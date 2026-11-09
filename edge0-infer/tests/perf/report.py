"""Measurement reports: one fixed JSON structure, written cell by cell, comparable across runs.

Format ``moe-perf-report/1``::

    {"format": ..., "identity": {engine, binary_sha256, tools_sha256, workspace, model, device},
     "protocol": {kind, ...}, "sides": {name: {command, configuration}},
     "cells": [{index, side, status, reason, metrics, memory, snapshot_before, snapshot_after, ...}],
     "noise": {...}, "pairs": [...], "decision": null | {...}}

Commands:
  report.py compare --baseline <report> --candidate <report>   exit 0 pass, 1 regression, 2 input error
  report.py baseline --report <report> --sides <name>... --threshold <fraction> --out <file>
  report.py render --out <file.md> <report.json>... [--append <file.md>]...
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

FORMAT = 'moe-perf-report/1'
STATUSES = ('ok', 'failed', 'not_run', 'out_of_scope')
# Comparison metrics of an ABBA report and the noise entry that bounds each of them. The noise entries
# use the pairing names: ``prefill_seconds`` is the first token time (prefill speed derives from it).
METRICS = ('first_token_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second')
LOWER_IS_BETTER = frozenset({'first_token_seconds'})
NOISE_KEY = {'first_token_seconds': 'prefill_seconds', 'prefill_tokens_per_second': 'prefill_seconds',
             'decode_tokens_per_second': 'decode_tokens_per_second'}
NOISE_FLOOR = .02
IDENTITY_KEYS = ('engine', 'binary_sha256', 'tools_sha256', 'workspace', 'model', 'device')
EXIT_PASS, EXIT_REGRESSION, EXIT_INPUT = 0, 1, 2


class ReportError(ValueError):
    """The report is malformed or incomplete (exit code 2 for compare)."""


# ---------------------------------------------------------------- identity helpers

def sanitize(text):
    """Replace local paths by environment variable names, '.' for the working directory, $HOME for home."""
    for key in ['MOE_ASSETS_35B', 'MOE_SOURCE_35B', 'MOE_TINY']:
        if os.environ.get(key):
            text = text.replace(os.environ[key], '$' + key)
    text = text.replace(str(Path.cwd()), '.')
    return text.replace(str(Path.home()), '$HOME')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tools_digest(paths):
    return {Path(p).name: file_sha256(p) for p in paths}


def workspace_state(root, limit=50):
    """Git state of the working tree now: commit, and the changed relative paths (names only)."""
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f'git {args[0]} failed')
        return result.stdout
    try:
        commit = git('rev-parse', 'HEAD').strip()
        changed = [line[3:] for line in git('status', '--porcelain', '--untracked-files=normal').splitlines()]
    except (RuntimeError, OSError) as error:
        return {'available': False, 'reason': str(error)}
    return dict(commit=commit, modified=bool(changed), changed_count=len(changed), changed_paths=sorted(changed)[:limit])


def build_consistency(engine, workspace):
    """Whether the binary was built from the tree that is measured from now."""
    if not isinstance(engine, dict) or not isinstance(workspace, dict) or workspace.get('available') is False:
        return dict(matches=None, note='build or workspace identity unavailable')
    same = (engine.get('source_commit') == workspace.get('commit')
            and (engine.get('source_modified') == 'yes') == workspace.get('modified'))
    return dict(matches=same, note='' if same else 'build and workspace differ')


# ---------------------------------------------------------------- cells

def memory_from_snapshot(snapshot):
    def peak(group):
        values = snapshot[group].get('values')
        return None if values is None else values['peak_bytes']
    return dict(ledger_peak_bytes=snapshot['ledger']['peak_bytes'], rss_peak_bytes=peak('rss'),
                physical_peak_bytes=peak('physical'), memory_snapshot=snapshot)


def failed_cell(index, side, reason, status='failed', **extra):
    cell = dict(index=index, side=side, status=status, reason=reason, metrics={}, memory={},
                snapshot_before=None, snapshot_after=None)
    cell.update(extra)
    return cell


def ok_cell(index, side, metrics, memory, before, after, **extra):
    cell = dict(index=index, side=side, status='ok', reason='', metrics=metrics, memory=memory,
                snapshot_before=before, snapshot_after=after)
    cell.update(extra)
    return cell


def side_cells(report, side):
    return [cell for cell in report['cells'] if cell['side'] == side]


def median_range(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), count=len(values))


# ---------------------------------------------------------------- writing

class ReportWriter:
    """Writes ``<out>.partial.jsonl`` while measuring and ``<out>.json`` / ``<out>.md`` at the end.

    The partial file starts with one header record (identity and protocol); each finished process appends
    a ``cell`` record at once, so cells already measured survive a later failure. The partial file is
    removed only when the final report has been written.
    """

    def __init__(self, out):
        self.json_path, self.md_path = Path(str(out) + '.json'), Path(str(out) + '.md')
        self.partial_path = Path(str(out) + '.partial.jsonl')
        for path in (self.json_path, self.md_path, self.partial_path):
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite existing evidence: {path}')
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record):
        with self.partial_path.open('a') as f:
            f.write(sanitize(json.dumps(record)) + '\n')
            f.flush()

    def start(self, identity, protocol, sides):
        self._append(dict(type='header', format=FORMAT, identity=identity, protocol=protocol, sides=sides))

    def update_identity(self, **fields):
        self._append(dict(type='identity', fields=fields))

    def add_cell(self, cell):
        self._append(dict(type='cell', cell=cell))

    def finish(self, report):
        text = sanitize(json.dumps(report, indent=2)) + '\n'
        self.json_path.write_text(text)
        self.md_path.write_text(sanitize(render(report)))
        self.partial_path.unlink()


def load_partial(path):
    """The report as far as it was written: header, identity updates and the cells so far."""
    report = None
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record['type'] == 'header':
            report = dict(format=record['format'], identity=record['identity'], protocol=record['protocol'],
                          sides=record['sides'], cells=[], noise=None, pairs=[], decision=None)
        elif report is None:
            raise ReportError('Partial report does not start with a header record')
        elif record['type'] == 'identity':
            report['identity'].update(record['fields'])
        elif record['type'] == 'cell':
            report['cells'].append(record['cell'])
    if report is None:
        raise ReportError('Empty partial report')
    return report


def load(path):
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f'Cannot read report {path}: {error}')
    return report


# ---------------------------------------------------------------- validation and comparison

def validate(report, name='report', sides=None):
    """Problems that make a report unusable as a comparison input."""
    if not isinstance(report, dict) or report.get('format') != FORMAT:
        return [f'{name}: unsupported or missing format']
    problems = []
    identity = report.get('identity')
    if not isinstance(identity, dict):
        problems.append(f'{name}: identity missing')
    else:
        for key in IDENTITY_KEYS:
            if identity.get(key) in (None, '', {}, []):
                problems.append(f'{name}: identity.{key} missing')
    for key in ('protocol', 'sides'):
        if not isinstance(report.get(key), dict) or not report[key]:
            problems.append(f'{name}: {key} missing')
    cells = report.get('cells')
    if not isinstance(cells, list) or not cells:
        problems.append(f'{name}: no cells')
        return problems
    for cell in cells:
        if sides is not None and isinstance(cell, dict) and cell.get('side') not in sides:
            continue
        if not isinstance(cell, dict) or cell.get('status') not in STATUSES:
            problems.append(f'{name}: cell with invalid status')
        elif cell['status'] == 'out_of_scope' and report.get('protocol', {}).get('kind') == 'sweep':
            if not cell.get('reason'):
                problems.append(f'{name}: out-of-scope cell needs a reason')
        elif cell['status'] != 'ok':
            problems.append(f"{name}: cell {cell.get('index')} is {cell['status']}: {cell.get('reason', '')}")
    return problems


def device_key(report):
    device = report['identity']['device']
    return device.get('model'), device.get('memory_bytes')


def side_metric(report, side, metric):
    values = []
    for cell in side_cells(report, side):
        value = cell['metrics'].get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ReportError(f'cell {cell.get("index")} has no usable {metric}')
        values.append(value)
    return values


COMPARABLE_KINDS = ('abba', 'sweep')


def normalized_configuration(configuration):
    """Supply only the stage defaults absent from reports predating mixed stages."""
    if not isinstance(configuration, dict):
        return configuration
    result = copy.deepcopy(configuration)
    for key, default in [('prefill_device', result.get('device')), ('stage_policy', 'fixed')]:
        if key not in result:
            result[key] = default
            sources = result.setdefault('sources', {})
            if isinstance(sources, dict):
                sources.setdefault(key, 'platform')
    return result


def compare(baseline, candidate):
    """(exit code, output lines). Reasons for exit code 2 are returned as lines and nothing is compared.

    A metric regresses when the candidate median is worse than the baseline median by more than the baseline
    threshold (the larger of the baseline noise, if any, the baseline's fixed threshold, if any, and 2 percent)
    and more than half of the candidate's processes are worse than the baseline median. Only the baseline's sides
    are compared; other sides of the candidate are ignored, including failed cells in them.
    """
    base_problems = validate(baseline, 'baseline')
    if base_problems:
        return EXIT_INPUT, base_problems
    sides = sorted(baseline['sides'])
    problems = validate(candidate, 'candidate', sides)
    if problems:
        return EXIT_INPUT, problems
    kinds = {baseline['protocol'].get('kind'), candidate['protocol'].get('kind')}
    if len(kinds) != 1 or not kinds <= set(COMPARABLE_KINDS):
        return EXIT_INPUT, ['compare only supports reports of the same kind: ' + ', '.join(COMPARABLE_KINDS)]
    if baseline['protocol'] != candidate['protocol']:
        return EXIT_INPUT, ['protocols differ: ' + ', '.join(
            sorted(k for k in set(baseline['protocol']) | set(candidate['protocol'])
                   if baseline['protocol'].get(k) != candidate['protocol'].get(k)))]
    if device_key(baseline) != device_key(candidate):
        return EXIT_INPUT, [f'devices differ: {device_key(baseline)} vs {device_key(candidate)}']
    if not set(sides) <= set(candidate['sides']):
        return EXIT_INPUT, ['candidate lacks sides of the baseline']
    for side in sides:
        if (normalized_configuration(baseline['sides'][side].get('configuration')) !=
                normalized_configuration(candidate['sides'][side].get('configuration'))):
            return EXIT_INPUT, [f'effective configuration of side {side} differs']
    noise, fixed = baseline.get('noise'), baseline.get('fixed_threshold')
    if not isinstance(fixed, (int, float)) or isinstance(fixed, bool) or fixed < 0:
        fixed = None
    if not isinstance(noise, dict) and fixed is None:
        return EXIT_INPUT, ['baseline has neither a noise measurement nor a fixed threshold']
    lines, regression = [], False
    try:
        for side in sides:
            for metric in METRICS:
                base_values, cand_values = side_metric(baseline, side, metric), side_metric(candidate, side, metric)
                if len(base_values) != len(cand_values) or len(base_values) < 3:
                    return EXIT_INPUT, [f'side {side}: process counts differ or are too small']
                base_median, cand_median = statistics.median(base_values), statistics.median(cand_values)
                change = cand_median / base_median - 1
                worse = (lambda v: v > base_median) if metric in LOWER_IS_BETTER else (lambda v: v < base_median)
                worse_amount = change if metric in LOWER_IS_BETTER else -change
                measured = noise.get(NOISE_KEY[metric]) if isinstance(noise, dict) else None
                allowed = max(NOISE_FLOOR, measured or 0, fixed or 0)
                worse_count = sum(worse(v) for v in cand_values)
                verdict = worse_amount > allowed and worse_count * 2 > len(cand_values)
                regression = regression or verdict
                lines.append(f'side {side} {metric}: baseline median {base_median:.6g}, candidate median {cand_median:.6g}, '
                             f'change {change:+.2%}, threshold {allowed:.2%}, {worse_count} of {len(cand_values)} processes worse '
                             f'-> {"REGRESSION" if verdict else "ok"}')
    except ReportError as error:
        return EXIT_INPUT, [str(error)]
    return (EXIT_REGRESSION if regression else EXIT_PASS), lines


def make_baseline(report, sides, threshold):
    """A baseline from the given sides of a report: identity, protocol, configuration, per-cell metrics and
    memory numbers, a fixed threshold; no system snapshots and no raw memory snapshots."""
    problems = validate(report, 'report', set(sides))
    if problems:
        raise ReportError('; '.join(problems))
    missing = [s for s in sides if s not in report['sides']]
    if missing:
        raise ReportError(f'report has no side {missing}')
    cells = []
    for cell in report['cells']:
        if cell['side'] in sides:
            memory = {k: v for k, v in cell['memory'].items() if k != 'memory_snapshot'}
            cells.append(dict(index=cell['index'], side=cell['side'], status=cell['status'], reason=cell['reason'],
                              metrics=cell['metrics'], memory=memory, snapshot_before=None, snapshot_after=None))
    return dict(format=FORMAT, identity=report['identity'], protocol=report['protocol'],
                sides={s: {k: v for k, v in report['sides'][s].items() if k != 'command'} | {'command': report['sides'][s].get('command')}
                       for s in sides},
                cells=cells, noise=None, fixed_threshold=threshold, pairs=[], decision=None)


# ---------------------------------------------------------------- rendering (layout only, no new judgements)

def _number(value, digits=4):
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines


def _residency(cell):
    shot = cell.get('snapshot_before') or {}
    value = (shot.get('residency') or {}).get('base.gguf') if isinstance(shot.get('residency'), dict) else None
    if isinstance(value, dict) and 'resident_bytes' in value:
        return f"{value['resident_bytes'] / 2**30:.2f}"
    return 'n/a'


def render(report):
    """Markdown for one report. The numbers are copied from the report, never recomputed into a verdict."""
    kind = report['protocol'].get('kind')
    identity = report['identity']
    out = [f'## Report ({kind})', '']
    engine = identity.get('engine') or {}
    out += [f"- Engine: version {engine.get('version')}, source commit {engine.get('source_commit')}, "
            f"modified {engine.get('source_modified')}, GGML {engine.get('ggml_origin')} {engine.get('ggml_version')}",
            f"- Device: {json.dumps(identity.get('device'), sort_keys=True)}",
            f"- Workspace at measurement: {json.dumps(identity.get('workspace'), sort_keys=True)}",
            f"- Binary SHA-256: {json.dumps(identity.get('binary_sha256'), sort_keys=True)}",
            f"- Protocol: {json.dumps(report['protocol'], sort_keys=True)}", '']
    if kind == 'turns':
        out += render_turns(report)
        return '\n'.join(out) + '\n'
    out += ['### Effective configuration', '']
    for side, entry in sorted(report['sides'].items()):
        configuration = entry.get('configuration') or {}
        out.append(f"- {side}: {json.dumps(configuration.get('model_options'), sort_keys=True)}, context {configuration.get('context')}")
    out += ['', '### Per-process cells', '']
    rows = []
    for cell in report['cells']:
        m, mem = cell.get('metrics') or {}, cell.get('memory') or {}
        rows.append([cell['index'], cell['side'], cell['status'], _number(m.get('first_token_seconds')),
                     _number(m.get('prefill_tokens_per_second'), 1), _number(m.get('decode_tokens_per_second'), 3),
                     _gib(mem.get('ledger_peak_bytes')), _gib(mem.get('rss_peak_bytes')), _gib(mem.get('physical_peak_bytes')),
                     _residency(cell)])
    out += _table(['Cell', 'Side', 'Status', 'First token (s)', 'Prefill (token/s)', 'Decode (token/s)',
                   'Ledger peak (GiB)', 'RSS peak (GiB)', 'Physical peak (GiB)', 'base.gguf resident before (GiB)'], rows)
    out += ['', '### Per-side summary (median, minimum, maximum)', '']
    rows = []
    for side in sorted(report['sides']):
        for metric in METRICS:
            values = [c['metrics'][metric] for c in side_cells(report, side) if c['status'] == 'ok' and metric in c['metrics']]
            if values:
                s = median_range(values)
                rows.append([side, metric, _number(s['median'], 4), _number(s['minimum'], 4), _number(s['maximum'], 4), s['count']])
    out += _table(['Side', 'Metric', 'Median', 'Minimum', 'Maximum', 'Processes'], rows)
    if report.get('pairs'):
        out += ['', '### Pairs (B relative to A)', '']
        out += _table(['Pair', 'A cell', 'B cell', 'First token', 'Decode'],
                      [[i, p['a'], p['b'], f"{p['relative']['prefill_seconds']:+.2%}",
                        f"{p['relative']['decode_tokens_per_second']:+.2%}"] for i, p in enumerate(report['pairs'], 1)])
    if kind == 'sweep':
        out += render_sweep(report)
    else:
        out += ['', f"Noise: {json.dumps(report.get('noise'))}", '', f"Decision: {json.dumps(report.get('decision'))}", '']
    return '\n'.join(out) + '\n'


def render_sweep(report):
    out = ['', '### Outliers (annotation only; nothing was repeated)', '']
    summary = report.get('summary') or {}
    found = False
    for name, entry in summary.items():
        for item in entry.get('outliers', []):
            found = True
            seen = item['observed']
            out.append(f"- {name}, cell {item['cell']}, {item['metric']} = {item['value']:.4g} ({item['relative_to_median']:+.1%} from the median); "
                       f"before the process: non-system top {seen['non_system_top_before']}, above threshold {seen['above_threshold_before']}, "
                       f"load average {seen['loadavg_before']}, base.gguf resident {seen['base_resident_gib_before']} GiB")
    if not found:
        out.append('None.')
    out += ['', '### Out of scope', '']
    exclusions = [f"- {name}, cell {item['cell']}: {item['reason']}" for name, entry in summary.items() for item in entry.get('exclusions', [])]
    out += exclusions or ['None.']
    out += ['', '### Failures', '']
    found = False
    for name, entry in summary.items():
        for failure in entry.get('failures', []):
            found = True
            out.append(f"- {name}, cell {failure['cell']}: {failure['reason']}")
    if not found:
        out.append('None.')
    return out + ['']


def _gib(value):
    return 'n/a' if value is None else f'{value / 2**30:.3f}'


def render_turns(report):
    out = ['### Turns summary', '']
    summary = report.get('summary') or {}
    for combo in sorted(summary):
        out += [f'#### {combo}', '']
        rows = []
        for turn in summary[combo]:
            rows.append([turn['turn'], json.dumps(turn['reuse_methods'], sort_keys=True), turn['input_count'],
                         turn.get('generated_identical'),
                         _stat(turn['reused_tokens']), _stat(turn['computed_tokens']),
                         _stat(turn['first_token_seconds']), _stat(turn['cold_first_token_seconds']),
                         _stat(turn['cold_computed_tokens'])])
        out += _table(['Turn', 'Reuse methods', 'Input tokens', 'Generated tokens identical', 'Reused', 'Computed', 'First token (s)',
                       'Cold first token (s)', 'Cold computed'], rows)
        out.append('')
    out += ['### Processes', '']
    out += _table(['Cell', 'Combination', 'Status', 'Reason'],
                  [[c['index'], c['side'], c['status'], c.get('reason', '')] for c in report['cells']])
    return out + ['']


def _stat(entry):
    if entry is None:
        return 'n/a'
    return f"{entry['median']:.4g} ({entry['minimum']:.4g} to {entry['maximum']:.4g})"


# ---------------------------------------------------------------- command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--baseline', required=True, type=Path)
    c.add_argument('--candidate', required=True, type=Path)
    b = sub.add_parser('baseline')
    b.add_argument('--report', required=True, type=Path)
    b.add_argument('--sides', required=True, nargs='+')
    b.add_argument('--threshold', required=True, type=float)
    b.add_argument('--out', required=True, type=Path)
    r = sub.add_parser('render')
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--title', default='Measurement report')
    r.add_argument('--append', action='append', default=[], type=Path, help='Markdown files copied verbatim after the reports')
    r.add_argument('reports', nargs='*', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        try:
            code, lines = compare(load(args.baseline), load(args.candidate))
        except ReportError as error:
            code, lines = EXIT_INPUT, [str(error)]
        print('\n'.join(lines), file=sys.stderr if code == EXIT_INPUT else sys.stdout)
        return code
    if args.command == 'baseline':
        try:
            baseline = make_baseline(load(args.report), args.sides, args.threshold)
        except (ReportError, KeyError) as error:
            print(f'Cannot make baseline: {error}', file=sys.stderr)
            return EXIT_INPUT
        if args.out.exists():
            print(f'Refusing to overwrite {args.out}', file=sys.stderr)
            return EXIT_INPUT
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(sanitize(json.dumps(baseline, indent=1, sort_keys=True)) + '\n')
        return 0
    try:
        parts = [f'# {args.title}', ''] + [render(load(p)) for p in args.reports]
        parts += [p.read_text() for p in args.append]
        text = '\n'.join(parts)
    except (ReportError, KeyError, OSError) as error:
        print(f'Cannot render: {error}', file=sys.stderr)
        return EXIT_INPUT
    args.out.write_text(sanitize(text))
    return 0


if __name__ == '__main__':
    sys.exit(main())
