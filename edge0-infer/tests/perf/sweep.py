"""Plain measurement sweep: several named configurations, a few processes each, medians and ranges.

This is the "measure once and report" tool, not a comparison: no pairing, no noise estimate, no stop conditions.
The configurations run in rotation (``--runs`` processes each). A process that fails is recorded as
``failed`` and the sweep goes on; nothing is repeated. Outliers are only annotated with what the snapshots show.
Background load is recorded in the snapshots and never blocks.

Report kind ``sweep`` in the format of ``report.py``; ``summary`` holds medians, ranges and outlier notes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import abba
import hosts
import report as reports

# name -> (prompt file, input tokens with the chat template, timed output tokens). Only macOS Metal defaults to the long
# prompt, whose prefill speed is the steady one; every other platform and device defaults to the medium prompt.
PROMPTS = {'long': ('tests/fixtures/bench/library-report-prompt.txt', 3084, 100),
           'medium': ('tests/fixtures/bench/library-report-prompt-head.txt', 627, 100),
           'short': ('tests/fixtures/bench/short-story-prompt.txt', 19, 512)}
PROMPT, INPUT_COUNT, _ = PROMPTS['long']


def default_prompt(platform, backend):
    return 'long' if (platform, backend) == ('macos', 'metal') else 'medium'
# name -> extra bench options
CONFIGS = {
    'predicted': [],
    'actual': ['--model-option', 'routing=actual'],
    'ctx16384': ['--context', '16384'],
    'direct': ['--model-option', 'expert_read=direct'],
}
OUTLIER_RELATIVE = .10          # distance from the configuration's median that earns an annotation
OUTLIER_METRICS = ('decode_tokens_per_second', 'first_token_seconds')
SUMMARY_METRICS = reports.METRICS


def build_command(cli, model, extra, backend='metal', prompt=PROMPT, ntok=100):
    return [str(cli), 'bench', '--model', str(model), '--device', backend, '--user-file', str(prompt),
            '--warmup', '10', '--ntok', str(ntok), '--runs', '1', *extra]


def build_protocol(runs, cooldown, prompt=PROMPT, backend='metal', name='long', configs=tuple(CONFIGS), options=()):
    """The measurement protocol recorded in the report (what a later report must match to be comparable)."""
    file, count, ntok = PROMPTS[name]
    # Recorded only when given, so a default protocol still equals the stored baseline protocol.
    extra = dict(options=list(options)) if options else {}
    return dict(kind='sweep', backend=backend, prompt=dict(file=file, sha256=reports.file_sha256(Path(prompt)), input_count=count),
                warmup=10, ntok=ntok, runs=1, runs_per_configuration=runs, order=list(configs), cooldown_seconds=cooldown, **extra,
                outlier_rule=f'relative distance from the configuration median above {OUTLIER_RELATIVE:.0%}')


def counted(cell):
    """Cells that count toward medians: the successful ones."""
    return cell['status'] == 'ok'


def observed(cell):
    """What the snapshot shows that could explain an odd process (names and numbers only)."""
    before = cell.get('snapshot_before') or {}
    load = before.get('background_load') or {}
    residency = (before.get('residency') or {}).get('base.gguf') if isinstance(before.get('residency'), dict) else None
    loadavg = before.get('load')
    return dict(
        non_system_top_before=[f"{e['name']} {e['cpu_percent']:.0f}%" for e in (load.get('non_system_top') or [])[:3]],
        above_threshold_before=[f"{e['name']} {e['cpu_percent']:.0f}%" for e in (load.get('other') or [])],
        loadavg_before=loadavg if isinstance(loadavg, str) else None,
        base_resident_gib_before=round(residency['resident_bytes'] / 2**30, 2) if isinstance(residency, dict) and 'resident_bytes' in residency else None)


def summarize(cells, order):
    """Per configuration: counts, medians and ranges, memory maxima and outlier notes."""
    result = {}
    for name in order:
        mine = [c for c in cells if c['side'] == name]
        used = [c for c in mine if counted(c)]
        entry = dict(processes=len(mine), ok=sum(c['status'] == 'ok' for c in mine), failed=sum(c['status'] == 'failed' for c in mine),
                     out_of_scope=sum(c['status'] == 'out_of_scope' for c in mine),
                     exclusions=[dict(cell=c['index'], reason=c['reason']) for c in mine if c['status'] == 'out_of_scope'],
                     metrics={}, memory_max_bytes={}, outliers=[],
                     failures=[dict(cell=c['index'], reason=c['reason']) for c in mine if c['status'] == 'failed'])
        for metric in SUMMARY_METRICS:
            values = [c['metrics'][metric] for c in used]
            if values:
                entry['metrics'][metric] = reports.median_range(values)
        for group in ('ledger_peak_bytes', 'rss_peak_bytes', 'physical_peak_bytes'):
            values = [c['memory'][group] for c in used if c['memory'].get(group) is not None]
            if values:
                entry['memory_max_bytes'][group] = max(values)
        if len(used) >= 3:
            for metric in OUTLIER_METRICS:
                median = entry['metrics'][metric]['median']
                for c in used:
                    relative = c['metrics'][metric] / median - 1
                    if abs(relative) > OUTLIER_RELATIVE:
                        entry['outliers'].append(dict(cell=c['index'], metric=metric, value=c['metrics'][metric],
                                                      relative_to_median=relative, observed=observed(c)))
        result[name] = entry
    return result


def run_sweep(host, cli, model, protocol, writer, *, raw_dir, tools, root=Path('.'), sleep=time.sleep, runs=3):
    """Run the sweep and write the report. Returns the final report. Never stops on a failed process."""
    identity = dict(engine=None, binary_sha256=reports.file_sha256(cli), tools_sha256=reports.tools_digest(tools),
                    workspace=reports.workspace_state(root), model=None, device=host.device())
    prompt, order, options = protocol['prompt']['file'], protocol['order'], protocol.get('options', [])
    sides = {name: dict(command=build_command(cli, model, CONFIGS[name] + options, protocol['backend'], prompt, protocol['ntok']), configuration=None)
             for name in order}
    writer.start(identity, protocol, sides)
    Path(raw_dir).mkdir(parents=True, exist_ok=False)
    cells, configuration_of, index = [], {}, 0
    model_dir = Path(model)
    for name in [n for _ in range(runs) for n in order]:
        index += 1
        unsupported = getattr(host, 'unsupported_configuration', lambda name: None)(name)
        if unsupported:
            cell = reports.failed_cell(index, name, unsupported)
            cell['status'] = 'out_of_scope'
            writer.add_cell(cell); cells.append(cell)
            continue
        if index > 1:
            remaining = protocol['cooldown_seconds']
            while remaining > 0:
                sleep(min(30, remaining))
                remaining -= min(30, remaining)
        print(f'sweep process {index} ({name})', flush=True)

        def fail(reason, **more):
            cell = reports.failed_cell(index, name, reason, **more)
            writer.add_cell(cell)
            cells.append(cell)
        try:
            host.preflight()
        except hosts.PreflightError as error:
            fail(str(error))
            continue
        before = host.snapshot(model_dir)
        result = host.run(sides[name]['command'])
        (Path(raw_dir) / f'{index}.stdout').write_text(result.stdout)
        (Path(raw_dir) / f'{index}.stderr').write_text(result.stderr)
        after = host.snapshot(model_dir)
        if result.returncode:
            fail(f'exit code {result.returncode}', snapshot_before=before, snapshot_after=after)
            continue
        try:
            row, summary = abba.parse_output(result.stdout, input_count=protocol['prompt']['input_count'], ntok=protocol['ntok'])
            metrics = dict(first_token_seconds=row['prefill_seconds'], prefill_tokens_per_second=row['prefill_tokens_per_second'],
                           decode_tokens_per_second=row['decode_tokens_per_second'])
            memory = reports.memory_from_snapshot(row['memory_snapshot'])
        except (ValueError, KeyError, TypeError) as error:
            fail(f'unusable benchmark output: {error}', snapshot_before=before, snapshot_after=after)
            continue
        model_identity = dict(adapter=summary['model']['adapter'], assets=summary['model']['assets'],
                              assets_verified=summary['model']['assets_verified'])
        if identity['engine'] is None:
            identity['engine'], identity['model'] = summary['engine'], model_identity
            identity['build_consistency'] = reports.build_consistency(identity['engine'], identity['workspace'])
            writer.update_identity(engine=identity['engine'], model=model_identity, build_consistency=identity['build_consistency'])
        elif summary['engine'] != identity['engine'] or model_identity != identity['model']:
            fail('engine or model identity changed between processes', snapshot_before=before, snapshot_after=after)
            continue
        configuration = dict(summary['model']['loading'])
        if 'runtime' in summary:
            configuration['runtime'] = summary['runtime']
        configuration_of.setdefault(name, configuration)
        cell = reports.ok_cell(index, name, metrics, memory, before, after, configuration=configuration)
        writer.add_cell(cell)
        cells.append(cell)
    for name in sides:
        sides[name]['configuration'] = configuration_of.get(name)
    final = dict(format=reports.FORMAT, identity=identity, protocol=protocol, sides=sides, cells=cells, noise=None, pairs=[],
                 decision=None, summary=summarize(cells, order))
    writer.finish(final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path, help='edge0-infer binary')
    parser.add_argument('--model', required=True, type=Path, help='Asset directory')
    parser.add_argument('--out', required=True, type=Path, help='Report prefix (.json and .md)')
    parser.add_argument('--raw', required=True, type=Path, help='New raw-output directory')
    parser.add_argument('--runs', type=int, default=3, help='Processes per configuration')
    parser.add_argument('--cooldown', type=float, default=30, help='Idle seconds between processes')
    parser.add_argument('--platform', help='Measurement platform (default: this machine)')
    parser.add_argument('--prompt', choices=PROMPTS, help='Input prompt (default: medium on Android, Windows and iOS; long on macOS)')
    parser.add_argument('--device', choices=['cpu', 'metal'], help='Backend device (default: the platform default, Metal on macOS)')
    parser.add_argument('--configs', default='predicted',
                        help=f'Comma-separated configurations from {", ".join(CONFIGS)}, or "all" (default: predicted)')
    parser.add_argument('--option', action='append', default=[],
                        help='Extra bench argument passed to every configuration; repeat it, e.g. --option=--threads --option=4')
    args = parser.parse_args(argv)
    if args.runs < 1 or not 0 <= args.cooldown <= 60:
        parser.error('Runs must be positive and cooldown between 0 and 60 seconds')
    configs = list(CONFIGS) if args.configs == 'all' else [n.strip() for n in args.configs.split(',') if n.strip()]
    unknown = [n for n in configs if n not in CONFIGS]
    if not configs or unknown or len(set(configs)) != len(configs):
        parser.error(f'Unknown, duplicate or empty configurations: {args.configs}')
    try:
        host = hosts.select(args.platform)
    except hosts.UnsupportedPlatform as error:
        parser.error(str(error))
    for suffix in ('.json', '.md', '.partial.jsonl'):
        if Path(str(args.out) + suffix).exists():
            parser.error(f'Refusing to overwrite existing evidence: {args.out}{suffix}')
    backend = args.device or host.backend
    name = args.prompt or default_prompt(host.name, backend)
    protocol = build_protocol(args.runs, args.cooldown, prompt=PROMPTS[name][0], backend=backend, name=name,
                              configs=configs, options=args.option)
    here = Path(__file__).resolve().parent
    try:
        run_sweep(host, args.cli, args.model, protocol, reports.ReportWriter(args.out), raw_dir=args.raw,
                  tools=[here / n for n in ('sweep.py', 'abba.py', 'hosts.py', 'report.py')] +
                        ([here.parent / 'platform' / n for n in ('android.py', 'android_cli.py', 'android-cli')] if host.name == 'android' else
                         [here.parent / 'platform' / n for n in ('ios.py', 'ios_cli.py', 'ios-cli')] if host.name == 'ios' else []),
                  runs=args.runs)
    except FileExistsError as error:
        print(f'Sweep stopped: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
