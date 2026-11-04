"""Conversation reuse measurement: first token per turn, reused against cold, for several configurations.

Runs ``bench --turns`` processes round-robin over the combinations (default: predicted routing with thinking off,
predicted with thinking on, actual routing with thinking off), several processes each, through a measurement host,
and writes a report of kind ``turns`` in the format of ``report.py``.

Every process is checked against the conversation file (turn count, thinking flags, file digest) and against its
expected effective configuration; a process that does not match is a failed cell and stops the measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosts
import report as reports

CONTEXT = 16384
FIXTURES = Path('tests/fixtures/bench')
# name -> (conversation file, extra bench options, expected routing)
COMBINATIONS = {
    'predicted-thinking-off': (FIXTURES / 'turns-library.json', [], 'predicted'),
    'predicted-thinking-on': (FIXTURES / 'turns-library-thinking.json', [], 'predicted'),
    'actual-thinking-off': (FIXTURES / 'turns-library.json', ['--model-option', 'routing=actual'], 'actual'),
}


class MeasurementStopped(RuntimeError):
    """A cell failed; the failed cell is in the partial report and measurement stopped."""


def conversation_info(path):
    """Digest, turn count and per-turn thinking flags of a conversation file."""
    data = Path(path).read_bytes()
    turns = json.loads(data)['turns']
    return dict(file=str(path), sha256=hashlib.sha256(data).hexdigest(), turn_count=len(turns),
                thinking=[bool(t.get('thinking', False)) for t in turns])


def build_command(cli, model, turns_file, extra):
    return [str(cli), 'bench', '--model', str(model), '--turns', str(turns_file), '--device', 'metal', '--context', str(CONTEXT),
            '--temperature', '0', '--presence-penalty', '0', *extra]


def parse_output(stdout):
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    turns = [row for row in rows if row.get('type') == 'turn']
    summaries = [row for row in rows if row.get('type') == 'summary']
    if len(summaries) != 1 or rows[-1].get('type') != 'summary' or any(k not in summaries[0] for k in ('engine', 'model', 'memory_snapshot')):
        raise ValueError('Expected one final summary row with engine, model and memory')
    return turns, summaries[0]


def check_process(turns, summary, info, routing):
    """Reason the process does not match the conversation and configuration, or None."""
    if len(turns) != info['turn_count'] or summary.get('turns') != info['turn_count']:
        return f"turn count {len(turns)} differs from the conversation's {info['turn_count']}"
    for expected, row in zip(info['thinking'], turns):
        if row['thinking'] != expected:
            return f"turn {row['turn']} ran with thinking={row['thinking']}, the conversation asks for {expected}"
        if row.get('stop_reason') == 'failed' or row['cold'].get('stop_reason') == 'failed':
            return f"turn {row['turn']} failed"
    loading = summary['model']['loading']
    if loading.get('context') != CONTEXT:
        return f"effective context {loading.get('context')} differs from {CONTEXT}"
    if loading.get('model_options', {}).get('routing') != routing:
        return f"effective routing {loading.get('model_options', {}).get('routing')} differs from {routing}"
    return None


def cell_metrics(turns):
    metrics = {}
    for row in turns:
        t = row['turn']
        metrics.update({f'turn{t}_first_token_seconds': row['first_token_seconds'], f'turn{t}_reused_tokens': row['reused_tokens'],
                        f'turn{t}_computed_tokens': row['computed_tokens'],
                        f'turn{t}_cold_first_token_seconds': row['cold']['first_token_seconds'],
                        f'turn{t}_cold_computed_tokens': row['cold']['computed_tokens']})
    return metrics


def cell_turns(turns):
    """Per-turn rows without the token arrays; the generated tokens are kept as a digest for cross-process comparison."""
    kept = ('turn', 'thinking', 'input_count', 'reuse_method', 'reused_tokens', 'computed_tokens', 'checkpoint_restored',
            'checkpoint_boundary', 'first_token_seconds', 'generated_count', 'stop_reason', 'cold')
    return [dict({k: row[k] for k in kept},
                 input_sha256=hashlib.sha256(json.dumps(row['input']).encode()).hexdigest(),
                 generated_sha256=hashlib.sha256(json.dumps(row['generated']).encode()).hexdigest()) for row in turns]


def summarize(cells):
    """Per combination and turn: reuse methods, counts, medians and ranges, and whether generated tokens agree."""
    result = {}
    for combo in sorted({c['side'] for c in cells}):
        group = [c for c in cells if c['side'] == combo and c['status'] == 'ok']
        rows = []
        for t in range(len(group[0]['turns']) if group else 0):
            per = [c['turns'][t] for c in group]
            methods = {}
            for row in per:
                methods[row['reuse_method']] = methods.get(row['reuse_method'], 0) + 1
            rows.append(dict(
                turn=t + 1, processes=len(per), reuse_methods=methods, input_count=per[0]['input_count'],
                inputs_identical=len({r['input_sha256'] for r in per}) == 1,
                generated_identical=len({r['generated_sha256'] for r in per}) == 1,
                reused_tokens=reports.median_range([r['reused_tokens'] for r in per]),
                computed_tokens=reports.median_range([r['computed_tokens'] for r in per]),
                first_token_seconds=reports.median_range([r['first_token_seconds'] for r in per]),
                cold_first_token_seconds=reports.median_range([r['cold']['first_token_seconds'] for r in per]),
                cold_computed_tokens=reports.median_range([r['cold']['computed_tokens'] for r in per])))
        result[combo] = rows
    return result


def run_turns(host, combos, protocol, writer, *, model, raw_dir, tools, root=Path('.'), sleep=time.sleep):
    """combos: name -> dict(command, routing, info). Runs round-robin, ``protocol['runs_per_combination']`` rounds."""
    host.preflight()
    identity = dict(engine=None, binary_sha256={n: reports.file_sha256(c['command'][0]) for n, c in combos.items()},
                    tools_sha256=reports.tools_digest(tools), workspace=reports.workspace_state(root), model=None, device=host.device())
    sides = {n: dict(command=c['command'], configuration=None) for n, c in combos.items()}
    writer.start(identity, protocol, sides)
    Path(raw_dir).mkdir(parents=True, exist_ok=False)
    order = [name for _ in range(protocol['runs_per_combination']) for name in combos]
    cells, configuration_of = [], {}
    for position, name in enumerate(order):
        n = position + 1
        if position:
            remaining = protocol['cooldown_seconds']
            while remaining > 0:
                sleep(min(30, remaining))
                remaining -= min(30, remaining)
        print(f'turns process {n}/{len(order)} ({name})', flush=True)
        combo = combos[name]

        def stop(reason, **extra):
            writer.add_cell(reports.failed_cell(n, name, reason, **extra))
            raise MeasurementStopped(f'process {n} ({name}): {reason}')
        try:
            host.preflight()
            host.check_background_load()
        except hosts.PreflightError as error:
            stop(str(error))
        if reports.file_sha256(combo['info']['file']) != combo['info']['sha256']:
            stop('conversation file changed since the measurement started')
        before = host.snapshot(model)
        result = host.run(combo['command'])
        (Path(raw_dir) / f'{n}.stdout').write_text(result.stdout)
        (Path(raw_dir) / f'{n}.stderr').write_text(result.stderr)
        after = host.snapshot(model)
        if result.returncode:
            stop(f'exit code {result.returncode}', snapshot_before=before, snapshot_after=after)
        try:
            turns, summary = parse_output(result.stdout)
            problem = check_process(turns, summary, combo['info'], combo['routing'])
        except (ValueError, KeyError, TypeError) as error:
            stop(f'unusable output: {error}', snapshot_before=before, snapshot_after=after)
        if problem:
            stop(problem, snapshot_before=before, snapshot_after=after)
        model_identity = dict(adapter=summary['model']['adapter'], assets=summary['model']['assets'],
                              assets_verified=summary['model']['assets_verified'])
        if identity['engine'] is None:
            identity['engine'], identity['model'] = summary['engine'], model_identity
            identity['build_consistency'] = reports.build_consistency(identity['engine'], identity['workspace'])
            writer.update_identity(engine=identity['engine'], model=model_identity, build_consistency=identity['build_consistency'])
        elif summary['engine'] != identity['engine'] or model_identity != identity['model']:
            stop('engine or model identity changed between processes', snapshot_before=before, snapshot_after=after)
        configuration = summary['model']['loading']
        if configuration_of.setdefault(name, configuration) != configuration:
            stop('effective configuration changed between processes of one combination', snapshot_before=before, snapshot_after=after)
        cell = reports.ok_cell(n, name, cell_metrics(turns), reports.memory_from_snapshot(summary['memory_snapshot']), before, after,
                               configuration=configuration, turns=cell_turns(turns))
        writer.add_cell(cell)
        cells.append(cell)
    for name in sides:
        sides[name]['configuration'] = configuration_of[name]
    final = dict(format=reports.FORMAT, identity=identity, protocol=protocol, sides=sides, cells=cells, noise=None, pairs=[],
                 decision=None, summary=summarize(cells))
    writer.finish(final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True, type=Path, help='edge0-infer binary')
    parser.add_argument('--model', required=True, type=Path, help='Asset directory')
    parser.add_argument('--out', required=True, type=Path, help='Report prefix (.json and .md)')
    parser.add_argument('--raw', required=True, type=Path, help='New raw-output directory')
    parser.add_argument('--runs', type=int, default=3, help='Processes per combination')
    parser.add_argument('--cooldown', type=float, default=30, help='Idle seconds between processes')
    parser.add_argument('--platform', help='Measurement platform (default: this machine)')
    args = parser.parse_args(argv)
    if args.runs < 1 or not 0 <= args.cooldown <= 60:
        parser.error('Runs must be positive and cooldown between 0 and 60 seconds')
    try:
        host = hosts.select(args.platform)
    except hosts.UnsupportedPlatform as error:
        parser.error(str(error))
    combos = {name: dict(command=build_command(args.cli, args.model, file, extra), routing=routing, info=conversation_info(file))
              for name, (file, extra, routing) in COMBINATIONS.items()}
    protocol = dict(kind='turns', backend='metal', context=CONTEXT, runs_per_combination=args.runs, cooldown_seconds=args.cooldown,
                    order=list(combos), conversations={n: {k: c['info'][k] for k in ('file', 'sha256', 'turn_count', 'thinking')}
                                                       for n, c in combos.items()})
    here = Path(__file__).resolve().parent
    try:
        run_turns(host, combos, protocol, reports.ReportWriter(args.out), model=args.model, raw_dir=args.raw,
                  tools=[here / n for n in ('turns.py', 'hosts.py', 'report.py')])
    except (MeasurementStopped, hosts.PreflightError, FileExistsError) as error:
        print(f'Measurement stopped: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
