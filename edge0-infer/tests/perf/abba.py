"""Serial ABBA benchmarking with fixed workload and explicit retention decisions.

Processes run through a measurement host (``hosts.py``) and are recorded cell by cell in the report format of
``report.py``.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosts
import report as reports

LIMIT = 4 * 1024**3
ORDER = 'ABBAABBA'
METRICS = ('prefill_seconds', 'decode_tokens_per_second')
SHORT_PROMPTS = {'chat-intro': ('tests/fixtures/bench/edge0-chat-intro.txt', 17),
                 'long-mixed': ('tests/fixtures/bench/edge0-long-mixed.txt', 358)}


def pairs(processes):
    if len(processes) != 8 or any(len(p) != 1 for p in processes):
        raise ValueError('ABBAABBA requires eight processes with one run each')
    return [(processes[a][0], processes[b][0]) for a, b in [(0, 1), (3, 2), (4, 5), (7, 6)]]


def relative(paired, metric):
    values = []
    for a, b in paired:
        x, y = a[metric], b[metric]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in (x, y)):
            raise ValueError('Metrics must be positive finite numbers')
        values.append(y / x - 1)
    return values


def noise_floor(processes):
    paired = pairs(processes)
    return {key: max(.02, max(abs(v) for v in relative(paired, key))) for key in METRICS}


def decide(processes, noise, target, correct):
    if target not in METRICS:
        raise ValueError('Unknown target metric')
    paired = pairs(processes)
    changes = {key: [(-v if key == 'prefill_seconds' else v) for v in relative(paired, key)] for key in METRICS}
    for key in METRICS:
        if key not in noise or not math.isfinite(noise[key]) or noise[key] < .02:
            raise ValueError('Noise must be finite and at least 2 percent')
    other = next(key for key in METRICS if key != target)
    peaks = [b['memory_snapshot']['physical']['values']['peak_bytes'] for _, b in paired]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in peaks):
        raise ValueError('Physical peak measurements are required')
    conditions = {
        'target_majority': sum(v > 0 for v in changes[target]) >= 3,
        'target_above_noise': statistics.mean(changes[target]) > noise[target],
        'other_within_noise': statistics.mean(changes[other]) >= -noise[other],
        'physical_limit': all(v <= LIMIT for v in peaks),
        'correctness': correct is True,
    }
    return dict(retain=all(conditions.values()), conditions=conditions, improvements=changes,
                mean_improvement={k: statistics.mean(v) for k, v in changes.items()}, target=target)


def check_command(command, short_prompt=None):
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError('Commands must be JSON arrays of strings')
    required = {'--device': 'metal', '--warmup': '10', '--ntok': '100', '--runs': '1',
                '--user-file': SHORT_PROMPTS[short_prompt][0] if short_prompt else 'tests/fixtures/bench/library-report-prompt.txt'}
    if command[1:2] != ['bench']:
        raise ValueError('Command must invoke bench')
    for key, value in required.items():
        if command.count(key) != 1 or command[command.index(key)+1] != value:
            raise ValueError(f'Fixed workload requires {key} {value}')
    if command.count('--model') != 1 or '--breakdown' in command or '--tokens' in command:
        raise ValueError('Model required; breakdown and token overrides are excluded from ABBA')
    return Path(command[command.index('--model')+1]).resolve()


def checked_rows(stdout, short_prompt=None, input_count=None, *, ntok=100):
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    runs = [row for row in rows if row.get('type') == 'run']
    if len(runs) != 1 or runs[0]['run'] != 1:
        raise ValueError('Expected exactly one run')
    for row in runs:
        if any(row[key] != value for key, value in dict(input_count=input_count or (SHORT_PROMPTS[short_prompt][1] if short_prompt else 3084), accepted_tokens=11 + ntok,
                                                       warmup=10, decode_forwards=ntok, stop_reason='length').items()):
            raise ValueError('Benchmark changed the fixed workload')
    return runs


def cooldown_seconds(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 60:
        raise argparse.ArgumentTypeError('Cooldown must be finite and between 0 and 60 seconds')
    return value


def schedule(cooldown, sleep=time.sleep):
    for index, side in enumerate(ORDER):
        if index:
            remaining = cooldown
            while remaining > 0:
                interval = min(30, remaining)
                sleep(interval)
                remaining -= interval
        yield index, side




class MeasurementStopped(RuntimeError):
    """A cell failed; the failed cell is in the partial report and measurement stopped."""


def cell_row(cell):
    """The row view of a cell that the pairing and decision functions work on."""
    return dict(prefill_seconds=cell['metrics']['first_token_seconds'],
                decode_tokens_per_second=cell['metrics']['decode_tokens_per_second'],
                memory_snapshot=cell['memory']['memory_snapshot'])


def parse_output(stdout, short_prompt=None, input_count=None, *, ntok=100):
    """(run row, summary row) of a bench process; raises ValueError when the workload is not the fixed one."""
    runs = checked_rows(stdout, short_prompt, input_count, ntok=ntok)
    summaries = [row for row in (json.loads(line) for line in stdout.splitlines() if line.strip()) if row.get('type') == 'summary']
    if len(summaries) != 1 or 'engine' not in summaries[0] or 'model' not in summaries[0]:
        raise ValueError('Expected one summary row with engine and model identity')
    return runs[0], summaries[0]


def build_protocol(a, args_cooldown, prompt_path, input_count):
    prompt = Path(prompt_path)
    return dict(kind='abba', backend='metal',
                prompt=dict(file=str(prompt), sha256=reports.file_sha256(prompt), input_count=input_count),
                warmup=10, ntok=100, runs=1, order=ORDER, cooldown_seconds=args_cooldown)


def run_measurement(host, commands, protocol, writer, *, model, raw_dir, tools, root=Path('.'), noise_report=None,
                    noise_only=False, record_only=False, target='decode_tokens_per_second', correct=False,
                    short_prompt=None, sleep=time.sleep):
    """Run the eight processes and write the report. Returns the final report."""
    host.preflight()
    identity = dict(engine=None, binary_sha256={s: reports.file_sha256(c[0]) for s, c in commands.items()},
                    tools_sha256=reports.tools_digest(tools), workspace=reports.workspace_state(root),
                    model=None, device=host.device())
    sides = {s: dict(command=c, configuration=None) for s, c in commands.items()}
    writer.start(identity, protocol, sides)
    Path(raw_dir).mkdir(parents=True, exist_ok=False)
    cells, configuration_of = [], {}
    for index, side in schedule(protocol['cooldown_seconds'], sleep):
        n = index + 1
        print(f'ABBAABBA process {n}/8 ({side})', flush=True)

        def stop(reason, **extra):
            cell = reports.failed_cell(n, side, reason, **extra)
            writer.add_cell(cell)
            raise MeasurementStopped(f'process {n} ({side}): {reason}')
        try:
            host.preflight()
            host.check_background_load()
        except hosts.PreflightError as error:
            stop(str(error))
        before = host.snapshot(model)
        result = host.run(commands[side])
        (Path(raw_dir) / f'{n}.stdout').write_text(result.stdout)
        (Path(raw_dir) / f'{n}.stderr').write_text(result.stderr)
        after = host.snapshot(model)
        if result.returncode:
            stop(f'exit code {result.returncode}', snapshot_before=before, snapshot_after=after)
        try:
            row, summary = parse_output(result.stdout, short_prompt)
            metrics = dict(first_token_seconds=row['prefill_seconds'], prefill_tokens_per_second=row['prefill_tokens_per_second'],
                           decode_tokens_per_second=row['decode_tokens_per_second'])
            memory = reports.memory_from_snapshot(row['memory_snapshot'])
        except (ValueError, KeyError, TypeError) as error:
            stop(f'unusable benchmark output: {error}', snapshot_before=before, snapshot_after=after)
        model_identity = dict(adapter=summary['model']['adapter'], assets=summary['model']['assets'],
                              assets_verified=summary['model']['assets_verified'])
        if identity['engine'] is None:
            identity['engine'], identity['model'] = summary['engine'], model_identity
            consistency = reports.build_consistency(identity['engine'], identity['workspace'])
            identity['build_consistency'] = consistency
            writer.update_identity(engine=identity['engine'], model=model_identity, build_consistency=consistency)
        elif summary['engine'] != identity['engine'] or model_identity != identity['model']:
            stop('engine or model identity changed between processes', snapshot_before=before, snapshot_after=after)
        configuration = summary['model']['loading']
        if side in configuration_of and configuration_of[side] != configuration:
            stop('effective configuration changed between processes of one side', snapshot_before=before, snapshot_after=after)
        configuration_of[side] = configuration
        cell = reports.ok_cell(n, side, metrics, memory, before, after, configuration=configuration,
                               workload={k: row[k] for k in ('input_count', 'accepted_tokens', 'warmup', 'decode_forwards',
                                                              'stop_reason', 'end_token_count', 'first_end_token_position')})
        writer.add_cell(cell)
        cells.append(cell)
    for side in sides:
        sides[side]['configuration'] = configuration_of[side]
    runs = [[cell_row(c)] for c in cells]
    if noise_only:
        noise = noise_floor(runs)
    else:
        noise = noise_report['noise']
    paired = pairs(runs)
    index_of = {id(r[0]): c['index'] for r, c in zip(runs, cells)}
    pair_list = [dict(a=index_of[id(x)], b=index_of[id(y)], relative={m: y[m] / x[m] - 1 for m in METRICS}) for x, y in paired]
    decision = None if noise_only or record_only else decide(runs, noise, target, correct)
    final = dict(format=reports.FORMAT, identity=identity, protocol=protocol, sides=sides, cells=cells,
                 noise=noise, pairs=pair_list, decision=decision)
    writer.finish(final)
    return final


def noise_matches(noise_report, protocol):
    """Why a noise report cannot serve this measurement, or None."""
    if noise_report.get('format') != reports.FORMAT or noise_report.get('protocol', {}).get('kind') != 'abba':
        return 'Noise report must be an abba report in the current format'
    if not isinstance(noise_report.get('noise'), dict):
        return 'Noise report has no noise measurement'
    for key in ('cooldown_seconds', 'warmup', 'ntok', 'runs', 'order'):
        if noise_report['protocol'].get(key) != protocol[key]:
            return f'Noise report differs in {key}'
    if noise_report['protocol']['prompt'].get('sha256') != protocol['prompt']['sha256']:
        return 'Noise report used another input file'
    if noise_report['protocol'].get('pressure') is not None:
        return 'Noise report was measured under the removed memory-pressure condition'
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True, help='A command as a JSON string array (no shell)')
    parser.add_argument('--b', required=True, help='B command as a JSON string array (no shell)')
    parser.add_argument('--out', required=True, type=Path, help='Report prefix (.json and .md)')
    parser.add_argument('--raw', required=True, type=Path, help='New raw-output directory')
    parser.add_argument('--cooldown', type=cooldown_seconds, default=30, help='Idle seconds between processes; must match the noise report')
    parser.add_argument('--noise', type=Path, help='A/A report JSON providing noise')
    parser.add_argument('--target', choices=METRICS, default='decode_tokens_per_second')
    parser.add_argument('--correct', action='store_true', help='Corresponding correctness checks passed')
    parser.add_argument('--noise-only', action='store_true', help='Measure A/A noise, no retention decision')
    parser.add_argument('--record-only', action='store_true', help='Report metrics without a retention decision')
    parser.add_argument('--short-prompt', choices=SHORT_PROMPTS, help='Fixed short input; requires --record-only and cannot measure noise')
    parser.add_argument('--platform', help='Measurement platform (default: this machine)')
    args = parser.parse_args(argv)
    if args.short_prompt and (not args.record_only or args.noise_only):
        parser.error('Short prompts are record-only comparisons with existing noise')
    try:
        host = hosts.select(args.platform)
    except hosts.UnsupportedPlatform as error:
        parser.error(str(error))
    a, b = json.loads(args.a), json.loads(args.b)
    model = check_command(a, args.short_prompt)
    if check_command(b, args.short_prompt) != model:
        parser.error('Both commands must use the same model')
    if args.noise_only and a != b:
        parser.error('Noise measurement requires identical commands')
    if not args.noise_only and args.noise is None:
        parser.error('A/A noise report is required')
    for path in (Path(str(args.out) + suffix) for suffix in ('.json', '.md', '.partial.jsonl')):
        if path.exists():
            parser.error(f'Refusing to overwrite existing evidence: {path}')
    prompt, count = SHORT_PROMPTS[args.short_prompt] if args.short_prompt else ('tests/fixtures/bench/library-report-prompt.txt', 3084)
    protocol = build_protocol(a, args.cooldown, prompt, count)
    noise_report = None
    if args.noise:
        try:
            noise_report = reports.load(args.noise)
        except reports.ReportError as error:
            parser.error(str(error))
        reason = noise_matches(noise_report, protocol)
        if reason:
            parser.error(reason)
    here = Path(__file__).resolve().parent
    try:
        run_measurement(host, dict(A=a, B=b), protocol, reports.ReportWriter(args.out), model=model, raw_dir=args.raw,
                        tools=[here / name for name in ('abba.py', 'hosts.py', 'report.py')],
                        noise_report=noise_report, noise_only=args.noise_only, record_only=args.record_only,
                        target=args.target, correct=args.correct, short_prompt=args.short_prompt)
    except (MeasurementStopped, hosts.PreflightError) as error:
        print(f'Measurement stopped: {error}', file=sys.stderr)
        return 1
    final = reports.load(str(args.out) + '.json')
    print(json.dumps(dict(noise=final['noise'], decision=final['decision'])), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
