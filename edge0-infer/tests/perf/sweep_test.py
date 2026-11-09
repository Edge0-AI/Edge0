"""Sweep: rotation order, failures that do not stop it, protocol, medians and outlier notes (fake host)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hosts
import report as reports
spec = importlib.util.spec_from_file_location('sweep', HERE / 'sweep.py')
sweep = importlib.util.module_from_spec(spec)
sys.modules['sweep'] = sweep
spec.loader.exec_module(sweep)

ENGINE = dict(version='0.1.0', source_commit='e' * 40, source_modified='no', source_reason='', ggml_origin='o', ggml_version='v')
ASSETS = [dict(role='base', filename='base.gguf', bytes=1, sha256='0' * 64)]
REPO = HERE.parents[1]
BASELINE = HERE / 'baselines' / 'macos-metal-Mac17,3-16GiB.json'


def bench_stdout(decode=20.0, first=6.0, input_count=3084):
    memory = {'ledger': {'peak_bytes': 1}, 'rss': {'values': {'peak_bytes': 2}}, 'physical': {'values': {'peak_bytes': 3}}}
    run = dict(type='run', run=1, input_count=input_count, accepted_tokens=111, warmup=10, decode_forwards=100, stop_reason='length',
               decode_tokens_per_second=decode, prefill_seconds=first, prefill_tokens_per_second=input_count / first,
               end_token_count=0, first_end_token_position=None, memory_snapshot=memory)
    summary = dict(type='summary', engine=ENGINE, model=dict(adapter='edge0_35b', assets=ASSETS, assets_verified=False,
                                                           loading=dict(context=4096, model_options={})))
    return json.dumps(run) + '\n' + json.dumps(summary) + '\n'


class FakeHost:
    name = 'macos'

    def __init__(self, outputs, busy=False):
        self.outputs, self.calls, self.busy = outputs, 0, busy
        self.commands, self.preflights = [], 0

    def preflight(self):
        self.preflights += 1

    def check_background_load(self):
        raise AssertionError('the sweep must not block on background load')

    def device(self):
        return dict(platform='macos', model='M', chip='C', memory_bytes=16 * 2**30, os_version='1')

    def snapshot(self, model):
        return dict(platform='macos', time=0, residency={'base.gguf': dict(resident_bytes=2**30)}, load='vm.loadavg: { 1 1 1 }',
                    background_load=dict(threshold=15.0, other=[dict(name='Busy', cpu_percent=30.0)] if self.busy else [], system=[],
                                         non_system_top=[dict(name='Busy', cpu_percent=30.0)] if self.busy else []))

    def run(self, argv):
        self.commands.append(list(argv))
        item = self.outputs(self.calls, argv) if callable(self.outputs) else self.outputs
        self.calls += 1
        return item if isinstance(item, hosts.CommandResult) else hosts.CommandResult(0, item, '')


class SweepTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-sweep-')
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.cli = self.root / 'cli'
        self.cli.write_bytes(b'x')

    def protocol(self):
        return dict(kind='sweep', backend='metal', prompt=dict(file=sweep.PROMPT, sha256='x', input_count=3084), warmup=10, ntok=100,
                    runs=1, runs_per_configuration=3, order=list(sweep.CONFIGS), cooldown_seconds=30)

    def run_sweep(self, host, sleeps=None, protocol=None):
        writer = reports.ReportWriter(self.root / 'out' / 'sweep')
        self.writer = writer
        return sweep.run_sweep(host, self.cli, Path('model'), protocol or self.protocol(), writer, raw_dir=self.root / 'raw', tools=[self.cli],
                               root=self.root, sleep=(sleeps.append if sleeps is not None else lambda s: None))

    def test_selected_configuration_with_extra_options(self):
        protocol = dict(self.protocol(), order=['predicted'], options=['--threads', '4'])
        host = FakeHost(bench_stdout())
        report = self.run_sweep(host, protocol=protocol)
        self.assertEqual([c['side'] for c in report['cells']], ['predicted'] * 3)
        self.assertEqual(list(report['summary']), ['predicted'])
        self.assertTrue(all(c[-2:] == ['--threads', '4'] for c in host.commands))
        self.assertEqual(reports.validate(json.loads(self.writer.json_path.read_text())), [])

    def test_default_prompt_by_platform_and_backend(self):
        cases = [('macos', 'metal'), ('macos', 'cpu'), ('android', 'cpu'), ('windows', 'cpu'), ('ios', 'metal')]
        self.assertEqual([sweep.default_prompt(*c) for c in cases], ['long', 'medium', 'medium', 'medium', 'medium'])

    def test_medium_prompt_checks_its_own_input_count(self):
        file, count, _ = sweep.PROMPTS['medium']
        protocol = dict(self.protocol(), order=['predicted'], prompt=dict(file=file, sha256='x', input_count=count))
        report = self.run_sweep(FakeHost(bench_stdout(input_count=count)), protocol=protocol)
        self.assertEqual([c['status'] for c in report['cells']], ['ok'] * 3)
        host = FakeHost(bench_stdout())
        holder = tempfile.TemporaryDirectory(prefix='moe-sweep-')
        self.addCleanup(holder.cleanup)
        report = sweep.run_sweep(host, self.cli, Path('model'), protocol, reports.ReportWriter(Path(holder.name) / 'sweep'),
                                 raw_dir=Path(holder.name) / 'raw', tools=[self.cli], root=self.root, sleep=lambda s: None)
        self.assertEqual([c['status'] for c in report['cells']], ['failed'] * 3)
        self.assertIn('fixed workload', report['cells'][0]['reason'])

    def test_prompt_commands_and_output_windows(self):
        for name, expected_count, expected_ntok in [('long', 3084, 100), ('medium', 627, 100), ('short', 19, 512)]:
            with self.subTest(prompt=name):
                file, count, ntok = sweep.PROMPTS[name]
                protocol = sweep.build_protocol(3, 30, prompt=REPO / file, name=name, configs=['predicted'])
                self.assertEqual((count, ntok, protocol['ntok']), (expected_count, expected_ntok, expected_ntok))
                host = FakeHost(bench_stdout(input_count=count).replace('"accepted_tokens": 111', f'"accepted_tokens": {11+ntok}')
                                .replace('"decode_forwards": 100', f'"decode_forwards": {ntok}'))
                with tempfile.TemporaryDirectory() as temp:
                    result = sweep.run_sweep(host, self.cli, Path('model'), protocol, reports.ReportWriter(Path(temp)/'out'),
                                             raw_dir=Path(temp)/'raw', tools=[self.cli], root=self.root, sleep=lambda s: None)
                expected = [str(self.cli), 'bench', '--model', 'model', '--device', 'metal', '--user-file', file,
                            '--warmup', '10', '--ntok', str(expected_ntok), '--runs', '1']
                self.assertEqual(host.commands, [expected]*3)
                self.assertEqual([c['status'] for c in result['cells']], ['ok']*3)
                if name == 'short':
                    with self.assertRaises(ValueError):
                        sweep.abba.parse_output(bench_stdout(input_count=count), input_count=count, ntok=512)

    def test_invalid_configuration_names_are_refused(self):
        base = ['--cli', str(self.cli), '--model', 'model', '--out', str(self.root / 'o'), '--raw', str(self.root / 'r')]
        for configs in ['nope', 'predicted,predicted', ',']:
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                sweep.main(base + ['--configs', configs])
            self.assertEqual(raised.exception.code, 2)

    def test_android_exclusions(self):
        host = FakeHost(bench_stdout())
        host.backend = 'cpu'
        host.unsupported_configuration = lambda name: 'direct unsupported' if name == 'direct' else None
        report = self.run_sweep(host, protocol=dict(self.protocol(), backend='cpu'))
        self.assertEqual(host.calls, 9)
        self.assertTrue(all(c[c.index('--device')+1] == 'cpu' for c in host.commands))
        self.assertEqual(report['summary']['direct']['out_of_scope'], 3)
        self.assertEqual(report['summary']['direct']['failed'], 0)
        self.assertTrue(all(c['reason'] for c in report['cells'] if c['status'] == 'out_of_scope'))
        self.assertEqual(reports.validate(report), [])

    def test_rotation_order_and_report(self):
        host, sleeps = FakeHost(bench_stdout()), []
        report = self.run_sweep(host, sleeps)
        names = [c['side'] for c in report['cells']]
        self.assertEqual(names, ['predicted', 'actual', 'ctx16384', 'direct'] * 3)
        self.assertEqual(sum(sleeps), 11 * 30)
        self.assertEqual(reports.validate(json.loads(self.writer.json_path.read_text())), [])
        self.assertIn(('--model-option', 'routing=actual'), {tuple(c[-2:]) for c in host.commands})
        self.assertIn(('--context', '16384'), {tuple(c[-2:]) for c in host.commands})
        self.assertIn(('--model-option', 'expert_read=direct'), {tuple(c[-2:]) for c in host.commands})
        for c in host.commands:
            self.assertEqual(c[c.index('--warmup') + 1:c.index('--warmup') + 2], ['10'])

    def test_failed_process_is_recorded_and_the_sweep_continues(self):
        def outputs(call, argv):
            return hosts.CommandResult(3, '', 'boom') if call in (1, 5) else bench_stdout()
        host = FakeHost(outputs)
        report = self.run_sweep(host)
        self.assertEqual(host.calls, 12)
        self.assertEqual([c['status'] for c in report['cells']].count('failed'), 2)
        self.assertEqual(report['cells'][1]['status'], 'failed')
        self.assertIn('exit code 3', report['cells'][1]['reason'])
        self.assertEqual(report['summary']['actual']['failed'], 2)
        self.assertEqual(report['summary']['actual']['ok'], 1)
        self.assertEqual(report['summary']['actual']['metrics']['decode_tokens_per_second']['count'], 1)

    def test_unusable_output_and_changed_identity_are_failed_cells(self):
        def outputs(call, argv):
            if call == 2:
                return 'not json\n'
            if call == 3:
                return bench_stdout().replace('"source_modified": "no"', '"source_modified": "yes"')
            return bench_stdout()
        report = self.run_sweep(FakeHost(outputs))
        self.assertIn('unusable', report['cells'][2]['reason'])
        self.assertIn('identity changed', report['cells'][3]['reason'])
        self.assertEqual(sum(c['status'] == 'ok' for c in report['cells']), 10)

    def test_medians_ranges_and_outlier_annotation_with_load_names(self):
        values = {0: 20.0, 4: 21.0, 8: 14.0}          # processes 1, 5, 9 are `predicted`
        host = FakeHost(lambda call, argv: bench_stdout(decode=values.get(call, 20.0)))
        original = host.snapshot
        def snapshot(model):
            host.busy = host.calls == 8
            return original(model)
        host.snapshot = snapshot
        report = self.run_sweep(host)
        summary = report['summary']['predicted']
        self.assertEqual((summary['metrics']['decode_tokens_per_second']['median'], summary['metrics']['decode_tokens_per_second']['minimum'],
                          summary['metrics']['decode_tokens_per_second']['maximum']), (20.0, 14.0, 21.0))
        outliers = summary['outliers']
        self.assertEqual([(o['cell'], o['metric']) for o in outliers], [(9, 'decode_tokens_per_second')])
        self.assertEqual(outliers[0]['observed']['above_threshold_before'], ['Busy 30%'])
        self.assertEqual(report['summary']['actual']['outliers'], [])
        text = reports.render(report)
        self.assertIn('Busy 30%', text)
        self.assertNotIn('/', json.dumps(outliers))

    def test_background_load_never_blocks(self):
        report = self.run_sweep(FakeHost(bench_stdout(), busy=True))
        self.assertEqual(sum(c['status'] == 'ok' for c in report['cells']), 12)

    def test_acpower_loss_fails_the_process_and_continues(self):
        host = FakeHost(bench_stdout())
        original = host.preflight
        def preflight():
            original()
            if host.preflights == 3:
                raise hosts.PreflightError('Measurement requires AC power throughout the comparison')
        host.preflight = preflight
        report = self.run_sweep(host)
        self.assertEqual(report['cells'][2]['status'], 'failed')
        self.assertEqual(host.calls, 11)


class ProtocolTest(unittest.TestCase):
    def test_default_protocol_equals_the_committed_baseline_protocol(self):
        baseline = json.loads(BASELINE.read_text())
        protocol = sweep.build_protocol(3, 30, prompt=REPO / sweep.PROMPT)
        self.assertEqual(protocol, baseline['protocol'])
        self.assertEqual(protocol['order'], ['predicted', 'actual', 'ctx16384', 'direct'])

    def test_medium_prompt_is_recorded_and_used(self):
        file, count, _ = sweep.PROMPTS['medium']
        protocol = sweep.build_protocol(3, 30, prompt=REPO / file, backend='cpu', name='medium')
        self.assertEqual((protocol['prompt']['file'], protocol['prompt']['input_count']), (file, 627))
        command = sweep.build_command('cli', 'model', [], 'cpu', file)
        self.assertEqual(command[command.index('--user-file') + 1], file)


if __name__ == '__main__':
    unittest.main()
