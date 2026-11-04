"""Counterexamples for the ABBA retention decision."""
import importlib.util
from pathlib import Path
import unittest
import argparse
import json
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('abba', HERE / 'abba.py')
abba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abba)
import hosts
import report as reports


def row(first=10, decode=10, peak=abba.LIMIT):
    return dict(prefill_seconds=first, decode_tokens_per_second=decode,
                memory_snapshot={'physical': {'values': {'peak_bytes': peak}}})


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.runs = [[row() if side == 'A' else row(10, 11)] for side in abba.ORDER]
        self.noise = dict.fromkeys(abba.METRICS, .02)

    def decision(self):
        return abba.decide(self.runs, self.noise, 'decode_tokens_per_second', True)

    def test_all_conditions(self):
        self.assertTrue(self.decision()['retain'])

    def test_three_better_below_noise(self):
        for index in [1, 2, 5]:
            self.runs[index] = [row(10, 10.1)]
        self.runs[6] = [row(10, 9.9)]
        result = self.decision()
        self.assertTrue(result['conditions']['target_majority'])
        self.assertFalse(result['conditions']['target_above_noise'])
        self.assertFalse(result['retain'])

    def test_other_metric_regression(self):
        self.runs[1][0] = row(11, 11)
        self.assertFalse(self.decision()['conditions']['other_within_noise'])

    def test_memory(self):
        self.runs[6][0] = row(10, 11, abba.LIMIT+1)
        self.assertFalse(self.decision()['conditions']['physical_limit'])

    def test_correctness_required(self):
        self.assertFalse(abba.decide(self.runs, self.noise, abba.METRICS[1], False)['retain'])

    def test_majority_required(self):
        self.runs[5] = self.runs[6] = [row()]
        result = self.decision()
        self.assertTrue(result['conditions']['target_above_noise'])
        self.assertFalse(result['retain'])

    def test_lower_latency_is_better(self):
        for index in [1, 2, 5, 6]:
            self.runs[index] = [row(9, 10)]
        self.assertTrue(abba.decide(self.runs, self.noise, abba.METRICS[0], True)['retain'])

    def test_pairing(self):
        self.assertEqual(abba.pairs([[i] for i in range(8)]), [(0, 1), (3, 2), (4, 5), (7, 6)])

    def test_noise_maximum_and_floor(self):
        self.assertEqual(abba.noise_floor([[row()]]*8), self.noise)
        self.runs[6][0] = row(8, 8)
        self.assertAlmostEqual(abba.noise_floor(self.runs)[abba.METRICS[0]], .2)
        self.assertAlmostEqual(abba.noise_floor(self.runs)[abba.METRICS[1]], .2)

    def test_cooldown(self):
        for seconds in [0, 30, 60, 31.5]:
            sleeps = []
            self.assertEqual(list(abba.schedule(seconds, sleeps.append)), list(enumerate(abba.ORDER)))
            self.assertEqual(sum(sleeps), 7 * seconds)
            self.assertTrue(all(0 < value <= 30 for value in sleeps))
            self.assertEqual(abba.cooldown_seconds(str(seconds)), seconds)
        for value in ['-1', '61', 'nan', 'inf']:
            with self.assertRaises(argparse.ArgumentTypeError):
                abba.cooldown_seconds(value)

    def test_short_workload(self):
        command = ['binary', 'bench', '--model', 'model', '--device', 'metal',
                   '--warmup', '10', '--ntok', '100', '--runs', '1', '--user-file',
                   'tests/fixtures/bench/library-report-prompt.txt']
        self.assertEqual(abba.check_command(command), Path('model').resolve())
        wrong=command.copy()
        wrong[-1]='tests/fixtures/bench/edge0-long-prompt.txt'
        with self.assertRaises(ValueError):
            abba.check_command(wrong)
        for flag, old in [('--ntok', '200'), ('--runs', '2')]:
            invalid = command.copy()
            invalid[invalid.index(flag) + 1] = old
            with self.assertRaises(ValueError):
                abba.check_command(invalid)
        record = dict(type='run', run=1, input_count=3084, accepted_tokens=111,
                      warmup=10, decode_forwards=100, stop_reason='length')
        self.assertEqual(abba.checked_rows(json.dumps(record)), [record])
        with self.assertRaises(ValueError):
            abba.checked_rows(json.dumps(record) + '\n' + json.dumps(record))

    def test_invalid_metrics(self):
        for value in [0, -1, float('nan'), float('inf')]:
            self.runs[1][0]['prefill_seconds'] = value
            with self.assertRaises(ValueError):
                self.decision()
        with self.assertRaises(ValueError):
            abba.pairs(self.runs[:3])

    def test_short_prompt_inputs_do_not_change_default_workload(self):
        for name, (path, count) in abba.SHORT_PROMPTS.items():
            command = ['binary', 'bench', '--model', 'model', '--device', 'metal',
                       '--warmup', '10', '--ntok', '100', '--runs', '1', '--user-file', path]
            self.assertEqual(abba.check_command(command, name), Path('model').resolve())
            with self.assertRaises(ValueError):
                abba.check_command(command)
            record = dict(type='run', run=1, input_count=count, accepted_tokens=111,
                          warmup=10, decode_forwards=100, stop_reason='length')
            self.assertEqual(abba.checked_rows(json.dumps(record), name), [record])
            with self.assertRaises(ValueError):
                abba.checked_rows(json.dumps(record))
            record['decode_forwards'] = 99
            with self.assertRaises(ValueError):
                abba.checked_rows(json.dumps(record), name)

    def test_short_prompt_requires_record_only(self):
        import subprocess
        import sys
        script = str(Path(__file__).with_name('abba.py'))
        for flags in [[], ['--record-only', '--noise-only']]:
            result = subprocess.run([sys.executable, script, '--a', '[]', '--b', '[]',
                                     '--out', 'unused', '--raw', 'unused',
                                     '--short-prompt', 'chat-intro', *flags], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Short prompts are record-only', result.stderr)


SUMMARY_ENGINE = dict(version='0.1.0', source_commit='a' * 40, source_modified='no', ggml_origin='o', ggml_version='v')
LOADING = dict(directory='model', device='MTL0', context=4096, prefill_chunk=4096, model_options={'routing': 'predicted'}, sources={})
ASSETS = [dict(role='base', filename='base.gguf', bytes=1, sha256='0' * 64)]


def bench_stdout(decode=20.0, first=6.0, peak=3 * 1024**3, loading=LOADING):
    memory = {'ledger': {'peak_bytes': 1}, 'rss': {'values': {'peak_bytes': 2}}, 'physical': {'values': {'peak_bytes': peak}}}
    run = dict(type='run', run=1, input_count=3084, accepted_tokens=111, warmup=10, decode_forwards=100, stop_reason='length',
               decode_tokens_per_second=decode, prefill_seconds=first, prefill_tokens_per_second=3084 / first,
               end_token_count=0, first_end_token_position=None, memory_snapshot=memory)
    summary = dict(type='summary', engine=SUMMARY_ENGINE, model=dict(adapter='edge0_35b', assets=ASSETS, assets_verified=False, loading=loading))
    return json.dumps(run) + '\n' + json.dumps(summary) + '\n'


class FakeHost:
    name = 'fake'

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    busy_from = None

    def preflight(self):
        pass

    def check_background_load(self):
        if self.busy_from is not None and self.calls >= self.busy_from:
            raise hosts.BackgroundLoadError('background load: Some Helper (30%) above 15% CPU')

    def device(self):
        return dict(platform='fake', model='M', chip='C', memory_bytes=16 * 1024**3, os_version='1')

    def snapshot(self, model):
        return dict(platform='fake', time=0, residency={'base.gguf': dict(resident_bytes=1024)})

    def run(self, argv):
        item = self.outputs[self.calls]
        self.calls += 1
        return item if isinstance(item, hosts.CommandResult) else hosts.CommandResult(0, item, '')


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-abba-')
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.binary = self.root / 'bin'
        self.binary.write_bytes(b'binary')
        self.prompt = self.root / 'prompt.txt'
        self.prompt.write_text('prompt')
        command = [str(self.binary), 'bench', '--model', 'model']
        self.commands = dict(A=command, B=command)

    def protocol(self):
        return dict(kind='abba', backend='metal', prompt=dict(file=str(self.prompt), sha256=reports.file_sha256(self.prompt), input_count=3084),
                    warmup=10, ntok=100, runs=1, order=abba.ORDER, cooldown_seconds=0)

    def measure(self, host, **options):
        writer = reports.ReportWriter(self.root / 'out' / 'report')
        return writer, abba.run_measurement(host, self.commands, self.protocol(), writer, model=Path('model'),
                                            raw_dir=self.root / 'raw', tools=[self.binary], root=self.root, sleep=lambda s: None,
                                            **options)

    def test_new_format_cells_and_noise(self):
        writer, report = self.measure(FakeHost([bench_stdout()] * 8), noise_only=True)
        self.assertEqual(report['format'], reports.FORMAT)
        self.assertEqual([c['index'] for c in report['cells']], list(range(1, 9)))
        self.assertEqual([c['side'] for c in report['cells']], list(abba.ORDER))
        self.assertTrue(all(c['status'] == 'ok' and c['reason'] == '' for c in report['cells']))
        self.assertEqual(report['cells'][0]['metrics']['decode_tokens_per_second'], 20.0)
        self.assertEqual(report['cells'][0]['memory']['physical_peak_bytes'], 3 * 1024**3)
        self.assertEqual(report['noise'], dict.fromkeys(abba.METRICS, .02))
        self.assertEqual([(p['a'], p['b']) for p in report['pairs']], [(1, 2), (4, 3), (5, 6), (8, 7)])
        self.assertIsNone(report['decision'])
        self.assertEqual(report['sides']['A']['configuration'], LOADING)
        self.assertEqual(report['identity']['engine'], SUMMARY_ENGINE)
        self.assertEqual(report['identity']['model']['assets'], ASSETS)
        self.assertTrue(writer.json_path.exists() and writer.md_path.exists())
        self.assertFalse(writer.partial_path.exists())
        self.assertEqual(reports.validate(json.loads(writer.json_path.read_text())), [])

    def test_failed_process_keeps_measured_cells_in_partial_file(self):
        outputs = [bench_stdout()] * 4 + [hosts.CommandResult(3, '', 'boom')]
        host = FakeHost(outputs)
        writer = reports.ReportWriter(self.root / 'out' / 'report')
        with self.assertRaises(abba.MeasurementStopped):
            abba.run_measurement(host, self.commands, self.protocol(), writer, model=Path('model'), raw_dir=self.root / 'raw',
                                 tools=[self.binary], root=self.root, sleep=lambda s: None, noise_only=True)
        self.assertEqual(host.calls, 5)
        self.assertFalse(writer.json_path.exists())
        partial = reports.load_partial(writer.partial_path)
        self.assertEqual([c['status'] for c in partial['cells']], ['ok'] * 4 + ['failed'])
        self.assertIn('exit code 3', partial['cells'][4]['reason'])
        self.assertEqual(partial['identity']['engine'], SUMMARY_ENGINE)

    def test_workload_change_is_a_failed_cell(self):
        wrong = bench_stdout().replace('"decode_forwards": 100', '"decode_forwards": 99')
        writer = reports.ReportWriter(self.root / 'out' / 'report')
        with self.assertRaises(abba.MeasurementStopped):
            abba.run_measurement(FakeHost([wrong]), self.commands, self.protocol(), writer, model=Path('model'), raw_dir=self.root / 'raw',
                                 tools=[self.binary], root=self.root, sleep=lambda s: None, noise_only=True)
        self.assertEqual(reports.load_partial(writer.partial_path)['cells'][0]['status'], 'failed')

    def test_identity_change_between_processes_fails(self):
        changed = bench_stdout().replace('"source_modified": "no"', '"source_modified": "yes"')
        with self.assertRaises(abba.MeasurementStopped):
            self.measure(FakeHost([bench_stdout(), changed]), noise_only=True)

    def test_background_load_fails_the_cell_and_stops(self):
        host = FakeHost([bench_stdout()] * 8)
        host.busy_from = 3
        writer = reports.ReportWriter(self.root / 'out' / 'report')
        with self.assertRaises(abba.MeasurementStopped) as raised:
            abba.run_measurement(host, self.commands, self.protocol(), writer, model=Path('model'), raw_dir=self.root / 'raw',
                                 tools=[self.binary], root=self.root, sleep=lambda s: None, noise_only=True)
        self.assertIn('background load', str(raised.exception))
        self.assertEqual(host.calls, 3)
        cells = reports.load_partial(writer.partial_path)['cells']
        self.assertEqual([c['status'] for c in cells], ['ok', 'ok', 'ok', 'failed'])
        self.assertIn('Some Helper', cells[3]['reason'])
        self.assertFalse(writer.json_path.exists())

    def test_record_only_with_noise_report_has_no_decision(self):
        _, report = self.measure(FakeHost([bench_stdout()] * 8), record_only=True, noise_report=dict(noise=dict.fromkeys(abba.METRICS, .03)))
        self.assertIsNone(report['decision'])
        self.assertEqual(report['noise'], dict.fromkeys(abba.METRICS, .03))

    def test_noise_report_must_match_protocol(self):
        protocol = self.protocol()
        good = dict(format=reports.FORMAT, protocol=protocol, noise=dict.fromkeys(abba.METRICS, .02))
        self.assertIsNone(abba.noise_matches(good, protocol))
        for key, value in [('cooldown_seconds', 10), ('ntok', 99)]:
            bad = json.loads(json.dumps(good))
            bad['protocol'][key] = value
            self.assertIn(key, abba.noise_matches(bad, protocol))
        bad = json.loads(json.dumps(good))
        bad['protocol']['prompt']['sha256'] = 'x'
        self.assertIn('input file', abba.noise_matches(bad, protocol))

    def test_noise_report_measured_under_memory_pressure_is_refused(self):
        protocol = self.protocol()
        bad = dict(format=reports.FORMAT, protocol=dict(protocol, pressure=dict(leave_gib=8, requested_bytes=1)),
                   noise=dict.fromkeys(abba.METRICS, .02))
        self.assertIn('removed memory-pressure condition', abba.noise_matches(bad, protocol))

    def test_noise_report_with_null_or_missing_pressure_is_accepted(self):
        protocol = self.protocol()
        for extra in (dict(pressure=None), {}):
            old = dict(format=reports.FORMAT, protocol=dict(protocol, **extra), noise=dict.fromkeys(abba.METRICS, .02))
            self.assertIsNone(abba.noise_matches(old, protocol))


if __name__ == '__main__':
    unittest.main()
