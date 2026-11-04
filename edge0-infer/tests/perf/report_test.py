"""Report comparison, writer and rendering: pure logic on constructed reports."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('report', HERE / 'report.py')
report = importlib.util.module_from_spec(spec)
sys.modules['report'] = report
spec.loader.exec_module(report)

LOADING = {'context': 4096, 'model_options': {'routing': 'predicted'}}
PROTOCOL = dict(kind='abba', backend='metal', prompt=dict(file='p.txt', sha256='ab', input_count=3084), warmup=10, ntok=100,
                runs=1, order='ABBAABBA', cooldown_seconds=30)


def make(decode=(20, 20, 20, 20), first=6.0, noise=.04, side_b=None):
    """A complete report. ``decode`` gives the four decode values of side A; side B gets ``side_b`` or the same."""
    values = {'A': list(decode), 'B': list(side_b or decode)}
    cells = []
    position = {'A': 0, 'B': 0}
    for index, side in enumerate('ABBAABBA', 1):
        d = values[side][position[side]]
        position[side] += 1
        cells.append(report.ok_cell(index, side, dict(first_token_seconds=first, prefill_tokens_per_second=3084 / first,
                                                      decode_tokens_per_second=float(d)),
                                    dict(ledger_peak_bytes=1, rss_peak_bytes=2, physical_peak_bytes=3, memory_snapshot={}), {}, {}))
    return dict(format=report.FORMAT,
                identity=dict(engine={'version': '1'}, binary_sha256={'A': 'x', 'B': 'x'}, tools_sha256={'abba.py': 'y'},
                              workspace={'commit': 'c'}, model={'adapter': 'edge0_35b', 'assets': []},
                              device={'model': 'Mac17,3', 'memory_bytes': 17179869184}),
                protocol=copy.deepcopy(PROTOCOL),
                sides={'A': dict(command=['x'], configuration=copy.deepcopy(LOADING)), 'B': dict(command=['x'], configuration=copy.deepcopy(LOADING))},
                cells=cells, noise={'prefill_seconds': noise, 'decode_tokens_per_second': noise}, pairs=[], decision=None)


class CompareTest(unittest.TestCase):
    def test_optional_process_values(self):
        snapshot = {'ledger': {'peak_bytes': 1}, 'rss': {'values': {'current_bytes': None, 'peak_bytes': 23}},
                    'physical': {'values': {'current_bytes': 12, 'peak_bytes': None}}}
        result = report.memory_from_snapshot(snapshot)
        self.assertEqual(result['rss_peak_bytes'], 23)
        self.assertIsNone(result['physical_peak_bytes'])
        self.assertIsNone(result['memory_snapshot']['rss']['values']['current_bytes'])

    def code(self, candidate, baseline=None):
        return report.compare(baseline or make(), candidate)[0]

    def test_identical_reports_pass(self):
        self.assertEqual(self.code(make()), 0)

    def test_difference_inside_noise_passes(self):
        self.assertEqual(self.code(make(decode=(19.4,) * 4)), 0)  # 3% lower, noise 4%

    def test_regression_beyond_noise(self):
        code, lines = report.compare(make(), make(decode=(18.8,) * 4))  # 6% lower, noise 4%
        self.assertEqual(code, 1)
        self.assertTrue(any('REGRESSION' in line for line in lines))

    def test_three_of_four_worse_is_enough(self):
        candidate = make(decode=(18.8, 18.8, 18.8, 21))
        self.assertEqual(report.compare(make(), candidate)[0], 1)

    def test_only_two_worse_passes_even_if_median_drop_is_large(self):
        candidate = make(decode=(10, 10, 20, 20))
        self.assertEqual(self.code(candidate), 0)

    def test_slower_first_token_is_a_regression(self):
        self.assertEqual(self.code(make(first=6.6)), 1)  # 10% slower, noise 4%

    def test_noise_floor_two_percent(self):
        baseline = make(noise=.0)
        self.assertEqual(self.code(make(decode=(19.7,) * 4), baseline), 0)  # 1.5% below the 2% floor
        self.assertEqual(self.code(make(decode=(19.3,) * 4), baseline), 1)  # 3.5%

    def test_protocol_difference_is_refused(self):
        for key, value in [('cooldown_seconds', 10), ('ntok', 99)]:
            candidate = make()
            candidate['protocol'][key] = value
            code, lines = report.compare(make(), candidate)
            self.assertEqual(code, 2)
            self.assertIn(key, lines[0])
        candidate = make()
        candidate['protocol']['prompt']['sha256'] = 'zz'
        self.assertEqual(self.code(candidate), 2)

    def test_device_difference_is_refused(self):
        candidate = make()
        candidate['identity']['device']['memory_bytes'] = 8 * 2**30
        self.assertEqual(self.code(candidate), 2)
        candidate = make()
        candidate['identity']['device']['model'] = 'Other'
        self.assertEqual(self.code(candidate), 2)

    def test_configuration_difference_is_refused(self):
        candidate = make()
        candidate['sides']['B']['configuration']['model_options'] = {'routing': 'actual'}
        self.assertEqual(self.code(candidate), 2)

    def test_stage_defaults_preserve_old_baselines_without_mutation(self):
        baseline = make()
        for side in baseline['sides'].values():
            side['configuration'].update(device='CPU', sources={'device': 'request'})
        candidate = copy.deepcopy(baseline)
        for side in candidate['sides'].values():
            side['configuration'].update(prefill_device='CPU', stage_policy='fixed')
            side['configuration']['sources'].update(prefill_device='platform', stage_policy='platform')
        saved = copy.deepcopy(baseline)
        self.assertEqual(report.compare(baseline, candidate)[0], 0)
        self.assertEqual(report.compare(candidate, baseline)[0], 0)
        self.assertEqual(baseline, saved)
        candidate['sides']['A']['configuration']['prefill_device'] = 'MTL0'
        code, messages = report.compare(baseline, candidate)
        self.assertEqual(code, 2)
        self.assertIn('effective configuration', messages[0])
        candidate['sides']['A']['configuration']['prefill_device'] = 'CPU'
        candidate['sides']['A']['configuration']['sources']['prefill_device'] = 'request'
        self.assertEqual(report.compare(baseline, candidate)[0], 2)
        candidate['sides']['A']['configuration']['sources']['prefill_device'] = 'platform'
        candidate['sides']['A']['configuration']['stage_policy'] = 'other'
        self.assertEqual(report.compare(baseline, candidate)[0], 2)

    def test_failed_cell_is_input_error(self):
        candidate = make()
        candidate['cells'][4] = report.failed_cell(5, 'A', 'exit code 1')
        self.assertEqual(self.code(candidate), 2)
        self.assertEqual(report.compare(candidate, make())[0], 2)

    def test_missing_identity_and_bad_format_are_input_errors(self):
        candidate = make()
        del candidate['identity']['binary_sha256']
        self.assertEqual(self.code(candidate), 2)
        candidate = make()
        candidate['format'] = 'other'
        self.assertEqual(self.code(candidate), 2)
        self.assertEqual(self.code({}), 2)

    def test_baseline_without_noise_is_input_error(self):
        baseline = make()
        baseline['noise'] = None
        self.assertEqual(self.code(make(), baseline), 2)

    def test_other_kinds_are_not_compared(self):
        a, b = make(), make()
        a['protocol']['kind'] = b['protocol']['kind'] = 'turns'
        self.assertEqual(report.compare(a, b)[0], 2)
        self.assertEqual(report.compare(make(), b)[0], 2)

    def test_command_line_exit_codes(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            base, good, bad = (Path(directory) / n for n in ('base.json', 'good.json', 'bad.json'))
            base.write_text(json.dumps(make()))
            good.write_text(json.dumps(make(decode=(19.9,) * 4)))
            bad.write_text(json.dumps(make(decode=(18,) * 4)))
            run = lambda cand: subprocess.run([sys.executable, '-I', str(HERE / 'report.py'), 'compare', '--baseline', str(base),
                                               '--candidate', str(cand)], capture_output=True, text=True).returncode
            self.assertEqual((run(good), run(bad), run(Path(directory) / 'missing.json')), (0, 1, 2))


def make_sweep(decode_predicted=(20, 20, 20), decode_actual=(14, 14, 14), extra_failed=False, noise=None):
    cells, index = [], 0
    for side, values in (('predicted', decode_predicted), ('actual', decode_actual), ('direct', (15, 15, 15))):
        for d in values:
            index += 1
            cells.append(report.ok_cell(index, side, dict(first_token_seconds=6.0, prefill_tokens_per_second=514.0,
                                                          decode_tokens_per_second=float(d)),
                                        dict(ledger_peak_bytes=1, rss_peak_bytes=2, physical_peak_bytes=3, memory_snapshot={}), {}, {}))
    if extra_failed:
        cells[-1] = report.failed_cell(index, 'direct', 'exit code 1')
    protocol = dict(PROTOCOL, kind='sweep', order=['predicted', 'actual', 'direct'])
    sides = {n: dict(command=['x'], configuration={'name': n}) for n in ('predicted', 'actual', 'direct')}
    full = dict(make(), protocol=protocol, sides=sides, cells=cells, noise=noise)
    return full


class FixedThresholdTest(unittest.TestCase):
    def baseline(self, **options):
        return report.make_baseline(make_sweep(**options), ['predicted', 'actual'], .05)

    def test_baseline_keeps_two_sides_without_snapshots(self):
        base = self.baseline()
        self.assertEqual(sorted(base['sides']), ['actual', 'predicted'])
        self.assertEqual({c['side'] for c in base['cells']}, {'actual', 'predicted'})
        self.assertTrue(all(c['snapshot_before'] is None and 'memory_snapshot' not in c['memory'] for c in base['cells']))
        self.assertEqual(base['fixed_threshold'], .05)
        self.assertLess(len(json.dumps(base)), 256 * 1024)

    def test_same_report_passes_and_ignores_other_sides_failures(self):
        self.assertEqual(report.compare(self.baseline(), make_sweep())[0], 0)
        self.assertEqual(report.compare(self.baseline(), make_sweep(extra_failed=True))[0], 0)

    def test_fixed_threshold_decides_without_noise(self):
        self.assertEqual(report.compare(self.baseline(), make_sweep(decode_predicted=(18.8,) * 3))[0], 1)   # 6% > 5%
        self.assertEqual(report.compare(self.baseline(), make_sweep(decode_predicted=(19.2,) * 3))[0], 0)   # 4% < 5%

    def test_larger_of_noise_and_threshold_applies(self):
        base = self.baseline()
        base['noise'] = {'prefill_seconds': .02, 'decode_tokens_per_second': .08}
        self.assertEqual(report.compare(base, make_sweep(decode_predicted=(18.6,) * 3))[0], 0)             # 7% < 8%
        self.assertEqual(report.compare(base, make_sweep(decode_predicted=(18.0,) * 3))[0], 1)             # 10% > 8%

    def test_majority_of_three_processes_is_required(self):
        self.assertEqual(report.compare(self.baseline(), make_sweep(decode_predicted=(10, 10, 21)))[0], 1)  # median 10, two worse
        self.assertEqual(report.compare(self.baseline(), make_sweep(decode_predicted=(10, 20, 21)))[0], 0)  # median 20

    def test_protocol_change_is_refused(self):
        candidate = make_sweep()
        candidate['protocol']['cooldown_seconds'] = 10
        self.assertEqual(report.compare(self.baseline(), candidate)[0], 2)

    def test_candidate_without_baseline_side_is_refused(self):
        candidate = make_sweep()
        del candidate['sides']['actual']
        candidate['cells'] = [c for c in candidate['cells'] if c['side'] != 'actual']
        self.assertEqual(report.compare(self.baseline(), candidate)[0], 2)

    def test_neither_noise_nor_threshold_is_refused(self):
        base = self.baseline()
        base['fixed_threshold'] = None
        self.assertEqual(report.compare(base, make_sweep())[0], 2)

    def test_baseline_cli_and_compare_cli(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            source, out = Path(directory) / 's.json', Path(directory) / 'b' / 'base.json'
            source.write_text(json.dumps(make_sweep()))
            run = lambda *a: subprocess.run([sys.executable, '-I', str(HERE / 'report.py'), *a], capture_output=True, text=True).returncode
            self.assertEqual(run('baseline', '--report', str(source), '--sides', 'predicted', 'actual', '--threshold', '0.05', '--out', str(out)), 0)
            self.assertEqual(run('baseline', '--report', str(source), '--sides', 'predicted', '--threshold', '0.05', '--out', str(out)), 2)  # exists
            self.assertEqual(run('compare', '--baseline', str(out), '--candidate', str(source)), 0)
            self.assertEqual(run('baseline', '--report', str(source), '--sides', 'nothing', '--threshold', '0.05', '--out', str(out) + '2'), 2)


class WriterTest(unittest.TestCase):
    def test_partial_then_final(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            writer = report.ReportWriter(Path(directory) / 'r')
            full = make()
            writer.start(full['identity'], full['protocol'], full['sides'])
            writer.add_cell(full['cells'][0])
            self.assertEqual(len(report.load_partial(writer.partial_path)['cells']), 1)
            writer.update_identity(engine={'version': '2'})
            self.assertEqual(report.load_partial(writer.partial_path)['identity']['engine'], {'version': '2'})
            writer.finish(full)
            self.assertFalse(writer.partial_path.exists())
            self.assertEqual(json.loads(writer.json_path.read_text())['format'], report.FORMAT)
            self.assertIn('Per-side summary', writer.md_path.read_text())
            with self.assertRaises(FileExistsError):
                report.ReportWriter(Path(directory) / 'r')

    def test_existing_partial_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            Path(directory, 'r.partial.jsonl').write_text('')
            with self.assertRaises(FileExistsError):
                report.ReportWriter(Path(directory) / 'r')


class RenderTest(unittest.TestCase):
    def test_render_does_not_change_inputs_and_copies_numbers(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            source, extra, out = (Path(directory) / n for n in ('r.json', 'extra.md', 'out.md'))
            source.write_text(json.dumps(make(decode=(21.5,) * 4)))
            extra.write_text('## Appended\n')
            before = source.read_bytes()
            code = report.main(['render', '--out', str(out), '--append', str(extra), str(source)])
            self.assertEqual(code, 0)
            self.assertEqual(source.read_bytes(), before)
            text = out.read_text()
            self.assertIn('21.500', text)
            self.assertTrue(text.rstrip().endswith('## Appended'))

    def test_render_rejects_unreadable_report(self):
        with tempfile.TemporaryDirectory(prefix='moe-report-') as directory:
            self.assertEqual(report.main(['render', '--out', str(Path(directory) / 'o.md'), str(Path(directory) / 'none.json')]), 2)


if __name__ == '__main__':
    unittest.main()
