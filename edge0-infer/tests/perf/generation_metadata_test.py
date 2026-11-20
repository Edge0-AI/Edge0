"""Generation provenance through native report writers, baselines and comparison."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report
import abba
import abba_test
import sweep
import sweep_test

GENERATION = dict(temperature=0.0, top_k=20, top_p=0.8, max_tokens=111,
                  seed=None, thinking=False, sources=dict(temperature='application', top_p='model'))


def with_generation(output, generation):
    rows = [json.loads(line) for line in output.splitlines()]
    for row in rows:
        if generation is not None:
            row['generation'] = copy.deepcopy(generation)
    return '\n'.join(json.dumps(row) for row in rows) + '\n'


class EffectiveConfigurationTest(unittest.TestCase):
    def test_preserves_generation_and_runtime_without_aliasing(self):
        summary = dict(model=dict(loading=dict(device='cpu')),
                       runtime=dict(cpu_threads_source='platform_default'),
                       generation=dict(thinking=False, seed=None,
                                       sources=dict(thinking='model')))
        original = copy.deepcopy(summary)
        actual = report.effective_configuration(summary)
        self.assertEqual(actual, dict(device='cpu', runtime=summary['runtime'],
                                      generation=summary['generation']))
        actual['generation']['sources']['thinking'] = 'application'
        actual['runtime']['cpu_threads_source'] = 'request'
        self.assertEqual(summary, original)

    def test_legacy_metadata_stays_absent(self):
        summary = dict(model=dict(loading=dict(device='cpu')))
        actual = report.effective_configuration(summary)
        self.assertEqual(actual, dict(device='cpu'))
        self.assertNotIn('generation', actual)
        self.assertNotIn('runtime', actual)

    def test_normalization_preserves_distinct_observed_generation(self):
        first = report.effective_configuration(dict(model=dict(loading=dict(device='cpu')),
                                                   generation=dict(thinking=False)))
        second = copy.deepcopy(first)
        second['generation']['thinking'] = True
        self.assertNotEqual(report.normalized_configuration(first),
                            report.normalized_configuration(second))
        legacy = report.effective_configuration(dict(model=dict(loading=dict(device='cpu'))))
        self.assertNotEqual(report.normalized_configuration(first),
                            report.normalized_configuration(legacy))


class SavedReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='moe-generation-report-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'fixture executable identity')

    def measure(self, kind, name, generation):
        writer = report.ReportWriter(self.root / name)
        if kind == 'sweep':
            output = with_generation(sweep_test.bench_stdout(), generation)
            host = sweep_test.FakeHost(output)
            protocol = dict(kind='sweep', backend='metal',
                            prompt=dict(file=sweep.PROMPT, sha256='fixture', input_count=3084),
                            warmup=10, ntok=100, runs=1, runs_per_configuration=3,
                            order=['predicted'], cooldown_seconds=0)
            result = sweep.run_sweep(host, self.binary, Path('model'), protocol, writer,
                                    raw_dir=self.root / (name + '-raw'), tools=[self.binary],
                                    root=self.root, runs=3, sleep=lambda seconds: None)
        else:
            output = with_generation(abba_test.bench_stdout(), generation)
            host = abba_test.FakeHost([output] * 8)
            command = [str(self.binary), 'bench', '--model', 'model']
            protocol = dict(kind='abba', backend='metal',
                            prompt=dict(file='prompt', sha256='fixture', input_count=3084),
                            warmup=10, ntok=100, runs=1, order=abba.ORDER, cooldown_seconds=0)
            result = abba.run_measurement(host, dict(A=command, B=command), protocol, writer,
                                         model=Path('model'), raw_dir=self.root / (name + '-raw'),
                                         tools=[self.binary], root=self.root, sleep=lambda seconds: None,
                                         noise_only=True)
        saved = report.load(writer.json_path)
        self.assertEqual(saved, result)
        self.assertFalse(writer.partial_path.exists())
        for side in saved['sides'].values():
            if generation is None:
                self.assertNotIn('generation', side['configuration'])
            else:
                self.assertEqual(side['configuration']['generation'], generation)
        for cell in saved['cells']:
            if generation is not None:
                self.assertEqual(cell['configuration']['generation'], generation)
        return saved, writer.json_path

    def test_saved_sweep_and_abba_compare_generation_and_missing_metadata(self):
        for kind in ('sweep', 'abba'):
            with self.subTest(kind=kind):
                first, _ = self.measure(kind, kind + '-first', GENERATION)
                same, same_path = self.measure(kind, kind + '-same', GENERATION)
                sampled = dict(GENERATION, temperature=0.8, seed=7)
                changed, changed_path = self.measure(kind, kind + '-sampled', sampled)
                legacy, legacy_path = self.measure(kind, kind + '-legacy', None)
                baseline = report.make_baseline(first, list(first['sides']), 0.05)
                baseline_path = self.root / (kind + '-baseline.json')
                baseline_path.write_text(json.dumps(baseline))
                for candidate, path, expected in ((same, same_path, 0),
                                                  (changed, changed_path, 2),
                                                  (legacy, legacy_path, 2)):
                    self.assertEqual(report.compare(baseline, candidate)[0], expected)
                    command = subprocess.run([sys.executable, '-I', str(Path(report.__file__)), 'compare',
                                              '--baseline', str(baseline_path), '--candidate', str(path)],
                                             capture_output=True, text=True)
                    self.assertEqual(command.returncode, expected, command.stdout + command.stderr)
                    if expected == 2:
                        self.assertIn('effective configuration', command.stdout + command.stderr)
                legacy_baseline = report.make_baseline(legacy, list(legacy['sides']), 0.05)
                self.assertEqual(report.compare(legacy_baseline, same)[0], 2)
                self.assertEqual(report.compare(legacy_baseline, legacy)[0], 0)

    def test_run_summary_generation_disagreement_is_refused(self):
        rows = [json.loads(line) for line in with_generation(abba_test.bench_stdout(), GENERATION).splitlines()]
        for mismatch in ('changed', 'missing_run', 'missing_summary', 'null', 'array'):
            bad = copy.deepcopy(rows)
            if mismatch == 'changed':
                bad[0]['generation']['seed'] = 7
            elif mismatch in ('missing_run', 'missing_summary'):
                del bad[0 if mismatch == 'missing_run' else 1]['generation']
            else:
                for row in bad:
                    row['generation'] = None if mismatch == 'null' else []
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(ValueError, 'generation configurations differ'):
                abba.parse_output('\n'.join(json.dumps(row) for row in bad))

    def test_sweep_generation_drift_is_preserved_as_failure(self):
        host = sweep_test.FakeHost(lambda index, argv: with_generation(
            sweep_test.bench_stdout(), GENERATION if index == 0 else dict(GENERATION, seed=7)))
        writer = report.ReportWriter(self.root / 'drift')
        protocol = dict(kind='sweep', backend='metal',
                        prompt=dict(file=sweep.PROMPT, sha256='fixture', input_count=3084),
                        warmup=10, ntok=100, runs=1, runs_per_configuration=3,
                        order=['predicted'], cooldown_seconds=0)
        result = sweep.run_sweep(host, self.binary, Path('model'), protocol, writer,
                                raw_dir=self.root / 'drift-raw', tools=[self.binary],
                                root=self.root, runs=3, sleep=lambda seconds: None)
        self.assertEqual([cell['status'] for cell in result['cells']], ['ok', 'failed', 'failed'])
        self.assertEqual(result['summary']['predicted']['failed'], 2)
        self.assertTrue(all('effective configuration changed' in cell['reason'] for cell in result['cells'][1:]))
        self.assertEqual(report.load(writer.json_path), result)


if __name__ == '__main__':
    unittest.main()
