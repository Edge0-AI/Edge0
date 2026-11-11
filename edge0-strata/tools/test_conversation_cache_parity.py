"""CPU-only checks that the model-level gate cannot pass with incomplete evidence."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_parity import STATE_KEYS, engine_args, pressure_budget_hint, verify_admission, verify_exchange, verify_pressure, verify_results


def fixture():
    state = {k: '1234' for k in STATE_KEYS}
    def record(name):
        return {'name': name, 'ids': [123], 'finish': 'length', 'state': dict(state), 'reused': 100}
    info = {'expert_slots': 100, 'kv': 'int8', 'kv_resident': 32768, 'context': 65536,
            'spec': 1, 'mtp_max': 0, 'lookup': 0, 'cvec': '0'}
    return {'baseline': [record('A'), record('A+')],
            'candidate': [record(n) for n in ('A', 'B', 'A+', 'B-again', 'A+-checkpoint')],
            'engine_info': {'baseline': dict(info), 'candidate': dict(info)}}


class ParityGate(unittest.TestCase):
    def test_native_iq_single_token_arguments(self):
        cfg = {'args': ['--spec', '4', '--mtp-max-t', '4']}
        args = engine_args(cfg, 400, 1)
        values = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
        self.assertEqual(values['--spec'], '2')
        self.assertEqual(values['--mtp-max-t'], '1')
        self.assertEqual(values['--suffix-draft'], '0')
        self.assertEqual(values['--adapt-swaps'], '0')
        self.assertEqual(values['--conversation-cache-mib'], '400')
        self.assertEqual(cfg['args'], ['--spec', '4', '--mtp-max-t', '4'])

    def test_complete_evidence(self):
        verify_results(fixture(), 100, 1)

    def test_incomplete_or_different_evidence_fails(self):
        cases = {
            'missing state': lambda d: d['candidate'][2]['state'].clear(),
            'missing spare key': lambda d: d['candidate'][2]['state'].pop('dead'),
            'empty output': lambda d: d['candidate'][2]['ids'].clear(),
            'missing request': lambda d: d['candidate'].pop(),
            'state differs': lambda d: d['candidate'][2]['state'].update(gdn='ffff'),
            'output differs': lambda d: d['candidate'][2].update(ids=[456]),
            'live not restored': lambda d: d['candidate'][2].update(reused=0),
            'checkpoint not restored': lambda d: d['candidate'][4].update(reused=0),
            'different residency': lambda d: d['engine_info']['candidate'].update(expert_slots=99),
            'cancellation': lambda d: d['candidate'][2].update(finish='cancel'),
            'initial output differs': lambda d: d['candidate'][0].update(ids=[789]),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                data = copy.deepcopy(fixture())
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify_results(data, 100, 1)

    def test_speculative_gate_still_requires_output_parity(self):
        data = fixture()
        data['candidate'][2]['state']['gdn'] = 'ffff'
        verify_results(data, 100, 4)
        data['candidate'][2]['ids'] = [456]
        with self.assertRaises(AssertionError):
            verify_results(data, 100, 4)

    def test_dry_run_does_not_read_config_or_start_engine(self):
        with tempfile.TemporaryDirectory(prefix='strata-parity-dry-') as directory:
            output = Path(directory) / 'not-created'
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('conversation_cache_parity.py')),
                                     '--config', str(Path(directory) / 'missing.json'),
                                     '--engine', str(Path(directory) / 'missing-engine'), '--output', str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No model loaded', result.stdout)
            self.assertFalse(output.exists())


def pressure_fixture():
    data = fixture()
    record = data['baseline'][0]
    for key in ('baseline', 'candidate'):
        data[key] = [{**copy.deepcopy(record), 'name': name, 'reused': 0}
                     for name in ('A', 'B', 'C', 'A-again')]
    data['engine_info']['candidate'].update(conversation_cache_mib=400, conversation_cache_slots=4)
    data['pressure'] = {'skips': 0, 'parks': [
        {'parked': 1, 'bytes': 266000000, 'evictions': n} for n in range(3)]}
    return data


class PressureGate(unittest.TestCase):
    def test_pressure_budget_hint_uses_individual_sizes(self):
        data = pressure_fixture()
        for park in data['pressure']['parks']:
            park['snapshot_bytes'] = 266 * 1024 * 1024
        self.assertIn('--cache-mib 266..531', pressure_budget_hint(data, 600))
        data['candidate'][-1]['reused'] = 100
        with self.assertRaisesRegex(AssertionError, r'Configured --cache-mib 600.*266\.00 MiB.*266\.\.531'):
            verify_pressure(data, 600)

    def test_pressure_budget_hint_handles_missing_or_incompatible_sizes(self):
        data = pressure_fixture()
        self.assertIn('No snapshot sizes', pressure_budget_hint(data, 600))
        for park, size in zip(data['pressure']['parks'], [1, 2, 10]):
            park['snapshot_bytes'] = size * 1024 * 1024
        self.assertIn('No whole-MiB budget', pressure_budget_hint(data, 600))

    def test_physical_memory_admission_evidence(self):
        data = pressure_fixture()
        data['pressure'] = {'memory_skips': 3, 'skips': 0, 'parks': []}
        data['engine_info']['candidate']['conversation_cache_min_free_mib'] = 999999
        verify_admission(data, 400, 999999)
        for mutate in (
            lambda d: d['pressure'].update(memory_skips=0, skips=3),
            lambda d: d['pressure'].update(memory_skips=2),
            lambda d: d['engine_info']['candidate'].update(conversation_cache_min_free_mib=2560),
            lambda d: d['candidate'][3].update(reused=10),
        ):
            broken = copy.deepcopy(data)
            mutate(broken)
            with self.assertRaises(AssertionError):
                verify_admission(broken, 400, 999999)

    def test_incoming_exchange_evidence(self):
        data = fixture()
        data['engine_info']['candidate']['conversation_cache_mib'] = 400
        data['pressure'] = {'skips': 2, 'parks': [
            {'parked': 1, 'bytes': 266000000, 'evictions': 0},
            {'parked': 1, 'bytes': 384000000, 'evictions': 0}]}
        verify_exchange(data, 100, 1, 400)
        for skips in (0, 1, 3):
            data['pressure']['skips'] = skips
            with self.assertRaises(AssertionError):
                verify_exchange(data, 100, 1, 400)

    def test_byte_eviction_evidence(self):
        verify_pressure(pressure_fixture(), 400)

    def test_oversized_evidence(self):
        data = pressure_fixture()
        data['engine_info']['candidate']['conversation_cache_mib'] = 1
        data['pressure'] = {'skips': 3, 'parks': []}
        verify_pressure(data, 1, oversized=True)
        data['pressure']['skips'] = 2
        with self.assertRaises(AssertionError):
            verify_pressure(data, 1, oversized=True)

    def test_insufficient_pressure_evidence_rejected(self):
        cases = {
            'slot pressure': lambda d: d['pressure']['parks'][1].update(parked=4),
            'no eviction': lambda d: d['pressure']['parks'][-1].update(evictions=0),
            'too large individually': lambda d: d['pressure'].update(skips=1),
            'missing snapshots': lambda d: d['pressure']['parks'].pop(),
            'over budget': lambda d: d['pressure']['parks'][0].update(bytes=500 * 1024 * 1024),
            'cache hit': lambda d: d['candidate'][-1].update(reused=100),
            'state mismatch': lambda d: d['candidate'][-1]['state'].update(gdn='ffff'),
            'empty state': lambda d: d['candidate'][-1]['state'].clear(),
            'wrong output': lambda d: d['candidate'][-1].update(ids=[456]),
            'wrong budget': lambda d: d['engine_info']['candidate'].update(conversation_cache_mib=8192),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                data = pressure_fixture()
                mutate(data)
                with self.assertRaises(AssertionError):
                    verify_pressure(data, 400)


if __name__ == '__main__':
    unittest.main()
