"""Offline checks for image/steering isolation evidence and generated fixtures."""
import copy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_isolation import STATE_KEYS, fixtures, verify
from tools.gguf_reader import GGUFFile


def evidence(scenario):
    image = scenario == 'image'
    names = (['image-A', 'text-B', 'image-A-return', 'image-B', 'image-A-again', 'grid-changed', 'image-A-final']
             if image else ['off', 'on', 'text-B', 'off-return', 'on-return', 'off-again'])
    hits = [2, 4, 6] if image else [3, 4, 5]
    records = [{'name': name, 'ids': [123], 'finish': 'length', 'reused': 0,
                'state': {k: str(i) for k in STATE_KEYS}} for i, name in enumerate(names)]
    candidate = copy.deepcopy(records)
    for i in hits:
        candidate[i]['reused'] = 100
    info = dict(expert_slots=8000, kv='int8', kv_resident=32768, context=262144,
                spec=2, mtp_max=1, lookup=0, cvec=0 if image else 'add:1-47')
    return {'baseline': records, 'candidate': candidate,
            'engine_info': {'baseline': dict(info), 'candidate': dict(info)}}


class IsolationGate(unittest.TestCase):
    def test_complete_evidence(self):
        for scenario in ('image', 'add', 'project'):
            verify(evidence(scenario), scenario)

    def test_incomplete_or_wrong_evidence_rejected(self):
        for scenario in ('image', 'add'):
            cases = {
                'incompatible reuse': lambda d: d['candidate'][1].update(reused=5),
                'no parked hit': lambda d: d['candidate'][-1].update(reused=0),
                'state mismatch': lambda d: d['candidate'][-1]['state'].update(gdn='ffff'),
                'missing state': lambda d: d['candidate'][-1]['state'].clear(),
                'empty output': lambda d: d['candidate'][-1]['ids'].clear(),
                'missing request': lambda d: d['candidate'].pop(),
                'different residency': lambda d: d['engine_info']['candidate'].update(expert_slots=1),
            }
            for name, mutate in cases.items():
                with self.subTest(scenario=scenario, case=name):
                    data = evidence(scenario)
                    mutate(data)
                    with self.assertRaises(AssertionError):
                        verify(data, scenario)

    def test_generated_fixtures(self):
        with tempfile.TemporaryDirectory(prefix='strata-isolation-') as directory:
            p = Path(directory)
            fixtures(p, 2560)
            cv = GGUFFile(p / 'control.gguf')
            self.assertEqual(cv.metadata['general.architecture'], 'controlvector')
            self.assertEqual([t.name for t in cv.tensors], ['direction.1', 'direction.20'])
            self.assertTrue(all(t.elements == 2560 for t in cv.tensors))
            a, b, grid = [(p / (name + '.sve')).read_bytes() for name in ('image-a', 'image-b', 'image-grid')]
            self.assertEqual(struct.unpack('<5i', a[:20]), (0x31455653, 4, 2, 2, 2560))
            self.assertEqual(len(a), 20 + 4 * 2560 * 4)
            self.assertEqual(a[:20], b[:20])
            self.assertNotEqual(a[20:], b[20:])
            self.assertNotEqual(a[:20], grid[:20])
            self.assertEqual(a[20:], grid[20:])

    def test_dry_run_does_not_read_missing_config(self):
        with tempfile.TemporaryDirectory(prefix='strata-isolation-dry-') as directory:
            p = Path(directory)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('conversation_cache_isolation.py')),
                                     '--config', str(p / 'missing.json'), '--engine', str(p / 'missing'),
                                     '--output', str(p / 'not-created'), '--scenario', 'image'],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('no model loaded', result.stdout)
            self.assertFalse((p / 'not-created').exists())


if __name__ == '__main__':
    unittest.main()
