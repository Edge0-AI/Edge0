"""Offline checks for the synthetic known-answer soak acceptance gate."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_soak import STATE_KEYS, answer_text, verify


def evidence():
    lengths = [2048, 40000, 120000]
    def record(index):
        return {'ids': [100 + index], 'text': ['MANGO', 'CEDAR', 'RAVEN'][index],
                'finish': 'stop', 'reused': lengths[index], 'cache_bytes': 100000,
                'state': {key: str(index) for key in STATE_KEYS}}
    info = dict(expert_slots=8000, kv='int8', kv_resident=32768, context=131072,
                spec=2, mtp_max=1, lookup=0, cvec=0, conversation_cache_mib=8192)
    return {'requested_cycles': 30, 'engine_info': {'baseline': {**info, 'conversation_cache_mib': 0}, 'candidate': info},
            'contexts': [{'expected': record(i)['text'], 'prompt_tokens': lengths[i], 'baseline': record(i)} for i in range(3)],
            'cycles': [{'cycle': i, 'context': i % 3, **record(i % 3)} for i in range(30)]}


class SoakGate(unittest.TestCase):
    def test_complete_evidence(self):
        verify(evidence())

    def test_bad_or_incomplete_evidence(self):
        cases = {
            'missing cycle': lambda r: r['cycles'].pop(),
            'too few requested': lambda r: r.update(requested_cycles=29),
            'missing context': lambda r: r['contexts'].pop(),
            'duplicate lengths': lambda r: r['contexts'][0].update(prompt_tokens=40000),
            'reordered cycle': lambda r: r['cycles'][2].update(context=0),
            'no streaming': lambda r: r['engine_info']['candidate'].update(kv_resident=0),
            'not near context limit': lambda r: (r['engine_info']['baseline'].update(context=262144),
                                                 r['engine_info']['candidate'].update(context=262144)),
            'different GPU residency': lambda r: r['engine_info']['candidate'].update(expert_slots=7999),
            'no restore': lambda r: r['cycles'][9].update(reused=0),
            'wrong answer despite parity': lambda r: (r['contexts'][0]['baseline'].update(text='WRONG'),
                                                      r['cycles'][0].update(text='WRONG')),
            'wrong token output': lambda r: r['cycles'][5].update(ids=[999]),
            'missing spare fingerprint': lambda r: r['cycles'][4]['state'].pop('dead'),
            'different state': lambda r: r['cycles'][4]['state'].update(gdn='bad'),
            'cancelled': lambda r: r['cycles'][3].update(finish='cancel'),
            'over budget': lambda r: r['cycles'][3].update(cache_bytes=2**40),
            'retained allocation grows': lambda r: r['cycles'][-1].update(cache_bytes=200000),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                results = copy.deepcopy(evidence())
                mutate(results)
                with self.assertRaises(AssertionError):
                    verify(results)

    def test_answer_normalization_does_not_hide_extra_text(self):
        self.assertEqual(answer_text(' MANGO<|im_end|>\n'), 'mango')
        self.assertNotEqual(answer_text('The code is MANGO.'), 'mango')

    def test_dry_run_never_reads_config_or_starts_model(self):
        with tempfile.TemporaryDirectory(prefix='strata-soak-dry-') as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('conversation_cache_soak.py')),
                                     '--config', str(root / 'missing.json'), '--engine', str(root / 'missing'),
                                     '--output', str(root / 'output')], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No model loaded', result.stdout)
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
