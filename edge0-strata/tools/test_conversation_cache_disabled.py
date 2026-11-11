"""The upstream comparison must reject mismatches and incomplete evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_disabled import NAMES, SETTINGS, STATE_KEYS, common_args, state_hashes, verify


def fixture():
    requests = [dict(name=n, ids=[1, 2], max_new=16, expected='4' if n == 'short' else None) for n in NAMES]
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    baseline = dict(name='upstream', args=['--kv', 'int8'], info={key: 1 for key in SETTINGS},
                    request_digest=digest, cache_events=[], records=[])
    for name in NAMES:
        baseline['records'].append(dict(name=name, ids=[4], text='4', finish='stop', prompt_tokens=2,
                                        reused=2 if name in ('A-live', 'A-checkpoint') else 0,
                                        state={key: 'abc' for key in STATE_KEYS}))
    candidate = copy.deepcopy(baseline)
    candidate['name'] = 'candidate-1'
    candidate['info']['conversation_cache_mib'] = 0
    candidate['info']['conversation_disk_mib'] = 0
    return dict(arms=[baseline, candidate], requests=requests, request_digest=digest)


class DisabledCacheGateTest(unittest.TestCase):
    def test_reference_fixture(self):
        verify(fixture())

    def test_upstream_hash_schema(self):
        line = 'strata serve: STATE_HASH L=2 gdn=ab ple=cd tail=ef pooled=12 kv=34 mtp=56 stale=78 ple_prev=1,2'
        self.assertEqual(set(state_hashes(line)[0]), set(STATE_KEYS))
        self.assertEqual(state_hashes(line), state_hashes(line + ' dead=99'))
        # the candidate's pooled= is the upstream extent; pooled_full= (with the spare row) is not compared
        self.assertEqual(state_hashes(line), state_hashes(line + ' dead=99 pooled_full=ff', candidate=True))
        self.assertNotEqual(state_hashes(line), state_hashes(line.replace('pooled=12', 'pooled=ff') +
                                                           ' pooled_full=12', candidate=True))
        with self.assertRaises(AssertionError):
            state_hashes(line, candidate=True)
        with self.assertRaises(AssertionError):
            state_hashes(line.replace('kv=34 ', ''))

    def test_rejects_bad_evidence(self):
        mutations = [
            lambda r: r['arms'].pop(),
            lambda r: r['requests'].pop(),
            lambda r: r['requests'][0]['ids'].append(3),
            lambda r: r['arms'][1]['args'].append('--different'),
            lambda r: r['arms'][1].update(request_digest='different'),
            lambda r: r['arms'][1]['records'].pop(),
            lambda r: r['arms'][1]['info'].update(expert_slots=2),
            lambda r: r['arms'][1]['info'].pop('kv'),
            lambda r: r['arms'][1]['info'].update(conversation_cache_mib=8192),
            lambda r: r['arms'][1]['info'].pop('conversation_cache_mib'),
            lambda r: r['arms'][1]['info'].update(conversation_disk_mib=16384),
            lambda r: r['arms'][1]['cache_events'].append('disk cache: spilled'),
            lambda r: r['arms'][1]['records'][0].update(ids=[]),
            lambda r: r['arms'][1]['records'][0].update(ids=[5]),
            lambda r: r['arms'][1]['records'][0].update(finish='disconnect'),
            lambda r: r['arms'][1]['records'][0].update(prompt_tokens=1),
            lambda r: r['arms'][1]['records'][0]['state'].pop('gdn'),
            lambda r: r['arms'][1]['records'][0]['state'].update(kv='bad'),
            lambda r: r['arms'][1]['records'][0].update(reused=1),
            lambda r: [a['records'][0].update(text='5') for a in r['arms']],
            lambda r: [a['records'][2].update(reused=0) for a in r['arms']],
            lambda r: [a['records'][5].update(reused=2) for a in r['arms']],
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                result = fixture()
                mutation(result)
                with self.assertRaises(AssertionError):
                    verify(result)

    def test_optimized_python_still_rejects(self):
        code = ('from tools.test_conversation_cache_disabled import fixture, verify; '
                'r=fixture(); r["arms"][1]["records"][0]["ids"]=[5]; verify(r)')
        p = subprocess.run([sys.executable, '-O', '-c', code], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('output differs', p.stderr)

    def test_cache_options_omitted_from_shared_arguments(self):
        cfg = {'args': ['--pack', '/model', '--conversation-cache-mib', '8192',
                        '--conversation-cache-disk', '/cache', '--kv', 'int8']}
        args = common_args(cfg, 4)
        self.assertEqual(args[:4], ['--pack', '/model', '--kv', 'int8'])
        self.assertFalse(any(x.startswith('--conversation-cache-') for x in args))

    def test_dry_run_does_not_load_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'absent'
            p = subprocess.run([sys.executable, 'tools/conversation_cache_disabled.py', '--config', '/absent',
                                '--upstream', '/absent', '--candidate', '/absent', '--output', str(out)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
