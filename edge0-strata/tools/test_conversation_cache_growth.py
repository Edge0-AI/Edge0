"""Negative controls for the incremental capture model gate."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # runnable as tools/test_x.py from the repo root too

from tools.conversation_cache_growth import NAMES, SECRET, SETTINGS, STATE_KEYS, verify


def fixture():
    lengths = [8, 2, 12, 2, 16, 2, 20, 2, 12, 2, 15, 2]
    requests = [dict(name=name, ids=[1]*n, max_new=16, expected=name.startswith('A-'))
                for name, n in zip(NAMES, lengths)]
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    full = dict(name='full', exe_sha256='fixed', args=['--conversation-cache-mib','1'], request_digest=digest,
                info={key: 1 for key in SETTINGS}, draft_verifications=4,
                parks=[dict(bytes=100,snapshot_bytes=100,reused_kv_bytes=0) for _ in range(3)],
                records=[dict(name=name, ids=[1], text=SECRET, finish='stop', reused=1,
                              prompt_tokens=n, state={k:'abc' for k in STATE_KEYS}) for name,n in zip(NAMES,lengths)])
    full['info'].update(conversation_cache_mib=1,conversation_cache_slots=4)
    incremental = copy.deepcopy(full)
    incremental['name'] = 'incremental'
    for p in incremental['parks']:
        p['reused_kv_bytes'] = 50
    return dict(arms=[full,incremental],requests=requests,request_digest=digest,cache_mib=1)


class GrowthGate(unittest.TestCase):
    def test_complete_evidence(self):
        verify(fixture())

    def test_incomplete_or_incorrect_evidence_fails(self):
        mutations = [
            lambda r:r['requests'].pop(),
            lambda r:r['requests'][0]['ids'].append(2),
            lambda r:r['arms'][1].update(exe_sha256='different'),
            lambda r:r.update(cache_mib=2),
            lambda r:r['arms'][1]['records'].pop(),
            lambda r:r['arms'][1]['records'][2].update(ids=[2]),
            lambda r:r['arms'][1]['records'][2].update(text='wrong'),
            lambda r:r['arms'][1]['records'][2].update(reused=0),
            lambda r:r['arms'][1]['records'][2].update(finish='cancel'),
            lambda r:r['arms'][1]['records'][2]['state'].pop('kv'),
            lambda r:r['arms'][1]['info'].update(expert_slots=2),
            lambda r:r['arms'][1].update(draft_verifications=0),
            lambda r:r['arms'][0]['parks'][0].update(reused_kv_bytes=1),
            lambda r:[p.update(reused_kv_bytes=0) for p in r['arms'][1]['parks']],
            lambda r:r['arms'][1]['parks'][0].update(bytes=2*1024*1024),
            lambda r:r['arms'][1]['parks'][0].update(reused_kv_bytes=101),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                result = fixture()
                mutate(result)
                with self.assertRaises(AssertionError):
                    verify(result)

    def test_optimized_python_rejects_missing_reuse(self):
        code = ('from tools.test_conversation_cache_growth import fixture,verify; '
                'r=fixture(); [p.update(reused_kv_bytes=0) for p in r["arms"][1]["parks"]]; verify(r)')
        run = subprocess.run([sys.executable,'-O','-c',code],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('did not reuse KV bytes',run.stderr)

    def test_dry_run_leaves_server_and_output_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'absent'
            run = subprocess.run([sys.executable,'tools/conversation_cache_growth.py','--config','/absent',
                                  '--engine','/absent','--output',str(output)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
