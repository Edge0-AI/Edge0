"""Logprobs with "parallel" (batch slots): python -m unittest serve.test_logprobs_parallel.

The fake engine of serve/test_parallel.py behind the real StrataEngine and Service, writing an LP line after each
T line when the GEN / BGEN line asks for logprobs.  A typed decision (max_tokens 1) is read on the control lines
(an admission: BGEN, one T, DONE, BADM 0) and gets its logprobs; longer replies are refused with a 400, since the
slot's BT lines carry none.
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import serve.test_parallel as tp

# LP after each T when asked: the token's log-probability is -(prompt length) / 1000 (so two requests can be told
# apart) and "A" (65) is its one alternative at -1.5 -- or the named ids themselves, at -1, when the line carries
# `allowed=` (the engine's own branch: it prints those and no top-K)
_T_LINE = '            print(f"T {t}", flush=True)\n'
assert tp.FAKE_BATCH.count(_T_LINE) == 1
FAKE_LP = tp.FAKE_BATCH.replace(_T_LINE, _T_LINE + (
    '            if " logprobs=" in line:\n'
    '                al = [x[8:] for x in line.split() if x.startswith("allowed=")]\n'
    '                cs = (" " + " ".join(i + ":-1.000000" for i in al[0].split(","))) if al else " 65:-1.500000"\n'
    '                print(f"LP {-len(ids) / 1000:.6f}" + cs, flush=True)\n'))


class LogprobsWithBatchSlots(unittest.TestCase):
    start, tearDown, get = tp.ParallelService.start, tp.ParallelService.tearDown, tp.ParallelService.get

    def start_lp(self, fake=FAKE_LP, slots=2):
        with mock.patch.object(tp, "FAKE_BATCH", fake):
            self.start(slots)

    def chat(self, text, max_tokens=1, logprobs=True, allowed=None):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none"}
        if logprobs:
            body |= {"logprobs": True, "top_logprobs": 1}
        if allowed:
            body["allowed_token_ids"] = allowed
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def entries(reply):
        return ((reply["choices"][0].get("logprobs") or {}).get("content")) or []

    def test_a_decision_gets_its_logprobs(self):
        self.start_lp()
        r = self.chat("Is it A or B?")
        self.assertEqual(r["choices"][0]["message"]["content"], "o")
        e = self.entries(r)
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["token"], "o")
        self.assertLess(e[0]["logprob"], 0)
        self.assertEqual(e[0]["top_logprobs"], [{"token": "A", "logprob": -1.5, "bytes": [65]}])
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))   # an admission

    def test_the_named_ids_reach_the_slot_and_are_what_is_reported(self):
        # allowed_token_ids travels in the same BGEN keys as logprobs, so the engine's own branch answers with
        # exactly those ids: the admission path carries the whole feature, not just the top-K one
        self.start_lp()
        e = self.entries(self.chat("Is it A or B?", allowed=[65]))
        self.assertEqual(len(e), 1)
        self.assertEqual([x["token"] for x in e[0]["top_logprobs"]], ["A"])
        self.assertEqual(e[0]["top_logprobs"][0]["logprob"], 0)      # the whole distribution: just that id

    def test_two_at_once_each_get_their_own(self):
        self.start_lp()
        short, long_ = "A or B?", "A or B? " + "x" * 300
        alone = {t: self.entries(self.chat(t))[0]["logprob"] for t in (short, long_)}
        self.assertNotEqual(alone[short], alone[long_])
        for _ in range(5):
            got, errs = {}, []

            def run(t):
                try:
                    got[t] = self.entries(self.chat(t))[0]["logprob"]
                except Exception as exc:          # noqa: BLE001 - reported below
                    errs.append(repr(exc))
            th = [threading.Thread(target=run, args=(t,)) for t in (short, long_, short, long_)]
            for x in th:
                x.start()
            for x in th:
                x.join(60)
            self.assertEqual(errs, [])
            self.assertEqual(got, alone)

    def test_not_asked_nothing_changes(self):
        self.start_lp()
        r = self.chat("Is it A or B?", logprobs=False)
        self.assertEqual(r["choices"][0]["message"]["content"], "o")
        self.assertIsNone(r["choices"][0].get("logprobs"))

    def test_longer_replies_are_refused_with_batch_slots(self):
        self.start_lp()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.chat("Is it A or B?", max_tokens=8)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("max_tokens 1", cm.exception.read().decode())
        # the server is still serving, and a reply that does not ask is untouched
        self.assertEqual(self.chat("still serving?", max_tokens=16, logprobs=False)["choices"][0]["message"]["content"],
                         "ok, done.")

    def test_an_engine_without_lp_lines(self):
        self.start_lp(fake=tp.FAKE_BATCH)          # an older engine: the key is ignored, no LP line at all
        r = self.chat("Is it A or B?")
        self.assertEqual(r["choices"][0]["message"]["content"], "o")
        self.assertEqual(self.entries(r), [])


if __name__ == "__main__":
    unittest.main()
