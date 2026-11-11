"""#465 (part 2): the parallel slots when a long prompt gives way, a window fails, a slot leaks or a conversation returns.

The same fake engine and helpers as serve/test_parallel.py (FakeEngineCase); split out because the whole file took
about two minutes in one piece."""
import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock
from serve import test_parallel as tp


class ParallelServiceYield(tp.FakeEngineCase):
    """Requests that give way to each other, and what happens to a slot when the engine misbehaves."""

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_the_way_is_asked_between_far_apart_progress_lines(self):
        """#656 when the PP lines come far apart (the pipelined groups' read in pieces of one chunk: one PP per piece,
        at its end): the BYIELD goes out while the engine is quiet, so the read gives way at the first boundary after
        the short request came.  Asked only at a PP line, it reached the engine after that boundary's check and the
        read gave way one piece later."""
        self.start(2, more=["--pp-every", "80"])   # a boundary every 80 chunks (~0.8 s): one in this read
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        yields = [x for x in self.log.read_text().splitlines() if x.startswith("YIELD ")]
        self.assertEqual([int(x.split()[2]) for x in yields], [80 * 32], yields)

    def test_three_requests_long_long_short_do_not_deadlock(self):
        """ENGINE_REVIEW finding 1 (0.1.41 check): two long prompts fill both slots, a short one arrives a moment later
        and wants one of them (the yield path).  The reviewer's repro: long, long, short at 0.25 s steps.  Every
        request must be answered; none may wait on the control lock that the yielding read holds."""
        self.start(2)
        texts = ["long " * 600, "lung " * 600, "short"]
        res, errors = {}, []

        def go(t):
            t0 = time.time()
            try:
                r = self.chat(t, max_tokens=64)
                res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
            except Exception as e:      # noqa: BLE001 - a timeout is the failure this test looks for
                errors.append((t[:6], repr(e)))
        threads = []
        for t in texts:
            th = threading.Thread(target=go, args=(t,))
            th.start()
            threads.append(th)
            time.sleep(0.25)
        for th in threads:
            th.join(45)
        self.assertFalse(any(th.is_alive() for th in threads), "a request never finished (deadlock)")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 3)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_an_admission_is_asked_between_far_apart_progress_lines(self):
        """As above for a read admitted beside a decoding slot (BGEN, `ask()`): with one PP line per 80 chunks the
        short request's BYIELD still reaches the first boundary."""
        self.start(3, more=["--pp-every", "80"])
        bg = threading.Thread(target=lambda: self.chat("LONGREPLY background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        log = self.log.read_text().splitlines()
        admitted = [x.split() for x in log if x.startswith("BGEN ") and int(x.split()[2]) > 2600]
        self.assertTrue(admitted, log)                                 # the long read was an admission
        yields = [x.split()[1:] for x in log if x.startswith("YIELD ")]
        self.assertEqual(yields[:1], [[admitted[0][1], str(80 * 32)]], log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_no_way_given_to_a_long_one_between_far_apart_progress_lines(self):
        """The ask between PP lines keeps the rule: a waiting request about as long gets no BYIELD there either."""
        self.start(2, more=["--pp-every", "80"])
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # #997 #890: a batch window fails while every request reads its own slot's lines, so no request reads the
        # engine's ERR line before it exits; the server's note on the death gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_nonfinite_in_batch_mode(self):
        """#879 in --batch mode: DONE ... nonfinite on the solo path or at an admission (nothing emitted yet) sends the
        request again once; BDONE ... nonfinite in a slot ends the reply that had started, as "length"."""
        self.start(2)
        self.assertEqual(self.chat("NFSOLO alone")["choices"][0]["message"]["content"], "ok, done.")
        gens = [x for x in self.log.read_text().splitlines() if x.startswith("GEN")]
        self.assertEqual(len(gens), 2, gens)                              # sent again once
        res = self.race("first question LONGREPLY", "NFADMIT second")
        self.assertEqual(res["NFADMIT second"][0], "ok, done.")
        self.assertEqual(res["first question LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        out = {}

        def go(t):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 64, "reasoning_effort": "none"}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out[t] = json.loads(r.read().decode())["choices"][0]
        a = threading.Thread(target=go, args=("third question LONGREPLY",))
        a.start()
        time.sleep(0.15)
        go("NFSLOT fourth")
        a.join(30)
        c = out["NFSLOT fourth"]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok", "length"))
        self.assertEqual(out["third question LONGREPLY"]["message"]["content"], "ok, " + "la " * 15 + "done.")
        # the engine dropped that slot's state: the server does not route the conversation's next turn to it
        self.assertFalse([h for h in self.engine.slot_held if bytes(t & 255 for t in h).find(b"NFSLOT") >= 0])
        with self.svc.status_lock:
            self.assertEqual(self.svc.live_reqs, {})

    def test_a_continued_request_reports_its_own_prompt_reuse(self):
        """A request STOPped on the solo path and continued in a slot (or back on the solo path) is sent again as its
        prompt + what it generated, and the engine holds all of that: the usage must still split the request's OWN
        prompt (input_tokens + cache_read_input_tokens = the prompt, input_tokens >= 1).  input_tokens 0 makes Claude
        Code count the prompt twice (message_start's input_tokens + cache_read_input_tokens): "Prompt is too long"
        at half the context."""
        self.start(2, slot_cache=True, reuse=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.message("LONGREPLY please", 200)))
        long.start()
        time.sleep(0.05)
        self.message("short", 200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)                    # it was continued: the case under test
        u = res["long"]["usage"]
        prompt = self.svc.history[[h["output_tokens"] for h in self.svc.history].index(
            max(h["output_tokens"] for h in self.svc.history))]["prompt_total"]
        self.assertGreaterEqual(u["input_tokens"], 1, u)
        self.assertEqual(u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"], prompt, u)
        self.assertEqual(u["cache_read_input_tokens"], 3, u)    # the first read's reuse, not the continuation's

    def test_a_slot_request_ended_by_its_stop_token_reports_the_slot_leg(self):
        """#1869: the consumer leaves at the stop token, before the BDONE: the request kept the admission's one token and
        its few ms.  Its figures are the slot leg's now (finish=stop was the broken case, finish=length read the BDONE)."""
        import serve.server as server
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):          # the fake engine's stop token
            a = threading.Thread(target=lambda: res.setdefault("a", self.chat("List three primes please", 200)))
            b = threading.Thread(target=lambda: res.setdefault("b", self.chat("Name two rivers in Germany", 200)))
            a.start(); b.start(); a.join(30); b.join(30)
        self.assertEqual(len(self.svc.history), 2)
        for h in self.svc.history:
            self.assertEqual(h["engine_generated"], h["output_tokens"], h)
            self.assertGreaterEqual(h["decode_ms"], 1.0, h)
        for r in res.values():
            t = r["timings"]
            self.assertEqual(t["prompt_n"] + t["cache_n"], r["usage"]["prompt_tokens"], t)
            self.assertEqual(t["predicted_n"], 10, t)
        self.assertFalse(any(self.engine.slot_busy))                 # the slots are free (no BDONE left for a wait)

    def test_a_request_moved_between_solo_and_slot_keeps_its_totals(self):
        """#1869/#1870: a request that read its prompt and decoded on the solo path, was STOPped when another arrived and
        went on in a slot (its admission copies the held prefix: prompt_read 1, one token) reports the prompt it read and
        the decode time and tokens of ALL its legs, not the last one's."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.25)                                   # a few solo tokens first
        self.chat("short", max_tokens=200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)           # solo, then continued in a slot: the case under test
        n = res["long"]["usage"]["completion_tokens"]
        row = max(self.svc.history, key=lambda h: h["output_tokens"])
        self.assertEqual(row["engine_generated"], n, row)
        self.assertGreater(row["decode_ms"], 20.0 * 3, row)          # the solo leg's STEP-long tokens are in it
        self.assertGreater(row["prompt_read"], 1, row)                # the first read, not the copy leg's 1
        t = res["long"]["timings"]
        self.assertEqual(t["prompt_n"] + t["cache_n"], res["long"]["usage"]["prompt_tokens"], t)
        self.assertEqual(t["cache_n"], 3, t)                          # the first read's reuse only
        self.assertGreater(t["predicted_ms"], 60.0, t)

    def test_a_leaked_busy_slot_does_not_hang_the_next_request(self):
        """#1603: a slot whose busy flag has no owner (no request holds it, no BDONE wait is running) kept every later
        request waiting for a slot nothing would free: running 0, queued 0, and the frozen-engine check never looked.
        The waiting request is now counted (admitting), the slot is given back after ORPHAN_SLOT_S, and it is
        answered."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.4
        with e.slot_cv:
            e.slot_busy[0] = e.slot_busy[1] = True       # the leak: busy, claimed by nobody, nothing ending them
        # the solo path needs a quiet engine; a busy flag sends the request to the slot path, which waits
        res, errs = {}, []
        def go():
            try:
                res["r"] = self.chat("after the leak", max_tokens=16)
            except Exception as ex:                       # noqa: BLE001
                errs.append(ex)
        t = threading.Thread(target=go)
        t0 = time.time()
        t.start()
        seen = []
        while t.is_alive() and time.time() - t0 < 20:
            live = self.get("/metrics")["live"]
            seen.append((live.get("running"), live.get("queued"), live.get("admitting")))
            time.sleep(0.05)
        t.join(5)
        self.assertEqual(errs, [])
        self.assertFalse(t.is_alive(), "the request hung on a slot nobody owns")
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        # while it waited it was visible: running as a request, and counted as waiting for a slot - never 0 / 0 / 0
        self.assertTrue(any(r == 1 and a == 1 for r, q, a in seen), seen)
        first = next(i for i, x in enumerate(seen) if sum(v or 0 for v in x))
        self.assertTrue(all(sum(v or 0 for v in x) for x in seen[first:-1]), seen)   # never invisible once it arrived
        self.assertEqual(e.admitting, 0)
        self.assertEqual(e.slot_claims, set())

    def test_an_owned_busy_slot_is_never_taken(self):
        """The leak check frees only slots with no owner: a slot a request holds stays busy however long it waits."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.0
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
            e.slot_busy[1] = True
            e.slot_releasing.add(1)
            self.assertEqual(e._reap_orphan_slots(), [])
            self.assertEqual(e.slot_busy, [True, True])
            e.slot_claims.discard(0)
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertEqual(e.slot_busy, [False, True])

    def test_a_parallel_request_does_not_wait_for_the_fifo_on_a_loaded_engine(self):
        """#1603: parallel requests used to take the FIFO (held by an image encode, /v1/vram, load) just to find the
        engine loaded; between 'queued' and 'running' they were neither.  Loaded: no fifo.  Unloaded: they wait for it
        and are counted as queued the whole time."""
        self.start(2)
        self.assertEqual(self.chat("warm")["choices"][0]["message"]["content"], "ok, done.")
        self.svc.fifo.acquire()
        try:
            r = self.chat("with the fifo held")             # would block here before
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        finally:
            self.svc.fifo.release()
        import serve.server as server
        script, real = Path(self.tmp.name) / "fake_strata.py", server.subprocess.Popen
        patch = mock.patch.object(server.subprocess, "Popen",      # the reload starts the fake again
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.unload()
        self.svc.fifo.acquire()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.chat("reloads", max_tokens=16)))
        t.start()
        try:
            deadline = time.time() + 10
            live = {}
            while time.time() < deadline:
                live = self.get("/metrics")["live"]
                if live.get("queued"):
                    break
                time.sleep(0.05)
            self.assertEqual(live.get("queued"), 1, live)    # waiting for the fifo: queued, not nothing
            self.assertEqual(live.get("running"), 0, live)
        finally:
            self.svc.fifo.release()
        t.join(60)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["live"]["queued"], 0)




if __name__ == "__main__":
    unittest.main()
