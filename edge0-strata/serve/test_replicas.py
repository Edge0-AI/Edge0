"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.replicas import (BLOCK, ReplicaConfigError, ReplicaEngine, Router, cpu_sets, gpu_groups, remove_arena,
                            replica_cfg, shared_arena_plan, sweep_stale_arenas)
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_a_number_cuts_the_gpu_list(self):
        self.assertEqual(gpu_groups({"gpu": [0, 1, 2, 3], "replicas": 2}), [[0, 1], [2, 3]])
        self.assertEqual(gpu_groups({"gpu": "0,1,2,3", "replicas": 4}), [[0], [1], [2], [3]])
        self.assertEqual(gpu_groups({"gpu": [0, 1], "replicas": 1}), [])
        self.assertEqual(gpu_groups({"gpu": [0, 1]}), [])

    def test_named_groups(self):
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}, {"gpus": [2]}]}), [[0, 1], [2]])
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}]}), [])

    def test_mistakes_are_said(self):
        for cfg in ({"gpu": [0, 1, 2], "replicas": 2}, {"replicas": 2}, {"replicas": "2"},
                    {"replicas": [{"gpus": [0, 1]}, {"gpus": [1, 2]}]}, {"replicas": [{"gpus": []}, {"gpus": [1]}]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                gpu_groups(cfg)

    def test_replica_cfg(self):
        cfg = {"gpu": [0, 1, 2, 3], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "env": {"A": "1"},
               "expert_profile_save": "/p/prof.bin",
               "replicas": [{"gpus": [0, 1], "env": {"B": 2}}, {"gpus": [2, 3], "args": ["--y"]}]}
        a, b = replica_cfg(cfg, 0, [0, 1]), replica_cfg(cfg, 1, [2, 3])
        self.assertEqual((a["gpu"], b["gpu"]), ([0, 1], [2, 3]))
        self.assertEqual((a["log"], b["log"]), ("/l/strata.r0.log", "/l/strata.r1.log"))
        self.assertEqual(a["env"], {"A": "1", "B": "2"})
        self.assertEqual(b["args"], ["--x", "--y"])
        self.assertNotIn("replicas", a)
        self.assertNotIn("hip_ordinal", a)
        self.assertEqual(a["expert_profile_save"], "/p/prof.r0.bin")
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched
        self.assertEqual(replica_cfg({"gpu": [0, 1], "args": []}, 0, [3])["gpu"], 3)

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield 7


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)

    def test_selected_replica_is_reported_for_solo_requests_too(self):
        engine = ReplicaEngine([FakeEngine(), FakeEngine()])
        engine.inflight[0] = 1
        cancel = threading.Event()
        selected = []
        cancel._strata_replica_selected = selected.append
        self.assertEqual(list(engine.generate([1, 2], 1, {}, cancel)), [7])
        self.assertEqual(selected, [1])


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def post(self, messages, max_tokens=64):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64):
        return self.post([{"role": "user", "content": text}], max_tokens)

    def served(self):
        return [sum(1 for l in p.read_text().splitlines() if l.split()[0] in ("GEN", "BGEN")) if p.exists() else 0
                for p in self.logs]

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_terminal_telemetry_correlates_second_replica_without_global_sample_claim(self):
        self.start()
        self.svc.replica_gpu_groups = [[0, 1], [2, 3]]
        second = self.engine.engines[1]
        request_id = "abcdef012345"
        second.slot_live[1] = {"request_id": request_id, "state": "reading", "prompt_tokens": 100,
                               "prompt_read": 12, "generated": 0, "started": time.time(), "first_token": None}
        first = self.engine.engines[0]
        first.terminal_batch_sample = {"windows": 2, "processed_rows": 8, "active_rows": 4,
                                       "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        second.terminal_batch_sample = {"windows": 2, "processed_rows": 6, "active_rows": 4,
                                        "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        first.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [800_000, 600_000],
                                       "observed_monotonic": time.monotonic()}
        second.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [700_000, 500_000],
                                        "observed_monotonic": time.monotonic()}
        self.svc.backend = "cuda"
        for child, cards in ((first, [0, 1]), (second, [2, 3])):
            child.spawn = (*child.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                              "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))},
                           *child.spawn[5:])
            child.terminal_stages = [
                {"stage": 0, "layer_start": 0, "layer_end": 19, "engine_ordinal": 0},
                {"stage": 1, "layer_start": 20, "layer_end": 39, "engine_ordinal": 1}]
        with self.svc.status_lock:
            self.svc.terminal_requests[request_id] = {"id": request_id, "state": "waiting",
                                                       "queued_at": time.time(), "admitted_at": time.time(),
                                                       "prompt_tokens": 100}
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            snapshot = json.load(response)
        slot = snapshot["work"]["slots"][5]
        self.assertEqual((slot["slot"], slot["replica"], slot["group"], slot["request_id"]),
                         (5, 1, 1, request_id))
        request = snapshot["work"]["requests"][0]
        self.assertEqual((request["slot"], request["replica"], request["group"], request["state"]),
                         (5, 1, 1, "reading"))
        self.assertEqual(snapshot["work"]["groups"], 2)
        self.assertIsNone(snapshot["work"]["effective_rows_window"])
        self.assertEqual([(row["replica"], row["batch_sample_rows"], row["batch_sample_active_rows"])
                          for row in snapshot["work"]["replica_batches"]], [(0, 8, 4), (1, 6, 4)])
        self.assertEqual([(row["replica"], row["stage_in_flight_us"])
                          for row in snapshot["work"]["replica_stage_samples"]],
                         [(0, [800_000, 600_000]), (1, [700_000, 500_000])])
        self.assertNotIn("stage_in_flight_us", snapshot["work"])
        self.assertEqual(snapshot["topology"]["stages"], [])
        self.assertEqual(snapshot["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)
        self.assertEqual(snapshot["topology"]["replica_devices"],
                         [{"replica": 0, "devices": [0, 1]}, {"replica": 1, "devices": [2, 3]}])
        second.spawn = (*second.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "CUDA_VISIBLE_DEVICES": "9,3"}, *second.spawn[5:])
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            mismatch = json.load(response)
        self.assertIsNone(mismatch["topology"]["replica_stages"][1]["stages"][0]["device_index"])
        self.assertEqual(mismatch["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


class SharedArena(unittest.TestCase):
    """The opt-in shared expert arena of the replicas (shared_arena_plan): when it is used, when it is not, what a stale
    file means."""
    CFG = {"replica_shared_arena": True, "args": ["--pack", "/p"], "gpu": [0, 1]}

    def plan(self, cfg=None, n=2, tmp=None, **kw):
        return shared_arena_plan(cfg or self.CFG, n, shm=tmp, platform=kw.pop("platform", "linux"),
                                 environ=kw.pop("environ", {}), free_bytes=kw.pop("free_bytes", 100 << 30),
                                 use_memfd=kw.pop("use_memfd", False), **kw)

    def test_off_by_default_and_for_one_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.plan({"args": []}, tmp=d), (None, ""))
            self.assertEqual(self.plan(n=1, tmp=d), (None, ""))
            self.assertEqual(self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "0"}), (None, ""))

    def test_on_by_config_or_environment(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d)
            self.assertTrue(path and path.startswith(d) and path.endswith(".bin"), path)
            self.assertIn("once", note)
            path2, _ = self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "1"})
            self.assertTrue(path2)
            self.assertEqual(path, path2)      # (one server: one file)

    def test_windows_gets_a_named_section(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, platform="win32")
            self.assertTrue(path and path.startswith("win:Local") and "strata-replicas-" in path, path)
            self.assertIn("Windows section", note)
            remove_arena(path)                 # (nothing to remove: harmless)

    def test_only_where_there_is_an_arena_to_share(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self.plan(tmp=d, platform="darwin")[0])
            for flag in ("--mmap-experts", "--resident-experts", "--resident-budget-gib", "--shared-expert-arena"):
                self.assertIsNone(self.plan({**self.CFG, "args": ["--pack", "/p", flag]}, tmp=d)[0], flag)
            self.assertIsNone(self.plan({**self.CFG, "replicas": [{"gpus": [0], "args": ["--mmap-experts"]}, {"gpus": [1]}]}, tmp=d)[0])
            self.assertIsNone(self.plan({**self.CFG, "arch": "deepseek41"}, tmp=d)[0])

    def test_no_room_or_no_shm_says_so_and_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, free_bytes=2 << 30)
            self.assertIsNone(path)
            self.assertIn("2.0 GiB free", note)
            path, note = self.plan(tmp=d + "/nope")
            self.assertIsNone(path)
            self.assertIn("not usable", note)

    def test_a_stale_file_goes_a_live_one_stays(self):
        with tempfile.TemporaryDirectory() as d:
            uid = __import__("os").getuid() if hasattr(__import__("os"), "getuid") else 0
            dead = Path(d) / f"strata-replicas-{uid}-2999999.bin"
            live = Path(d) / f"strata-replicas-{uid}-{__import__('os').getpid()}.bin"
            other = Path(d) / f"strata-replicas-{uid + 1}-2999999.bin"
            junk = Path(d) / "strata-replicas-x.bin"
            for f in (dead, live, other, junk):
                f.write_bytes(b"x")
            with mock.patch("serve.replicas._pid_alive", side_effect=lambda pid: pid != 2999999):
                gone = sweep_stale_arenas(d, uid)
            self.assertEqual(gone, [dead.name])
            self.assertFalse(dead.exists())
            self.assertTrue(live.exists() and other.exists() and junk.exists())

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_is_not_limited_by_dev_shm_and_is_closed_at_the_end(self):
        import os
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=100 << 30, use_memfd=True)
        self.assertTrue(path and path.startswith(f"/proc/{os.getpid()}/fd/"), path)
        fd = int(path.rsplit("/", 1)[1])
        os.ftruncate(fd, 1 << 20)
        self.assertEqual(os.fstat(fd).st_size, 1 << 20)
        with open(path, "r+b") as other:          # (how an engine opens it)
            other.seek(100)
            other.write(b"x")
        remove_arena(path)
        with self.assertRaises(OSError):
            os.fstat(fd)
        remove_arena(path)                         # (twice: harmless)

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_arena_warns_when_the_engines_will_not_all_fit(self):
        cfg = {**self.CFG, "model_name": "qwen3.8-flash-next-iq3_s"}
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=57 << 30, use_memfd=True)
        try:
            self.assertTrue(path)
            self.assertIn("WARNING", note)
            self.assertIn("may not fit", note)
        finally:
            remove_arena(path)
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=120 << 30, use_memfd=True)
        try:
            self.assertNotIn("WARNING", note)
        finally:
            remove_arena(path)

    def test_a_memfd_needs_free_ram(self):
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=2 << 30, use_memfd=True)
        self.assertIsNone(path)
        self.assertIn("2.0 GiB of RAM", note)

    def test_removing_twice_or_nothing_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")
            remove_arena(str(f))
            remove_arena(str(f))
            remove_arena(None)
            self.assertFalse(f.exists())

    def test_the_replica_engine_unlinks_the_file_at_close(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")

            class E:
                def close(self):
                    pass
            r = ReplicaEngine([E(), E()])
            r.__dict__["arena_file"] = str(f)
            r.close()
            self.assertFalse(f.exists())


if __name__ == "__main__":
    unittest.main()
