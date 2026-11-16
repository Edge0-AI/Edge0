"""Tests for tools/calibrate.py and setup's use of it, without a GPU: a stand-in engine whose decode speed is a
function of the settings it runs with.

    python -m unittest tools.test_calibrate
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate as CAL  # noqa: E402


class FakeEngine:
    """Speed = f(pcie_frac, spec_min_p, workers): the GEN line's tune keys arrive as `strata_tune`.
    `dies`: setting values whose engine exits during its first generation (the #267 class, mid-measurement)."""

    def __init__(self, args, speed, info_workers=6, starts=None, adapt=None, read=None, reads=None, dies=()):
        self.args = list(args)
        self.read, self.reads = read, reads      # prompt read speed = read(prefill_pipe); the values it read with
        self.adapt_bonus = (adapt or {}).get(CAL.arg_value(args, "--adapt-swaps"), 1.0)
        w = CAL.arg_value(args, "--pool-workers")
        self.workers = int(w) if w else info_workers
        self.info = {"pool_workers": info_workers, "pcie_frac": 0.55, "spec_min_p": float(CAL.arg_value(args, "--spec-min-p") or 0)}
        self.speed = speed
        self.last = {}
        self.proc = None
        self.dies = CAL.arg_value(args, "--adapt-swaps") in dies or CAL.arg_value(args, "--pool-workers") in dies
        self.dead = False
        if starts is not None:
            starts.append(list(args))

    def generate(self, ids, max_new, sampling, cancel):
        if self.dies and not self.dead:
            self.dead = True
            raise RuntimeError("the engine exited: ERR verify: the adaptive tier's copies for layer 47 never landed"
                               " (the copy stream itself is not drained); restart the engine")
        tune = sampling.get("strata_tune") or {}
        rate = self.speed(tune.get("pcie_frac", 0.55), tune.get("spec_min_p", self.info["spec_min_p"]), self.workers)
        rate *= self.adapt_bonus
        for _ in range(max_new):
            yield 1
        self.last = {"generated": max_new, "decode_ms": max_new / rate * 1000.0}
        if "prefill_pipe" in tune and self.read is not None:
            n = len(ids)
            self.last.update(prompt_tokens=n, prompt_read=n, prompt_ms=n / self.read(tune["prefill_pipe"]) * 1000.0)
            if self.reads is not None:
                self.reads.append(tune["prefill_pipe"])


BASE = ["--pack", "p", "--spec", "4", "--spec-min-p", "0.5", "--max-context", "8192"]


class Calibrate(unittest.TestCase):
    def run_with(self, speed, workers=6, base=BASE, adapt=None, read=None, reads=None, dies=()):
        starts = []
        res = CAL.measure(base, [[1, 2, 3]] * 3,
                          lambda a: FakeEngine(a, speed, workers, starts, adapt, read, reads, dies),
                          say=lambda *_: None, long_prompt=lambda i, n: [1] * n)
        return res, starts

    def test_prefill_pipe_on_a_layer_split(self):
        # two stages that read fastest with b/a 1536: kept, and only one engine start measures it
        split = BASE + ["--layer-split", "auto"]
        reads = []
        res, _ = self.run_with(lambda f, p, w: 50.0, base=split, reads=reads,
                               read=lambda v: {1536: 600.0, 768: 560.0, 1: 520.0}.get(v, 500.0))
        self.assertEqual(res["settings"].get("--prefill-pipe"), "1536")
        self.assertEqual(sorted(set(reads)), sorted(set(CAL.PIPE_VALUES)))
        self.assertIn("pipe_sweep", res["report"])
        self.assertEqual(CAL.arg_value(CAL.apply(split, res["settings"]), "--prefill-pipe"), "1536")

    def test_prefill_pipe_noise_stays_off(self):
        # 2% faster evened out: below MIN_GAIN, so the engine default (off) stays
        res, _ = self.run_with(lambda f, p, w: 50.0, base=BASE + ["--layer-split", "auto"],
                               read=lambda v: 510.0 if v == 1 else 500.0)
        self.assertNotIn("--prefill-pipe", res["settings"])

    def test_prefill_pipe_must_win_every_round(self):
        # 1536 looks 12% faster in the sweep and the median, but one paired round reads slower: noise, it stays off
        calls = {}

        def read(v):
            calls[v] = calls.get(v, 0) + 1
            if v != 1536:
                return 500.0
            return [560.0, 560.0, 560.0, 480.0, 560.0, 560.0][min(calls[v], 6) - 1]
        res, _ = self.run_with(lambda f, p, w: 50.0, base=BASE + ["--layer-split", "auto"], read=read)
        self.assertNotIn("--prefill-pipe", res["settings"])
        self.assertEqual(len(res["report"]["pipe_confirm"]["1536"]), 3)

    def test_prefill_pipe_reads_the_same_text(self):
        # every read: the same stretch of text after a header of its own (only the chunks differ between values)
        tok = type("T", (), {"encode": lambda s, t, **k: [ord(c) for c in t]})()
        make = CAL.doc_prompt(tok)
        a, b = make(1, 300), make(2, 300)
        self.assertNotEqual(a[:40], b[:40])
        self.assertEqual(a[-320:], b[-320:])

    def test_prefill_pipe_only_on_a_layer_split(self):
        reads = []
        res, _ = self.run_with(lambda f, p, w: 50.0, read=lambda v: 600.0 if v == 1536 else 500.0, reads=reads)
        self.assertEqual(reads, [])
        self.assertNotIn("--prefill-pipe", res["settings"])
        # an old calibration's value goes when a new one does not keep it
        self.assertIsNone(CAL.arg_value(CAL.apply(BASE + ["--prefill-pipe", "800"], {}), "--prefill-pipe"))

    def test_prefill_pipe_follows_the_engines_stage_count(self):
        # --layer-split that left one prompt stage (the engine says prefill_stages=1): nothing to measure
        reads = []

        class OneStage(FakeEngine):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.info["prefill_stages"] = 1
        res = CAL.measure(BASE + ["--layer-split", "auto"], [[1, 2, 3]] * 3,
                          lambda a: OneStage(a, lambda f, p, w: 50.0, 6, None, None,
                                             lambda v: 600.0 if v == 1536 else 500.0, reads),
                          say=lambda *_: None, long_prompt=lambda i, n: [1] * n)
        self.assertEqual(reads, [])
        self.assertNotIn("--prefill-pipe", res["settings"])

    def test_defaults_kept_when_flat(self):
        res, _ = self.run_with(lambda f, p, w: 50.0)
        self.assertEqual(res["settings"], {})

    def test_measure_reports_bounded_stages_and_stops_between_samples(self):
        events = []
        stopped = threading.Event()

        def progress(stage, completed, total):
            events.append((stage, completed, total))
            if stage == "pcie_share" and completed == 1:
                stopped.set()

        with self.assertRaises(CAL.CalibrationCancelled):
            CAL.measure(BASE, [[1, 2, 3]] * 3,
                        lambda args: FakeEngine(args, lambda *_: 50.0),
                        say=lambda *_: None, progress=progress, cancel_event=stopped)
        self.assertIn(("loading", 1, 1), events)
        self.assertIn(("pcie_share", 1, len(CAL.PCIE_FRACS) * CAL.SWEEP_ROUNDS), events)
        self.assertNotIn("draft_floor", [stage for stage, _, _ in events])

    def test_active_generation_receives_shared_cancel_event(self):
        stopped = threading.Event()
        seen = []

        class Engine:
            def generate(self, ids, max_new, sampling, cancel):
                seen.append(cancel)
                cancel.set()
                yield None

        session = CAL.Session(Engine(), [[1, 2, 3]], stopped)
        with self.assertRaises(CAL.CalibrationCancelled):
            session.rate()
        self.assertEqual(seen, [stopped])

    def test_successful_sweep_covers_every_progress_stage(self):
        events = []
        CAL.measure(BASE, [[1, 2, 3]] * 3,
                    lambda args: FakeEngine(args, lambda *_: 50.0),
                    say=lambda *_: None, progress=lambda *item: events.append(item))
        stages = list(dict.fromkeys(stage for stage, _, _ in events))
        self.assertEqual(stages, ["loading", "pcie_share", "draft_floor", "confirmation",
                                  "cpu_workers", "expert_tier"])
        for stage in stages:
            readings = [(done, total) for current, done, total in events if current == stage]
            self.assertEqual(readings[0][0], 0)
            self.assertEqual(readings[-1][0], readings[-1][1])
            self.assertTrue(all(before[0] <= after[0] and before[1] == after[1]
                                for before, after in zip(readings, readings[1:])))

    def test_small_gain_is_noise(self):
        # 2% better at pcie 0.35: below MIN_GAIN, so the default stays
        res, _ = self.run_with(lambda f, p, w: 51.0 if abs(f - 0.35) < 1e-6 else 50.0)
        self.assertEqual(res["settings"], {})

    def test_finds_pcie_and_min_p(self):
        def speed(f, p, w):
            return 50.0 + (10.0 if abs(f - 0.2) < 1e-6 else 0.0) + (5.0 if abs(p - 0.7) < 1e-6 else 0.0)
        res, _ = self.run_with(speed)
        self.assertEqual(res["settings"].get("--pcie-frac"), "0.20")
        self.assertEqual(res["settings"].get("--spec-min-p"), "0.70")
        self.assertNotIn("--pool-workers", res["settings"])

    def test_fewer_workers(self):
        # a hybrid CPU: half the workers is 20% faster
        res, starts = self.run_with(lambda f, p, w: 60.0 if w == 3 else 50.0, workers=6)
        self.assertEqual(res["settings"].get("--pool-workers"), "3")
        # one start per worker count and per expert-tier candidate, plus the sweep
        self.assertEqual(len(starts), 1 + len(CAL.worker_candidates(6)) + len(CAL.ADAPT_CANDIDATES))
        self.assertEqual(CAL.arg_value(starts[0], "--spec-min-p"), "0.5")   # measured against the product default

    def test_a_failed_restart_keeps_the_measurements_1337(self):
        # the top PCIe share wins by far; every later restart (the worker counts, the expert tier) fails to start
        n = {"starts": 0}

        def start(args):
            n["starts"] += 1
            if n["starts"] > 1:
                raise RuntimeError("the engine exited before it was ready: cudaMalloc failed (213 MiB free)")
            return FakeEngine(args, lambda f, p, w: 80.0 if f == 0.75 else 50.0, 6)
        said = []
        res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=said.append)
        self.assertEqual(res["settings"].get("--pcie-frac"), "0.75")        # steps 1-3 survived
        self.assertGreater(len(res["report"]["failed_starts"]), 1)
        self.assertTrue(any("did not start" in x and "cudaMalloc" in x for x in said))
        self.assertEqual(res["report"]["workers"], {})
        self.assertEqual(res["report"]["adapt"], {})

    def test_one_failing_worker_count_only_drops_that_candidate_1337(self):
        def start(args):
            if CAL.arg_value(args, "--pool-workers") == "3":
                raise RuntimeError("out of memory")
            return FakeEngine(args, lambda f, p, w: 60.0 if w == 2 else 50.0, 6)
        res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=lambda *_: None)
        self.assertEqual(res["settings"].get("--pool-workers"), "2")
        self.assertEqual(list(res["report"]["failed_starts"]), ["3 workers"])

    def test_adaptive_tier(self):
        # a slow-RAM PC: swapping 160 experts per round is 10% faster, 80 is 2% (noise)
        res, starts = self.run_with(lambda f, p, w: 50.0, adapt={"160": 1.10, "80": 1.02})
        self.assertEqual([res["settings"].get(f) for f in CAL.ADAPT_FLAGS], ["1", "160", "0.97"])
        flat, _ = self.run_with(lambda f, p, w: 50.0, adapt={"160": 1.02})
        self.assertFalse(any(f in flat["settings"] for f in CAL.ADAPT_FLAGS))
        # the candidates are measured with the worker count chosen before
        res, starts = self.run_with(lambda f, p, w: 60.0 if w == 3 else 50.0, workers=6, adapt={"160": 1.10})
        self.assertEqual(res["settings"].get("--pool-workers"), "3")
        self.assertTrue(all(CAL.arg_value(a, "--pool-workers") == "3" for a in starts[-len(CAL.ADAPT_CANDIDATES):]))
        a = CAL.apply(BASE, res["settings"])
        self.assertEqual(CAL.arg_value(a, "--adapt-swaps"), "160")
        self.assertIsNone(CAL.arg_value(CAL.apply(a, {}), "--adapt-swaps"))   # an old calibration's tier goes

    def test_engine_death_during_tier_measurement_drops_only_that_candidate(self):
        # the 160-swap candidate's engine dies during its own measurement (the adaptive gate's #267 livelock):
        # that candidate loses, 80 (10% faster) is still chosen, and the run returns with the earlier results
        res, _ = self.run_with(lambda f, p, w: 50.0, adapt={"80": 1.10}, dies=("160",))
        self.assertEqual(res["settings"].get("--adapt-swaps"), "80")
        self.assertEqual(list(res["report"]["adapt"]), ["default", "1/80/0.97"])
        self.assertIn("the expert tier 1/160/0.97", res["report"]["failed_starts"])

    def test_engine_death_during_a_worker_measurement_keeps_the_run(self):
        # 4 workers' engine dies mid-measurement: 3 and 2 are still measured, the default count stays, the tier runs
        res, _ = self.run_with(lambda f, p, w: 50.0, workers=6, dies=("4",))
        self.assertNotIn("--pool-workers", res["settings"])
        self.assertEqual(sorted(int(k.split()[0]) for k in res["report"]["workers"]), [2, 3, 6])
        self.assertIn("4 workers", res["report"]["failed_starts"])
        self.assertEqual(list(res["report"]["adapt"]), ["default", "1/80/0.97", "1/160/0.97"])

    def test_old_calibration_is_the_baseline_reset(self):
        # a config tuned earlier: the measurement starts from the product defaults, not from those values
        base = CAL.apply(BASE, {"--pcie-frac": "0.20", "--pool-workers": "3", "--spec-min-p": "0.70"})
        _, starts = self.run_with(lambda f, p, w: 50.0, base=base)
        self.assertIsNone(CAL.arg_value(starts[0], "--pcie-frac"))
        self.assertIsNone(CAL.arg_value(starts[0], "--pool-workers"))
        self.assertEqual(CAL.arg_value(starts[0], "--spec-min-p"), "0.5")

    def test_sh_stream_is_measured_on_hip_only(self):
        # #1272 / #816: the shared-expert fork is measured off (the HIP default) against on, each with a restart and
        # the engine variable set explicitly; it is kept only when it wins by more than MIN_GAIN
        def run(bonus, hip=True):
            starts, envs = [], []

            def start(args, env=None):
                envs.append(dict(env or {}))
                fork = (env or {}).get(CAL.SH_STREAM) == "1"
                return FakeEngine(args, lambda f, p, w: 50.0 * (bonus if fork else 1.0), 6, starts)
            res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=lambda *_: None, hip=hip)
            return res, starts, envs
        res, starts, envs = run(1.08)
        self.assertEqual(res["env"], {CAL.SH_STREAM: "1"})
        self.assertEqual(envs[-2:], [{CAL.SH_STREAM: "0"}, {CAL.SH_STREAM: "1"}])
        self.assertEqual(starts[-1], starts[-2])                      # the same args for both: only the variable differs
        self.assertGreater(res["report"]["tok_s"], 52.0)              # the reported speed is the fork's
        flat, _, _ = run(1.02)                                        # within noise: the default stays
        self.assertEqual(flat["env"], {})
        self.assertEqual(set(flat["report"]["sh_stream"]), {"0", "1"})
        slower, _, _ = run(0.80)
        self.assertEqual(slower["env"], {})
        cuda, _, envs = run(1.50, hip=False)                          # not AMD: no step, no engine variable
        self.assertEqual(cuda["env"], {})
        self.assertNotIn("sh_stream", cuda["report"])
        self.assertTrue(all(e == {} for e in envs))

    def test_a_failed_fork_start_keeps_the_default(self):
        def start(args, env=None):
            if (env or {}).get(CAL.SH_STREAM) == "1":
                raise RuntimeError("out of memory")
            return FakeEngine(args, lambda f, p, w: 50.0, 6)
        res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=lambda *_: None, hip=True)
        self.assertEqual(res["env"], {})
        self.assertIn(f"{CAL.SH_STREAM}=1", res["report"]["failed_starts"])

    def test_engine_death_during_a_fork_measurement_keeps_the_run(self):
        # the forked engine exits mid-measurement (#267's class): that candidate loses, the run returns
        def start(args, env=None):
            e = FakeEngine(args, lambda f, p, w: 50.0, 6)
            if (env or {}).get(CAL.SH_STREAM) == "1":
                e.dies = True
            return e
        res = CAL.measure(BASE, [[1, 2, 3]] * 3, start, say=lambda *_: None, hip=True)
        self.assertEqual(res["env"], {})
        self.assertEqual(list(res["report"]["sh_stream"]), ["0"])
        self.assertIn(f"{CAL.SH_STREAM}=1", res["report"]["failed_starts"])

    def test_apply_env(self):
        old = {"STRATA_ROCBLAS_TUNING": "tools/hip", CAL.SH_STREAM: "1"}
        self.assertEqual(CAL.apply_env(old, {}), {"STRATA_ROCBLAS_TUNING": "tools/hip"})   # an older choice goes
        self.assertEqual(CAL.apply_env({"X": "1"}, {CAL.SH_STREAM: "1"}), {"X": "1", CAL.SH_STREAM: "1"})
        self.assertEqual(CAL.apply_env(None, None), {})

    def test_apply(self):
        a = CAL.apply(BASE, {"--pcie-frac": "0.35", "--pool-workers": "4"})
        self.assertEqual(CAL.arg_value(a, "--pcie-frac"), "0.35")
        self.assertEqual(CAL.arg_value(a, "--pool-workers"), "4")
        self.assertEqual(CAL.arg_value(a, "--spec-min-p"), "0.5")
        b = CAL.apply(a, {})                                # back to the defaults
        self.assertIsNone(CAL.arg_value(b, "--pcie-frac"))
        self.assertIsNone(CAL.arg_value(b, "--pool-workers"))
        self.assertEqual(b.count("--spec-min-p"), 1)

    def test_engine_args_are_the_servers(self):
        # #447: a "gpu" list is a layer split only with two or more cards; "0,2" is one too; split_skip_if_fits counts
        helper = {"args": BASE + ["--expert-cache-device1", "1800"], "gpu": [0], "env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
        self.assertEqual(CAL.engine_args(helper), helper["args"])                 # one stage + a helper card: no split
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": 1}), BASE)
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": [0, 2]}), BASE + ["--layer-split", "auto"])
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": "0,2", "layer_split": "18"}),
                         BASE + ["--layer-split", "18"])
        self.assertEqual(CAL.engine_args({"args": BASE, "gpu": [0, 1], "split_skip_if_fits": True}),
                         BASE + ["--layer-split", "auto", "--split-skip-if-fits"])
        own = BASE + ["--layer-split", ""]                                       # the config's own value wins
        self.assertEqual(CAL.engine_args({"args": own, "gpu": [0, 1]}), own)

    def test_run_measures_with_the_servers_args(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d)
            (tok / "vocab.json").write_text(json.dumps({"a": 0, "b": 1}))
            (tok / "merges.txt").write_text("")
            (tok / "token_type.json").write_text(json.dumps([1, 1]))
            seen = []
            saved = CAL.measure
            CAL.measure = lambda args, ids_list, start_engine, say=print, extra=(), long_prompt=None, hip=False: (
                seen.append((args, hip)) or {})
            fake = type("ST", (), {"Tokenizer": lambda *a: type("T", (), {"encode": lambda s, t, **k: [0]})()})
            try:
                with mock.patch.dict(sys.modules, {"strata_tokenizer": fake}):
                    CAL.run({"tokenizer": d, "args": list(BASE), "gpu": [0]}, start_engine=lambda a: None)
                    CAL.run({"tokenizer": d, "args": list(BASE), "gpu": [0], "backend": "hip"}, start_engine=lambda a: None)
            finally:
                CAL.measure = saved
        self.assertEqual(seen, [(BASE, False), (BASE, True)])         # the shared-expert step runs on AMD only

    def test_engine_error(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata-x.log"
            log.write_text("strata generate: an old error\n", encoding="utf-8")
            since = log.stat().st_size
            with open(log, "a", encoding="utf-8") as f:
                f.write("loading ...\nstrata generate: --expert-cache-remote with a layer split needs a GPU that runs "
                        "no stage (2 visible, 2 used by the split)\n\n")
            self.assertEqual(CAL.engine_error(str(log), since), "strata generate: --expert-cache-remote with a layer "
                             "split needs a GPU that runs no stage (2 visible, 2 used by the split)")
            with open(log, "a", encoding="utf-8") as f:
                f.write("Segmentation fault\n")
            self.assertIn("--expert-cache-remote", CAL.engine_error(str(log), since))   # the engine's own line first
            self.assertEqual(CAL.engine_error(str(log), log.stat().st_size), None)   # nothing new since
            self.assertIsNone(CAL.engine_error(None))
            self.assertIsNone(CAL.engine_error(str(Path(d) / "missing.log")))

    def test_worker_candidates(self):
        self.assertEqual(CAL.worker_candidates(6), [6, 4, 3, 2])
        self.assertEqual(CAL.worker_candidates(23), [23, 15, 12, 6])        # a quarter: 4 beat 15 on 8P + 16E
        self.assertEqual(CAL.worker_candidates(23, (7,)), [23, 15, 12, 6, 7])   # P-cores - 1
        self.assertEqual(CAL.worker_candidates(35, (17,)), [35, 23, 18, 9, 17])  # 2 sockets: one socket's cores - 1
        self.assertEqual(CAL.worker_candidates(6, (6, 1, 9)), [6, 4, 3, 2])      # none above the default, none below 2
        self.assertEqual(CAL.worker_candidates(3), [3, 2])
        self.assertEqual(CAL.worker_candidates(1), [1])

    def test_pick(self):
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [53, 52, 60]}, "a"), "b")
        self.assertEqual(CAL.pick({"a": [50, 51, 49], "b": [51, 51.5, 51]}, "a"), "a")
        self.assertEqual(CAL.pick({}, "a"), "a")


class SetupIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = self.tmp.name
        os.environ["XDG_CONFIG_HOME"] = self.tmp.name
        import setup as S
        self.S = S
        self.saved = (S.gpu_info, S.cpu_info, S.ram_gb, CAL.run)
        S.gpu_info = lambda pick=None: {"name": "RTX Test", "vram_gb": 12.0, "arch": "120", "count": 1, "index": 0}
        S.cpu_info = lambda: ("Test CPU", True, True)
        S.ram_gb = lambda: 64.0

    def tearDown(self):
        self.S.gpu_info, self.S.cpu_info, self.S.ram_gb, CAL.run = self.saved
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_calibrate_config_writes_and_remembers(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg = {"exe": "x", "args": list(BASE), "model_name": "qwen3.8-flash-next-q2_0"}
        cfg_path.write_text(json.dumps(cfg))
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {"--pcie-frac": "0.20"}, "report": {"tok_s": 61.2}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        written = json.loads(cfg_path.read_text())
        self.assertEqual(CAL.arg_value(written["args"], "--pcie-frac"), "0.20")
        saved = self.S.saved_calibration(written)
        self.assertEqual(saved["settings"], {"--pcie-frac": "0.20"})
        # another model on the same PC has no calibration yet
        self.assertIsNone(self.S.saved_calibration({**written, "model_name": "swift-iq2_xs"}))
        # another context size is another key (its KV cache changes the expert cache)
        other = dict(written, args=CAL.with_arg(written["args"], "--max-context", "131072"))
        self.assertIsNone(self.S.saved_calibration(other))

    def test_calibrate_config_writes_the_env_choice(self):
        # HIP: the measured engine variable goes into the config's "env" and the settings file; a re-run that finds it
        # no longer faster removes it again, and leaves the config's other variables alone
        cfg_path = Path(self.tmp.name) / "strata-iq3_s.json"
        cfg = {"exe": "x", "args": list(BASE), "model_name": "m", "backend": "hip", "env": {"STRATA_ROCBLAS_TUNING": "t"}}
        cfg_path.write_text(json.dumps(cfg))
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {}, "env": {CAL.SH_STREAM: "1"}, "report": {"tok_s": 70.1}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        written = json.loads(cfg_path.read_text())
        self.assertEqual(written["env"], {"STRATA_ROCBLAS_TUNING": "t", CAL.SH_STREAM: "1"})
        self.assertEqual(self.S.saved_calibration(written)["env"], {CAL.SH_STREAM: "1"})
        CAL.run = lambda c, say=print, start_engine=None: {"settings": {}, "env": {}, "report": {"tok_s": 65.0}}
        self.assertTrue(self.S.calibrate_config(cfg_path))
        self.assertEqual(json.loads(cfg_path.read_text())["env"], {"STRATA_ROCBLAS_TUNING": "t"})

    def test_failed_calibration_keeps_defaults(self):
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg_path.write_text(json.dumps({"exe": "x", "args": list(BASE), "model_name": "m"}))

        def boom(*a, **k):
            raise RuntimeError("the engine exited before it was ready")
        CAL.run = boom
        self.assertFalse(self.S.calibrate_config(cfg_path))
        self.assertEqual(json.loads(cfg_path.read_text())["args"], BASE)
        self.assertIsNone(self.S.saved_calibration({"args": BASE, "model_name": "m"}))

    def test_failed_calibration_says_the_engines_reason(self):
        # #447: the engine's last error line is printed, and the --calibrate run repeats the failure before it starts
        import contextlib
        import io
        log = Path(self.tmp.name) / "strata-q2_0.log"
        log.write_text("strata generate: an error of an earlier start\n", encoding="utf-8")
        cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        cfg_path.write_text(json.dumps({"exe": "x", "args": list(BASE), "model_name": "m", "log": str(log)}))

        def boom(*a, **k):
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata generate: --expert-cache-remote with a layer split needs a GPU that runs no stage\n")
            raise RuntimeError(f"the engine exited before it was ready (see {log})")
        CAL.run = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(self.S.calibrate_config(cfg_path))
        self.assertIn("the engine said: strata generate: --expert-cache-remote with a layer split", out.getvalue())
        self.assertNotIn("an earlier start", out.getvalue())
        out = io.StringIO()
        with mock.patch.object(self.S, "ROOT", Path(self.tmp.name)), \
                mock.patch.object(self.S, "data_folder", lambda d, *a, **k: (Path(self.tmp.name) / "data", [])), \
                mock.patch.object(self.S, "update_installed_engine", lambda *a: None), \
                mock.patch.object(sys, "argv", ["setup.py", "--calibrate", "--no-start"]), \
                contextlib.redirect_stdout(out):
            self.assertEqual(self.S.main(), 0)
        self.assertIn("this PC is NOT tuned: the tuning failed (the reason is above); the model keeps the default "
                      "settings", out.getvalue())


if __name__ == "__main__":
    unittest.main()
