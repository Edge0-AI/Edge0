"""The auto-tuner's decisions, tested with a stub engine whose speed is computed from the flags it is given."""

from __future__ import annotations

import threading
import unittest

from strata_control import autotune, tune_knobs
from strata_control.tune_session import StartFailed

from tests.tuning.stub_session import FakeClock, StubSession, flag

BASE = ["--pack", "P", "--native", "m.gguf", "--expert-cache", "auto", "--prefill", "auto", "--spec", "4",
        "--spec-min-p", "0.5", "--mtp", "D", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768",
        "--vram-reserve-mib", "1536"]


def make_ctx(**kw):
    ctx = tune_knobs.Context(backend="cuda", family="flash-next", gpu_count=1, vram_mib=12000, ram_gib=64, model_gib=60.0,
                             context_tokens=131072, base_args=list(BASE), logical_threads=12, **kw)
    ctx.facts = {"workers": 6, "slots": 3000}
    return ctx


def speed_with(gains: dict[str, float], base=60.0):
    """tok/s as a function of the flags: each (flag, value) in `gains` multiplies the speed."""
    def model(args, env, tune=None):
        v = base
        for (f, val), g in gains.items():
            if f.startswith("STRATA_"):
                if env.get(f) == val:
                    v *= g
            elif flag(args, f) == val:
                v *= g
        if tune:
            for k, g in (gains.get(("tune", None)) or {}).items() if False else []:
                pass
        return v
    return model


def make_ctx_small():
    c = make_ctx()
    c.context_tokens = 16384
    c.base_args = ["--max-context", "16384"]
    return c


def run_tuner(model, *, opts=None, ctx=None, clock=None, session_kw=None, cancel=None, progress=None, **kw):
    clock = clock or FakeClock()
    session = StubSession(model=model, clock=clock, **(session_kw or {}))
    t = autotune.Tuner(session, BASE, {}, ctx or make_ctx(), opts or autotune.Options(), clock=clock, sleep=clock.sleep,
                       system=session.system, total_ram_mib=64000, total_vram_mib=12000, quiet=lambda: None,
                       cancel=cancel, progress=progress, **kw)
    return t, session, t.run()


class RulesTest(unittest.TestCase):
    def test_never_list(self):
        ctx = make_ctx(arch="gfx1201")
        bad = [({"--expert-cache": 0}, {}), ({"--kv-resident": 20480}, {}), ({"--pcie-frac": 0.75}, {}),
               ({"--spec": 6}, {}), ({"--spec": 2}, {}), ({"--lookup-chain": 4}, {}), ({"--adapt-every": 100000}, {}),
               ({"--spec": 5}, {}),   # outside the tuner's values: still not tried (a saved preset may keep it, #3844)
               ({"--prefill": 512}, {}), ({"--ple-io": "ram"}, {}), ({}, {"STRATA_SH_STREAM": "1"}),
               ({}, {"STRATA_RDNA2_SGEMM": "all"}), ({}, {"STRATA_HIP_WMMA": "1"}), ({"--pool-workers": 8}, {})]
        for args, env in bad:
            self.assertTrue(tune_knobs.check_setting(args, env, ctx), (args, env))
        ok = [({"--spec": 3}, {}), ({"--kv-resident": 65536}, {}), ({"--pool-workers": 4}, {}), ({"--prefill": "auto:32768"}, {}),
              ({"--adapt-every": 1, "--adapt-swaps": 80, "--adapt-decay": 0.97}, {}), ({"--pcie-frac": 0.2}, {})]
        for args, env in ok:
            self.assertEqual(tune_knobs.check_setting(args, env, ctx), [], (args, env))

    def test_kv_resident_rules_follow_the_context(self):
        long = make_ctx()                               # 128K, the config's own --kv-resident is 32768
        self.assertEqual(tune_knobs.check_setting({"--kv-resident": 65536}, {}, long), [])
        self.assertEqual(tune_knobs.check_setting({"--kv-resident": 49152}, {}, long), [])
        self.assertTrue(tune_knobs.check_setting({"--kv-resident": 20480}, {}, long))      # below the context and not measured
        short = make_ctx()
        short.context_tokens = 32768
        short.base_args = ["--max-context", "32768"]
        self.assertTrue(tune_knobs.check_setting({"--kv-resident": 20480}, {}, short))     # -29.5% at 32K
        self.assertEqual(tune_knobs.check_setting({"--kv-resident": 65536}, {}, make_ctx_small()), [])   # above the context: no streaming at all

    def test_long_context_candidates_exist_and_are_exact(self):
        ctx = make_ctx()
        ctx.vram_mib = 32624                            # the R9700: class B
        ctx.facts = {"workers": 31}
        arms, _ = tune_knobs.plan_arms(ctx, mode="quick")
        ids = [p.arm.id for p in arms]
        self.assertEqual(ids[:2], ["kv_resident_64k", "prefill_32k"])
        self.assertTrue(all(p.arm.answer == "exact" for p in arms))
        short = make_ctx()
        short.vram_mib = 32624
        short.context_tokens = 32768
        short.base_args = ["--max-context", "32768"]
        self.assertNotIn("kv_resident_64k", [p.arm.id for p in tune_knobs.plan_arms(short, mode="quick")[0]])

    def test_a_prompt_knob_is_judged_on_prompt_speed_and_may_not_cost_decode(self):
        ctx = make_ctx()
        ctx.vram_mib = 32624
        def pp(args, env):
            return 1500.0 * (1.06 if flag(args, "--prefill") == "auto:32768" else 1.0)
        # decode unchanged, long prompt +6%: wins on the prompt metric
        clock = FakeClock()
        sess = StubSession(model=lambda a, e, tune=None: 60.0, pp=pp, clock=clock, noise=0.001)
        t = autotune.Tuner(sess, BASE, {}, ctx, autotune.Options(budget_s=3600), clock=clock, sleep=clock.sleep, system=sess.system,
                           total_ram_mib=64000, total_vram_mib=32000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(r.settings["args"], {"--prefill": "auto:32768"}, r.notes)
        # the same prompt gain with a 6% decode loss does not
        sess2 = StubSession(model=lambda a, e, tune=None: 60.0 * (0.94 if flag(a, "--prefill") == "auto:32768" else 1.0), pp=pp, clock=FakeClock(), noise=0.001)
        c2 = sess2.clock
        t2 = autotune.Tuner(sess2, BASE, {}, ctx, autotune.Options(budget_s=3600), clock=c2, sleep=c2.sleep, system=sess2.system,
                            total_ram_mib=64000, total_vram_mib=32000, quiet=lambda: None)
        self.assertEqual(t2.run().settings["args"], {})

    def test_only_the_best_single_knob_is_kept(self):
        ctx = make_ctx()
        ctx.vram_mib = 32624
        model = speed_with({("--kv-resident", "65536"): 1.06})
        pp = lambda a, e: 1500.0 * (1.07 if flag(a, "--prefill") == "auto:32768" else 1.0)
        clock = FakeClock()
        sess = StubSession(model=model, pp=pp, clock=clock, noise=0.001)
        t = autotune.Tuner(sess, BASE, {}, ctx, autotune.Options(budget_s=3600), clock=clock, sleep=clock.sleep, system=sess.system,
                           total_ram_mib=64000, total_vram_mib=32000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(len(r.settings["args"]), 1)
        self.assertTrue(any("only the best one" in n for n in r.notes))

    def test_a_box_whose_default_is_not_repeatable_says_so(self):
        ctx = make_ctx()
        self.assertFalse(tune_knobs.default_unsteady(ctx))
        ctx.facts = {"workers": 6, "resident_ram": True}            # the RTX 3060 in resident-RAM mode
        self.assertTrue(tune_knobs.default_unsteady(ctx))
        sy = make_ctx()
        sy.backend = "sycl"
        self.assertTrue(tune_knobs.default_unsteady(sy))
        from strata_control import presets
        steady = {"answers": {"class": "exact", "default_repeatable": True}}
        unsteady = {"answers": {"class": "exact", "default_repeatable": False}}
        self.assertIn("identical", presets.notice(steady))
        self.assertIn("even with the standard setup", presets.notice(unsteady))
        # the class of the knob does not change: exact knobs are still auto-picked there, nonrepeatable ones stay behind their switch
        ids = [p.arm.id for p in tune_knobs.plan_arms(ctx, mode="quick")[0]]
        self.assertNotIn("adapt_async", ids)

    def test_hybrid_cpu_affinity_never(self):
        ctx = make_ctx()
        ctx.hybrid_cpu = True
        self.assertTrue(tune_knobs.check_setting({"--pool-affinity": "auto"}, {}, ctx))
        ctx.hybrid_cpu = False
        self.assertEqual(tune_knobs.check_setting({"--pool-affinity": "auto"}, {}, ctx), [])

    def test_boolean_flags_are_present_or_absent(self):
        self.assertEqual(tune_knobs.apply_to_argv(["--a", "1"], {"--kv-grow": True}), ["--a", "1", "--kv-grow"])
        self.assertEqual(tune_knobs.apply_to_argv(["--kv-grow", "--a", "1"], {"--kv-grow": None}), ["--a", "1"])
        self.assertNotIn("0", tune_knobs.apply_to_argv(["--kv-grow"], {"--kv-grow": False}))

    def test_candidates_are_single_knob_changes_of_the_users_config(self):
        ctx = make_ctx(allow_placement=True)
        arms, skipped = tune_knobs.plan_arms(ctx, mode="thorough")
        self.assertTrue(arms)
        for p in arms:
            argv = tune_knobs.apply_to_argv(BASE, p.args)
            # everything the user had that the arm does not name is untouched (Rule 1)
            for f in ("--expert-cache", "--kv", "--spec-min-p", "--vram-reserve-mib", "--max-context"):
                if f not in p.args:
                    self.assertEqual(flag(argv, f), flag(BASE, f), (p.arm.id, f))
            self.assertEqual(tune_knobs.check_setting(p.args, p.env, ctx), [])

    def test_answer_changing_arms_need_the_users_tick(self):
        ctx = make_ctx()
        ids = {p.arm.id for p in tune_knobs.plan_arms(ctx, mode="thorough")[0]}
        self.assertNotIn("spec3", ids)
        self.assertNotIn("adapt_async", ids)
        ctx2 = make_ctx(allow_placement=True)
        self.assertIn("spec3", {p.arm.id for p in tune_knobs.plan_arms(ctx2, mode="thorough")[0]})
        ctx3 = make_ctx(allow_nonrepeatable=True)
        ctx3.facts = {"workers": 6, "resident_ram": True}
        self.assertIn("adapt_async", {p.arm.id for p in tune_knobs.plan_arms(ctx3, mode="thorough")[0]})

    def test_engine_without_a_flag_drops_the_arm_with_a_reason(self):
        ctx = make_ctx()
        ctx.engine_flags = ["--spec"]
        arms, skipped = tune_knobs.plan_arms(ctx, mode="thorough")
        self.assertFalse(any(p.arm.id == "kv_resident_64k" for p in arms))
        self.assertTrue(any("does not accept" in s["reason"] for s in skipped))

    def test_qwen36_and_sycl_lists(self):
        q = make_ctx()
        q.family = "qwen36"
        q.facts = {"workers": 6, "hybrid_experts": True}
        ids = [p.arm.id for p in tune_knobs.plan_arms(q, mode="thorough")[0]]
        self.assertIn("q36_workers_explicit", ids)
        self.assertNotIn("kv_resident_64k", ids)
        s = make_ctx(allow_placement=True)
        s.backend = "sycl"
        ids = [p.arm.id for p in tune_knobs.plan_arms(s, mode="thorough")[0]]
        self.assertEqual(set(ids) - {"adapt_decay_09", "pcie_start_02"}, set())

    def test_workers_arm_never_goes_below_a_quarter(self):
        c = make_ctx()
        c.facts = {"workers": 4}
        self.assertEqual(tune_knobs.dynamic_value("workers-1", c), 3)       # one step down at most
        self.assertIsNone(tune_knobs.dynamic_value("workers-2", c))          # never two below (spec 6.2 C)
        c.facts = {"workers": 2}
        self.assertIsNone(tune_knobs.dynamic_value("workers-1", c))


class TunerTest(unittest.TestCase):
    def test_finds_a_real_gain_and_changes_one_knob(self):
        model = speed_with({("--kv-resident", "65536"): 1.08})
        t, s, r = run_tuner(model, ctx=make_ctx())
        self.assertEqual(r.outcome, "improved", r.notes)
        self.assertEqual(r.settings["args"], {"--kv-resident": 65536})
        self.assertAlmostEqual(r.gain_pct, 8.0, delta=2.0)
        self.assertEqual(r.answers["class"], "exact")
        for args, env in s.runs:                           # every run kept the user's own flags
            self.assertEqual(flag(args, "--expert-cache"), "auto")
            self.assertEqual(flag(args, "--kv"), "int8")
        self.assertLessEqual(r.seconds, 1200 + 90 * 2)

    def test_no_gain_keeps_the_default(self):
        t, s, r = run_tuner(lambda a, e, tune=None: 60.0, session_kw={"noise": 0.02})
        self.assertEqual(r.outcome, "default_confirmed")
        self.assertEqual(r.settings, {"args": {}, "env": {}, "per_request": {}})
        self.assertIn("already the fastest", r.message)

    def test_a_gain_below_the_margin_is_not_taken(self):
        model = speed_with({("--kv-resident", "65536"): 1.02})
        t, s, r = run_tuner(model, session_kw={"noise": 0.001})
        self.assertEqual(r.settings["args"], {})

    def test_losing_candidates_are_dropped_after_one_loss(self):
        model = speed_with({("--kv-resident", "65536"): 0.85})
        t, s, r = run_tuner(model, session_kw={"noise": 0.001})
        arm = [a for a in r.arms if a["id"] == "kv_resident_64k"][0]
        self.assertEqual(arm["status"], "dropped")
        self.assertEqual(sum(1 for args, _ in s.runs if flag(args, "--kv-resident") == "65536"), 1)

    def test_machine_drift_is_not_mistaken_for_a_gain(self):
        clock = FakeClock()
        counter = {"n": 0}

        def model(args, env, tune=None):                    # the machine slows 1.2% per engine start
            return 60.0 * (1.0 - 0.012 * len(session.runs)) * (1.03 if flag(args, "--kv-resident") == "65536" else 1.0)
        session = StubSession(model=model, clock=clock, noise=0.0005)
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(), clock=clock, sleep=clock.sleep, system=session.system,
                           total_ram_mib=64000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(r.settings["args"], {}, r.notes)

    def test_candidate_that_does_not_fit_is_never_started(self):
        model = speed_with({("--kv-resident", "65536"): 1.10})
        clock = FakeClock()
        session = StubSession(model=model, clock=clock, avail=9000, rss=lambda a, e: 12000.0)
        calls = {"n": 0}
        real = session.system

        def system():
            out = real()
            out["memory"]["available_mib"] = 40000 if not session.runs else 9000     # after the base run only 9 GB is free
            return out
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(wait_ram_s=30), clock=clock, sleep=clock.sleep,
                           system=system, total_ram_mib=64000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(len(session.runs), 1, "only the base was started")
        self.assertEqual(r.outcome, "default_confirmed")
        self.assertTrue(any(a["status"] == "skipped" and "memory" in a["reason"] for a in r.arms))

    def test_memory_wait_gives_up_when_nothing_is_freed(self):
        # #3490: a loaded model keeps the memory; ten minutes of waiting per candidate made a tuning last for hours
        model = speed_with({})
        clock = FakeClock()
        session = StubSession(model=model, clock=clock)

        def system():
            out = session.system()
            out["memory"]["available_mib"] = 40000 if not session.runs else 3000
            return out
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(per_request=False), clock=clock,
                           sleep=clock.sleep, system=system, total_ram_mib=64000, quiet=lambda: None)
        before = clock()
        t.run()
        self.assertLess(clock() - before, 600.0, "waited for the whole ten minutes per candidate")

    def test_nothing_compared_is_inconclusive_not_default_confirmed(self):
        model = speed_with({})
        clock = FakeClock()
        session = StubSession(model=model, clock=clock)

        def system():
            out = session.system()
            out["memory"]["available_mib"] = 40000 if not session.runs else 3000
            return out
        t = autotune.Tuner(session, BASE, {}, make_ctx(), autotune.Options(wait_ram_s=30, per_request=False), clock=clock,
                           sleep=clock.sleep, system=system, total_ram_mib=64000, quiet=lambda: None)
        r = t.run()
        self.assertEqual(r.outcome, "inconclusive")
        self.assertIn("could not be tried", r.message)

    def test_first_oom_stops_the_search(self):
        def fail(args, env):
            return StartFailed("OOM", "out of memory") if flag(args, "--kv-resident") == "65536" else None
        t, s, r = run_tuner(speed_with({}), session_kw={"fail": fail})
        self.assertTrue(any("ran out of memory" in n for n in r.notes))
        self.assertEqual(sum(1 for a, _ in s.runs if flag(a, "--kv-resident") == "65536"), 1)
        self.assertEqual(r.settings["args"], {})

    def test_unsupported_candidate_fails_alone(self):
        def fail(args, env):
            return StartFailed("UNSUPPORTED", "unknown argument: --kv-resident") if flag(args, "--kv-resident") == "65536" else None
        ctx = make_ctx()
        ctx.facts = {"workers": 6}
        t, s, r = run_tuner(speed_with({}), ctx=ctx, session_kw={"fail": fail})
        failed = [a for a in r.arms if a["status"] == "failed"]
        self.assertTrue(failed and "UNSUPPORTED" in failed[0]["reason"])
        self.assertEqual(r.settings["args"], {})

    def test_room_goal_refuses_a_faster_candidate_that_takes_more_memory(self):
        model = speed_with({("--kv-resident", "65536"): 1.1})
        rss = lambda a, e: 20000.0 if flag(a, "--kv-resident") == "65536" else 12000.0
        t, s, r = run_tuner(model, opts=autotune.Options(goal="room", ram_wish_pct=20, reserve_mib=1536), session_kw={"rss": rss})
        self.assertEqual(r.settings["args"], {})
        self.assertTrue(any("more memory" in a["reason"] for a in r.arms))
        t, s, r = run_tuner(model, opts=autotune.Options(goal="speed"), session_kw={"rss": rss})
        self.assertEqual(r.settings["args"], {"--kv-resident": 65536})

    def test_cancel_changes_nothing(self):
        cancel = threading.Event()
        model = speed_with({("--kv-resident", "65536"): 1.1})
        n = {"i": 0}

        def progress(p):
            if p["stage"] == "rounds":
                cancel.set()
        t, s, r = run_tuner(model, cancel=cancel, progress=progress)
        self.assertEqual(r.outcome, "cancelled")
        self.assertEqual(r.settings["args"], {})

    def test_the_time_chosen_bounds_the_number_of_restarts(self):
        model = speed_with({})
        t, s, r = run_tuner(model, session_kw={"run_s": 300.0}, opts=autotune.Options(budget_s=1200))
        self.assertLessEqual(len(s.runs), 1200 / 300 + 1)
        t, s, r = run_tuner(model, session_kw={"run_s": 60.0}, opts=autotune.Options(budget_s=1200))
        self.assertGreater(len(s.runs), 4)

    def test_progress_reports_stage_and_time_left(self):
        seen = []
        t, s, r = run_tuner(speed_with({}), progress=seen.append)
        self.assertTrue(any(p["stage"] == "base" for p in seen))
        etas = [p["eta_s"] for p in seen if p.get("eta_s") is not None]
        self.assertTrue(etas)
        self.assertLessEqual(etas[-1], etas[0])
        self.assertEqual(seen[-1]["stage"], "done")

    def test_progress_carries_the_speed_of_every_measurement_so_far(self):
        seen = []
        t, s, r = run_tuner(speed_with({("--spec", "6"): 1.2}), progress=seen.append)
        last = [p for p in seen if p.get("speeds")][-1]["speeds"]
        self.assertEqual(last[0]["label"], "base")
        self.assertAlmostEqual(last[0]["tok_s"], 60.0, places=1)
        self.assertTrue(all(isinstance(x["tok_s"], float) and x["tok_s"] > 0 for x in last))
        self.assertGreater(len(last), 1, "the candidates' speeds follow the base")

    def test_per_request_value_needs_every_round(self):
        wins = {"n": 0}

        def model(args, env, tune=None):
            if tune and tune.get("spec_min_p") == 0.7:
                return 60.0 * 1.06
            return 60.0
        t, s, r = run_tuner(model, session_kw={"noise": 0.002})
        self.assertEqual(r.settings["per_request"].get("spec_min_p"), 0.7)
        self.assertEqual(r.outcome, "improved")

        calls = {"n": 0}

        def noisy(args, env, tune=None):
            if tune and tune.get("spec_min_p") == 0.7:
                calls["n"] += 1
                return 60.0 * (1.06 if calls["n"] % 3 else 0.99)      # one round in three it does not win
            return 60.0
        t, s2, r = run_tuner(noisy, session_kw={"noise": 0.0})
        self.assertNotIn("spec_min_p", r.settings["per_request"])

    def test_sending_any_setting_is_not_a_gain(self):
        # the p620 artifact: every request that carried a setting read ~10% faster than the plain default, even a setting equal
        # to what the engine already uses. The control point catches it.
        def model(args, env, tune=None):
            return 60.0 * (1.10 if tune else 1.0)
        t, s, r = run_tuner(model, session_kw={"noise": 0.002})
        self.assertEqual(r.settings["per_request"], {})
        self.assertNotEqual(r.outcome, "improved")

    def test_a_real_per_request_gain_is_confirmed_in_a_fresh_engine(self):
        def model(args, env, tune=None):
            return 60.0 * (1.06 if tune and tune.get("spec_min_p") == 0.7 else 1.0)
        t, s, r = run_tuner(model, session_kw={"noise": 0.002})
        self.assertEqual(r.settings["per_request"], {"spec_min_p": 0.7})
        self.assertEqual(r.per_request_confirm["status"], "PASS")
        self.assertGreaterEqual(sum(1 for x in r.per_request_confirm["ratios"] if x > 1.0), 4)
        self.assertAlmostEqual(r.gain_pct, 6.0, delta=1.5)

    def test_thorough_confirms_with_five_pairs(self):
        model = speed_with({("--kv-resident", "65536"): 1.06})
        t, s, r = run_tuner(model, opts=autotune.Options(mode="thorough", budget_s=7200), session_kw={"noise": 0.003})
        self.assertEqual(r.outcome, "improved")
        self.assertEqual(r.pairs, 5)
        self.assertGreaterEqual(r.faster_pairs, 4)

    def test_prompt_regression_blocks_a_decode_win(self):
        model = speed_with({("--kv-resident", "65536"): 1.08})
        pp = lambda a, e: 1500.0 * (0.8 if flag(a, "--kv-resident") == "65536" else 1.0)
        t, s, r = run_tuner(model, session_kw={"pp": pp, "noise": 0.001})
        self.assertEqual(r.settings["args"], {})

    def test_base_that_cannot_start_is_reported(self):
        t, s, r = run_tuner(speed_with({}), session_kw={"fail": lambda a, e: StartFailed("FAIL_BOOT", "x")})
        self.assertEqual(r.outcome, "base_failed")
        self.assertEqual(len(s.runs), 1)

    def test_coding_metric_uses_prompt_speed(self):
        m = autotune.metric({"decode_tok_s": 100.0, "pp_tok_s": 400.0}, "coding")
        self.assertAlmostEqual(m, 200.0)
        self.assertEqual(autotune.metric({"decode_tok_s": 100.0, "pp_tok_s": 400.0}, "chat"), 100.0)


class DecisionRulesTest(unittest.TestCase):
    def test_threshold_follows_the_noise(self):
        self.assertAlmostEqual(autotune.accept_threshold(0.03, 0.0, 0.0), 1.03)
        self.assertAlmostEqual(autotune.accept_threshold(0.03, 0.025, 0.0), 1.05)
        self.assertAlmostEqual(autotune.accept_threshold(0.03, 0.0, 0.04), 1.08)

    def test_is_win(self):
        self.assertTrue(autotune.is_win([1.05, 1.06], [1.0], threshold=1.03))
        self.assertFalse(autotune.is_win([1.05, 0.99], [1.0], threshold=1.03))
        self.assertFalse(autotune.is_win([1.04, 1.04], [0.9], threshold=1.03))
        self.assertTrue(autotune.is_win([1.05, 1.06, 1.04, 1.05, 0.99], [1.0], threshold=1.03, need=4))


class RecheckTest(unittest.TestCase):
    def _preset(self):
        return {"settings": {"args": {"--host-core": "last"}, "env": {}, "per_request": {}},
                "measured": {"decode_tok_s": 60.0, "pp4k_tok_s": 1500.0, "rss_peak_mib": 12000.0, "vram_peak_mib": 10000.0}}

    def test_keep_when_as_fast(self):
        s = StubSession(model=lambda a, e, tune=None: 59.0, noise=0.0)
        out = autotune.recheck(s, BASE, self._preset(), ctx=make_ctx())
        self.assertEqual(out["verdict"], "keep")
        self.assertEqual(len(s.runs), 1)
        self.assertEqual(flag(s.runs[0][0], "--host-core"), "last")

    def test_retune_suggested_when_slower(self):
        s = StubSession(model=lambda a, e, tune=None: 50.0, noise=0.0)
        out = autotune.recheck(s, BASE, self._preset(), ctx=make_ctx())
        self.assertEqual(out["verdict"], "retune")
        self.assertIn("slower", out["reason"])

    def test_engine_update_adds_a_baseline_start(self):
        s = StubSession(model=lambda a, e, tune=None: 60.0 if flag(a, "--host-core") else 62.0, noise=0.0)
        out = autotune.recheck(s, BASE, self._preset(), ctx=make_ctx(), engine_changed=True)
        self.assertEqual(len(s.runs), 2)
        self.assertEqual(out["verdict"], "retune")

    def test_a_start_that_fails_suggests_retune(self):
        s = StubSession(model=lambda a, e, tune=None: 60.0, fail=lambda a, e: StartFailed("UNSUPPORTED", "unknown argument"))
        out = autotune.recheck(s, BASE, self._preset(), ctx=make_ctx())
        self.assertEqual(out["verdict"], "retune")


if __name__ == "__main__":
    unittest.main()
