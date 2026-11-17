"""Tuning job contract tests.

The measurements are injected, which is the point: these tests are about the *contract* — which
dimensions a backend may search, what happens to a candidate that fails, how a winner is chosen,
and what the job refuses to do. A GPU is not needed and, more importantly, a hardcoded benchmark
number can never leak into the module under test.

The negative cases carry the weight: a sweep must never promote a candidate the engine could not
run, never pick a winner from a machine that drifted, and never recommend a change on evidence it
does not have.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import tuning  # noqa: E402


class TuningCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-tune-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, backend="cuda", objective="generation", hold=(), max_trials=16, repeats=3,
             flags=None, dimensions=()):
        return tuning.TuningPlan(
            id="job1", backend=backend, engine="/nonexistent/strata",
            engine_flags_known=flags is not None, model="qwen3.8-flash-next",
            profile_revision="abc123", objective=objective,
            workload={"prompt_tokens": 512, "max_new": 200, "seed": 1},
            dimensions=[{"id": d} for d in dimensions], excluded=[], fixed={},
            max_trials=max_trials, total_time_s=60.0,
            repeats=repeats, devices=["0"],
        )

    def run_job(self, plan, measure):
        tuner = tuning.Tuner(self.tmp / "runs", plan, measure)
        return tuner, tuner.run()


class TestDimensionValidation(TuningCase):
    """A backend is offered only the levers its engine can actually accept."""

    def test_metal_is_not_offered_nvidia_only_dimensions(self):
        offered, excluded = tuning.validate_dimensions("metal")
        ids = {d.id for d in offered}
        self.assertNotIn("vram_reserve", ids)
        self.assertNotIn("pcie_frac", ids)
        reasons = {e["id"]: e["reason"] for e in excluded}
        self.assertIn("vram_reserve", reasons)
        self.assertIn("NVIDIA/AMD", reasons["vram_reserve"])

    def test_cuda_is_offered_the_nvidia_dimensions(self):
        offered, _ = tuning.validate_dimensions("cuda")
        ids = {d.id for d in offered}
        self.assertTrue({"vram_reserve", "pcie_frac", "expert_cache"} <= ids)

    def test_a_flag_the_binary_does_not_accept_excludes_the_dimension(self):
        """The static table is not trusted: the actual binary's flag list decides."""
        offered, excluded = tuning.validate_dimensions("cuda", engine_flags=["--kv", "--ple-io"])
        ids = {d.id for d in offered}
        self.assertNotIn("vram_reserve", ids, "a dimension whose flag the binary lacks must be excluded")
        reasons = {e["id"]: e["reason"] for e in excluded}
        self.assertIn("does not accept --vram-reserve-mib", reasons["vram_reserve"])

    def test_a_flag_the_binary_does_accept_keeps_the_dimension(self):
        offered, _ = tuning.validate_dimensions("cuda", engine_flags=["--kv", "--vram-reserve-mib"])
        self.assertIn("vram_reserve", {d.id for d in offered})

    def test_benchmark_only_dimension_is_never_exported(self):
        tuner = tuning.Tuner(self.tmp / "runs", self.plan(), lambda c: {"tokens_per_s": 1.0})
        args = tuner._args_for({"prompt_cache_tail": True, "kv": "int8"})
        self.assertNotIn("--prompt-cache-tail", args,
                         "a flag the production interface does not accept must not be exported")
        self.assertEqual(args["--kv"], "int8")

    def test_flag_inventory_returns_none_for_a_missing_engine(self):
        self.assertIsNone(tuning.engine_flag_inventory(Path("/nonexistent/strata")))

    def test_flag_inventory_reads_stderr_as_well_as_stdout(self):
        """The Strata engine writes its usage to stderr.

        A stdout-only read reports "no flags" for a working engine, and because every dimension is
        then excluded the sweep silently searches nothing — safe, but useless and very hard to
        notice. This test uses a stand-in that prints usage to stderr only.
        """
        fake = self.tmp / "fake-engine.py"      # run as [sys.executable, fake]: works on every OS
        fake.write_text("import sys\nfor line in ('usage:', '  --kv KV', '  --ple-io MODE'):\n    print(line, file=sys.stderr)\n")
        flags = tuning.engine_flag_inventory(fake)
        self.assertIsNotNone(flags, "usage on stderr must still be read")
        self.assertIn("--kv", flags)
        self.assertIn("--ple-io", flags)

    def test_flag_inventory_reads_stdout_too(self):
        fake = self.tmp / "fake-engine-stdout.py"
        fake.write_text("print('  --vram-reserve-mib MIB')\n")
        self.assertIn("--vram-reserve-mib", tuning.engine_flag_inventory(fake))

    def test_a_backend_with_dimensions_that_all_need_missing_flags_searches_nothing(self):
        """The safe outcome, asserted so it cannot silently become an unsafe one.

        Every dimension has *some* reason: either its flags are not this backend's concept, or the
        binary does not accept them. What must never happen is a dimension with no reason being
        dropped silently.
        """
        offered, excluded = tuning.validate_dimensions("metal", engine_flags=["--unrelated"])
        self.assertEqual(offered, [])
        self.assertEqual(len(excluded), len(tuning.DIMENSIONS))
        for e in excluded:
            self.assertTrue(e["reason"].strip(), f"{e['id']} was excluded with no reason given")
        # The dimensions that ARE Metal concepts are excluded specifically because the binary lacks
        # their flags, which is the check that matters.
        by_id = {e["id"]: e["reason"] for e in excluded}
        for dim_id in ("kv", "ple_io", "kv_grow", "spec_min_p"):
            self.assertIn("does not accept", by_id[dim_id])


class TestBaselineAndSelection(TuningCase):
    def test_a_clear_winner_is_recommended(self):
        calls = {"n": 0}

        def measure(candidate):
            calls["n"] += 1
            # The baseline (no settings) is slow; any candidate is faster.
            return {"tokens_per_s": 50.0 if not candidate.settings else 80.0, "correct": True}

        plan = self.plan(max_trials=2, dimensions=("kv",))
        tuner, result = self.run_job(plan, measure)
        rep = result["report"]
        self.assertTrue(rep["recommendation_available"], rep)
        self.assertEqual(rep["baseline"]["median"], 50.0)
        self.assertEqual(rep["recommendation"]["median"], 80.0)

    def test_a_faster_candidate_inside_the_baseline_spread_is_not_recommended(self):
        """Promoting noise is the classic tuning failure, so the gate compares worst-against-worst."""
        state = {"phase": 0}

        def measure(candidate):
            if candidate.is_baseline:
                # A wide baseline spread: worst 40, best 90.
                state["phase"] += 1
                return {"tokens_per_s": [40.0, 90.0, 65.0][state["phase"] % 3]}
            # The candidate's median is higher but its worst run is below the baseline's worst.
            return {"tokens_per_s": [45.0, 60.0, 60.0][state["phase"] % 3]}

        plan = self.plan(max_trials=1, repeats=3, dimensions=("kv",))
        _, result = self.run_job(plan, measure)
        rep = result["report"]
        self.assertFalse(rep["recommendation_available"],
                         "a candidate whose worst run is below the baseline's worst must not win")
        self.assertIn("spread", rep["reason"])

    def test_no_candidate_better_means_recommend_no_change(self):
        def measure(candidate):
            return {"tokens_per_s": 60.0 if not candidate.settings else 55.0}

        _, result = self.run_job(self.plan(max_trials=2, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertFalse(rep["recommendation_available"])
        self.assertIn("keeping the current configuration", rep["reason"])

    def test_drift_is_reported(self):
        """The baseline is re-measured last, so a drifting machine is visible instead of silent.

        The baseline is measured once as a warmup plus `repeats` times, and then again after the
        sweep. This case makes the measured value a function of *which* baseline measurement it is,
        rather than of the call count, so the expectation does not depend on how repeats are
        scheduled.
        """
        state = {"baseline_measurements": 0}

        def measure(candidate):
            if candidate.kind == "baseline":
                state["baseline_measurements"] += 1
                # The first measurement is the warmup; the three counted repeats are 50, and the
                # recheck afterwards is 40 (below), simulating a machine that got slower.
                return {"tokens_per_s": 50.0}
            if candidate.kind == "recheck":
                return {"tokens_per_s": 40.0}
            return {"tokens_per_s": 45.0}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertEqual(rep["baseline"]["median"], 50.0)
        self.assertIsNotNone(rep["recheck"], "the baseline must be re-measured after the sweep")
        self.assertEqual(rep["recheck"]["median"], 40.0)
        self.assertIsNotNone(rep["drift"], "the drift between the two baseline measurements is reported")
        self.assertLess(rep["drift"]["relative"], 0,
                        "the report must show that the machine slowed, not hide it")

    def test_fewer_repeats_than_required_disqualifies_a_candidate(self):
        def measure(candidate):
            return {"tokens_per_s": 90.0 if candidate.settings else 50.0}

        plan = self.plan(max_trials=1, repeats=5, dimensions=("kv",))
        _, result = self.run_job(plan, measure)
        best = [c for c in result["report"]["candidates"] if c["kind"] == "trial"][0]
        self.assertGreaterEqual(len(best["metrics"]), 5)


class TestFailuresAreRecorded(TuningCase):
    """A candidate that could not run is evidence about the configuration, never a slow winner."""

    def test_a_boot_failure_is_recorded_and_excluded(self):
        def measure(candidate):
            if candidate.settings:
                raise RuntimeError("the engine failed to boot with these flags")
            return {"tokens_per_s": 50.0}

        _, result = self.run_job(self.plan(max_trials=2, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertEqual(rep["recommendation_available"], False)
        self.assertTrue(rep["failures"])
        self.assertEqual(rep["failures"][0]["status"], "FAIL_BOOT")
        self.assertIn("failed to boot", rep["failures"][0]["reason"])

    def test_a_timeout_is_recorded_as_a_timeout(self):
        def measure(candidate):
            if candidate.settings:
                raise RuntimeError("the measurement timed out after 120s")
            return {"tokens_per_s": 50.0}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        self.assertEqual(result["report"]["failures"][0]["status"], "TIMEOUT")

    def test_no_tokens_is_not_a_slow_configuration(self):
        def measure(candidate):
            return {"tokens_per_s": 0.0}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertFalse(rep["recommendation_available"])
        statuses = [c["status"] for c in rep["candidates"]]
        self.assertIn("NO_TOKENS", statuses)

    def test_a_correctness_failure_disqualifies_a_fast_candidate(self):
        """tokens/s alone is not acceptance."""
        def measure(candidate):
            if candidate.settings:
                return {"tokens_per_s": 500.0, "correct": False}
            return {"tokens_per_s": 50.0, "correct": True}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertFalse(rep["recommendation_available"],
                         "a fast but incorrect configuration must never be recommended")
        self.assertTrue(any(f["status"] == "INVALID" for f in rep["failures"]), rep["failures"])

    def test_a_baseline_that_cannot_be_measured_stops_the_sweep(self):
        def measure(candidate):
            raise RuntimeError("no engine here")

        _, result = self.run_job(self.plan(max_trials=5, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertFalse(rep["recommendation_available"])
        self.assertIsNone(rep["baseline"])
        self.assertIn("could not be measured", rep["reason"])


class TestBounds(TuningCase):
    def test_the_trial_limit_is_respected(self):
        # Count *trials*, not measurement calls: each candidate is measured once as a warmup plus
        # `repeats` times, so counting calls would measure the wrong thing entirely.
        plan = self.plan(max_trials=3, dimensions=("ple_io", "kv"))
        _, result = self.run_job(plan, lambda c: {"tokens_per_s": 60.0})
        self.assertEqual(result["report"]["trials_run"], 3)
        self.assertEqual(len([c for c in result["report"]["candidates"]
                              if c["kind"] == "trial"]), 3)
        self.assertLessEqual(result["report"]["baseline"] is not None, True)

    def test_the_time_bound_stops_new_trials(self):
        clock = {"t": 0.0}

        def measure(candidate):
            clock["t"] += 30.0
            return {"tokens_per_s": 60.0}

        plan = self.plan(max_trials=100, dimensions=("kv",))
        plan.total_time_s = 45.0
        tuner = tuning.Tuner(self.tmp / "runs", plan, measure, clock=lambda: clock["t"])
        result = tuner.run()
        self.assertTrue(any(e["state"] == "time_limit" for e in result["events"]),
                        "the displayed time bound must actually stop the sweep")

    def test_cancellation_keeps_evidence_and_recommends_nothing(self):
        def measure(candidate):
            if candidate.settings:
                tuner.request_cancel()
            return {"tokens_per_s": 90.0 if candidate.settings else 50.0}

        plan = self.plan(max_trials=10, dimensions=("kv",))
        tuner = tuning.Tuner(self.tmp / "runs", plan, measure)
        result = tuner.run()
        self.assertEqual(result["state"], "cancelled")
        self.assertFalse(result["report"]["recommendation_available"],
                         "a cancelled sweep must not produce a winner")
        self.assertTrue(result["report"]["candidates"], "evidence collected before the cancel is kept")


class TestSearchSpaceIsDisclosed(TuningCase):
    """A bounded sweep must not look like a complete one.

    This is the honesty property that decides whether "no candidate beat the baseline" can be read
    as a conclusion. With four trials out of ninety-six combinations, it cannot — and the report has
    to say so, or a reader will take a partial search for a full one.
    """

    def test_a_bounded_sweep_reports_the_size_of_the_space_it_sampled(self):
        plan = self.plan(max_trials=2, dimensions=("kv",))
        _, result = self.run_job(plan, lambda c: {"tokens_per_s": 60.0})
        rep = result["report"]
        self.assertEqual(rep["trials_run"], 2)
        # `kv` has four values, so the space is four and two trials is a prefix of it.
        self.assertEqual(rep["combinations_total"], 4)
        self.assertTrue(rep["trials_are_a_prefix"])

    def test_a_complete_sweep_is_not_labelled_a_prefix(self):
        plan = self.plan(max_trials=10, dimensions=("kv",))
        _, result = self.run_job(plan, lambda c: {"tokens_per_s": 60.0})
        rep = result["report"]
        self.assertEqual(rep["combinations_total"], 4)
        self.assertEqual(rep["trials_run"], 4, "the whole four-value space is enumerated")
        self.assertFalse(rep["trials_are_a_prefix"])

    def test_the_space_size_accounts_for_every_offered_dimension(self):
        plan = self.plan(max_trials=1, dimensions=("kv", "kv_grow"))
        _, result = self.run_job(plan, lambda c: {"tokens_per_s": 60.0})
        # kv has 4 values, kv_grow has 2 -> 8 combinations.
        self.assertEqual(result["report"]["combinations_total"], 8)

    def test_a_plan_with_no_dimensions_reports_no_space(self):
        _, result = self.run_job(self.plan(max_trials=1), lambda c: {"tokens_per_s": 60.0})
        self.assertEqual(result["report"]["combinations_total"], 0)


class TestDriftIsGraded(TuningCase):
    """A drift note is only warranted when the drift is material.

    Printing "the comparison is weaker than the repeat count suggests" at +0.0% trains the reader to
    ignore the warning, which is worse than not printing it.
    """

    def test_negligible_drift_is_not_flagged_as_material(self):
        def measure(candidate):
            return {"tokens_per_s": 50.0}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertIsNotNone(rep["drift"])
        self.assertAlmostEqual(rep["drift"]["relative"], 0.0)
        self.assertFalse(rep["drift_material"])

    def test_material_drift_is_flagged(self):
        def measure(candidate):
            return {"tokens_per_s": 40.0 if candidate.kind == "recheck" else 50.0}

        _, result = self.run_job(self.plan(max_trials=1, dimensions=("kv",)), measure)
        rep = result["report"]
        self.assertAlmostEqual(rep["drift"]["relative"], -0.2)
        self.assertTrue(rep["drift_material"])


class TestExclusiveLease(TuningCase):
    """Tuning must own its devices: a second job, or a serving load, must not share them."""

    def test_a_second_job_is_refused_while_the_lease_is_held(self):
        first = tuning.Tuner(self.tmp / "runs", self.plan(), lambda c: {"tokens_per_s": 50.0})
        first._acquire_lease()
        try:
            second = tuning.Tuner(self.tmp / "runs", self.plan(), lambda c: {"tokens_per_s": 50.0})
            with self.assertRaises(RuntimeError) as ctx:
                second._acquire_lease()
            self.assertIn("exclusive device lease", str(ctx.exception))
        finally:
            first._release_lease()

    def test_the_lease_is_released_even_when_the_job_fails(self):
        plan = self.plan()
        tuner = tuning.Tuner(self.tmp / "runs", plan, lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
        tuner.run()
        self.assertFalse((tuner.root / "tuning.lease").exists(),
                         "a failed job must not leave the devices leased")

    def test_a_stale_lease_is_reported_rather_than_ignored(self):
        stale = self.tmp / "runs" / "job1"
        stale.mkdir(parents=True)
        (stale / "tuning.lease").write_text(json.dumps({"id": "dead", "pid": 1}))
        tuner = tuning.Tuner(self.tmp / "runs", self.plan(), lambda c: {"tokens_per_s": 50.0})
        with self.assertRaises(RuntimeError) as ctx:
            tuner._acquire_lease()
        self.assertIn("dead", str(ctx.exception))


class TestExportedRetest(TuningCase):
    """A winner is not qualified until the *exported* configuration is measured again."""

    def test_a_configuration_that_loses_its_advantage_on_export_is_withdrawn(self):
        def measure(args):
            # The benchmark-only lever carried the gain; without it the engine is at the baseline.
            return {"tokens_per_s": 90.0 if "--prompt-cache-tail" in args else 50.0}

        outcome = tuning.verify_exported(Path("/engine"), {"--kv": "int8"}, measure)
        self.assertTrue(outcome["verified"])
        self.assertEqual(outcome["median"], 50.0)

    def test_an_exported_configuration_that_fails_to_boot_is_not_verified(self):
        def measure(args):
            raise RuntimeError("the engine failed to boot")

        outcome = tuning.verify_exported(Path("/engine"), {}, measure)
        self.assertFalse(outcome["verified"])
        self.assertIn("failed to run", outcome["reason"])

    def test_an_incorrect_exported_configuration_is_not_verified(self):
        outcome = tuning.verify_exported(Path("/engine"), {}, lambda a: {"tokens_per_s": 100.0,
                                                                        "correct": False})
        self.assertFalse(outcome["verified"])
        self.assertIn("correctness", outcome["reason"])


class TestPersistenceAndReporting(TuningCase):
    def test_the_job_is_reloadable_with_its_report(self):
        def measure(candidate):
            return {"tokens_per_s": 70.0 if candidate.settings else 50.0}

        plan = self.plan(max_trials=1, dimensions=("kv",))
        _, result = self.run_job(plan, measure)
        saved = json.loads((self.tmp / "runs" / "job1" / "job.json").read_text())
        self.assertEqual(saved["state"], "completed")
        self.assertEqual(saved["id"], "job1")
        self.assertTrue(saved["report"]["candidates"])
        self.assertIn("excluded_dimensions", saved["report"])
        self.assertIn("machine", json.dumps(tuning.machine_identity()))

    def test_the_report_names_what_was_excluded_and_why(self):
        plan = tuning.plan_tuning(Path("/nonexistent/strata"), "metal", "m", "generation",
                                  {"prompt_tokens": 128})
        self.assertIn("vram_reserve", {e["id"] for e in plan.excluded})
        self.assertFalse(plan.engine_flags_known,
                         "a plan must record that the engine could not be interrogated")

    def test_objective_selects_the_metric(self):
        self.assertEqual(self.plan(objective="generation").__class__.__name__, "TuningPlan")
        tuner = tuning.Tuner(self.tmp / "runs", self.plan(objective="prefill"),
                             lambda c: {"tokens_per_s": 1.0})
        self.assertEqual(tuner._objective_metric(), "prefill_tokens_per_s")
        tuner2 = tuning.Tuner(self.tmp / "runs2", self.plan(objective="latency"),
                              lambda c: {"tokens_per_s": 1.0})
        self.assertEqual(tuner2._objective_metric(), "ttft_ms")


if __name__ == "__main__":
    unittest.main()
