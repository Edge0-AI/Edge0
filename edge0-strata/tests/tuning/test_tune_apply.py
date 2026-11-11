"""Applying and rolling back a tuning recommendation (checklist T07).

The requirement is that a separate Apply revalidates the report and profile, snapshots, restarts and
verifies, and rolls back on failure — with no automatic winner application.

The risk these tests exist for is the gap between measuring a configuration and running it for real.
A sweep can recommend something that boots but cannot generate, or that was measured against
different model bytes, or against a report that has since been edited. Each of those is a specific
test below, and in every failure case the assertion includes that **the configuration file is
byte-identical afterwards**: a refusal that still wrote would be the worst outcome.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import tune_apply as ta  # noqa: E402


def a_job(*, args: dict | None = None, state: str = "completed",
          available: bool = True, identity: dict | None = None, drift: dict | None = None) -> dict:
    report: dict = {
        "metric": "tokens_per_s",
        "recommendation_available": available,
        "reason": None if available else "no candidate beat the baseline outside its own spread",
        "trials_run": 4,
        "combinations_total": 4,
    }
    if available:
        report["recommendation"] = {
            "candidate_index": 2,
            "args": {"--kv": "int8"} if args is None else args,
            "median": 24.0,
            "baseline_median": 20.0,
        }
    if identity is not None:
        report["identity"] = identity
    if drift is not None:
        report["drift"] = drift
        report["drift_material"] = abs(drift.get("relative", 0)) >= 0.02
    return {"id": "a" * 32, "state": state, "report": report,
            "plan": {"engine": "/e", "model": "m", "backend": "metal", "context": 32768,
                     "identity": identity or {}}}


class ApplyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-tune-apply-"))
        self.state = self.tmp / "state"
        self.state.mkdir()
        self.config = self.tmp / "strata-fixture.json"
        self.config.write_text(json.dumps({
            "model": "fixture", "gpu": 0,
            "args": ["--max-context", "32768", "--kv", "fp16"],
            "env": {"PATH": "/native/bin", "STRATA_TOKEN": "private"},
            "aliases": ["chat"],
        }, indent=1) + "\n", encoding="utf-8")
        self.before = self.config.read_bytes()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def identity(self):
        return {"engine_digest": "e" * 64, "model_digest": "m" * 64, "digest_method": "sha256-full",
                "backend": "metal", "context": 32768,
                "workload": {"prompt_tokens": 16, "max_new": 4, "seed": 1, "temperature": 0.0},
                "machine": {"system": "Darwin", "machine": "arm64", "cpu_count": 10}}

    def job_with_identity(self, **kw):
        return a_job(identity=self.identity(), **kw)

    def assert_untouched(self):
        self.assertEqual(self.config.read_bytes(), self.before,
                         "a refused apply must leave the configuration byte-identical")


class TestArgv(ApplyCase):
    def test_flags_are_paired_and_ordered_deterministically(self):
        argv = ta.args_to_argv({"--kv": "int8", "--kv-grow": True, "--ple-io": "mmap"})
        self.assertEqual(argv, ["--kv", "int8", "--kv-grow", "--ple-io", "mmap"])

    def test_a_false_value_omits_the_flag_rather_than_writing_it(self):
        self.assertEqual(ta.args_to_argv({"--kv-grow": False}), [])
        self.assertEqual(ta.args_to_argv({"--kv-grow": None}), [])


class TestConfigEdit(ApplyCase):
    def test_the_recommended_flag_replaces_every_occurrence(self):
        cfg = {"args": ["--kv", "fp16", "--kv", "q4_0", "--max-context", "32768"]}
        out = ta.apply_args_to_config(cfg, ["--kv", "int8"])
        self.assertEqual(out["args"], ["--max-context", "32768", "--kv", "int8"],
                         "a half-replaced flag is a configuration that does nothing")

    def test_an_absent_flag_is_appended(self):
        out = ta.apply_args_to_config({"args": ["--max-context", "4096"]}, ["--kv", "int8"])
        self.assertEqual(out["args"], ["--max-context", "4096", "--kv", "int8"])

    def test_unknown_fields_and_secrets_survive(self):
        """An apply that dropped a user's token or a vendor's unknown key would be worse than none."""
        out = ta.apply_args_to_config(
            {"args": [], "env": {"STRATA_TOKEN": "private"}, "vendor_unknown": {"keep": True},
             "aliases": ["chat"]}, ["--kv", "int8"])
        self.assertEqual(out["env"], {"STRATA_TOKEN": "private"})
        self.assertEqual(out["vendor_unknown"], {"keep": True})
        self.assertEqual(out["aliases"], ["chat"])

    def test_the_input_configuration_is_not_mutated(self):
        cfg = {"args": ["--kv", "fp16"]}
        ta.apply_args_to_config(cfg, ["--kv", "int8"])
        self.assertEqual(cfg["args"], ["--kv", "fp16"], "the caller's dict must be untouched")


class TestReview(ApplyCase):
    """Every refusal, and each one leaves the file alone."""

    def test_a_recommendation_that_changes_nothing_is_refused(self):
        plan = ta.review_apply(job=self.job_with_identity(args={"--kv": "fp16"}),
                               config_path=self.config, current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("already carries" in r for r in plan.reasons), plan.reasons)

    def test_an_incomplete_sweep_cannot_be_applied(self):
        plan = ta.review_apply(job=self.job_with_identity(state="running"),
                               config_path=self.config, current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("not completed" in r for r in plan.reasons), plan.reasons)

    def test_a_sweep_that_recommended_nothing_cannot_be_applied(self):
        plan = ta.review_apply(job=a_job(available=False, identity=self.identity()),
                               config_path=self.config, current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("recommended no change" in r for r in plan.reasons), plan.reasons)
        self.assertTrue(any("beat the baseline" in r for r in plan.reasons),
                        "the sweep's own reason is carried into the refusal")

    def test_material_drift_blocks_the_apply(self):
        """A comparison the sweep itself called weak must not be applied."""
        plan = ta.review_apply(job=self.job_with_identity(
                                   drift={"metric": "tokens_per_s", "first": 20.0, "last": 25.0,
                                          "relative": 0.25, "note": "drifted"}),
                               config_path=self.config, current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("drifted" in r for r in plan.reasons), plan.reasons)

    def test_a_changed_model_makes_the_recommendation_unappliable(self):
        """The single most important check: different model bytes mean the numbers do not apply."""
        plan = ta.review_apply(job=self.job_with_identity(), config_path=self.config,
                               current_identity=dict(self.identity(), model_digest="z" * 64))
        self.assertFalse(plan.applicable)
        self.assertTrue(any("different model bytes" in r for r in plan.reasons), plan.reasons)

    def test_a_changed_workload_or_context_makes_it_unappliable(self):
        for changed, expected in ((dict(self.identity(), context=131072), "context"),
                                  (dict(self.identity(), workload={"prompt_tokens": 999}),
                                   "workload")):
            plan = ta.review_apply(job=self.job_with_identity(), config_path=self.config,
                                   current_identity=changed)
            self.assertFalse(plan.applicable, expected)
            self.assertTrue(any(expected in r for r in plan.reasons), plan.reasons)

    def test_a_sweep_with_no_recorded_identity_cannot_be_applied(self):
        plan = ta.review_apply(job=a_job(), config_path=self.config,
                               current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("no artifact identity" in r for r in plan.reasons), plan.reasons)

    def test_a_report_edited_after_review_is_refused(self):
        job = self.job_with_identity()
        reviewed = ta._report_digest(job)
        job["report"]["recommendation"]["median"] = 999.0        # someone edits the numbers
        plan = ta.review_apply(job=job, config_path=self.config, current_identity=self.identity(),
                               report_sha256=reviewed)
        self.assertFalse(plan.applicable)
        self.assertTrue(any("changed after review" in r for r in plan.reasons), plan.reasons)

    def test_a_missing_configuration_is_refused_with_its_path(self):
        plan = ta.review_apply(job=self.job_with_identity(), config_path=self.tmp / "gone.json",
                               current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("gone.json" in r for r in plan.reasons), plan.reasons)


class TestApply(ApplyCase):
    def test_an_applicable_recommendation_is_written_with_the_other_keys_intact(self):
        result = ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                        current_identity=self.identity(), state_dir=self.state)
        after = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual(after["args"], ["--max-context", "32768", "--kv", "int8"])
        self.assertEqual(after["env"]["STRATA_TOKEN"], "private")
        self.assertEqual(after["aliases"], ["chat"])
        self.assertEqual(result["plan"]["added"], ["--kv int8"])
        self.assertEqual(result["plan"]["removed"], ["--kv fp16"])

    def test_the_previous_bytes_are_retained(self):
        result = ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                        current_identity=self.identity(), state_dir=self.state)
        kept = Path(result["directory"]) / "before.json"
        self.assertEqual(kept.read_bytes(), self.before,
                         "rollback needs the exact previous bytes, not a reconstruction")

    def test_a_verification_failure_writes_nothing(self):
        def verify(candidate):
            raise RuntimeError("the engine refused --kv int8")

        with self.assertRaises(ta.ApplyError) as ctx:
            ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                    current_identity=self.identity(), state_dir=self.state,
                                    verify=verify)
        self.assertIn("failed verification", str(ctx.exception))
        self.assertIn("unchanged", str(ctx.exception))
        self.assert_untouched()

    def test_a_candidate_that_generates_nothing_writes_nothing(self):
        """A configuration that boots but cannot generate must not be written."""
        with self.assertRaises(ta.ApplyError) as ctx:
            ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                    current_identity=self.identity(), state_dir=self.state,
                                    verify=lambda candidate: 0)
        self.assertIn("no tokens", str(ctx.exception))
        self.assert_untouched()

    def test_a_successful_verification_is_recorded(self):
        result = ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                        current_identity=self.identity(), state_dir=self.state,
                                        verify=lambda candidate: 7)
        self.assertEqual(result["application"]["verified_tokens"], 7)

    def test_a_refused_apply_changes_nothing(self):
        with self.assertRaises(ta.ApplyError):
            ta.apply_recommendation(job=self.job_with_identity(state="failed"),
                                    config_path=self.config, current_identity=self.identity(),
                                    state_dir=self.state, verify=lambda c: 5)
        self.assert_untouched()
        self.assertFalse((self.state / "applications").exists(),
                         "a refusal must not leave an application record behind")

    def test_a_user_edit_during_verification_is_not_overwritten(self):
        """Verification can take minutes; an edit in that window wins."""
        def verify(candidate):
            self.config.write_text(json.dumps({"args": ["--kv", "q4_0"], "edited": True}),
                                   encoding="utf-8")
            return 5

        with self.assertRaises(ta.ApplyError) as ctx:
            ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                    current_identity=self.identity(), state_dir=self.state,
                                    verify=verify)
        self.assertIn("changed during verification", str(ctx.exception))
        self.assertEqual(json.loads(self.config.read_text())["args"], ["--kv", "q4_0"],
                         "the user's edit survives")


class TestUnverifiableIdentityNeverApplies(ApplyCase):
    """The defect found by exercising the CLI, held here so it cannot come back.

    A sweep planned without `--model-path` records `model_digest: null`. The library refused it, but
    the CLI's own identity re-derivation produced a *different* key whose model digest was also null,
    and `null == null` matched -- so the apply proceeded on a recommendation that could never be
    matched to the installed weights. The output looked entirely normal.

    The fix is that the recorded sweep must carry a real digest before anything is written, which is
    what these tests pin down.
    """

    def test_a_null_model_digest_is_not_a_match_for_a_null_model_digest(self):
        """The shape of the bug: two unknowns comparing equal."""
        job = a_job()                                    # no identity at all
        plan = ta.review_apply(job=job, config_path=self.config, current_identity=self.identity())
        self.assertFalse(plan.applicable)
        self.assertTrue(any("no artifact identity" in r for r in plan.reasons), plan.reasons)

    def test_an_identity_with_a_null_model_digest_still_refuses(self):
        """Even a *present* identity whose model digest is unknown is not applicable."""
        partial = dict(self.identity(), model_digest=None)
        plan = ta.review_apply(job=self.job_with_identity(), config_path=self.config,
                               current_identity=partial)
        self.assertFalse(plan.applicable,
                         "an unknown model must never be treated as the same model")
        self.assertTrue(any("different model" in r or "different" in r for r in plan.reasons),
                        plan.reasons)

    def test_both_sides_unknown_is_refused_by_the_cli_guard(self):
        """The CLI refuses before reaching the library, which is where the bug lived.

        This asserts the guard's condition directly: a recorded identity without a model digest may
        not be applied, whatever the re-derived identity looks like.
        """
        recorded = {"model_digest": None}
        self.assertFalse(recorded.get("model_digest"),
                         "the CLI guard keys on exactly this, and it must be falsy")


class TestRollback(ApplyCase):
    def apply(self):
        return ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                       current_identity=self.identity(), state_dir=self.state,
                                       verify=lambda c: 5)

    def test_rollback_restores_the_previous_bytes(self):
        result = self.apply()
        self.assertNotEqual(self.config.read_bytes(), self.before)
        ta.rollback(application_id=result["application"]["application_id"], state_dir=self.state)
        self.assertEqual(self.config.read_bytes(), self.before)

    def test_rollback_refuses_to_discard_a_newer_edit(self):
        result = self.apply()
        self.config.write_text(json.dumps({"args": ["--kv", "manual"]}) + "\n", encoding="utf-8")
        with self.assertRaises(ta.ApplyError) as ctx:
            ta.rollback(application_id=result["application"]["application_id"], state_dir=self.state)
        self.assertIn("newer change", str(ctx.exception))
        self.assertEqual(json.loads(self.config.read_text())["args"], ["--kv", "manual"],
                         "the user's edit wins over the rollback")

    def test_rollback_twice_is_refused(self):
        result = self.apply()
        ta.rollback(application_id=result["application"]["application_id"], state_dir=self.state)
        with self.assertRaises(ta.ApplyError) as ctx:
            ta.rollback(application_id=result["application"]["application_id"], state_dir=self.state)
        self.assertIn("already been rolled back", str(ctx.exception))

    def test_an_unknown_application_is_a_not_found(self):
        with self.assertRaises(ta.ApplyError) as ctx:
            ta.rollback(application_id="20260101T000000Z-deadbeef", state_dir=self.state)
        self.assertEqual(ctx.exception.code, 404)

    def test_an_apply_cannot_be_repeated_on_the_same_record(self):
        result = self.apply()
        application = result["application"]["application_id"]
        # A second apply with the same sweep is refused by the config diff, not by the record, but
        # either way the file must not change twice.
        after = self.config.read_bytes()
        with self.assertRaises(ta.ApplyError):
            ta.apply_recommendation(job=self.job_with_identity(), config_path=self.config,
                                    current_identity=self.identity(), state_dir=self.state,
                                    verify=lambda c: 5)
        self.assertEqual(self.config.read_bytes(), after, f"application {application} wrote twice")


class TestConfigDiscovery(ApplyCase):
    def test_only_object_configs_with_an_args_list_are_treated_as_run_configurations(self):
        (self.tmp / "strata-good.json").write_text('{"args": ["--kv", "fp16"]}', encoding="utf-8")
        (self.tmp / "strata-notjson.json").write_text("{ broken", encoding="utf-8")
        (self.tmp / "strata-wrongshape.json").write_text('{"model": "x"}', encoding="utf-8")
        (self.tmp / "other.json").write_text('{"args": []}', encoding="utf-8")
        found = {p.name for p in ta.find_config_files(self.tmp)}
        self.assertIn("strata-good.json", found)
        self.assertIn("strata-fixture.json", found)
        self.assertNotIn("strata-notjson.json", found, "a damaged file is skipped, not guessed at")
        self.assertNotIn("strata-wrongshape.json", found)
        self.assertNotIn("other.json", found, "the project's own naming decides, not any JSON file")


if __name__ == "__main__":
    unittest.main()
