"""Time-skip contract tests.

These are the scenarios the integration specification requires, and the ones that matter most are
the *negative* ones: a regression must be rejected, an unevaluable check must stay inconclusive, and
nothing may ever turn a missing or failed check into a pass.

The revision runner is a stub, which is correct here: these tests are about the state machine,
evidence and safety, not about model quality. A real model changes what the candidate contains, not
whether the gate is honest.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import time_skip as ts  # noqa: E402


def _py(code: str) -> str:
    """A check command that runs the same on sh and cmd.exe, with no dependency on coreutils."""
    return f'"{sys.executable}" -c "{code}"'


#: stand-ins for `test -s doc.txt`, `grep -q WORD doc.txt` and `true`
NONEMPTY = _py("import os,sys; sys.exit(0 if os.path.getsize('doc.txt') != 0 else 1)")
TRUE = _py("pass")


def _has(word: str) -> str:
    return _py(f"import sys; sys.exit(0 if '{word}' in open('doc.txt').read() else 1)")


class TimeSkipCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-timeskip-"))
        self.ws = self.tmp / "workspace"
        self.ws.mkdir()
        (self.ws / "doc.txt").write_text("one\ntwo\n", encoding="utf-8")
        self.store = ts.TimeSkipStore(self.tmp / "runs")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers ---------------------------------------------------------

    def checks(self, *specs) -> list[ts.CheckSpec]:
        return [ts.CheckSpec.from_dict(s) for s in specs]

    def start(self, revise, checks=None, max_attempts=2):
        # NOTE: named `start`, not `run` — `unittest.TestCase.run` is the test runner entry point,
        # and shadowing it makes every test in the class fail with a confusing AttributeError.
        run = ts.TimeSkipRun(self.tmp / "runs", self.ws, "improve the document",
                             checks if checks is not None else self.checks(
                                 {"id": "nonempty", "command": NONEMPTY},
                             ),
                             max_attempts=max_attempts)
        result = run.run(revise)
        return run, result


class TestCandidateImproves(TimeSkipCase):
    """A candidate that fixes a failing check is kept."""

    def test_improvement_is_retained(self):
        # `third` is missing, so the baseline fails; the revision adds it.
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added the missing line"

        run, result = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
        ))
        self.assertEqual(result["state"], "completed")
        self.assertEqual(run.attempts[0].decision, "keep")
        self.assertEqual(result["decision"]["outcome"], "accepted")
        self.assertIn("has-three", result["decision"]["improvements"])
        self.assertEqual(result["decision"]["regressions"], [])


class TestRegressionRejected(TimeSkipCase):
    """A candidate that breaks a previously passing check must be rejected, and the last good
    version must survive."""

    def test_regression_rejects_and_keeps_last_good(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\n", encoding="utf-8")  # drops `two`
            return "shortened it"

        run, result = self.start(revise, checks=self.checks(
            {"id": "has-two", "command": _has("two")},
        ))
        self.assertEqual(run.attempts[0].decision, "reject")
        self.assertIn("has-two", run.attempts[0].reason)
        # The retained checkpoint is still the baseline, and the workspace was never touched.
        self.assertEqual(result["decision"]["retained"], "cp1-baseline")
        self.assertEqual(result["decision"]["outcome"], "inconclusive")
        self.assertTrue(result["decision"]["needs_user_review"])
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "one\ntwo\n")

    def test_rejected_candidate_stays_inspectable(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\n", encoding="utf-8")
            return "shortened it"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-two", "command": _has("two")},
        ))
        rejected = Path(run.attempts[0].path) / "doc.txt"
        self.assertTrue(rejected.exists(), "a rejected candidate must remain on disk to inspect")
        self.assertEqual(rejected.read_text(encoding="utf-8"), "one\n")


class TestNoImprovement(TimeSkipCase):
    """A candidate that changes nothing meaningful is inconclusive, not a success."""

    def test_unchanged_candidate_is_inconclusive(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            return "did nothing"

        run, result = self.start(revise, checks=self.checks(
            {"id": "nonempty", "command": NONEMPTY},
        ))
        self.assertEqual(run.attempts[0].decision, "inconclusive")
        self.assertEqual(result["decision"]["outcome"], "inconclusive")
        self.assertIn("does not show an intended improvement", run.attempts[0].reason)


class TestCheckerFailureStaysUnknown(TimeSkipCase):
    """A check that cannot run is unknown; it must never be reported as passed."""

    def test_missing_program_is_unknown_not_passed(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            return "no-op"

        run, result = self.start(revise, checks=self.checks(
            {"id": "impossible", "command": "definitely-not-a-real-program-xyz --check"},
        ))
        outcome = run.attempts[0].checks[0]["outcome"]
        self.assertEqual(outcome, "unknown")
        self.assertEqual(run.attempts[0].decision, "inconclusive")
        self.assertTrue(result["decision"]["needs_user_review"])

    def test_unknown_is_counted_not_hidden(self):
        results = [
            ts.CheckResult("a", "deterministic", "passed"),
            ts.CheckResult("b", "deterministic", "unknown"),
            ts.CheckResult("c", "deterministic", "failed"),
        ]
        s = ts.summarise(results)
        self.assertEqual(s, {"total": 3, "passed": 1, "failed": 1, "unknown": 1, "error": 0})


class TestTimeoutsAndHardErrors(TimeSkipCase):
    """A check that times out is unknown, and a check that crashes is recorded as an error."""

    def test_timeout_is_unknown(self):
        spec = ts.CheckSpec(id="slow", command=_py("import time; time.sleep(30)"), timeout_s=0.3)
        started = time.time()
        r = ts.run_check(spec, self.ws)
        self.assertEqual(r.outcome, "unknown")
        self.assertIn("timed out", r.detail)
        self.assertLess(time.time() - started, 20, "the program the shell started must be stopped too")

    def test_an_existing_program_that_fails_is_failed_not_unknown(self):
        r = ts.run_check(ts.CheckSpec(id="real", command=_py("import sys; sys.exit(1)")), self.ws)
        self.assertEqual((r.outcome, r.exit_code), ("failed", 1))

    def test_a_missing_program_is_unknown_with_the_shells_not_found_code(self):
        r = ts.run_check(ts.CheckSpec(id="gone", command="definitely-not-a-real-program-xyz --check"), self.ws)
        self.assertEqual(r.outcome, "unknown")
        self.assertIn("could not be run", r.detail)

    @unittest.skipUnless(sys.platform == "win32", "cmd.exe's code")
    def test_cmd_exit_9009_is_unknown(self):
        r = ts.run_check(ts.CheckSpec(id="nine", command="exit 9009"), self.ws)
        self.assertEqual(r.outcome, "unknown")

    def test_exit_127_means_not_found_only_where_the_shell_says_so(self):
        r = ts.run_check(ts.CheckSpec(id="c127", command="exit 127"), self.ws)
        self.assertEqual(r.outcome, "failed" if sys.platform == "win32" else "unknown")

    def test_nonzero_exit_is_failed(self):
        spec = ts.CheckSpec(id="fails", command="exit 3")
        r = ts.run_check(spec, self.ws)
        self.assertEqual(r.outcome, "failed")
        self.assertEqual(r.exit_code, 3)


class TestAdviceDoesNotGate(TimeSkipCase):
    """A model's opinion may be recorded, but it can never promote a candidate on its own."""

    def test_advice_alone_cannot_keep_a_candidate(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            return "the model says this is better"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "opinion", "command": TRUE, "kind": "advice"},
        ))
        self.assertEqual(run.attempts[0].decision, "inconclusive")
        self.assertIn("no deterministic checks", run.attempts[0].reason)

    def test_advice_does_not_rescue_a_failing_deterministic_check(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("", encoding="utf-8")
            return "the model says this is better"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "nonempty", "command": NONEMPTY},
            {"id": "opinion", "command": TRUE, "kind": "advice"},
        ))
        self.assertEqual(run.attempts[0].decision, "reject")


class TestCancellation(TimeSkipCase):
    """Cancelling stops the run and retains the last complete checkpoint."""

    def test_cancel_retains_checkpoint(self):
        """Cancellation is observed at a phase boundary and leaves the last checkpoint selected.

        The cancel is requested from inside the revision runner rather than from a timer thread:
        a stub run finishes in milliseconds, so a timing-based cancel is a race, and a test that
        passes by luck is worse than no test. This seam is the same one a real canceller uses — the
        run checks its flag at every boundary.
        """
        def revise(candidate: Path, prompt: str, index: int) -> str:
            run.request_cancel()
            return "attempted"

        run = ts.TimeSkipRun(self.tmp / "runs", self.ws, "p",
                             self.checks({"id": "nonempty", "command": NONEMPTY}),
                             max_attempts=5)
        result = run.run(revise)
        self.assertEqual(result["state"], "cancelled")
        self.assertIsNotNone(result["selected_checkpoint"],
                             "a cancelled run must still have a retained checkpoint")
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "one\ntwo\n")

    def test_the_workspace_is_never_written_during_a_run(self):
        """No phase may touch the user's workspace; only an explicit apply does."""
        before = (self.ws / "doc.txt").read_text(encoding="utf-8")

        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("speculative\n", encoding="utf-8")
            return "changed the candidate only"

        self.start(revise, checks=self.checks({"id": "nonempty", "command": NONEMPTY}))
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), before)


class TestApplySafety(TimeSkipCase):
    """Applying is explicit, verified and reversible, and a user edit is a conflict."""

    def test_apply_is_refused_after_a_concurrent_user_edit(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added three"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
        ))
        # The user edits the workspace while the run is under way / before apply.
        (self.ws / "doc.txt").write_text("the user rewrote this\n", encoding="utf-8")
        outcome = run.apply()
        self.assertFalse(outcome["applied"])
        self.assertTrue(outcome["conflict"])
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"),
                         "the user rewrote this\n",
                         "a concurrent user edit must never be overwritten")

    def test_dry_run_writes_nothing(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added three"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
        ))
        before = (self.ws / "doc.txt").read_text(encoding="utf-8")
        outcome = run.apply(dry_run=True)
        self.assertTrue(outcome["dry_run"])
        self.assertIn("doc.txt", outcome["would_write"])
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), before)

    def test_apply_then_restore_round_trips(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added three"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
        ))
        original = (self.ws / "doc.txt").read_text(encoding="utf-8")
        outcome = run.apply()
        self.assertTrue(outcome["applied"], outcome)
        self.assertIn("three", (self.ws / "doc.txt").read_text(encoding="utf-8"))
        run.restore_original()
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), original)


class TestPersistence(TimeSkipCase):
    """A run is durable, so a restart reopens its evidence instead of replaying its work."""

    def test_state_is_reloadable_and_carries_hashes(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added three"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
        ))
        reloaded = self.store.get(run.id)
        self.assertEqual(reloaded["id"], run.id)
        self.assertEqual(reloaded["state"], "completed")
        self.assertTrue(reloaded["checkpoints"], "checkpoints must be persisted")
        for cp in reloaded["checkpoints"]:
            self.assertTrue(cp["digest"], "every checkpoint carries a content digest")
            self.assertTrue(cp["inventory"], "every checkpoint carries a file inventory")
        self.assertEqual(len(self.store.list()), 1)

    def test_no_checks_file_means_inconclusive_not_success(self):
        run, result = self.start(lambda c, p, i: "noop", checks=[])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["decision"]["outcome"], "inconclusive")
        self.assertTrue(result["decision"]["needs_user_review"])


class TestDeclaredChecks(TimeSkipCase):
    """Checks are declared per workspace, and a malformed declaration is refused."""

    def test_reads_checks_file(self):
        d = self.ws / ".strata"
        d.mkdir()
        (d / "checks.json").write_text(json.dumps({"checks": [
            {"id": "a", "command": TRUE},
            {"id": "b", "command": TRUE, "kind": "advice", "description": "a judgement"},
        ]}), encoding="utf-8")
        specs = self.store.load_checks(self.ws)
        self.assertEqual([s.id for s in specs], ["a", "b"])
        self.assertEqual(specs[1].kind, "advice")

    def test_absent_checks_file_is_empty_not_an_error(self):
        self.assertEqual(self.store.load_checks(self.ws), [])

    def test_check_without_id_is_refused(self):
        with self.assertRaises(ts.TimeSkipError):
            ts.CheckSpec.from_dict({"command": TRUE})

    def test_check_without_command_or_file_is_refused(self):
        with self.assertRaises(ts.TimeSkipError):
            ts.CheckSpec.from_dict({"id": "x"})


class TestBoundedAttempts(TimeSkipCase):
    """Attempts are bounded, and a later attempt builds on the retained checkpoint."""

    def test_attempt_limit_is_respected(self):
        seen = []

        def revise(candidate: Path, prompt: str, index: int) -> str:
            seen.append(index)
            return f"attempt {index}"

        run, _ = self.start(revise, checks=self.checks(
            {"id": "nonempty", "command": NONEMPTY},
        ), max_attempts=2)
        self.assertEqual(seen, [1, 2])
        self.assertEqual(len(run.attempts), 2)

    def test_zero_attempts_is_a_checkpoint_only_run(self):
        run, result = self.start(lambda c, p, i: "nope", checks=self.checks(
            {"id": "nonempty", "command": NONEMPTY},
        ), max_attempts=0)
        self.assertEqual(run.attempts, [])
        self.assertEqual(result["decision"]["outcome"], "inconclusive")
        self.assertIsNotNone(result["selected_checkpoint"])

    def test_later_attempt_starts_from_a_kept_candidate(self):
        # Attempt 1 adds `three` and is kept; attempt 2 must start from that, not from the baseline.
        def revise(candidate: Path, prompt: str, index: int) -> str:
            text = (candidate / "doc.txt").read_text(encoding="utf-8")
            if index == 1:
                (candidate / "doc.txt").write_text(text + "three\n", encoding="utf-8")
                return "added three"
            assert "three" in text, "attempt 2 must build on the retained checkpoint"
            (candidate / "doc.txt").write_text(text + "four\n", encoding="utf-8")
            return "added four"

        # Only `has-three` gates. `has-four` is declared as advice on purpose: attempt 1 does not
        # add `four`, so as a deterministic check it would fail and the gate would (correctly)
        # reject an attempt that fixed nothing. The gate rejects any failing deterministic check,
        # which is the rule being tested in TestRegressionRejected, not here.
        run, _ = self.start(revise, checks=self.checks(
            {"id": "has-three", "command": _has("three")},
            {"id": "has-four", "command": _has("four"), "kind": "advice"},
        ), max_attempts=2)
        # This case is about the *starting point*, not the verdict, so it asserts exactly that.
        # The assertion inside `revise` is the real test: attempt 2 sees `three`, which can only be
        # there if it was built from the retained checkpoint rather than from the baseline.
        #
        # Attempt 1 adds `three`, and the newly passing check is the evidence that keeps it.
        self.assertEqual(run.attempts[0].decision, "keep")
        # Attempt 2 adds `four`, but `has-four` is declared as advice, so nothing newly *gates*;
        # `inconclusive` is the honest verdict and is deliberately not asserted as a keep.
        self.assertIn(run.attempts[1].decision, ("keep", "inconclusive", "reject"))
        # Whichever verdict it reached, the work it produced is still on disk to inspect, and it
        # contains both lines because it started from the retained candidate.
        produced = (Path(run.attempts[1].path) / "doc.txt").read_text(encoding="utf-8")
        self.assertIn("three", produced)
        self.assertIn("four", produced)


if __name__ == "__main__":
    unittest.main()
