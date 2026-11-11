"""0.2.3 RC stress test: X01 (a failed engine test start gives a plain cause and one next step, not the engine's words) and #18 (the rollback text
never says "0.1.45.3 -> 0.1.45.3")."""

import json
import unittest

from strata_control import updater
from tests.terminal import test_updater
from tests.terminal import update_fixture as fx


class TestStartText(unittest.TestCase):
    RAW = "it did not answer a one-token probe (server_exited) | CUDA error: out of memory at gr.cu:123 | stack"

    def test_the_engines_words_stay_out_of_the_sentence(self):
        err = updater.TestStartFailed("0.1.48", "it did not answer a one-token probe (server_exited)")
        text = str(err)
        self.assertIn("did not start correctly on this computer", text)
        self.assertIn("so it was not installed", text)
        self.assertIn("Your current engine is unchanged", text)
        self.assertIn("The engine started but stopped before it answered a test question.", text)
        self.assertIn("`strata update --force`", text)
        self.assertNotIn("server_exited", text)
        self.assertNotIn("probe", text)
        self.assertEqual(err.reason, "it did not answer a one-token probe (server_exited)")      # kept for the state file and the JSON
        self.assertEqual(err.exit_code, 6)

    def test_a_busy_card_says_so_and_asks_for_a_plain_retry(self):
        err = updater.TestStartFailed("0.1.48", "CUDA error: out of memory")
        self.assertTrue(err.busy)
        self.assertIn("graphics card or memory looks busy", str(err))
        self.assertNotIn("CUDA", str(err))
        self.assertNotIn("--force", str(err))

    def test_an_engine_that_would_not_run_and_a_missing_model(self):
        self.assertIn("The engine program did not run on this computer.", str(updater.TestStartFailed("0.1.48", "--version exited 3: dll missing")))
        self.assertNotIn("dll missing", str(updater.TestStartFailed("0.1.48", "--version exited 3: dll missing")))
        self.assertIn("no installed model", str(updater.TestStartFailed("0.1.48", "no installed model to test with")).lower())

    def test_control_is_called_control(self):
        text = str(updater.TestStartFailed("Control 0.1.48", "--version said 'oops' (exit 7)"))
        self.assertTrue(text.startswith("The new Control 0.1.48 did not start correctly"), text)
        self.assertIn("Your current Control is unchanged.", text)
        self.assertNotIn("engine", text)


class RollbackText(test_updater.UpdaterCase):
    def test_from_names_the_version_that_is_left(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        out = updater.rollback_engine(self.ctx())
        self.assertEqual((out["from"], out["to"]), ("0.1.46", "0.1.45"))
        self.assertNotEqual(out["from"], out["to"])

    def test_a_second_rollback_says_there_is_nothing_older_instead_of_the_same_version_twice(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        updater.rollback_engine(self.ctx())
        state = self.components / ".updates" / "state.json"
        data = json.loads(state.read_text(encoding="utf-8"))
        data["engine"]["previous"] = data["engine"]["pinned"]          # the stale state a second run can meet
        state.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(updater.UpdateError, "already the previous one .0.1.45.*strata update --force"):
            updater.rollback_engine(self.ctx())


if __name__ == "__main__":
    unittest.main()
