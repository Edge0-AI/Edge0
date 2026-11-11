"""0.2.3 RC stress test: #6 (A07/B03) the download size counts the draft layer and the picture reader, B03 `--plan` does not call an installed
engine missing, B10/E04 `--drafting off` leaves `--mtp <path>` out of the written config."""

import unittest
from unittest import mock

from strata_cli.launch import modelstate, settings as S
from strata_cli.launch.runner import Runner, derive_config
from strata_control import engine_component
from tests.terminal.launch_support import HW_12, Home, catalog


class DownloadSize(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def runner(self, **kw):
        values = dict(family="gyro", model="Gyro-S", context=32768, vision="cpu")
        values.update(kw)
        return Runner(S.LaunchSettings(**values), catalog=catalog(HW_12))

    def test_gyro_s_counts_the_model_the_draft_layer_and_the_picture_reader(self):
        runner = self.runner()
        state = runner.model_state()
        self.assertEqual([p.kind for p in state.parts], ["weights", "mtp", "mmproj"])
        self.assertAlmostEqual(state.remaining_gb, 58.5 + 5.2 + 0.9, places=1)
        title, text = runner.download_question()
        self.assertEqual(title, "Download about 65 GB?")
        plan = runner.download_plan()
        self.assertEqual([p["kind"] for p in plan["parts"]], ["weights", "mtp", "mmproj"])
        self.assertAlmostEqual(plan["gb"], 64.6, places=1)
        self.assertIn("the draft layer (5.2 GB)", plan["text"])

    def test_what_the_person_turns_off_is_not_counted(self):
        self.assertEqual([p.kind for p in self.runner(vision="off").model_state().parts], ["weights", "mtp"])
        self.assertEqual([p.kind for p in self.runner(mtp="off").model_state().parts], ["weights", "mmproj"])
        self.assertEqual([p.kind for p in self.runner(drafting="off").model_state().parts], ["weights", "mmproj"])
        self.assertAlmostEqual(self.runner(vision="off", mtp="off").model_state().remaining_gb, 58.5, places=1)

    def test_the_other_families_that_get_helpers_count_the_same(self):
        for family, model in (("qwen", "Q2_0"), ("swift", "IQ2_XS"), ("coder", "IQ3_XXS")):
            kinds = [p.kind for p in self.runner(family=family, model=model).model_state().parts]
            self.assertEqual(kinds, ["weights", "mtp", "mmproj"], family)

    def test_setup_is_told_the_draft_layer_is_off_for_drafting_off(self):
        args = self.runner(drafting="off").setup_args()
        self.assertEqual(args[args.index("--mtp") + 1], "off")
        self.assertTrue(modelstate.HELPER_FAMILIES.count("gyro"))


class PlanEngine(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def plan(self, report):
        runner = Runner(S.LaunchSettings(family="gyro", model="Gyro-S", context=32768, vision="off"), catalog=catalog(HW_12))
        with mock.patch.object(engine_component, "status", return_value=report):
            return runner.download_plan()

    def test_an_engine_that_is_installed_but_unusable_is_not_called_missing(self):
        out = self.plan({"usable": False, "installed": [{"version": "0.1.48"}], "notice": "x"})
        self.assertIs(out["engine_missing"], False)
        self.assertIs(out["engine_usable"], False)
        self.assertIn("An engine (0.1.48) is installed", out["text"])
        self.assertNotIn("downloaded first", out["text"])

    def test_no_engine_at_all_is_missing_and_a_usable_one_is_neither(self):
        out = self.plan({"usable": False, "installed": []})
        self.assertIs(out["engine_missing"], True)
        self.assertIn("downloaded first", out["text"])
        ok = self.plan({"usable": True, "installed": [{"version": "0.1.48"}]})
        self.assertEqual((ok["engine_missing"], ok["engine_usable"]), (False, True))


class DraftingOffConfig(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "port": 8080, "model_name": "m", "env": {},
            "args": ["--native", "x.gguf", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt"]}

    def test_drafting_off_writes_no_mtp_path(self):
        cfg = derive_config(self.BASE, S.LaunchSettings(drafting="off"), bool_flags=frozenset())
        self.assertIs(cfg["mtp"], False)
        self.assertEqual(cfg["drafting"], "off")
        for flag in ("--mtp", "--spec-min-p"):
            self.assertNotIn(flag, cfg["args"])
        self.assertIn("--mtp", self.BASE["args"])

    def test_auto_and_mtp_keep_the_draft_layer(self):
        for mode in ("auto", "mtp", None):
            cfg = derive_config(self.BASE, S.LaunchSettings(drafting=mode), bool_flags=frozenset())
            self.assertIn("--mtp", cfg["args"], mode)

    def test_an_installed_config_that_already_says_off_stays_off(self):
        cfg = derive_config({**self.BASE, "drafting": "off"}, S.LaunchSettings(), bool_flags=frozenset())
        self.assertNotIn("--mtp", cfg["args"])


if __name__ == "__main__":
    unittest.main()
