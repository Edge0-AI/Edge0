"""`strata launch` settles the engine before it touches the model. With no usable engine and none to get it does NOT refuse (owner decision,
2026-10-18): it warns in plain words (the reason from `strata update`, and the next step), asks whether to download the model anyway (the
answer is yes unless the person says no; --yes skips the question), downloads, and ends with a message that names the engine problem and the
next step, exit code 0. Nothing is started without an engine."""

from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import LaunchError, NO_ENGINE_STEP, Runner, no_engine_final, no_engine_warning
from strata_control import engine_component as ec

ROOT = Path(__file__).resolve().parents[2]

WRONG_FAMILY = ec.arch_notice(("gfx1100", "gfx1101", "gfx1102"), ["gfx1201"])
USABLE = {"usable": True, "engine": {"version": "0.1.45", "backend": "hip"}, "notice": None}
UNUSABLE = {"usable": False, "engine": None, "notice": "No engine for this PC was found in /x: the component folder holds no engine package. "
                                                       "The app never downloads an engine while it starts or serves."}


def qwen_catalog():
    """A catalog stand-in for a Qwen model: since 0.2.4 the runner asks it whether the family is DeepSeek (which has its own engine and skips this step);
    a bare Mock answers yes to everything."""
    cat = mock.Mock()
    cat.strata.multi_family.return_value = False
    return cat


class FakeProc:
    def __init__(self, events: list[dict], code: int = 0):
        self.stdout = iter(json.dumps(e) + "\n" for e in events)
        self._code = code

    def wait(self) -> int:
        return self._code


def update_events(status: str, note: str = "", code: int = 0) -> list[dict]:
    res = {"component": "engine", "status": status, "changed": status == "updated", "installed_before": None}
    if note:
        res["note"] = note
    return [{"event": "phase", "phase": "feed", "message": "Reading the signed update feed"},
            {"event": "result", "results": [res], "exit_code": code}]


def runner(spawned: list) -> Runner:
    r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=qwen_catalog(), emit=lambda e: None)
    r.check_cancel = lambda: None
    return r


class EngineBeforeModel(unittest.TestCase):
    def run_ensure(self, statuses: list[dict], events: list[dict], code: int = 0):
        """ensure_engine() with a fake `strata update` and a sequence of engine reports. Returns (spawn count, the error or None)."""
        spawns, err, _r = self.run_ensure_runner(statuses, events, code)
        return spawns, err

    def run_ensure_runner(self, statuses: list[dict], events: list[dict], code: int = 0, confirm=None, gb: float = 0.0,
                          assume_yes: bool = False):
        """ensure_engine() like run_ensure, returning (spawns, error or None, the runner with .events)."""
        spawns: list = []
        seen: list = []
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=qwen_catalog(), emit=seen.append,
                   confirm=confirm, assume_yes=assume_yes)
        r.events = seen
        r.check_cancel = lambda: None
        err = None
        with mock.patch.object(ec, "status", side_effect=statuses), \
                mock.patch.object(Runner, "_spawn", side_effect=lambda argv, **kw: spawns.append(argv) or FakeProc(events, code)), \
                mock.patch.object(Runner, "download_gb", return_value=gb):
            try:
                r.ensure_engine()
            except LaunchError as e:
                err = e
        return spawns, err, r

    def test_engine_present_runs_nothing(self):
        spawns, err = self.run_ensure([USABLE], [])
        self.assertIsNone(err)
        self.assertEqual(spawns, [])

    def test_no_engine_and_the_feed_has_one(self):
        spawns, err = self.run_ensure([UNUSABLE, USABLE], update_events("updated"))
        self.assertIsNone(err)
        self.assertEqual(len(spawns), 1)
        self.assertIn("update", spawns[0])

    def test_no_engine_and_the_feed_has_none_is_a_warning_not_a_refusal(self):
        note = "The feed has no engine package for this PC."
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", note))
        self.assertIsNone(err)
        self.assertEqual(r.engine_problem, note)
        warning = no_engine_warning(note)
        self.assertEqual(r.events[-1].text, warning)
        self.assertEqual(r.events[-1].level, "warn")
        self.assertIn("Strata couldn't find an engine for this computer (The feed has no engine package for this PC)", warning)
        self.assertIn("The model will download, but it may not run until you install the right engine (next step:", warning)
        self.assertIn(NO_ENGINE_STEP, warning)
        self.assertNotIn("The app never downloads", warning)          # not the generic line

    def test_no_graphics_card_is_said_plainly_with_no_feed_address(self):
        report = dict(UNUSABLE, backend=None)
        spawns, err, r = self.run_ensure_runner([report, report], update_events("none-for-this-pc"))
        self.assertIsNone(err)
        self.assertTrue(r.engine_problem.startswith("no graphics card that Strata can use was found"))
        text = r.events[-1].text
        self.assertIn("Strata needs an NVIDIA or AMD graphics card", text)
        self.assertNotIn("--feed", text)
        self.assertNotIn("<address>", text)
        self.assertNotIn("--feed", no_engine_final(r.engine_problem))

    def test_the_other_no_engine_texts_do_not_ask_for_a_feed_address_either(self):
        self.assertNotIn("<address>", NO_ENGINE_STEP)
        self.assertNotIn("--feed", no_engine_warning("The feed has no engine package for this PC."))

    def test_wrong_family_names_the_plain_reason_in_the_warning(self):
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", WRONG_FAMILY))
        self.assertIsNone(err)
        self.assertEqual(r.engine_problem, WRONG_FAMILY)
        text = r.events[-1].text
        self.assertIn("This build of Strata is for AMD RX 7000 cards. This computer has an AMD RX 9000 card (gfx1201).", text)
        self.assertIn("Ask for the RX 9000 build)", text)

    def test_installed_wrong_family_package_uses_its_plain_notice(self):
        report = {"usable": False, "engine": None, "notice": WRONG_FAMILY}
        spawns, err, r = self.run_ensure_runner([report, report], update_events("none-for-this-pc"))
        self.assertEqual(r.engine_problem, WRONG_FAMILY)

    def test_the_question_is_asked_with_the_warning_and_no_stops_before_the_download(self):
        asked = []
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", "No engine here."),
                                                confirm=lambda title, text: asked.append((title, text)) or False, gb=66.0)
        self.assertEqual(len(asked), 1)
        self.assertIn("Strata couldn't find an engine for this computer (No engine here)", asked[0][1])
        self.assertIn("Download the model now (about 66 GB)?", asked[0][1])
        self.assertIsInstance(err, LaunchError)
        self.assertEqual(err.code, 130)
        self.assertIn("Nothing was downloaded", err.text)

    def test_yes_goes_on_and_the_yes_flag_asks_nothing(self):
        asked = []
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", "No engine here."),
                                                confirm=lambda t, x: asked.append(1) or True, gb=66.0)
        self.assertIsNone(err)
        self.assertEqual(len(asked), 1)
        asked.clear()
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", "No engine here."),
                                                confirm=lambda t, x: asked.append(1) or False, gb=66.0, assume_yes=True)
        self.assertIsNone(err)
        self.assertEqual(asked, [])
        self.assertEqual(r.engine_problem, "No engine here.")

    def test_nothing_to_download_asks_nothing(self):
        asked = []
        spawns, err, r = self.run_ensure_runner([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", "No engine here."),
                                                confirm=lambda t, x: asked.append(1) or False, gb=0.0)
        self.assertIsNone(err)
        self.assertEqual(asked, [])

    def test_the_model_is_downloaded_and_nothing_is_started_without_an_engine(self):
        r = runner([])
        started = []
        with mock.patch.object(ec, "status", return_value=UNUSABLE), \
                mock.patch.object(Runner, "_spawn", return_value=FakeProc(update_events("none-for-this-pc", WRONG_FAMILY))), \
                mock.patch.object(Runner, "ensure_model", return_value=Path("/x/strata-q.json")) as model, \
                mock.patch.object(Runner, "build_config", return_value=Path("/x/launch.json")), \
                mock.patch.object(Runner, "ensure_profile"), mock.patch.object(Runner, "tune") as tune, \
                mock.patch.object(Runner, "start", side_effect=lambda: started.append(1)):
            result = r.run()
        model.assert_called_once()
        tune.assert_not_called()
        self.assertEqual(started, [])
        self.assertFalse(result.serving)
        self.assertEqual(result.urls, [])
        self.assertEqual(result.engine_problem, WRONG_FAMILY)
        self.assertEqual(result.final_text, no_engine_final(WRONG_FAMILY))
        self.assertIn("The model is ready, but Strata can't start it yet", result.final_text)
        self.assertIn("Ask for the RX 9000 build)", result.final_text)
        self.assertIn(NO_ENGINE_STEP, result.final_text)

    def test_the_setup_child_is_told_to_go_on_without_an_engine(self):
        r = runner([])
        r.engine_problem = "No engine here."
        seen = {}

        class Boom(Exception):
            pass

        def spawn(self, argv, env=None, cwd=None):
            seen["env"] = env
            raise Boom()

        cat = mock.MagicMock()
        cat.src = ROOT
        r._catalog = cat
        r.s = S.LaunchSettings(family="qwen", model="Q2_0", context=32768)
        with mock.patch.object(Runner, "installed_state", return_value=(None, None)), mock.patch.object(Runner, "_spawn", spawn):
            with self.assertRaises(Boom):
                r.ensure_model()
        self.assertEqual(seen["env"].get("STRATA_ALLOW_NO_ENGINE"), "1")
        self.assertNotIn("STRATA_NEED_ENGINE", seen["env"])

    def test_a_failed_update_still_reports_the_old_message(self):
        spawns, err = self.run_ensure([UNUSABLE], [{"event": "error", "error": {"message": "cannot reach the feed"}}], code=6)
        self.assertEqual(err.code, 6)
        self.assertIn("Cannot reach the feed", err.text)      # (the reason is capitalised where it joins the sentence)


class SetupControlStep(unittest.TestCase):
    """setup.py in the app's mode (and under `strata launch`): the engine step comes before the download step."""

    def load_setup(self):
        """A private copy of setup.py read in the app's mode; the shared `setup` module (other tests) is left as it is."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_control_copy", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1"}):
            spec.loader.exec_module(module)
        return module

    def step(self, exc: Exception, env: dict) -> tuple[object, str]:
        setup = self.load_setup()
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), mock.patch.object(ec, "resolve", side_effect=exc), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("llama")), \
                mock.patch.object(setup, "say", lambda *a, **k: out.write(" ".join(map(str, a)) + "\n")), \
                mock.patch.object(setup, "warn", lambda *a, **k: out.write("WARN " + " ".join(map(str, a)) + "\n")), \
                mock.patch.object(setup, "ok", lambda *a, **k: None):
            try:
                return setup.control_engine_step({}, "none"), out.getvalue()
            except SystemExit as e:
                return e, out.getvalue()

    def test_wrong_family_stops_before_the_download_for_launch(self):
        exc = ec.EngineMissing("nothing matched", Path("/c"), notice=WRONG_FAMILY)
        res, out = self.step(exc, {"STRATA_NEED_ENGINE": "1"})
        self.assertIsInstance(res, SystemExit)
        self.assertIn("Ask for the RX 9000 build.", out)
        self.assertIn("nothing was downloaded", out)

    def test_wrong_family_goes_on_with_a_warning_when_launch_said_go_on(self):
        exc = ec.EngineMissing("nothing matched", Path("/c"), notice=WRONG_FAMILY)
        res, out = self.step(exc, {"STRATA_ALLOW_NO_ENGINE": "1"})
        self.assertNotIsInstance(res, SystemExit)
        self.assertIn("no engine is installed for this PC yet", out)
        self.assertIn("The model is downloaded and prepared now", out)

    def test_no_engine_for_launch_still_stops_setup_when_run_on_its_own(self):
        res, out = self.step(ec.EngineMissing("the component folder holds no engine package", Path("/c")), {"STRATA_NEED_ENGINE": "1"})
        self.assertIsInstance(res, SystemExit)

    def test_the_app_with_a_wrong_family_engine_warns_and_still_downloads(self):
        """Owner decision 2026-10-18: the app downloads the model anyway and warns (its card says so)."""
        env = {k: v for k, v in os.environ.items() if k != "STRATA_NEED_ENGINE"}
        exc = ec.EngineMissing("nothing matched", Path("/c"), notice=WRONG_FAMILY)
        with mock.patch.dict(os.environ, env, clear=True):
            res, out = self.step(exc, {})
        self.assertNotIsInstance(res, SystemExit)
        self.assertIn("Ask for the RX 9000 build.", out)
        self.assertIn("The model is downloaded and prepared now", out)

    def test_no_engine_stops_before_the_download_for_launch(self):
        res, out = self.step(ec.EngineMissing("the component folder holds no engine package", Path("/c")), {"STRATA_NEED_ENGINE": "1"})
        self.assertIsInstance(res, SystemExit)

    def test_the_app_without_an_engine_keeps_its_order(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_NEED_ENGINE"}
        with mock.patch.dict(os.environ, env, clear=True):
            res, out = self.step(ec.EngineMissing("the component folder holds no engine package", Path("/c")), {})
        self.assertNotIsInstance(res, SystemExit)
        self.assertIn("The model is downloaded and prepared now", out)


if __name__ == "__main__":
    unittest.main()
