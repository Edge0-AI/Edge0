"""The engine options a person set by hand survive setup, the app and every start (#3216 control vectors, #3215 and #3270
--vram-reserve-mib).  strata_control/user_args.py is the rule; setup.carry_over, serve/server.py engine_args, serve/runconfig.py
and strata_control/tune_apply.py use it.  Every outside effect is mocked (tools/test_setup_golden.py's install()).

    python -m unittest tools.test_user_args
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp  # noqa: E402
from serve import runconfig  # noqa: E402
from serve.server import engine_args  # noqa: E402
from strata_control import tune_apply, user_args  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

CVEC = ["--control-vector-scaled", "C:/vectors/mine.gguf:1.0", "--control-vector-layer-range", "4", "44",
        "--cvec-mode", "project", "--cvec-dir", "per-layer"]


def value(args, flag):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


class Rule(unittest.TestCase):
    def test_groups_and_merge(self):
        self.assertEqual(user_args.groups(["--a", "1", "--b", "--c", "x", "y"]), [["--a", "1"], ["--b"], ["--c", "x", "y"]])
        self.assertEqual(user_args.merge(["--a", "1", "--b", "2", "--a", "3"], ["--a", "9", "--z"]), ["--b", "2", "--a", "9", "--z"])
        self.assertEqual(user_args.merge(["--a", "1"], []), ["--a", "1"])

    def test_capture_tells_hand_edits_from_what_was_written(self):
        written = ["--max-context", "32768", "--vram-reserve-mib", "700", "--spec", "4"]
        cfg = {"args": written + CVEC, "args_written": written}
        self.assertEqual(user_args.capture(cfg), CVEC)
        cfg = {"args": ["--max-context", "32768", "--vram-reserve-mib", "0", "--spec", "4"], "args_written": written}
        self.assertEqual(user_args.capture(cfg), ["--vram-reserve-mib", "0"])
        self.assertEqual(user_args.capture({**cfg, "args": written}), [])                  # nothing changed
        # the context is setup's own choice: a hand edit of it is not taken as an override
        self.assertEqual(user_args.capture({"args": ["--max-context", "1"], "args_written": ["--max-context", "2"]},
                                           setup.SETUP_CHOICE_FLAGS), [])
        # an option taken out of "args" by hand is the person's no more
        self.assertEqual(user_args.capture({"args": written[:4], "args_written": written, "user_args": ["--spec", "2"]}), [])
        self.assertEqual(user_args.capture({"args": written, "args_written": written, "user_args": ["--spec", "2"]}), ["--spec", "2"])   # user_args stands
        # no copy of what was written: only user_args is known
        self.assertEqual(user_args.capture({"args": CVEC, "user_args": ["--a", "1"]}), ["--a", "1"])


class SetupRerun(unittest.TestCase):
    """The app's Load runs setup again whenever the context, KV type or pictures differ from the installed ones."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, *extra, configs=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra], configs=configs)

    def first(self):
        code, out, first, _ = self.run_setup("--context", "65536")
        self.assertEqual(code, 0, out)
        return first

    def test_3216_control_vector_arguments_survive_a_load(self):
        first = self.first()
        old = {**first, "args": first["args"] + CVEC, "args_written": first["args"]}      # typed into the JSON editor
        code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--control-vector-scaled"):][:9], CVEC)
        self.assertEqual(value(a, "--max-context"), "131072")                              # setup's own choice still applies
        self.assertEqual(cfg["user_args"], CVEC)
        # the second Load: the options are in user_args, so they stay
        code, out, again, _ = self.run_setup("--context", "65536", configs=[("strata-q2_0.json", cfg)])
        self.assertEqual(again["user_args"], CVEC)
        self.assertEqual(value(again["args"], "--cvec-mode"), "project")

    def test_3216_a_config_without_args_written_keeps_them_too(self):
        first = self.first()
        old = {**first, "args": first["args"] + CVEC}                                      # 0.2.0-beta: no copy of what was written
        code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["user_args"], CVEC)
        self.assertEqual(cfg["args"][-len(CVEC):], CVEC)

    def test_3215_a_reserve_of_zero_is_kept(self):
        first = self.first()
        written = list(first["args"])
        edited = [*written]
        edited[edited.index("--vram-reserve-mib") + 1] = "0"
        for old in ({**first, "args": edited, "args_written": written},                    # with the copy of what was written
                    {**first, "args": edited}):                                           # a 0.2.0-beta file
            code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", old)])
            self.assertEqual(code, 0, out)
            self.assertEqual(value(cfg["args"], "--vram-reserve-mib"), "0")
            self.assertEqual(cfg["user_args"], ["--vram-reserve-mib", "0"])

    def test_3270_a_lowered_reserve_is_not_reset(self):
        first = self.first()
        written = list(first["args"])
        for mib in ("1024", "256"):
            edited = [*written]
            edited[edited.index("--vram-reserve-mib") + 1] = mib
            code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", {**first, "args": edited, "args_written": written})])
            self.assertEqual(value(cfg["args"], "--vram-reserve-mib"), mib)

    def test_the_vision_default_is_not_taken_for_a_choice(self):
        first = self.first()                                                               # pictures on the GPU write 700 themselves
        self.assertEqual(value(first["args"], "--vram-reserve-mib"), "700")
        code, out, cfg, _ = self.run_setup("--context", "131072", "--vision", "cpu", configs=[("strata-q2_0.json", first)])
        self.assertNotIn("user_args", cfg)
        self.assertNotIn("--vram-reserve-mib", cfg["args"])                                # the CPU reader needs no reserve

    def test_an_option_given_on_the_command_line_wins_and_is_kept(self):
        first = self.first()
        old = {**first, "args": first["args"] + ["--vram-reserve-mib", "0"], "user_args": ["--vram-reserve-mib", "0"]}
        code, out, cfg, _ = self.run_setup("--context", "131072", "--vram-reserve-mib", "256", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["user_args"], ["--vram-reserve-mib", "256"])
        self.assertEqual(value(cfg["args"], "--vram-reserve-mib"), "256")

    def test_a_first_install_with_a_reserve_records_it(self):
        code, out, cfg, _ = self.run_setup("--vram-reserve-mib", "300")
        self.assertEqual(cfg["user_args"], ["--vram-reserve-mib", "300"])

    def test_setup_writes_args_written_and_a_rerun_with_no_edit_makes_no_user_args(self):
        first = self.first()
        code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", {**first, "args_written": first["args"]})])
        self.assertNotIn("user_args", cfg)


class Start(unittest.TestCase):
    def test_every_start_puts_the_persons_options_over_the_generated_ones(self):
        cfg = {"args": ["--pack", "/p", "--vram-reserve-mib", "1536", "--spec", "4"],
               "user_args": ["--vram-reserve-mib", "0"] + CVEC}
        with contextlib.redirect_stdout(io.StringIO()):
            args = engine_args(cfg)
        self.assertEqual(value(args, "--vram-reserve-mib"), "0")
        self.assertEqual(args.count("--vram-reserve-mib"), 1)
        self.assertEqual(args[args.index("--control-vector-scaled"):][:9], CVEC)
        self.assertEqual(cfg["args"][3], "1536")                                           # the config in memory is untouched

    def test_the_settings_page_records_the_reserve_as_the_persons(self):
        cfg = {"args": ["--pack", "/p", "--vram-reserve-mib", "1536"], "args_written": ["--pack", "/p", "--vram-reserve-mib", "1536"]}
        new, changed = runconfig.apply(cfg, {"vram_reserve_mib": 0})
        self.assertEqual(changed, ["vram_reserve_mib"])
        self.assertEqual(new["user_args"], ["--vram-reserve-mib", "0"])
        self.assertEqual(new["args_written"], new["args"])
        back, _ = runconfig.apply(new, {"vram_reserve_mib": None})
        self.assertNotIn("user_args", back)
        self.assertNotIn("--vram-reserve-mib", back["args"])

    def test_a_tuning_replaces_only_the_flags_it_sets(self):
        cfg = {"args": ["--pack", "/p", "--vram-reserve-mib", "0", *CVEC], "args_written": ["--pack", "/p"],
               "user_args": ["--vram-reserve-mib", "0", *CVEC]}
        out = tune_apply.apply_args_to_config(cfg, ["--vram-reserve-mib", "512", "--pcie-frac", "0.5"])
        self.assertEqual(value(out["args"], "--vram-reserve-mib"), "512")
        self.assertEqual(out["user_args"], CVEC)                                           # the tuned flag is no longer the old pin
        self.assertEqual(out["args_written"], out["args"])


class DisplayCards(unittest.TestCase):
    """The app keeps the safety reserve for a card that drives a display and less on one that does not (#3215)."""

    def test_nvidia_by_nvidia_smi(self):
        gpus = [{"vendor": "nvidia", "index": 0}, {"vendor": "nvidia", "index": 1}, {"vendor": "nvidia", "index": 2}]
        strata_mcp.mark_display(gpus, {0: False, 1: True}, [])
        self.assertIs(gpus[0]["drives_display"], False)
        self.assertIs(gpus[1]["drives_display"], True)
        self.assertNotIn("drives_display", gpus[2])                                        # not told: unknown

    def test_amd_by_the_windows_adapter_with_a_desktop(self):
        amd = {"vendor": "amd", "index": 1, "pci_id": 0x744C}
        gpus = [{"vendor": "amd", "index": 0, "pci_id": 0x164E}, amd]
        strata_mcp.mark_display(gpus, {}, ["PCI\\VEN_1002&DEV_164E&SUBSYS_1"])
        self.assertIs(gpus[0]["drives_display"], True)
        self.assertIs(amd["drives_display"], False)
        twins = [{"vendor": "amd", "pci_id": 0x744C}, {"vendor": "amd", "pci_id": 0x744C}]
        strata_mcp.mark_display(twins, {}, ["PCI\\VEN_1002&DEV_744C"])
        self.assertNotIn("drives_display", twins[0])                                       # two of one model: not told apart
        strata_mcp.mark_display(gpus[:1] + [{"vendor": "amd", "pci_id": 0x1}], {}, [])
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
