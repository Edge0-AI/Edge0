"""Tests for what setup keeps in a model's run config when it runs again (#629), the browser choice (#609) and the
image-token flag (#625).  Every outside effect is mocked (tools/test_setup_golden.py's install()).

    python -m unittest tools.test_setup_config
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


USER = {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "mcp_servers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}},
        "mcp": {"timeout_s": 60, "max_result_chars": 16000, "max_rounds": 8},
        "cors_origins": ["http://localhost:3000"], "reasoning_effort": "low"}


class CarryOver(unittest.TestCase):
    """#629: setup run again (a different --context) keeps the keys it does not write and keeps the old file as .bak."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, *extra, configs=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra],
                       configs=configs)

    def test_user_keys_survive_a_rerun(self):
        code, out, first, _ = self.run_setup("--context", "65536")
        self.assertEqual(code, 0, out)
        old = {**first, **USER, "args": first["args"] + ["--hand-option", "7"]}
        code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        for k, v in USER.items():
            self.assertEqual(cfg[k], v, k)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")       # the new choice is used
        self.assertEqual(a[a.index("--hand-option") + 1], "7")             # an option added by hand is kept (#3216)...
        self.assertEqual(cfg["user_args"], ["--hand-option", "7"])         # ...and recorded as the person's
        self.assertIn("kept from your earlier strata-q2_0.json: sampling, mcp_servers, mcp, cors_origins, "
                      "reasoning_effort", out)
        self.assertIn("kept as strata-q2_0.json.bak", out)
        self.assertIn("args --hand-option", out)                           # ...and the run says it was kept

    def test_a_first_install_is_unchanged(self):
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(code, 0, out)
        self.assertNotIn("kept from", out)
        self.assertNotIn(".bak", out)
        self.assertFalse(set(cfg) - setup.SETUP_KEYS)                     # setup writes only its own keys

    def test_setup_keys_are_rewritten(self):
        code, out, first, _ = self.run_setup()
        old = {**first, "model_name": "renamed", "port": 9999, "draft_vocab": "en", "host": "0.0.0.0"}
        code, out, cfg, _ = self.run_setup(configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], first["model_name"])
        self.assertEqual(cfg["port"], first["port"])
        self.assertNotIn("host", cfg)                                     # as before: --host is given to setup
        self.assertEqual(cfg["draft_vocab"], "en")                        # kept by its own rule (saved_draft_vocab)
        self.assertNotIn("kept from", out)

    def test_write_setup_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-x.json"
            mm = Path(d) / "mmproj-Q8_0.gguf"
            mm.write_bytes(b"")
            old = {"exe": "e", "args": ["--kv", "int8", "--threads", "8"], "sampling": {"temperature": 0.7},
                   "env": {"STRATA_RESIDENT_PIN": "0", "MY_VAR": "1"},
                   "vision": {"exe": "v", "mmproj": str(mm), "gpu": False, "max_tokens": 300, "min_tokens": 64}}
            p.write_text(json.dumps(old), encoding="utf-8")
            new = {"exe": "e2", "args": ["--kv", "int8"],
                   "vision": {"exe": "v2", "mmproj": str(Path(d) / "mmproj-BF16.gguf"), "gpu": False, "max_tokens": 300}}
            _, out = quiet(setup.write_setup_config, p, new)
            got = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(got["sampling"], {"temperature": 0.7})
            self.assertEqual(got["env"], {"MY_VAR": "1"})                 # setup's own env entry is setup's to write
            self.assertEqual(got["vision"]["mmproj"], str(mm))            # a Q8_0 mmproj of the user's own (#625)
            self.assertEqual(got["vision"]["min_tokens"], 64)
            self.assertEqual(got["exe"], "e2")
            self.assertEqual(json.loads((Path(d) / "strata-x.json.bak").read_text(encoding="utf-8")), old)
            self.assertIn("--threads", out)
            # the same again: nothing changes, so no new copy is made
            (Path(d) / "strata-x.json.bak").unlink()
            again = {"exe": "e2", "args": ["--kv", "int8"],
                     "vision": {"exe": "v2", "mmproj": str(Path(d) / "mmproj-BF16.gguf"), "gpu": False,
                                "max_tokens": 300}}
            quiet(setup.write_setup_config, p, again)
            self.assertFalse((Path(d) / "strata-x.json.bak").exists())
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), got)

    def test_an_unreadable_earlier_config_is_replaced_with_a_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-x.json"
            p.write_text("{cut off", encoding="utf-8")
            quiet(setup.write_setup_config, p, {"exe": "e", "args": []})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"exe": "e", "args": [], "args_written": []})
            self.assertEqual((Path(d) / "strata-x.json.bak").read_text(encoding="utf-8"), "{cut off")

    def test_display_tip_and_typed_ahead_flush(self):
        # #779: the tip only when nvidia-smi says the card drives a display; #841: typed-ahead keys are dropped on a tty
        with unittest.mock.patch.object(setup, "out", lambda cmd: "Enabled\n"):
            self.assertTrue(setup.gpu_drives_display({"index": 0}))
        with unittest.mock.patch.object(setup, "out", lambda cmd: "Disabled\n"), unittest.mock.patch.object(setup, "out") as o:
            o.return_value = "Disabled\n"
            self.assertFalse(setup.gpu_drives_display({"index": 1}))
            self.assertIn("1", o.call_args[0][0])
        with unittest.mock.patch.object(setup, "out", lambda cmd: ""):
            self.assertFalse(setup.gpu_drives_display({"index": 0}))
        with unittest.mock.patch.object(setup.sys, "stdin") as stdin:
            stdin.isatty.return_value = False
            setup.flush_typed_ahead()                      # a pipe: nothing touched
            stdin.fileno.assert_not_called()

    def test_a_learned_expert_profile_is_carried(self):
        # #775: --expert-profile / --expert-profile-save added by hand survive a setup run
        with tempfile.TemporaryDirectory() as d:
            mine = Path(d) / "learned.bin"
            mine.write_bytes(b"x")
            old = {"args": ["--kv", "int8", "--expert-profile", str(mine), "--expert-profile-save", str(mine)]}
            new = {"args": ["--kv", "int8", "--expert-profile", "/data/expert-profile.bin", "--expert-cache", "auto"]}
            self.assertEqual(setup.carry_over(old, new), ["args --expert-profile", "args --expert-profile-save"])
            self.assertEqual(setup.flag_value(new["args"], "--expert-profile"), str(mine))
            self.assertEqual(setup.flag_value(new["args"], "--expert-profile-save"), str(mine))
            # the shipped profile named again, or a file that is gone: setup's own stays
            gone = {"args": ["--expert-profile", str(Path(d) / "gone.bin")]}
            new = {"args": ["--expert-profile", "/data/expert-profile.bin"]}
            self.assertEqual(setup.carry_over(gone, new), [])
            self.assertEqual(new["args"], ["--expert-profile", "/data/expert-profile.bin"])
            same = {"args": ["--expert-profile", "/old/expert-profile.bin"]}
            self.assertEqual(setup.carry_over(same, new), [])

    def test_mtp_off_is_the_users_and_survives_a_setup_run_that_writes_the_draft_layer_back(self):
        old = {"args": ["--spec", "4", "--spec-min-p", "0.5"], "mtp": False}
        new = {"args": ["--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt", "--max-context", "32768"]}
        kept = setup.carry_over(old, new)
        self.assertIs(new["mtp"], False)
        self.assertEqual(new["args"], ["--spec", "4", "--max-context", "32768"])
        self.assertTrue(any(k.startswith("mtp off") for k in kept))
        on = {"args": ["--mtp", "/m/rt"]}
        setup.carry_over({"args": [], "mtp": True}, on)
        self.assertEqual(on["args"], ["--mtp", "/m/rt"])                           # on: setup's own arguments stay

    def test_a_missing_mmproj_is_not_carried(self):
        old = {"vision": {"mmproj": "/nowhere/mmproj-Q8_0.gguf", "gpu": True}}
        new = {"vision": {"mmproj": "/data/mmproj-BF16.gguf", "gpu": True}}
        self.assertEqual(setup.carry_over(old, new), [])
        self.assertEqual(new["vision"]["mmproj"], "/data/mmproj-BF16.gguf")

    def test_a_copy_set_up_like_the_last_one_keeps_them_too(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "other" / "strata-x.json"
            src.parent.mkdir()
            src.write_text(json.dumps({"exe": "old", "mcp_servers": {"a": {"command": "x"}}}), encoding="utf-8")
            p = Path(d) / "strata-x.json"
            quiet(setup.write_setup_config, p, {"exe": "new"}, src)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")),
                             {"exe": "new", "mcp_servers": {"a": {"command": "x"}}})
            self.assertFalse((Path(d) / "strata-x.json.bak").exists())   # nothing here was replaced


class NoBrowser(unittest.TestCase):
    """#609 (#631): --no-browser keeps the chat page from opening: "open_browser": false in the run config, no --open
    on the start's server command or in run-<model>.bat/.sh.  Without the flag nothing changes."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, *extra, configs=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra],
                       configs=configs)

    def test_at_setup(self):
        code, out, cfg, _ = self.run_setup("--no-browser")
        self.assertEqual(code, 0, out)
        self.assertIs(cfg["open_browser"], False)
        code, out, cfg, _ = self.run_setup()
        self.assertNotIn("open_browser", cfg)                             # not given: the config as before
        # setup run again without the flag keeps the choice (#629); --browser undoes it
        code, out, cfg, _ = self.run_setup(configs=[("strata-q2_0.json", {"open_browser": False})])
        self.assertIs(cfg["open_browser"], False)
        code, out, cfg, _ = self.run_setup("--browser", configs=[("strata-q2_0.json", {"open_browser": False})])
        self.assertIs(cfg["open_browser"], True)

    def test_run_script(self):
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(setup, "ROOT", Path(d)):
            text = lambda p: p.read_text(encoding="utf-8")   # noqa: E731
            self.assertIn('"--open"', text(setup.write_run_script("Q2_0", Path(d) / "strata-q2_0.json", 8080)))
            script = setup.write_run_script("Q2_0", Path(d) / "strata-q2_0.json", 8080, False)
            self.assertNotIn("--open", text(script))
            self.assertIn('"--port" "8080"', text(script))

    def start_cmd(self, cfg, keep=None):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--kv", "int8"], "gpu": 0, "gpus_asked": True, **cfg}))
            call = unittest.mock.Mock(return_value=0)
            with unittest.mock.patch.object(setup, "gpus", lambda: self.found), \
                    unittest.mock.patch.object(setup, "is_wsl", lambda: False), \
                    unittest.mock.patch.object(setup, "ensure_engine_for", lambda cards, path, c, yes: c), \
                    unittest.mock.patch.object(setup.subprocess, "call", call):
                _, out = quiet(setup.start, p, None, None, True, True, None, keep)
            return call.call_args[0][0], out, json.loads(p.read_text())

    def test_on_a_start(self):
        cmd, out, _ = self.start_cmd({})
        self.assertIn("--open", cmd)                                      # the default, as before
        self.assertIn("the browser opens when it is ready", out)
        cmd, out, _ = self.start_cmd({"open_browser": False})
        self.assertNotIn("--open", cmd)
        self.assertNotIn("the browser opens", out)
        self.assertIn("no browser", out)                                  # the Settings line says so
        cmd, out, saved = self.start_cmd({}, {"open_browser": False})     # START-HERE --no-browser: kept from now on
        self.assertNotIn("--open", cmd)
        self.assertIs(saved["open_browser"], False)
        self.assertIn("saved for this model: no browser", out)
        cmd, out, saved = self.start_cmd({"open_browser": False}, {"open_browser": True})
        self.assertIn("--open", cmd)
        self.assertIs(saved["open_browser"], True)


class VisionTokens(unittest.TestCase):
    """#625: --vision-tokens N sets the config's vision.max_tokens (the most image tokens a picture becomes); a setup
    run again keeps it for the same encoder device; without it the device's default, as before."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, *extra, configs=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra],
                       configs=configs)

    def test_the_flag(self):
        code, out, cfg, _ = self.run_setup("--vision", "cpu")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["max_tokens"], 300)               # the default, as before
        self.assertNotIn("--vision-tokens", out)
        code, out, cfg, _ = self.run_setup("--vision", "cpu", "--vision-tokens", "768")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["vision"]["max_tokens"], 768)
        self.assertIn("up to 768 image tokens", out)
        self.assertIn("longer to encode", out)                            # a note, not a cap
        code, out, cfg, _ = self.run_setup("--vision", "gpu", "--vision-tokens", "2048")
        self.assertEqual(cfg["vision"]["max_tokens"], 2048)
        self.assertIn("more VRAM", out)

    def test_kept_when_setup_runs_again(self):
        code, out, first, _ = self.run_setup("--vision", "cpu", "--vision-tokens", "768")
        old = [("strata-q2_0.json", first)]
        code, out, cfg, _ = self.run_setup("--vision", "cpu", "--context", "65536", configs=old)
        self.assertEqual(cfg["vision"]["max_tokens"], 768)
        code, out, cfg, _ = self.run_setup("--vision", "cpu", "--vision-tokens", "300", configs=old)
        self.assertEqual(cfg["vision"]["max_tokens"], 300)               # given again: the new value
        code, out, cfg, _ = self.run_setup("--vision", "gpu", configs=old)
        self.assertEqual(cfg["vision"]["max_tokens"], 1024)              # another device: its own default

    def test_refused_and_unused(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code, out, cfg, _ = self.run_setup("--vision-tokens", "0")
        self.assertEqual(code, 2)
        self.assertIn("--vision-tokens takes a number", err.getvalue())
        code, out, cfg, _ = self.run_setup("--vision", "no", "--vision-tokens", "768")
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", cfg)
        self.assertIn("images are off", out)


if __name__ == "__main__":
    unittest.main()
