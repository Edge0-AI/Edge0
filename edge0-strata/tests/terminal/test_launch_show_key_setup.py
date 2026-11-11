"""#3570: `strata launch --show-key` printed a key only for a launch (a running one or a preset). A model set up by setup.py keeps its key in
strata-<model>.json, so there the command said "no key" and people read the JSON by hand. It now reads that file when no launch has one."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import cli
from strata_cli.launch import cmd as launch_cmd


class ShowKeyOfSetupConfigs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def config(self, model_id, key="__none__", **extra):
        path = self.root / f"strata-{model_id}.json"
        body = {"exe": "x", "args": [], **extra}
        if key != "__none__":
            body["api_key"] = key
        path.write_text(json.dumps(body), encoding="utf-8")
        return {"id": model_id, "source_config": str(path)}

    def run_cli(self, entries, *argv, launch=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(launch_cmd, "launch_info", return_value=launch), \
                mock.patch("strata_control.models.list_installed", return_value={"models": entries}):
            code = cli.main(["launch", "--show-key", *argv])
        return code, out.getvalue().strip(), err.getvalue()

    def test_the_only_model_with_a_key_is_shown(self):
        code, out, err = self.run_cli([self.config("coder-iq1_m", "abc123-secret"), self.config("qwen-q2_0")])
        self.assertEqual((code, out), (0, "abc123-secret"))
        self.assertIn("coder-iq1_m", err)                      # the hint is on stderr: stdout stays the key alone, for scripts

    def test_a_launch_that_has_a_key_comes_first(self):
        code, out, _ = self.run_cli([self.config("coder-iq1_m", "from-the-file")], launch={"key": "from-the-launch"})
        self.assertEqual((code, out), (0, "from-the-launch"))

    def test_several_models_need_a_name_and_the_name_picks_one_in_any_case(self):
        entries = [self.config("coder-iq1_m", "key-one"), self.config("qwen-iq3_xxs", "key-two")]
        code, out, err = self.run_cli(entries)
        self.assertEqual(code, 5)
        self.assertEqual(out, "")
        self.assertIn("coder-iq1_m", err)
        self.assertIn("--model NAME", err)
        self.assertNotIn("key-one", err + out)                  # no key is printed while the choice is open
        code, out, _ = self.run_cli(entries, "--model", "IQ3_XXS")
        self.assertEqual((code, out), (0, "key-two"))

    def test_no_key_anywhere_and_a_name_that_matches_nothing_are_told_plainly(self):
        code, out, err = self.run_cli([self.config("qwen-q2_0")])
        self.assertEqual((code, out), (5, ""))
        self.assertIn("There is no API key", err)
        code, out, err = self.run_cli([self.config("qwen-q2_0", "none")], "--model", "iq1_m")
        self.assertEqual(code, 5)
        self.assertIn("iq1_m", err)

    def test_a_preset_name_never_falls_back_to_a_setup_config(self):
        code, out, err = self.run_cli([self.config("coder-iq1_m", "abc123-secret")], "--preset", "mine")
        self.assertEqual((code, out), (5, ""))
        self.assertNotIn("abc123", err)


if __name__ == "__main__":
    unittest.main()
