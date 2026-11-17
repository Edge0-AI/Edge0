"""The draft layer switch: "mtp": false in a run config starts the engine without --mtp (serve/server.py engine_args), however the
arguments got there, and strata_control/draft_layer.py is the one rule setup, the launcher and the server share.

    python -m unittest serve.test_draft_layer
"""
import contextlib
import io
import unittest

from serve import runconfig
from serve.server import engine_args
from strata_control import draft_layer

ARGS = ["--pack", "/p", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt", "--max-context", "32768", "--batch-mtp", "--kv", "int8"]


class DraftLayer(unittest.TestCase):
    def test_strip_keeps_everything_but_the_draft_layer(self):
        out, removed = draft_layer.strip(ARGS)
        self.assertEqual(out, ["--pack", "/p", "--spec", "4", "--max-context", "32768", "--kv", "int8"])
        self.assertEqual(removed, ["--spec-min-p", "--mtp", "--batch-mtp"])
        self.assertEqual(draft_layer.strip(["--mtp", "2", "--spec", "2"])[0], ["--spec", "2"])      # Qwen3.6: a draft count

    def test_only_an_explicit_false_turns_it_off(self):
        for cfg in ({}, {"mtp": True}, {"mtp": None}, {"mtp": 0}, {"mtp": "off"}, None):
            self.assertFalse(draft_layer.is_off(cfg), cfg)
        self.assertTrue(draft_layer.is_off({"mtp": False}))

    def test_the_server_applies_it_at_every_start(self):
        on = engine_args({"args": list(ARGS)})
        self.assertIn("--mtp", on)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            off = engine_args({"args": list(ARGS), "mtp": False})
        for flag in ("--mtp", "--spec-min-p", "--batch-mtp"):
            self.assertNotIn(flag, off)
        self.assertEqual(off[off.index("--spec") + 1], "4")
        self.assertIn("without the draft layer", out.getvalue())

    def test_the_web_settings_page_lists_it(self):
        cfg = {"args": []}
        self.assertTrue(any(k["key"] == "mtp" for k in runconfig.view(cfg, "strata-x.json")["keys"]))
        new, changed = runconfig.apply(cfg, {"mtp": False})
        self.assertIs(new["mtp"], False)
        self.assertEqual(changed, ["mtp"])
        with self.assertRaises(ValueError):
            runconfig.apply(cfg, {"mtp": "off"})


if __name__ == "__main__":
    unittest.main()
