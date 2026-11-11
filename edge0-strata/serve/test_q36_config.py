"""serve/test_q36_config.py - what the server hands the Qwen3.6-35B-A3B engine (arch qwen35moe): the config's pictures
("vision" section -> --vision) and parallel requests ("parallel": N -> --batch N), as for Flash-Next's engine.

    python -m unittest serve.test_q36_config
"""
from __future__ import annotations

import unittest
from unittest import mock

from serve.server import engine_args, parallel_args


def cfg(**kw):
    base = {"arch": "qwen35moe", "exe": "strata", "args": ["--native", "/m/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                                                           "--max-context", "32768", "--prefill", "auto", "--mtp", "2"]}
    base.update(kw)
    return base


class Q36Config(unittest.TestCase):
    def test_parallel_becomes_batch(self):
        a = engine_args(cfg(parallel=4))
        self.assertEqual(a[a.index("--batch") + 1], "4")
        self.assertNotIn("--batch", engine_args(cfg()))
        self.assertNotIn("--batch", engine_args(cfg(parallel=1)))

    def test_more_than_the_window_is_said(self):
        with mock.patch("builtins.print") as pr:
            a = engine_args(cfg(parallel=12))
        self.assertEqual(a[a.index("--batch") + 1], "12")          # the engine caps it at 8 and says so
        self.assertTrue(any("at most 8" in str(c) for c in pr.call_args_list))

    def test_a_vision_section_starts_the_engine_with_pictures(self):
        v = {"exe": "strata-vision", "mmproj": "/m/mmproj-F16.gguf", "model": "/m/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
             "gpu": False, "max_tokens": 300}
        self.assertIn("--vision", engine_args(cfg(vision=v)))
        self.assertIn("--vision", engine_args(cfg(vision=v, args=cfg()["args"] + ["--vision"])))
        self.assertEqual(engine_args(cfg(vision=v, args=cfg()["args"] + ["--vision"])).count("--vision"), 1)
        self.assertNotIn("--vision", engine_args(cfg()))

    def test_an_explicit_batch_in_the_args_wins(self):
        c = cfg(parallel=4, args=cfg()["args"] + ["--batch", "2"])
        self.assertEqual(parallel_args(c, c["args"]), [])


if __name__ == "__main__":
    unittest.main()
