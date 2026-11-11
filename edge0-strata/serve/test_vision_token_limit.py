"""I7 (tools and images review, 2026-10-22): a picture reader without vision.max_tokens in its config was unclamped, and #3690 saw one
picture become 93,700 image tokens.  The server gives every reader a limit (GPU 1024, CPU 300) unless the config names one, and logs a
picture that came back over 4 times the limit.

    python -m unittest serve.test_vision_token_limit
"""
import io
import unittest
from contextlib import redirect_stdout

import serve.server as server

BASE = {"exe": "strata-vision", "mmproj": "m.gguf", "model": "t.gguf"}


def limit_of(cfg):
    args = server.Vision.reader_args(cfg)
    return int(args[args.index("--max-tokens") + 1]) if "--max-tokens" in args else None


class TokenLimit(unittest.TestCase):
    def test_a_config_without_max_tokens_gets_the_default_of_its_kind(self):
        self.assertEqual(limit_of({**BASE, "gpu": True}), 1024)
        self.assertEqual(limit_of({**BASE, "gpu": False}), 300)
        self.assertEqual(limit_of(dict(BASE)), 300)

    def test_a_limit_the_person_set_is_kept_not_capped(self):
        self.assertEqual(limit_of({**BASE, "gpu": True, "max_tokens": 4096}), 4096)
        self.assertEqual(limit_of({**BASE, "max_tokens": "512"}), 512)

    def test_zero_still_means_no_limit(self):
        self.assertIsNone(limit_of({**BASE, "gpu": True, "max_tokens": 0}))

    def test_an_unreadable_value_falls_back_to_the_default(self):
        self.assertEqual(limit_of({**BASE, "gpu": True, "max_tokens": "lots"}), 1024)

    def test_min_tokens_is_unchanged(self):
        args = server.Vision.reader_args({**BASE, "gpu": True, "min_tokens": 768})
        self.assertEqual(args[args.index("--min-tokens") + 1], "768")


class Warn(unittest.TestCase):
    def reader(self, limit):
        v = server.Vision.__new__(server.Vision)
        v.max_tokens = limit
        return v

    def said(self, limit, tokens):
        out = io.StringIO()
        with redirect_stdout(out):
            self.reader(limit).warn_if_over_limit(tokens)
        return out.getvalue()

    def test_93700_tokens_under_a_1024_limit_is_logged_with_the_numbers(self):
        text = self.said(1024, 93700)
        self.assertIn("93700 image tokens", text)
        self.assertIn("limit of 1024", text)
        self.assertIn("warmed up at N image tokens", text)

    def test_a_picture_within_four_times_the_limit_is_quiet(self):
        self.assertEqual(self.said(1024, 1024), "")
        self.assertEqual(self.said(1024, 4096), "")

    def test_no_limit_no_warning(self):
        self.assertEqual(self.said(0, 93700), "")
        v = server.Vision.__new__(server.Vision)          # a reader object made without the attribute (older tests)
        v.warn_if_over_limit(93700)


if __name__ == "__main__":
    unittest.main()
