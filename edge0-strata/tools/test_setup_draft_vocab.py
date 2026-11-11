"""Tests for setup.py's draft-vocabulary choice (#287): a setup run again without --draft-vocab keeps the subset the
model's config chose before (cyrillic, fr, en), and refresh_draft_vocab copies the chosen shipped subset.  Pure file work
in a temporary folder - no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_draft_vocab
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class SavedChoice(unittest.TestCase):
    def test_a_config_keeps_its_choice(self):
        with tempfile.TemporaryDirectory() as d:
            for choice in setup.DRAFT_VOCABS:
                p = Path(d) / "strata-x.json"
                p.write_text(json.dumps({"args": [], "draft_vocab": choice}), encoding="utf-8")
                self.assertEqual(setup.saved_draft_vocab(p), choice)

    def test_no_choice_without_a_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.saved_draft_vocab(Path(d) / "missing.json"))
            for text in ("{}", "[1]", "not json", json.dumps({"draft_vocab": "klingon"})):
                p = Path(d) / "strata-y.json"
                p.write_text(text, encoding="utf-8")
                self.assertIsNone(setup.saved_draft_vocab(p), text)


class Refresh(unittest.TestCase):
    def test_cyrillic_replaces_a_shipped_subset_and_stays(self):
        if not (ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).exists():
            self.skipTest("data/draft_vocab_cyrillic.bin is not in this checkout")
        want = hashlib.sha256((ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((ROOT / "data" / setup.DRAFT_VOCABS["cjk"]).read_bytes())
            setup.refresh_draft_vocab(rt, "cyrillic")
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)
            setup.refresh_draft_vocab(rt, "cyrillic")                       # again: nothing changes
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)


    def test_fr_replaces_a_shipped_subset_and_back(self):
        # #597: the French subset is a shipped one too - a later --draft-vocab cjk (or en) replaces it again, while a
        # subset made by hand stays
        data = ROOT / "data"
        if not (data / setup.DRAFT_VOCABS["fr"]).exists():
            self.skipTest("data/draft_vocab_fr.bin is not in this checkout")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((data / setup.DRAFT_VOCABS["en"]).read_bytes())
            setup.refresh_draft_vocab(rt, "fr")
            self.assertEqual(sha(rt / "draft_vocab.bin"), sha(data / setup.DRAFT_VOCABS["fr"]))
            setup.refresh_draft_vocab(rt, "cjk")
            self.assertEqual(sha(rt / "draft_vocab.bin"), sha(data / setup.DRAFT_VOCABS["cjk"]))
            (rt / "draft_vocab.bin").write_bytes(bytes([1, 0, 0, 0]))   # made by hand: kept
            setup.refresh_draft_vocab(rt, "fr")
            self.assertEqual((rt / "draft_vocab.bin").read_bytes(), bytes([1, 0, 0, 0]))

    def test_fr_holds_the_en_subset_in_its_order(self):
        # built with tools/draft_vocab.py --base data/draft_vocab_en.bin --corpus ...: the base first, unchanged
        from array import array
        data = ROOT / "data"
        if not (data / setup.DRAFT_VOCABS["fr"]).exists():
            self.skipTest("data/draft_vocab_fr.bin is not in this checkout")
        en = list(array("i", (data / setup.DRAFT_VOCABS["en"]).read_bytes()))
        fr = list(array("i", (data / setup.DRAFT_VOCABS["fr"]).read_bytes()))
        self.assertEqual(fr[:len(en)], en)
        self.assertEqual(len(set(fr)), len(fr))
        self.assertEqual(fr[len(en):], sorted(fr[len(en):]))


class SharedFolder(unittest.TestCase):
    """setup.py --update refreshes the draft subset of every installed config, most recently used first.  Two configs
    that name one MTP folder: the folder keeps the subset of the one used last, not of the one used longest ago."""

    def setUp(self):
        self.data = ROOT / "data"
        if not all((self.data / setup.DRAFT_VOCABS[c]).exists() for c in ("cjk", "cyrillic", "en")):
            self.skipTest("the shipped draft subsets are not in this checkout")
        self.sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731

    def run_update(self, rt, cfgs):
        done: set = set()
        for cfg in cfgs:                                                    # update_install's loop, newest first
            setup.refresh_shared_draft_vocab(cfg, done)
        return self.sha(rt / "draft_vocab.bin")

    def test_the_most_recently_used_config_decides(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((self.data / setup.DRAFT_VOCABS["cyrillic"]).read_bytes())
            in_use = {"args": ["--mtp", str(rt), "--spec", "4"], "draft_vocab": "cyrillic"}
            older = {"args": ["--mtp", str(rt), "--spec", "4"]}             # no key: the default subset (cjk)
            self.assertEqual(self.run_update(rt, [in_use, older]), self.sha(self.data / setup.DRAFT_VOCABS["cyrillic"]))
            self.assertEqual(self.run_update(rt, [older, in_use]), self.sha(self.data / setup.DRAFT_VOCABS["cjk"]))

    def test_separate_folders_each_keep_their_own(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            for rt in (a, b):
                rt.mkdir()
                (rt / "draft_vocab.bin").write_bytes((self.data / setup.DRAFT_VOCABS["cjk"]).read_bytes())
            done: set = set()
            setup.refresh_shared_draft_vocab({"args": ["--mtp", str(a), "--spec", "4"], "draft_vocab": "cyrillic"}, done)
            setup.refresh_shared_draft_vocab({"args": ["--mtp", str(b), "--spec", "4"], "draft_vocab": "en"}, done)
            self.assertEqual(self.sha(a / "draft_vocab.bin"), self.sha(self.data / setup.DRAFT_VOCABS["cyrillic"]))
            self.assertEqual(self.sha(b / "draft_vocab.bin"), self.sha(self.data / setup.DRAFT_VOCABS["en"]))

    def test_a_config_without_a_draft_layer_is_skipped(self):
        done: set = set()
        setup.refresh_shared_draft_vocab({"args": ["--spec", "0"]}, done)
        setup.refresh_shared_draft_vocab({}, done)
        self.assertEqual(done, set())


class SmallCardNote(unittest.TestCase):
    """#474: a card under 14 GB is told about a smaller draft subset - a note only, and not when one was chosen."""

    def test_note_on_a_small_card_only(self):
        note = setup.draft_vocab_note(12.0, None)
        self.assertTrue(note)
        self.assertIn("--draft-vocab en", " ".join(note))
        self.assertIn("the draft head does not fit", " ".join(note))
        for vram, chosen in ((16.0, None), (24.0, None), (12.0, "en"), (12.0, "cyrillic"), (12.0, "fr"), (12.0, "cjk"),
                             (0.0, None)):
            self.assertEqual(setup.draft_vocab_note(vram, chosen), [], (vram, chosen))

    def test_sizes_follow_the_shipped_subsets(self):
        # the MiB the note gives scale with the subsets' token counts (4 bytes per token id in data/)
        sizes = {}
        for choice, name in setup.DRAFT_VOCABS.items():
            p = ROOT / "data" / name
            if not p.exists():
                self.skipTest(f"data/{name} is not in this checkout")
            sizes[choice] = p.stat().st_size // 4
        for choice in ("en", "cyrillic", "fr"):
            want = setup.DRAFT_VOCAB_MIB["cjk"] * sizes[choice] / sizes["cjk"]
            self.assertAlmostEqual(setup.DRAFT_VOCAB_MIB[choice], want, delta=3)


if __name__ == "__main__":
    unittest.main()
