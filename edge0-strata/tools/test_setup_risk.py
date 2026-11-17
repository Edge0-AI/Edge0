"""Tests for the owner's rule in setup.py - it recommends, it never forces: a choice bigger than the recommendation is
kept and its risk said (#406's 256K on 64 GB, #364/#384's two GPUs in the low-RAM mode, #403's RAM budget), and the
stops that stay are the ones a user can confirm (confirm_risk: --yes with an explicit flag, or y).  The defaults are
tools/test_setup_golden.py's.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_risk
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, card, install  # noqa: E402


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text, questions asked); stdin answers every input() (None: never asked)."""
    out, asked, code = io.StringIO(), [], None

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class ConfirmRisk(unittest.TestCase):
    """S6: one helper for the stops a user may take on."""

    def go(self, explicit, yes, stdin=None):
        return run(setup.confirm_risk, "it may not fit", explicit, yes, "it does not fit", "a smaller size",
                   stdin=stdin)

    def test_yes_with_an_explicit_choice_is_consent(self):
        code, out, asked = self.go(True, True)
        self.assertIsNone(code)
        self.assertIn("it may not fit", out)
        self.assertEqual(asked, [])

    def test_yes_alone_keeps_the_stop(self):
        code, out, asked = self.go(False, True)
        self.assertEqual(code, 1)
        self.assertIn("it does not fit", out)
        self.assertIn("a smaller size", out)

    def test_interactive_default_is_no(self):
        for explicit in (True, False):
            code, out, asked = self.go(explicit, False, stdin="")
            self.assertEqual(code, 1)
            self.assertIn("[n]", asked[0])
            self.assertIsNone(self.go(explicit, False, stdin="y")[0])

    def test_paging_with_an_explicit_model(self):
        code, out, asked = run(setup.confirm_paging, "IQ3_XXS", 54, "auto", True, True)
        self.assertIsNone(code, out)
        self.assertIn("installing IQ3_XXS with 54 GB of RAM, as you chose (--model)", out)


class Context(unittest.TestCase):
    """S1 (#406 #364): the RAM rule recommends, an explicit or picked context is kept with a note."""
    RAM64, GPU32 = PROFILES["64GB-1x32GB"]

    def test_the_ram_rule(self):
        self.assertEqual(setup.ram_ctx("IQ3_S", 63.7), 131072)
        self.assertEqual(setup.ram_ctx("IQ3_XXS", 63.7), 131072)
        self.assertEqual(setup.ram_ctx("IQ3_XXS", 95.8), 524288)      # 43 + 7.2 + 24 GB fits 96
        self.assertEqual(setup.ram_ctx("IQ3_S", 76.5), 131072)
        self.assertEqual(setup.ram_ctx("IQ3_S", 78.5), 262144)
        # #608: where the 200K estimate fits and the 256K one does not, the rule stays at 128K (200K is a menu step)
        self.assertLessEqual(setup.ctx_ram_need("IQ3_S", 204800), 78.5)
        self.assertGreater(setup.ctx_ram_need("IQ3_S", 262144), 77.5)
        self.assertEqual(setup.ram_ctx("IQ3_S", 77.5), 131072)
        for ram in range(16, 200):
            for model in ("IQ3_S", "IQ3_XXS", "Q2_0"):
                self.assertNotEqual(setup.ram_ctx(model, float(ram)), 204800)
        self.assertEqual(setup.ram_ctx("Q2_0", 31.9), 524288)          # other sizes: no RAM rule
        self.assertEqual(setup.ram_ctx("IQ3_XXS", 31.9, low_ram=True), 524288)   # the KV cache is in VRAM there
        self.assertIsNone(setup.ctx_ram_need("IQ3_XXS", 262144, low_ram=True))
        self.assertAlmostEqual(setup.ctx_ram_need("IQ3_S", 262144), 77.9, places=1)

    def test_an_explicit_256k_is_kept(self):
        code, out, cfg, _ = install(self.RAM64, self.GPU32, ["--family", "qwen", "--model", "IQ3_S", "--no-start",
                                                             "--context", "262144"])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")
        self.assertNotIn("--rope-scaling", cfg["args"])
        self.assertIn("256K with IQ3_S needs ~78 GB of RAM by setup's estimate", out)
        self.assertIn("this PC has 64. Kept as you chose: it may be slower or run out of RAM under load. 128K is the "
                      "recommended size.", out)

    def test_a_picked_256k_is_kept_and_the_menu_says_the_risk(self):
        code, out, cfg, asked = install(self.RAM64, self.GPU32, ["--family", "qwen", "--model", "IQ3_S", "--no-start"],
                                        answers={"Context?": "6"})
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")
        self.assertIn("256K tokens   (needs ~78 GB RAM, this PC has 64: may run out of memory)", out)
        self.assertIn("128K tokens   (recommended for your GPU)\n", out)
        self.assertIn("Kept as you chose", out)

    def test_a_200k_choice_is_kept(self):
        """#406: 200K was added between the 128K rule and 256K - it is inside the trained 262144, so no rope scaling,
        and a pick of it is kept with the note 256K gets instead of being capped to 128K."""
        code, out, cfg, _ = install(self.RAM64, self.GPU32, ["--family", "qwen", "--model", "IQ3_S", "--no-start"],
                                    answers={"Context?": "5"})
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "204800")
        self.assertNotIn("--rope-scaling", cfg["args"])
        self.assertIn("200K tokens   (needs ~77 GB RAM, this PC has 64: may run out of memory)", out)
        self.assertIn("200K with IQ3_S needs ~77 GB of RAM by setup's estimate", out)
        self.assertIn("Kept as you chose", out)

    def test_low_ram_mode_keeps_262k_without_a_ram_note(self):
        ram, found = PROFILES["32GB-2x24GB"]                              # #364
        code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "IQ3_XXS", "--no-start",
                                                 "--context", "262144"])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")
        self.assertNotIn("Kept as you chose: it may be slower", out)

    def test_the_earlier_install_keeps_its_context(self):
        """An update (a new copy of Strata) set up like the earlier install keeps its 256K instead of capping it."""
        with tempfile.TemporaryDirectory() as d:
            prev = Path(d) / "strata-iq3_s.json"
            prev.write_text(json.dumps({"args": ["--max-context", "262144", "--kv", "int8"], "port": 8080}))
            started = mock.Mock(return_value=0)
            code, out, cfg, _ = install(self.RAM64, self.GPU32, [], extra=[
                mock.patch.object(setup, "previous_config", lambda *a: prev),
                mock.patch.object(setup, "start", started)])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")
        self.assertIn("Kept as you chose", out)
        self.assertTrue(started.called)


class BrokenEarlierConfig(unittest.TestCase):
    """#459: a new copy of Strata set up like an earlier install skips an earlier config that does not parse (an
    empty strata-*.json crashed START-HERE with JSONDecodeError) and goes on as a fresh install; configs are written
    whole (a temporary file moved over the old one)."""
    RAM64, GPU32 = PROFILES["64GB-1x32GB"]

    def setup_with(self, files):
        with tempfile.TemporaryDirectory() as d:
            for i, (name, text) in enumerate(files):                  # in order, oldest first
                (Path(d) / name).write_text(text, encoding="utf-8")
                os.utime(Path(d) / name, (1_700_000_000 + i, 1_700_000_000 + i))
            return install(self.RAM64, self.GPU32, [], extra=[
                mock.patch.object(setup, "other_installs", lambda settings: [Path(d)]),
                mock.patch.object(setup, "start", mock.Mock(return_value=0))])

    def test_an_empty_config_is_skipped(self):
        for text, why in (("", "the file is empty"), ("{\"args\": [", "not valid JSON"), ("[1]", "not a JSON object")):
            with self.subTest(text=text):
                code, out, cfg, _ = self.setup_with([("strata-iq3_s.json", text)])
                self.assertEqual(code, 0, out)
                self.assertIn("skipped the earlier config ", out)
                self.assertIn(f"strata-iq3_s.json ({why}): setting this copy up without it", out)
                self.assertNotIn("Found your earlier install", out)
                self.assertIsNotNone(cfg)                                # the fresh install's config

    def test_the_newest_readable_config_is_used(self):
        good = json.dumps({"args": ["--max-context", "262144", "--kv", "int8"], "port": 8080})
        code, out, cfg, _ = self.setup_with([("strata-iq3_s.json", good), ("strata-q2_0.json", "")])
        self.assertEqual(code, 0, out)
        self.assertIn("strata-q2_0.json (the file is empty)", out)
        self.assertIn("Found your earlier install", out)
        self.assertIn("(iq3_s)", out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")

    def test_write_config_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_s.json"
            p.write_text("old", encoding="utf-8")
            setup.write_config(p, {"args": ["--kv", "int8"]})
            self.assertEqual(p.read_text(encoding="utf-8"), json.dumps({"args": ["--kv", "int8"]}, indent=1))
            self.assertEqual([f.name for f in Path(d).iterdir()], ["strata-iq3_s.json"])   # no .tmp left
            with mock.patch.object(Path, "write_text", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    setup.write_config(p, {"args": []})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"args": ["--kv", "int8"]})   # kept whole


class LowRamGpus(unittest.TestCase):
    """S2/S3 (#364 #384): one GPU recommended in the low-RAM mode, all of them when asked for."""

    def setUp(self):
        # the behaviour before RESIDENT_SPLIT_ENGINE (an engine from before 0.1.40); the tests of the newer engine
        # patch MIN_ENGINE themselves, and the default (MIN_ENGINE = 0.1.40) is covered by test_setup_golden
        p = mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))
        p.start()
        self.addCleanup(p.stop)

    def test_explicit_gpus_are_kept_with_the_mapped_variant(self):
        for prof, model in (("32GB-2x24GB", "IQ3_XXS"), ("32GB-2x24GB", "Q2_0"), ("47GB-2x16GB", "IQ3_XXS")):
            with self.subTest(prof=prof, model=model):
                ram, found = PROFILES[prof]
                code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", model, "--no-start",
                                                         "--gpus", "0,1"])
                self.assertEqual(code, 0, out)
                self.assertEqual(cfg["gpu"], [0, 1])
                self.assertEqual(cfg["layer_split"], "auto")
                self.assertIn("--mmap-experts", cfg["args"])
                self.assertNotIn("--resident-experts", cfg["args"])
                self.assertIn("as you chose (--gpus)", out)
                self.assertIn("RAM can fill up to 0 free", out)
                self.assertIn("low-RAM mode on 2 GPUs", out)

    def test_the_share_counts_every_card(self):
        ram, found = PROFILES["47GB-2x16GB"]                              # #384: 2 x 16 GB, 47 GB, IQ3_XXS at 64K
        code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "IQ3_XXS", "--no-start",
                                                 "--gpus", "0,1"])
        one = setup.low_ram_gpu_gb("IQ3_XXS", 15.9, 65536, "int8")
        self.assertIn(f"the GPUs hold ~{100 * 2 * one / setup.MODELS['IQ3_XXS']['arena_gb']:.0f}% of them", out)

    def test_asked_and_answered_two(self):
        ram, found = PROFILES["32GB-2x24GB"]
        code, out, cfg, asked = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start"],
                                        answers={"Low-RAM mode: which GPUs?": "2"})     # the roles are the default (Main last); the low-RAM question still defaults to one card
        self.assertEqual(code, 0, out)
        self.assertTrue(any("Low-RAM mode: which GPUs? [1]" in q for q in asked), asked)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("--mmap-experts", cfg["args"])
        self.assertIn("1) GPU 0 (NVIDIA GeForce RTX 3090, 24 GB) only", out)

    def test_yes_and_enter_keep_one_gpu(self):
        ram, found = PROFILES["32GB-2x24GB"]
        for answers in (None, ""):
            code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start"],
                                        answers=answers)
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["gpu"], 0)
            self.assertIn("--resident-experts", cfg["args"])

    def test_low_ram_resident_with_gpus(self):
        ram, found = PROFILES["32GB-2x24GB"]
        code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start", "--gpus", "0,1",
                                                 "--low-ram", "resident"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("--mmap-experts", cfg["args"])
        self.assertIn("--low-ram resident has no layer split yet", out)

    def test_low_ram_resident_alone_keeps_one_gpu_without_asking(self):
        ram, found = PROFILES["32GB-2x24GB"]
        code, out, cfg, asked = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start",
                                                     "--low-ram", "resident"], answers="")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertFalse(any("Low-RAM mode" in q for q in asked), asked)

    def test_low_ram_on_gpus_with_a_resident_split_engine(self):
        """#642: an engine that runs the resident variant on a layer split keeps the cards together, resident."""
        ram, found = PROFILES["32GB-2x24GB"]
        with mock.patch.object(setup, "MIN_ENGINE", setup.RESIDENT_SPLIT_ENGINE):
            code, out, cfg, asked = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start"],
                                            answers={"Use these roles?": "y"})     # the recommended roles: Main 0 (last), Helper 1
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertIn("--resident-experts", cfg["args"])
            self.assertFalse(any("Low-RAM mode" in q for q in asked), asked)
            self.assertIn("stay in RAM", out)
            code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start",
                                                     "--gpus", "0,1", "--low-ram", "resident"])
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertIn("--resident-experts", cfg["args"])
            self.assertNotIn("no layer split yet", out)


class StartOnSeveralGpus(unittest.TestCase):
    """A resident low-RAM config started on several GPUs reads the experts through the file cache (#364 #384)."""

    def setUp(self):
        # the behaviour before RESIDENT_SPLIT_ENGINE (an engine from before 0.1.40); the tests of the newer engine
        # patch MIN_ENGINE themselves, and the default (MIN_ENGINE = 0.1.40) is covered by test_setup_golden
        p = mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))
        p.start()
        self.addCleanup(p.stop)

    def test_split_mmap(self):
        cfg = {"args": ["--pack", "p", "--resident-experts", "--kv", "int8"]}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(setup.split_mmap(cfg))
            self.assertFalse(setup.split_mmap(cfg))
            self.assertFalse(setup.split_mmap({"args": ["--mmap-experts"]}))
        self.assertEqual(cfg["args"], ["--pack", "p", "--mmap-experts", "--kv", "int8"])
        self.assertIn("no layer split yet", out.getvalue())

    def offer(self, args, stdin):
        found = PROFILES["32GB-2x24GB"][1]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            cfg = {"args": list(args)}
            with mock.patch.object(setup, "gpus", lambda: found), \
                    mock.patch.object(setup, "engine_runs_on", lambda g: True):
                code, out, asked = run(setup.offer_together, p, cfg, stdin is None, stdin=stdin)
            return code, out, asked, json.loads(p.read_text())

    def test_offer_together_recommends_one_gpu_for_a_resident_config(self):
        code, out, asked, cfg = self.offer(["--resident-experts"], None)       # --yes: the recommendation
        self.assertIsNone(code, out)
        self.assertNotIn("gpu", cfg)
        self.assertEqual(cfg["args"], ["--resident-experts"])
        code, out, asked, cfg = self.offer(["--resident-experts"], "y")
        self.assertIn("[n]", asked[0])
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["args"], ["--mmap-experts"])   # #1447: no helper cache, no --remote-expert-opt
        code, out, asked, cfg = self.offer(["--mmap-experts"], None)           # other configs: as before
        self.assertEqual(cfg["gpu"], [0, 1])

    def test_resident_split_engine_keeps_a_resident_config(self):
        """#642: from RESIDENT_SPLIT_ENGINE a resident config is offered both cards as any other, and stays resident."""
        with mock.patch.object(setup, "MIN_ENGINE", setup.RESIDENT_SPLIT_ENGINE):
            cfg = {"args": ["--pack", "p", "--resident-experts", "--kv", "int8"]}
            self.assertFalse(setup.split_mmap(cfg))
            self.assertEqual(cfg["args"], ["--pack", "p", "--resident-experts", "--kv", "int8"])
            code, out, asked, cfg = self.offer(["--resident-experts"], None)   # --yes: the recommendation
            self.assertIsNone(code, out)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertEqual(cfg["args"], ["--resident-experts"])
            self.assertNotIn("OS file cache", out)

    def test_start_with_gpus(self):
        found = PROFILES["32GB-2x24GB"][1]
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--resident-experts"], "gpu": 0, "gpus_asked": True}))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: found), \
                    mock.patch.object(setup, "ensure_engine_for", lambda cards, path, cfg, yes: cfg), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup.subprocess, "call", call):
                code, out, _ = run(setup.start, p, None, [0, 1], False, True)
            cfg = json.loads(p.read_text())
        self.assertIsNone(code, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["args"], ["--mmap-experts"])   # #1447: no helper cache, no --remote-expert-opt
        self.assertIn("no layer split yet", out)
        self.assertTrue(call.called)


class SmallCard(unittest.TestCase):
    """S9: a card under 8 GB named with --gpus is a risk to confirm, not a stop."""
    FOUND = [card(0, "NVIDIA GeForce RTX 3090", 24.0, "86"), card(1, "NVIDIA GeForce RTX 2060", 6.0, "75")]

    def test_named_with_yes(self):
        code, out, cfg, _ = install(63.7, self.FOUND, ["--family", "qwen", "--model", "Q2_0", "--no-start",
                                                       "--gpus", "0,1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("GPU 1 (NVIDIA GeForce RTX 2060) has 6 GB of VRAM", out)
        self.assertIn("is used together with the others, as you chose", out)

    def test_named_and_enter_stops(self):
        code, out, cfg, asked = install(63.7, self.FOUND, ["--family", "qwen", "--model", "Q2_0", "--no-start",
                                                           "--gpus", "0,1"], answers="")
        self.assertEqual(code, 1)
        self.assertTrue(any("Use it anyway?" in q for q in asked), asked)

    def test_not_named_is_not_offered(self):
        code, out, cfg, _ = install(63.7, self.FOUND, ["--family", "qwen", "--model", "Q2_0", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)                                   # one card: as before


class SplitShortCard(unittest.TestCase):
    """#448: a second card too small to lend a split's prompt chunk: the first card alone recommended, the pair kept
    when named."""
    FOUND = [card(0, "NVIDIA RTX PRO 4500 Blackwell", 31.8, "120"), card(1, "NVIDIA GeForce RTX 3080", 10.0, "86")]

    def test_rule(self):
        self.assertEqual([g["index"] for g in setup.split_short(self.FOUND)], [1])
        self.assertEqual(setup.split_short(PROFILES["32GB-2x24GB"][1]), [])
        self.assertEqual(setup.split_short(PROFILES["47GB-2x16GB"][1]), [])
        eights = [card(i, "NVIDIA GeForce RTX 3070", 8.0, "86") for i in range(2)]
        self.assertEqual(setup.split_short(eights), [])                   # neither card alone reads more

    def test_yes_takes_the_recommended_roles(self):
        """The 10 GB card is above the 8 GB line, so it starts as a Helper (the owner's rule); the first card is Main and goes last."""
        code, out, cfg, _ = install(127.8, self.FOUND, ["--family", "qwen", "--model", "Q2_0", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [1, 0])
        self.assertEqual(cfg["gpu_roles"], {"0": "main", "1": "helper"})

    def test_named_pair_is_kept(self):
        code, out, cfg, _ = install(127.8, self.FOUND, ["--family", "qwen", "--model", "Q2_0", "--no-start",
                                                        "--gpus", "0,1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])

    def test_offer_together_defaults_to_one(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            with mock.patch.object(setup, "gpus", lambda: self.FOUND), \
                    mock.patch.object(setup, "engine_runs_on", lambda g: True):
                code, out, asked = run(setup.offer_together, p, {"args": ["--mmap-experts"]}, True)
            cfg = json.loads(p.read_text())
        self.assertIsNone(code, out)
        self.assertNotIn("gpu", cfg)
        self.assertIn("#448", out)


class RamFloor(unittest.TestCase):
    """S7: less RAM than the smallest model needs: a stop with --yes alone, a risk with --model."""
    FOUND = [card(0, "NVIDIA GeForce RTX 3060", 8.0, "86")]

    def test_yes_alone_stops(self):
        code, out, cfg, _ = install(19.9, self.FOUND, ["--family", "qwen", "--no-start"])
        self.assertEqual(code, 1)
        self.assertIn("--model NAME --yes", out)

    def test_an_explicit_model_goes_on(self):
        code, out, cfg, _ = install(19.9, self.FOUND, ["--family", "coder", "--model", "IQ1_M", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertIn("going on with 20 GB of RAM, as you chose", out)


class KvStreaming(unittest.TestCase):
    """8: --kv-streaming on|off|auto (auto: the RAM test, as before)."""
    RAM, FOUND = PROFILES["64GB-1x32GB"]

    def go(self, model, *flags):
        return install(self.RAM, self.FOUND, ["--family", "qwen", "--model", model, "--no-start", *flags])

    def test_off(self):
        code, out, cfg, _ = self.go("IQ3_XXS", "--kv-streaming", "off")
        self.assertEqual(code, 0, out)
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertIn("KV streaming off, as you chose", out)

    def test_on_past_the_ram_test(self):
        code, out, cfg, _ = self.go("IQ3_S", "--kv-streaming", "on")         # auto leaves it off on 64 GB
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--kv-resident"), "32768")
        self.assertIn("Kept as you chose (--kv-streaming on)", out)

    def test_k8v4_streams(self):
        code, out, cfg, _ = self.go("Q2_0", "--kv-streaming", "on", "--kv", "k8v4")
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--kv-resident"), "32768")
        self.assertEqual(arg(cfg, "--kv"), "k8v4")

    def test_on_where_it_cannot(self):
        code, out, cfg, _ = self.go("Q2_0", "--kv-streaming", "on", "--context", "32768")
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertIn("a context under 64K is not streamed", out)

    def test_budget_flag_on_another_size(self):
        code, out, cfg, _ = self.go("Q2_0", "--resident-budget-gib", "30")
        self.assertEqual(code, 0, out)
        self.assertNotIn("--resident-budget-gib", cfg["args"])
        self.assertIn("--resident-budget-gib is for UD-Q4_K_XL", out)


class RulesAgreeWithTheWrittenConfig(unittest.TestCase):
    """The launcher's preset diff asks setup's rules (`kv_streaming_wanted`, `low_ram_wanted`) what "auto" decides on
    this PC, instead of comparing the word "auto" with a config's flags.  That is only honest if those rules are
    exactly what setup writes into the model's config: the same PC, the same flags, every size and every
    --kv-streaming / --low-ram choice."""
    RAM64, CARDS64 = PROFILES["64GB-1x32GB"]
    RAM32, CARDS32 = PROFILES["32GB-2x24GB"]

    def go(self, ram, cards, model, *flags):
        return install(ram, cards, ["--family", "qwen", "--model", model, "--no-start", *flags])

    @mock.patch.object(setup, "is_wsl", lambda: False)   # #974: not WSL, as in install()
    def test_the_streaming_rule_is_the_flag_setup_writes(self):
        for model, ctx, kv, choice in itertools.product(("Q2_0", "IQ3_XXS", "IQ3_S"), ("32768", "131072"),
                                                        ("int8", "q4_0", "k8v4"), ("auto", "on", "off")):
            code, out, cfg, _ = self.go(self.RAM64, self.CARDS64, model, "--context", ctx, "--kv", kv,
                                        "--kv-streaming", choice)
            self.assertEqual(code, 0, out)
            self.assertEqual("--kv-resident" in cfg["args"],
                             setup.kv_streaming_wanted(model, int(ctx), kv, self.RAM64, choice),
                             f"{model} at {ctx} with {kv}, --kv-streaming {choice}: setup wrote {cfg['args']}")

    def test_the_low_ram_rule_is_the_flag_setup_writes(self):
        pcs = [(self.RAM64, self.CARDS64), (self.RAM32, self.CARDS32)]      # auto is on here, off there
        for (ram, cards), model, choice in itertools.product(pcs, ("Q2_0", "IQ3_XXS", "IQ3_S"),
                                                              ("auto", "on", "off", "mmap")):
            code, out, cfg, _ = self.go(ram, cards, model, "--low-ram", choice)
            self.assertEqual(code, 0, out)
            low = "--resident-experts" in cfg["args"] or "--mmap-experts" in cfg["args"]
            self.assertEqual(low, setup.low_ram_wanted(model, ram, choice),
                             f"{model} on {ram:.0f} GB of RAM, --low-ram {choice}: setup wrote {cfg['args']}")


if __name__ == "__main__":
    unittest.main()
