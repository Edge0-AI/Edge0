"""Tests for setup.py's low-RAM messages and the paging question (#250): the one-GPU warning says why and shows the
RAM math, and an explicit --low-ram off installs with --yes instead of stopping at "Install it anyway?" (default
n).  Mocked input - no GPU, no downloads.

    python -m unittest tools.test_setup_lowram
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text); `stdin` answers input() (None: never asked)."""
    out = io.StringIO()
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    code = None
    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class OneGpuWhy(unittest.TestCase):
    def test_auto_shows_the_ram_math(self):
        lines = setup.low_ram_one_gpu_why("IQ3_XXS", 46, "auto")       # the #250 report: 46 GB, IQ3_XXS
        text = "\n".join(lines)
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        need = arena + setup.LOW_RAM_HEADROOM_GB
        self.assertIn(f"{arena:.0f} + {setup.LOW_RAM_HEADROOM_GB} = {need:.0f} GB", text)
        self.assertIn("this PC has 46 GB", text)
        self.assertIn("layer split", text)                            # why one GPU is recommended
        self.assertIn("--gpus 0,1", text)                             # #364 #384: and the way to use them all
        self.assertIn("OS file cache", text)
        self.assertNotIn("--low-ram off", text)                       # (that pages the experts: not the way)

    def test_an_explicit_choice_says_so(self):
        text = "\n".join(setup.low_ram_one_gpu_why("Q2_0", 64, "resident"))
        self.assertIn("you chose the low-RAM mode (--low-ram resident)", text)


class PagingQuestion(unittest.TestCase):
    MODEL, RAM = "IQ3_XXS", 46                                       # 60 GB wanted, 46 here

    def test_yes_with_low_ram_off_installs(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "off", True)
        self.assertIsNone(code, out)
        self.assertEqual(asked, [])
        self.assertIn("installing IQ3_XXS with 46 GB of RAM, as you chose (--low-ram off)", out)

    def test_yes_with_auto_still_stops(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "auto", True)
        self.assertEqual(code, 1)
        self.assertIn("IQ3_XXS needs about 60 GB of RAM; this PC has 46 GB", out)
        self.assertIn("--low-ram off --yes", out)                    # the way to insist

    def test_interactive(self):
        for choice, answer, want in (("off", "", None), ("auto", "", 1), ("auto", "y", None), ("off", "n", 1)):
            with self.subTest(choice=choice, answer=answer):
                code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, choice, False, stdin=answer)
                self.assertEqual(code, want, out)
                self.assertEqual(len(asked), 1)
                self.assertIn("[y]" if choice == "off" else "[n]", asked[0])


class DraftLayerCountsInTheFit(unittest.TestCase):
    """The draft layer (MTP) takes about 1 GB of graphics memory out of the expert cache: every fit check counts it while it is on."""

    def test_the_card_holds_one_gb_less_with_mtp_on(self):
        on = setup.low_ram_gpu_gb("IQ3_XXS", 12.0)
        off = setup.low_ram_gpu_gb("IQ3_XXS", 12.0, mtp=False)
        self.assertAlmostEqual(off - on, setup.MTP_VRAM_GB)
        self.assertEqual(setup.MTP_VRAM_GB, 1.0)
        self.assertAlmostEqual(setup.low_ram_gpu_share("IQ3_XXS", 12.0, mtp=False) - setup.low_ram_gpu_share("IQ3_XXS", 12.0),
                               1.0 / setup.MODELS["IQ3_XXS"]["arena_gb"])

    def test_the_edge_of_what_fits_moves_by_the_draft_layer(self):
        arena = setup.MODELS["IQ1_M"]["arena_gb"]
        ram = arena + 6 - 7.0 + 0.5            # a card of 12 GB holds 7 GB beside the dense part; the draft layer needs one of them
        self.assertFalse(setup.low_ram_fits("IQ1_M", ram, 12.0))
        self.assertTrue(setup.low_ram_fits("IQ1_M", ram, 12.0, mtp=False))


if __name__ == "__main__":
    unittest.main()
