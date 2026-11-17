"""Tests for setup.py's model menu (0.1.44): the family order, NVFP4 hidden on AMD and Intel, the NVFP4 line's card note
(RTX 50 native FP4, older RTX slower than UD-IQ4_XS), the Unsloth sizes in order with images on, and the prebuilt
install removing a stale strata_kq library.

    python -m unittest tools.test_setup_menu
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class FamilyOrder(unittest.TestCase):
    def test_the_menu_order_on_nvidia(self):
        fams = setup.menu_families(False) + list(setup.OWN_PATH_FAMILIES)
        self.assertEqual(fams, ["qwen", "swift", "coder", "unsloth", "gyro", "nvfp4", "qwen36"])

    def test_the_default_stays_the_first_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertIn("GSQ-RCO", setup.FAMILIES["qwen"]["by"])

    def test_nvfp4_is_not_listed_on_amd_and_intel(self):
        fams = setup.menu_families(True)
        self.assertNotIn("nvfp4", fams)
        self.assertEqual(fams, ["qwen", "swift", "coder", "unsloth", "gyro"])
        self.assertIn("nvfp4", setup.FAMILIES)             # --family nvfp4 still exists, and says why not

    def test_the_unsloth_line_names_the_alternative_on_amd_and_intel(self):
        self.assertIn("UD-IQ4_XS", setup.family_menu_note("unsloth", True, None))
        self.assertEqual(setup.family_menu_note("unsloth", False, {"arch": 86}), "")
        self.assertEqual(setup.family_menu_note("coder", True, None), "")


class Nvfp4Note(unittest.TestCase):
    def test_rtx_50_has_native_fp4(self):
        line = setup.family_menu_note("nvfp4", False, {"arch": 120, "name": "RTX 5070"})
        self.assertIn("native FP4 on this card", line)

    def test_older_rtx_runs_but_is_slower_than_ud_iq4_xs(self):
        for arch in (75, 86, 89):
            line = setup.family_menu_note("nvfp4", False, {"arch": arch})
            self.assertIn("runs, but UD-IQ4_XS is faster on this card", line, arch)
            self.assertNotIn("native FP4", line)

    def test_an_avx2_cpu_is_told_its_rows_but_not_refused(self):      # #2347
        line = setup.family_menu_note("nvfp4", False, {"arch": 120}, avx2=True, avx512=False)
        self.assertIn("no AVX-512", line)
        self.assertIn("AVX2", line)
        self.assertIn("native FP4 on this card", line)               # the card part is unchanged
        self.assertTrue(setup.nvfp4_cpu_note(True, False))

    def test_an_avx512_cpu_and_a_cpu_without_avx2_get_no_cpu_note(self):
        self.assertEqual(setup.nvfp4_cpu_note(True, True), "")
        self.assertEqual(setup.nvfp4_cpu_note(False, False), "")     # the older-CPU build says its own piece
        line512 = setup.family_menu_note("nvfp4", False, {"arch": 120})
        self.assertNotIn("AVX", line512)
        self.assertEqual(line512, setup.family_menu_note("nvfp4", False, {"arch": 120}, avx2=True, avx512=True))

    def test_the_line_gives_the_download_and_the_ram_need(self):
        line = setup.family_menu_note("nvfp4", False, {"arch": 120})
        d = setup.MODELS["NVFP4"]
        self.assertIn(f"{d['download_gb']:.0f} GB download", line)
        self.assertIn(f"~{d['ram_gb']} GB of RAM", line)


class UnslothSizes(unittest.TestCase):
    def test_the_sizes_in_menu_order(self):
        names = [m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ())]
        self.assertEqual(names, ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"])

    def test_nothing_is_experimental_and_images_are_on(self):
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            d = setup.MODELS[m]
            self.assertFalse(d.get("experimental"), m)
            self.assertFalse(d.get("nvidia_only"), m)
            self.assertIs(d.get("vision"), True, m)
            self.assertIn("download_gb", d)
            self.assertIn("ram_gb", d)
        self.assertNotIn("experimental", setup.MODELS["UD-Q4_K_XL"]["about"].lower())
        self.assertTrue(setup.FAMILIES["unsloth"]["vision"])

    def test_the_family_line_lists_every_size(self):
        about = setup.FAMILIES["unsloth"]["about"]
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertIn(m, about)


class RecommendedSizes(unittest.TestCase):
    """0.1.44 (owner): IQ3_S, UD-Q4_K_XL and UD-IQ4_XS are the recommended sizes; recommend, never force."""

    def names(self, family):
        return [m for m in setup.MODELS if family in setup.MODELS[m].get("families", ("qwen", "swift"))]

    def test_qwen_and_swift_default_to_iq3_s_when_it_fits(self):
        for fam in ("qwen", "swift"):
            n = self.names(fam)
            self.assertIn("IQ3_S", n)
            self.assertEqual(setup.recommended_size(fam, n, 67.7), "IQ3_S")
            self.assertEqual(setup.recommended_size(fam, n, 191.0), "IQ3_S")      # not the largest: IQ3_S stays the pick
            self.assertEqual([m for m in n if setup.size_tag(fam, m)], ["IQ3_S"])  # the others are choices, unmarked

    def test_without_the_ram_the_largest_that_fits_and_iq3_s_stays_marked(self):
        n = self.names("qwen")
        self.assertEqual(setup.recommended_size("qwen", n, 61.0), "IQ3_XXS")      # IQ3_S needs 62 GB
        self.assertEqual(setup.recommended_size("qwen", n, 52.0), "IQ3_XXS")      # 48 GB tier (RAM + VRAM - 5 GB)
        self.assertEqual(setup.recommended_size("qwen", n, 40.0, 11.0), "IQ3_XXS")  # 40 + 11 - 5 = 46 GB together
        self.assertEqual(setup.recommended_size("qwen", n, 32.0, 12.0), "IQ2_XS")  # 44 - 5 = 39 GB: the 40 GB tier
        self.assertEqual(setup.recommended_size("swift", self.names("swift"), 32.0, 12.0), "IQ2_XS")
        self.assertEqual(setup.recommended_size("qwen", n, 32.0, 35.0), "IQ3_S")   # 67 - 5 = 62 GB together
        self.assertEqual(setup.size_tag("qwen", "IQ3_S"), " (recommended)")

    def test_3425_the_tiers_leave_5_gb_for_windows(self):
        # the reporter: RTX 5080 (16 GB) + 32 GB of RAM summed to 48 GB and got IQ3_XXS, which read ~11 GB from the SSD
        n = self.names("qwen")
        self.assertEqual(setup.TIER_SYSTEM_RESERVE_GB, 5.0)
        self.assertEqual(setup.recommended_size("qwen", n, 31.9, 15.9), "IQ2_XS")
        self.assertEqual(setup.tier_pick(31.9, 15.9), ("swift", "IQ2_XS"))
        # boundaries: floors 38 / 46 / 62 plus the 5 GB
        self.assertEqual(setup.tier_pick(43.0), ("swift", "IQ2_XS"))
        self.assertIsNone(setup.tier_pick(42.9))
        self.assertEqual(setup.tier_pick(51.0), ("qwen", "IQ3_XXS"))
        self.assertEqual(setup.tier_pick(50.9), ("swift", "IQ2_XS"))
        self.assertEqual(setup.tier_pick(67.0), ("qwen", "IQ3_S"))
        self.assertEqual(setup.tier_pick(66.9), ("qwen", "IQ3_XXS"))
        self.assertEqual(setup.tier_pick(40.0, 3.0), ("swift", "IQ2_XS"))      # VRAM counts the same as RAM

    def test_coder_has_nothing_marked(self):
        n = self.names("coder")
        self.assertEqual([setup.size_tag("coder", m) for m in n], [""] * len(n))
        self.assertEqual(setup.recommended_size("coder", n, 64.0), n[0])


class UnslothDefaultEnv(unittest.TestCase):
    """0.1.44 (owner): the env setup writes for Unsloth's files (review-quants.md section 5)."""

    def test_q5_and_q6_turn_the_k_quant_prompt_kernels_on_q4_does_not(self):
        f = setup.unsloth_default_env
        self.assertEqual(f("unsloth", "UD-Q5_K_XL", False, False)["STRATA_MMQ_KQ"], "1")
        self.assertEqual(f("unsloth", "UD-Q6_K_XL", True, True)["STRATA_MMQ_KQ"], "1")
        for m in ("UD-Q4_K_XL", "UD-IQ4_XS", "Q8_0"):
            self.assertNotIn("STRATA_MMQ_KQ", f("unsloth", m, False, False), m)

    def test_hc_q8_on_cuda_for_the_ud_files_only(self):
        f = setup.unsloth_default_env
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL"):
            self.assertEqual(f("unsloth", m, False, True)["STRATA_HC_Q8"], "1", m)
            self.assertNotIn("STRATA_HC_Q8", f("unsloth", m, True, True), m)          # not on AMD
        self.assertNotIn("STRATA_HC_Q8", f("unsloth", "Q8_0", False, True))

    def test_tail_skip_off_only_when_every_expert_is_in_ram(self):
        f = setup.unsloth_default_env
        self.assertEqual(f("unsloth", "UD-Q4_K_XL", False, False)["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertNotIn("STRATA_ROUTE_TAIL_SKIP", f("unsloth", "UD-Q4_K_XL", False, True))   # budget mode keeps the skip

    def test_other_families_are_untouched(self):
        for fam, m in (("qwen", "IQ3_S"), ("swift", "Q2_0"), ("coder", "IQ1_M"), ("nvfp4", "NVFP4")):
            self.assertEqual(setup.unsloth_default_env(fam, m, False, False), {}, fam)


class EnginePlaceholders(unittest.TestCase):
    def test_every_engine_minimum_is_this_release(self):
        self.assertEqual(setup.NVFP4_ENGINE, (0, 1, 44))
        self.assertEqual(setup.MIN_ENGINE, (0, 1, 46))
        for m in ("UD-Q5_K_XL", "UD-Q6_K_XL", "NVFP4", "NVFP4-NVIDIA"):
            self.assertEqual(setup.MODELS[m]["engine"], (0, 1, 44), m)
        # No minimum may name an engine newer than this tree (a placeholder for a release that does not exist yet would make setup
        # ask for an engine nobody can download). The tree's version is read from CMakeLists.txt, so this does not go stale each release.
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        tree = tuple(int(n) for n in re.search(r"project\(strata VERSION ([0-9.]+)", text).group(1).split("."))
        for what, minimum in [("MIN_ENGINE", setup.MIN_ENGINE), ("NVFP4_ENGINE", setup.NVFP4_ENGINE)] + [(m, setup.MODELS[m]["engine"]) for m in setup.MODELS if "engine" in setup.MODELS[m]]:
            self.assertLessEqual(tuple(minimum), tree[:len(minimum)], f"{what} {minimum} is ahead of the tree {tree}")


class KqLibrary(unittest.TestCase):
    def test_a_zip_without_strata_kq_does_not_leave_an_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            eng, tmp = Path(d) / "engine", Path(d) / "new"
            eng.mkdir()
            tmp.mkdir()
            (eng / setup.EXE).write_bytes(b"old")
            (eng / setup.KQ_LIB).write_bytes(b"old kq")
            (tmp / setup.EXE).write_bytes(b"new")
            setup.install_unpacked(tmp, eng)
            self.assertEqual((eng / setup.EXE).read_bytes(), b"new")
            self.assertFalse((eng / setup.KQ_LIB).exists())
            self.assertEqual((eng / setup.PREVIOUS_ENGINE / setup.KQ_LIB).read_bytes(), b"old kq")   # a rollback gets it back

    def test_a_zip_with_strata_kq_replaces_it(self):
        with tempfile.TemporaryDirectory() as d:
            eng, tmp = Path(d) / "engine", Path(d) / "new"
            eng.mkdir()
            tmp.mkdir()
            (eng / setup.KQ_LIB).write_bytes(b"old kq")
            (tmp / setup.EXE).write_bytes(b"new")
            (tmp / setup.KQ_LIB).write_bytes(b"new kq")
            setup.install_unpacked(tmp, eng)
            self.assertEqual((eng / setup.KQ_LIB).read_bytes(), b"new kq")


class ImageEncoderDefault(unittest.TestCase):
    """#2406: "images: yes" gets the CPU encoder on an NVIDIA card under 24 GB (the GPU encoder cost 3-5% of text speed on
    16 GB cards, 35-60% on an RTX 4080 SUPER under Windows); --vision gpu is kept with a note."""

    def test_yes_follows_the_card(self):
        f = setup.flash_vision_encoder
        self.assertEqual(f("yes", {"vram_gb": 12.0}), "cpu")
        self.assertEqual(f("yes", {"vram_gb": 15.9}), "cpu")
        self.assertEqual(f("yes", {"vram_gb": 24.0}), "gpu")
        self.assertEqual(f("yes", {"vram_gb": 32.0}), "gpu")
        self.assertEqual(f("yes", {}), "gpu")                    # unknown VRAM: the old default
        self.assertEqual(f("cpu", {"vram_gb": 32.0}), "cpu")
        self.assertEqual(f("none", {"vram_gb": 16.0}), "none")

    def test_gpu_asked_is_kept_with_a_note(self):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(setup.flash_vision_encoder("gpu", {"vram_gb": 16.0}), "gpu")
            self.assertEqual(setup.flash_vision_encoder("gpu", {"vram_gb": 32.0}), "gpu")
        self.assertEqual(out.getvalue().count("kept as you chose"), 1)
        self.assertIn("#2406", out.getvalue())


if __name__ == "__main__":
    unittest.main()
