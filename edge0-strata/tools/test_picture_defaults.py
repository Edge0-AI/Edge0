"""Owner rule 2026-10-18: pictures are on by default wherever the model and the engine package can read them; the reader runs on the
CPU under 24 GB of VRAM and on the GPU from 24 GB, on every make of card.  Checks setup.images_default and the shared rule.

    python -m unittest tools.test_picture_defaults
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import vision_reader as VR  # noqa: E402


class PictureDefaults(unittest.TestCase):
    def test_unsloth_default_is_on_for_every_make_and_size(self):
        models = [m for m, d in setup.MODELS.items() if d.get("budget") and d.get("vision") is not False]
        self.assertTrue(models)
        for m in models:
            for gpu in ({"vram_gb": 10.0}, {"vram_gb": 12.0}, {"vram_gb": 16.0}, {"vram_gb": 32.0},
                        {"vram_gb": 16.0, "arch": "gfx1201"}, {"vram_gb": 24.0, "arch": "gfx1100"}, {}):
                for win in (False, True):
                    old = setup.WIN
                    setup.WIN = win
                    try:
                        self.assertTrue(setup.images_default(m, "unsloth", 64.0, gpu), (m, gpu, win))
                        self.assertTrue(setup.images_default(m, "unsloth", 512.0, gpu), (m, gpu, win))
                    finally:
                        setup.WIN = old

    def test_yes_means_pictures_on_for_qwen_swift_and_coder(self):
        """setup --yes (and Enter at the question) on the original Flash-Next, Swift and Coder files: pictures on, the reader by the 24 GB rule,
        and no VRAM reserve for the CPU reader."""
        from test_setup_golden import PROFILES, install
        for key, gpu in (("64GB-1x32GB", True), ("96GB-1x16GB", False)):          # 31.8 GB: the GPU reader; 15.9 GB: the CPU reader
            ram, found = PROFILES[key]
            for fam, model in (("qwen", "Q2_0"), ("swift", "IQ3_S"), ("coder", "IQ1_M")):
                for answers in (None, ""):
                    code, out, cfg, _ = install(ram, found, ["--family", fam, "--model", model, "--no-start"], answers=answers)
                    where = (key, fam, model, answers)
                    self.assertEqual(code, 0, (where, out[-1500:]))
                    self.assertIn("--vision", cfg["args"], where)
                    self.assertEqual(cfg["vision"]["gpu"], gpu, where)
                    self.assertEqual("--vram-reserve-mib" in cfg["args"], gpu, where)       # only the GPU reader keeps VRAM free
                    self.assertIn("images: on", out, where)

    def test_pictures_off_stays_available(self):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["64GB-1x32GB"]
        for fam, model in (("qwen", "Q2_0"), ("swift", "IQ3_S"), ("coder", "IQ1_M")):
            code, out, cfg, _ = install(ram, found, ["--family", fam, "--model", model, "--no-start", "--vision", "no"])
            self.assertEqual(code, 0, out[-1500:])
            self.assertNotIn("--vision", cfg["args"])
            self.assertNotIn("vision", cfg)

    def test_the_cpu_reader_takes_no_vram_reserve(self):
        """So a card of 8 GB is no exception: the reserve is only for the GPU reader."""
        self.assertEqual(setup.VISION["cpu"]["reserve_mib"], 0)
        self.assertEqual(setup.VISION["gpu"]["reserve_mib"], 700)

    def test_one_rule_for_the_reader(self):
        self.assertEqual(setup.IMAGES_GPU_ENCODER_MIN_VRAM_GB, VR.GPU_MIN_VRAM_GB)
        for vram, want in ((8, "cpu"), (16, "cpu"), (23.0, "cpu"), (24, "gpu"), (32, "gpu"), (None, "gpu")):
            self.assertEqual(VR.default_mode(vram), want)
            self.assertEqual(VR.wanted("yes", vram), want)
            self.assertEqual(setup.hip_vision("yes", vram or 0.0), want if vram else "gpu")
        self.assertEqual(setup.hip_vision(None, 16.0), "cpu")
        self.assertEqual(setup.hip_vision("gpu", 16.0), "gpu")

class MacPictureDefaults(unittest.TestCase):
    """setup.py on a Mac: pictures on by default, on the Metal reader (strata_control/vision_reader.py)."""

    def test_nothing_asked_means_the_gpu_reader_for_every_mac_size(self):
        from strata_control import mac_fit as MF
        for model in MF.MAC_MODELS:
            for ram in (32.0, 64.0, 128.0):
                self.assertEqual(setup.mac_vision(None, model, ram), "gpu", (model, ram))
                self.assertEqual(setup.mac_vision("yes", model, ram), "gpu", (model, ram))

    def test_off_and_cpu_are_kept_as_asked(self):
        for off in ("no", "none", "off"):
            self.assertEqual(setup.mac_vision(off, "Q2_0", 32.0), "none")
        self.assertEqual(setup.mac_vision("cpu", "Q2_0", 32.0), "cpu")

    def test_a_folder_without_the_reader_turns_pictures_off(self):
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="macvis-"))
        (d / "strata").write_bytes(b"x")
        self.assertEqual(setup.mac_reader(d, "gpu").mode, "none")
        (d / "strata-vision").write_bytes(b"x")
        r = setup.mac_reader(d, "gpu")
        self.assertEqual((r.mode, r.exe.name), ("gpu", "strata-vision"))


if __name__ == "__main__":
    unittest.main()
