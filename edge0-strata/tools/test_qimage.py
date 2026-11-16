"""Tests for image generation's setup side (tools/image/qimage.py, setup.py --images): the pins, the quant and backend
choices, the CMake definitions, the source unpacking and the run config's "images" section.  No GPU, no network.

    python -m unittest tools.test_qimage
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "image"))
import qimage as QI  # noqa: E402
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Pins(unittest.TestCase):
    def test_shapes(self):
        for c in (QI.SD_CPP_COMMIT, QI.SD_GGML_COMMIT, *QI.REVISIONS.values()):
            self.assertRegex(c, r"^[0-9a-f]{40}$")
        for name, (repo, path, size, sha) in QI.FILES.items():
            self.assertIn(repo, QI.REVISIONS, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertGreater(size, 100 << 20)
        self.assertEqual(QI.file_url("Q8_0"), "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/"
                         "5d1f9f759ea877ae556cda88dbec958b62ad3809/qwen-image-2.1-turbo-Q8_0.gguf")
        self.assertEqual(QI.local_path(Path("m"), "vae"),
                         Path("m") / "qwen-image-2.1-turbo" / "qwen_image_2.1_vae_bf16.safetensors")

    def test_sigmas_are_the_models(self):
        """serve/images.py's schedule is model_index.json's sample_sigmas plus the terminal 0."""
        from serve import images
        sample = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        self.assertEqual([float(x) for x in images.TURBO_SIGMAS.split(",")], sample + [0.0])


class Choices(unittest.TestCase):
    def test_quant_by_memory(self):
        self.assertEqual([QI.pick_quant(g) for g in (4, 8, 8.5, 12, 12.1, 16, 32)],
                         ["Q4_K_M", "Q4_K_M", "Q6_K", "Q6_K", "Q8_0", "Q8_0", "Q8_0"])

    def test_backend(self):
        self.assertEqual(QI.pick_backend(None, False), "cpu")
        self.assertEqual(QI.pick_backend({"vendor": "nvidia"}, True), "cuda")
        self.assertEqual(QI.pick_backend({"vendor": "amd"}, False), "hip")
        self.assertEqual(QI.pick_backend({"vendor": "amd", "uma": True}, False), "vulkan")    # the 780M, Strix Halo
        self.assertEqual(QI.pick_backend({"vendor": "amd"}, True), "vulkan")                  # AMD on Windows
        self.assertEqual(QI.pick_backend({"vendor": "intel"}, False), "vulkan")
        self.assertEqual(QI.pick_backend({"vendor": "amd", "arch": "gfx1103"}, False), "vulkan")   # no uma mark
        self.assertEqual(QI.pick_backend({"vendor": "amd", "arch": "gfx1100"}, False), "hip")
        self.assertTrue(QI.is_apu({"arch": "gfx1151:xnack-"}))

    def test_default_args(self):
        self.assertEqual(QI.default_args("cuda"), ["--diffusion-fa"])
        self.assertEqual(QI.default_args("vulkan"), ["--diffusion-fa", "--vae-tiling", "--vae-tile-overlap", "0.25"])

    def test_setup_recommends_never_forces(self):
        ns = argparse.Namespace(image_backend=None, image_quant=None)
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 12.0, "arch": 86}, False)[:2],
                         ("cuda", "Q6_K"))
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 32.3, "uma": True, "arch": "gfx1103"},
                               True)[:2], ("vulkan", "Q8_0"))
        ns.image_quant = "Q8_0"                      # more than recommended: kept, with a warning
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = setup.image_backend_choice(ns, {"vram_gb": 8.0, "arch": 86}, False)
        self.assertEqual(r[:2], ("cuda", "Q8_0"))
        self.assertIn("recommends Q4_K_M", out.getvalue())
        ns.image_quant, ns.image_backend = None, "cpu"
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 24.0}, False)[:2], ("cpu", "Q4_K_M"))


class Build(unittest.TestCase):
    def test_cmake_defs(self):
        d = QI.cmake_defs("cuda", arch=86, cuda_root=Path("/usr/local/cuda"))
        self.assertIn("-DSD_CUDA=ON", d)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", d)
        self.assertIn("-DSD_WEBP=OFF", d)
        self.assertIn("-DSD_VULKAN=ON", QI.cmake_defs("vulkan"))
        with tempfile.TemporaryDirectory() as r:
            rocm = Path(r)
            (rocm / "lib" / "llvm" / "bin").mkdir(parents=True)
            d = QI.cmake_defs("hip", arch="gfx1103:sramecc-:xnack-", rocm=rocm)
            self.assertIn("-DGPU_TARGETS=gfx1103", d)
            self.assertIn(f"-DCMAKE_HIP_COMPILER={rocm / 'lib' / 'llvm' / 'bin' / 'clang++'}", d)
        with self.assertRaises(ValueError):
            QI.cmake_defs("hip", arch="gfx1103")
        with self.assertRaises(ValueError):
            QI.cmake_defs("metal")

    def test_unpack_puts_ggml_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z1, z2 = d / "sd.zip", d / "ggml.zip"
            with zipfile.ZipFile(z1, "w") as z:
                z.writestr(f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT}/CMakeLists.txt", "project(sd)")
                z.writestr(f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT}/ggml/", "")
            with zipfile.ZipFile(z2, "w") as z:
                z.writestr(f"ggml-{QI.SD_GGML_COMMIT}/CMakeLists.txt", "project(ggml)")
            src = QI.unpack_source(d, z1, z2)
            self.assertEqual((src / "ggml" / "CMakeLists.txt").read_text(), "project(ggml)")
            self.assertTrue(QI.source_ok(src))
            (src / ".strata-pin").write_text("0" * 40 + "\n" + QI.SD_GGML_COMMIT + "\n")
            self.assertFalse(QI.source_ok(src))          # another pin: unpacked again

    def test_build_is_skipped_when_the_stamp_matches(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "IMAGE_ENGINE", Path(d)):
            (Path(d) / QI.SD_EXE).write_bytes(b"x")
            (Path(d) / "BUILD.json").write_text(json.dumps({**QI.build_stamp("cuda", 86), "lib_dirs": [d]}))
            with mock.patch.object(setup, "cmake_build") as cb:
                exe, libs = quiet(setup.build_image_engine, "cuda", {"arch": 86})
            cb.assert_not_called()
            self.assertEqual((exe, libs), (Path(d) / QI.SD_EXE, [d]))
            with mock.patch.object(setup, "cmake_build") as cb, mock.patch.object(setup, "get_sd_cpp"), \
                    mock.patch.object(setup, "apply_native_runner"), \
                    mock.patch.object(setup, "find_nvcc", return_value=(None, None)):
                with self.assertRaises(SystemExit):     # another arch: compiled again (here: no nvcc, so it stops)
                    quiet(setup.build_image_engine, "cuda", {"arch": 89})


class Config(unittest.TestCase):
    def test_section(self):
        c = QI.image_config("vulkan", "Q8_0", Path("/e/sd-cli"), Path("/m"), uma=True, lib_dirs=["/l"])
        self.assertEqual(c["diffusion_model"], str(Path("/m/qwen-image-2.1-turbo/qwen-image-2.1-turbo-Q8_0.gguf")))
        self.assertEqual(c["args"], ["--diffusion-fa", "--vae-tiling", "--vae-tile-overlap", "0.25"])
        self.assertTrue(c["uma"])
        self.assertEqual(c["gpu_policy"], "auto")
        from serve.images import ImageGen
        with tempfile.TemporaryDirectory() as d:
            ImageGen({**c, "out_dir": d})              # serve accepts what setup writes

    def test_images_no_drops_the_section(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": "x", "images": {"exe": "sd-cli"}, "mine": 1}))
            quiet(setup.write_setup_config, p, {"exe": "y", "images": None})
            cfg = json.loads(p.read_text())
            self.assertNotIn("images", cfg)
            self.assertEqual(cfg["mine"], 1)
            p.write_text(json.dumps({"exe": "x", "images": {"exe": "sd-cli"}}))
            quiet(setup.write_setup_config, p, {"exe": "y"})      # not asked: carried over
            self.assertEqual(json.loads(p.read_text())["images"], {"exe": "sd-cli"})

    def test_flags(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        for flag in ("--images", "--image-quant", "--image-backend", "--image-gpu-policy"):
            self.assertRegex(src, re.escape(f'ap.add_argument("{flag}"'))


class Native(unittest.TestCase):
    """I2: the native DiT runner's patch for the pinned stable-diffusion.cpp, and its config switch."""

    def test_stamp_and_config(self):
        self.assertIn("native", QI.build_stamp("cuda", 86))
        self.assertIn("native", QI.build_stamp("hip", "gfx1103"))
        self.assertNotIn("native", QI.build_stamp("vulkan", None))
        c = QI.image_config("hip", "Q8_0", Path("/e/sd-cli"), Path("/m"), native=True)
        self.assertIs(c["native"], True)
        self.assertNotIn("native", QI.image_config("vulkan", "Q8_0", Path("/e/sd-cli"), Path("/m")))

    def test_apply_is_idempotent(self):
        sys.path.insert(0, str(QI.NATIVE_DIR))
        import apply as A
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d)
            (sd / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
            (sd / "src" / "model" / "diffusion").mkdir(parents=True)
            (sd / "src" / "pipeline").mkdir(parents=True)
            (sd / "src" / "pipeline" / "model_builders.cpp").write_text(
                "#include <x>\n" + A.INCLUDE_OLD + "int a;\n" + A.BUILDER_OLD + "            }\n", encoding="utf-8")
            (sd / "CMakeLists.txt").write_text("x\n" + A.CMAKE_OLD + "y\n", encoding="utf-8")
            A.apply(sd)
            first = [(sd / f).read_text(encoding="utf-8") for f in ("src/pipeline/model_builders.cpp", "CMakeLists.txt")]
            A.apply(sd)
            again = [(sd / f).read_text(encoding="utf-8") for f in ("src/pipeline/model_builders.cpp", "CMakeLists.txt")]
            self.assertEqual(first, again)
            self.assertIn("QwenImage21NativeRunner", first[0])
            self.assertEqual(first[1].count("SD_QI_NATIVE"), 1)
            self.assertTrue((sd / "ggml/src/ggml-cuda/qi21-native.cu").is_file())
            (sd / "CMakeLists.txt").write_text("something else\n", encoding="utf-8")
            with self.assertRaises(SystemExit):        # another sd.cpp: refused, not half-patched
                A.apply(sd)


if __name__ == "__main__":
    unittest.main()
