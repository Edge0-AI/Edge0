"""Tests for setup.py's NVFP4 family (ready-made packs, NVFP4_REPOS): every file fetched from its pinned revision and
checked against its size and SHA-256, a file both packs share linked instead of downloaded twice, a wrong file
refused, and the config's name read back as the same size.  A local file:// mirror - nothing is downloaded.

    python -m unittest tools.test_setup_nvfp4
"""
from __future__ import annotations

import contextlib
import hashlib
import io
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

A, B = "owner/pack-a", "owner/pack-b"
REV_A, REV_B = "a" * 40, "b" * 40


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class ReadyPacks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mirror = self.root / "mirror"
        self.models = self.root / "models"
        bodies = {
            (A, REV_A, "pack/index.txt"): b"index a", (A, REV_A, "pack/experts.bin"): b"experts a" * 50,
            (A, REV_A, "dense-a.gguf"): b"dense a", (A, REV_A, "ple-fp8.gguf"): b"the shared table" * 100,
            (B, REV_B, "pack/index.txt"): b"index b", (B, REV_B, "pack/experts.bin"): b"experts b" * 50,
            (B, REV_B, "dense-b.gguf"): b"dense b", (B, REV_B, "ple-fp8.gguf"): b"the shared table" * 100,
        }
        roles = {"pack/index.txt": "pack", "pack/experts.bin": "pack", "dense-a.gguf": "dense", "dense-b.gguf": "dense",
                 "ple-fp8.gguf": "ple"}
        repos = {A: {"revision": REV_A, "files": {}}, B: {"revision": REV_B, "files": {}}}
        for (repo, rev, path), body in bodies.items():
            f = self.mirror / repo / "resolve" / rev / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(body)
            repos[repo]["files"][path] = (roles[path], len(body), hashlib.sha256(body).hexdigest())
        repos[setup.NVFP4_FP4LOCAL_REPO] = setup.NVFP4_REPOS[setup.NVFP4_FP4LOCAL_REPO]   # listed, never fetched here
        models = {"NVFP4": {**setup.MODELS["NVFP4"], "repo": A}, "NVFP4-NVIDIA": {**setup.MODELS["NVFP4-NVIDIA"], "repo": B}}
        self.patches = [mock.patch.object(setup, "NVFP4_REPOS", repos),
                        mock.patch.dict(setup.MODELS, models),
                        mock.patch.dict(os.environ, {"HF_ENDPOINT": "file://" + self.mirror.as_posix()})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_fetch_checks_and_links_the_shared_file(self):
        quiet(setup.ready_fetch, "NVFP4", self.models)
        for f in setup.ready_files("NVFP4"):
            self.assertTrue(setup.ready_verified(setup.ready_where(self.models, f), f[5]), f[3])
        quiet(setup.ready_fetch, "NVFP4-NVIDIA", self.models)
        a = self.models / "owner--pack-a" / "ple-fp8.gguf"
        b = self.models / "owner--pack-b" / "ple-fp8.gguf"
        self.assertTrue(setup.ready_verified(b, setup.NVFP4_REPOS[B]["files"]["ple-fp8.gguf"][2]))
        self.assertTrue(os.path.samefile(a, b), "the shared table is linked, not a second copy")
        p = setup.ready_paths("NVFP4-NVIDIA", self.models)
        self.assertEqual(p["pack"], self.models / "owner--pack-b" / "pack")
        self.assertEqual(p["dense"].name, "dense-b.gguf")
        self.assertEqual(p["ple"], b)

    def test_a_second_run_downloads_nothing(self):
        quiet(setup.ready_fetch, "NVFP4", self.models)
        with mock.patch.object(setup, "download", side_effect=AssertionError("downloaded again")):
            quiet(setup.ready_fetch, "NVFP4", self.models)

    def test_a_wrong_file_is_refused_and_deleted(self):
        bad = self.mirror / A / "resolve" / REV_A / "dense-a.gguf"
        bad.write_bytes(b"dense X")                    # same size, other bytes
        with self.assertRaises(SystemExit):
            quiet(setup.ready_fetch, "NVFP4", self.models)
        self.assertFalse((self.models / "owner--pack-a" / "dense-a.gguf").exists())

    def test_config_name_reads_back_as_the_size(self):
        for model in ("NVFP4", "NVFP4-NVIDIA"):
            cfg = self.root / f"strata-{(setup.FAMILIES['nvfp4']['tag'] + model).lower()}.json"
            cfg.write_text('{"args": ["--max-context", "32768"]}', encoding="utf-8")
            ch = setup.choices_from_config(cfg)
            self.assertEqual((ch["family"], ch["model"]), ("nvfp4", model))


class Family(unittest.TestCase):
    def test_sizes_and_family(self):
        self.assertTrue(setup.FAMILIES["nvfp4"]["ready"])
        names = [m for m in setup.MODELS if "nvfp4" in setup.MODELS[m].get("families", ())]
        self.assertEqual(names, ["NVFP4", "NVFP4-NVIDIA", "NVFP4-QAD"])   # the GPTQ pack first: the default size
        for m in names:
            d = setup.MODELS[m]
            self.assertTrue(d["nvidia_only"])
            self.assertIs(d["vision"], True)   # the shared encoder reads pictures for every NVFP4 model (QAD: dense.gguf is the text model)
            self.assertRegex(d["license"], "Licen[sc]e|Agreement")
            self.assertEqual(d["engine"], setup.FP4LOCAL_ENGINE if m == "NVFP4-QAD" else setup.NVFP4_ENGINE)

    def test_gguf_names_skip_the_ready_family(self):
        self.assertIsNone(setup.gguf_choice("pack-a.gguf"))
        self.assertEqual(setup.gguf_choice("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"), ("qwen", "IQ3_XXS"))


class PreparedModel(unittest.TestCase):
    """NVFP4-QAD (NVIDIA's FP4-Local): the checkpoint is downloaded pinned, the pack is prepared on this PC."""

    def test_the_checkpoint_files_are_pinned(self):
        files = setup.ready_files("NVFP4-QAD")
        names = {f[3] for f in files}
        self.assertEqual(sum(1 for n in names if n.startswith("tensors/model-")), 21)
        self.assertTrue({"config.json", "model.safetensors.index.json", "tokenizer.json"} <= names)
        self.assertEqual(sum(f[4] for f in files) / 1e9 // 1, setup.MODELS["NVFP4-QAD"]["download_gb"] // 1)
        for f in files:
            self.assertEqual(f[0], "src")
            self.assertRegex(f[5], "^[0-9a-f]{64}$")
            self.assertEqual(f[2], "b9ce18a6887e18c089fdea8509a33e63429a3300")

    def test_paths_are_the_prepared_files(self):
        m = Path("/models")
        p = setup.ready_paths("NVFP4-QAD", m)
        d = m / "nvidia--Qwen3.8-Flash-Next-FP4-Local" / "prepared"
        self.assertEqual(p, {"pack": d / "pack", "dense": d / "dense.gguf", "ple": d / "ple-nvfp4.gguf", "mtp": d / "mtp"})
        self.assertNotIn("embd", p)

    def test_prepared_done_needs_every_mark(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(setup.prepared_done("NVFP4-QAD", Path(t)))
            d = setup.prepared_dir("NVFP4-QAD", Path(t))
            d.mkdir(parents=True)
            marks = ("ple-nvfp4.gguf.done", "dense.gguf.done", "pack.done", "draft.done")
            for m in marks[:-1]:
                (d / m).write_text("x")
                self.assertFalse(setup.prepared_done("NVFP4-QAD", Path(t)))
            (d / marks[-1]).write_text("x")
            self.assertTrue(setup.prepared_done("NVFP4-QAD", Path(t)))


class PreparedWithoutCheckpoint(unittest.TestCase):
    """A prepared NVFP4-QAD model needs none of the checkpoint's 95 GB of safetensors: the engine reads the prepared files."""

    def prepare(self, models: Path) -> None:
        d = setup.prepared_dir("NVFP4-QAD", models)
        d.mkdir(parents=True)
        for m in ("ple-nvfp4.gguf.done", "dense.gguf.done", "pack.done", "draft.done"):
            (d / m).write_text("x")

    def write_small_files(self, models: Path) -> None:
        for f in setup.ready_files("NVFP4-QAD"):
            if f[3].startswith("tensors/"):
                continue
            dst = setup.ready_where(models, f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"x")
            dst.with_name(dst.name + ".done").write_text(f"sha256 {f[5]}")

    def test_prepared_model_does_not_need_the_tensors(self):
        with tempfile.TemporaryDirectory() as t:
            models = Path(t)
            self.prepare(models)
            needed = setup.ready_needed("NVFP4-QAD", models)
            self.assertTrue(needed)
            self.assertFalse([f for f in needed if f[3].startswith("tensors/")])
            self.assertIn("config.json", {f[3] for f in needed})

    def test_unprepared_model_still_needs_the_tensors(self):
        with tempfile.TemporaryDirectory() as t:
            needed = setup.ready_needed("NVFP4-QAD", Path(t))
            self.assertEqual(len(needed), len(setup.ready_files("NVFP4-QAD")))
            self.assertEqual(sum(1 for f in needed if f[3].startswith("tensors/model-")), 21)

    def test_a_model_that_is_not_prepared_by_this_pc_keeps_all_its_files(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(len(setup.ready_needed("NVFP4", Path(t))), len(setup.ready_files("NVFP4")))

    def test_ready_fetch_downloads_nothing_for_a_prepared_model_without_tensors(self):
        with tempfile.TemporaryDirectory() as t:
            models = Path(t)
            self.prepare(models)
            self.write_small_files(models)
            with mock.patch.object(setup, "download", side_effect=AssertionError("downloaded")) as dl:
                setup.ready_fetch("NVFP4-QAD", models)
            dl.assert_not_called()

    def test_ready_fetch_wants_the_tensors_of_an_unprepared_model(self):
        with tempfile.TemporaryDirectory() as t:
            models = Path(t)
            self.write_small_files(models)
            with mock.patch.object(setup, "download", side_effect=RuntimeError("would download")):
                with self.assertRaises(RuntimeError):
                    setup.ready_fetch("NVFP4-QAD", models)


class MockedInstall(unittest.TestCase):
    """setup.main() --family nvfp4 --yes on mocked PCs (test_setup_golden's harness): the run config's model flags."""

    def run_install(self, prof, model=None, extra_argv=()):
        import test_setup_golden as G
        ram, found = G.PROFILES[prof]
        return G.install(ram, found, ["--family", "nvfp4", "--no-start", *(["--model", model] if model else []),
                                      *extra_argv])

    def flag(self, args, name):
        return args[args.index(name) + 1] if name in args else None

    def test_recommended_size_on_128gb(self):
        code, out, cfg, _ = self.run_install("128GB-1x24GB")
        self.assertEqual(code, 0, out[-2000:])
        a = cfg["args"]
        self.assertTrue(self.flag(a, "--pack").endswith("Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata/pack"))
        dense = self.flag(a, "--native")
        self.assertTrue(dense.endswith("/qwen-nvfp4-gptq-dense.gguf"))
        self.assertEqual(self.flag(a, "--native-dense-gguf"), dense)
        self.assertTrue(self.flag(a, "--ple-gguf").endswith("/ple-fp8.gguf"))
        self.assertTrue(self.flag(a, "--embd-gguf").endswith("/token-embd-bf16.gguf"))
        self.assertNotIn("--mmap-experts", a)              # 128 GB: every expert in RAM
        self.assertNotIn("--resident-experts", a)
        self.assertIn("vision", cfg)                       # --yes: pictures on, as for every model that can read them
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-nvfp4-nvfp4")
        self.assertIn("Qwen Community License", out)

    def test_images_use_the_dense_gguf_and_the_pinned_encoder(self):
        code, out, cfg, _ = self.run_install("128GB-1x24GB", extra_argv=["--vision", "gpu"])
        self.assertEqual(code, 0, out[-2000:])
        v = cfg["vision"]
        self.assertTrue(v["model"].endswith("/qwen-nvfp4-gptq-dense.gguf"))
        self.assertTrue(v["mmproj"].endswith("/mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
        self.assertTrue(v["gpu"])
        self.assertIn("--vision", cfg["args"])

    def test_nvidia_size_and_its_license(self):
        code, out, cfg, _ = self.run_install("128GB-1x24GB", "NVFP4-NVIDIA")
        self.assertEqual(code, 0, out[-2000:])
        self.assertIn("NVIDIA-NVFP4-Strata/pack", self.flag(cfg["args"], "--pack"))
        self.assertIn("NVIDIA Open Model License", out)

    def test_qad_size_is_prepared_and_uses_its_own_files(self):
        code, out, cfg, _ = self.run_install("128GB-1x24GB", "NVFP4-QAD")
        self.assertEqual(code, 0, out[-2500:])
        a = cfg["args"]
        base = "nvidia--Qwen3.8-Flash-Next-FP4-Local/prepared"
        self.assertTrue(self.flag(a, "--pack").endswith(base + "/pack"))
        self.assertTrue(self.flag(a, "--native").endswith(base + "/dense.gguf"))
        self.assertEqual(self.flag(a, "--native-dense-gguf"), self.flag(a, "--native"))
        self.assertTrue(self.flag(a, "--ple-gguf").endswith(base + "/ple-nvfp4.gguf"))
        self.assertNotIn("--embd-gguf", a)                  # its dense GGUF holds the token embedding
        self.assertTrue(self.flag(a, "--mtp").endswith(base + "/mtp/rt"))   # NVIDIA's own draft layer, not Qwen's
        self.assertIn("NVIDIA Open Model Agreement", out)
        self.assertIn("turned into a Strata pack on this PC", out)

    def test_qad_reads_pictures_with_its_prepared_dense_gguf(self):
        code, out, cfg, _ = self.run_install("128GB-1x24GB", "NVFP4-QAD", extra_argv=["--vision", "gpu"])
        self.assertEqual(code, 0, out[-2500:])
        v = cfg["vision"]
        self.assertTrue(v["model"].endswith("nvidia--Qwen3.8-Flash-Next-FP4-Local/prepared/dense.gguf"))
        self.assertTrue(v["mmproj"].endswith("/mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
        self.assertIn("--vision", cfg["args"])
        self.assertNotIn("images are not available", out)

    def test_96gb_keeps_every_expert_in_ram_64gb_takes_the_ram_budget(self):
        code, out, cfg, _ = self.run_install("96GB-1x16GB")    # 68 GB of experts + 10 GB beside them fit 95.8
        self.assertEqual(code, 0, out[-2000:])
        self.assertFalse({"--mmap-experts", "--resident-experts"} & set(cfg["args"]))
        code, out, cfg, _ = self.run_install("64GB-1x32GB")    # 64 GB: the RAM-budget mode (RAM less 24 GB)
        self.assertEqual(code, 0, out[-2000:])
        a = cfg["args"]
        self.assertFalse({"--mmap-experts", "--resident-experts"} & set(a))
        self.assertEqual(a[a.index("--resident-budget-gib") + 1], "40")

    def test_amd_is_refused_with_a_recommendation(self):
        import test_setup_golden as G
        ram, _ = G.PROFILES["128GB-1x24GB"]
        amd = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201"}]
        code, out, cfg, _ = G.install(ram, [], ["--family", "nvfp4", "--no-start", "--backend", "hip"],
                                      extra=[mock.patch.object(setup, "amd_gpus", lambda *a, **k: amd)])
        self.assertNotEqual(code, 0)
        self.assertIn("UD-IQ4_XS", out)


if __name__ == "__main__":
    unittest.main()
