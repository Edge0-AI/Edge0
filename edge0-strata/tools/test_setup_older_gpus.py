"""Older NVIDIA GPUs (experimental): Pascal / Volta run a second engine built with CUDA 12 (engine-cuda12/), chosen per
model config by its oldest card; --cuda 12|13 overrides it.  A PC whose cards the ready-made CUDA 13 engine runs on
gets exactly what it got before.  No GPU, no downloads.

    python -m unittest tools.test_setup_older_gpus
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup  # noqa: E402
from test_setup_choices import quiet  # noqa: E402
from test_setup_golden import PROFILES, install, argv_for  # noqa: E402


def card(i, name, vram, arch, driver="580.97"):
    return {"index": i, "name": name, "vram_gb": vram, "arch": arch, "driver": driver}


V100 = card(0, "Tesla V100-PCIE-32GB", 32.0, "70", "575.57")
P40 = card(0, "Tesla P40", 24.0, "61", "535.104")


class Choice(unittest.TestCase):
    def test_by_the_oldest_card(self):
        self.assertEqual(setup.cuda_choice(["120"]), (13, None))
        self.assertEqual(setup.cuda_choice([75, 86, 89]), (13, None))
        for archs in ([70], [61], [60], [70, 86]):
            tk, why = setup.cuda_choice(archs)
            self.assertEqual(tk, 12, archs)
            self.assertIn(f"sm_{min(archs)} is older than CUDA 13 supports", why)

    def test_a_driver_too_old_for_cuda13_gets_the_cuda12_engine(self):       # #3843
        self.assertEqual(setup.cuda_choice([86], None, 576)[0], 12)
        self.assertIn("older than CUDA 13 needs", setup.cuda_choice([86], None, 576)[1])
        self.assertEqual(setup.cuda_choice([86], None, 580), (13, None))
        self.assertEqual(setup.cuda_choice([86], None, 0), (13, None))          # driver unknown: as before
        self.assertEqual(setup.cuda_choice([86], None, 500), (13, None))        # too old for both: setup stops with the driver message
        self.assertEqual(setup.cuda_choice([86], "13", 576)[0], 13)             # chosen by hand: kept

    def test_the_override_is_honored_with_a_warning(self):
        tk, why = setup.cuda_choice([70], "13")
        self.assertEqual(tk, 13)                       # recommend, never force: kept, and said
        self.assertIn("will not run on that card", why)
        self.assertEqual(setup.cuda_choice([86], "12")[0], 12)
        self.assertIn("#220", setup.cuda_choice([86, 120], "12")[1])
        self.assertEqual(setup.cuda_choice([70], "auto")[0], 12)
        self.assertEqual(setup.cuda_choice([86], "auto"), (13, None))

    def test_engine_folders_and_configs(self):
        with mock.patch.object(setup, "ROOT", Path("/s")):
            self.assertEqual(setup.engine_dir(), Path("/s/engine"))
            self.assertEqual(setup.engine_dir(12), Path("/s/engine-cuda12"))
        self.assertEqual(setup.config_toolkit({"exe": "/s/engine/strata.exe"}), 13)
        self.assertEqual(setup.config_toolkit({"exe": "/s/engine-cuda12/strata.exe"}), 12)
        self.assertEqual(setup.config_toolkit({"exe": "x", "cuda": 12}), 12)
        self.assertEqual(setup.engine_defs([86], 12), ["-DSTRATA_EXPERIMENTAL_SM60=ON"])
        self.assertEqual(setup.engine_defs([86]), [])


class OptIn(unittest.TestCase):
    """Pascal / Volta cards are used only when chosen; a PC with a newer card keeps recommending the newer one."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_reasons(self):
        rtx = card(0, "RTX 3090", 24.0, "86")
        v100 = {**V100, "index": 1}
        self.assertIsNone(setup.old_gpus_opt_in([rtx]))
        self.assertIsNone(setup.old_gpus_opt_in([rtx, v100]))                  # the default: unchanged
        self.assertIn("you chose GPU 1", setup.old_gpus_opt_in([rtx, v100], [1]))
        self.assertIsNone(setup.old_gpus_opt_in([rtx, v100], [0]))
        self.assertEqual(setup.old_gpus_opt_in([rtx, v100], cuda="12"), "--cuda 12")
        self.assertIn("only kind", setup.old_gpus_opt_in([V100]))
        self.assertIsNone(setup.old_gpus_opt_in([V100], other=True))            # an AMD card it can use instead
        self.assertIsNone(setup.old_gpus_opt_in([card(0, "GTX 980", 4.0, "52")]))
        # #3218: a Maxwell card is never routed to the CUDA 12 engine (it holds no sm_52 code), with or without a switch
        m6000 = card(0, "Quadro M6000", 24.0, "52")
        self.assertTrue(setup.sm60_card("60") and setup.sm60_card("70") and not setup.sm60_card("75"))
        self.assertFalse(setup.sm60_card("52"))
        self.assertIn("not supported", setup.gpu_problem(m6000))
        self.assertIn("Maxwell only runs from a source build", setup.gpu_problem(m6000))
        self.assertNotIn("experimental: choose it", setup.gpu_problem(m6000))
        self.assertIsNone(setup.old_gpus_opt_in([m6000]))
        for env in ({"STRATA_EXPERIMENTAL_SM60": "1"}, {"STRATA_EXPERIMENTAL_SM52": "1"}):
            with mock.patch.dict(os.environ, env):
                self.assertFalse(setup.sm60_card("52"))
                self.assertIn("not supported", setup.gpu_problem(m6000))
                self.assertIsNone(setup.old_gpus_opt_in([m6000]))
        self.assertIsNone(setup.old_gpus_opt_in([m6000], named=[0], cuda="12"))   # naming it does not admit it either
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "1"}):
            self.assertEqual(setup.old_gpus_opt_in([rtx, v100]), "STRATA_EXPERIMENTAL_SM60=1")

    def test_maxwell_note_is_a_warning_with_the_recipe(self):
        m6000 = card(0, "Quadro M6000", 24.0, "52")
        note = setup.maxwell_note([m6000, {**m6000, "index": 1}])
        self.assertIn("GPU 0 (Quadro M6000), GPU 1 (Quadro M6000)", note)
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=52", note)
        self.assertIn("docs/OLDER_GPUS.md", note)
        self.assertNotIn("STRATA_EXPERIMENTAL_SM52", note)                       # no switch the user has to set
        self.assertEqual(setup.maxwell_note([V100, P40]), "")
        self.assertEqual(setup.maxwell_note([]), "")

    def test_named(self):
        self.assertEqual(setup.named_gpus(None, "0,2"), [0, 2])
        self.assertEqual(setup.named_gpus(1, None), [1])
        self.assertEqual(setup.named_gpus(None, "all"), [])
        self.assertEqual(setup.named_gpus(None, "x"), [])

    def test_the_gate_follows_the_opt_in(self):
        with mock.patch.object(setup, "OLD_GPUS", None):
            self.assertIsNotNone(setup.gpu_problem(V100))
        with mock.patch.object(setup, "OLD_GPUS", "you chose GPU 0"):
            self.assertIsNone(setup.gpu_problem(V100))
            self.assertIsNotNone(setup.gpu_problem(card(0, "GTX 980", 4.0, "52")))


class Libraries(unittest.TestCase):
    def test_cuda12_wheels_two_folders(self):
        with tempfile.TemporaryDirectory() as d:
            sp = Path(d) / "site-packages"
            names = {"cublas/bin/cublas64_12.dll", "cuda_runtime/bin/cudart64_12.dll", "cu13/bin/x86_64/cublas64_13.dll",
                     "cublas/lib/libcublas.so.12", "cuda_runtime/lib/libcudart.so.12", "cu13/lib/libcublas.so.13"}
            for n in names:
                f = sp / "nvidia" / n
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"")
            with mock.patch.object(sys, "path", [str(sp)]):
                for win in (True, False):
                    with mock.patch.object(setup, "WIN", win):
                        got12 = sorted(Path(x).relative_to(sp / "nvidia").as_posix() for x in setup.cuda_lib_dirs(12))
                        got13 = [Path(x).relative_to(sp / "nvidia").as_posix() for x in setup.cuda_lib_dirs()]
                    if win:
                        self.assertEqual(got12, ["cublas/bin", "cuda_runtime/bin"])
                        self.assertEqual(got13, ["cu13/bin/x86_64"])
                    else:
                        self.assertEqual(got12, ["cublas/lib", "cuda_runtime/lib"])
                        self.assertEqual(got13, ["cu13/lib"])

    def test_wheels_pinned_to_12(self):
        self.assertTrue(all("-cu12==12." in w for w in setup.CUDA12_WHEELS))
        self.assertEqual(setup.CUDA12_ASSET, "strata-windows-x64-cuda12.zip" if setup.WIN else
                         "strata-linux-x64-cuda12.zip")


class BuildTools(unittest.TestCase):
    """The compile path: a CUDA 12 engine looks for a 12.x toolkit (STRATA_NVCC picks one), sm_120 needs 12.8+."""

    def tools(self, gpu, nvcc):
        seen = []

        def find(below=None):
            seen.append(below)
            return nvcc(below)

        with mock.patch.object(setup, "find_nvcc", find), mock.patch.object(setup, "find_vcvars", lambda cuda_v=None: "vcvars"), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n):
            got, text = quiet(setup.install_build_tools, gpu, True)
        return got, seen, text

    def test_cuda12_on_a_newer_card(self):
        got, seen, _ = self.tools({"arch": "86", "archs": [86], "toolkit": 12},
                                  lambda below: ("nvcc12", (12, 9)) if below else ("nvcc13", (13, 0)))
        self.assertEqual((got[0], seen), ("nvcc12", [(13, 0)]))

    def test_sm120_in_a_cuda12_engine_warns_and_needs_12_8(self):
        got, _, text = self.tools({"arch": "70", "archs": [70, 120]},
                                  lambda below: ("nvcc12", (12, 9)) if below else ("nvcc13", (13, 0)))
        self.assertEqual(got[0], "nvcc12")
        self.assertIn("#220", text)
        with self.assertRaises(SystemExit):
            self.tools({"arch": "70", "archs": [70, 120]}, lambda below: ("nvcc12", (12, 6)) if below else (None, None))


def fake_prebuilt(calls):
    """get_prebuilt that installs an engine in the folder of the toolkit it is asked for (and says which)."""
    def get(url_base, gpu, vision, updating=False, toolkit=13):
        calls.append(toolkit)
        eng = setup.engine_dir(toolkit)
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.39", "source": "release", "archs": [86], "ptx": True,
                                                    "cuda": "12.9" if toolkit == 12 else "13.0", "vision": "gpu"}))
        (eng / setup.EXE).write_bytes(b"")
        (eng / setup.VEXE).write_bytes(b"")                    # pictures are on by default: the engine carries the reader
        return eng
    return get


class Install(unittest.TestCase):
    """setup.main() on mocked PCs (tools/test_setup_golden.py's harness)."""

    def run_setup(self, ram, found, argv, env=None):
        calls, libs = [], []
        extra = [mock.patch.object(setup, "get_prebuilt", fake_prebuilt(calls)),
                 mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: libs.append(tk)),
                 mock.patch.object(setup, "OLD_GPUS", None),
                 mock.patch.object(setup, "cuda12_min_driver", lambda feed=False: (575, "575.51", "the published figure")),   # never the real feed
                 mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "", "STRATA_CUDA": "", **(env or {})})]
        code, out, cfg, asked = install(ram, found, argv, extra=extra)
        return code, out, cfg, calls, libs

    def test_a_v100_alone_gets_the_cuda12_engine(self):
        code, out, cfg, calls, libs = self.run_setup(95.0, [V100], argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((calls, libs), ([12], [12]))
        self.assertEqual(cfg["cuda"], 12)
        self.assertEqual(cfg["exe"], "<T>/engine-cuda12/<EXE>")
        self.assertIn("sm_70 is older than CUDA 13 supports", out)
        self.assertIn("experimental", out)

    def test_a_maxwell_pc_is_warned_with_the_recipe_and_gets_no_engine(self):
        """#3218: no ready-made engine holds sm_52 code, so none is installed; the way forward is printed, with or
        without any switch (a user never has to set one)."""
        m6000 = card(0, "Quadro M6000", 24.0, "52")
        for env in (None, {"STRATA_EXPERIMENTAL_SM60": "1"}, {"STRATA_EXPERIMENTAL_SM52": "1"}):
            with self.subTest(env=env):
                code, out, cfg, calls, libs = self.run_setup(95.0, [m6000], argv_for("qwen", "IQ3_XXS"), env)
                self.assertNotEqual(code, 0, out[-3000:])
                self.assertEqual((calls, libs), ([], []))
                self.assertIsNone(cfg)
                self.assertIn("Maxwell (compute capability 5.2)", out)
                self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=52", out)
                self.assertIn("docs/OLDER_GPUS.md", out)

    def test_a_maxwell_card_beside_a_newer_one_is_not_a_warning(self):
        rtx = card(0, "NVIDIA GeForce RTX 3090", 24.0, "86")
        code, out, cfg, calls, _ = self.run_setup(95.0, [rtx, card(1, "GTX 980", 4.0, "52")], argv_for("qwen", "IQ3_XXS"))
        self.assertEqual((code, calls), (0, [13]), out[-3000:])
        self.assertNotIn("Maxwell (compute capability", out)

    def test_a_p40_with_an_older_driver(self):
        code, out, cfg, calls, _ = self.run_setup(95.0, [P40], argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])                 # driver 535: enough for CUDA 12
        self.assertIn("575.51", out)                            # #4247: said with the package's own figure, and as a warning, not a stop
        self.assertNotIn("works with driver 528", out)
        self.assertEqual(calls, [12])

    def test_a_newer_card_beside_a_v100_is_unchanged(self):
        """The RTX 3090 is recommended as before; the V100 is listed with how to choose it."""
        rtx = card(0, "NVIDIA GeForce RTX 3090", 24.0, "86")
        found = [rtx, {**V100, "index": 1}]
        code, out, cfg, calls, libs = self.run_setup(95.0, found, argv_for("qwen", "IQ3_XXS"))
        code0, _, cfg0, _ = install(95.0, [rtx], argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((calls, libs), ([13], [13]))
        self.assertNotIn("cuda", cfg)
        self.assertEqual(cfg["exe"], "<T>/engine/<EXE>")
        self.assertIn("choose it with --gpu 1", out)
        skip = ("gpu", "gpus_asked", "lib_dirs")       # one card vs two; the fake engine's library folders
        self.assertEqual({k: v for k, v in cfg.items() if k not in skip}, {k: v for k, v in cfg0.items() if k not in skip})

    def test_naming_the_v100_beside_a_newer_card(self):
        found = [card(0, "NVIDIA GeForce RTX 3090", 24.0, "86"), {**V100, "index": 1}]
        code, out, cfg, calls, _ = self.run_setup(95.0, found, argv_for("qwen", "IQ3_XXS") + ["--gpu", "1"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(calls, [12])
        self.assertEqual((cfg["cuda"], cfg["gpu"]), (12, 1))
        self.assertIn("you chose GPU 1", out)

    def test_cuda12_asked_for_on_a_newer_card(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for argv, env in ((["--cuda", "12"], None), ([], {"STRATA_CUDA": "12"})):
            with self.subTest(argv=argv, env=env):
                code, out, cfg, calls, _ = self.run_setup(ram, found, argv_for("qwen", "IQ3_XXS") + argv, env)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual((calls, cfg["cuda"]), ([12], 12))
                self.assertIn("--cuda 12 (as you chose", out)

    def test_an_old_driver_gets_cuda12_by_itself(self):                     # #3843: said in one line, not a stop
        code, out, cfg, calls, _ = self.run_setup(127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89", "550.54")],
                                                  argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((calls, cfg["cuda"]), ([12], 12))
        self.assertIn("older than CUDA 13 needs (580): this model runs the CUDA 12 engine", out)
        code, out, _, _, _ = self.run_setup(127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89", "470.10")],
                                            argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 1)                                                # too old for both: the stop stays
        self.assertIn("is too old", out)
        self.assertNotIn("--cuda 12", out)
        code, out, cfg, calls, _ = self.run_setup(127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89", "550.54")],
                                                  argv_for("qwen", "IQ3_XXS") + ["--cuda", "12"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(calls, [12])

    def test_cuda13_forced_on_a_v100_is_said(self):
        code, out, cfg, calls, _ = self.run_setup(95.0, [V100], argv_for("qwen", "IQ3_XXS") + ["--cuda", "13"])
        self.assertIn("will not run on that card", out)


class Start(unittest.TestCase):
    """At a start: a Pascal / Volta card added to a CUDA 13 model moves that model to the CUDA 12 engine; updates
    look at both engine folders."""

    def test_an_old_card_added_moves_the_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg_path = root / "strata-iq3_xxs.json"
            cfg = {"exe": str(root / "engine" / setup.EXE), "args": [], "lib_dirs": ["<cu13>"]}
            calls = []
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "get_prebuilt", fake_prebuilt(calls)), \
                    mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None), \
                    mock.patch.object(setup, "cuda_lib_dirs", lambda tk=13: [f"<cu{tk}>"]), \
                    mock.patch.object(setup, "gpu_info", lambda i=None: None):
                got, text = quiet(setup.ensure_engine_for, [card(0, "RTX 3090", 24.0, "86"), {**V100, "index": 1}],
                                  cfg_path, dict(cfg), True)
            self.assertEqual(calls, [12])
            self.assertEqual(Path(got["exe"]).parent.name, "engine-cuda12")
            self.assertEqual((got["cuda"], got["lib_dirs"]), (12, ["<cu12>"]))
            self.assertIn("moves to the experimental CUDA 12 engine", text)
            self.assertEqual(json.loads(cfg_path.read_text())["cuda"], 12)

    def test_update_looks_at_both_engines(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            seen = []
            real = setup.update_installed_engine

            def spy(url_base, toolkit=None):
                if toolkit is None:
                    return real(url_base, toolkit)
                seen.append(toolkit)
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "update_installed_engine", spy):
                spy("u")
                self.assertEqual(seen, [13])
                (root / "engine-cuda12").mkdir()
                (root / "engine-cuda12" / "BUILD.json").write_text("{}")
                seen.clear()
                spy("u")
                self.assertEqual(seen, [13, 12])


if __name__ == "__main__":
    unittest.main()
