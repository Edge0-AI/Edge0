"""Tests for setup.py --backend sycl, the experimental Intel Arc engine (sycl/, PR #423): it warns, stops with a clear
message on Windows, and on Linux hands the run to sycl/setup_intel.py without its --backend flag. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_stops_with_the_docs_pointer(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, ("exit", 1))
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("docs/INTEL_ARC.md", out)
        call.assert_not_called()

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_the_parser_accepts_sycl(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"--backend", choices=\["cuda", "hip", "sycl"\]')

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_setup_intel_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870: setup.py grew write_run_script's fourth argument and the port's replacement still took three, so
        setup died with a TypeError after the download.  Every `S.<name> = <replacement>` in setup_intel.py must accept
        all the parameters of setup.<name> (or take *args / **kwargs)."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_intel's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_arc_pro_b60_is_known_under_both_pci_ids(self):
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn('"e211": ("Arc Pro B60", 24.0)', src)
        self.assertIn('"e221": ("Arc Pro B60", 24.0)', src)


class CMakeOption(unittest.TestCase):
    def test_sycl_is_off_by_default_and_exclusive(self):
        src = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
        self.assertIn('option(STRATA_ENABLE_SYCL "EXPERIMENTAL', src)
        self.assertRegex(src, r'option\(STRATA_ENABLE_SYCL "[^"]*" OFF\)')
        self.assertIn("add_subdirectory(sycl)", src)


class SyclVersion(unittest.TestCase):
    def setUp(self):
        from sycl import setup_intel
        self.mod = setup_intel

    def test_version_matches_the_source_and_satisfies_native_models(self):
        version = self.mod.sycl_version()
        self.assertNotEqual(version, "0")
        self.assertEqual(version, setup.source_version())
        numbers = tuple(int(x) for x in version.split("."))
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(model=model):
                self.assertGreaterEqual(numbers, setup.MODELS[model].get("engine", setup.UNSLOTH_ENGINE))

    def test_version_follows_a_source_hotfix_with_a_variable_sycl_project(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.41.2 LANGUAGES CXX)\n", encoding="utf-8")
            (root / "sycl").mkdir()
            (root / "sycl" / "CMakeLists.txt").write_text(
                "set(STRATA_SYCL_VERSION 0.1.40.2)\n"
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n", encoding="utf-8")
            with mock.patch.object(self.mod, "ROOT", root), mock.patch.object(setup, "ROOT", root):
                self.assertEqual(self.mod.sycl_version(), "0.1.41.2")


class IntelIds(unittest.TestCase):
    """sycl/setup_intel.py names a card by its PCI device id. The Arc Pro B60s measured on 2x B60 (`lspci -nn` 8086:e211,
    bmg-g21) must not fall through to "Intel GPU e211 (xe)" with a guessed VRAM size."""

    @staticmethod
    def table():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.INTEL_ARC

    def test_b60_e211_is_named_with_its_vram(self):
        self.assertEqual(self.table()["e211"], ("Arc Pro B60", 24.0))

    def test_the_ids_that_were_listed_before_are_kept(self):
        t = self.table()
        self.assertEqual(t["e223"][0], "Arc Pro B70")
        self.assertEqual(t["e221"][0], "Arc Pro B60")


class ToSycl(unittest.TestCase):
    """The config sycl/setup_intel.py writes: an explicit --vram-reserve-mib is kept, the B-series streams its experts,
    the A-series (i915) cannot (a single pinned host allocation above a few GB fails there) and loads them into RAM."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / "a.gguf"),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", *extra]}

    def test_b_series_streams_and_defaults_the_reserve(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("STRATA_VERIFY_NO_HOST", c.get("env", {}))

    def test_xe_ranks_the_experts_with_the_sycl_profile_and_i915_keeps_the_shipped_one(self):
        m = self.mod()
        shipped = str(m.MOUNT / "data" / "expert-profile.bin")
        other = str(m.MOUNT / "mine" / "learned.bin")
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile-sycl.bin"))
        c = m.to_sycl(self.cfg(m, ["--expert-profile", other]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("learned.bin"))   # the user's own file stays
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile.bin"))   # the CPU computes the misses there

    def test_an_explicit_reserve_is_kept_on_both_drivers(self):
        m = self.mod()
        for vram, drv in ((32.0, "xe"), (8.0, "i915")):
            c = m.to_sycl(self.cfg(m, ["--vram-reserve-mib", "500"]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, vram, drv)
            self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")

    def test_a_series_loads_experts_into_ram_and_leaves_no_host_off(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")
        self.assertEqual(c["env"]["STRATA_SYCL_BIN"], "build-sycl/strata")

    def test_gyro_memory_warning_reads_the_real_ram(self):
        # setup's RAM rule is fed 1024 GB on the Intel path; Gyro counts VRAM + RAM, so it must see the real RAM
        m = self.mod()
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_for_test", ROOT / "setup.py")
        s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s)
        have = m.real_mem_have(s.mem_have, 16.0, 1024.0)
        self.assertEqual(have("Gyro-S", 1024.0, 8.0), 24.0)              # 16 GB RAM + 8 GB VRAM: below Gyro-S's 27.6, setup warns
        self.assertLess(have("Gyro-S", 1024.0, 8.0), s.MODELS["Gyro-S"]["ram_gb"])
        self.assertEqual(have("Gyro-S", 40.0, 8.0), 48.0)                # a real RAM figure passes through
        self.assertEqual(have("Q2_0", 1024.0, 8.0), 16.0)                # the other sizes count RAM only, as before

    def test_a_series_keeps_gyros_ngram_table_on_the_ssd(self):
        # Gyro's table is half of the 58-92 GB file: --ple-io ram would not fit beside the experts' arena on a 64 GB PC
        m = self.mod()
        cfg = self.cfg(m)
        cfg["args"][cfg["args"].index("--native") + 1] = str(m.MOUNT / "d" / "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf")
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--ple-io", c["args"])
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")

    def test_split_short_rule(self):
        # #1980: 2x B60 24 GB - IQ3_XXS (42.9 GB) was refused under NO_HOST=1, IQ2_XS (35.5 GB) loads fully
        m = self.mod()
        self.assertTrue(m.split_likely_short(42.9, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(35.5, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(50.3, [32.0, 32.0]))   # IQ3_S on 2x B70: 79%
        self.assertTrue(m.split_likely_short(77.0, [32.0, 32.0]))
        self.assertFalse(m.split_likely_short(0, [24.0]))
        self.assertFalse(m.split_likely_short(40.0, []))

    def test_small_card_rule(self):
        m = self.mod()
        self.assertTrue(m.small_card(8.0, "i915"))
        self.assertTrue(m.small_card(24.0, "i915"))
        self.assertTrue(m.small_card(10.0, "xe"))
        self.assertFalse(m.small_card(32.0, "xe"))

    def test_pictures_follow_the_one_reader_rule_and_the_vision_section_survives(self):
        """Intel Arc: pictures on gives the CPU reader (strata-vision) on the host under 24 GB and, until a 24 GB+ Arc has run the SYCL reader,
        also from 24 GB; only --vision gpu asks for the SYCL reader (strata_control/vision_reader.py is the one place for the rule)."""
        m = self.mod()
        for asked in (None, "yes", "cpu", "on"):
            self.assertEqual(m.intel_vision(asked, 8.0), "cpu")
            self.assertEqual(m.intel_vision(asked, 32.0), "cpu")
        self.assertEqual(m.intel_vision("gpu", 8.0), "gpu")
        self.assertEqual(m.intel_vision("gpu", 32.0), "gpu")
        for asked in ("no", "none", "off"):
            self.assertEqual(m.intel_vision(asked, 32.0), "none")
        section = {"exe": "/h/engine-sycl/strata-vision", "mmproj": "/h/models/mmproj.gguf", "model": "/h/models/m.gguf", "gpu": False,
                   "max_tokens": 300}
        cfg = self.cfg(m, ["--vision"])
        cfg["vision"] = section
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"], section)                              # host paths: the reader runs on the host, not in the container
        self.assertIn("--vision", c["args"])

    def test_pictures_are_on_by_default_on_every_arc_and_a_battlemage_is_told_it_is_new(self):
        """Owner decision 2026-10-22: an Arc B-series card runs like any Arc. The first picture on the Arc Pro B70 reset the GPU (xe, 2026-10-19) is a
        known risk, not a refusal: pictures are on (the CPU reader) and setup prints one plain line about it."""
        m = self.mod()
        for driver, devid in (("i915", "56a1"), ("i915", ""), ("xe", "56a0"), ("xe", "e223"), ("xe", "e20b"), ("xe", "e211"), ("xe", "")):
            self.assertTrue(m.pictures_on_by_default(driver, devid), (driver, devid))
        self.assertTrue(m.is_b_series("xe", "e223"))                        # B70
        self.assertTrue(m.is_b_series("xe", "e20b"))                        # B580
        self.assertFalse(m.is_b_series("i915", "56a1"))                     # A750
        self.assertFalse(m.is_b_series("xe", "56a0"))                       # A770 on xe
        self.assertIn("new to Strata", m.B_SERIES_LINE)
        self.assertIn("issues", m.B_SERIES_LINE)
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn("B_SERIES_LINE", src)
        self.assertNotIn('sys.argv += ["--vision", "none"]', src)          # no B-series card has pictures switched off for it
        self.assertIn("S.hip_vision = intel_vision", src)
        self.assertIn("S.build_vision_cpu(", src)                           # the CPU reader is compiled by setup's own step
        self.assertIn("build_vision_sycl(", src)                            # --vision gpu: the SYCL reader, when the oneAPI compiler is there

    def test_the_sycl_reader_is_built_only_when_asked_and_the_cpu_reader_stays_without_the_compiler(self):
        from unittest import mock
        m = self.mod()
        with mock.patch.object(m, "oneapi_compiler", return_value=None), mock.patch.object(m.S, "warn") as warn:
            self.assertFalse(m.build_vision_sycl_reader(Path("/nonexistent"), "/llama"))
        self.assertIn("oneAPI compiler", warn.call_args[0][0])
        self.assertIn("pictures are read on the CPU", warn.call_args[0][0])

    def test_the_container_shares_dev_shm_with_the_server_for_the_embeddings(self):
        sh = (ROOT / "sycl" / "serve" / "strata-sycl.sh").read_text(encoding="utf-8")
        self.assertIn("-v /dev/shm:/dev/shm", sh)
        self.assertIn("STRATA_SYCL_SHM", sh)

    def test_a_picture_that_cannot_reach_the_container_is_said_once(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("server_intel_t", ROOT / "sycl" / "serve" / "server_intel.py")
        src = (ROOT / "sycl" / "serve" / "server_intel.py").read_text(encoding="utf-8")
        self.assertIn("install_shared_embeddings()", src.split('if __name__ == "__main__":')[1])
        sys.path.insert(0, str(ROOT / "sycl" / "serve"))
        try:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except ImportError as e:                                           # no oneAPI / xe telemetry import here: the source check above stands
            self.skipTest(str(e))
        finally:
            sys.path.pop(0)
        out = io.StringIO()
        with mock.patch.object(m.S, "combined_embeddings_path", lambda d, n: Path("/tmp/x/req.sve")), contextlib.redirect_stdout(out):
            m.install_shared_embeddings()
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
        self.assertEqual(out.getvalue().count("/dev/shm has no room"), 1)


if __name__ == "__main__":
    unittest.main()
