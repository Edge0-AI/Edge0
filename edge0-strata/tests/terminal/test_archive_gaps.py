"""Things a user of the Terminal archive (no setup.py, no tools/calibrate.py, an engine package) hit on real boxes."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import calibration, tune_apply, tuning  # noqa: E402
from strata_control.profiles import ProfileError  # noqa: E402
from strata_control.targets import host_platform  # noqa: E402


class BooleanFlags(unittest.TestCase):
    def test_a_boolean_flag_is_on_or_absent_never_given_a_value(self):
        self.assertEqual(tuning.flag_argv("--kv-grow", "1"), ["--kv-grow"])
        self.assertEqual(tuning.flag_argv("--kv-grow", "0"), [])
        self.assertEqual(tuning.flag_argv("--kv-grow", True), ["--kv-grow"])
        self.assertEqual(tuning.flag_argv("--kv-grow", False), [])
        self.assertEqual(tuning.flag_argv("--spec-min-p", "0.5"), ["--spec-min-p", "0.5"])
        self.assertEqual(tuning.flag_argv("--spec", "1"), ["--spec", "1"])      # a numeric flag keeps its 1

    def test_a_candidate_with_kv_grow_off_does_not_become_kv_grow_0(self):
        self.assertEqual(tune_apply.args_to_argv({"--kv-grow": "0", "--kv": "int8"}), ["--kv", "int8"])
        self.assertEqual(tune_apply.args_to_argv({"--kv-grow": "1"}), ["--kv-grow"])

    def test_the_harness_argv_follows_the_same_rule(self):
        from strata_control import tune_engine
        harness = tune_engine.EngineHarness(Path("engine"), {"--kv-grow": "0", "--pool-workers": "0", "--host-core": "first"})
        argv = harness._argv()
        self.assertNotIn("--kv-grow", argv)
        self.assertIn("--pool-workers", argv)
        self.assertEqual(argv[argv.index("--pool-workers") + 1], "0")


class EnginePackageEnv(unittest.TestCase):
    def test_a_package_folder_goes_first_on_the_library_path(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "engine-component.json").write_text("{}", encoding="utf-8")
            (folder / "strata").write_text("x", encoding="utf-8")
            with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/old", "PATH": "/old"}):
                env = tuning.engine_env(folder / "strata")
            var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            self.assertTrue(env[var].startswith(str(folder.resolve())))
            self.assertTrue(env[var].endswith("/old") or env[var].endswith("old"))

    def test_a_plain_engine_keeps_the_callers_environment(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "strata").write_text("x", encoding="utf-8")
            self.assertIsNone(tuning.engine_env(Path(d) / "strata"))


class CalibrateWithoutAFullFolder(unittest.TestCase):
    def test_missing_setup_is_a_refusal_in_words_not_a_traceback(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_INSTALL_DIR", None)
            with mock.patch("importlib.import_module", side_effect=ImportError("no module")):
                with self.assertRaises(ProfileError) as ctx:
                    calibration._full_folder("tools.calibrate")
        self.assertEqual(ctx.exception.code if hasattr(ctx.exception, "code") else 5, 5)
        self.assertIn("STRATA_INSTALL_DIR", str(ctx.exception))
        self.assertIn("tune run --mode api", str(ctx.exception))

    def test_tool_availability_is_reported_for_the_plan(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_INSTALL_DIR", None)
            with mock.patch("importlib.util.find_spec", return_value=None):
                self.assertFalse(calibration._tools_available())
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "tools").mkdir()
                (Path(d) / "setup.py").write_text("", encoding="utf-8")
                (Path(d) / "tools" / "calibrate.py").write_text("", encoding="utf-8")
                os.environ["STRATA_INSTALL_DIR"] = d
                with mock.patch("importlib.util.find_spec", return_value=None):
                    self.assertTrue(calibration._tools_available())


@unittest.skipIf(host_platform() == "linux-aarch64", "there is no HIP engine for arm64 Linux (BACKEND_MATRIX)")
class HipPackageForAnotherCard(unittest.TestCase):
    """A HIP package built for gfx1200/gfx1201 must not be chosen for a Phoenix iGPU (gfx1103): P2006 resolved it as usable."""

    def _kfd(self, root: Path, versions):
        for number, value in enumerate(versions):
            node = root / str(number)
            node.mkdir()
            (node / "properties").write_text("cpu_cores_count 1\ngfx_target_version " + value + "\n", encoding="utf-8")

    def test_gfx_target_version_becomes_the_gfx_name(self):
        from strata_control import engine_component as ec
        with tempfile.TemporaryDirectory() as d:
            self._kfd(Path(d), ["0", "110003", "120001", "90010", "100300"])
            self.assertEqual(ec.hip_archs_present(d), ["gfx1103", "gfx1201", "gfx90a", "gfx1030"])
        self.assertEqual(ec.hip_archs_present("/nonexistent/kfd"), [])

    def test_a_package_without_the_cards_arch_is_not_chosen_and_one_with_it_is(self):
        from strata_control import engine_component as ec
        from tests.terminal.test_engine_component import make_package, rewrite
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = make_package(root, backend="hip")
            rewrite(folder, schema_version=2, archs=["gfx1200", "gfx1201"], build_id="test")
            with mock.patch.object(ec, "hip_archs_present", return_value=["gfx1103"]):
                with self.assertRaises(ec.EngineMissing) as ctx:
                    ec.resolve(root, backend="hip")
            self.assertIn("gfx1103", str(ctx.exception))
            self.assertIn("gfx1201", str(ctx.exception))
            with mock.patch.object(ec, "hip_archs_present", return_value=["gfx1201"]):
                self.assertEqual(ec.resolve(root, backend="hip").backend, "hip")
            with mock.patch.object(ec, "hip_archs_present", return_value=[]):      # cannot tell: not a refusal
                self.assertEqual(ec.resolve(root, backend="hip").backend, "hip")


class DeviceAccess(unittest.TestCase):
    def test_a_render_node_the_user_cannot_open_is_named(self):
        from strata_control import doctor
        if not sys.platform.startswith("linux"):
            self.skipTest("Linux device nodes")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "dri").mkdir()
            node = Path(d) / "dri" / "renderD128"
            node.write_text("", encoding="utf-8")
            os.chmod(node, 0)
            if os.access(node, os.R_OK):
                self.skipTest("running as a user that can open anything")
            issue = doctor._device_access_issue("sycl", d)
            self.assertEqual(issue["code"], "GPU_DEVICE_ACCESS")
            self.assertIn("render", issue["message"])
            os.chmod(node, 0o666)
            self.assertIsNone(doctor._device_access_issue("sycl", d))
            self.assertIsNone(doctor._device_access_issue("cuda", d))
            self.assertIsNone(doctor._device_access_issue("hip", tempfile.gettempdir() + "/none"))


class ArchiveContents(unittest.TestCase):
    def test_the_fingerprint_table_models_import_reads_is_in_the_payload(self):
        sys.path.insert(0, str(ROOT / "packaging"))
        import package_terminal
        self.assertIn("data/gguf_fingerprints.json", package_terminal.PAYLOAD)
        self.assertTrue((ROOT / "data" / "gguf_fingerprints.json").is_file())


if __name__ == "__main__":
    unittest.main()
