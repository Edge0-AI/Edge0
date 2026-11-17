"""Setup's MTP download, the Gyro low-RAM pack and the FP4-Local conversion packages (#3193 #3214 #3257 #3227): the draft layer's download is run again when it stops, every tool
setup runs ends with setup, and Gyro's low-RAM pack gets --compat-bf16 like its first pack.  Everything mocked; nothing downloaded.

    python -m unittest tools.test_setup_dl_fix
"""
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_unsloth as U  # noqa: E402


class FetchMtp(unittest.TestCase):
    def run_with(self, codes):
        calls = []

        def fake_run(cmd, cwd=None, env=None, check=True, quiet=False):
            calls.append((cmd, env, check))
            return mock.Mock(returncode=codes[min(len(calls) - 1, len(codes) - 1)])
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup.time, "sleep", lambda s: None), \
                mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "warn", lambda *a, **k: None):
            try:
                setup.fetch_mtp(pathlib.Path("mtp"), {"A": "1"})
                result = "ok"
            except SystemExit as e:
                result = e.code
        return result, calls

    def test_a_stopped_download_is_run_again_and_continues(self):
        result, calls = self.run_with([1, 1, 0])
        self.assertEqual((result, len(calls)), ("ok", 3))
        self.assertTrue(all(c[0][2] == "fetch" and c[1]["A"] == "1" and c[2] is False for c in calls))

    def test_a_download_that_never_works_ends_setup_with_a_hint(self):
        out = []
        with mock.patch.object(setup, "fail", lambda msg, hint=None: out.append((msg, hint)) or (_ for _ in ()).throw(SystemExit(1))):
            result, calls = self.run_with([1])
        self.assertEqual((result, len(calls)), (1, 3))
        self.assertIn("modelscope", out[0][1])

    def test_the_source_reaches_the_fetch(self):
        with mock.patch.object(setup, "model_source", lambda: "modelscope"):
            _, calls = self.run_with([0])
        self.assertEqual(calls[0][1]["STRATA_SOURCE"], "modelscope")


class RunEndsWithSetup(unittest.TestCase):
    def test_tools_are_told_to_end_when_setup_ends(self):
        seen = {}

        def fake_subprocess_run(cmd, **kw):
            seen.update(kw)
            return mock.Mock(returncode=0)
        with mock.patch.object(setup.subprocess, "run", fake_subprocess_run), mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup.acquisition, "require", lambda *a, **k: None):
            setup.run([sys.executable, "x.py"])
            self.assertEqual(seen["env"]["STRATA_EXIT_WITH_PARENT"], "1")
            self.assertIn("PATH", seen["env"] if os.name != "nt" else {k.upper(): v for k, v in seen["env"].items()})
            setup.run([sys.executable, "x.py"], env={"B": "2", "STRATA_EXIT_WITH_PARENT": "0"})
            self.assertEqual(seen["env"]["STRATA_EXIT_WITH_PARENT"], "0", "a value the caller set is kept")
            self.assertEqual(seen["env"]["B"], "2")


class GyroLowRam(U.Base):
    def test_the_low_ram_pack_gets_compat_bf16_like_the_first_pack(self):
        code, out, _ = self.main(["--family", "gyro", "--low-ram", "on", "--context", "8192"], m="Gyro-S", family=False, ram=31.9, version="0.1.45.1")
        packs = [r for r in self.runs if any(x.endswith("iq_pack.py") for x in r)]
        self.assertTrue(packs, (code, out[-1500:]))
        self.assertTrue(all("--compat-bf16" in r for r in packs), packs)
        self.assertTrue(any("--experts-bin" in r for r in packs), "the low-RAM pack ran")


class Fp4LocalPackages(unittest.TestCase):
    """#3227: torch, transformers and safetensors come from a pinned requirements file, right before a prepare that runs."""

    def test_the_requirements_file_pins_what_the_converter_imports(self):
        pkgs = setup.nvfp4_local_packages()
        self.assertEqual(sorted({p.split("==")[0] for p in pkgs}), sorted(setup.NVFP4_LOCAL_MODULES))
        self.assertTrue(all("==" in p for p in pkgs), pkgs)

    def test_missing_packages_are_installed_once_with_the_pins(self):
        calls = []
        with mock.patch("importlib.util.find_spec", lambda name: None), mock.patch.object(setup, "CONTROL", False), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "pip_install", lambda pkgs, what, extra=(): calls.append((list(pkgs), list(extra)))):
            setup.ensure_nvfp4_local_packages()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], setup.nvfp4_local_packages())
        self.assertEqual(bool(calls[0][1]), not setup.WIN)               # the CPU wheel index only on Linux

    def test_nothing_is_installed_when_they_are_there(self):
        with mock.patch("importlib.util.find_spec", lambda name: object()), \
                mock.patch.object(setup, "pip_install", side_effect=AssertionError("pip ran")):
            setup.ensure_nvfp4_local_packages()

    def test_the_app_says_so_instead_of_installing(self):
        with mock.patch("importlib.util.find_spec", lambda name: None), mock.patch.object(setup, "CONTROL", True), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "pip_install", side_effect=AssertionError("pip ran")), \
                self.assertRaises(SystemExit):
            setup.ensure_nvfp4_local_packages()

    def test_the_tool_names_the_missing_package_when_run_by_hand(self):
        import nvfp4_local_pack as P
        with mock.patch("importlib.util.find_spec", lambda name: None if name == "transformers" else object()):
            with self.assertRaises(SystemExit) as cm:
                P.require_packages()
        self.assertIn("transformers", str(cm.exception))
        self.assertIn("requirements-nvfp4-local.txt", str(cm.exception))
        self.assertNotIn("torch,", str(cm.exception))


class Fp4LocalOrder(unittest.TestCase):
    def test_the_packages_come_before_the_prepare_run(self):
        import test_setup_golden as G
        import test_setup_nvfp4 as N
        events = []
        with mock.patch.object(setup, "ensure_nvfp4_local_packages", lambda: events.append("packages")):
            ram, found = G.PROFILES["128GB-1x24GB"]
            _ = G.install(ram, found, ["--family", "nvfp4", "--no-start", "--model", "NVFP4-QAD"])
        self.assertEqual(events, ["packages"])


if __name__ == "__main__":
    unittest.main()
