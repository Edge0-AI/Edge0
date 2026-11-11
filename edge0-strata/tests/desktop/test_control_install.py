"""The desktop app's installer: tools/strata_mcp.py and setup.py run on Control's Python in STRATA_CONTROL_INSTALL mode.

What the owner asked for: a user installs the app, picks a model in it and downloads and prepares it, like any normal user.
What must stay true: a start or a serve never acquires anything (tests/terminal/test_no_acquisition.py), and the install
itself downloads model files only - no pip, no git, no compiler, no engine fetch, only the bundled preparation tools.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_setup(home: Path, name: str = "setup_control_mode"):
    """setup.py imported as the app's installer: the mode is read when the module loads."""
    env = {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(home)}
    with mock.patch.dict(os.environ, env):
        spec = importlib.util.spec_from_file_location(name, ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ControlModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "install"
        self.home.mkdir()
        self.S = load_setup(self.home)

    def test_the_folder_everything_is_written_to_is_the_install_folder_not_the_bundle(self):
        S = self.S
        self.assertTrue(S.CONTROL)
        self.assertEqual(S.home(), self.home.resolve())
        self.assertEqual(S.ROOT, ROOT)
        # without the mode nothing moves: HOME is the source folder, as it always was
        spec = importlib.util.spec_from_file_location("setup_plain_mode", ROOT / "setup.py")
        plain = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "0"}):
            spec.loader.exec_module(plain)
        self.assertFalse(plain.CONTROL)
        self.assertEqual(plain.home(), plain.ROOT)

    def test_only_this_python_running_a_bundled_tool_may_run(self):
        S = self.S
        tool = ROOT / "tools" / "pack_index.py"
        for refused in (
            [sys.executable, "-m", "pip", "install", "numpy"],
            ["pip", "install", "numpy"],
            ["git", "clone", "https://example.invalid/x.git"],
            ["cmake", "--build", "."],
            ["winget", "install", "Nvidia.CUDA"],
            [sys.executable, str(self.home / "evil.py")],                 # a script outside the bundled tools
            [sys.executable, str(ROOT / "setup.py"), "--build"],
            [sys.executable],
        ):
            with self.subTest(refused=refused), mock.patch.object(S.subprocess, "run", side_effect=AssertionError("ran")):
                with self.assertRaises(S.acquisition.AcquisitionRefused):
                    S.run(refused)
        done = subprocess.CompletedProcess([], 0)
        with mock.patch.object(S.subprocess, "run", return_value=done) as runner:
            S.run([sys.executable, str(tool), "--help"], check=False)
        runner.assert_called_once()

    def test_no_package_is_installed(self):
        S = self.S
        with mock.patch.object(S, "run", side_effect=AssertionError("pip")), \
                mock.patch.object(S, "control_missing_packages", return_value=[]):
            S.pip_install(["numpy==1"], "the packages")      # the app ships them; nothing runs
        with mock.patch.object(S, "control_missing_packages", return_value=["numpy"]), \
                self.assertRaises(SystemExit):
            S.pip_install(["numpy==1"], "the packages")

    def test_the_engine_is_the_component_and_a_missing_one_does_not_stop_the_download(self):
        S = self.S
        llama = self.home / "third_party" / "llama.cpp"
        (llama / "ggml").mkdir(parents=True)
        (llama / "ggml" / "CMakeLists.txt").write_text("", encoding="utf-8")
        (llama / "gguf-py").mkdir()
        empty = Path(self.tmp.name) / "components"
        empty.mkdir()
        gpu = {"index": 0, "vram_gb": 12.0}
        with mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(empty)}), \
                mock.patch.object(S, "download", side_effect=AssertionError("fetched")), \
                mock.patch.object(S, "build_engine", side_effect=AssertionError("built")), \
                mock.patch.object(S, "get_prebuilt", side_effect=AssertionError("fetched an engine")):
            found, eng, meta, libs, chosen, vision = S.engine_step(mock.Mock(build=False, prebuilt=None), gpu, False, [], 13, "gpu")
        self.assertEqual(found.resolve(), llama.resolve())
        self.assertEqual(libs, [])
        self.assertEqual(vision, "none")                                   # images are not in the first app release
        self.assertEqual(meta["source"], "local")
        self.assertEqual(S.engine_exe(eng), "component:strata-engine")

    def test_the_run_config_goes_to_the_install_folder_and_the_data_folder_sits_beside_it(self):
        S = self.S
        self.assertEqual(S.installed_configs(), [])
        (self.home / "strata-q2_0.json").write_text(json.dumps({"exe": "component:strata-engine", "args": ["--native", "x.gguf"],
                                                                "tokenizer": "t"}), encoding="utf-8")
        with mock.patch.object(S, "model_config", return_value=True):
            self.assertEqual([p.name for p in S.installed_configs()], ["strata-q2_0.json"])
        with mock.patch.dict(os.environ, {"APPDATA": self.tmp.name, "XDG_CONFIG_HOME": self.tmp.name}):
            data, _ = S.data_folder_in_place()
        self.assertEqual(data, self.home.resolve().parent / "Strata-data")

    @unittest.skipIf(os.name == "nt", "a launcher is a shell script: Linux and macOS bundles")
    def test_the_bundle_keeps_one_real_python_and_launchers_for_the_other_names(self):
        sys.path.insert(0, str(ROOT / "apps" / "desktop" / "scripts"))
        try:
            spec = importlib.util.spec_from_file_location("stage_control_for_launchers", ROOT / "apps" / "desktop" / "scripts" / "stage-control.py")
            stage = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stage)
        finally:
            sys.path.pop(0)
        folder = Path(self.tmp.name) / "bundle"
        bindir = folder / "python" / "bin"
        bindir.mkdir(parents=True)
        real = bindir / "python3.12"
        real.write_text("#!/bin/sh"+chr(10)+"echo real "+chr(34)+"$@"+chr(34)+chr(10))
        real.chmod(0o755)
        os.symlink("python3.12", bindir / "python3")
        os.symlink("python3", bindir / "python")
        self.assertEqual(stage.dedupe_python(folder), ["python", "python3"])
        for name in ("python", "python3"):
            f = bindir / name
            self.assertFalse(f.is_symlink())
            self.assertLess(f.stat().st_size, 400, "a launcher, not a second copy of the interpreter")
            self.assertTrue(os.access(f, os.X_OK))
        out = subprocess.run([str(bindir / "python"), "a b", "c"], capture_output=True, text=True).stdout
        self.assertEqual(out.strip(), "real a b c")
        self.assertEqual(stage.dedupe_python(folder), [], "run twice: nothing left to change")
        # no external program: with an empty PATH (and, in the AppImage, libraries that make `dirname` fail) the launcher still starts the
        # interpreter, from another folder and through a relative name. Ubuntu 26.04: "dirname: libsystemd.so.0: version LIBSYSTEMD_254 not found".
        for cwd, name in ((str(folder), "python/bin/python3"), (str(bindir), "./python"), (str(bindir.parent), "bin/python")):
            r = subprocess.run(["/bin/sh", name, "x"], capture_output=True, text=True, cwd=cwd, env={"PATH": ""})
            self.assertEqual(r.stdout.strip(), "real x", (cwd, name, r.stderr))

    def test_the_bundle_lists_everything_the_install_imports(self):
        sys.path.insert(0, str(ROOT / "apps" / "desktop" / "scripts"))
        try:
            spec = importlib.util.spec_from_file_location("stage_control_for_test", ROOT / "apps" / "desktop" / "scripts" / "stage-control.py")
            stage = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stage)
        finally:
            sys.path.pop(0)
        tools = stage.local_imports(ROOT, stage.INSTALLER_TOOLS)
        for need in ("strata_mcp", "iq_pack", "strata_pack", "canonical_xcheck", "pack_layer", "gguf_reader", "gguf_writer", "_paths"):
            self.assertIn(need, tools)
        out = Path(self.tmp.name) / "control"
        out.mkdir()
        record = stage.add_installer(out)
        for name in ("setup.py", "ref/load.py", "tools/strata_mcp.py", "tools/iq_pack.py", "data/draft_vocab.bin"):
            self.assertTrue((out / name).is_file(), name)
        self.assertIn("tools/strata_pack.py", record["files"])
        # every module setup.py's preparation steps run exists in the bundle
        for tool in ("iq_pack", "strata_pack", "pack_index", "strata_tokenizer", "mtp_fetch", "mtp_pack", "mtp_rt"):
            self.assertTrue((out / "tools" / f"{tool}.py").is_file(), tool)


class InstallerServerTests(unittest.TestCase):
    """tools/strata_mcp.py with its source (setup.py, serve/) apart from the install folder, as the app runs it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "install"
        self.root.mkdir()
        spec = importlib.util.spec_from_file_location("strata_mcp_for_test", ROOT / "tools" / "strata_mcp.py")
        self.mcp = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(self.root)}):
            spec.loader.exec_module(self.mcp)

    def test_the_catalog_comes_from_setup_py_in_the_bundle(self):
        s = self.mcp.Strata(self.root, ROOT)
        self.assertEqual(s.root, self.root.resolve())
        self.assertEqual(s.src, ROOT)
        models, families, contexts, source = s.tables()
        self.assertEqual(source, "setup.py")
        self.assertIn("qwen", families)
        self.assertIn("IQ3_XXS", models)
        self.assertEqual(s.python_override, sys.executable)               # Control's own Python runs the install

    def test_the_install_job_runs_on_control_python_without_a_venv_and_may_fetch_only_for_itself(self):
        s = self.mcp.Strata(self.root, ROOT)
        spec_path = self.root / "job.json"
        spec = {"job": "j1", "root": str(self.root), "src": str(ROOT), "control": True, "python": sys.executable,
                "base_python": sys.executable, "create_venv": False, "args": ["--yes", "--no-start"],
                "result": str(self.root / "result.json")}
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        seen = {}

        class FakePopen:
            pid = 4242

            def __init__(self, cmd, **kw):
                seen["cmd"], seen["kw"] = cmd, kw

            def wait(self):
                return 0

        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1", "PYTHONPATH": "x", "PYTHONHOME": "y"}), \
                mock.patch.object(self.mcp.subprocess, "Popen", FakePopen), \
                mock.patch.object(self.mcp, "proc_identity", return_value=None):
            self.assertEqual(self.mcp.install_job(str(spec_path)), 0)
            self.assertEqual(os.environ["STRATA_NO_ACQUIRE"], "1", "this process keeps the switch")
        self.assertEqual(seen["cmd"][:2], [sys.executable, str(ROOT / "setup.py")])
        env = seen["kw"]["env"]
        self.assertEqual(env["STRATA_CONTROL_INSTALL"], "1")
        self.assertEqual(env["STRATA_INSTALL_DIR"], str(self.root))
        self.assertNotIn("STRATA_NO_ACQUIRE", env, "the explicit install job may fetch the chosen model's files")
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertFalse((self.root / ".venv").exists(), "no virtual environment, no pip")
        self.assertIsNotNone(s)

    def test_a_two_file_model_has_one_combined_progress(self):
        """The bar went 70 GB -> 0 -> up again because each file reported its own total. One total for the whole model now."""
        folder = self.root / "Strata-data" / "models" / "IQ3_S"
        folder.mkdir(parents=True)
        job = {"folder": str(folder), "total_gb": 0.004}
        step5 = {"step": "5/7: downloading Qwen3.8-Flash-Next IQ3_S"}
        seen = []
        # file 1 downloading, then complete, then file 2 starts: the combined figure only goes up
        for name, size in (("a.gguf.part", 400_000), ("a.gguf.part", 1_500_000)):
            (folder / name).write_bytes(b"x" * size)
            seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        (folder / "a.gguf.part").rename(folder / "a.gguf")
        (folder / "b.gguf.part").write_bytes(b"x" * 100_000)
        seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        (folder / "b.gguf.part").write_bytes(b"x" * 1_000_000)
        seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        done = [d["done_gb"] for d in seen]
        self.assertEqual(done, sorted(done), done)
        self.assertEqual(seen[-1]["total_gb"], 0.004)
        self.assertAlmostEqual(seen[-1]["remaining_gb"], 0.004 - seen[-1]["done_gb"], places=2)
        self.assertTrue(all(d["file"] == "model files" for d in seen))
        # not a download any more once setup is preparing the files
        self.assertNotIn("download", self.mcp.combined_progress({"step": "6/7: preparing the model for Strata", "download": {"x": 1}}, job))
        # a resumed install counts what is already on disk
        self.assertGreater(self.mcp.combined_progress(dict(step5), job)["download"]["done_gb"], 0)

    def test_a_model_in_parts_counts_every_part_and_the_picture_reader(self):
        """QA 2026-10-18 (6): the bar stopped at 99 % while the picture reader downloaded, and a first whole part looked like a finished
        model. Every part is counted, the picture reader is a part of its own, and 'done' needs all of them."""
        S = self.mcp.Strata(self.root, ROOT).setup_module()
        models = self.root / "Strata-data" / "models"
        files = self.mcp.expected_files(S, "qwen", "IQ3_S", models, True)
        self.assertEqual([f["role"] for f in files], ["model", "model", "vision"])
        self.assertEqual(Path(files[2]["path"]).parent, models, "the picture reader sits beside the model folders")
        folder = Path(files[0]["path"]).parent
        folder.mkdir(parents=True)
        job = {"folder": str(folder), "total_gb": 0.004, "files": files}
        step5 = {"step": "5/7: downloading Qwen3.8-Flash-Next IQ3_S"}
        # the first part is whole, the second not started: the model is NOT downloaded
        Path(files[0]["path"]).write_bytes(b"x" * 2_000_000)
        one = self.mcp.combined_progress(dict(step5), job)
        model = next(x for x in one["parts"] if x["kind"] == "model")
        self.assertEqual((model["files_done"], model["files_total"], model["state"]), (1, 2, "downloading"))
        self.assertLess(one["download"]["percent"], 99)
        # both parts whole, the picture reader under way: still counted, and the bar is not at its end
        Path(files[1]["path"]).write_bytes(b"x" * 2_000_000)
        Path(files[2]["path"]).with_name(Path(files[2]["path"]).name + ".part").write_bytes(b"x" * 400_000)
        two = self.mcp.combined_progress(dict(step5), job)
        vision = next(x for x in two["parts"] if x["kind"] == "vision")
        self.assertEqual(vision["state"], "downloading")
        self.assertEqual(two["download"]["file"], "picture reader")
        self.assertGreater(two["download"]["done_gb"], one["download"]["done_gb"])
        self.assertGreater(two["download"]["total_gb"], 0.004, "the picture reader is part of the total")
        self.assertLessEqual(two["download"]["percent"], 99)
        # all whole
        Path(files[2]["path"]).with_name(Path(files[2]["path"]).name + ".part").unlink()
        Path(files[2]["path"]).write_bytes(b"x" * 907_543_008)
        three = self.mcp.combined_progress(dict(step5), job)
        self.assertTrue(all(x["state"] == "done" for x in three["parts"]))
        self.assertEqual(three["download"]["done_gb"], three["download"]["total_gb"])
        # not a download once setup prepares the files
        self.assertNotIn("download", self.mcp.combined_progress({"step": "6/7: preparing the model for Strata", "download": {"x": 1}}, job))

    def test_a_pack_with_many_files_reports_the_whole_pack(self):
        S = self.mcp.Strata(self.root, ROOT).setup_module()
        files = self.mcp.expected_files(S, "nvfp4", "NVFP4", self.root / "models", False)
        self.assertGreater(len(files), 5)
        self.assertTrue(all(f["bytes"] for f in files), "a ready-made pack pins every file's size")
        big = next(f for f in files if f["path"].endswith("experts.bin"))
        Path(big["path"]).parent.mkdir(parents=True)
        Path(big["path"] + ".part").write_bytes(b"x" * 1000)
        prog = self.mcp.combined_progress({"step": "5/7: downloading"}, {"files": files, "total_gb": 128.0})
        model = prog["parts"][0]
        self.assertEqual((model["kind"], model["state"], model["files_done"], model["files_total"]), ("model", "downloading", 0, len(files)))
        self.assertAlmostEqual(model["total_gb"], sum(f["bytes"] for f in files) / 1e9, places=2)

    def test_a_config_with_only_some_parts_is_not_ready(self):
        """The config names the first part only; 'ready' used to mean that part exists."""
        data = self.root / "models" / "x"
        data.mkdir(parents=True)
        first = data / "Model-00001-of-00003.gguf"
        first.write_bytes(b"x")
        self.assertEqual(len(self.mcp.shard_gaps([str(first)])), 2)
        (data / "Model-00002-of-00003.gguf").write_bytes(b"x")
        (data / "Model-00003-of-00003.gguf.part").write_bytes(b"x")
        self.assertEqual([Path(g).name for g in self.mcp.shard_gaps([str(first)])], ["Model-00003-of-00003.gguf"])
        (data / "Model-00003-of-00003.gguf.part").unlink()
        (data / "Model-00003-of-00003.gguf").write_bytes(b"x")
        self.assertEqual(self.mcp.shard_gaps([str(first)]), [])
        self.assertEqual(self.mcp.shard_gaps([str(data / "single.gguf")]), [])

    @unittest.skipUnless(os.name == "nt" or sys.platform.startswith("linux"), "a PC with a GPU probe")
    def test_a_real_install_reaches_the_model_download_with_nothing_else_acquired(self):
        """setup.py in the app's mode, as the install job runs it: steps 1-4 pass (PC check, packages bundled, engine component
        optional, llama.cpp source already present) and step 5 asks for the model file. The model host is a folder that does
        not exist, so nothing is downloaded; pip, git, cmake and winget shims that record any call must stay unused."""
        import setup as plain
        if not plain.gpus():
            self.skipTest("no NVIDIA GPU here: setup stops at step 1")
        import shutil
        if shutil.disk_usage(self.tmp.name).free < 130 * 10**9:
            self.skipTest("less than 130 GB free: setup stops at its disk check before the download step")
        home = Path(self.tmp.name) / "home"
        shims = Path(self.tmp.name) / "shims"
        for p in (home / "third_party" / "llama.cpp" / "ggml", home / "third_party" / "llama.cpp" / "gguf-py", shims):
            p.mkdir(parents=True)
        (home / "third_party" / "llama.cpp" / "ggml" / "CMakeLists.txt").write_text("", encoding="utf-8")
        log = Path(self.tmp.name) / "shim.log"
        for tool in ("pip", "pip3", "git", "cmake", "ninja", "winget", "curl", "wget", "cl", "make"):
            if os.name == "nt":
                (shims / f"{tool}.cmd").write_text(f"@echo {tool} %* >> \"{log}\"\r\n", encoding="utf-8")
            else:
                (shims / tool).write_text(f"#!/bin/sh\necho {tool} \"$@\" >> '{log}'\n", encoding="utf-8")
                (shims / tool).chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_NO_ACQUIRE", "PYTHONPATH", "VIRTUAL_ENV")}
        env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(home), APPDATA=str(Path(self.tmp.name) / "appdata"),
                   XDG_CONFIG_HOME=str(Path(self.tmp.name) / "xdg"), STRATA_COMPONENTS_DIR=str(Path(self.tmp.name) / "components"),
                   HF_ENDPOINT=(Path(self.tmp.name) / "no-such-hf").as_uri(), PATH=str(shims) + os.pathsep + env.get("PATH", ""),
                   PYTHONIOENCODING="utf-8")
        done = subprocess.run([sys.executable, str(ROOT / "setup.py"), "--yes", "--no-start", "--family", "qwen", "--model", "Q2_0",
                               "--context", "32768", "--vision", "no", "--experimental-speed-projection", "off"],
                              capture_output=True, text=True, env=env, cwd=str(ROOT), stdin=subprocess.DEVNULL, timeout=300)
        out = done.stdout + done.stderr
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("bundled with the app", out)
        self.assertIn("Step 5: downloading", out)
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", out)
        self.assertFalse(log.exists(), "something ran pip, git, cmake or a package manager: " + (log.read_text() if log.exists() else ""))


if __name__ == "__main__":
    unittest.main()


class FailedStartTests(unittest.TestCase):
    def test_a_failed_start_carries_the_last_lines_the_server_wrote(self):
        """The app turns `server_exited` into words and keeps the reason under Details: Control reports the log's last lines."""
        from strata_control import runtime
        owner = [c for _, c in vars(runtime).items() if isinstance(c, type) and hasattr(c, "_log_tail")][0]
        with tempfile.TemporaryDirectory() as directory:
            fake = mock.Mock()
            fake.state_path = Path(directory) / "app-1.json"
            fake.name = "app-1"
            self.assertIsNone(owner._log_tail(fake))                       # no log: nothing, no exception
            (Path(directory) / "app-1.server.log").write_text("loading\n\nmemory guard: 5.1 GiB free is under 6 GiB\nthe engine process ended: exit code 3\n", encoding="utf-8")
            tail = owner._log_tail(fake)
            self.assertIn("memory guard: 5.1 GiB free", tail)
            self.assertTrue(tail.endswith("exit code 3"))
            self.assertNotIn("loading", tail.split(" | ")[0] if tail.count(" | ") > 2 else "")


class DeleteModelTests(unittest.TestCase):
    """Deleting a downloaded model removes its own files and run config, never what another installed model still uses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "install"
        self.root.mkdir()
        self.data = base / "Strata-data"
        spec = importlib.util.spec_from_file_location("strata_mcp_for_delete", ROOT / "tools" / "strata_mcp.py")
        self.mcp = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(self.root)}):
            spec.loader.exec_module(self.mcp)
        self.s = self.mcp.Strata(self.root, ROOT)
        self.tools = self.mcp.Tools(self.s)

        def make(path: Path, size: int):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * size)
            return path
        self.make = make
        mtp = self.data / "mtp" / "rt"
        make(mtp / "experts.bin", 3000)
        shared_ple = make(self.data / "models" / "Q2_0" / "shard-2.gguf", 2000)     # the same second shard in both models
        for tag, size in (("iq3_s", 10000), ("q2_0", 4000)):
            folder = self.data / "models" / tag
            first = make(folder / "shard-1.gguf", size)
            make(first.with_name(first.name + ".done"), 1)
            pack = self.data / "packs" / tag
            make(pack / "tokenizer" / "vocab.json", 500)
            (self.root / f"strata-{tag}.json").write_text(json.dumps({"exe": "component:strata-engine", "args": [
                "--pack", str(pack), "--native", str(first), "--ple-gguf", str(shared_ple), "--mtp", str(mtp), "--max-context", "32768"]}), encoding="utf-8")
            (self.root / f"strata-{tag}.log").write_text("log", encoding="utf-8")

    def test_the_plan_names_what_goes_and_what_stays(self):
        plan = self.tools.strata_remove(model="iq3_s")["plan"]
        removed = {Path(x["path"]).name for x in plan["remove"]}
        kept = {Path(x["path"]).name for x in plan["keep"]}
        self.assertIn("shard-1.gguf", removed)
        self.assertIn("iq3_s", removed)                                    # its pack folder
        self.assertIn("strata-iq3_s.json", removed)
        self.assertEqual(kept, {"shard-2.gguf", "rt"})                     # the shared PLE shard and the shared MTP draft
        self.assertEqual(plan["bytes"], 10000 + 1 + 500 + 3 + (self.root / "strata-iq3_s.json").stat().st_size)

    def test_confirm_deletes_only_that_model_and_the_other_still_has_everything(self):
        result = self.tools.strata_remove(model="iq3_s", confirm=True)
        self.assertEqual(result["failed"], [])
        self.assertFalse((self.data / "models" / "iq3_s").exists())        # its folder is gone with its file
        self.assertFalse((self.data / "packs" / "iq3_s").exists())
        self.assertFalse((self.root / "strata-iq3_s.json").exists())
        self.assertTrue((self.data / "models" / "Q2_0" / "shard-2.gguf").is_file())
        self.assertTrue((self.data / "mtp" / "rt" / "experts.bin").is_file())
        self.assertTrue((self.data / "models" / "q2_0" / "shard-1.gguf").is_file())
        self.assertTrue((self.data / "packs" / "q2_0" / "tokenizer" / "vocab.json").is_file())
        self.assertTrue((self.root / "strata-q2_0.json").is_file())

    def test_the_last_model_takes_its_shared_files_with_it(self):
        self.tools.strata_remove(model="iq3_s", confirm=True)
        self.tools.strata_remove(model="q2_0", confirm=True)
        self.assertFalse((self.data / "mtp" / "rt").exists())
        self.assertFalse((self.data / "models" / "Q2_0" / "shard-2.gguf").exists())

    def test_nothing_outside_strata_s_folders_is_touched_and_each_model_reports_its_own_disk_use(self):
        outside = Path(self.tmp.name) / "elsewhere" / "mine.gguf"
        self.make(outside, 777)
        cfg = self.root / "strata-q2_0.json"
        data = json.loads(cfg.read_text())
        data["args"] += ["--ple-gguf", str(outside)]
        cfg.write_text(json.dumps(data), encoding="utf-8")
        plan = self.tools.strata_remove(model="q2_0")["plan"]
        self.assertIn("mine.gguf", {Path(x["path"]).name for x in plan["keep"]})
        self.tools.strata_remove(model="q2_0", confirm=True)
        self.assertTrue(outside.is_file())
        described = self.s.describe_config(self.root / "strata-iq3_s.json")
        self.assertGreater(described["disk_bytes"], 10000)

    def test_files_strata_downloaded_into_another_models_folder_are_deleted_too(self):
        # The app installs into its own Models folder while settings.json names another data folder (#3716).
        other = Path(self.tmp.name) / "Documents" / "Strata"
        shard = self.make(other / "Models" / "IQ3" / "m-00001.gguf", 5000)
        self.make(shard.with_name(shard.name + ".done"), 1)
        pack = other / "packs" / "iq3"
        self.make(pack / "dense.bin", 300)
        self.make(pack / "conversions.json", 2)
        mine = self.make(Path(self.tmp.name) / "mine" / "own.gguf", 99)
        (self.root / "strata-iq3x.json").write_text(json.dumps({"exe": "component:strata-engine", "args": [
            "--pack", str(pack), "--native", str(shard), "--ple-gguf", str(mine)]}), encoding="utf-8")
        result = self.tools.strata_remove(model="iq3x", confirm=True)
        self.assertEqual(result["failed"], [])
        self.assertFalse(shard.exists())
        self.assertFalse(pack.exists())
        self.assertTrue(mine.is_file(), "a file the person brought stays")

    def test_every_part_of_a_multi_part_model_goes_with_it_and_a_part_another_model_uses_stays(self):
        # The run config names only the first part; the others used to stay on disk (#3716).
        folder = self.data / "models" / "IQ3_XXS"
        parts = [self.make(folder / f"Big-0000{i}-of-00003.gguf", 5000) for i in (1, 2, 3)]
        for part in parts:
            self.make(part.with_name(part.name + ".done"), 1)
        (self.root / "strata-iq3xxs.json").write_text(json.dumps({"exe": "component:strata-engine", "args": [
            "--native", str(parts[0]), "--max-context", "32768"]}), encoding="utf-8")
        plan = self.tools.strata_remove(model="iq3xxs")["plan"]
        removed = {Path(x["path"]).name for x in plan["remove"]}
        self.assertTrue({p.name for p in parts} <= removed, removed)
        result = self.tools.strata_remove(model="iq3xxs", confirm=True)
        self.assertEqual(result["failed"], [])
        self.assertEqual([p.exists() for p in parts], [False, False, False])
        self.assertFalse(folder.exists())

    def test_a_part_that_another_model_names_is_kept_when_a_multi_part_model_is_deleted(self):
        folder = self.data / "models" / "IQ3_XXS"
        parts = [self.make(folder / f"Big-0000{i}-of-00002.gguf", 5000) for i in (1, 2)]
        for part in parts:
            self.make(part.with_name(part.name + ".done"), 1)
        (self.root / "strata-iq3xxs.json").write_text(json.dumps({"exe": "component:strata-engine", "args": ["--native", str(parts[0])]}), encoding="utf-8")
        (self.root / "strata-twin.json").write_text(json.dumps({"exe": "component:strata-engine", "args": ["--native", str(parts[0])]}), encoding="utf-8")
        self.tools.strata_remove(model="iq3xxs", confirm=True)
        self.assertTrue(all(p.is_file() for p in parts), "the twin config still uses every part")


class LocalGgufInstallTests(unittest.TestCase):
    def test_a_plan_from_a_local_folder_adopts_the_files_and_needs_no_download_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "install"
            root.mkdir()
            local = Path(tmp) / "mine"
            local.mkdir()
            spec = importlib.util.spec_from_file_location("strata_mcp_for_local", ROOT / "tools" / "strata_mcp.py")
            mcp = importlib.util.module_from_spec(spec)
            with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(root)}):
                spec.loader.exec_module(mcp)
            tools = mcp.Tools(mcp.Strata(root, ROOT))
            plan, args = tools.install_plan("qwen", "IQ3_S", 32768, "no", None, None, None, None, None, None, None, str(local))
            self.assertIn("--gguf-dir", args)
            self.assertEqual(args[args.index("--gguf-dir") + 1], str(local))
            self.assertLessEqual(plan["disk_needed_gb"], 9)           # the 8 GB for preparing, not the download
            with self.assertRaises(mcp.ToolError):
                tools.install_plan("qwen", "IQ3_S", 32768, "no", None, None, None, None, None, None, None, str(local / "missing"))
