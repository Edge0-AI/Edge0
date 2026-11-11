"""A start through the owner runtime finds its engine in the component folder, or says clearly that there is none.

The engine here is the fake one the runtime tests use (a script behind a manifest-declared entrypoint), so this proves
the path the installed app takes: run config "component:strata-engine" -> `strata runtime start` -> the server resolves
the package -> the engine runs from the component folder -> stop ends it. It also proves the absence is a notice, not a
fetch: with an empty component folder and the no-acquire switch set, a start fails with the typed message, starts no
engine, and the state ends up stopped."""

import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_control import components, engine_component as ec, profiles, runtime

WINDOWS = os.name == "nt"
FAKE_ENGINE = (
    "import sys, os\nfrom pathlib import Path\n"
    "Path(os.environ['FAKE_ENGINE_MARK']).write_text(os.path.abspath(__file__) + '\\n' + ' '.join(sys.argv[1:]))\n"
    "print('READY 4096 stop', flush=True)\n"
    "for line in sys.stdin:\n"
    "    if line.strip() == 'QUIT': break\n"
    "    if line.startswith(('GEN ', 'BGEN ')):\n"
    "        slot = int(line.split()[1]) if line.startswith('BGEN ') else None\n"
    "        print('T 97', flush=True)\n"
    "        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)\n"
    "        if slot is not None: print(f'BADM {slot} 0', flush=True)\n")


def install_fake_engine(components_dir: Path) -> Path:
    folder = components_dir / "engine" / f"0.1.45-{components.host_platform()}-cuda"
    folder.mkdir(parents=True)
    (folder / "engine_main.py").write_text(FAKE_ENGINE, encoding="utf-8")
    if WINDOWS:
        entry = "engine.cmd"
        (folder / entry).write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine_main.py" %*\r\n'.encode())
    else:
        entry = "engine"
        (folder / entry).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
        (folder / entry).chmod(0o755)
    files = [{"path": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
              "bytes": (folder / name).stat().st_size} for name in (entry, "engine_main.py")]
    (folder / ec.MANIFEST).write_text(json.dumps({
        "schema_version": 1, "id": "strata-engine", "version": "0.1.45", "platform": components.host_platform(),
        "backend": "cuda", "entrypoint": entry, "files": files}), encoding="utf-8")
    return folder


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class ComponentStart(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-component-start-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.components = self.root / "components"
        self.components.mkdir()
        env = mock.patch.dict(os.environ, {
            "STRATA_PROFILE_DIR": str(self.root / "profiles"), "STRATA_RUNTIME_DIR": str(self.root / "runtime"),
            "STRATA_COMPONENTS_DIR": str(self.components), "STRATA_ENGINE_BACKEND": "cuda",
            "STRATA_NO_ACQUIRE": "1", "FAKE_ENGINE_MARK": str(self.root / "ran.txt")})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_API_KEY", None)
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (tokenizer / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
        (tokenizer / "merges.txt").write_text("a b")
        (tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": "component:strata-engine", "backend": "cuda",
                                            "tokenizer": str(tokenizer), "args": [], "cwd": str(self.root),
                                            "model_name": "fake", "host": "127.0.0.1", "port": 0}), encoding="utf-8")
        profiles.create("app", str(self.config))
        self.addCleanup(self._stop)

    def _stop(self):
        try:
            if runtime._read_state("app") is not None:
                runtime.stop("app", timeout=12)
        except (OSError, profiles.ProfileError):
            pass

    def _cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([*args, "--output", "json"])
        return code, json.loads(out.getvalue())

    def test_start_runs_the_engine_from_the_component_folder_and_stop_ends_it(self):
        folder = install_fake_engine(self.components)
        code, started = self._cli("runtime", "start", "--profile", "app", "--wait", "30", "--shutdown-timeout", "2")
        self.assertEqual(code, 0, started)
        self.assertEqual(started["state"], "ready")
        self.assertTrue(started["generation_ready"])
        ran = (self.root / "ran.txt").read_text().splitlines()[0]
        self.assertEqual(Path(ran).resolve(), (folder / "engine_main.py").resolve())
        status = self._cli("runtime", "status", "--profile", "app")[1]
        self.assertEqual(status["instance_id"], started["instance_id"])
        code, stopped = self._cli("runtime", "stop", "--profile", "app", "--wait", "20")
        self.assertEqual((code, stopped["state"]), (0, "stopped"))
        deadline = time.monotonic() + 5
        while runtime._read_state("app") is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNone(runtime._read_state("app"))

    def test_a_packaged_stub_engine_starts_and_a_changed_library_does_not(self):
        """The package is made by packaging/package_engine.py (schema 2: archs, min_driver, build_id), the start hashes
        every file: one flipped byte in a library the stub never loads still refuses the start, and no engine runs."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
        import package_engine
        src = self.root / "stub-src"
        src.mkdir()
        (src / "engine_main.py").write_text(FAKE_ENGINE, encoding="utf-8")
        if WINDOWS:
            entry = "strata.cmd"
            (src / entry).write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine_main.py" %*\r\n'.encode())
            library = "cudart64_13.dll"
        else:
            entry = "strata"
            (src / entry).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
            library = "libcudart.so.13"
        (src / library).write_bytes(b"a runtime library the stub never loads " * 200)
        (src / "BUILD.json").write_text(json.dumps({"version": "0.1.45"}))
        out = self.components / "engine"
        out.mkdir()
        folder = package_engine.build(src, "cuda", "0.1.45", components.host_platform(), out, entry=entry,
                                      archs=("sm_86", "compute_89"), min_driver={"nvidia": "580.65"},
                                      build_id="stub-1", variant="cuda13", include=["engine_main.py"])
        patches = (mock.patch.object(ec, "cuda_archs_present", return_value=["sm_86"]),
                   mock.patch.object(ec, "nvidia_driver_version", return_value="610.47"))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        code, started = self._cli("runtime", "start", "--profile", "app", "--wait", "30", "--shutdown-timeout", "2")
        log = self.root / "runtime" / "app.server.log"
        self.assertEqual(code, 0, (started, log.read_text(errors="replace")[-1500:] if log.exists() else "no server log"))
        self.assertEqual(started["state"], "ready")
        self.assertEqual(Path((self.root / "ran.txt").read_text().splitlines()[0]).resolve(),
                         (folder / "engine_main.py").resolve())
        self.assertEqual(self._cli("runtime", "stop", "--profile", "app", "--wait", "20")[1]["state"], "stopped")
        deadline = time.monotonic() + 5
        while runtime._read_state("app") is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        (self.root / "ran.txt").unlink()
        body = bytearray((folder / library).read_bytes())
        body[100] ^= 0xFF                                       # same size; only the full hash sees it
        (folder / library).write_bytes(bytes(body))
        code, result = self._cli("runtime", "start", "--profile", "app", "--wait", "20", "--shutdown-timeout", "2")
        self.assertEqual(code, 5, result)                      # the refusal is the first thing the user sees: no host, no server
        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "ran.txt").exists(), "a package with a changed library must not run")
        self.assertIn("integrity check", json.dumps(result))
        self.assertIn(library, json.dumps(result))
        self.assertIn("not started", json.dumps(result))
        self.assertIsNone(runtime._read_state("app"), "no runtime host may exist after the refusal")

    def test_no_engine_is_a_notice_and_starts_nothing(self):
        code, result = self._cli("runtime", "start", "--profile", "app", "--wait", "20", "--shutdown-timeout", "2")
        self.assertNotEqual(code, 0)
        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "ran.txt").exists(), "no engine may run")
        log = (self.root / "runtime" / "app.server.log")
        text = log.read_text(errors="replace") if log.exists() else ""
        self.assertIn("No engine for this PC", text + json.dumps(result))
        self.assertIn("never downloads", text + json.dumps(result))
        self.assertIn(runtime.status("app")["state"], ("stopped", "failed"))
        self.assertEqual(sorted(p.name for p in self.components.iterdir()), [],
                         "looking for an engine must not create or fetch anything")

    def test_a_tampered_engine_is_not_started(self):
        folder = install_fake_engine(self.components)
        (folder / "engine_main.py").write_text(FAKE_ENGINE + "# changed\n", encoding="utf-8")
        code, result = self._cli("runtime", "start", "--profile", "app", "--wait", "20", "--shutdown-timeout", "2")
        self.assertNotEqual(code, 0)
        self.assertFalse((self.root / "ran.txt").exists(), "a changed package must not run")

    def test_the_package_folder_is_first_on_the_library_path(self):
        from serve import server
        folder = install_fake_engine(self.components)
        cfg = server.with_component_engine({"exe": "component:strata-engine", "backend": "cuda", "lib_dirs": ["/other"]})
        self.assertEqual(Path(cfg["exe"]).parent.resolve(), folder.resolve())
        self.assertEqual(cfg["lib_dirs"][0], str(folder / ("engine.cmd" if WINDOWS else "engine")).rsplit(os.sep, 1)[0])
        self.assertEqual(cfg["lib_dirs"][1:], ["/other"])
        env = server.child_env(cfg)
        var = "PATH" if WINDOWS else "LD_LIBRARY_PATH"
        self.assertTrue(env[var].startswith(str(folder)))
        plain = {"exe": "/some/where/strata", "args": []}
        self.assertIs(server.with_component_engine(plain), plain)
        with self.assertRaises(SystemExit) as caught:
            for entry in (folder / "engine.cmd", folder / "engine"):
                entry.unlink(missing_ok=True)
            server.with_component_engine({"exe": "component:strata-engine", "backend": "cuda"})
        self.assertEqual(caught.exception.code, 5)

    def test_a_picture_reader_path_from_an_older_engine_folder_follows_the_installed_package(self):
        """After an engine update the folder name changes; the run config's absolute vision exe must not stop the server."""
        from serve import server
        folder = install_fake_engine(self.components)
        reader = folder / "strata-vision"
        reader.write_text("x", encoding="utf-8")
        gone = str(self.components / "engine" / "0.0.1-old" / "strata-vision")
        cfg = server.with_component_engine({"exe": "component:strata-engine", "backend": "cuda",
                                            "vision": {"exe": gone, "cpu_exe": gone, "mmproj": "/m.gguf"}})
        self.assertEqual(cfg["vision"]["exe"], str(reader))
        self.assertEqual(cfg["vision"]["cpu_exe"], str(reader))
        self.assertEqual(cfg["vision"]["mmproj"], "/m.gguf")
        ok = {"exe": "component:strata-engine", "backend": "cuda", "vision": {"exe": str(reader)}}
        self.assertEqual(server.with_component_engine(ok)["vision"], {"exe": str(reader)})
        none = server.with_component_engine({"exe": "component:strata-engine", "backend": "cuda", "vision": {"exe": gone.replace("strata-vision", "other")}})
        self.assertTrue(none["vision"]["exe"].endswith("other"))        # no such file in the package: left as it is

    def test_the_build_json_lib_dirs_follow_the_package_folder(self):
        """The Windows HIP package keeps the ROCm DLLs in rocm/bin (BUILD.json lib_dirs): strata.exe does not load without them on PATH."""
        from serve import server
        folder = install_fake_engine(self.components)
        (folder / "rocm" / "bin").mkdir(parents=True)
        (folder / "BUILD.json").write_text(json.dumps({"lib_dirs": ["rocm/bin", "../escape", "missing", "/abs"]}), encoding="utf-8")
        cfg = server.with_component_engine({"exe": "component:strata-engine", "backend": "cuda", "lib_dirs": ["/other"]})
        self.assertEqual(cfg["lib_dirs"], [str(folder), str(folder / "rocm" / "bin"), "/other"])
        env = server.child_env(cfg)
        var = "PATH" if WINDOWS else "LD_LIBRARY_PATH"
        self.assertTrue(env[var].startswith(str(folder) + os.pathsep + str(folder / "rocm" / "bin")))

    def test_doctor_names_the_missing_engine(self):
        from strata_control.doctor import inspect
        report = inspect(self.config)
        engine = next(row for row in report["components"] if row["name"] == "engine")
        self.assertEqual(engine["status"], "missing")
        install_fake_engine(self.components)
        report = inspect(self.config)
        engine = next(row for row in report["components"] if row["name"] == "engine")
        self.assertEqual(engine["status"], "ok")


if __name__ == "__main__":
    unittest.main()
