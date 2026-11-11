"""Starting and serving never acquire anything.

The owner's rule: a user must never download random packages from the internet while starting or serving. Everything
that installs, downloads or compiles lives behind an explicit command (SETUP, UPDATE, `strata install`). Three layers
are checked here:

* the import graph of the start/serve entry points does not reach installer code;
* no module on the start/serve side runs pip, git, cmake, npm or a package manager, or opens a URL, outside a short
  list of request-time and loopback uses, each with its reason;
* a real start (setup.main with an installed model) with the network and every subprocess mocked makes no request and
  runs no installer, and a start that WOULD need one stops with a typed error that names the explicit command.
"""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import setup
from strata_cli import cli
from strata_control import acquisition

ROOT = Path(__file__).resolve().parents[2]

#: Modules the start/serve entry points must never load.
INSTALLER_MODULES = {"setup", "pip", "ensurepip", "venv", "huggingface_hub", "modelscope", "metal.setup_mac",
                     "sycl.setup_intel"}
#: Files that may be loaded on the start/serve side besides the control, cli and serve packages.
ALLOWED_TOOLS = {"tools.strata_tokenizer", "tools.gguf_reader"}   # read-only: a tokenizer and a GGUF header reader

#: argv words that mean "acquire": a subprocess whose literal argv contains one of these is an installer.
INSTALL_WORDS = {"pip", "pip3", "git", "cmake", "ninja", "npm", "npx", "pnpm", "yarn", "winget", "apt-get", "apt",
                 "dnf", "pacman", "brew", "curl", "wget", "choco"}

#: (file, call) -> why a network call there is not an acquisition. Request-time and loopback only.
NETWORK_ALLOWED = {
    ("serve/server.py", "urlopen"): "an image_url a client sends in a chat request is fetched at request time",
    ("serve/mcp.py", "urlopen"): "the MCP servers the user configured, at request time",
    ("strata_control/tune_api.py", "urlopen"): "the loopback engine/server of a tuning job the user started",
    ("strata_control/tune_session.py", "urlopen"): "the loopback engine (127.0.0.1) the auto-tuner a user started measures",
    ("strata_control/loopback.py", "urlopen"): "the one opener for this computer's own servers (no proxy); any other address is passed on unchanged",
    ("strata_control/components.py", "urlopen"): "the component fetcher: only `strata install`/UPDATE call it",
    ("strata_control/dsv41_engine.py", "urlopen"): "the DeepSeek engine's one 10-second 'is the feed reachable' check, called only by strata launch and "
                                                  "setup (test_the_deepseek_engine_check_is_reached_only_from_launch_and_setup); never while serving",
}
#: Modules that import the installer on purpose, with the explicit command that reaches them.
INSTALLER_IMPORT_ALLOWED = {
    "strata_control/calibration.py": "`strata engine calibrate`: an explicit tuning command, imports setup lazily",
}

SERVING_DIRS = ("serve", "strata_cli", "strata_control")


def _sources():
    for d in SERVING_DIRS:
        for p in sorted((ROOT / d).glob("*.py")):
            if p.name.startswith("test_") or p.name == "__init__.py":
                continue
            yield p
    yield ROOT / "strata.py"


class ImportGraphTests(unittest.TestCase):
    def test_start_and_serve_modules_do_not_load_installer_code(self):
        script = (
            "import sys, os, json\n"
            "sys.path.insert(0, os.getcwd())\n"
            "import strata_cli.cli as c\n"
            "import serve.server\n"
            "from strata_control import runtime\n"
            "c.parser()\n"
            "root = os.getcwd()\n"
            "mods = {n: os.path.relpath(m.__file__, root).replace(os.sep, '/') for n, m in list(sys.modules.items())\n"
            "        if getattr(m, '__file__', None) and os.path.abspath(m.__file__).startswith(root)}\n"
            "print(json.dumps({'mods': mods, 'names': sorted(sys.modules)}))\n")
        r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        got = json.loads(r.stdout.strip().splitlines()[-1])
        loaded = set(got["names"])
        self.assertEqual(sorted(loaded & INSTALLER_MODULES), [])
        for name, rel in got["mods"].items():
            top = rel.split("/")[0]
            self.assertIn(top, ("serve", "strata_cli", "strata_control"),
                          f"{name} ({rel}) is loaded by starting/serving but is not part of it")
        self.assertNotIn("strata_control.calibration", got["mods"])
        self.assertNotIn("strata_control.components", got["mods"])

    def test_static_imports_of_the_installer_are_explicit_commands_only(self):
        found = {}
        for p in _sources():
            rel = p.relative_to(ROOT).as_posix()
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    top = n.split(".")[0]
                    if top == "setup" or n.startswith(("tools.", "metal.", "sycl.")) and n not in ALLOWED_TOOLS:
                        found.setdefault(rel, []).append(n)
        for rel in found:
            self.assertIn(rel, INSTALLER_IMPORT_ALLOWED, f"{rel} imports installer code: {found[rel]}")

    def test_the_deepseek_engine_check_is_reached_only_from_launch_and_setup(self):
        """strata_control/dsv41_engine.py (the engine's self-upgrade, 0.2.4.2) reaches the network; serving must never import it."""
        importers = set()
        for d in SERVING_DIRS:
            for p in sorted((ROOT / d).glob("*.py")):
                if p.name.startswith("test_") or p.name == "dsv41_engine.py":
                    continue
                tree = ast.parse(p.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else                         [node.module] + [f"{node.module}.{a.name}" for a in node.names] if isinstance(node, ast.ImportFrom) and node.module else []
                    if any(n.endswith("dsv41_engine") for n in names):
                        importers.add(p.relative_to(ROOT).as_posix())
        importers.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "strata_cli" / "launch").glob("*.py")
                         if "dsv41_engine" in p.read_text(encoding="utf-8") and not p.name.startswith("test_"))
        importers.discard("strata_cli/launch/runner.py")
        if "dsv41_engine" in (ROOT / "setup.py").read_text(encoding="utf-8"):
            importers.add("setup.py")
        importers.discard("setup.py")
        self.assertEqual(sorted(importers), [], "only strata_cli/launch/runner.py and setup.py may use the DeepSeek engine check")

    def test_no_installer_subprocess_or_network_call_on_the_serving_side(self):
        bad = []
        for p in _sources():
            rel = p.relative_to(ROOT).as_posix()
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                if name in ("urlopen", "urlretrieve") and (rel, name) not in NETWORK_ALLOWED \
                        and (rel, "urlopen") not in NETWORK_ALLOWED:
                    bad.append(f"{rel}:{node.lineno} {name}")
                if name in ("run", "call", "Popen", "check_call", "check_output") and node.args:
                    argv = node.args[0]
                    words = []
                    if isinstance(argv, (ast.List, ast.Tuple)):
                        words = [e.value for e in argv.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    elif isinstance(argv, ast.Constant) and isinstance(argv.value, str):
                        words = argv.value.split()
                    hit = INSTALL_WORDS & {Path(w).stem if w.endswith(".exe") else w for w in words}
                    if hit:
                        bad.append(f"{rel}:{node.lineno} subprocess {sorted(hit)}")
        self.assertEqual(bad, [])
        for key, why in NETWORK_ALLOWED.items():
            self.assertTrue(why)


class _Recorder:
    """Stands in for the network and for every way of starting a program; records what a start tries."""

    def __init__(self):
        self.network = []
        self.programs = []

    def urlopen(self, req, *a, **k):
        self.network.append(getattr(req, "full_url", str(req)))
        raise AssertionError("a start reached the network: " + self.network[-1])

    def program(self, argv, *a, **k):
        argv = [str(x) for x in (argv if isinstance(argv, (list, tuple)) else str(argv).split())]
        self.programs.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def installers(self):
        return [a for a in self.programs
                if INSTALL_WORDS & {Path(w).stem.lower() if w.lower().endswith(".exe") else w for w in a}
                or "pip" in a or "-m" in a and "pip" in a]


class StartNeverAcquiresTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "Strata"
        self.root.mkdir()
        self.data = Path(self.tmp.name) / "Strata-data"
        (self.data / "models").mkdir(parents=True)
        self.gguf = self.data / "models" / "m.gguf"
        self.gguf.write_bytes(b"GGUF")
        eng = self.root / "engine"
        eng.mkdir()
        (eng / setup.EXE).write_bytes(b"engine")
        # an engine older than this Strata expects: a start used to download the new one
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.1", "source": "prebuilt", "backend": "cuda",
                                                    "archs": [86]}))
        self.cfg = self.root / "strata-q2_0.json"
        self.cfg.write_text(json.dumps({
            "exe": str(eng / setup.EXE), "args": ["--native", str(self.gguf), "--max-context", "8192"],
            "tokenizer": str(self.data / "tok.json"), "model_name": "Q2_0", "port": 8080}))
        self.settings_home = Path(self.tmp.name) / "appdata"
        self.settings_home.mkdir()
        env = {"APPDATA": str(self.settings_home), "XDG_CONFIG_HOME": str(self.settings_home)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        setup.settings_path().parent.mkdir(parents=True, exist_ok=True)
        setup.settings_path().write_text(json.dumps({"data_dir": str(self.data), "installs": [str(self.root)]}))
        self.settings_before = setup.settings_path().read_bytes()
        # pip_install keeps its stamp in sys.prefix: never in the interpreter running the tests
        prefix = mock.patch.object(sys, "prefix", str(self.root))
        prefix.start()
        self.addCleanup(prefix.stop)

    def run_setup(self, rec, *argv, gpus=None):
        out = io.StringIO()
        started = []
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(setup, "ROOT", self.root))
            st.enter_context(mock.patch.object(sys, "argv", ["setup.py", *argv]))
            st.enter_context(mock.patch.object(urllib.request, "urlopen", rec.urlopen))
            for n in ("run", "call", "Popen", "check_call", "check_output"):
                st.enter_context(mock.patch.object(subprocess, n, rec.program))
            st.enter_context(mock.patch.object(setup, "gpus", lambda: gpus or []))
            st.enter_context(mock.patch.object(setup, "call_server", create=True))
            st.enter_context(mock.patch.object(setup.subprocess, "call", lambda cmd, *a, **k: (
                started.append(cmd), setattr(self, "env_of_start", (k.get("env") or {}).get(acquisition.ENV)), 0)[-1]))
            st.enter_context(contextlib.redirect_stdout(out))
            try:
                code = setup.main()
            except SystemExit as e:
                code = e.code
            except acquisition.AcquisitionRefused as e:
                code = e
        return code, out.getvalue(), started

    def test_a_start_with_an_old_engine_makes_no_request_and_runs_no_installer(self):
        rec = _Recorder()
        code, text, started = self.run_setup(rec, "--yes", "--no-browser")
        self.assertEqual(code, 0, text)
        self.assertEqual(rec.network, [])
        self.assertEqual(rec.installers(), [])
        self.assertEqual(len(started), 1, text)                        # the server was started
        self.assertIn("server.py", " ".join(started[0]))
        self.assertEqual(self.env_of_start, "1")                         # the server is told, too
        self.assertIn("is older than this Strata expects", text)
        self.assertTrue("UPDATE" in text or "update.sh" in text, text)     # UPDATE.bat on Windows, ./update.sh elsewhere

    def test_a_start_moves_copies_and_creates_nothing(self):
        before = sorted(p.relative_to(self.tmp.name).as_posix() for p in Path(self.tmp.name).rglob("*"))
        stamps = {p: p.stat().st_mtime_ns for p in Path(self.tmp.name).rglob("*") if p.is_file()}
        rec = _Recorder()
        code, text, _ = self.run_setup(rec, "--yes", "--no-browser")
        self.assertEqual(code, 0, text)
        after = sorted(p.relative_to(self.tmp.name).as_posix() for p in Path(self.tmp.name).rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(setup.settings_path().read_bytes(), self.settings_before)
        for p, ns in stamps.items():
            if p != self.cfg:                                          # a start touches the config (most recently used)
                self.assertEqual(p.stat().st_mtime_ns, ns, p)

    def test_a_card_the_engine_has_no_code_for_is_a_typed_error_naming_the_command(self):
        rec = _Recorder()
        card = {"index": 0, "name": "RTX 5090", "arch": "120", "vram_gb": 32.0, "driver": "580.0"}
        with mock.patch.object(setup, "engine_runs_on", lambda g, tk=13: False), \
                mock.patch.object(setup, "old_gpus_opt_in", lambda *a, **k: []):
            code, text, started = self.run_setup(rec, "--yes", "--no-browser", gpus=[card])
        self.assertIsInstance(code, acquisition.AcquisitionRefused)
        self.assertIn("RTX 5090", str(code))
        self.assertIn("setup", str(code).lower())
        self.assertEqual((rec.network, rec.installers(), started), ([], [], []))

    def test_main_script_reports_the_refusal_and_exits_5(self):
        e = acquisition.AcquisitionRefused("an engine for sm_120")
        self.assertEqual(e.exit_code, 5)
        self.assertEqual(e.code, "acquisition_refused")
        self.assertIn("never install or download", str(e))

    def test_every_acquiring_primitive_refuses_while_a_model_starts(self):
        rec = _Recorder()
        calls = [
            lambda: setup.download("https://example.invalid/x.bin", self.root / "x.bin"),
            lambda: setup.download("file:///nonexistent", self.root / "y.bin"),
            lambda: setup.pip_install(["definitely-not-installed==1.0"], "a test package"),
            lambda: setup.run([sys.executable, "-m", "pip", "install", "x"]),
            lambda: setup._urlopen(urllib.request.Request("https://example.invalid/")),
            lambda: setup.reachable("https://example.invalid/"),
            lambda: setup.get_prebuilt("https://example.invalid/", {"arch": "86"}, "none"),
            lambda: setup.cmake_build(self.root, self.root / "b", "strata", [], None, ""),
            lambda: setup.pip_cuda_libs(13),
        ]
        with mock.patch.object(urllib.request, "urlopen", rec.urlopen), \
                mock.patch.object(subprocess, "run", rec.program), mock.patch.object(subprocess, "Popen", rec.program):
            with acquisition.forbidden():
                for i, call in enumerate(calls):
                    with self.subTest(i=i):
                        with self.assertRaises(acquisition.AcquisitionRefused) as cm:
                            call()
                        self.assertIn("Run:", str(cm.exception))
        self.assertEqual((rec.network, rec.programs), ([], []))

    def test_the_environment_switch_forbids_a_whole_process(self):
        with mock.patch.dict(os.environ, {acquisition.ENV: "1"}):
            with self.assertRaises(acquisition.AcquisitionRefused):
                setup.pip_install(["definitely-not-installed==1.0"], "a test package")

    def test_update_still_installs(self):
        """The explicit commands keep working: outside a start, the install step runs."""
        installed = []
        with tempfile.TemporaryDirectory() as prefix, mock.patch.object(sys, "prefix", prefix),                 mock.patch.object(setup, "run", lambda cmd, **k: installed.append([str(c) for c in cmd])):
            setup.pip_install(["somepkg==1.0"], "a test package")
        self.assertEqual(len(installed), 1)
        self.assertIn("pip", installed[0])


class CliServeIsOfflineTests(unittest.TestCase):
    def test_serve_plan_reaches_no_network_and_runs_no_program(self):
        rec = _Recorder()
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "strata-x.json"
            gguf = Path(d) / "m.gguf"
            gguf.write_bytes(b"GGUF")
            tok = Path(d) / "t.json"
            tok.write_text("{}")
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"x")
            cfg.write_text(json.dumps({"exe": str(exe), "args": ["--native", str(gguf)], "tokenizer": str(tok)}))
            out = io.StringIO()
            with mock.patch.object(urllib.request, "urlopen", rec.urlopen), \
                    mock.patch.object(subprocess, "run", rec.program), mock.patch.object(subprocess, "Popen", rec.program), \
                    mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(Path(d) / "profiles")}), \
                    contextlib.redirect_stdout(out):
                code = cli.main(["serve", "--profile", str(cfg), "--plan", "--output", "json"])
        self.assertEqual(code, 0, out.getvalue())
        # Reading the hardware is not acquiring anything: nvidia-smi / hipInfo / rocminfo and PowerShell's Get-CimInstance (display adapters) only
        # print what is here. Whether a PC runs them depends on the PC (a machine with nvidia-smi on PATH does), so the check is that nothing else runs.
        probes = {"nvidia-smi", "hipinfo", "rocminfo", "rocm-smi", "vulkaninfo"}

        def read_only(argv):
            stem = Path(argv[0]).stem.lower()
            return stem in probes or (stem in ("powershell", "pwsh") and any("Get-CimInstance" in a for a in argv))
        self.assertEqual(rec.network, [])
        self.assertEqual([a for a in rec.programs if not read_only(a)], [])
        self.assertNotIn("setup", sys.modules.get("strata_cli.cli").__dict__.get("__name__", ""))

    def test_strata_install_without_the_installer_is_a_typed_error(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": d}), \
                mock.patch.object(cli, "ROOT", Path(d)), contextlib.redirect_stderr(err):
            code = cli.main(["install"])
        self.assertEqual(code, 5)
        self.assertIn("installer", err.getvalue())


if __name__ == "__main__":
    unittest.main()
