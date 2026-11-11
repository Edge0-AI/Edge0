"""The close-path matrix: every way a Strata server or app can end, and what must be true afterwards.

After each case: no process of the tree that was running is left, the port is free, and the runtime's lease can be
taken again (nothing holds it). With a real engine, `--vram` also checks that the GPU's memory came back.

Run it directly (the pytest wrapper test_lifecycle_matrix.py runs the fake-engine cases on every platform):

    python tests/terminal/lifecycle_matrix.py --scenarios all --report matrix.json
    python tests/terminal/lifecycle_matrix.py --engine real --config /data/run.json --vram --scenarios serve-sigterm

The engine is a small script behind the same manifest-declared entry point a real engine package has, unless
`--engine real` names a run config. It never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from strata_control import components, engine_component as ec, procs, profiles, runtime  # noqa: E402

WINDOWS = os.name == "nt"
LINUX = sys.platform.startswith("linux")
BACKEND = "metal" if sys.platform == "darwin" else "cuda"      # the family the fake package claims on this OS

# Control words (STOP) are taken on a thread of their own, as the real engine does; everything else in order.
FAKE_ENGINE = r'''
import os, sys, threading, time
from pathlib import Path
mark = Path(os.environ["FAKE_ENGINE_MARK"])
(mark / "engine.pid").write_text(str(os.getpid()))
stop = threading.Event()
inbox = []
cv = threading.Condition()
def note(name): (mark / name).write_text(str(time.time()))
def reader():
    for raw in sys.stdin:
        word = raw.split()[0] if raw.split() else ""
        if word == "STOP":
            stop.set(); note("stop"); continue
        with cv:
            inbox.append(raw); cv.notify()
    with cv:
        inbox.append(None); cv.notify()
threading.Thread(target=reader, daemon=True).start()
print("READY 4096 stop", flush=True)
while True:
    with cv:
        while not inbox: cv.wait()
        line = inbox.pop(0)
    if line is None:                       # the server closed our stdin
        note("eof"); sys.exit(0)
    word = line.split()[0] if line.split() else ""
    if word == "QUIT":
        note("quit"); sys.exit(0)
    if word in ("GEN", "BGEN"):
        slow = int(os.environ.get("FAKE_ENGINE_SLOW_STEPS", "0"))
        slot = int(line.split()[1]) if word == "BGEN" else None
        steps = slow if (slow and not os.environ.get("FAKE_ENGINE_FAST_FIRST")) else 1
        stop.clear()
        note("busy")
        for _ in range(steps):
            print("T 97", flush=True)
            if stop.wait(0.2):
                break
        print("DONE 1 1 1.0 1.0 length 0 0 0", flush=True)
        if slot is not None: print(f"BADM {slot} 0", flush=True)
'''


if WINDOWS:   # a parent that ignores Ctrl+C hands that to its children; the matrix must start servers that do not
    import ctypes
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def install_fake_engine(components_dir: Path) -> Path:
    folder = components_dir / "engine" / f"0.1.45-{components.host_platform()}-{BACKEND}"
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
        "backend": BACKEND, "entrypoint": entry, "files": files}), encoding="utf-8")
    return folder


def vram_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return sum(int(v) for v in out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Rig:
    """A throwaway Strata home: state, runtime, profile and (fake or real) engine config."""

    def __init__(self, root: Path, *, real_config: Path | None = None, real_components: Path | None = None,
                 slow_steps: int = 0):
        self.root = root
        self.real = real_config is not None
        self.port = free_port()
        self.components = root / "components"
        self.components.mkdir()
        self.mark = root / "mark"
        self.mark.mkdir()
        self.state = root / "state"
        self.state.mkdir()
        self.env = {**os.environ, "STRATA_PROFILE_DIR": str(root / "profiles"), "STRATA_RUNTIME_DIR": str(root / "runtime"),
                    "STRATA_COMPONENTS_DIR": str(self.components), "STRATA_ENGINE_BACKEND": BACKEND,
                    "STRATA_NO_ACQUIRE": "1", "FAKE_ENGINE_MARK": str(self.mark), "LOCALAPPDATA": str(self.state),
                    "XDG_STATE_HOME": str(self.state), "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
                    "FAKE_ENGINE_SLOW_STEPS": str(slow_steps)}
        self.env.pop("STRATA_API_KEY", None)
        for key in ("STRATA_PROFILE_DIR", "STRATA_RUNTIME_DIR", "LOCALAPPDATA", "XDG_STATE_HOME", "STRATA_COMPONENTS_DIR",
                    "STRATA_NO_ACQUIRE", "STRATA_ENGINE_BACKEND", "FAKE_ENGINE_MARK", "FAKE_ENGINE_SLOW_STEPS"):
            os.environ[key] = self.env[key]               # this process reads the same folders (profiles, runtime)
        if real_components is not None:
            self.env["STRATA_COMPONENTS_DIR"] = os.environ["STRATA_COMPONENTS_DIR"] = str(real_components)
        if real_config is not None:
            cfg = json.loads(real_config.read_text(encoding="utf-8"))
            cfg.update({"host": "127.0.0.1", "port": self.port})
            if cfg.get("log"):
                cfg["log"] = str(root / "engine.log")                  # never the real install's log
            self.config = root / "run.json"
            self.config.write_text(json.dumps(cfg), encoding="utf-8")
        else:
            install_fake_engine(self.components)
            from tools.strata_tokenizer import BYTE_TO_UNICODE
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            (tokenizer / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
            (tokenizer / "merges.txt").write_text("a b")
            (tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
            self.config = root / "run.json"
            self.config.write_text(json.dumps({"exe": "component:strata-engine", "backend": BACKEND,
                                               "tokenizer": str(tokenizer), "args": [], "cwd": str(root),
                                               "model_name": "fake", "host": "127.0.0.1", "port": self.port}),
                                   encoding="utf-8")
        profiles.create("app", str(self.config))

    # ------------------------------------------------------------------------------------- starting
    def serve_argv(self) -> list[str]:
        return [sys.executable, str(ROOT / "strata.py"), "serve", "--profile", "app", "--port", str(self.port),
                "--host", "127.0.0.1", "--shutdown-timeout", "5"]

    def wait_ready(self, proc=None, seconds=None) -> None:
        seconds = seconds or (480 if self.real else 120)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"the server exited with {proc.returncode} before it was ready")
            if port_open(self.port):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3).read()
                    return
                except OSError:
                    pass
            time.sleep(0.25)
        raise RuntimeError("the server was not ready in time")

    def tree(self, pid: int) -> list[procs.Ident]:
        """The process and everything below it, with the facts that tell them from a later process."""
        found = []
        try:
            family = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
        except psutil.Error:
            return found
        for p in family:
            token = procs.start_token(p.pid)
            if token:
                found.append(procs.Ident(p.pid, token, tuple(procs.command_line(p.pid) or [])))
        return found

    def engine_idents(self) -> list[procs.Ident]:
        """Every live process that is this run's engine (found by its recorded launch, not by a name)."""
        return [procs.Ident(r["pid"], r["token"], tuple(r["cmd"])) for r in procs.records()
                if r.get("role") in ("engine", "server") and procs.start_token(r["pid"]) == r["token"]]

    # ------------------------------------------------------------------------------------- checking
    def survivors(self, idents) -> list[str]:
        return [f"{i.pid} {' '.join(i.cmd)[:90]}" for i in idents if procs.start_token(i.pid) == i.token]

    def wait_gone(self, idents, seconds=45) -> list[str]:
        deadline = time.monotonic() + seconds
        left = self.survivors(idents)
        while left and time.monotonic() < deadline:
            time.sleep(0.25)
            left = self.survivors(idents)
        return left

    def lease_free(self) -> bool:
        try:
            fd = runtime.acquire_activity_lease(shared=False)
        except Exception:  # noqa: BLE001
            return False
        runtime._unlock_file(fd)
        os.close(fd)
        return True

    def marks(self) -> list[str]:
        return sorted(p.name for p in self.mark.iterdir())

    def clear_marks(self) -> None:
        for p in self.mark.iterdir():
            p.unlink()


# ------------------------------------------------------------------------------------------- actions

def _default_signals() -> None:
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)


def spawn_serve(rig: Rig, *, new_console: bool = False, session: bool = True) -> subprocess.Popen:
    log = open(rig.root / "serve.log", "ab")
    kw: dict = {}
    if WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_CONSOLE if new_console else subprocess.CREATE_NEW_PROCESS_GROUP
        if new_console:
            info = subprocess.STARTUPINFO()
            info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            info.wShowWindow = 7                               # minimized, not activated
            kw["startupinfo"] = info
    elif session:
        kw["start_new_session"] = True                         # its own process group, like a job in a shell
        kw["preexec_fn"] = _default_signals                    # a harness started in the background inherits "ignore"
    proc = subprocess.Popen(rig.serve_argv(), cwd=ROOT, env=rig.env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, **kw)
    log.close()
    return proc


def windows_console_event(pid: int, kind: str) -> None:
    """Ctrl+C ('ctrl-c') or the window's close box ('close') on the console `pid` is attached to."""
    code = (
        "import ctypes, sys, time\n"
        "k = ctypes.windll.kernel32\n"
        "k.FreeConsole()\n"
        f"assert k.AttachConsole({pid}), 'AttachConsole failed'\n"
        "if sys.argv[1] == 'ctrl-c':\n"
        "    k.SetConsoleCtrlHandler(None, 1)\n"
        "    k.GenerateConsoleCtrlEvent(0, 0)\n"
        "    time.sleep(1.5)\n"
        "else:\n"
        "    hwnd = k.GetConsoleWindow()\n"
        "    assert hwnd, 'no console window'\n"
        "    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)\n"
        "    time.sleep(1.0)\n"
    )
    # for "close" the helper is attached to the console that is closing, so the system ends it too: that exit is expected
    done = subprocess.run([sys.executable, "-c", code, kind], creationflags=subprocess.DETACHED_PROCESS, timeout=30)
    if kind == "ctrl-c" and done.returncode != 0:
        raise RuntimeError(f"could not send Ctrl+C (helper exit {done.returncode})")


def kill_hard(pid: int) -> None:
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)      # this one process, not its tree
    else:
        os.kill(pid, signal.SIGKILL)


def pty_close(rig: Rig):
    """A terminal: the server runs as the session leader of a pseudo-terminal; closing the master is the window closing."""
    import pty
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        _default_signals()
        os.execve(sys.executable, rig.serve_argv(), rig.env)
    return pid, master


# ------------------------------------------------------------------------------------------ scenarios

class Result(dict):
    pass


def finish(rig: Rig, name: str, idents, *, expect_graceful: bool | None, note: str = "", vram_before=None,
           vram: bool = False, wait: float = 45, stopped_text: str | None = None, vram_during=None) -> Result:
    left = rig.wait_gone(idents, wait)
    problems = []
    if left:
        problems.append(f"processes left: {left}")
    deadline = time.monotonic() + 10                      # a closing process's sockets go a moment after it
    while port_open(rig.port) and time.monotonic() < deadline:
        time.sleep(0.2)
    if port_open(rig.port):
        problems.append("port still open")
    if not rig.lease_free():
        problems.append("lease still held")
    marks = rig.marks()
    if expect_graceful is True and not rig.real and not ({"quit", "eof"} & set(marks)):
        problems.append(f"engine was not asked to quit (marks {marks})")
    if vram and vram_before is not None:
        deadline = time.monotonic() + 30
        used = vram_used_mib()
        while used is not None and used > vram_before + 300 and time.monotonic() < deadline:
            time.sleep(1)
            used = vram_used_mib()
        if used is not None and used > vram_before + 300:
            problems.append(f"VRAM not freed: {used} MiB used, {vram_before} before")
        note += f" vram {vram_before}->{vram_during}->{used} MiB"
        if vram_during is not None and vram_during <= vram_before + 500:
            problems.append(f"the engine never held GPU memory ({vram_during} MiB while running), so the check proves nothing")
    return Result(name=name, ok=not problems, problems=problems, graceful=("quit" in marks or "eof" in marks),
                  marks=marks, note=note.strip(), stopped=stopped_text)


def serve_case(rig: Rig, name: str, action, *, graceful: bool | None, vram: bool = False,
               new_console: bool = False, wait: float = 45) -> Result:
    rig.clear_marks()
    before = vram_used_mib() if vram else None
    proc = spawn_serve(rig, new_console=new_console)
    try:
        rig.wait_ready(proc)
        idents = rig.tree(proc.pid)
        idents += [i for i in rig.engine_idents() if i.pid not in {x.pid for x in idents}]
        if not any("engine" in " ".join(i.cmd) or "--serve" in " ".join(i.cmd) for i in idents):
            raise RuntimeError(f"no engine in the tree: {[' '.join(i.cmd)[:60] for i in idents]}")
        during = vram_used_mib() if vram else None
        t0 = time.monotonic()
        action(rig, proc)
        out = finish(rig, name, idents, expect_graceful=graceful, vram_before=before, vram=vram, wait=wait,
                     vram_during=during)
        out["seconds"] = round(time.monotonic() - t0, 1)
        text = (rig.root / "serve.log").read_text(errors="replace") if (rig.root / "serve.log").exists() else ""
        out["stopped"] = "[strata] stopped" in text
        if graceful is True and not out["stopped"]:
            out["problems"].append("the server did not print its clean-stop line"); out["ok"] = False
        return out
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for i in rig.tree(proc.pid):
            if procs.start_token(i.pid) == i.token:
                try:
                    psutil.Process(i.pid).kill()
                except psutil.Error:
                    pass
        keep = os.environ.get("STRATA_MATRIX_LOGS")                 # keep the server's and engine's logs as evidence
        if keep:
            Path(keep).mkdir(parents=True, exist_ok=True)
            for log in ("serve.log", "engine.log"):
                if (rig.root / log).exists():
                    (Path(keep) / f"{name}.{log}").write_bytes((rig.root / log).read_bytes())
        (rig.root / "serve.log").write_bytes(b"")


def sigint_group(rig, proc):                      # Ctrl+C in a terminal: SIGINT to the whole foreground group
    os.killpg(proc.pid, signal.SIGINT)


def sigterm(rig, proc):
    os.kill(proc.pid, signal.SIGTERM)


def sighup(rig, proc):                            # what ssh dropping the connection does to the shell's job
    os.killpg(proc.pid, signal.SIGHUP)


def kill9_parent(rig, proc):
    kill_hard(proc.pid)


def win_ctrl_break(rig, proc):
    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)


def win_ctrl_c(rig, proc):
    windows_console_event(proc.pid, "ctrl-c")


def win_console_close(rig, proc):
    windows_console_event(proc.pid, "close")


def busy_sigterm(rig, proc):
    """A request is running when the stop arrives: it must be cancelled, not waited for."""
    import threading
    started = threading.Event()

    def ask():
        try:
            body = json.dumps({"model": "fake", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4000 if rig.real else 200,
                               "stream": True}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{rig.port}/v1/chat/completions", data=body,
                                         headers={"content-type": "application/json"})
            started.set()
            urllib.request.urlopen(req, timeout=60).read()
        except OSError:
            pass
    threading.Thread(target=ask, daemon=True).start()
    started.wait(5)
    deadline = time.monotonic() + 10
    while not rig.real and "busy" not in rig.marks() and time.monotonic() < deadline:
        time.sleep(0.1)
    if rig.real:
        time.sleep(4)                               # the real engine is reading the prompt / generating by now
    (sigterm if not WINDOWS else win_ctrl_break)(rig, proc)


def nohup_case(rig: Rig, name: str = "serve-nohup-keeps-serving", *, vram: bool = False) -> Result:
    """`nohup strata serve &` asks to outlive its terminal: a hangup that was ignored at start is still ignored, and the
    server stops on SIGTERM as usual. (An ordinary server stops on SIGHUP: serve-sighup.)"""
    rig.clear_marks()
    before = vram_used_mib() if vram else None
    log = open(rig.root / "serve.log", "ab")

    def ignore_hangup():
        _default_signals()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    proc = subprocess.Popen(rig.serve_argv(), cwd=ROOT, env=rig.env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=ignore_hangup)
    log.close()
    try:
        rig.wait_ready(proc)
        idents = rig.tree(proc.pid)
        idents += [i for i in rig.engine_idents() if i.pid not in {x.pid for x in idents}]
        during = vram_used_mib() if vram else None
        os.killpg(proc.pid, signal.SIGHUP)                  # the terminal closes
        time.sleep(3)
        problems = []
        if proc.poll() is not None or not port_open(rig.port):
            problems.append("the server stopped on a hangup it was told to ignore")
        t0 = time.monotonic()
        os.kill(proc.pid, signal.SIGTERM)
        out = finish(rig, name, idents, expect_graceful=True, vram_before=before, vram=vram, vram_during=during)
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["problems"] = problems + out["problems"]
        out["ok"] = not out["problems"]
        return out
    finally:
        if proc.poll() is None:
            proc.kill()
        for i in rig.tree(proc.pid):
            if procs.start_token(i.pid) == i.token:
                try:
                    psutil.Process(i.pid).kill()
                except psutil.Error:
                    pass


def pty_close_case(rig: Rig, name: str = "serve-terminal-closed", *, vram: bool = False) -> Result:
    rig.clear_marks()
    before = vram_used_mib() if vram else None
    pid, master = pty_close(rig)
    try:
        class _P:                                   # just enough of Popen for wait_ready
            returncode = None

            def poll(self_inner):
                done, _ = os.waitpid(pid, os.WNOHANG)
                self_inner.returncode = 0 if done else None
                return 0 if done else None
        rig.wait_ready(_P())
        idents = rig.tree(pid)
        idents += [i for i in rig.engine_idents() if i.pid not in {x.pid for x in idents}]
        during = vram_used_mib() if vram else None
        t0 = time.monotonic()
        os.close(master)                            # the window is closed
        out = finish(rig, name, idents, expect_graceful=True, vram_before=before, vram=vram, vram_during=during)
        out["seconds"] = round(time.monotonic() - t0, 1)
        return out
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass


# --- the detached service (`strata runtime start`) ------------------------------------------------

def runtime_start(rig: Rig, die_with: int | None = None) -> dict:
    argv = [sys.executable, str(ROOT / "strata.py"), "runtime", "start", "--profile", "app", "--wait", "400" if rig.real else "240",
            "--shutdown-timeout", "5", "--output", "json"]
    if die_with:
        argv += ["--die-with-pid", str(die_with)]
    out = subprocess.run(argv, cwd=ROOT, env=rig.env, capture_output=True, text=True, timeout=500)
    try:
        return json.loads(out.stdout)
    except ValueError:
        raise RuntimeError(f"runtime start gave {out.returncode}: {out.stdout[-300:]} {out.stderr[-300:]}") from None


def runtime_stop(rig: Rig) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "strata.py"), "runtime", "stop", "--profile", "app", "--wait", "60",
                          "--output", "json"], cwd=ROOT, env=rig.env, capture_output=True, text=True, timeout=120)
    try:
        return json.loads(out.stdout)
    except ValueError:
        return {"ok": False, "raw": out.stdout[-300:] + out.stderr[-300:]}


def owner_pid(rig: Rig) -> int:
    return runtime._read_state("app")["host_pid"]


def sleeper_process() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def runtime_case(rig: Rig, name: str, what: str, *, vram: bool = False) -> Result:
    rig.clear_marks()
    before = vram_used_mib() if vram else None
    holder = sleeper_process() if what == "die-with" else None
    try:
        started = runtime_start(rig, die_with=holder.pid if holder else None)
        if started.get("state") != "ready":
            raise RuntimeError(f"not ready: {started}")
        owner = owner_pid(rig)
        idents = rig.tree(owner)
        during = vram_used_mib() if vram else None
        t0 = time.monotonic()
        graceful = True
        note = ""
        if what == "stop":
            res = runtime_stop(rig)
            note = f"stop -> {res.get('state')}"
        elif what == "terminal-gone":
            # the command that started it is long gone; the service must still be serving
            time.sleep(1.5)
            note = "still serving: " + str(port_open(rig.port))
            if not port_open(rig.port):
                raise RuntimeError("the background service died with its terminal")
            res = runtime_stop(rig)
        elif what == "owner-killed":
            kill_hard(owner)
            graceful = None
        elif what == "server-killed":
            server = next(i for i in idents if "serve.server" in " ".join(i.cmd))
            kill_hard(server.pid)
            graceful = None
            note = "server killed, owner left alive"
            idents = [i for i in idents if "strata_control.runtime" not in " ".join(i.cmd) and "conhost" not in " ".join(i.cmd)]
        elif what == "die-with":
            holder.kill(); holder.wait(timeout=10)
        if what == "server-killed":
            # the server and engine are gone; the owner host stays to report the failure until it is told to stop
            out = finish(rig, name, idents, expect_graceful=graceful, vram_before=before, vram=vram, note=note, wait=60,
                     vram_during=during)
            out["problems"] = [p for p in out["problems"] if p != "lease still held"]
            res = runtime_stop(rig)
            out["stop_after"] = res.get("state")
            left = rig.wait_gone(rig.tree(owner) or [], 30)
            if left or not rig.lease_free():
                out["problems"].append(f"after runtime stop: {left} lease_free={rig.lease_free()}")
            out["ok"] = not out["problems"]
            return out
        out = finish(rig, name, idents, expect_graceful=graceful, vram_before=before, vram=vram, note=note, wait=60,
                     vram_during=during)
        out["seconds"] = round(time.monotonic() - t0, 1)
        if what == "owner-killed":
            res = runtime_stop(rig)                       # the next stop clears the private state of a lost owner
            out["stop_after"] = res.get("state")
            if runtime._read_state("app") is not None:
                out["problems"].append("state file left after stop"); out["ok"] = False
        return out
    finally:
        if holder and holder.poll() is None:
            holder.kill()
        try:
            if runtime._read_state("app") is not None:
                runtime_stop(rig)
        except Exception:  # noqa: BLE001
            pass
        for i in rig.engine_idents():
            try:
                psutil.Process(i.pid).kill()
            except psutil.Error:
                pass


def windows_press(pid: int, text: str) -> None:
    """Type `text` into the console `pid` is attached to (what a person at the keyboard does)."""
    code = (
        "import ctypes, sys\n"
        "from ctypes import wintypes\n"
        "k = ctypes.windll.kernel32\n"
        "class Key(ctypes.Structure):\n"
        "    _fields_ = [('down', wintypes.BOOL), ('repeat', wintypes.WORD), ('vk', wintypes.WORD), ('scan', wintypes.WORD),\n"
        "                ('char', wintypes.WCHAR), ('state', wintypes.DWORD)]\n"
        "class Rec(ctypes.Structure):\n"
        "    _fields_ = [('kind', wintypes.WORD), ('pad', wintypes.WORD), ('key', Key)]\n"
        "k.FreeConsole()\n"
        f"assert k.AttachConsole({pid}), 'AttachConsole failed'\n"
        "k.CreateFileW.restype = wintypes.HANDLE\n"
        "h = k.CreateFileW('CONIN$', 0xC0000000, 3, None, 3, 0, None)\n"
        "recs = []\n"
        "for ch in sys.argv[1]:\n"
        "    for down in (1, 0):\n"
        "        recs.append(Rec(1, 0, Key(down, 1, ord(ch.upper()), 0, ch, 0)))\n"
        "arr = (Rec * len(recs))(*recs)\n"
        "n = wintypes.DWORD()\n"
        "assert k.WriteConsoleInputW(h, arr, len(recs), ctypes.byref(n)), 'WriteConsoleInput failed'\n"
    )
    subprocess.run([sys.executable, "-c", code, text], creationflags=subprocess.DETACHED_PROCESS, check=True, timeout=30)


def monitor_case(rig: Rig, how: str) -> Result:
    """The Textual monitor starts no engine. Quitting it (`q`) or killing it ends the monitor and leaves a service
    started elsewhere untouched; nothing of the monitor is left behind."""
    rig.clear_marks()
    started = runtime_start(rig)
    if started.get("state") != "ready":
        raise RuntimeError(f"not ready: {started}")
    owner = owner_pid(rig)
    service = rig.tree(owner)
    argv = [sys.executable, str(ROOT / "strata.py"), "monitor", "--url", f"http://127.0.0.1:{rig.port}"]
    env = {**rig.env, "TERM": "xterm-256color", "COLUMNS": "120", "LINES": "40"}
    master = child = None
    if WINDOWS:
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 7
        monitor = subprocess.Popen(argv, cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=info)
        pid, alive = monitor.pid, lambda: monitor.poll() is None
    else:
        import fcntl, pty, struct, termios
        child, master = pty.fork()
        if child == 0:
            os.chdir(ROOT)
            _default_signals()
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))     # a terminal that has a size
            os.execve(sys.executable, argv, env)
        pid = child
        gone = []

        def drain():                                  # a TUI paints without pause; a full pty buffer would freeze it
            while True:
                try:
                    if not os.read(master, 65536):
                        return
                except OSError:
                    return
        import threading
        threading.Thread(target=drain, daemon=True).start()

        def alive():
            if gone:
                return False
            try:
                done, _ = os.waitpid(child, os.WNOHANG)
            except ChildProcessError:
                done = child
            if done:
                gone.append(True)
            return not done
    problems = []
    time.sleep(4)
    if not alive():
        problems.append("the monitor did not stay up (is Textual installed here?)")
    mine = rig.tree(pid)
    if how == "quit":
        if WINDOWS:
            windows_press(pid, "q")
        else:
            os.write(master, b"q")
    else:
        kill_hard(pid)
    deadline = time.monotonic() + 15
    while alive() and time.monotonic() < deadline:
        time.sleep(0.2)
    if alive():
        problems.append(f"the monitor did not end on '{how}'")
    if master is not None:
        os.close(master)
    left = rig.wait_gone(mine, 15)
    if left:
        problems.append(f"monitor processes left: {left}")
    if not port_open(rig.port) or rig.survivors(service) != [f"{i.pid} {' '.join(i.cmd)[:90]}" for i in service
                                                           if procs.start_token(i.pid) == i.token]:
        problems.append("the monitor's end disturbed the service")
    runtime_stop(rig)
    left = rig.wait_gone(service, 45)
    if left:
        problems.append(f"service left after stop: {left}")
    return Result(name=f"monitor-{how}-leaves-service-alone", ok=not problems, problems=problems, marks=rig.marks(), note="")


SCENARIOS = {
    # foreground `strata serve`
    "serve-ctrl-c": lambda rig, v: serve_case(rig, "serve-ctrl-c", win_ctrl_c if WINDOWS else sigint_group,
                                              graceful=True, vram=v, new_console=WINDOWS),
    "serve-sigterm": lambda rig, v: serve_case(rig, "serve-sigterm", sigterm, graceful=True, vram=v),
    "serve-sighup": lambda rig, v: serve_case(rig, "serve-sighup", sighup, graceful=True, vram=v),
    "serve-ctrl-break": lambda rig, v: serve_case(rig, "serve-ctrl-break", win_ctrl_break, graceful=True, vram=v),
    "serve-console-close": lambda rig, v: serve_case(rig, "serve-console-close", win_console_close, graceful=None,
                                                     vram=v, new_console=True),
    "serve-kill-parent": lambda rig, v: serve_case(rig, "serve-kill-parent", kill9_parent, graceful=None, vram=v),
    "serve-busy-stop": lambda rig, v: serve_case(rig, "serve-busy-stop", busy_sigterm, graceful=True, vram=v),
    "serve-terminal-closed": lambda rig, v: pty_close_case(rig, vram=v),
    "serve-nohup-keeps-serving": lambda rig, v: nohup_case(rig, vram=v),
    # `strata runtime start` (a background service by design)
    "runtime-stop": lambda rig, v: runtime_case(rig, "runtime-stop", "stop", vram=v),
    "runtime-survives-terminal": lambda rig, v: runtime_case(rig, "runtime-survives-terminal", "terminal-gone", vram=v),
    "runtime-owner-killed": lambda rig, v: runtime_case(rig, "runtime-owner-killed", "owner-killed", vram=v),
    "runtime-server-killed": lambda rig, v: runtime_case(rig, "runtime-server-killed", "server-killed", vram=v),
    "runtime-die-with": lambda rig, v: runtime_case(rig, "runtime-die-with", "die-with", vram=v),
    # the TUI
    "monitor-quit": lambda rig, v: monitor_case(rig, "quit"),
    "monitor-killed": lambda rig, v: monitor_case(rig, "kill"),
}
POSIX_ONLY = {"serve-sigterm", "serve-sighup", "serve-terminal-closed", "serve-nohup-keeps-serving"}
WINDOWS_ONLY = {"serve-ctrl-break", "serve-console-close"}


def applicable(name: str) -> bool:
    if WINDOWS:
        return name not in POSIX_ONLY
    return name not in WINDOWS_ONLY


def run(names: list[str], *, real_config: Path | None = None, real_components: Path | None = None,
        vram: bool = False, slow_steps: int = 600) -> list[Result]:
    results = []
    for name in names:
        saved = dict(os.environ)                      # a Rig points this process at its own folders; give them back
        with tempfile.TemporaryDirectory(prefix="strata-matrix-") as tmp:
            rig = Rig(Path(tmp), real_config=real_config, real_components=real_components, slow_steps=slow_steps if name == "serve-busy-stop" else 0)
            try:
                result = SCENARIOS[name](rig, vram)
            except Exception as exc:  # noqa: BLE001
                result = Result(name=name, ok=False, problems=[f"{type(exc).__name__}: {exc}"], marks=[])
            results.append(result)
            print(("PASS " if result["ok"] else "FAIL ") + name, result.get("seconds", ""), result.get("problems") or "",
                  result.get("note", ""), flush=True)
            try:
                runtime_stop(rig)
            except Exception:  # noqa: BLE001
                pass
            for ident in rig.engine_idents():
                try:
                    psutil.Process(ident.pid).kill()
                except psutil.Error:
                    pass
        os.environ.clear()
        os.environ.update(saved)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--config", help="a real engine's run config (then no fake engine is used)")
    ap.add_argument("--components", help="the engine component folder for --config (STRATA_COMPONENTS_DIR)")
    ap.add_argument("--vram", action="store_true", help="check nvidia-smi memory returns")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    names = [n for n in SCENARIOS if applicable(n)] if a.scenarios == "all" else a.scenarios.split(",")
    results = run(names, real_config=Path(a.config) if a.config else None,
                  real_components=Path(a.components) if a.components else None, vram=a.vram)
    if a.report:
        Path(a.report).write_text(json.dumps({"platform": sys.platform, "engine": "real" if a.config else "fake",
                                              "results": results}, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
