"""serve/lifeline.py - a process Strata starts ends when the process that started it ends, however that ends.

The graceful path is the server's own (QUIT to the engine, drain, exit). This module is the guarantee behind it
for the cases where nothing gets to run: kill -9, a crash, the console window closed, logoff, power-off.

* Windows: the child joins a kill-on-close job object whose only handle is ours (serve/winjob.py).
* Linux: ``PR_SET_PDEATHSIG`` = SIGKILL, set in the child before it execs. The kernel ties the signal to the
  THREAD that forked the child, not to the process, so every start goes through one long-lived "anchor" thread;
  a request thread that starts an engine and then finishes would otherwise take the engine with it.
* macOS (no such call): a small watcher process holds a pipe to us. When we die the pipe closes, the watcher
  sees EOF and ends the child, after checking it is still the same process (PID, start time). A child that
  ended on its own, or was ended by us, is released first, so the watcher never signals a reused PID.

``popen`` is ``subprocess.Popen`` plus the guarantee. ``release`` is called once the child has been reaped.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from pathlib import Path

from serve import winjob
from strata_control import procs

_REAL_POPEN = subprocess.Popen                  # tests replace subprocess.Popen; isinstance needs the class
LINUX = sys.platform.startswith("linux")
MODE = (os.environ.get("STRATA_LIFELINE") or "").strip().lower()        # "watcher" forces the portable path (tests)

_libc = None
if LINUX:
    try:
        import ctypes
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    except (OSError, AttributeError):
        _libc = None

PR_SET_PDEATHSIG = 1


class _Anchor:
    """One thread that lives as long as the process and performs every Popen, so PDEATHSIG's "parent" is it."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="strata-lifeline-anchor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, future = self._jobs.get()
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)

    def call(self, fn):
        if threading.current_thread() is self._thread:
            return fn()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, future))
        return future.result()


_anchor: _Anchor | None = None
_anchor_lock = threading.Lock()


def _the_anchor() -> _Anchor:
    global _anchor
    with _anchor_lock:
        if _anchor is None:
            _anchor = _Anchor()
        return _anchor


def _die_with_parent(parent: int, then=None):
    """The preexec function of a child: SIGKILL when the starting thread ends; and if it ended already, now."""
    def run() -> None:
        _libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        if os.getppid() != parent:                 # the parent died between fork and prctl
            os.kill(os.getpid(), signal.SIGKILL)
        if then is not None:
            then()
    return run


# ---------------------------------------------------------------------------------------- the portable watcher

_WATCHER = r'''
import os, signal, subprocess, sys, time
pid, token = int(sys.argv[1]), sys.argv[2]
for s in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
    signal.signal(s, signal.SIG_IGN)               # it ends with its pipe, not with the terminal
def start_of(p):
    # Start time ONLY. `ps`'s stat column changes while a process runs (R while it starts, S while it
    # sleeps), so including it made this token disagree with the one taken at start and the watcher
    # spared the child it was meant to end.
    try:
        with open("/proc/%d/stat" % p, "rb") as f:   # closed explicitly: start_of runs up to 42 times per child
            raw = f.read().decode("utf-8", "replace")
        fields = raw[raw.rindex(")") + 2:].split()
        if fields[0] == "Z":                       # exited; only its parent has not collected it
            return None
        return "l:" + fields[19]
    except Exception:
        pass
    if os.path.exists("/proc/self/stat"):          # Linux, and that PID has no /proc entry any more
        return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(p)], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None
    return None if not out else "p:" + out
os.set_blocking(0, True)
data = b""
while True:
    chunk = os.read(0, 16)
    if not chunk:
        break
    data += chunk
    if b"x" in data:                                # released: the child is gone or was ended on purpose
        sys.exit(0)
if token and start_of(pid) == token:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        sys.exit(0)
    for _ in range(40):
        time.sleep(0.25)
        if start_of(pid) != token:
            sys.exit(0)
    if start_of(pid) == token:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
'''

_watchers: dict[int, subprocess.Popen] = {}
_watchers_lock = threading.Lock()


def _start_token(pid: int) -> str:
    """The same string for the life of one process, different for a later process with that PID.

    Identity is the start time alone. The previous version ran `ps -o lstart=,stat=` and kept the whole
    line, so the token changed whenever the process changed state - measured here: one child reported
    '... R' once and '... S' eleven times inside 150 ms. The watcher then compared its own reading with
    the token from start time, saw "different process", and exited without ending the child, which is
    exactly the flake `serve/test_lifeline.py` reported as "the child outlived a killed server".
    """
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            rest = fields[fields.rindex(")") + 2:].split()
            return "" if rest[0] == "Z" else "l:" + rest[19]      # starttime, clock ticks since boot
        except (OSError, ValueError, IndexError):
            # procfs IS there and the entry is not readable: the process is gone (or is not ours), and there is
            # nothing to fall back to. Without procfs at all, fall through to `ps` - otherwise this side would
            # return "" while the watcher's start_of() returns "p:...", the two can never match, and the child
            # outlives the server: exactly the failure this change removes, restored for that environment.
            if os.path.exists("/proc/self/stat"):
                return ""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return "" if not out else "p:" + out


def _watch(child: subprocess.Popen) -> None:
    if not isinstance(child, _REAL_POPEN) or getattr(child, "pid", None) is None:
        return
    try:
        watcher = subprocess.Popen([sys.executable, "-S", "-c", _WATCHER, str(child.pid), _start_token(child.pid)],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   close_fds=True, start_new_session=True)
    except OSError:
        return
    with _watchers_lock:
        _watchers[child.pid] = watcher


def release(child: subprocess.Popen | None) -> None:
    """The child has been reaped (or ended on purpose): stop watching it. Harmless where nothing is watching."""
    if child is None:
        return
    with _watchers_lock:
        watcher = _watchers.pop(child.pid, None)
    if watcher is None:
        return
    try:
        watcher.stdin.write(b"x")
        watcher.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        watcher.wait(timeout=5)
    except subprocess.TimeoutExpired:
        watcher.kill()


# ---------------------------------------------------------------------------------------------------- popen

def mechanism() -> str:
    """Which guarantee a start gets on this platform: job, pdeathsig, watcher."""
    if os.name == "nt":
        return "job"
    if LINUX and _libc is not None and MODE != "watcher":
        return "pdeathsig"
    return "watcher"


_children: set = set()
_children_lock = threading.Lock()


def end_children(grace: float = 3.0) -> None:
    """Whatever this process started and has not ended yet: terminate, then kill. Runs at interpreter exit, so an
    exception or a signal that unwinds main() before its own cleanup still leaves nothing behind (the engine and
    the others have already been closed on every normal path, so this finds nothing then)."""
    with _children_lock:
        alive = [c for c in _children if c.poll() is None]
    for child in alive:
        try:
            child.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + grace
    for child in alive:
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError):
            try:
                child.kill()
            except OSError:
                pass


atexit.register(end_children)


def _after_exit(child: subprocess.Popen, token: str | None) -> None:
    child.wait()
    with _children_lock:
        _children.discard(child)
    release(child)
    if token is not None:
        procs.drop_record(child.pid, token)


ISOLATED_ROLES = ("engine", "vision")      # speak a stdin protocol with us: the server alone decides when they stop


def _own_group(then=None):
    def run() -> None:
        os.setpgid(0, 0)
        if then is not None:
            then()
    return run


def popen(args, role: str = "engine", **kw) -> subprocess.Popen:
    """``subprocess.Popen(args, **kw)`` for a process that must not outlive us.

    An engine or image encoder is also taken out of the terminal's reach (its own process group on POSIX, a new
    process group on Windows): Ctrl+C, a closing terminal or a console Break go to the server only, which then ends
    the engine the clean way (STOP, QUIT, wait) instead of the engine dying under a request that could have been
    finished or cancelled. What ties the engine's life to ours is the guarantee below, not the shared terminal.

    It also leaves a record (strata_control/procs.py) so that, if everything here dies anyway, the next start can
    find this process by PID, start time and command line and end it."""
    how = mechanism()
    isolate = role in ISOLATED_ROLES
    if isolate and os.name == "nt":
        kw["creationflags"] = kw.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    if isolate and os.name != "nt" and "start_new_session" not in kw:
        kw["preexec_fn"] = _own_group(kw.pop("preexec_fn", None))
    if how == "pdeathsig":
        parent, then = os.getpid(), kw.pop("preexec_fn", None)
        kw["preexec_fn"] = _die_with_parent(parent, then)
        child = _the_anchor().call(lambda: subprocess.Popen(args, **kw))
    else:
        child = subprocess.Popen(args, **kw)
        if how == "job":
            winjob.contain(child)
        else:
            _watch(child)
    if isinstance(child, _REAL_POPEN):
        with _children_lock:
            _children.add(child)
    try:
        token = procs.write_record(child.pid, role, profile=os.environ.get("STRATA_RUNTIME_PROFILE") or None)
        threading.Thread(target=_after_exit, args=(child, token), name="strata-lifeline-reaper", daemon=True).start()
    except Exception:  # noqa: BLE001 - a record is a safety net, never a reason to fail a start
        pass
    return child


# ------------------------------------------------------------------------------------------- stop signals

_console_handler = None            # the ctypes callback must stay referenced for the life of the process


def install_stop_handlers(request_stop, finished: threading.Event | None = None) -> None:
    """Make every ordinary way a server is asked to end take the clean path (`request_stop(reason)`).

    * POSIX: SIGTERM (docker stop, systemd, `kill`), SIGHUP (the terminal closed, the ssh connection dropped; not when it
      was ignored at start, which is what `nohup` does on purpose).
      SIGINT keeps Python's own handler (Ctrl+C, and a second Ctrl+C to stop waiting).
    * Windows: SIGBREAK, and the console's close, logoff and shutdown events. Those arrive on a thread of their own and
      the system ends the process a few seconds after the handler returns, so the handler waits for `finished` (up to
      4.5 s) and then returns; whatever is still running is ended by the job object.
    Must be called from the main thread."""
    def in_main(reason: str):
        def handler(signum, frame):
            request_stop(reason)
        return handler
    for name, reason in (("SIGTERM", "sigterm"), ("SIGHUP", "sighup"), ("SIGBREAK", "sigbreak")):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            # `nohup strata serve ... &` asks to outlive its terminal: a hangup that is ignored at start stays ignored
            if name == "SIGHUP" and signal.getsignal(sig) == signal.SIG_IGN:
                continue
            signal.signal(sig, in_main(reason))
        except (ValueError, OSError):                       # not the main thread
            pass
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    events = {2: "console-close", 5: "logoff", 6: "shutdown"}
    prototype = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def on_console_event(event: int) -> bool:
        reason = events.get(event)
        if reason is None:
            return False                                    # Ctrl+C and Ctrl+Break: Python's own handling
        request_stop(reason)
        if finished is not None:
            finished.wait(4.5)
        return True

    global _console_handler
    _console_handler = prototype(on_console_event)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
