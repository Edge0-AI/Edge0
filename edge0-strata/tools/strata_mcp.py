#!/usr/bin/env python3
"""tools/strata_mcp.py - an MCP server that lets an AI assistant install, start, check and stop Strata.

    claude mcp add strata -- python /path/to/Strata/tools/strata_mcp.py

Add it to Claude Code, Claude Desktop, Cursor, VS Code, Codex ... (docs/MCP_SERVER.md has every client's snippet) and
ask your AI "install Strata for this PC", "start Strata", "is Strata running?".  It speaks MCP (JSON-RPC 2.0, one
message per line) over stdin/stdout, uses the Python standard library only (it runs before setup made .venv; Python
3.10 or newer), and offers a fixed set of tools: strata_status, strata_models, strata_install, strata_start,
strata_stop, strata_logs, strata_benchmark, strata_connect_info.

Safety: no tool takes a shell command or a free path.  Every argument is checked against setup.py's own choices
(models, families, contexts ...); the only path argument (install's data_dir) must be the Strata data folder, a folder
inside the Strata folder, or a new folder named Strata* outside the system folders.  Setup and the model run as
background processes this server starts itself; it keeps their process id and start time in .strata-mcp/ and only
ever stops a process it started (a Strata started elsewhere, e.g. from its own window, is only asked over its API to
unload the model).

Not to be confused with serve/mcp.py: that is Strata's MCP *client* (the model calling your tools from the chat page).
This file is an MCP *server* for the AI assistant you already use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "2025-06-18"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")   # the revisions this server answers in
VERSION = "0.1.0"
WIN = os.name == "nt"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
# The desktop app runs this under Strata Control's Python (STRATA_CONTROL_INSTALL=1): the install runs on that Python, makes no
# .venv and installs no package (setup.py's own CONTROL mode).
CONTROL = os.environ.get("STRATA_CONTROL_INSTALL") == "1"
DEFAULT_PORT = 8080
STATE_DIR = ".strata-mcp"                             # in the Strata folder (git-ignored)

# What setup.py offers, used only when setup.py cannot be imported (the numbers are setup's MODELS / FAMILIES /
# CONTEXTS; tools/test_strata_mcp.py checks they still match).
FALLBACK_MODELS = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3},
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "vision": True},
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic): K-quant experts, between UD-IQ4_XS and UD-Q5_K_XL in quality",
                   "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",), "budget": True,
                   "vision": True},                       # #967: images allowed (setup warns: untested with this file)
    "UD-Q5_K_XL": {"about": "5-bit (Unsloth Dynamic): Q5_K experts with Q8_0 downs, close to Q8_0 in quality",
                   "download_gb": 158.3, "ram_gb": 108, "arena_gb": 98.4, "families": ("unsloth",), "budget": True,
                   "vision": True},
    "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic): Q6_K experts with Q8_0 downs",
                   "download_gb": 169.2, "ram_gb": 118, "arena_gb": 109.2, "families": ("unsloth",), "budget": True,
                   "vision": True},
    "Q8_0": {"about": "8-bit, the reference quality (every expert Q8_0); for a 192 GB PC",
             "download_gb": 188.2, "ram_gb": 136, "arena_gb": 128.3, "families": ("unsloth",), "budget": True,
             "vision": True},
    "Gyro-S": {"about": "the smallest option, for PCs with little memory (27.6 GB of VRAM + RAM together); lower quality "
                        "than the 2-3-bit sizes, and the 2-bit Q2_0 decodes faster",
               "engine": (0, 1, 45, 1),
               "download_gb": 58.5, "ram_gb": 27.6, "arena_gb": 25.8, "families": ("gyro",), "mem_total": True,
               "vision": True},
    "Gyro-M": {"about": "about IQ3_XXS quality with about 20% less GPU memory (35 GB of VRAM + RAM together); decodes "
                        "about 16-21% slower than Q2_0 on 12 GB cards",
               "engine": (0, 1, 45, 1),
               "download_gb": 92.0, "ram_gb": 35.0, "arena_gb": 32.1, "families": ("gyro",), "mem_total": True,
               "vision": True},
    "NVFP4": {"about": "4-bit NVFP4 experts, GPTQ (sergqwer's strata-nvfp4 pack); RTX 50 has FP4 tensor cores, other "
                       "NVIDIA cards run it at 8-bit precision", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
              "families": ("nvfp4",), "vision": True},
    "NVFP4-NVIDIA": {"about": "NVIDIA's own NVFP4 checkpoint (plain rounding), converted by sergqwer; the same size "
                              "and speed as NVFP4", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
                     "families": ("nvfp4",), "vision": True},
    "NVFP4-QAD": {"about": "NVIDIA's FP4, trained for FP4 - slightly faster on short prompts; a 102 GB download "
                           "that this PC turns into a Strata pack once; it reads pictures",
                  "download_gb": 102.2, "ram_gb": 80, "arena_gb": 68.0, "families": ("nvfp4",), "vision": True},
}
FALLBACK_FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "about": "the original model", "tag": ""},
    "swift": {"title": "Swift 1.5", "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its "
                                             "authors' numbers)", "tag": "swift-"},
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "about": "half the experts (code, tools, images kept): needs ~32 GB "
                                                            "of RAM, faster; weaker outside coding", "tag": "coder-"},
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "about": "UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL or Q8_0 "
                                                                  "(94 to 188 GB downloads): every expert in RAM from "
                                                                  "~80 GB (UD-IQ4_XS) to ~136 GB (Q8_0) of RAM, else "
                                                                  "part of them read from the SSD",
                "tag": "unsloth-", "vision": True},
    "gyro": {"title": "Qwen3.8-Flash-Next (Gyro)", "about": "Gyro-S (27.6 GB of memory) and Gyro-M (35 GB): the smallest "
                                                              "sizes, for PCs with little memory; 58 and 92 GB "
                                                              "downloads (the files include the n-gram table)",
             "tag": "gyro-"},
    "nvfp4": {"title": "Qwen3.8-Flash-Next (NVFP4)", "about": "4-bit NVFP4 experts, NVIDIA cards; a 128 GB download, "
                                                              "~80 GB of RAM to keep every expert in RAM",
              "tag": "nvfp4-"},
}
OWN_PATH_ENGINE = (0, 1, 45)   # the first engine that runs Qwen3.6-35B-A3B with pictures and parallel requests (setup.py MIN_ENGINE note)
MEMORY_TIERS = ((62.0, "qwen", "IQ3_S"), (46.0, "qwen", "IQ3_XXS"), (38.0, "swift", "IQ2_XS"))   # as setup.py's: RAM + VRAM floors of the owner's 64 / 48 / 40 GB tiers
# #3425 (owner's rule, same as setup.py's TIER_SYSTEM_RESERVE_GB): the floors count RAM + VRAM less the 5 GB Windows and the system keep
TIER_SYSTEM_RESERVE_GB = 5.0
GYRO_ROOM_GB = 4.0     # recommend() suggests a Gyro size only when VRAM + RAM exceed its memory figure by this much
FALLBACK_CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
BENCH_PROMPT = ("Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle. "
                "Plain prose, no title.")

if WIN:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
    _k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _QUERY, _TERMINATE, _SYNCHRONIZE, _STILL_ACTIVE = 0x1000, 0x0001, 0x00100000, 259

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    _k32.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    _k32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_BREAKAWAY_FROM_JOB = 0x200, 0x08000000, 0x01000000


def log(msg: str) -> None:
    """Diagnostics go to stderr: stdout carries only the protocol."""
    try:
        sys.stderr.write(f"[strata-mcp] {msg}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


# ------------------------------------------------------------------------------------------------ processes
def proc_identity(pid) -> str | None:
    """A token for a RUNNING process: its start time (so a reused process id is never mistaken for ours), or None
    when no such process runs."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if WIN:
        h = _k32.OpenProcess(_QUERY, False, pid)
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not _k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                return None
            return f"win:{(c.dwHighDateTime << 32) | c.dwLowDateTime}"
        finally:
            _k32.CloseHandle(h)
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc/self/stat").exists():
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            return None
        if fields[0] in ("Z", "X"):                     # a zombie has ended
            return None
        return f"linux:{fields[19]}"                    # field 22 of stat: the start time in clock ticks
    try:                                                # elsewhere (no /proc): alive, without a start time
        os.kill(pid, 0)
        return "pid"
    except OSError:
        return None


def proc_alive(pid, ident) -> bool:
    return bool(ident) and proc_identity(pid) == ident


def proc_terminate(pid, ident) -> bool:
    """Ask a process WE started to end (POSIX: SIGTERM, the server's clean path; Windows: TerminateProcess).  Never
    touches a process whose start time differs from the one recorded."""
    if not proc_alive(pid, ident):
        return False
    if WIN:
        h = _k32.OpenProcess(_TERMINATE | _QUERY, False, int(pid))
        if not h:
            return False
        try:
            return bool(_k32.TerminateProcess(h, 1))
        finally:
            _k32.CloseHandle(h)
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def _rmtree_ignore_missing(func, path, exc_info):
    """shutil.rmtree error hook: a file that vanished while its folder was being removed is already gone."""
    exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    if isinstance(exc, FileNotFoundError):
        return
    raise exc


def proc_children_map() -> dict:
    """Windows: {parent pid: [child pids]} of every running process (a Toolhelp snapshot); {} elsewhere or when it fails."""
    if not WIN:
        return {}
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)                  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return {}
    kids: dict = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            kids.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _k32.CloseHandle(snap)
    return kids


def proc_tree_snapshot(pid) -> list:
    """Every process below `pid` as [(pid, identity)], taken NOW: once a middle process of the tree ends, taskkill /T can no longer reach
    what it started (#3257: a cancelled install left tools/mtp_fetch.py writing).  Windows only; [] elsewhere (a process group does that)."""
    kids = proc_children_map()
    out, seen, todo = [], {int(pid)}, [int(pid)]
    while todo:
        for child in kids.get(todo.pop(), []):
            if child in seen:
                continue
            seen.add(child)
            ident = proc_identity(child)
            if ident:
                out.append((child, ident))
                todo.append(child)
    return out


def kill_snapshot(procs) -> None:
    """End what a proc_tree_snapshot() listed and still runs (the same process: its start time still matches)."""
    for pid, ident in reversed(list(procs)):
        if proc_alive(pid, ident):
            proc_terminate(pid, ident)


def proc_kill_tree(pid, ident) -> bool:
    """End a process WE started together with the processes it started (setup's pip / cmake, the engine)."""
    if not proc_alive(pid, ident):
        return False
    if WIN:
        tree = proc_tree_snapshot(pid)                  # before taskkill: it walks parent ids, and a break in the chain hides the rest
        r = subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW)
        kill_snapshot(tree)
        return r.returncode == 0 or not proc_alive(pid, ident)
    try:
        os.killpg(int(pid), signal.SIGKILL)             # started as a session leader: its group is its own
        return True
    except OSError:
        try:
            os.kill(int(pid), signal.SIGKILL)
            return True
        except OSError:
            return False


def end_group_leftovers(pgid) -> None:
    """POSIX: after the server WE started (a session leader) is gone, end what is left of its process group - an
    engine it started but could not close (e.g. stopped while loading, before its own SIGTERM handler existed).
    While any member lives, the group id cannot belong to anybody else."""
    if WIN or not pgid:
        return
    for sig, wait in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(int(pgid), sig)
        except OSError:
            return                                      # no such group left
        end = time.time() + wait
        while time.time() < end:
            try:
                os.killpg(int(pgid), 0)
            except OSError:
                return
            time.sleep(0.25)


def wait_gone(pid, ident, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not proc_alive(pid, ident):
            return True
        time.sleep(0.25)
    return not proc_alive(pid, ident)


def spawn_detached(cmd: list, cwd: Path, log_path: Path, env: dict | None = None) -> subprocess.Popen:
    """Start a background process whose output goes to log_path.  It keeps running when the AI client (and this
    server) ends: its own process group / session, no console window on Windows."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        kw = dict(cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                  env=env, close_fds=True)
        if WIN:
            flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            try:                                        # out of the AI client's job, if it lets us
                return subprocess.Popen(cmd, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kw)
            except OSError:
                return subprocess.Popen(cmd, creationflags=flags, **kw)
        return subprocess.Popen(cmd, start_new_session=True, **kw)


def run_quiet(cmd, timeout=30) -> str:
    """A command's stdout ('' when it is missing or fails), without a console window flashing up on Windows."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                              creationflags=CREATE_NO_WINDOW if WIN else 0).stdout or ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def read_tail(path: Path, lines: int, max_bytes: int = 512 * 1024) -> list[str]:
    """The last lines of a log; a progress line rewritten with \\r counts as its last version."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for raw in data.split("\n"):
        part = [p for p in raw.split("\r") if p.strip()]
        if part:
            out.append(part[-1].rstrip())
    return out[-lines:]


def dir_size_gb(path: Path, limit_files: int = 20000) -> float:
    total, n = 0, 0
    for base, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(base, f)).st_size
            except OSError:
                pass
            n += 1
            if n >= limit_files:
                return round(total / 1e9, 1)
    return round(total / 1e9, 1)


# ------------------------------------------------------------------------------------------------ HTTP
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never through a proxy


def http_json(url: str, method="GET", body=None, api_key=None, timeout=3.0):
    """(status, JSON or None); status 0 when nothing answers."""
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw or b"null")
    except ValueError:
        return status, None


def port_free(port: int, host="127.0.0.1") -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ validation
class ToolError(Exception):
    """A clear message for the AI (and its user): the tool result is an error, the server goes on."""


def check_schema(schema: dict, args) -> dict:
    """The tools' JSON schemas, enforced: object, known properties only, types, enums, ranges, patterns."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    for k in args:
        if k not in props:
            raise ToolError(f"unknown argument {k!r}; this tool takes: {', '.join(props) or 'no arguments'}")
    for k in schema.get("required", []):
        if k not in args:
            raise ToolError(f"missing argument {k!r}")
    for k, v in args.items():
        p = props[k]
        t = p.get("type")
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            if isinstance(v, float) and v.is_integer():
                v = args[k] = int(v)
            elif isinstance(v, str) and re.fullmatch(r"\d{1,7}", v.strip()):
                v = args[k] = int(v.strip())              # some clients send numbers as strings
            else:
                raise ToolError(f"{k} must be a whole number")
        if t == "boolean" and not isinstance(v, bool):
            if v in ("true", "false"):
                v = args[k] = v == "true"
            else:
                raise ToolError(f"{k} must be true or false")
        if t == "string":
            if not isinstance(v, str):
                raise ToolError(f"{k} must be a string")
            if len(v) > p.get("maxLength", 300) or "\x00" in v or "\n" in v or "\r" in v:
                raise ToolError(f"{k} is too long or contains control characters")
            if "pattern" in p and not re.fullmatch(p["pattern"], v):
                raise ToolError(f"{k}={v!r} is not valid ({p.get('description', p['pattern'])})")
        if "enum" in p and v not in p["enum"]:
            raise ToolError(f"{k} must be one of: {', '.join(map(str, p['enum']))} (got {v!r})")
        if t == "integer":
            if "minimum" in p and v < p["minimum"] or "maximum" in p and v > p["maximum"]:
                raise ToolError(f"{k} must be between {p.get('minimum')} and {p.get('maximum')} (got {v})")
    return args


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def norm(p) -> Path:
    """A path to compare: absolute, links and Windows' short names (AI-SER~1) resolved, case folded on Windows."""
    return Path(os.path.normcase(os.path.realpath(os.path.abspath(str(p)))))


def system_folders() -> list[Path]:
    if WIN:
        env = [os.environ.get(k) for k in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramData",
                                            "ProgramW6432")]
        return [norm(p) for p in env if p]
    return [norm(p) for p in ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/proc", "/run", "/sbin",
                              "/sys", "/usr", "/var", "/snap")]


# ------------------------------------------------------------------------------------------------ Strata
class Strata:
    """One Strata folder: what is installed there, the PC, and the processes this server started."""

    def __init__(self, root: Path | str = DEFAULT_ROOT, src: Path | str | None = None):
        self.root = Path(root).resolve()
        # where setup.py, tools/ and serve/ are (the app: inside Strata Control); `root` is where the installs live
        self.src = Path(src).resolve() if src else self.root
        self.state_dir = self.root / STATE_DIR
        self.lock = threading.Lock()                    # install / start / stop one at a time
        self.children: list[subprocess.Popen] = []      # reaped now and then (no zombies on POSIX)
        self._setup = None
        self._setup_tried = False
        self._hw_cache = None
        # tests: the interpreter that runs setup / the server. The app's Control runs the installer on its own Python.
        self.python_override: str | None = sys.executable if CONTROL else None
        self.health_timeout = 2.0

    # ---- setup.py's own tables and logic
    def setup_module(self):
        """setup.py imported as a module (its tables and its pure checks), or None.  Its command runner is replaced
        by one without console windows; nothing of its install path runs."""
        if not self._setup_tried:
            self._setup_tried = True
            try:
                if str(self.src) not in sys.path:
                    sys.path.insert(0, str(self.src))   # setup.py imports strata_control
                spec = importlib.util.spec_from_file_location("strata_setup_for_mcp", self.src / "setup.py")
                mod = importlib.util.module_from_spec(spec)
                real = sys.stdout
                sys.stdout = sys.stderr                 # nothing it prints may reach the protocol stream
                try:
                    spec.loader.exec_module(mod)
                finally:
                    sys.stdout = real
                if all(hasattr(mod, a) for a in ("MODELS", "FAMILIES", "CONTEXTS")):
                    mod.out = lambda cmd: run_quiet(cmd, 60)
                    self._setup = mod
            except (Exception, SystemExit) as e:        # noqa: BLE001 - any failure: the built-in tables
                log(f"setup.py could not be imported ({e}); using the built-in model table")
        return self._setup

    def tables(self):
        s = self.setup_module()
        if s is not None:
            return s.MODELS, {**s.FAMILIES, **self.own_families()}, list(s.CONTEXTS), "setup.py"
        return FALLBACK_MODELS, FALLBACK_FAMILIES, FALLBACK_CONTEXTS, "built-in table (setup.py not importable)"

    def own_families(self) -> dict:
        """setup.py's OWN_PATH_FAMILIES (Qwen3.6-35B-A3B): families with an engine path of their own.  They are not in FAMILIES
        (whose entries are Flash-Next files with packs), so the model list used to leave them out and the app's Qwen3.6 page
        showed no versions at all.  Each is returned as a family entry (title, about, tag, license) of this server's table."""
        s = self.setup_module()
        out = {}
        for f, d in (getattr(s, "OWN_PATH_FAMILIES", None) or {}).items():
            out[f] = {"title": d["title"], "about": d.get("about"), "tag": d["tag"], "license": d.get("license"),
                      "vision": bool(d.get("vision", True)), "own_path": True}
        for f, d in (getattr(s, "MULTI_FILE_FAMILIES", None) or {}).items():     # DeepSeek-V4.1-Flash: nine shards, its own engine
            out[f] = {"title": d["title"], "about": d.get("about"), "tag": d["tag"], "license": d.get("license"),
                      "vision": bool(d.get("vision", False)), "own_path": True, "multi_file": True}
        return out

    def own_size_names(self) -> list:
        """The size names of the families that have an engine path of their own (Qwen3.6, DeepSeek-V4.1-Flash): the install tool accepts them as `model`."""
        s = self.setup_module()
        names = []
        for table in ("OWN_PATH_FAMILIES", "MULTI_FILE_FAMILIES"):
            for fam in (getattr(s, table, None) or {}).values():
                names.extend(n for n in fam.get("models", {}) if n not in names)
        return names

    def multi_family(self, family) -> dict | None:
        """setup.py's MULTI_FILE_FAMILIES entry (DeepSeek-V4.1-Flash), else None."""
        return (getattr(self.setup_module(), "MULTI_FILE_FAMILIES", None) or {}).get(family)

    def own_size(self, family, model) -> dict | None:
        """One size of an own-path family as a model-table row for install_plan / model_table (None: not an own-path size).
        The download is the GGUF; the picture reader (mmproj, ~0.9 GB) is counted by the plan when pictures are on."""
        big = self.multi_family(family)
        if big and model in big["models"]:
            # nine shards; the memory the weights need is VRAM + RAM together WITHOUT the n-gram table (it stays on the SSD)
            gb = sum(big["sha256"][n][0] for n in [*big["models"][model]["shards"], *big["models"][model].get("shared", [])]) / 1e9
            return {"about": big["models"][model]["about"], "download_gb": round(gb, 1), "ram_gb": big["models"][model]["ram_gb"],
                    "arena_gb": round(float(big["models"][model]["ram_gb"]) - 17.0, 1), "mem_total": True, "vision": bool(big.get("vision", True)),
                    "engine": None, "families": (family,), "multi_file": True}
        d = (getattr(self.setup_module(), "OWN_PATH_FAMILIES", None) or {}).get(family)
        if not d or model not in d["models"]:
            return None
        size_gb = d["sha256"][d["file"].format(q=model)][0] / 1e9
        return {"about": d["models"][model]["about"], "download_gb": round(size_gb, 1), "ram_gb": d["models"][model]["ram_gb"],
                # the routed experts are ~93% of the file (setup.own_path_parallel_fits_all); the dense part and buffers ~1 GB more
                "arena_gb": round(size_gb * 0.93, 1), "vision": bool(d.get("vision", True)),
                "engine": OWN_PATH_ENGINE, "families": (family,)}

    def sizes_of(self, models, family) -> list:
        """The family's sizes as setup lists them: an experimental one last (never the default)."""
        own = (getattr(self.setup_module(), "OWN_PATH_FAMILIES", None) or {}).get(family) or self.multi_family(family)
        if own:
            return list(own["models"])
        return sorted((m for m in models if family in models[m].get("families", ("qwen", "swift"))),
                      key=lambda m: bool(models[m].get("experimental")))

    # ---- paths
    def settings_path(self) -> Path:
        if os.environ.get("STRATA_SETTINGS_FILE"):
            return Path(os.environ["STRATA_SETTINGS_FILE"])
        if WIN:
            return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Strata" / "settings.json"
        return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"

    def data_dir(self) -> Path:
        try:
            s = json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        d = s.get("data_dir") if isinstance(s, dict) else None
        return Path(d) if d else self.root.parent / "Strata-data"

    def allowed_roots(self) -> list[Path]:
        return [norm(self.root), norm(self.data_dir())]

    def allowed(self, p) -> bool:
        p = norm(p)
        return any(is_inside(p, r) for r in self.allowed_roots())

    def venv_python(self) -> Path:
        return self.root / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")

    def run_python(self) -> str:
        return self.python_override or str(self.venv_python())

    def state(self, name: str) -> dict:
        try:
            d = json.loads((self.state_dir / f"{name}.json").read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, name: str, d: dict | None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        f = self.state_dir / f"{name}.json"
        if d is None:
            f.unlink(missing_ok=True)
        else:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            os.replace(tmp, f)

    def reap(self):
        self.children = [p for p in self.children if p.poll() is None]

    # ---- what is installed
    def configs(self) -> list[Path]:
        return sorted((p for p in self.root.glob("strata-*.json") if not p.name.endswith(".shared-settings.json")),
                      key=lambda p: p.stat().st_mtime, reverse=True)   # #346: the Chat settings file is no config

    @staticmethod
    def read_config(path: Path) -> dict:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            return cfg if isinstance(cfg, dict) else {}
        except (OSError, ValueError):
            return {}

    def describe_config(self, path: Path, usage: dict | None = None) -> dict:
        cfg = self.read_config(path)
        a = cfg.get("args") if isinstance(cfg.get("args"), list) else []

        def val(k):
            return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None
        tag = path.stem[len("strata-"):]
        missing = [p for p in [cfg.get("exe"), *[x for x in a if isinstance(x, str) and x.endswith(".gguf")]]
                   if not p or not Path(p).exists()]
        gaps = shard_gaps([x for x in a if isinstance(x, str) and x.endswith(".gguf")])   # a model in parts is whole only with every part
        missing += gaps
        script = self.root / f"run-{tag}.{'bat' if WIN else 'sh'}"
        vis = cfg.get("vision")
        own_usage = (usage or self.disk_usage())["models"].get(tag) or {"gb": 0.0, "bytes": 0, "shared_bytes": 0}
        plan = self.removal_plan(path)
        return {"model": tag, "config": path.name, "model_name": cfg.get("model_name"),
                "context": int(val("--max-context")) if str(val("--max-context") or "").isdigit() else None,
                "kv": val("--kv"), "images": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict)
                else ("on" if cfg.get("arch") == "deepseek41" and str(val("--vision") or "on").lower() not in ("off", "no", "false", "0") else "off"),
                "port": cfg.get("port", DEFAULT_PORT), "host": cfg.get("host", "127.0.0.1"),
                "gpu": cfg.get("gpu"), "gpu_roles": cfg.get("gpu_roles"), "api_key_set": bool(cfg.get("api_key")),
                "low_ram": "--resident-experts" in a or "--mmap-experts" in a,
                "start_script": script.name if script.exists() else None,
                "ready": not missing and bool(cfg), "missing_files": missing[:3], "parts_missing": len(gaps),
                "disk_gb": own_usage["gb"], "disk_bytes": own_usage["bytes"],
                "shared_bytes": own_usage["shared_bytes"],
                # what Delete would remove (files in Strata's own folders that no other model uses); the person's own GGUF folders stay
                "removable_gb": plan["gb"], "kept_files": len(plan["keep"]),
                "last_used": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))}

    # ---- what a model owns on disk (Delete, and the per-model disk usage in Storage)
    FILE_FLAGS = ("--native", "--ple-gguf", "--embd-gguf", "--native-dense-gguf")
    DIR_FLAGS = ("--pack", "--mtp")

    def model_paths(self, cfg: dict) -> tuple[list, list]:
        """(the GGUF files, the folders) a run config names: --native/--ple-gguf/... and --pack/--mtp, plus any other .gguf."""
        a = [x for x in (cfg.get("args") or []) if isinstance(x, str)]
        files = [Path(a[i + 1]) for i, x in enumerate(a[:-1]) if x in self.FILE_FLAGS]
        files += [Path(x) for x in a if x.lower().endswith(".gguf")]
        dirs = [Path(a[i + 1]) for i, x in enumerate(a[:-1]) if x in self.DIR_FLAGS]
        seen, once = set(), []
        for f in files:                                 # one file named by several flags is still one file
            if norm(f) not in seen:
                seen.add(norm(f))
                once.append(f)
        seen_d, once_d = set(), []
        for d in dirs:
            if norm(d) not in seen_d:
                seen_d.add(norm(d))
                once_d.append(d)
        return once, once_d

    @staticmethod
    def with_shards(files: list) -> list:
        """The files plus the other parts of every multi-part GGUF among them (name-00001-of-00004.gguf: the config names only the
        first), so deleting a model takes all its parts and never leaves 100 GB behind (#3716)."""
        out, seen = [], set()
        for f in files:
            names = [Path(f)]
            m = SHARD_NAME.match(str(f))
            if m:
                names += [Path(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, int(m.group(3)) + 1)]
            for n in names:
                key = norm(n)
                if key not in seen:
                    seen.add(key)
                    out.append(n)
        return out

    @staticmethod
    def tree_bytes(path: Path) -> int:
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        total = 0
        for base, _d, names in os.walk(path):
            for n in names:
                try:
                    total += os.stat(os.path.join(base, n)).st_size
                except OSError:
                    pass
        return total

    @staticmethod
    def file_key(path: Path):
        """What makes two names the same file on disk: device + inode (hard links), else the resolved path."""
        try:
            st = os.stat(path)
            if st.st_ino:
                return ("ino", st.st_dev, st.st_ino), st.st_size
            return ("path", str(norm(path))), st.st_size
        except OSError:
            return None, 0

    def used_files(self, cfg: dict) -> dict:
        """Every file a model really uses, from its run config: all parts of a multi-part GGUF (the config names the first), the
        picture reader, the packs and the draft layer folders.  {file identity: bytes}, wherever they live (the Strata folders or the
        person's own)."""
        files, dirs = self.model_paths(cfg)
        paths = list(files)
        vis = cfg.get("vision")
        if isinstance(vis, dict) and isinstance(vis.get("mmproj"), str):
            paths.append(Path(vis["mmproj"]))
        for f in list(paths):                           # the other parts of a multi-part file
            m = SHARD_NAME.match(str(f))
            if m:
                paths += [Path(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, int(m.group(3)) + 1)]
        for d in dirs:
            for base, _sub, names in os.walk(d):
                paths += [Path(base) / n for n in names]
        out = {}
        for p in paths:
            if p.is_file():
                key, size = self.file_key(p)
                if key is not None:
                    out[key] = size
        return out

    def disk_usage(self) -> dict:
        """What the installed models take on disk, from the real files: each model's own files, the files shared between models
        (the picture reader, the draft layer) counted ONCE, and the total.  A model's own figure leaves out the shared files and
        every other model's; rows plus 'shared' add up to the total."""
        per = {c.stem[len("strata-"):]: self.used_files(self.read_config(c)) for c in self.configs()}
        owners: dict = {}
        for files in per.values():
            for k in files:
                owners[k] = owners.get(k, 0) + 1
        models, shared = {}, {}
        for tag, files in per.items():
            own = sum(b for k, b in files.items() if owners[k] == 1)
            models[tag] = {"bytes": own, "gb": round(own / 1e9, 2), "files": len(files),
                           "shared_bytes": sum(b for k, b in files.items() if owners[k] > 1)}
            shared.update({k: b for k, b in files.items() if owners[k] > 1})
        shared_bytes = sum(shared.values())
        total = sum(m["bytes"] for m in models.values()) + shared_bytes
        return {"models": models, "shared_bytes": shared_bytes, "shared_gb": round(shared_bytes / 1e9, 2),
                "total_bytes": total, "total_gb": round(total / 1e9, 2)}

    @staticmethod
    def made_by_strata(p: Path, is_dir: bool) -> bool:
        """A download or pack Strata wrote itself, even in a models folder the person picked or the app chose: a finished GGUF has its
        .done mark beside it, a prepared pack has its conversion record. The person's own GGUF files have neither (#3716)."""
        try:
            if is_dir:
                return any((p / n).is_file() for n in ("conversions.json", "experts.bin.src.json"))
            return p.with_name(p.name + ".done").is_file()
        except OSError:
            return False

    def removal_plan(self, config: Path) -> dict:
        """What deleting this model removes and what it keeps. A file or folder another installed model's config still names
        (the shared MTP draft, a shared PLE shard) is kept; so is anything outside Strata's own folders."""
        cfg = self.read_config(config)
        files, dirs = self.model_paths(cfg)
        files = self.with_shards(files)
        used_files, used_dirs = set(), set()
        for other in self.configs():
            if norm(other) == norm(config):
                continue
            f, d = self.model_paths(self.read_config(other))
            f = self.with_shards(f)
            used_files |= {norm(x) for x in f}
            used_dirs |= {norm(x) for x in d}
        remove, keep = [], []

        def shared(p: Path, is_dir: bool) -> bool:
            n = norm(p)
            if n in used_files or n in used_dirs or any(is_inside(u, n) for u in used_files | used_dirs) and is_dir:
                return True
            return any(is_inside(n, u) for u in used_dirs)

        for p, is_dir in [(x, False) for x in files] + [(x, True) for x in dirs]:
            if not p.exists():
                continue
            if not self.allowed(p) and not self.made_by_strata(p, is_dir):
                keep.append({"path": str(p), "reason": "outside Strata's folders"})
            elif shared(p, is_dir):
                keep.append({"path": str(p), "reason": "another installed model still uses it"})
            else:
                remove.append({"path": str(p), "kind": "folder" if is_dir else "file", "bytes": self.tree_bytes(p)})
                if not is_dir:
                    for extra in (p.with_name(p.name + ".done"), p.with_name(p.name + ".part"),
                                  p.with_name(p.name + ".part.chunks")):
                        if extra.exists():
                            remove.append({"path": str(extra), "kind": "file", "bytes": self.tree_bytes(extra)})
        tag = config.stem[len("strata-"):]
        for extra in (config, config.with_name(config.name + ".bak"), config.with_name(f"strata-{tag}.log"),
                      self.root / f"run-{tag}.bat", self.root / f"run-{tag}.sh"):
            if extra.exists():
                remove.append({"path": str(extra), "kind": "file", "bytes": self.tree_bytes(extra)})
        total = sum(x["bytes"] for x in remove)
        return {"model": tag, "remove": remove, "keep": keep, "bytes": total, "gb": round(total / 1e9, 2)}

    def find_config(self, model: str | None) -> Path:
        have = self.configs()
        if not have:
            raise ToolError("no model is installed in " + str(self.root) + " yet: install one first (strata_install)")
        if not model:
            return have[0]                              # the most recently used, as setup starts it
        m = model.strip().lower()
        exact = [c for c in have if c.stem[len("strata-"):] == m]
        if exact:
            return exact[0]
        by_name = [c for c in have if str(self.read_config(c).get("model_name", "")).lower() == m]
        if by_name:
            return by_name[0]
        ends = [c for c in have if c.stem.endswith("-" + m)]
        if len(ends) == 1:
            return ends[0]
        raise ToolError(f"no installed model {model!r}; installed: " +
                        ", ".join(c.stem[len("strata-"):] for c in have))

    def engine_info(self) -> dict:
        for d in ("engine",):
            try:
                meta = json.loads((self.root / d / "BUILD.json").read_text())
                return {"folder": d, "version": meta.get("version"), "source": meta.get("source"),
                        "backend": meta.get("backend")}
            except (OSError, ValueError):
                continue
        return {"folder": None, "version": None}

    def installed(self) -> dict:
        data = self.data_dir()
        items = {}
        for sub in ("models", "packs"):
            for base in dict.fromkeys([data, self.root]):
                d = base / sub
                if d.is_dir():
                    for c in sorted(d.iterdir()):
                        if c.is_dir():
                            ggufs = list(c.glob("*.gguf"))
                            entry = {"folder": str(c), "size_gb": dir_size_gb(c)}
                            if sub == "models":
                                entry["complete"] = bool(ggufs) and all(
                                    g.with_name(g.name + ".done").exists() for g in ggufs)
                                entry["downloading"] = bool(list(c.glob("*.part")))
                            items.setdefault(sub, []).append(entry)
        mtp = data / "mtp" / "rt" / "experts.bin"
        usage = self.disk_usage()
        return {"strata_folder": str(self.root), "data_folder": str(data), "data_folder_exists": data.is_dir(),
                "python_env": self.venv_python().exists(), "engine": self.engine_info(),
                "models": [self.describe_config(c, usage) for c in self.configs()],
                "usage": usage,
                "downloads": items.get("models", []), "prepared": items.get("packs", []),
                "draft_layer": mtp.exists()}

    # ---- the PC
    def hardware(self, fresh=False) -> dict:
        if self._hw_cache and not fresh and time.time() - self._hw_cache[0] < 600:
            hw = dict(self._hw_cache[1])
        else:
            hw = self._hardware()
            self._hw_cache = (time.time(), hw)
            hw = dict(hw)
        hw["disk_free_gb"] = self.disk_free(self.data_dir())
        return hw

    @staticmethod
    def disk_free(p: Path) -> float | None:
        p = Path(p)
        while not p.exists() and p.parent != p:
            p = p.parent
        try:
            return round(shutil.disk_usage(p).free / 1e9, 1)
        except OSError:
            return None

    def _hardware(self) -> dict:
        S = self.setup_module()
        hw = {"os": f"{platform.system()} {platform.release()}", "os_version": platform.version(),
              "machine": platform.machine(), "python": platform.python_version()}
        if not WIN and Path("/proc/version").exists():
            try:
                hw["wsl"] = "microsoft" in Path("/proc/version").read_text().lower()
            except OSError:
                pass
        try:
            hw["ram_gb"] = round(S.ram_gb(), 1) if S else round(own_ram_gb(), 1)
        except Exception:                               # noqa: BLE001
            hw["ram_gb"] = round(own_ram_gb(), 1)
        if S and WIN:
            try:
                hw["page_file_gb"] = round(S.page_file_gb(), 1)
            except Exception:                           # noqa: BLE001
                pass
        try:
            if S:
                name, avx2, avx512 = S.cpu_info()
            else:
                name, avx2, avx512 = platform.processor() or "unknown CPU", None, None
            hw["cpu"] = {"name": name, "cores": physical_cores(), "threads": os.cpu_count(), "avx2": avx2, "avx512": avx512}
        except Exception:                               # noqa: BLE001
            hw["cpu"] = {"name": platform.processor(), "cores": physical_cores(), "threads": os.cpu_count()}
        if sys.platform == "darwin":                    # a Mac: one pool of unified memory, the Metal engine (strata_control/mac_fit.py)
            return self._mac_hardware(hw)
        nv = []
        try:
            nv = S.gpus() if S else own_nvidia_gpus()
        except Exception:                               # noqa: BLE001
            nv = own_nvidia_gpus()
        for g in nv:
            g["vendor"] = "nvidia"
            g["vram_gb"] = round(g["vram_gb"], 1)
            try:
                g["problem"] = S.gpu_problem(g) if S else (None if int(g["arch"]) >= 75 else "older than RTX 20")
            except Exception:                           # noqa: BLE001
                g["problem"] = None
            g["usable"] = g["problem"] is None
        amd = []
        try:                                            # Linux: the kernel's KFD topology; Windows: the display adapters' PCI ids + registry (setup.amd_gpus_win)
            amd = S.amd_gpus() if S else []
            for g in amd:
                g["vendor"] = "amd"
                g["vram_gb"] = round(g["vram_gb"], 1)
                g["problem"] = S.amd_problem(g)
                g["usable"] = g["problem"] is None
        except Exception:                               # noqa: BLE001
            amd = []
        if not WIN and not amd and shutil.which("rocm-smi"):
            amd = rocm_smi_gpus()
        if not WIN and not amd:                         # no setup module, no rocm-smi (ROCm from a Python venv): the kernel's own list
            amd = kfd_gpus()
        # A PC whose only NVIDIA cards are Pascal / Volta (two GTX 1080s) and that has no AMD card Strata can use: setup runs them with the
        # CUDA 12 engine (old_gpus_opt_in), so the app must not call it "no supported graphics card" and start both cards Off (#3655).
        # A PC with a newer card keeps recommending that one; naming an old card is still the opt-in.
        try:
            if S and nv and S.old_gpus_opt_in(nv, (), None, any(g.get("usable") for g in amd)):
                for g in nv:
                    if S.sm60_card(g["arch"]) and g.get("problem") and "older than the RTX 20" in str(g["problem"]):
                        g["problem"], g["usable"], g["engine_variant"] = None, True, "cuda12"
        except Exception:                               # noqa: BLE001 - the hardware page never breaks over this
            pass
        if any(g.get("usable") for g in nv):
            for g in amd:                             # a Ryzen's integrated graphics beside an NVIDIA card: the engine runs on the NVIDIA card
                if g.get("uma") and g.get("usable"):
                    g["usable"], g["problem"] = False, "integrated graphics: Strata uses the NVIDIA card in this computer"
        try:                                            # does the card drive a display?  (the app keeps less VRAM free on one that does not, #3215)
            mark_display(nv + amd, nvidia_display_active() if nv else {}, windows_desktop_adapters() if WIN and amd else [])
        except Exception:                               # noqa: BLE001 - unknown stays unknown: the app then keeps its safety default
            pass
        hw["gpus"] = nv + amd
        for g in hw["gpus"]:                            # each make numbers its cards from 0 (the engine drives one make): a stable id tells two cards apart
            g.setdefault("id", f"{g.get('vendor') or 'gpu'}:{g.get('index')}")
        try:                                            # how fast each card is (SMs x clock): the app and the Terminal pick the Main card by it
            from strata_control import gpu_roles
            gpu_roles.add_speeds(hw["gpus"])
        except Exception:                               # noqa: BLE001 - unknown speed: the card with the most memory is Main
            pass
        self._engine_code(hw)
        self._engine_vendor(hw)
        if WIN:
            hw["display_adapters"] = windows_video_controllers()
        hw["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
        return hw

    @staticmethod
    def _engine_code(hw: dict) -> None:
        """An AMD card the installed engine package has no code for is not one Strata can run on (the engine would stop on it, #3203): flag it
        unusable with the plain reason, so the card lists and "Automatic" leave it out.  Only a package that lists its archs speaks here."""
        amd = [g for g in hw.get("gpus", []) if g.get("vendor") == "amd" and g.get("arch")]
        if not amd:
            return
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            eng = rep.get("engine") or {}
            built = [a for a in eng.get("archs") or [] if str(a).startswith("gfx")]
            if not rep.get("usable") or eng.get("backend") != "hip" or not built:
                return
        except Exception:                               # noqa: BLE001 - never break the hardware page over this
            return
        for g in amd:
            if g["arch"] not in built and g.get("usable") is not False:
                g["usable"] = False
                g["problem"] = (ec.no_package_note([g["arch"]]) or f"this Strata engine has no code for {g['arch']} (it is built for {', '.join(built)})")

    @staticmethod
    def _engine_vendor(hw: dict) -> None:
        """One engine drives the cards of ONE make (CUDA: NVIDIA, HIP: AMD, ...): note which one for this PC (the installed engine's, else the one
        setup fetches first) and, when the other make has the bigger card, the one plain line that says so. The model list below only counts
        the cards this engine can drive; a model is never split across makes (strata_control/gpu_fit.py)."""
        try:
            from strata_control import gpu_fit
            backend = None
            try:
                from strata_control import engine_component as ec
                rep = ec.status()
                backend = (rep.get("engine") or {}).get("backend") if rep.get("usable") else None
            except Exception:                           # noqa: BLE001
                backend = None
            vendor = gpu_fit.pick_vendor(hw.get("gpus", []), backend)
            hw["engine_vendor"] = vendor
            note = gpu_fit.other_vendor_note(hw.get("gpus", []), vendor)
            if note:
                hw["gpu_note"] = note
        except Exception:                               # noqa: BLE001 - never break the hardware page over this
            pass

    @staticmethod
    def engine_gpus(hw: dict | None) -> list:
        """The usable cards the engine of this PC drives (all usable cards when the make is not known)."""
        hw = hw or {}
        vendor = hw.get("engine_vendor")
        return [g for g in hw.get("gpus", []) if g.get("usable") and (not vendor or not g.get("vendor") or g.get("vendor") == vendor)]

    # ---- a Mac (Apple Silicon): unified memory, the Metal engine.  The rules are strata_control/mac_fit.py's (the same module
    # metal/setup_mac.py uses), so the app, the Terminal and a Mac that builds the engine itself judge a model the same way.
    def mac(self):
        """strata_control/mac_fit.py, or None when this Strata folder does not carry it."""
        try:
            if str(self.src) not in sys.path:
                sys.path.insert(0, str(self.src))
            from strata_control import mac_fit
            return mac_fit
        except Exception as e:                          # noqa: BLE001
            log(f"strata_control/mac_fit.py could not be imported ({e})")
            return None

    def _mac_hardware(self, hw: dict) -> dict:
        mf = self.mac()
        info = mf.mac_info() if mf else {"chip": platform.processor() or "Apple silicon", "ram_gb": hw.get("ram_gb") or 0.0,
                                         "macos": platform.mac_ver()[0], "arm64": platform.machine() == "arm64", "clt": "x"}
        problems = mf.mac_problems(info, need_compiler=False) if mf else []
        ram = round(float(info.get("ram_gb") or hw.get("ram_gb") or 0.0), 1)
        hw["ram_gb"] = ram
        hw["cpu"] = {"name": info["chip"], "cores": os.cpu_count(), "avx2": None, "avx512": None}
        # The graphics memory IS the RAM here: vram_gb stays 0 so nothing is counted twice (the pages that fit RAM + VRAM add them up).
        hw["gpus"] = [{"index": 0, "name": info["chip"], "vendor": "apple", "vram_gb": 0.0, "unified_gb": ram, "arch": "apple-gpu",
                       "problem": problems[0] if problems else None, "usable": not problems}]
        hw["mac"] = {"chip": info["chip"], "macos": info.get("macos"), "unified_gb": ram}
        hw["nvidia_smi"] = False
        return hw

    def _mac_sizes(self, models: dict, family: str = "qwen") -> list:
        """The sizes the app offers on a Mac: the GSQ-RCO files of the Qwen family (two shards, no repack) and agentionai's Gyro files (one
        file, no repack). Unsloth's many-shard files need metal/tools/metal_repack.py, which the Terminal's setup uses; the app does not
        offer them yet."""
        mf = self.mac()
        return [m for m, d in mf.MAC_MODELS.items() if d.get("family", "qwen") == family and m in models] if mf else []

    def _mac_verdict(self, model: str, ram: float) -> str:
        mf = self.mac()
        if ram < mf.MIN_STREAM_RAM_GB:
            return "does not fit"
        if not mf.streams(model, ram):
            return "fits"
        speed = mf.STREAM_MEASURED.get(model)
        if speed:
            return (f"part of its experts in memory, the rest read from the SSD (slower: {speed} measured on a 32 GB Mac mini)")
        return "slow here: part of its experts read from the SSD (slower, not measured on a Mac of this size)"

    def _mac_recommend(self, hw: dict, models: dict, families: dict) -> dict:
        mf = self.mac()
        ram = float(hw.get("ram_gb") or 0)
        usable = [g for g in hw.get("gpus", []) if g.get("usable")]
        if not mf or not usable:
            why = (hw.get("gpus") or [{}])[0].get("problem") or "this Mac cannot run the Metal engine"
            return {"family": None, "model": None, "why": why}
        if ram < mf.MIN_STREAM_RAM_GB:
            return {"family": None, "model": None,
                    "why": f"{ram:.0f} GB of unified memory: Strata needs {mf.MIN_STREAM_RAM_GB:.0f} GB or more on a Mac (32 GB or more is what it was tested on)"}
        model, notes = mf.recommend_model(ram)
        notes = [n for n in notes if "UD-Q4_K_XL" not in n]          # the app does not offer the Unsloth files yet
        if model not in self._mac_sizes(models):
            model = "Q2_0"
        if mf.streams(model, ram):
            speed = mf.STREAM_MEASURED.get(model)
            why = (f"{ram:.0f} GB of unified memory: no size fits whole, so {model}'s experts are read from the SSD as it answers"
                   + (f" ({speed} measured on a 32 GB Mac mini)" if speed else ""))
        else:
            why = f"{ram:.0f} GB of unified memory: {model} stays in memory (the best size that does)"
        ctx = mf.recommend_context(ram)
        return {"family": "qwen", "model": model, "title": families["qwen"]["title"] + " " + model, "context": ctx,
                "backend": "metal", "gpu": {k: usable[0].get(k) for k in ("index", "name", "vram_gb")},
                "download_gb": models[model]["download_gb"], "why": why, "notes": notes,
                "setup_command": f"./setup.sh --setup --yes --family qwen --model {model} --context {ctx}"}

    def _mac_model_table(self, hw: dict) -> list:
        models, families, _ctx, _src = self.tables()
        ram = float(hw.get("ram_gb") or 0)
        have = {c.stem[len("strata-"):] for c in self.configs()}
        out = []
        for family in ("qwen", "gyro"):                # Gyro after the regular sizes, as on a PC: never the default
            fd = families.get(family)
            if fd is None:
                continue
            sizes = []
            for m in self._mac_sizes(models, family):
                d = models[m]
                tag = (fd.get("tag", "") + m).lower()
                sizes.append({"model": m, "about": d["about"], "download_gb": d["download_gb"], "experimental": False,
                              "images": d.get("vision", fd.get("vision", True)) is not False,   # the Mac engine carries the picture reader (Metal)
                              "ram_needed_gb": d["ram_gb"], "experts_gb": d.get("arena_gb"), "mem_total": False,
                              "memory_gb": round(float(d.get("arena_gb") or d["ram_gb"]) + 1.0, 1),
                              "on_this_pc": self._mac_verdict(m, ram) if ram else None, "installed": tag in have, "id": tag})
            if sizes:
                out.append({"family": family, "title": fd["title"], "about": fd.get("about"), "experimental": False,
                            "images": any(x["images"] for x in sizes), "sizes": sizes})
        ds = self._mac_deepseek(hw, have)                # DeepSeek-V4.1-Flash on the Mac (0.2.7.1): after Qwen, never the default
        if ds is not None:
            out.append(ds)
        return out

    def _mac_deepseek(self, hw: dict, have: set) -> dict | None:
        """DeepSeek-V4.1-Flash for a Mac: the official sets that were run on one (strata_control/dsv41_fit.MAC_SET_PACK), with the Mac fit words.
        None when setup.py has no DeepSeek family (an older Terminal)."""
        big = self.multi_family("deepseek41")
        if not big:
            return None
        from strata_control import dsv41_fit
        mf, ram = self.mac(), float(hw.get("ram_gb") or 0)
        sizes = []
        for m in big["models"]:
            if m not in dsv41_fit.MAC_SET_PACK:
                continue
            d = self.own_size("deepseek41", m)
            if d is None:
                continue
            tag = (big.get("tag", "") + m).lower()
            sizes.append({"model": m, "about": d["about"], "download_gb": d["download_gb"], "experimental": False,
                          "images": False, "images_reader": "none",       # the Mac engine's picture tower: see docs/DEEPSEEK_MAC.md
                          "ram_needed_gb": d["ram_gb"], "experts_gb": d.get("arena_gb"), "mem_total": True, "multi_file": True,
                          "disk_gb": self.dsv41_disk_gb(d), "engine_min": None,
                          "memory_gb": round(float(d["ram_gb"]), 1),
                          "on_this_pc": ("does not fit: this Mac has too little memory" if ram and ram < mf.MIN_STREAM_RAM_GB
                                         else self.dsv41_words(hw, m, installed=tag in have)),
                          "installed": tag in have, "id": tag, "gpu_options": None})
        if not sizes:
            return None
        return {"family": "deepseek41", "title": big["title"], "about": big.get("about"), "experimental": False,
                "images": False, "sizes": sizes}

    def recommend(self, hw: dict) -> dict:
        """setup.py's own rules: the size it preselects (IQ3_XXS from 60 GB of RAM, else the first, Q2_0), the
        Coder below the full model's RAM (setup's RAM table), the low-RAM mode when the GPU makes up for the RAM, and
        the context it preselects for the card's VRAM (32K under 14 GB, 64K under 20 GB, else 128K)."""
        models, families, _ctx, _src = self.tables()
        if hw.get("mac"):
            return self._mac_recommend(hw, models, families)
        S = self.setup_module()
        usable = self.engine_gpus(hw)
        ram = hw.get("ram_gb") or 0
        if not usable:
            why = ("no NVIDIA RTX 20-series-or-newer GPU found (nvidia-smi did not list one)"
                   + ("; the AMD graphics in this PC is not a card Strata supports (an RX 7000 or 9000 series card, or a Ryzen AI Max, is needed)" if WIN and any(
                       "amd" in str(a.get("name", "")).lower() or "radeon" in str(a.get("name", "")).lower()
                       for a in hw.get("display_adapters", [])) else ""))
            return {"family": None, "model": None, "why": why}
        best = max(usable, key=lambda g: (round(g["vram_gb"]), -g.get("index", 0)))
        vram = best["vram_gb"]
        backend = "hip" if best.get("vendor") == "amd" else "cuda"

        def low_fits(m):
            try:
                return S.low_ram_fits(m, ram, vram) if S else ram - 6 + max(0.0, vram - 5 - 1.0) >= models[m]["arena_gb"]
            except Exception:                           # noqa: BLE001
                return False
        notes = []
        # 0.2.0, the owner's memory tiers (README "The other sizes"; setup.py MEMORY_TIERS holds the same floors): RAM + VRAM
        # together.  An APU's shared memory is the RAM itself, so only its BIOS carve-out counts as VRAM.
        tvram = float(best.get("dedicated_gb", 0.0)) if best.get("uma") else float(vram)
        tier = next(((f, n) for lo, f, n in MEMORY_TIERS if ram + tvram - TIER_SYSTEM_RESERVE_GB >= lo), None)
        if tier and tier[1] in models and tier[0] in families:
            fam, model = tier
            total = f"{ram:.0f} GB of RAM + {tvram:.0f} GB of VRAM = {ram + tvram:.0f} GB, {TIER_SYSTEM_RESERVE_GB:.0f} GB of it kept for Windows and the system"
            if model == "IQ2_XS":
                why = (f"{total}, the 40 GB tier: Swift 1.5's IQ2_XS (thinks for a shorter time)"
                       + ("; Gyro-M (35 GB) is the other choice there, close to IQ3_XXS in quality but 16-21% slower at "
                          "writing answers than Q2_0" if "Gyro-M" in models and best.get("vendor") in ("nvidia", "amd", "intel") else ""))
            elif model == "IQ3_XXS":
                why = f"{total}, the 48 GB tier: IQ3_XXS (better quality, slower)"
            else:
                why = (f"{total}, the 64 GB tier: IQ3_S (matches the full model, the slowest); Swift 1.5's IQ3_S is an "
                       "equal choice")
        elif ram >= 60:
            fam, model, why = "qwen", "IQ3_XXS", f"{ram:.0f} GB of RAM: setup's own pick from 60 GB (better quality)"
        elif ram >= models["Q2_0"]["ram_gb"] - 4:
            fam, model, why = "qwen", "Q2_0", f"{ram:.0f} GB of RAM: the full model's fastest size fits"
        elif ram >= models["IQ1_M"]["ram_gb"] - 4:
            fam, model, why = "coder", "IQ1_M", (f"{ram:.0f} GB of RAM: the full model needs ~48 GB; the Coder "
                                                 "(half the experts, best for code) needs ~32 GB")
        elif low_fits("IQ1_M"):
            fam, model, why = "coder", "IQ1_M", (f"{ram:.0f} GB of RAM is below the Coder's 32 GB, but the GPU's "
                                                 f"{vram:.0f} GB make up for it (setup's low-RAM mode: slower)")
        elif "Gyro-S" in models and ram + vram >= models["Gyro-S"]["ram_gb"] + GYRO_ROOM_GB:
            # Gyro (agentionai): only where the sizes above do not fit.  Their memory figure (Gyro-S 27.6 GB, Gyro-M 35 GB)
            # is VRAM + RAM together and the system needs room beside it, so GYRO_ROOM_GB more is asked for here.
            # Gyro-M when this PC has that much, else Gyro-S.
            model = "Gyro-M" if ram + vram >= models["Gyro-M"]["ram_gb"] + GYRO_ROOM_GB else "Gyro-S"
            fam, why = "gyro", (f"{ram:.0f} GB of RAM is below the Coder's 32 GB, but RAM + VRAM = {ram + vram:.0f} GB "
                                f"holds {model} ({models[model]['ram_gb']} GB of memory in total; the quality is lower "
                                "than the 2-3-bit sizes, and it needs the whole download on a fast SSD)")
        else:
            return {"family": None, "model": None, "why": f"{ram:.0f} GB of RAM: Strata needs 32 GB or more "
                                                          "(the smallest model, the Coder, keeps ~23 GB of experts in "
                                                          "RAM)"}
        ctx = 32768 if vram < 14 else 65536 if vram < 20 else 131072
        # #3425: the tiers count RAM + VRAM together, but the desktop keeps about 10 GB of the RAM and the card keeps ~6 GB for
        # its dense weights, so a PC at the floor of its tier reads part of the experts from the SSD at every answer.  Said
        # plainly beside the pick (the same arithmetic as the app's "This computer" card); the pick itself stays the owner's tier.
        arena = float(models[model].get("arena_gb") or 0)
        spill = arena - max(0.0, tvram - 6.0) - max(0.0, ram - 10.0)
        if arena and spill >= 4.0:
            notes.append(f"about {spill:.0f} GB of {model} would be read from the SSD while answering (this PC's RAM leaves about 10 GB "
                         "for the system and the card holds the rest): expect a few tokens per second on a SATA or USB drive, more "
                         "on a fast NVMe. A smaller size fits better; it is a recommendation, every size stays a choice")
        if vram < 11:
            notes.append("less than 12 GB of VRAM: it runs, but slowly (most experts stay on the CPU)")
        if hw.get("cpu", {}).get("avx2") is False:      # #623: a warning, not a stop (setup compiles for it)
            notes.append("this CPU has no AVX2: EXPERIMENTAL and slow - setup compiles the engine on this PC for the "
                         "older CPU (10-20 minutes), and the CPU's share of the experts runs a few times slower")
        if backend == "hip":
            notes.append("AMD (experimental): pictures are on; the reader runs on the CPU under 24 GB of VRAM and on the GPU from 24 GB")
        cmd = (("START-HERE.bat --setup" if WIN else "./setup.sh --setup") +
               f" --yes --family {fam} --model {model} --context {ctx}" + (" --backend hip" if backend == "hip" else ""))
        return {"family": fam, "model": model, "title": families[fam]["title"] + " " + model, "context": ctx,
                "backend": backend, "gpu": {k: best.get(k) for k in ("index", "name", "vram_gb")},
                "download_gb": models[model]["download_gb"], "why": why, "notes": notes, "setup_command": cmd}

    def model_table(self, hw: dict | None) -> list:
        if (hw or {}).get("mac"):
            return self._mac_model_table(hw)
        models, families, _ctx, _src = self.tables()
        S = self.setup_module()
        ram = (hw or {}).get("ram_gb")
        usable = self.engine_gpus(hw)
        vram = max((g["vram_gb"] for g in usable), default=0.0)
        have = {c.stem[len("strata-"):] for c in self.configs()}
        out = []
        for f, fd in families.items():
            sizes = []
            for m in self.sizes_of(models, f):
                if f == "deepseek41" and self.dsv41_not_here(m):      # the Unsloth 1-bit sets on a DGX Spark: not offered until a Spark run passes
                    continue
                own = self.own_size(f, m)
                d = own or models[m]
                tag = (fd.get("tag", "") + m).lower()
                verdict = None
                if ram is not None:
                    have_gb = ram + vram if d.get("mem_total") else ram     # Gyro: VRAM and RAM together
                    verdict = "fits" if have_gb >= d["ram_gb"] else "tight" if have_gb >= d["ram_gb"] - 8 else "does not fit"
                    try:
                        if own and own.get("multi_file"):       # DeepSeek-V4.1-Flash: strata_control/dsv41_fit.py (cards + RAM, the n-gram table on the SSD)
                            verdict = self.dsv41_words(hw, m, installed=tag in have)
                        elif own:                       # its own engine path: no budget or low-RAM mode of Flash-Next's to judge by
                            pass
                        elif d.get("budget") and S and hasattr(S, "unsloth_budget") and not S.unsloth_budget(m, ram):
                            verdict = ("experimental: " if d.get("experimental") else "") + "fits (all of its experts in RAM)"
                        elif d.get("budget"):
                            verdict = (("experimental: " if d.get("experimental") else "")
                                       + "part of its experts in RAM, the rest read from the SSD"
                                       if ram >= d["ram_gb"] else
                                       "slow here: part of its experts in RAM, the rest read from the SSD (setup "
                                       "asks first)" if ram >= 48 else "does not fit")
                        elif not d.get("mem_total") and S and S.low_ram_needed(m, ram) and S.low_ram_fits(m, ram, vram):
                            verdict = "fits in the low-RAM mode (slower: the GPU holds part of the experts)"
                    except Exception:                   # noqa: BLE001
                        pass
                sizes.append({"model": m, "about": d["about"], "download_gb": d["download_gb"],
                              "experimental": bool(d.get("experimental")),
                              "images": d.get("vision", fd.get("vision", True)) is not False,
                              # which picture reader "pictures on" gets on this PC (strata_control/vision_reader.py, the one rule: the CPU reader
                              # under 24 GB of graphics memory, the GPU reader from 24 GB; the person can switch it per model)
                              "images_reader": self.images_reader(vram),
                              "ram_needed_gb": d["ram_gb"], "experts_gb": d.get("arena_gb"),
                              # RAM + VRAM the weights need together, WITHOUT the n-gram table (that stays on the SSD): Gyro states it
                              # (mem_total), for the rest the experts plus ~1 GB for the dense part (Swift IQ2_XS: 35.5 + 1 = 36.5)
                              "mem_total": bool(d.get("mem_total")), "multi_file": bool(own and own.get("multi_file")),
                              # DeepSeek only: the free disk the set needs, from the set itself (the download: model files, n-gram table, drafter and
                              # metadata, plus the pack folder and room to work, strata_control/dsv41_fit.DISK_HEADROOM_GB): 2.00 and 2.25 differ
                              "disk_gb": self.dsv41_disk_gb(d) if own and own.get("multi_file") else None,
                              # the first engine that reads this size's files (setup.py "engine"); the app says "needs a newer engine"
                              # with the update step when the installed one is older. None: any engine that setup accepts.
                              "engine_min": ".".join(map(str, d["engine"])) if d.get("engine") else None,
                              "memory_gb": round(float(d["ram_gb"]) if d.get("mem_total") else float(d.get("arena_gb") or d["ram_gb"]) + 1.0, 1),
                              "on_this_pc": verdict, "installed": tag in have, "id": tag,
                              # which graphics cards to use for it (strata_control/gpu_fit.py, the one rule app / Terminal / setup share):
                              # only with two or more cards that count; "recommended" names the pre-selected option
                              "gpu_options": (self.dsv41_options(hw, m) if own and own.get("multi_file")
                                              else None if own else self.gpu_options(d, ram, hw))})   # Qwen3.6 runs on one card so far
            out.append({"family": f, "title": fd["title"], "about": fd.get("about"),
                        "experimental": bool(fd.get("experimental")),
                        "images": any(x["images"] for x in sizes) if sizes else fd.get("vision", True) is not False,
                        "sizes": sizes})
        return out

    @staticmethod
    def dsv41_not_on_mac() -> tuple:
        """The DeepSeek sets the Metal engine cannot run yet (IQ1_S experts): strata_control/dsv41_fit.MAC_NOT_YET."""
        from strata_control import dsv41_fit
        return dsv41_fit.MAC_NOT_YET

    @staticmethod
    def dsv41_not_here(model: str) -> str | None:
        """The plain line when this computer (a Mac, a DGX Spark) is not offered `model`, else None: strata_control/dsv41_fit.not_here."""
        from strata_control import dsv41_fit
        return dsv41_fit.not_here(model)

    @staticmethod
    def dsv41_not_on_mac_line() -> str:
        from strata_control import dsv41_fit
        return dsv41_fit.MAC_NOT_YET_LINE

    @staticmethod
    def dsv41_disk_gb(d: dict) -> float:
        """The free disk a DeepSeek set needs: its whole download plus the pack and room to work (the card text says "about")."""
        from strata_control import dsv41_fit
        return round(float(d["download_gb"]) + dsv41_fit.DISK_HEADROOM_GB, 1)

    def dsv41_plan(self, hw, disk_free=None, only=None, model=None) -> dict:
        """strata_control/dsv41_fit.plan for this computer; `only`: the card numbers a launch names (else every card that counts)."""
        from strata_control import dsv41_fit
        cards = list((hw or {}).get("gpus", []))
        if only:
            cards = [c for c in cards if int(c.get("index", -1)) in {int(i) for i in only}]
        return dsv41_fit.plan(cards, (hw or {}).get("ram_gb"), disk_free, vendor=(hw or {}).get("engine_vendor"), model=model, named=bool(only))

    def dsv41_words(self, hw, model=None, installed: bool = False) -> str:
        """The catalog's `on_this_pc` text for DeepSeek-V4.1-Flash, with the disk it needs (download and preparation) beside the memory: a set that
        fits in memory but not on the disk is "tight", never refused (strata_cli/launch/catalog.verdict reads its first words)."""
        words = self._dsv41_memory_words(hw, model)
        free = (hw or {}).get("disk_free_gb")
        if installed or free is None or words.startswith("does not fit"):
            return words
        pl = self.dsv41_plan(hw, free, model=model)
        from strata_control import dsv41_fit
        left = self.dsv41_left_gb(model, pl["download_gb"])                   # only what still has to be downloaded (files already here count as done)
        need = left + dsv41_fit.DISK_HEADROOM_GB                             # + the pack (MBs) and room to work
        if left > 0.05 and free < need:
            return (f"tight: it needs about {need:.0f} GB of free disk space (the rest of the download and its preparation) and this computer has "
                    f"{free:.0f} GB free; free some space or choose another drive when you install it ({words})")
        return words

    def dsv41_left_gb(self, model, default: float) -> float:
        """GB of this DeepSeek set's files still to be downloaded: the files already in Strata's models folder (the right size; the first release's
        names count too) are done, so the disk check asks only for the rest.  Files the person keeps elsewhere (--model-path) are handled by the plan.
        `default`: what to answer when the tables cannot be read (the whole download)."""
        S = self.setup_module()
        try:
            fam = S.MULTI_FILE_FAMILIES["deepseek41"]
            md = fam["models"][model]
            models = self.data_dir() / "models"
            folder, shared = models / (fam["tag"] + model), models / (fam["tag"] + "shared")
            left = 0
            for where, names in ((folder, md["shards"]), (shared, md.get("shared", []))):
                for n in names:
                    for cand in (n, getattr(S, "DSV41_LEGACY_NAMES", {}).get(n)):
                        if cand and (where / cand).is_file() and (where / cand).stat().st_size == S.DSV41_ALL_FILES[cand][0]:
                            break
                    else:
                        left += S.DSV41_ALL_FILES[n][0]
            return left / 1e9
        except Exception:                                    # noqa: BLE001 - a hint for the menu: when unsure, count the whole download
            return default

    def _dsv41_memory_words(self, hw, model=None) -> str:
        pl = self.dsv41_plan(hw, model=model)
        v = pl["verdict"]
        if v == "no_card":
            return "does not fit: it needs one or more graphics cards"
        if pl.get("mac"):                       # an Apple Silicon Mac (0.2.7.1): one pool of unified memory, the rest of the experts streams from the SSD
            streamed, kept = pl["streamed_gb"], pl["vram_experts_gb"]
            if pl["mac_verdict"] == "fits":
                return f"fits: every expert stays in this Mac's memory ({pl['experts_gb']:.0f} GB)"
            if pl["mac_verdict"] == "small":
                return (f"slow here: this Mac's memory holds about {kept:.0f} GB of the {pl['experts_gb']:.0f} GB of experts and the other "
                        f"{streamed:.0f} GB are read from the SSD for every word; it can still be installed (nothing this small was measured, "
                        "a Mac with 64 GB or more suits it better)")
            return (f"runs on this Mac's memory: about {kept:.0f} GB of the {pl['experts_gb']:.0f} GB of experts stay in memory and the other "
                    f"{streamed:.0f} GB are read from the SSD while it answers; measured 12-19 tokens a second on an M5 Max with 128 GB")
        if pl.get("uma"):                       # Ryzen AI Max (Strix Halo, Windows): the graphics memory is the RAM, counted once
            return (f"runs on this PC's shared memory: about {pl['vram_experts_gb']:.0f} GB of the {pl['experts_gb']:.0f} GB of experts stay "
                    f"in memory and the other {pl['streamed_gb']:.0f} GB are read from the SSD; measured 14-16 tokens a second on code and "
                    "10-11 on chat")
        if pl.get("variant"):                   # an official IST-DAS set (strata_control/dsv41_fit.py): measured on 46-64 GB, never a refusal
            if v == "spills":
                return (f"slow here: this computer has {pl['ram_gb']:.0f} GB of RAM and the set was measured with 46-64 GB, so most of its "
                        "experts are read from the SSD; it can still be installed"
                        + ("" if pl["variant"] == "IST-DAS-1.75bpw" else " (the smaller 1.75 bpw version suits this computer better)"))
            if v == "tight":
                return ("tight: runs, with the experts RAM cannot hold read from the SSD (about 5-7 tokens a second measured on 46-64 GB "
                        "of RAM and a 12 GB card)")
            if v == "fits":
                return (f"fits: {pl['experts_in_vram']:,} of {pl['experts_total']:,} experts in graphics memory, the rest cached in RAM "
                        f"({pl['ram_needed_gb']:.0f} GB)")
            return "not checked (this computer's memory could not be read)"
        if v == "spills" and pl["spill_gb"] >= 100:       # most of the CPU's experts would come from the disk every word
            return (f"does not fit: about {pl['spill_gb']:.0f} GB of its experts would be read from the SSD while it answers "
                    f"(slow: it needs ~{pl['ram_needed_gb']:.0f} GB of RAM beside {pl['vram_experts_gb']:.0f} GB of graphics memory; "
                    "it can still be installed)")
        if v == "spills":
            return (f"part of its experts are read from the SSD (slow here: needs ~{pl['ram_needed_gb']:.0f} GB of RAM beside "
                    f"{pl['vram_experts_gb']:.0f} GB of graphics memory; it can still be installed)")
        if v == "tight":
            return f"tight: RAM is close to the {pl['ram_needed_gb']:.0f} GB the CPU's share of the experts needs"
        if v == "fits":
            return (f"fits: {pl['experts_in_vram']:,} of {pl['experts_total']:,} experts in graphics memory, the rest on the CPU "
                    f"({pl['ram_needed_gb']:.0f} GB of RAM)")
        return "not checked (this computer's memory could not be read)"

    def dsv41_options(self, hw, model=None) -> dict | None:
        """gpu_fit-shaped card choices for DeepSeek-V4.1-Flash: every card (recommended: the model is bigger than any one card), or the
        largest alone; each with the placement words."""
        from strata_control import dsv41_fit
        try:
            return dsv41_fit.options(list((hw or {}).get("gpus", [])), (hw or {}).get("ram_gb"), (hw or {}).get("engine_vendor"), model=model)
        except Exception:                                  # noqa: BLE001 - the choice is a convenience: never break the model list
            return None

    @staticmethod
    def images_reader(vram) -> str:
        try:
            from strata_control import vision_reader
            return vision_reader.default_mode(vram) if vram else "cpu"
        except Exception:                                  # noqa: BLE001
            return "cpu"

    @staticmethod
    def gpu_options(d: dict, ram, hw) -> dict | None:
        try:
            from strata_control import gpu_fit
            mem = float(d["ram_gb"]) if d.get("mem_total") else float(d.get("arena_gb") or d["ram_gb"]) + 1.0
            res = gpu_fit.options(round(mem, 1), ram, (hw or {}).get("gpus", []), (hw or {}).get("engine_vendor"))
            if len(res["options"]) >= 2:
                return res
            # one card the engine can drive: no choice to offer, but a bigger card of the other make is still worth one plain line
            return {"options": [], "recommended": None, "other_vendor_note": res.get("other_vendor_note")} if res.get("other_vendor_note") else None
        except Exception:                                  # noqa: BLE001 - the choice is a convenience: never break the model list
            return None

    # ---- the running server
    def ports(self, port: int | None) -> list[int]:
        if port:
            return [port]
        ps = [self.state("server").get("port")] + [self.describe_config(c)["port"] for c in self.configs()]
        return list(dict.fromkeys(int(p) for p in [*ps, DEFAULT_PORT] if isinstance(p, int)))

    def api_key_for(self, port: int) -> str | None:
        """The key a config of this folder set for that port (used for our own requests; never shown)."""
        def real(cfg):                                   # "api_key": "none" = served without a key on purpose
            k = cfg.get("api_key")
            return k if k and str(k).strip().lower() != "none" else None
        for c in self.configs():
            cfg = self.read_config(c)
            if real(cfg) and int(cfg.get("port", DEFAULT_PORT)) == port:
                return cfg["api_key"]
        for c in self.configs():
            if real(self.read_config(c)):
                return self.read_config(c)["api_key"]
        return None

    def probe(self, port: int, deep=True) -> dict | None:
        """What answers on 127.0.0.1:port: a Strata server's facts, {"other": True} for something else, None for
        nothing."""
        base = f"http://127.0.0.1:{port}"
        st, health = http_json(base + "/health", timeout=self.health_timeout)
        if st == 0:
            return None
        if st != 200 or not isinstance(health, dict) or health.get("service") != "strata":
            return {"port": port, "other": True, "note": "something else answers on this port (not a Strata server)"}
        info = {"port": port, "url": base, "model": health.get("model"), "context": health.get("max_context"),
                "images": health.get("images"), "loaded": health.get("loaded"),
                "api_key_required": bool(health.get("api_key"))}
        if not deep:
            return info
        key = self.api_key_for(port)
        st, v1 = http_json(base + "/v1/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(v1, dict):
            info["engine_version"] = v1.get("engine")
            info["uptime_s"] = v1.get("uptime_s")
            info["requests_served"] = (v1.get("activity") or {}).get("requests")
            m = v1.get("machine") or {}
            if m.get("gpu"):
                info["gpu"] = m["gpu"]
            if m.get("ram"):
                info["ram"] = m["ram"]
            if v1.get("last_timings"):
                lt = v1["last_timings"]
                info["last_request"] = {k: lt.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n",
                                                               "predicted_per_second") if k in lt}
        elif st == 401:
            info["note"] = "the server wants an API key this folder's configs do not have"
        st, s = http_json(base + "/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(s, dict):
            info["busy"] = bool(s.get("busy"))
            info["activity"] = {k: s.get(k) for k in ("phase", "generated", "tokens_per_s", "queued", "elapsed_s")
                                if s.get(k) is not None}
        if "gpu" not in info:                           # older servers: the Monitor's numbers
            st, met = http_json(base + "/metrics", api_key=key, timeout=self.health_timeout)
            if st == 200 and isinstance(met, dict):
                hwn = met.get("hardware") or {}
                if hwn.get("gpu_mem_total"):
                    info["gpu"] = {"used_mib": round(hwn.get("gpu_mem_used", 0) / 2**20),
                                   "total_mib": round(hwn["gpu_mem_total"] / 2**20)}
                if hwn.get("ram_total"):
                    info["ram"] = {"used_gib": round(hwn.get("ram_used", 0) / 2**30, 1),
                                   "total_gib": round(hwn["ram_total"] / 2**30, 1)}
                info["engine_version"] = (met.get("engine") or {}).get("version")
        return info

    def tracked_server(self) -> dict | None:
        """The server this MCP server started, while it still runs (else its record is cleared)."""
        s = self.state("server")
        if not s:
            return None
        if proc_alive(s.get("pid"), s.get("ident")):
            return s
        return {**s, "ended": True}

    # ---- install job
    def install_job(self) -> dict | None:
        j = self.state("install")
        if not j:
            return None
        res = self.state("install-result")
        running = proc_alive(j.get("pid"), j.get("ident"))
        log_path = Path(j.get("log", ""))
        tail = read_tail(log_path, 1500) if log_path.name else []   # a step's heading must stay inside it: the pack tools print hundreds of lines
        prog = install_progress(tail)
        prog = combined_progress(prog, j)
        out = {"running": running, "started": j.get("started"), "model": j.get("model"), "family": j.get("family"),
               "log": str(log_path), **prog}
        if not running:
            if res.get("job") == j.get("job") and "exit_code" in res:
                out["exit_code"] = res["exit_code"]
                out["finished"] = res.get("ended")
                out["result"] = "installed" if res["exit_code"] == 0 else "failed"
            else:
                out["result"] = "ended unexpectedly (no exit code recorded)"
        else:
            out["result"] = "running"
        out["last_lines"] = tail[-12:]
        return out


SHARD_NAME = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
DEFAULT_MMPROJ_BYTES = 907_543_008          # the Flash-Next picture reader (mmproj BF16), when a family does not pin its size


def shard_gaps(ggufs) -> list:
    """The parts of a multi-part GGUF (name-00001-of-00004.gguf) that are not there whole: a missing part, or one still being
    downloaded (<part>.part).  A config that names only the first part used to look finished with one part of four on disk."""
    gaps = []
    for g in ggufs:
        m = SHARD_NAME.match(str(g))
        if not m:
            continue
        for i in range(1, int(m.group(3)) + 1):
            sib = Path(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf")
            if not sib.is_file() or sib.with_name(sib.name + ".part").exists():
                gaps.append(str(sib))
    return gaps


def expected_files(S, family: str, model: str, models_dir: Path, vision: bool) -> list:
    """The files setup fetches for one install, with the sizes it can know before the download starts:
    [{path, bytes|None, role}] where role is "model" (the shards or the pack's files) or "vision" (the picture reader).
    Empty when the setup module cannot say (the caller then falls back to the folder scan)."""
    out = []
    try:
        big = (getattr(S, "MULTI_FILE_FAMILIES", None) or {}).get(family)
        if big:                                         # DeepSeek-V4.1-Flash: its nine shards in the model's folder
            folder = Path(models_dir) / (big["tag"] + model)
            shared = Path(models_dir) / (big["tag"] + "shared")        # the Engram, the drafter and the metadata shard of the official sets
            return ([{"path": str(folder / n), "bytes": big["sha256"][n][0], "role": "model"} for n in big["models"][model]["shards"]]
                    + [{"path": str(shared / n), "bytes": big["sha256"][n][0], "role": "model"} for n in big["models"][model].get("shared", [])])
        own = (getattr(S, "OWN_PATH_FAMILIES", None) or {}).get(family)
        if own:                                         # Qwen3.6: the GGUF and the picture reader in the model's folder
            folder = Path(models_dir) / (own["tag"] + model)
            name = own["file"].format(q=model)
            out.append({"path": str(folder / name), "bytes": own["sha256"][name][0], "role": "model"})
            if vision:
                out.append({"path": str(folder / own["mmproj"]), "bytes": own["sha256"][own["mmproj"]][0], "role": "vision"})
            return out
        fam, d = S.FAMILIES[family], S.MODELS[model]
        if d.get("repo"):                               # NVFP4: a ready-made pack, many files with pinned sizes
            for f in S.ready_needed(model, Path(models_dir)):
                out.append({"path": str(S.ready_where(Path(models_dir), f)), "bytes": f[4], "role": "model"})
        else:
            folder = Path(models_dir) / (fam.get("tag", "") + model)
            for i in range(1, S.model_shards(fam, model) + 1):
                out.append({"path": str(folder / S.model_file(fam, model, i)), "bytes": None, "role": "model"})
        if vision:                                      # beside the model folders, not inside them (setup step 5)
            size = (fam.get("mmproj_sha256") or (None,))[0] or DEFAULT_MMPROJ_BYTES
            out.append({"path": str(Path(models_dir) / fam.get("mmproj", S.MMPROJ)), "bytes": size, "role": "vision"})
    except Exception:                                   # noqa: BLE001 - progress is a convenience: never break the install
        return []
    return out


def files_progress(prog: dict, job: dict, files: list) -> dict:
    """One honest download figure for a model in parts: every part counted (finished, partial, waiting), the picture reader as a part
    of its own, and a part list the screen can name ("part 2 of 4").  The finished state needs EVERY part: a first shard that is whole
    does not make the model downloaded."""
    log = prog.get("download") or {}
    cur_gb = float(log.get("total_gb") or 0)
    rows = []
    for f in files:
        p = Path(f["path"])
        part = p.with_name(p.name + ".part")
        try:
            if part.is_file():
                state, have = "downloading", part.stat().st_size
            elif p.is_file():
                state, have = "done", p.stat().st_size
            else:
                state, have = "waiting", 0
        except OSError:
            state, have = "waiting", 0
        total = f.get("bytes") or (have if state == "done" else 0)
        if state == "downloading" and cur_gb and p.name in str(log.get("file", "")):   # the log line names this file
            # The .part file is as long as the furthest byte written, which the parallel pieces push to the full size well before the
            # download is done: the pieces setup counts are the truth.
            have = min(have, int(float(log.get("done_gb") or 0) * 1e9)) if have else have
            if not total:
                total = int(cur_gb * 1e9)
        rows.append({"role": f.get("role", "model"), "state": state, "have": have, "total": max(total, have)})
    plan_total = float(job.get("total_gb") or 0) * 1e9
    parts = []
    for role, name in (("model", "Model files"), ("vision", "Picture reader")):
        mine = [r for r in rows if r["role"] == role]
        if not mine:
            continue
        have = sum(r["have"] for r in mine)
        known = sum(r["total"] for r in mine)
        total = known if all(r["total"] for r in mine) else max(plan_total if role == "model" else 0, known, have)
        k = sum(1 for r in mine if r["state"] == "done")
        state = "done" if k == len(mine) else "downloading" if any(r["state"] == "downloading" for r in mine) or k else "waiting"
        parts.append({"kind": role, "name": name, "state": state, "done_gb": round(min(have, total) / 1e9, 3),
                      "total_gb": round(total / 1e9, 3), "files_done": k, "files_total": len(mine)})
    total_gb = sum(x["total_gb"] for x in parts)
    done_gb = min(sum(x["done_gb"] for x in parts), total_gb)
    active = next((x for x in parts if x["state"] != "done"), parts[-1] if parts else None)
    if active and active["state"] == "waiting" and any(x["state"] == "done" for x in parts):
        active = dict(active, state="downloading")
    prog["download"] = {"file": "picture reader" if active and active["kind"] == "vision" else "model files",
                        "done_gb": round(done_gb, 3), "total_gb": round(total_gb, 3),
                        "remaining_gb": round(max(0.0, total_gb - done_gb), 3),
                        "percent": min(99, int(100 * done_gb / total_gb)) if total_gb else 0}
    prog["parts"] = parts
    return prog


def draft_progress(prog: dict, job: dict, dl: dict) -> dict:
    """Setup step 6 downloads the ~5 GB of MTP tensors (tools/mtp_fetch.py prints `MTP draft layer: 1.2 / 5.0 GB (24%)`).  The figure
    the app shows is the whole job's: the model files (done by now) plus this part, so the bar does not fall back to 0 (#3257)."""
    done_gb, total_gb = float(dl.get("done_gb") or 0), float(dl.get("total_gb") or 0)
    parts = []
    if job.get("files"):
        base = files_progress({}, job, job["files"])
        parts = [dict(p, state="done", done_gb=p["total_gb"]) for p in base.get("parts", []) if p.get("state") == "done"]
    parts.append({"kind": "draft", "name": "Draft layer", "state": "done" if int(dl.get("percent") or 0) >= 100 else "downloading",
                  "done_gb": round(done_gb, 3), "total_gb": round(total_gb, 3)})
    total = sum(p["total_gb"] for p in parts)
    done = min(sum(p["done_gb"] for p in parts), total)
    prog["parts"] = parts
    prog["download"] = {"file": "Draft layer", "done_gb": round(done, 3), "total_gb": round(total, 3),
                        "remaining_gb": round(max(0.0, total - done), 3),
                        "percent": min(99, int(100 * done / total)) if total else 0}
    return prog


def combined_progress(prog: dict, job: dict) -> dict:
    """One total for the whole model: the bytes of all its files (finished, partial and the one in flight, read from the
    model's folder) over the download size, instead of each file's own 0-100% (a two-file model went 70 GB -> 0 -> up
    again). Only while the download step runs; later steps are not a download."""
    folder, total = job.get("folder"), job.get("total_gb")
    step = str(prog.get("step") or "")
    dl = prog.get("download") or {}
    if step.startswith("6/") and str(dl.get("file", "")).startswith("MTP draft layer"):
        return draft_progress(prog, job, dl)
    if step.startswith("5/") and str(dl.get("file", "")).startswith("checking "):
        # setup is hashing files that are all on disk: the figure is the check's own (not a download at "99%")
        if job.get("files"):
            prog = files_progress(prog, job, job["files"])
        prog["download"] = {"file": "checking files", "done_gb": dl["done_gb"], "total_gb": dl["total_gb"],
                            "remaining_gb": round(max(0.0, dl["total_gb"] - dl["done_gb"]), 3), "percent": int(dl["percent"])}
        return prog
    if job.get("files") and step.startswith("5/"):      # the install knows its files: count each of them
        return files_progress(prog, job, job["files"])
    if not folder or not total or not step.startswith("5/"):
        prog.pop("download", None)
        return prog
    done = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and (entry.name.endswith(".gguf") or entry.name.endswith(".part")):
                done += entry.stat().st_size
    except OSError:
        pass
    if not done:                                        # nothing on disk yet: the log's own line is all there is
        return prog
    done_gb = min(done / 1e9, float(total))
    prog["download"] = {"file": "model files", "done_gb": round(done_gb, 3), "total_gb": float(total),
                        "remaining_gb": round(max(0.0, float(total) - done_gb), 3),
                        "percent": min(99, int(100 * done_gb / float(total))) if total else 0}
    # The parts of the whole job, for a progress bar that sums them (the job that fetches the engine adds its own part):
    # {kind, name, done_gb, total_gb, state}
    prog["parts"] = [{"kind": "model", "name": "Model files", "done_gb": prog["download"]["done_gb"],
                      "total_gb": float(total), "state": "downloading"}]
    return prog


def install_progress(lines: list[str]) -> dict:
    """setup's progress from its output: the step, a download's percentage, a failure."""
    step, title, dl, err = None, None, None, None
    for i, line in enumerate(lines):
        m = re.search(r"=== Step (\d+): (.+?) ===", line)
        if m:
            step, title, err, dl = int(m.group(1)), m.group(2), None, None      # a download figure belongs to its own step
        m = re.match(r"\s*(\S.*?): +([\d.]+) / ([\d.]+) GB \((\d+)%\)", line)
        if m:
            dl = {"file": m.group(1), "done_gb": float(m.group(2)), "total_gb": float(m.group(3)),
                  "percent": int(m.group(4))}
        if line.strip().startswith("[X]"):
            err = line.strip()[3:].strip()
            if i + 1 < len(lines) and lines[i + 1].startswith("       "):
                err += " - " + lines[i + 1].strip()
    out = {"step": f"{step}/7: {title}" if step else None}
    if dl:
        out["download"] = dl
    if err:
        out["error"] = err
    words = progress_words_for(step, lines, dl)
    if words:
        out["label"], out["phase"] = words
    return out


def physical_cores() -> int | None:
    """The CPU's physical cores (os.cpu_count() counts logical threads: 16 on an 8-core SMT chip, and the app's Work profile took
    80% of that for --pool-workers, more than the cores, #3417).  Falls back to the thread count when the machine cannot say."""
    try:
        from strata_control import tune_machine
        n = tune_machine.cpu().get("physical")
        if isinstance(n, int) and n > 0:
            return n
    except Exception:                                   # noqa: BLE001 - a hardware report must never fail on this
        pass
    return os.cpu_count()


def progress_words_for(step, lines, download):
    """(the plain sentence, its phase) for what setup is doing now (strata_control/progress_words.py, the words the Terminal uses too);
    None when this Strata folder does not carry the module."""
    try:
        from strata_control import progress_words
    except ImportError:
        return None
    w = progress_words.describe(step, lines, download)
    return w.label, w.phase


def own_ram_gb() -> float:
    if WIN:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        *((n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev"))]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.tp / 2**30
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / 2**30
    except OSError:
        pass
    return 0.0


def own_nvidia_gpus() -> list:
    s = run_quiet(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                   "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            try:
                mem_gb = float(mem) / 1024.0
            except ValueError:
                if platform.machine().lower() not in ("aarch64", "arm64"):
                    raise            # as before on x86: a card whose memory nvidia-smi does not report is not listed
                # a DGX Spark (GB10): one memory for the CPU and the GPU, memory.total is [N/A]; counted as setup.py's gpus() counts it
                # (the RAM less the OS's 6 GB, none of it room beside the RAM)
                shared = max(0.0, own_ram_gb() - 6)
                found.append({"index": int(idx), "name": name, "vram_gb": shared, "arch": cc.replace(".", ""), "driver": drv,
                              "uma": True, "dedicated_gb": 0.0, "shared_gb": shared})
                continue
            found.append({"index": int(idx), "name": name, "vram_gb": mem_gb,
                          "arch": cc.replace(".", ""), "driver": drv})
        except ValueError:
            continue
    return found


def nvidia_display_active() -> dict:
    """{GPU index: True when a display is attached to the card} from nvidia-smi (display_active Enabled); {} when it says nothing."""
    found = {}
    for line in run_quiet(["nvidia-smi", "--query-gpu=index,display_active", "--format=csv,noheader"]).strip().splitlines():
        idx, _, state = line.partition(",")
        if idx.strip().isdigit() and state.strip().lower() in ("enabled", "disabled"):
            found[int(idx)] = state.strip().lower() == "enabled"
    return found


def windows_desktop_adapters() -> list:
    """The PNPDeviceIDs of the Windows display adapters that have a desktop resolution (a monitor is attached to them)."""
    out = []
    for line in run_quiet(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                           "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID + '|' + $_.CurrentHorizontalResolution }"]
                          ).splitlines():
        pnp, _, res = line.strip().rpartition("|")
        if pnp and res.strip().isdigit() and int(res) > 0:
            out.append(pnp)
    return out


def mark_display(gpus: list, nvidia_active: dict, desktop_pnp: list) -> None:
    """Sets g["drives_display"] (True / False) on the cards that can be told, from nvidia-smi (NVIDIA) and the Windows adapters that have a desktop
    resolution (AMD and Intel, matched by PCI device id; two cards of one model are not told apart: left unknown).  A card left without the key is
    unknown, and the app treats an unknown card as one that drives a display."""
    desk = [str(p).upper() for p in desktop_pnp]
    for g in gpus:
        if g.get("vendor") == "nvidia" and g.get("index") in nvidia_active:
            g["drives_display"] = nvidia_active[g["index"]]
        elif g.get("vendor") == "amd" and isinstance(g.get("pci_id"), int) and desktop_pnp is not None and desktop_pnp:
            tag = f"DEV_{g['pci_id']:04X}"
            twins = [x for x in gpus if x.get("vendor") == "amd" and x.get("pci_id") == g["pci_id"]]
            if len(twins) == 1:
                g["drives_display"] = any(tag in p and "VEN_1002" in p for p in desk)


def windows_video_controllers() -> list:
    s = run_quiet(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"])
    try:
        d = json.loads(s) if s.strip() else []
    except ValueError:
        return []
    d = [d] if isinstance(d, dict) else d
    return [{"name": x.get("Name"), "driver": x.get("DriverVersion")} for x in d if isinstance(x, dict)]


KFD_SUPPORTED = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "gfx1031", "gfx1151")


def kfd_gpus(sysfs: str = "/sys") -> list:
    """AMD cards from the kernel's KFD topology and the amdgpu driver's files, with no ROCm tool and no setup module: a PC that has only the
    driver plus the engine's own runtime has no rocm-smi, amd-smi or rocminfo, and the cards were then not counted at all (a 4-card PC ran on one).
    Numbered as HIP numbers them (the GPU nodes in order, CPU nodes skipped)."""
    base = Path(sysfs) / "class/kfd/kfd/topology/nodes"
    out = []
    try:
        nodes = sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError:
        return out
    for node in nodes:
        try:
            props = dict(line.partition(" ")[::2] for line in (node / "properties").read_text().splitlines())
            ver, simd = int(props.get("gfx_target_version") or 0), int(props.get("simd_count") or 0)
        except (OSError, ValueError):
            continue
        if ver == 0 or simd == 0:
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(sysfs) / f"class/drm/renderD{props.get('drm_render_minor', '').strip()}/device"
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        ok = arch in KFD_SUPPORTED
        out.append({"index": len(out), "name": name, "vram_gb": round(vram, 1), "arch": arch, "driver": "amdgpu", "vendor": "amd",
                    "usable": ok, "problem": None if ok else f"not supported - Strata's AMD backend does not list {arch}"})
    return out


def rocm_smi_gpus() -> list:
    s = run_quiet(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    try:
        d = json.loads(s)
    except ValueError:
        return []
    out = []
    for i, (card, v) in enumerate(sorted(d.items())):
        if not isinstance(v, dict) or not card.lower().startswith("card"):
            continue
        total = next((v[k] for k in v if "total" in k.lower() and "vram" in k.lower()), 0)
        try:
            vram = int(total) / 2**30
        except (TypeError, ValueError):
            vram = 0.0
        out.append({"index": i, "name": v.get("Card Series") or v.get("Card SKU") or card, "vram_gb": round(vram, 1),
                    "vendor": "amd", "usable": None, "problem": "listed by rocm-smi (setup checks the card itself)"})
    return out


# ------------------------------------------------------------------------------------------------ the tools
GPUS_PATTERN = r"all|\d{1,2}(,\d{1,2}){1,7}"
MODEL_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,48}"


def tool_defs(models: dict, families: dict, contexts: list) -> list:
    port = {"type": "integer", "minimum": 1024, "maximum": 65535,
            "description": "the server's port (default: the installed model's port, usually 8080)"}
    installed_model = {"type": "string", "pattern": MODEL_ID_PATTERN, "maxLength": 49,
                       "description": "an installed model's id as strata_status lists it (e.g. 'q2_0', "
                                      "'coder-iq1_m', 'swift-iq3_xxs'); default: the most recently used one"}
    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    return [
        {"name": "strata_status", "title": "Strata status",
         "description": "Is Strata (the local AI model server) running, and what is installed? Reports the running "
                        "server (model, engine version, context, busy or idle, VRAM/RAM in use), the installed models "
                        "and data folder, an install in progress, this PC's hardware (OS, CPU, RAM, GPUs, free disk) "
                        "and the model size setup recommends for this PC. Read-only; call it first.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "hardware": {"type": "boolean", "description": "include the hardware check and the recommendation "
                                                            "(default true; takes a second or two)"}}},
         "annotations": ro},
        {"name": "strata_models", "title": "Strata models",
         "description": "The models and sizes Strata's setup offers (families: qwen = Qwen3.8-Flash-Next, swift = "
                        "Swift 1.5, coder = the Coder, unsloth = Unsloth's ~4-bit), with download size, RAM needed, "
                        "whether each fits this PC, which are installed, the context lengths, and the recommendation. "
                        "Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
         "annotations": ro},
        {"name": "strata_remove", "title": "Delete an installed model",
         "description": "Delete an installed model's own files (its model GGUFs, its pack folder, its run config) and nothing "
                        "another installed model still uses (a shared MTP draft or PLE file). Without confirm=true it only "
                        "returns the plan (what would be removed and how big, what is kept). Stop the model first.",
         "inputSchema": {"type": "object", "additionalProperties": False, "required": ["model"],
                         "properties": {"model": installed_model, "confirm": {"type": "boolean"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": True}},
        {"name": "strata_install", "title": "Install Strata",
         "description": "Install a Strata model on this PC by running Strata's setup non-interactively in the "
                        "background (Python packages, the engine, a 58-111 GB model download, preparation, start "
                        "script). Without confirm=true it only returns the plan (what would be downloaded, disk "
                        "space): show it to the user and call again with confirm=true once they agree. Returns at "
                        "once; call it again (no arguments needed) to see the progress, or with cancel=true to stop "
                        "it (it resumes later). Defaults: the recommended model and context for this PC, pictures on where the model can read them. "
                        "After it finishes, start the model with strata_start.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "family": {"type": "string", "enum": list(families),
                        "description": "qwen = Qwen3.8-Flash-Next (the original), swift = Swift 1.5 (thinks "
                                       "shorter), coder = the Coder (half the experts, ~32 GB RAM, for code), "
                                       "unsloth = Unsloth's 4- to 8-bit files (UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, "
                                       "UD-Q6_K_XL, Q8_0): every expert in RAM where they fit, else part of them read "
                                       "from the SSD"},
             "model": {"type": "string", "enum": list(models),
                       "description": "the size (quantization): see strata_models"},
             "context": {"type": "integer", "minimum": 1024, "maximum": 2097152, "examples": contexts,
                         "description": "context length in tokens (more needs more VRAM). Any whole number works, "
                                        "for example one typed into a model's run config by hand; the usual sizes are "
                                        f"{', '.join(str(c) for c in contexts)}"},
             "vision": {"type": "string", "enum": ["no", "yes", "gpu", "cpu"],
                        "description": "let the model read images (~0.9 GB more download). Default yes where the model can "
                                       "read them (not on a Mac): yes = the best reader for this card (the CPU "
                                       "under 24 GB of VRAM, the GPU from 24 GB, any make), gpu / cpu = that "
                                       "reader, no = text only"},
             "kv": {"type": "string", "enum": ["int8", "q4_0", "k8v4"],
                    "description": "KV cache precision above 8K context (default int8)"},
             "gpu": {"type": "integer", "minimum": 0, "maximum": 63,
                     "description": "one GPU, as nvidia-smi numbers them (default: the one with the most VRAM)"},
             "gpus": {"type": "string", "pattern": GPUS_PATTERN, "maxLength": 40,
                      "description": "several GPUs sharing the model: '0,2' or 'all' (experimental)"},
             "backend": {"type": "string", "enum": ["auto", "cuda", "hip"],
                         "description": "auto (default), cuda = NVIDIA, hip = AMD on Linux (experimental)"},
             "low_ram": {"type": "string", "enum": ["auto", "on", "off", "resident", "mmap"],
                         "description": "setup's low-RAM mode (default auto)"},
             "port": {**port, "description": "the port the model will listen on (default 8080)"},
             "data_dir": {"type": "string", "maxLength": 260,
                          "description": "where the model files go (default: the Strata-data folder next to the "
                                         "Strata folder). Only the current data folder, a folder inside the Strata "
                                         "folder, or a new absolute path whose last folder is named Strata... "
                                         "(e.g. D:\\Strata-data) outside system folders"},
             "gguf_dir": {"type": "string", "maxLength": 260,
                          "description": "a folder that already holds this model's GGUF files under their original names: setup "
                                         "checks them (size and SHA-256) and does not download them again"},
             "confirm": {"type": "boolean", "description": "true = really start the install (after the user agreed "
                                                           "to the plan and the download size)"},
             "cancel": {"type": "boolean", "description": "true = stop the install in progress"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                         "openWorldHint": True}},
        {"name": "strata_start", "title": "Start Strata",
         "description": "Start an installed Strata model in the background (as its start script does, without "
                        "opening a browser) and wait until it answers. Loading takes 1-3 minutes and the PC can be "
                        "slow meanwhile; if it is not ready within wait_seconds this returns 'loading' - check again "
                        "with strata_status. The model keeps running until strata_stop.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "model": installed_model, "port": port,
             "gpu": {"type": "integer", "minimum": 0, "maximum": 63,
                     "description": "run on this GPU for this start (as nvidia-smi numbers them)"},
             "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 600,
                              "description": "how long to wait for it to be ready (default 50: some AI apps end "
                                             "a tool call after 60 s)"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_stop", "title": "Stop Strata",
         "description": "Stop the Strata model: the engine is unloaded through the server's own API (it frees its "
                        "VRAM and RAM itself), then the server this tool started is ended. A Strata started "
                        "elsewhere (e.g. its own window) is only unloaded - close its window to end it. Refuses while "
                        "a request is running unless force=true.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "force": {"type": "boolean", "description": "stop even while a request is running"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_logs", "title": "Strata logs",
         "description": "The last lines of a Strata log: 'server' (the model server this tool started: loading, "
                        "requests, errors), 'engine' (the engine's own log for a model), 'setup' (the install). "
                        "Default: the setup log while an install runs, else the server log. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "source": {"type": "string", "enum": ["server", "engine", "setup"]},
             "model": installed_model,
             "lines": {"type": "integer", "minimum": 1, "maximum": 500, "description": "default 80"}}},
         "annotations": ro},
        {"name": "strata_benchmark", "title": "Strata speed test",
         "description": "A short fixed request (greedy, thinking off) through the running Strata server; reports "
                        "the output speed (tokens/s) and prompt speed. Takes 10-60 s. Refuses while the model is "
                        "busy.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "max_tokens": {"type": "integer", "minimum": 16, "maximum": 512, "description": "default 200"}}},
         "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}},
        {"name": "strata_connect_info", "title": "Connect to Strata",
         "description": "How to point other apps at the local Strata server: the OpenAI and Anthropic base URLs, "
                        "the model id, whether an API key is needed, and ready-to-paste settings for Claude Code, "
                        "Codex CLI, the OpenAI SDK, curl, Cursor/Continue/Open WebUI. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"port": port}},
         "annotations": ro},
    ]


class Tools:
    def __init__(self, strata: Strata):
        self.s = strata

    def defs(self) -> list:
        models, families, contexts = self.s.tables()[:3]
        return tool_defs({**{name: {} for name in self.s.own_size_names() if name not in models}, **models}, families, contexts)

    def call(self, name: str, args) -> dict:
        defs = {d["name"]: d for d in self.defs()}
        if name not in defs:
            raise KeyError(name)
        args = check_schema(defs[name]["inputSchema"], args)
        return getattr(self, name)(**args)

    # ---- strata_status
    def strata_status(self, port=None, hardware=True) -> dict:
        s = self.s
        s.reap()
        servers = []
        for p in s.ports(port):
            info = s.probe(p)
            if info is not None:
                servers.append(info)
        tracked = s.tracked_server()
        out = {}
        running = [x for x in servers if not x.get("other")]
        if tracked and not tracked.get("ended"):
            mine = next((x for x in running if x["port"] == tracked["port"]), None)
            if mine:
                mine["started_by"] = f"this MCP server (pid {tracked['pid']})"
            else:
                out["loading"] = {"model": tracked.get("model"), "port": tracked["port"], "pid": tracked["pid"],
                                  "since": tracked.get("started"),
                                  "last_log_line": (read_tail(Path(tracked.get("log", "")), 1) or [None])[-1]}
        elif tracked and tracked.get("ended"):
            out["last_start"] = {"model": tracked.get("model"), "ended": True,
                                 "last_log_lines": read_tail(Path(tracked.get("log", "")), 6)}
        for x in running:
            x.setdefault("started_by", "outside this MCP server (its own window, a start script or a service)")
        out["running"] = running
        others = [x for x in servers if x.get("other")]
        if others:
            out["ports_used_by_other_programs"] = others
        if running:
            r = running[0]
            summary = (f"Strata is running on port {r['port']}: {r.get('model')}"
                       + (", busy" if r.get("busy") else ", idle")
                       + ("" if r.get("loaded", True) else " (model unloaded; the next request loads it)"))
        elif "loading" in out:
            summary = f"Strata is loading {out['loading']['model']} (port {out['loading']['port']})"
        else:
            summary = "Strata is not running"
        inst = s.installed()
        job = s.install_job()
        if job:
            out["install"] = job
            if job["running"]:
                summary += f"; an install is running ({job.get('step') or 'starting'})"
        if not inst["models"]:
            summary += "; no model is installed yet"
        else:
            summary += "; installed: " + ", ".join(m["model"] for m in inst["models"])
        out["installed"] = inst
        if hardware:
            hw = s.hardware()
            out["hardware"] = hw
            out["recommendation"] = s.recommend(hw)
        return {"summary": summary, **out}

    # ---- strata_models
    def strata_models(self) -> dict:
        s = self.s
        hw = s.hardware()
        _m, _f, contexts, src = s.tables()
        rec = s.recommend(hw)
        return {"summary": "Model families and sizes setup offers" + (
                    f"; recommended for this PC: {rec['title']}, {rec['context'] // 1024}K context"
                    if rec.get("model") else f"; no model fits this PC: {rec['why']}"),
                "families": s.model_table(hw), "contexts": contexts,
                "context_note": "longer contexts need more VRAM (fewer experts fit on the GPU); past 262144 setup adds "
                                "rope scaling (experimental)",
                "recommendation": rec, "source": src,
                "this_pc": {"ram_gb": hw.get("ram_gb"),
                            "gpus": [{k: g.get(k) for k in ("index", "name", "vram_gb", "usable", "vendor")}
                                     for g in hw.get("gpus", [])],
                            **({"engine_vendor": hw["engine_vendor"]} if hw.get("engine_vendor") else {}),
                            **({"gpu_note": hw["gpu_note"]} if hw.get("gpu_note") else {})}}

    # ---- strata_remove
    def strata_remove(self, model=None, confirm=False) -> dict:
        s = self.s
        config = s.find_config(model)
        plan = s.removal_plan(config)
        if not confirm:
            return {"summary": f"Plan: delete {plan['model']} ({plan['gb']:.1f} GB). Nothing was done yet.", "plan": plan}
        with s.lock:
            job = s.install_job()
            if job and job["running"]:
                raise ToolError("an install is running: wait for it to finish or stop it first")
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended") and tracked.get("model") == plan["model"]:
                raise ToolError("this model is running: stop it first")
            failed = []
            # The plan can list a folder and files inside it, or a file that is already gone (an earlier delete, a file moved by hand):
            # a path that is not there is already removed, not an error. Only what is still there afterwards counts as a failure.
            for item in sorted(plan["remove"], key=lambda x: (x["path"] == str(config), x["path"])):
                p = Path(item["path"])
                try:
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p, onerror=_rmtree_ignore_missing)
                    else:
                        p.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    if p.exists() or p.is_symlink():
                        failed.append(f"{p.name}: {e}")
            files, dirs = s.model_paths(s.read_config(config)) if config.exists() else ([], [])
            for p in {x.parent for x in [Path(i["path"]) for i in plan["remove"]] if x.suffix == ".gguf"}:
                try:
                    p.rmdir()                           # the model's own folder, when nothing else is in it
                except OSError:
                    pass
        return {"summary": f"Deleted {plan['model']} ({plan['gb']:.1f} GB)" + (f"; could not remove {len(failed)} item(s)" if failed else ""),
                "plan": plan, "failed": failed}

    # ---- strata_install
    def check_data_dir(self, raw: str) -> Path:
        s = self.s
        if raw.startswith("\\\\") or raw.startswith("//"):
            raise ToolError("data_dir: network (UNC) paths are not allowed; use a local drive")
        p = Path(os.path.expandvars(raw)).expanduser()
        if not p.is_absolute():
            raise ToolError("data_dir must be an absolute path (e.g. D:\\Strata-data or /mnt/big/Strata-data)")
        if ".." in p.parts:
            raise ToolError("data_dir must not contain '..'")
        n = norm(p)
        current = norm(s.data_dir())
        if n == current or is_inside(n, norm(s.root)) or n == norm(s.root.parent / "Strata-data"):
            pass
        else:
            if not p.name.lower().startswith("strata"):
                raise ToolError(f"data_dir {raw!r}: a new data folder must be named Strata... (e.g. "
                                f"{Path(p.anchor or '/') / 'Strata-data'}), so that no other folder is filled with "
                                "model files")
            if not p.parent.is_dir():
                raise ToolError(f"data_dir {raw!r}: its parent folder {p.parent} does not exist")
            if any(n == f or is_inside(n, f) for f in system_folders()):
                raise ToolError(f"data_dir {raw!r} is inside a system folder")
            if p.exists() and not p.is_dir():
                raise ToolError(f"data_dir {raw!r} is a file")
        return p

    def strata_install(self, family=None, model=None, context=None, vision=None, kv=None, gpu=None, gpus=None,
                       backend=None, low_ram=None, port=None, data_dir=None, confirm=False, cancel=False,
                       gguf_dir=None) -> dict:
        s = self.s
        with s.lock:
            job = s.install_job()
            if cancel:
                if not job or not job["running"]:
                    return {"summary": "no install is running", "install": job}
                j = s.state("install")
                res = s.state("install-result")
                child = res.get("child_pid") if res.get("job") == j.get("job") else None
                tree = proc_tree_snapshot(j["pid"]) if proc_alive(j["pid"], j["ident"]) else []   # setup.py and what it runs (#3257)
                if child:
                    proc_kill_tree(child, res.get("child_ident"))
                proc_kill_tree(j["pid"], j["ident"])
                kill_snapshot(tree)
                wait_gone(j["pid"], j["ident"], 15)
                return {"summary": "the install was stopped; calling strata_install again resumes it (downloads "
                                   "continue where they stopped)", "install": s.install_job()}
            if job and job["running"]:
                return {"summary": f"an install is running: {job.get('step') or 'starting'}"
                                   + (f", downloading {job['download']['file']} {job['download']['percent']}%"
                                      if job.get("download") else ""),
                        "install": job, "follow": "call strata_install again for progress, or strata_logs "
                                                  "source=setup for the full output"}
            given = [x for x in (family, model, context, vision, kv, gpu, gpus, backend, low_ram, port, data_dir)
                     if x is not None]
            if job and not given and not confirm:
                return {"summary": f"the last install {job['result']}" + (f": {job['error']}" if job.get("error") else ""),
                        "install": job,
                        "next": "strata_start" if job.get("exit_code") == 0 else
                                "fix the problem above and call strata_install again with confirm=true (finished "
                                "steps are skipped)"}
            plan, args = self.install_plan(family, model, context, vision, kv, gpu, gpus, backend, low_ram, port,
                                           data_dir, gguf_dir)
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended"):
                raise ToolError("Strata is running (started by strata_start): stop it first with strata_stop - setup "
                                "may update the engine files it uses")
            if not confirm:
                return {"summary": f"Plan: install {plan['title']} ({plan['download_gb']:.0f} GB download"
                                   f"{' at most; part of it is already here' if plan['partly_downloaded'] else ''}). "
                                   "Nothing was done yet: ask the user, then call strata_install again with the "
                                   "same arguments and confirm=true.", "plan": plan}
            if plan["disk_short"]:
                raise ToolError(plan["disk_short"])
            return self.launch_install(plan, args)

    def install_plan(self, family, model, context, vision, kv, gpu, gpus, backend, low_ram, port, data_dir, gguf_dir=None):
        s = self.s
        models, families, contexts, _src = s.tables()
        if gpu is not None and gpus is not None:
            raise ToolError("give gpu (one card) or gpus (several), not both")
        if sys.version_info < (3, 10) or (WIN and sys.maxsize <= 2**32):
            raise ToolError("this Python is too old or 32-bit: Strata's setup needs 64-bit Python 3.10+; run "
                            + ("START-HERE.bat" if WIN else "./setup.sh") + " once instead (it installs Python)")
        hw = s.hardware()
        rec = s.recommend(hw)
        if model is None and family is None:
            if not rec.get("model"):
                raise ToolError("no model was given and none fits this PC: " + rec["why"])
            family, model = rec["family"], rec["model"]
            context = context or rec["context"]
        elif model is None:
            sizes = s.sizes_of(models, family)
            model = rec["model"] if rec.get("family") == family else ("IQ3_XXS" if "IQ3_XXS" in sizes and
                                                                     hw.get("ram_gb", 0) >= 60 else sizes[0])
        elif family is None:
            fams = [f for f in families if model in s.sizes_of(models, f)]
            family = "qwen" if "qwen" in fams else fams[0]
        if family not in families:
            raise ToolError(f"unknown family {family}; the families: {', '.join(families)}")
        sizes = s.sizes_of(models, family)
        if model not in sizes:
            raise ToolError(f"{families[family]['title']} has no {model}; its sizes: {', '.join(sizes)}")
        md = s.own_size(family, model) or models[model]   # Qwen3.6-35B-A3B: its sizes are not in setup's MODELS
        mac = bool(hw.get("mac"))
        if mac:                                         # a Mac: no card to pick, no KV choice, no RAM mode: unified memory decides
            if rec.get("model") is None and model is not None and s.mac() and hw.get("ram_gb", 0) < s.mac().MIN_STREAM_RAM_GB:
                raise ToolError(f"{model} does not fit this Mac: {rec['why']}")
            gpu = gpus = kv = low_ram = None
            if md.get("multi_file"):                    # DeepSeek on a Mac (0.2.7.1): the sets that were run there, 8K like everywhere, text only
                from strata_control import dsv41_fit
                if model in dsv41_fit.MAC_NOT_YET:
                    raise ToolError(dsv41_fit.MAC_NOT_YET_LINE)
                if model not in dsv41_fit.MAC_SET_PACK:
                    raise ToolError(f"{model} is not offered on a Mac: DeepSeek runs there as " + " or ".join(dsv41_fit.MAC_SET_PACK))
                if vision in ("yes", "gpu", "cpu"):
                    raise ToolError("images are not available with DeepSeek on a Mac yet: use vision=no")
                vision = "no"
                context = context or getattr(s.setup_module(), "DSV41_CONTEXT", 8192)
            context = context or (s.mac().recommend_context(hw.get("ram_gb") or 0) if s.mac() else 32768)
        if context is None and md.get("multi_file"):
            context = getattr(s.setup_module(), "DSV41_CONTEXT", 8192)      # the context the engine was checked at
        if md.get("multi_file"):                          # DeepSeek-V4.1-Flash reads up to 1,048,576 positions (Project Maya #442): a size past that cannot run
            from strata_control import dsv41_context
            if dsv41_context.problem(int(context)):
                raise ToolError(dsv41_context.problem(int(context)))
        if context is None:
            usable = s.engine_gpus(hw)
            vram = max((g["vram_gb"] for g in usable), default=12)
            context = 32768 if vram < 14 else 65536 if vram < 20 else 131072
            if md.get("budget") and (not hasattr(s.setup_module(), "unsloth_budget") or
                                                s.setup_module().unsloth_budget(model, hw.get("ram_gb") or 0.0)):
                context = 8192 if vram < 14 else 32768
        if vision in ("yes", "gpu", "cpu") and md.get("vision", families[family].get("vision")) is False:
            raise ToolError(f"images are not available with {families[family]['title']} {model} yet: use vision=no")
        if vision is None:
            # owner rule (2026-10-18): pictures are on by default wherever the model can read them and the engine has a reader; setup picks
            # the reader (strata_control/vision_reader.py: the CPU under 24 GB, the GPU from 24 GB, every make of card; a Mac reads on its GPU).
            can_read = md.get("vision", families[family].get("vision")) is not False
            vision = "yes" if can_read else "no"
        target = self.check_data_dir(data_dir) if data_dir else s.data_dir()
        tag = (families[family].get("tag", "") + model)
        have_dir = target / "models" / tag
        partly = (have_dir.is_dir() and any(have_dir.glob("*.gguf*"))) or bool(gguf_dir)   # a local folder: nothing to download
        need = (8 if partly else md["download_gb"] + 8) + (1 if vision in ("yes", "gpu", "cpu") else 0)
        free = s.disk_free(target)
        short = None
        if free is not None and free < need:
            short = (f"not enough free disk space for {model}: about {need:.0f} GB needed at {target}, "
                     f"{free:.0f} GB free; free some space or choose data_dir on a bigger drive (e.g. D:\\Strata-data)")
        args = ["--yes", "--no-start", "--family", family, "--model", model, "--context", str(context),
                "--vision", vision or "no"]
        if family in ("qwen", "coder"):
            args += ["--experimental-speed-projection", "off"]
        own = bool(s.own_families().get(family))
        for flag, v in (("--kv", kv), ("--gpu", gpu), ("--gpus", gpus), ("--low-ram", low_ram), ("--port", port),
                        ("--data-dir", str(target) if data_dir else None)):
            if own and flag in ("--kv", "--low-ram"):      # Flash-Next settings: Qwen3.6's engine path does not use them (setup would warn)
                continue
            if v is not None:
                args += [flag, str(v)]
        if gguf_dir:
            local = Path(os.path.expandvars(gguf_dir)).expanduser()
            if not local.is_absolute() or not local.is_dir() or ".." in local.parts:
                raise ToolError("gguf_dir must be an existing absolute folder")
            args += ["--gguf-dir", str(local)]
        if backend in ("cuda", "hip"):
            args += ["--backend", backend]
        fit = None
        ram = hw.get("ram_gb")
        if ram is not None and md.get("multi_file"):
            fit = s.dsv41_words(hw, model)
        elif ram is not None:
            need_ram = md["ram_gb"]
            fit = "fits" if ram >= need_ram else f"needs ~{need_ram} GB of RAM, this PC has {ram:.0f} GB"
        installed = (s.root / f"strata-{tag.lower()}.json").exists()
        plan = {"title": f"{families[family]['title']} {model}", "family": family, "model": model,
                "context": context, "images": vision or "no", "download_gb": md["download_gb"],
                "partly_downloaded": partly, "ram": fit, "data_folder": str(target), "model_folder": str(have_dir),
                "disk_free_gb": free,
                "disk_needed_gb": round(need), "disk_short": short, "already_installed": installed,
                "setup_command": ("START-HERE.bat " if WIN else "./setup.sh ") + " ".join(args),
                "takes": "about 20-90 minutes, mostly the download; the PC stays usable",
                "recommended_for_this_pc": rec if rec.get("model") else None}
        if installed:
            plan["note"] = ("this model is already installed: running setup again updates the engine and applies "
                            "the options (finished steps are skipped)")
        if families[family].get("license"):
            plan["license"] = families[family]["license"]
        if families[family].get("experimental") or md.get("experimental"):
            plan["experimental"] = True
        return plan, args

    def launch_install(self, plan: dict, args: list) -> dict:
        s = self.s
        s.state_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"{int(time.time())}-{os.getpid()}"
        log_path = s.state_dir / "setup.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== strata-mcp: install {plan['title']} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        spec = {"job": job_id, "root": str(s.root), "src": str(s.src), "control": CONTROL,
                "python": s.run_python(), "base_python": sys.executable,
                "create_venv": s.python_override is None and not s.venv_python().exists(),
                "args": args, "result": str(s.state_dir / "install-result.json")}
        spec_path = s.state_dir / "install-job.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        s.save_state("install-result", None)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        p = spawn_detached([sys.executable, str(Path(__file__).resolve()), "--install-job", str(spec_path)],
                           s.root, log_path, env)
        s.children.append(p)
        time.sleep(0.2)
        ident = proc_identity(p.pid)
        if ident is None and p.poll() and s.state("install-result").get("job") != job_id:
            raise ToolError("setup did not start: " + " | ".join(read_tail(log_path, 5)))
        folder = plan.get("model_folder")
        files = []
        if s.setup_module() is not None and folder:     # every file of this install, for a progress that counts all the parts
            files = expected_files(s.setup_module(), plan["family"], plan["model"], Path(folder).parent,
                                   plan.get("images") in ("yes", "gpu", "cpu"))
        s.save_state("install", {"job": job_id, "pid": p.pid, "ident": ident, "log": str(log_path),
                                 "model": plan["model"], "family": plan["family"], "folder": folder, "files": files,
                                 "total_gb": plan.get("download_gb"),
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": args})
        return {"summary": f"Install of {plan['title']} started in the background (about 20-90 minutes, mostly the "
                           f"{plan['download_gb']:.0f} GB download). Call strata_install again to see the progress; "
                           "when it is done, start it with strata_start.",
                "plan": plan, "log": str(log_path), "pid": p.pid}

    # ---- strata_start
    def strata_start(self, model=None, port=None, gpu=None, wait_seconds=50) -> dict:
        s = self.s
        with s.lock:
            job = s.install_job()
            if job and job["running"]:
                raise ToolError("an install is still running (strata_install shows its progress); start the model "
                                "when it is done")
            cfg_path = s.find_config(model)
            desc = s.describe_config(cfg_path)
            if not desc["ready"]:
                raise ToolError(f"{cfg_path.name} refers to missing files ({', '.join(desc['missing_files'])}): run "
                                "the install again (strata_install with confirm=true) to repair it")
            port = port or desc["port"] or DEFAULT_PORT
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended"):
                info = s.probe(tracked["port"])
                return {"summary": f"Strata is already {'running' if info else 'loading'} ({tracked.get('model')}, "
                                   f"port {tracked['port']}, started by this MCP server)"
                                   + ("" if tracked.get("model") == desc["model"] else
                                      ": stop it first (strata_stop) to start another model"),
                        "server": info}
            info = s.probe(port, deep=False)
            if info and not info.get("other"):
                return {"summary": f"a Strata server already runs on port {port} ({info.get('model')}), started "
                                   "outside this MCP server; stop it (close its window) or use another port",
                        "server": info}
            if info or not port_free(port):
                raise ToolError(f"port {port} is used by another program; choose another port (port=...)")
            py = s.run_python()
            if not Path(py).exists():
                raise ToolError("Strata's Python environment (.venv) is missing: run the install again")
            cmd = [py, str(s.root / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
                   "--port", str(port)]
            if gpu is not None:
                cmd += ["--gpu", str(gpu)]
            log_path = s.state_dir / "server.log"
            s.state_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== strata-mcp: start {desc['model']} on port {port} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                os.utime(cfg_path)                      # the most recently used model, as setup's start marks it
            except OSError:
                pass
            p = spawn_detached(cmd, s.root, log_path, env)
            s.children.append(p)
            time.sleep(0.1)
            ident = proc_identity(p.pid)
            s.save_state("server", {"pid": p.pid, "ident": ident, "port": port, "model": desc["model"],
                                    "config": cfg_path.name, "log": str(log_path),
                                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        return self.wait_ready(p, ident, port, desc, wait_seconds, log_path)

    def wait_ready(self, p, ident, port, desc, wait_seconds, log_path) -> dict:
        s = self.s
        end = time.time() + wait_seconds
        while True:
            if p.poll() is not None or not proc_alive(p.pid, ident):
                tail = read_tail(log_path, 15)
                end_group_leftovers(p.pid)
                s.save_state("server", None)
                raise ToolError(f"Strata stopped while starting (exit code {p.poll()}). Its last output:\n"
                                + "\n".join(tail))
            info = s.probe(port)
            if info and not info.get("other"):
                return {"summary": f"Strata is running: {info.get('model')} on http://127.0.0.1:{port}/v1 "
                                   "(OpenAI and Anthropic APIs; chat page at http://127.0.0.1:" + str(port) + "/)",
                        "server": {**info, "pid": p.pid}}
            if time.time() >= end:
                return {"summary": f"Strata is loading {desc['model']} (this takes 1-3 minutes; the PC may be slow "
                                   "meanwhile). Check with strata_status in a minute.",
                        "loading": {"pid": p.pid, "port": port,
                                    "last_log_line": (read_tail(log_path, 1) or [None])[-1]}}
            time.sleep(1.0)

    # ---- strata_stop
    def strata_stop(self, port=None, force=False) -> dict:
        s = self.s
        with s.lock:
            tracked = s.tracked_server()
            if tracked and tracked.get("ended"):
                s.save_state("server", None)
                tracked = None
            if tracked:
                port = tracked["port"]
                key = s.api_key_for(port)
                st, body = http_json(f"http://127.0.0.1:{port}/status", api_key=key, timeout=3)
                if st == 200 and isinstance(body, dict) and body.get("busy") and not force:
                    raise ToolError("a request is running on Strata right now; wait for it, or call strata_stop "
                                    "with force=true to stop anyway")
                st, body = http_json(f"http://127.0.0.1:{port}/unload", "POST", api_key=key, timeout=120)
                unloaded = (body or {}).get("status") if isinstance(body, dict) else None
                if st == 409 and not force:
                    raise ToolError("Strata is busy (a request is running or queued); call again with force=true "
                                    "to stop anyway")
                proc_terminate(tracked["pid"], tracked["ident"])   # POSIX: SIGTERM = the server's own Ctrl+C path
                clean = wait_gone(tracked["pid"], tracked["ident"], 30)
                if not clean:
                    proc_kill_tree(tracked["pid"], tracked["ident"])
                    wait_gone(tracked["pid"], tracked["ident"], 15)
                s.reap()
                gone = not proc_alive(tracked["pid"], tracked["ident"])
                if gone:
                    end_group_leftovers(tracked["pid"])
                    s.save_state("server", None)
                return {"summary": "Strata stopped" + ("" if gone else " - but its process still runs; see strata_logs")
                                   + (" (the engine unloaded itself first)" if unloaded == "unloaded" else ""),
                        "model": tracked.get("model"), "port": port, "engine_unloaded": unloaded,
                        "ended_cleanly": clean}
            for p in s.ports(port):
                info = s.probe(p, deep=False)
                if info and not info.get("other"):
                    key = s.api_key_for(p)
                    st, body = http_json(f"http://127.0.0.1:{p}/unload", "POST", api_key=key, timeout=120)
                    r = body.get("status") if isinstance(body, dict) else None
                    if st == 409:
                        raise ToolError("Strata is busy (a request is running); try again when it is done")
                    if st == 401:
                        raise ToolError("Strata (started outside this MCP server) wants an API key this folder's "
                                        "configs do not have; close its window to stop it")
                    return {"summary": f"Strata on port {p} was started outside this MCP server (its own window or a "
                                       f"service), so only its model was unloaded ({r}): its VRAM and RAM are free, "
                                       "but the server still listens and loads the model again on the next request. "
                                       "Close its window to end it completely.",
                            "port": p, "engine_unloaded": r}
            return {"summary": "Strata is not running"}

    # ---- strata_logs
    def strata_logs(self, source=None, model=None, lines=80) -> dict:
        s = self.s
        if source is None:
            job = s.install_job()
            source = "setup" if job and job["running"] else "server"
        if source == "setup":
            path = s.state_dir / "setup.log"
        elif source == "server":
            path = s.state_dir / "server.log"
        else:
            cfg = s.find_config(model or (s.state("server").get("model")))
            raw = s.read_config(cfg).get("log")
            if not raw:
                raise ToolError(f"{cfg.name} names no engine log")
            path = Path(raw)
            if not s.allowed(path):
                raise ToolError(f"{cfg.name}'s log {raw} is outside the Strata folder and its data folder: not shown")
        if not path.exists():
            return {"summary": f"no {source} log yet ({path})", "log": str(path), "lines": []}
        tail = read_tail(path, lines)
        return {"summary": f"last {len(tail)} lines of the {source} log", "log": str(path), "lines": tail}

    # ---- strata_benchmark
    def strata_benchmark(self, port=None, max_tokens=200) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if target is None:
            raise ToolError("Strata is not running: start it first (strata_start)")
        p = target["port"]
        key = s.api_key_for(p)
        st, stat = http_json(f"http://127.0.0.1:{p}/status", api_key=key, timeout=3)
        if st == 200 and isinstance(stat, dict) and (stat.get("busy") or stat.get("queued")):
            raise ToolError("Strata is busy with a request; run the speed test when it is idle")
        body = {"model": target.get("model") or "strata", "max_tokens": max_tokens, "temperature": 0,
                "reasoning_effort": "none", "stream": False,
                "messages": [{"role": "user", "content": BENCH_PROMPT}]}
        t0 = time.time()
        st, r = http_json(f"http://127.0.0.1:{p}/v1/chat/completions", "POST", body, api_key=key, timeout=600)
        wall = time.time() - t0
        if st != 200 or not isinstance(r, dict):
            msg = (r or {}).get("error", {}).get("message") if isinstance(r, dict) else None
            raise ToolError(f"the request failed (HTTP {st}): {msg or 'no answer'}")
        usage, tm = r.get("usage") or {}, r.get("timings") or {}
        n = usage.get("completion_tokens") or tm.get("predicted_n") or 0
        out = {"model": r.get("model"), "port": p, "prompt_tokens": usage.get("prompt_tokens"),
               "output_tokens": n, "wall_s": round(wall, 2),
               "output_tok_s": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
               "prompt_tok_s": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
               "wall_tok_s": round(n / wall, 1) if wall > 0 and n else None,
               "finish_reason": ((r.get("choices") or [{}])[0]).get("finish_reason"),
               "settings": f"greedy (temperature 0), thinking off, max_tokens {max_tokens}, a fixed story prompt"}
        speed = out["output_tok_s"] or out["wall_tok_s"]
        out["summary"] = (f"{n} tokens at {speed} tokens/s" + (f" (prompt {out['prompt_tok_s']} tokens/s)"
                                                              if out["prompt_tok_s"] else "")
                          + f", {wall:.1f} s in total")
        out["note"] = ("output speed depends on the context length, the GPU's free VRAM and what else the PC does; "
                       "the first request after a start is slower")
        return out

    # ---- strata_connect_info
    def strata_connect_info(self, port=None) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if port is None:
            cfgs = s.configs()
            port = target["port"] if target else (s.describe_config(cfgs[0])["port"] if cfgs else DEFAULT_PORT)
        model = (target or {}).get("model") or "strata"
        key_needed = (target or {}).get("api_key_required")
        if key_needed is None:
            key_needed = any(s.describe_config(c)["api_key_set"] for c in s.configs())
        key = "<your api_key from strata-<model>.json>" if key_needed else "none"
        base = f"http://127.0.0.1:{port}"
        return {
            "summary": (f"Strata {'runs' if target else 'will run'} at {base}: OpenAI base URL {base}/v1, Anthropic "
                        f"base URL {base}" + (" (an API key is required)" if key_needed else " (any API key works)")),
            "running": bool(target), "openai_base_url": f"{base}/v1", "anthropic_base_url": base,
            "anthropic_messages_endpoint": f"{base}/v1/messages", "chat_page": base + "/", "model": model,
            "api_key": ("required: the \"api_key\" value in the Strata folder's strata-<model>.json"
                        if key_needed else "not required (send any value)"),
            "examples": {
                "claude_code": {"env": {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key,
                                        "ANTHROPIC_MODEL": "claude-sonnet-4-5"},
                                "note": "Claude Code needs a Claude model name it knows; Strata ignores the name"},
                "codex_cli": {"file": "~/.codex/config.toml",
                              "toml": f'model = "{model}"\nmodel_provider = "strata"\n\n[model_providers.strata]\n'
                                      f'name = "Strata"\nbase_url = "{base}/v1"\nwire_api = "chat"'},
                "openai_python": (f'from openai import OpenAI\nclient = OpenAI(base_url="{base}/v1", api_key="{key}")\n'
                                  f'r = client.chat.completions.create(model="{model}", messages=[{{"role": "user", '
                                  f'"content": "Hello!"}}])\nprint(r.choices[0].message.content)'),
                "curl": (f"curl {base}/v1/chat/completions -H \"Content-Type: application/json\" "
                         + (f"-H \"Authorization: Bearer {key}\" " if key_needed else "")
                         + f"-d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": "
                           "\"Hello!\"}]}'"),
                "openai_compatible_apps": {"base_url": f"{base}/v1", "api_key": key, "model": model,
                                           "apps": "Cursor (Settings > Models > OpenAI base URL override), "
                                                   "Continue, Open WebUI, LM Studio clients, aider "
                                                   f"(--openai-api-base {base}/v1)"}},
            "other_devices": "the server listens on this PC only; to reach it from your network, run setup with "
                             "--host 0.0.0.0 --api-key <secret> (docs/DETAILS.md, 'Using it')"}


# ------------------------------------------------------------------------------------------------ MCP over stdio
INSTRUCTIONS = ("Strata runs a large AI model (Qwen3.8-Flash-Next, a mixture of experts) locally on this PC's GPU and "
                "RAM, with an OpenAI- and Anthropic-compatible server on http://127.0.0.1:8080. Call strata_status "
                "first. To install: strata_install (it returns a plan; get the user's OK for the 58-111 GB download, "
                "then call it with confirm=true; it runs in the background - call it again for progress). Then "
                "strata_start, strata_stop, strata_logs for problems, strata_connect_info to point other apps at it.")


class McpServer:
    def __init__(self, strata: Strata, out=None):
        self.strata = strata
        self.tools = Tools(strata)
        self.out = out
        self.write_lock = threading.Lock()
        self.protocol = PROTOCOL

    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            self.out.write(data)
            self.out.flush()

    @staticmethod
    def result(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    def handle(self, msg) -> dict | None:
        """One JSON-RPC message -> its response (None for a notification or a response)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self.error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
        if "method" not in msg:
            return None                                 # a response to something we never ask: ignored
        method, rid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        if "id" not in msg:
            return None                                 # notifications (initialized, cancelled ...): nothing to say
        if not isinstance(params, dict):
            return self.error(rid, -32602, "params must be an object")
        if method == "initialize":
            asked = str(params.get("protocolVersion") or PROTOCOL)
            self.protocol = asked if asked in PROTOCOLS else PROTOCOL
            return self.result(rid, {"protocolVersion": self.protocol,
                                     "capabilities": {"tools": {"listChanged": False}},
                                     "serverInfo": {"name": "strata", "title": "Strata", "version": VERSION},
                                     "instructions": INSTRUCTIONS})
        if method == "ping":
            return self.result(rid, {})
        if method == "tools/list":
            return self.result(rid, {"tools": self.tools.defs()})
        if method == "tools/call":
            name = params.get("name")
            try:
                res = self.tools.call(name, params.get("arguments"))
                return self.result(rid, self.content(res, error=False))
            except KeyError:
                return self.error(rid, -32602, f"unknown tool: {name}")
            except ToolError as e:
                return self.result(rid, self.content({"error": str(e)}, error=True))
            except Exception as e:                      # noqa: BLE001 - a bug must not end the server
                log("tool failed:\n" + traceback.format_exc())
                return self.result(rid, self.content({"error": f"internal error in {name}: {e}"}, error=True))
        if method in ("resources/list", "prompts/list"):
            return self.result(rid, {method.split("/")[0]: []})
        return self.error(rid, -32601, f"method not found: {method}")

    def content(self, res: dict, error: bool) -> dict:
        text = json.dumps(res, indent=1, ensure_ascii=False)
        out = {"content": [{"type": "text", "text": text}], "isError": error}
        if self.protocol >= "2025-06-18":
            out["structuredContent"] = res
        return out

    def serve(self, inp) -> None:
        """Read JSON-RPC lines until stdin closes; tool calls run in their own threads (a start waits for the model
        while pings and status calls are still answered)."""
        threads = []
        for raw in inp:
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(self.error(None, -32700, "parse error"))
                continue
            batch = isinstance(msg, list)
            if batch or (isinstance(msg, dict) and msg.get("method") == "tools/call"):
                t = threading.Thread(target=self._run, args=(msg,), daemon=True)
                t.start()
                threads.append(t)
                threads = [x for x in threads if x.is_alive()]
            else:
                self._run(msg)
        for t in threads:
            t.join(timeout=5)

    def _run(self, msg) -> None:
        if isinstance(msg, list):
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            if out:
                data = (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")
                with self.write_lock:
                    self.out.write(data)
                    self.out.flush()
            return
        r = self.handle(msg)
        if r is not None:
            self.send(r)


# ------------------------------------------------------------------------------------------------ the install job
def install_job(spec_path: str) -> int:
    """The background half of strata_install (this file started again with --install-job): make .venv like
    START-HERE.bat / setup.sh do when it is missing, run setup, record its exit code."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    root = Path(spec["root"])
    result = Path(spec["result"])

    def record(**kw):
        try:
            old = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        except (OSError, ValueError):
            old = {}
        tmp = result.with_suffix(".tmp")
        tmp.write_text(json.dumps({**old, "job": spec["job"], **kw}, indent=1), encoding="utf-8")
        os.replace(tmp, result)

    py = Path(spec["python"])
    src = Path(spec.get("src") or root)
    try:
        if spec.get("create_venv"):
            venv = root / ".venv"
            if py.exists() and subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True).returncode:
                print("strata-mcp: .venv has no pip (an earlier run stopped half-way): making it again", flush=True)
                shutil.rmtree(venv, ignore_errors=True)
            if not py.exists():
                print(f"strata-mcp: creating Strata's Python environment in {venv} ...", flush=True)
                r = subprocess.run([spec["base_python"], "-m", "venv", str(venv)])
                if r.returncode or not py.exists():
                    print("\n  [X]  could not create the Python environment (.venv)", flush=True)
                    print("       Linux: install python3-venv (sudo apt install python3-venv) or run ./setup.sh "
                          "once in a terminal", flush=True)
                    record(exit_code=1, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                    return 1
        cmd = [str(py), str(src / "setup.py"), *spec["args"]]
        print("strata-mcp: running " + " ".join(cmd), flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        if spec.get("control"):                         # the app: the installer's own mode, writing to `root`
            env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(root), PYTHONNOUSERSITE="1")
            for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
                env.pop(key, None)
            # This job is the explicit install the user asked for (it downloads model files and the preparation sources). The
            # server that started it, and every start or serve, keep STRATA_NO_ACQUIRE; setup.py's CONTROL mode still runs
            # no pip, git, compiler or engine fetch.
            env.pop("STRATA_NO_ACQUIRE", None)
        p = subprocess.Popen(cmd, cwd=str(src if spec.get("control") else root), stdin=subprocess.DEVNULL, env=env)
        record(child_pid=p.pid, child_ident=proc_identity(p.pid))
        rc = p.wait()
    except Exception as e:                              # noqa: BLE001
        print(f"\n  [X]  strata-mcp could not run setup: {e}", flush=True)
        rc = 1
    print(f"strata-mcp: setup ended with exit code {rc}", flush=True)
    record(exit_code=rc, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strata's MCP server (stdio): install, start, check and stop Strata "
                                             "from an AI assistant. See docs/MCP_SERVER.md.")
    ap.add_argument("--root", help="the Strata folder (default: the one this file is in)")
    ap.add_argument("--src", help="where setup.py, tools/ and serve/ are, when that is not the Strata folder (the app)")
    ap.add_argument("--install-job", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.install_job:
        return install_job(a.install_job)
    if sys.version_info < (3, 10):
        log("Python 3.10 or newer is needed")
        return 2
    root = Path(a.root).expanduser().resolve() if a.root else DEFAULT_ROOT
    src = Path(a.src).expanduser().resolve() if a.src else root
    if not (src / "setup.py").is_file() or not (src / "serve" / "server.py").is_file():
        log(f"{src} is not a Strata folder (no setup.py / serve/server.py)")
        return 2
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                             # stray prints must never corrupt the protocol stream
    server = McpServer(Strata(root, src), out)
    log(f"ready (Strata folder {root})")
    try:
        server.serve(sys.stdin.buffer)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
