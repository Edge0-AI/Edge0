"""Which processes Strata started, and how to tell one of them from a stranger.

Every engine, image encoder and server that Strata starts leaves a small private record: its PID, the time it
started, its command line, and the same two things for the process that started it. When that parent is gone and
the child is still running, the child is a leftover. A leftover is ended only when ALL of these still hold at the
moment of the signal:

* the process at that PID started at the recorded time (a later process that reused the number does not match);
* its command line is the recorded one (PID reuse plus the same start time is not possible, but this is also what
  makes an unrelated program impossible to hit by a corrupted record);
* the recorded parent is not running as that same process.

A PID that is only a number is never signalled. If the identity of a process cannot be read (no command line on
this platform and no psutil), it is left alone and reported.

Also here: the one-time check for engines left by earlier Strata versions, which wrote no record: an orphan of
exactly the engine executable that is about to be started, running ``--serve`` with no live parent.
"""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from strata_control import paths
from strata_control import atomic as _atomic_mod

WINDOWS = os.name == "nt"
SCHEMA = 1
INIT_NAMES = {"systemd", "init", "launchd", "upstart"}     # parents that adopt orphans

try:  # psutil ships in every Strata runtime (requirements.txt, the Control lock files)
    import psutil
except ImportError:  # pragma: no cover - the Linux /proc fallback below still identifies processes
    psutil = None


# ---------------------------------------------------------------------------------------------- identity

def start_token(pid: int) -> str | None:
    """A string that is the same for the lifetime of one process and different for a later process with the same
    PID. None when there is no such process or its start cannot be read."""
    if type(pid) is not int or pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            fields = raw[raw.rindex(")") + 2:].split()
            if fields[0] == "Z":                       # a zombie has exited; only its parent has not collected it
                return None
            return "l:" + fields[19]                   # starttime, clock ticks since boot (field 22)
        except (OSError, ValueError, IndexError):
            return None
    if WINDOWS:
        from strata_control import winproc
        started = winproc.process_started(pid)
        return None if started is None else f"w:{started}"
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return None
            return f"p:{p.create_time():.3f}"
        except (psutil.Error, OSError):
            return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=,stat=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None if not out or out.endswith("Z") else "s:" + out.rsplit(None, 1)[0]


def started_epoch(pid: int) -> float | None:
    """When the process at `pid` was created, in seconds since 1970, or None when there is no such process or the time cannot be read.
    A number recorded in an old file can belong to a later process; a creation time after the file was written proves it does."""
    if type(pid) is not int or pid <= 0:
        return None
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except (psutil.Error, OSError):
            return None
    if WINDOWS:
        from strata_control import winproc
        ticks = winproc.process_started(pid)
        return None if ticks is None else (ticks - 116444736000000000) / 1e7     # 100 ns ticks since 1601
    if sys.platform.startswith("linux"):
        token = start_token(pid)
        try:
            boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
            return boot + int(token[2:]) / os.sysconf("SC_CLK_TCK") if token else None
        except (OSError, ValueError, StopIteration):
            return None
    return None


def command_line(pid: int) -> list[str] | None:
    """The process's argv, or None when it cannot be read (then it is never treated as a match)."""
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part] or None
        except OSError:
            return None
    if psutil is not None:
        try:
            return list(psutil.Process(pid).cmdline()) or None
        except (psutil.Error, OSError):
            return None
    if not WINDOWS:
        try:
            out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return out.split() or None
    return None


def parent_of(pid: int) -> int | None:
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            return int(raw[raw.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return None
    if psutil is not None:
        try:
            return psutil.Process(pid).ppid()
        except (psutil.Error, OSError):
            return None
    return None


def exe_path(pid: int) -> str | None:
    if sys.platform.startswith("linux"):
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None
    if psutil is not None:
        try:
            return psutil.Process(pid).exe() or None
        except (psutil.Error, OSError):
            return None
    return None


def same_file(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


@dataclass(frozen=True)
class Ident:
    pid: int
    token: str
    cmd: tuple[str, ...]


def identify(pid: int) -> Ident | None:
    """The live process at `pid`, with the facts a record is later compared with."""
    token = start_token(pid)
    cmd = command_line(pid)
    if token is None or not cmd:
        return None
    return Ident(pid, token, tuple(cmd))


def is_same(pid: int, token: str, cmd: list[str] | tuple[str, ...] | None) -> bool:
    """True only when the process now at `pid` started at `token` and runs exactly `cmd`."""
    if start_token(pid) != token:
        return False
    now = command_line(pid)
    return bool(now) and list(now) == list(cmd or [])


# ---------------------------------------------------------------------------------------------- records

def procs_dir(create: bool = True) -> Path:
    """The owner-private folder of launch records (beside runtime/, under the per-user state)."""
    base = paths.state_base() / "procs"
    if create and not base.exists():
        _private_mkdir(base, parents=True, exist_ok=True)
        if WINDOWS:
            from strata_control import private_files
            private_files.restrict_to_current_user(base, directory=True)
    return base


def write_record(pid: int, role: str, *, profile: str | None = None, cmd: list[str] | None = None,
                 parent_pid: int | None = None) -> str | None:
    """Record that this process started `pid`; returns the process's start token (None: nothing was written).
    Best effort: a record that cannot be written never stops a launch."""
    try:
        token = start_token(pid)
        command = list(cmd) if cmd else command_line(pid)
        parent_pid = os.getpid() if parent_pid is None else parent_pid
        parent_token = start_token(parent_pid)
        if token is None or not command or parent_token is None:
            return None
        record = {"schema": SCHEMA, "pid": pid, "token": token, "cmd": command, "role": role,
                  "profile": profile, "parent": {"pid": parent_pid, "token": parent_token},
                  "written_at": time.time()}
        path = procs_dir() / f"{pid}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        _atomic_mod.replace(tmp, path)
        return token
    except (OSError, ValueError):
        return None


def drop_record(pid: int, token: str | None = None) -> None:
    """Delete the record of `pid`; with `token`, only if it is the record of that process (not of a later one)."""
    path = procs_dir(create=False) / f"{pid}.json"
    try:
        if token is not None:
            record = _load(path)
            if record is None or record["token"] != token:
                return
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _load(path: Path) -> dict | None:
    try:
        if path.stat().st_size > 16 * 1024:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = (isinstance(record, dict) and record.get("schema") == SCHEMA and type(record.get("pid")) is int and
          isinstance(record.get("token"), str) and isinstance(record.get("cmd"), list) and
          all(isinstance(part, str) for part in record["cmd"]) and isinstance(record.get("parent"), dict) and
          type(record["parent"].get("pid")) is int and isinstance(record["parent"].get("token"), str))
    return record if ok else None


def records(profile: str | None = None) -> list[dict]:
    base = procs_dir(create=False)
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("*.json")):
        record = _load(path)
        if record is None:
            continue
        if profile is None or record.get("profile") == profile:
            found.append(record)
    return found


# ---------------------------------------------------------------------------------------------- leftovers

def leftovers(profile: str | None = None, *, forget_stale: bool = True) -> tuple[list[dict], list[dict]]:
    """(to_end, unverifiable). `to_end`: recorded processes that still run as exactly what was recorded whose
    parent is no longer that process. Records of processes that have gone, or whose PID now belongs to something
    else, are deleted (when `forget_stale`); the stranger is not touched. `unverifiable`: still running but their
    command line cannot be read here, so they are reported and left alone."""
    to_end, unverifiable = [], []
    for record in records(profile):
        pid = record["pid"]
        token = start_token(pid)
        if token is None or token != record["token"]:
            if forget_stale:
                drop_record(pid)
            continue
        parent = record["parent"]
        if start_token(parent["pid"]) == parent["token"]:
            continue                                  # its starter is still there: not a leftover
        if is_same(pid, record["token"], record["cmd"]):
            to_end.append(record)
        elif command_line(pid) is None:
            unverifiable.append(record)
        elif forget_stale:
            drop_record(pid)                          # same start time, different command line: not ours
    return to_end, unverifiable


def _signal_verified(record: dict, sig: int) -> bool:
    """Signal the recorded process, but only if it is still the recorded process at this instant."""
    if not is_same(record["pid"], record["token"], record["cmd"]):
        return False
    try:
        os.kill(record["pid"], sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def end_one(record: dict, *, grace: float = 8.0, settle: float = 10.0, say=None) -> str:
    """End one verified leftover: a polite signal where the OS has one, then a kill after `grace` seconds, then up to
    `settle` seconds for it to be gone (#4051: an engine holding tens of GB of pinned memory can take longer than the
    10 s that were enough for a small process; `say` is told every ~10 s while it is waited for).
    Returns 'gone' (already not running), 'ended' or 'stuck'."""
    pid = record["pid"]
    if not is_same(pid, record["token"], record["cmd"]):
        drop_record(pid)
        return "gone"
    # once signalled, the start token alone says it still runs: an exiting process has an empty command line on Linux
    def running():
        return start_token(pid) == record["token"]
    if not WINDOWS:
        _signal_verified(record, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and running():
            time.sleep(0.1)
        if is_same(pid, record["token"], record["cmd"]):
            _signal_verified(record, signal.SIGKILL)
    else:
        if is_same(pid, record["token"], record["cmd"]):
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)      # one PID, verified just above
    started = time.monotonic()
    deadline = started + settle
    told = 0
    while time.monotonic() < deadline and running():
        time.sleep(0.1)
        waited = int((time.monotonic() - started) // 10)
        if say is not None and waited > told:
            told = waited
            say(f"[strata] still waiting for process {pid} to give its memory back ({told * 10} s)")
    if running():
        return "stuck"
    drop_record(pid)
    return "ended"


def end_leftovers(profile: str | None = None, *, grace: float = 8.0, settle: float = 10.0, say=None) -> dict:
    """End every verified leftover (children first: a record whose parent is itself a leftover goes last)."""
    to_end, unverifiable = leftovers(profile)
    children_first = sorted(to_end, key=lambda r: r["written_at"], reverse=True)
    result = {"ended": [], "gone": [], "stuck": [], "unverifiable": [
        {"pid": r["pid"], "role": r.get("role")} for r in unverifiable]}
    for record in children_first:
        outcome = end_one(record, grace=grace, settle=settle, say=say)
        result[outcome].append({"pid": record["pid"], "role": record.get("role"), "profile": record.get("profile")})
        if say is not None and outcome != "gone":
            say(f"[strata] leftover {record.get('role', 'process')} {record['pid']} {outcome}")
    return result


# ---------------------------------------------------------------------------- engines older versions left

def orphans_of(engine_exe: str) -> list[Ident]:
    """Processes that are exactly this engine executable, running ``--serve``, whose parent no longer exists.

    Earlier Strata versions wrote no record. Such an engine is not a service anybody started on purpose: the
    server that owned it is gone, and it holds the GPU. Only the executable that is about to be started is
    matched, never another installation's, and a process whose parent is alive is never an orphan."""
    if psutil is None or not engine_exe:
        return []
    mine = {os.getpid()}
    try:
        # pid 1 and a subreaper such as `systemd --user` (an ancestor of every desktop terminal) adopt orphans: they are not starters
        mine.update(q.pid for q in psutil.Process().parents() if q.pid != 1 and q.name().lower() not in INIT_NAMES)
    except psutil.Error:
        pass
    found = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        try:
            cmd = p.info.get("cmdline") or []
            if "--serve" not in cmd or p.pid in mine:
                continue
            if not same_file(p.exe(), engine_exe):
                continue
            parent = p.info.get("ppid")
            if parent in mine:
                continue                              # started by this server (or by a process above it)
            if parent and parent != 1:
                try:
                    q = psutil.Process(parent)
                    adopted = q.name().lower() in INIT_NAMES      # an init that adopts orphans did not start it
                    if not adopted and q.create_time() <= p.info["create_time"] + 1.0                             and q.status() != psutil.STATUS_ZOMBIE:
                        continue                      # a live parent started it: not an orphan
                except psutil.NoSuchProcess:
                    pass
            ident = identify(p.pid)
            if ident is not None:
                found.append(ident)
        except (psutil.Error, OSError):
            continue
    return found


def end_orphans(engine_exe: str, *, grace: float = 8.0, settle: float = 10.0, say=None) -> list[dict]:
    """End the orphans of `engine_exe` (see `orphans_of`), re-verified one by one."""
    ended = []
    for ident in orphans_of(engine_exe):
        record = {"pid": ident.pid, "token": ident.token, "cmd": list(ident.cmd), "role": "engine", "written_at": 0}
        outcome = end_one(record, grace=grace, settle=settle, say=say)
        ended.append({"pid": ident.pid, "outcome": outcome})
        if say is not None and outcome != "gone":
            say(f"[strata] an engine left behind by an earlier run (pid {ident.pid}) {outcome}")
    return ended
