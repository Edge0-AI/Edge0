"""Move the model files (models, packs, mtp) from one data folder to another (0.2.2, Settings > Storage > Models folder).

The desktop app and the Terminal both use this: ``strata models move``.  Rules, in the order they matter:

* Nothing is deleted until its copy is whole and checked.  One file at a time: copy to ``name.part``, compare the size, read the copy
  back and compare its SHA-256 with the original's (and with the pinned SHA-256 in the file's ``.done`` mark when it has one), rename it
  into place, only then remove the original.  The extra space needed is never more than the move itself.
* Stopping is safe.  The file in flight is discarded (its ``.part`` is removed), files already moved stay, and the run configs are
  rewritten for every model whose files are all in the new folder.  The data folder in the settings is changed by the caller only
  when the whole move is done, so the app is never pointed at an incomplete folder.
* The same drive is a rename (instant).  Another drive is a copy.
* A crash leaves at worst a model in both folders; ``name.part`` files are never trusted.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ITEMS = ("models", "packs", "mtp")
STATE_FILE = "models-move.json"
MARGIN = 1.05                       # room beside the files themselves: 5 %
CHUNK = 8 * 1024 * 1024


class MoveError(Exception):
    """A move that stopped for a reason the person can act on. ``code`` is one of: not_found, not_writable, inside, disk_full, verify, busy, io."""

    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code = code
        self.details = details


class Stopped(Exception):
    """The person asked to stop."""


@dataclass
class Entry:
    rel: str            # path under the data folder, with "/" (models/<tag>/file.gguf)
    size: int
    group: str          # the model it belongs to: "models/<tag>", "packs/<tag>", "mtp"


# ----------------------------------------------------------------------------------------------- what is there

def _item_dirs(data: Path) -> list[Path]:
    """The models / packs / mtp folders of a data folder (any spelling of the name: ``Models`` and ``models`` are one library)."""
    try:
        names = sorted(p for p in data.iterdir() if p.is_dir() and p.name.lower() in ITEMS)
    except OSError:
        return []
    return names


def inventory(data: Path) -> list[Entry]:
    """Every file Strata keeps in a data folder: nothing else is touched. Unfinished downloads (``.part``) are left where they are."""
    found: list[Entry] = []
    for top in _item_dirs(data):
        for root, _dirs, files in os.walk(top):
            for name in files:
                if name.endswith(".part"):
                    continue
                path = Path(root) / name
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                rel = path.relative_to(data).as_posix()
                parts = rel.split("/")
                kind = parts[0].lower()
                group = kind if kind == "mtp" or len(parts) < 3 else parts[1].lower()   # a model's GGUF files and its pack are one model
                found.append(Entry(rel, size, group))
    found.sort(key=lambda e: (e.group, e.rel))
    return found


def groups(entries: list[Entry]) -> list[dict]:
    """The models in a list of files, for the person: name (the folder name), bytes."""
    out: dict[str, int] = {}
    for e in entries:
        out[e.group] = out.get(e.group, 0) + e.size
    return [{"group": g, "name": g, "bytes": b} for g, b in out.items()]


def _near(path: Path) -> Path:
    """The folder that exists and holds ``path`` (or ``path`` itself)."""
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def same_drive(a: Path, b: Path) -> bool:
    """Both folders are on one volume (a move is then a rename). STRATA_MOVE_ALWAYS_COPY=1 copies anyway: for testing the copy path on one disk."""
    if os.environ.get("STRATA_MOVE_ALWAYS_COPY") == "1":
        return False
    try:
        return os.stat(_near(a)).st_dev == os.stat(_near(b)).st_dev
    except OSError:
        return False


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _inside(child: Path, parent: Path) -> bool:
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    return c != p and c.startswith(p.rstrip(os.sep) + os.sep)


def _write_test(folder: Path) -> str | None:
    """None when files can be written in ``folder`` (made if missing); else the raw refusal."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".strata-write-test", dir=folder)
        os.close(fd)
        os.remove(name)
        return None
    except OSError as e:
        return f"{errno.errorcode.get(e.errno, e.errno)}: {e.strerror or e}, {e.filename or folder}"


# ----------------------------------------------------------------------------------------------- plan

def plan(old: Path, new: Path, *, free=None, write_test: bool = True) -> dict:
    """What picking ``new`` for the models folder means, without changing anything.

    Returns ``{ok, issue, old, new, same, same_drive, free_bytes, models_bytes, needed_bytes, models, choices}``.  ``issue`` is None or
    ``{code, message, details}``: not_found (the folder or its drive is gone), not_writable, inside (one folder inside the other),
    same (nothing to do).  ``choices`` says which of move / keep / empty can be done and why not.
    """
    old, new = Path(old), Path(new)
    free = free or (lambda p: shutil.disk_usage(p).free)
    entries = inventory(old) if old.is_dir() else []
    total = sum(e.size for e in entries)
    result = {"ok": False, "issue": None, "old": str(old), "new": str(new), "same": False, "same_drive": False, "free_bytes": 0,
              "models_bytes": total, "needed_bytes": 0, "models": groups(entries), "choices": {}}

    def refuse(code, message, details=""):
        result["issue"] = {"code": code, "message": message, "details": details}
        result["choices"] = {k: {"possible": False, "reason": message} for k in ("move", "keep", "empty")}
        return result

    if _same(old, new):
        result["same"] = True
        return refuse("same", "That is the folder Strata already uses.")
    if _inside(new, old) or _inside(old, new):
        return refuse("inside", "Pick a folder outside the current one.")
    if not new.exists():
        # a folder that will be made is fine when its parent is there; a drive that is not plugged in is not
        drive = Path(new.anchor) if new.anchor else None
        if drive is not None and not drive.exists():
            return refuse("not_found", f"Strata can't find {new}.", f"{drive} does not exist")
    if write_test:
        raw = _write_test(new)
        if raw:
            return refuse("not_writable", f"Strata can't save files in {new}.", raw)
    result["same_drive"] = same_drive(old, new)
    try:
        result["free_bytes"] = int(free(_near(new)))
    except OSError:
        result["free_bytes"] = 0
    # a same-drive move is a rename: it needs no room
    needed = 0 if result["same_drive"] else int(total * MARGIN)
    result["needed_bytes"] = needed
    can_move = bool(entries) and result["free_bytes"] >= needed
    reason_move = "" if can_move else ("You have no models to move." if not entries
                                       else f"Your models need {_gb(needed)}, this drive has {_gb(result['free_bytes'])} free.")
    result["choices"] = {"move": {"possible": can_move, "reason": reason_move}, "keep": {"possible": True, "reason": ""},
                         "empty": {"possible": True, "reason": ""}}
    result["ok"] = True
    return result


def _gb(n: int) -> str:
    g = n / 1e9
    return f"{g:.0f} GB" if g >= 10 else f"{g:.1f} GB"


# ----------------------------------------------------------------------------------------------- the move

def _sha256_of(path: Path, stop: Callable[[], bool], tick: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if stop():
                raise Stopped()
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            if tick:
                tick(len(block))
    return h.hexdigest()


def _pinned(src: Path) -> str | None:
    """The SHA-256 a finished download wrote in its ``.done`` mark (setup.py: ``sha256 <hex>``), when there is one."""
    mark = src.with_name(src.name + ".done")
    try:
        m = re.search(r"sha256\s+([0-9a-fA-F]{64})", mark.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return m.group(1).lower() if m else None


def _disk_full(e: OSError) -> bool:
    return e.errno in (errno.ENOSPC, errno.EDQUOT) or getattr(e, "winerror", None) in (112, 39)


def _copy_verified(src: Path, dst: Path, size: int, stop: Callable[[], bool], tick: Callable[[int], None]) -> None:
    """Copy src to dst (through dst.part), prove the copy is the same file, rename it into place. The source is left alone."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    try:
        with open(src, "rb") as fin, open(part, "wb") as fout:
            while True:
                if stop():
                    raise Stopped()
                block = fin.read(CHUNK)
                if not block:
                    break
                h.update(block)
                fout.write(block)
                tick(len(block))
            fout.flush()
            os.fsync(fout.fileno())
        if part.stat().st_size != size:
            raise MoveError("verify", f"The copy of {src.name} has the wrong size.", f"{part.stat().st_size} != {size}")
        want = h.hexdigest()
        pinned = _pinned(src)
        if pinned and pinned != want:
            raise MoveError("verify", f"{src.name} does not match its saved checksum, so it was not moved.", f"sha256 {want} != pinned {pinned}")
        got = _sha256_of(part, stop)
        if got != want:
            raise MoveError("verify", f"The copy of {src.name} did not match the original.", f"sha256 {got} != {want}")
        os.replace(part, dst)
    except OSError as e:
        _quiet_remove(part)
        if _disk_full(e):
            raise MoveError("disk_full", "The drive ran out of space.", f"{errno.errorcode.get(e.errno, e.errno)}: {e.strerror}, {e.filename or dst}") from e
        raise MoveError("io", f"Strata could not copy {src.name}.", f"{type(e).__name__}: {e}") from e
    except BaseException:
        _quiet_remove(part)
        raise


def _quiet_remove(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _read_state(new: Path) -> dict:
    try:
        return json.loads((new / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(new: Path, **fields) -> None:
    try:
        tmp = new / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(fields, indent=1), encoding="utf-8")
        os.replace(tmp, new / STATE_FILE)
    except OSError:
        pass


def _save_state(old: Path, new: Path, **fields) -> None:
    """The state goes in the new folder and, as a pointer, in the old one: the old folder stays the data folder until the move is done,
    so that is where the next start looks."""
    _write_state(new, **fields)
    _write_state(old, **fields)


def _clear_state(*folders: Path) -> None:
    for f in folders:
        _quiet_remove(Path(f) / STATE_FILE)


def _same_path(a, b) -> bool:
    try:
        return _same(Path(a), Path(b))
    except (OSError, ValueError, TypeError):
        return False


def interrupted(data: Path) -> dict | None:
    """The unfinished move recorded in the data folder (``models-move.json``), for 'A move was interrupted. Finish it or leave it'.

    A move that is running right now (its process is alive) is not interrupted.  A stopped or failed move that left files in both folders is.
    Returns ``{state, from, to, bytes}`` or None.
    """
    data = Path(data)
    st = _read_state(data)
    if st.get("state") not in ("running", "stopped", "failed") or not st.get("from") or not st.get("to"):
        return None
    if not _same_path(st["from"], data):                       # a pointer left in some other folder is not ours
        return None
    if not Path(st["from"]).is_dir():
        return None
    pid, token = st.get("pid"), st.get("token")
    if st["state"] == "running" and isinstance(pid, int) and token and pid != os.getpid():
        try:
            from strata_control import procs
            if procs.start_token(pid) == token:
                return None
        except Exception:
            pass
    return st


def repoint_config(cfg_file: Path, old: Path, new: Path) -> bool:
    """A run config whose model files moved from ``old`` to ``new`` names them there (each path only when its file is now there and no longer
    at the old place). Same rule as setup.py repoint_config. Returns whether the file changed."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            o = str(old)
            nv, no = os.path.normcase(v), os.path.normcase(o)
            if nv.startswith(no + os.sep) or nv.startswith(no + "/"):
                rest = v[len(o):]
                if re.split(r"[\\/]", rest.lstrip("\\/"))[0].lower() in ITEMS:
                    n = str(new) + rest
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    fixed = fix(cfg)
    if fixed == cfg:
        return False
    tmp = cfg_file.with_name(cfg_file.name + ".move-tmp")
    tmp.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, cfg_file)
    return True


def config_files(install: Path | None) -> list[Path]:
    """The ``strata-<model>.json`` run configs of an install folder."""
    if not install or not Path(install).is_dir():
        return []
    return [p for p in Path(install).glob("strata-*.json") if not p.name.endswith(".shared-settings.json")]


def _prune(folder: Path, stop_at: Path) -> None:
    """Remove empty folders from ``folder`` upward, never ``stop_at`` itself."""
    p = folder
    while p != stop_at and _inside(p, stop_at):
        try:
            p.rmdir()
        except OSError:
            return
        p = p.parent


def move(old: Path, new: Path, *, on_progress: Callable[[dict], None] | None = None, cancelled: Callable[[], bool] | None = None,
         configs: list[Path] | None = None, now: Callable[[], float] = time.time) -> dict:
    """Move the model files from ``old`` to ``new``. Returns the summary; raises MoveError (nothing lost) or Stopped.

    ``summary = {done: bool, stopped: bool, moved: [group], remaining: [group], bytes_moved, total_bytes, files_moved, same_drive}``.
    """
    old, new = Path(old), Path(new)
    emit = on_progress or (lambda _e: None)
    stop = cancelled or (lambda: False)
    p = plan(old, new, write_test=True)
    if p["issue"]:
        raise MoveError(p["issue"]["code"], p["issue"]["message"], p["issue"]["details"])
    entries = inventory(old)
    total = sum(e.size for e in entries)
    if not p["choices"]["move"]["possible"] and entries:
        raise MoveError("disk_full", p["choices"]["move"]["reason"])
    # unfinished copies of an earlier run are never trusted
    for top in _item_dirs(new):
        for root, _d, files in os.walk(top):
            for name in files:
                if name.endswith(".part"):
                    _quiet_remove(Path(root) / name)
    same = p["same_drive"]
    todo = {e.group: [] for e in entries}
    for e in entries:
        todo[e.group].append(e)
    left = {g: len(v) for g, v in todo.items()}
    moved_groups: list[str] = []
    before = _read_state(old)
    if not (_same_path(before.get("from", ""), old) and _same_path(before.get("to", ""), new)):
        before = {}
    try:
        from strata_control import procs
        token = procs.start_token(os.getpid())
    except Exception:
        token = None
    state = {"state": "running", "from": str(old), "to": str(new), "started": now(), "bytes": total, "pid": os.getpid(), "token": token,
             "files": sorted({e.rel for e in entries} | set(before.get("files") or []))}
    _save_state(old, new, **state)
    done_bytes = 0
    files_moved = 0
    last_emit = [0.0]
    started = now()
    seen_inodes: dict[tuple, Path] = {}
    base_event = lambda **kw: {"event": "progress", "total": total, "same_drive": same, **kw}

    def progress(current: str, group: str, index: int, force: bool = False):
        t = now()
        if not force and t - last_emit[0] < 0.25:
            return
        last_emit[0] = t
        rate = done_bytes / max(t - started, 0.001)
        emit(base_event(done=done_bytes, file=current, group=group, index=index, count=len(entries),
                        seconds_left=(int((total - done_bytes) / rate) if rate > 0 and not same else 0)))

    def repoint(only_whole: bool = True):
        for cfg in (configs or []):
            try:
                repoint_config(cfg, old, new)
            except OSError:
                pass

    try:
        for index, e in enumerate(entries, 1):
            if stop():
                raise Stopped()
            src, dst = old / e.rel, new / e.rel
            progress(e.rel, e.group, index, force=True)
            if not src.exists():
                done_bytes += e.size
                left[e.group] -= 1
            else:
                if same:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if dst.exists():
                            os.remove(dst)
                        os.replace(src, dst)
                    except OSError as err:
                        raise MoveError("io", f"Strata could not move {src.name}.", f"{type(err).__name__}: {err}") from err
                    done_bytes += e.size
                else:
                    ino = None
                    try:
                        st = src.stat()
                        ino = (st.st_dev, st.st_ino) if st.st_nlink > 1 and st.st_ino else None
                    except OSError:
                        pass
                    if ino and ino in seen_inodes and seen_inodes[ino].exists():
                        # the same file under two names (a shared pack): one copy, the second name links to it
                        try:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            if dst.exists():
                                os.remove(dst)
                            os.link(seen_inodes[ino], dst)
                        except OSError:
                            _copy_verified(src, dst, e.size, stop, lambda n: None)
                        done_bytes += e.size
                    else:
                        resumed = False
                        if dst.exists() and dst.stat().st_size == e.size:
                            # a crash between the rename and the delete: the copy may be whole; prove it before the original goes
                            resumed = _sha256_of(dst, stop) == _sha256_of(src, stop)
                        if resumed:
                            done_bytes += e.size
                        else:
                            def tick(n, group=e.group, rel=e.rel, idx=index):
                                nonlocal done_bytes
                                done_bytes += n
                                progress(rel, group, idx)
                            before = done_bytes
                            try:
                                _copy_verified(src, dst, e.size, stop, tick)
                            except BaseException:
                                done_bytes = before
                                raise
                    if ino:
                        seen_inodes[ino] = dst
                    try:
                        os.remove(src)
                    except OSError as err:
                        raise MoveError("io", f"Strata copied {src.name} but could not remove the old copy.", f"{type(err).__name__}: {err}") from err
                _prune(src.parent, old)
                files_moved += 1
                left[e.group] -= 1
            if left[e.group] == 0:
                moved_groups.append(e.group)
                repoint()
        repoint()
        state.update(state="done", finished=now())
        _clear_state(old, new)
        return {"done": True, "stopped": False, "moved": moved_groups, "remaining": [], "bytes_moved": done_bytes, "total_bytes": total,
                "files_moved": files_moved, "same_drive": same}
    except Stopped:
        repoint()
        state.update(state="stopped")
        if files_moved:
            _save_state(old, new, **state)
        else:
            _clear_state(old, new)
        remaining = [g for g, n in left.items() if n > 0]
        return {"done": False, "stopped": True, "moved": moved_groups, "remaining": remaining, "bytes_moved": done_bytes, "total_bytes": total,
                "files_moved": files_moved, "same_drive": same}
    except MoveError as err:
        repoint()
        state.update(state="failed", error=err.code)
        if files_moved:
            _save_state(old, new, **state)
        else:
            _clear_state(old, new)
        err.summary = {"moved": moved_groups, "remaining": [g for g, n in left.items() if n > 0], "bytes_moved": done_bytes, "total_bytes": total,
                       "files_moved": files_moved}
        raise


def leave(old: Path, new: Path, *, configs: list[Path] | None = None) -> dict:
    """'Leave it': give up an unfinished move and keep the old folder as the complete one.

    Only the files the move itself listed are touched.  A copy in the new folder whose original is still in the old one is removed; a file
    that exists only in the new folder (its original went after a checked copy) is put back in the old folder, so nothing is lost.
    ``.part`` files are removed and the run configs point at the old folder again.  Returns ``{removed, restored, bytes_removed}``.
    """
    old, new = Path(old), Path(new)
    st = _read_state(old) or _read_state(new)
    rels = [r for r in (st.get("files") or []) if isinstance(r, str) and r and ".." not in Path(r).parts and not Path(r).is_absolute()]
    removed = restored = bytes_removed = 0
    for rel in rels:
        src, dst = old / rel, new / rel
        _quiet_remove(Path(str(dst) + ".part"))
        if not dst.exists():
            continue
        try:
            size = dst.stat().st_size
            if src.exists():
                os.remove(dst)
                removed += 1
                bytes_removed += size
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                if same_drive(old, new):
                    os.replace(dst, src)
                else:
                    _copy_verified(dst, src, size, lambda: False, lambda n: None)
                    os.remove(dst)
                restored += 1
        except OSError as e:
            raise MoveError("io", f"Strata could not clean up {dst.name}.", f"{type(e).__name__}: {e}") from e
        _prune(dst.parent, new)
    for cfg in (configs or []):
        try:
            repoint_config(cfg, new, old)
        except OSError:
            pass
    _clear_state(old, new)
    return {"removed": removed, "restored": restored, "bytes_removed": bytes_removed}
