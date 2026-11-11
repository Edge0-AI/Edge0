"""Opt-in: give the model files' page cache back when the server stops (STRATA_DROP_CACHE_ON_EXIT=1).

Linux keeps a stopped model's GGUF/pack files in the page cache (MemAvailable counts it, `free` shows it as used), which
helps a restart of the SAME model.  On a host that then starts a different, large model (a dual-socket server whose second
NUMA node holds the old model's cache) some owners prefer the pages back at once.  Off by default.  Only this server's own
model files are advised, and only after the engine process has ended (its mappings are gone, so the pages can go).
"""
from __future__ import annotations

import os
from pathlib import Path


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return os.name != "nt" and env.get("STRATA_DROP_CACHE_ON_EXIT", "").strip().lower() in ("1", "true", "yes", "on")


def model_files(paths) -> list[str]:
    """The regular files named by `paths` (a file, or a directory walked recursively); missing ones are skipped."""
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        q = Path(p)
        try:
            if q.is_file():
                out.append(str(q))
            elif q.is_dir():
                out.extend(str(f) for f in sorted(q.rglob("*")) if f.is_file())
        except OSError:
            continue
    return out


def drop(paths, advise=None) -> int:
    """posix_fadvise(DONTNEED) on every file of `paths`; returns how many files were advised.  Never raises."""
    advise = advise or getattr(os, "posix_fadvise", None)
    if advise is None:
        return 0
    n = 0
    for f in model_files(paths):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            advise(fd, 0, 0, getattr(os, "POSIX_FADV_DONTNEED", 4))
            n += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return n


def drop_after_exit(engine, env=None, advise=None) -> int:
    """Called once the engine has been closed.  0 when not opted in or the engine still runs."""
    if not enabled(env):
        return 0
    proc = getattr(engine, "proc", None)
    if proc is not None and proc.poll() is None:
        return 0
    spawn = getattr(engine, "spawn", None)
    args = list(spawn[1]) if spawn else []
    wanted = {"--native", "--pack", "--mmproj", "--model"}
    paths = [v for k, v in zip(args, args[1:]) if k in wanted]
    n = drop(paths or [getattr(engine, "model_path", None)], advise)
    print(f"[strata] STRATA_DROP_CACHE_ON_EXIT: released the page cache of {n} model files", flush=True)
    return n
