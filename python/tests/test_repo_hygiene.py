"""Repository-hygiene guards for a clean open-source release.

These tests import nothing from ``edge0`` (and nothing from ``mlx``), so
they can run on any platform, independent of whatever MLX wheels happen
to be available on the CI runner's platform. They enforce the "no
dependency on the author's machine" rule:

* no personal / hardcoded absolute paths in shipped code, docs or scripts;
* the MLX backend boundary is respected (``import mlx`` only inside
  ``edge0/backends/mlx/``);
* no obvious secrets (API keys / private keys) in the tree;
* README/NOTICE references resolve.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

# Repo root (this file lives at <root>/python/tests/): the hygiene guards
# scan the whole multi-platform repo, not just the Python subproject.
ROOT = pathlib.Path(__file__).resolve().parents[2]

_EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", ".agents",
                 ".codex", "vendor", "wt", ".edge0", "models", "node_modules",
                 ".gradle", "build-dl", "target", "dist", "build"}
_EXCLUDE_PARTS = {"egg-info"}
_EXCLUDE_SUFFIXES = {".pyc", ".so", ".dylib", ".bin", ".safetensors", ".npz",
                     ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf"}

# NOTE: patterns are assembled with + so this file never contains the
# literal string it is looking for.
_USR = "Users"
_NAS = "nas"
_HOME = "home"
_ROOT = "root"
_LOCAL_PATH_PATTERNS = [
    re.compile(r"/" + _USR + r"/[\w.-]+/"),
    re.compile(r"/" + _NAS + r"/"),
    re.compile(r"/" + _HOME + r"/[\w.-]+/"),
    re.compile(r"/" + _ROOT + r"(?:/|$)"),
    re.compile(r"C:" + re.escape("\\") + _USR + re.escape("\\")),
    re.compile(r"HPC-\d"),
    re.compile(r"ali-pod\d"),
]

_MLX_IMPORT = re.compile(r"^\s*(?:from\s+mlx(?:\s|\.)|import\s+mlx(?:\s|\.))",
                         re.MULTILINE)

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
]


def _git_ls_files() -> list[str] | None:
    """Repo-relative paths of the files that are (or would be) committed:
    tracked files plus untracked-but-not-gitignored ones.  Materialized
    build areas (``vendor/``, ``wt/``, ``models/``, ``.venv/``) are
    gitignored, so this naturally excludes them — the hygiene guards check
    *shipped* code, not downloaded third-party dependencies.  Returns None
    if git is unavailable, so the caller can fall back to a directory walk.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard",
             "-z"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def _iter_text_files():
    shipped = _git_ls_files()
    if shipped is not None:
        for rel in shipped:
            path = ROOT / rel
            if path.suffix in _EXCLUDE_SUFFIXES or not path.is_file():
                continue
            yield path
        return
    # Fallback (no git): walk the tree, skipping vcs / build / vendor dirs.
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        if any(part in _EXCLUDE_PARTS for part in rel.parts):
            continue
        if path.suffix in _EXCLUDE_SUFFIXES or path.is_dir():
            continue
        yield path


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def test_no_hardcoded_local_paths():
    hits = []
    for path in _iter_text_files():
        text = _read(path)
        if not text:
            continue
        for pat in _LOCAL_PATH_PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(f"{path.relative_to(ROOT)}: {m.group(0)!r}")
    assert not hits, "hardcoded local paths found:\n" + "\n".join(hits)


def test_mlx_imports_stay_inside_the_backend():
    bad = []
    for path in (ROOT / "python" / "src" / "edge0").rglob("*.py"):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if _MLX_IMPORT.search(text) and "backends/mlx" not in rel.as_posix():
            bad.append(str(rel))
    assert not bad, (
        "MLX imports must live under edge0/backends/mlx/ only:\n"
        + "\n".join(bad))


def test_no_secrets():
    hits = []
    for path in _iter_text_files():
        text = _read(path)
        if not text:
            continue
        for pat in _SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                hits.append(f"{path.relative_to(ROOT)}: {m.group(0)!r}")
    assert not hits, "possible secrets found:\n" + "\n".join(hits)


def test_notice_link_resolves():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\[NOTICE\]\(([^)]+)\)", readme)
    assert m, "README should link the NOTICE file"
    target = ROOT / m.group(1)
    assert target.is_file(), f"README NOTICE link target missing: {target}"
