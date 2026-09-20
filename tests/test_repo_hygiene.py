"""Repository-hygiene guards for a clean open-source release.

These tests import nothing from ``edge0`` (and nothing from ``mlx``), so
they can run on any platform — including the Linux CI job where MLX has no
wheels.  They enforce the "no dependency on the author's machine" rule:

* no personal / hardcoded absolute paths in shipped code, docs or scripts;
* the MLX backend boundary is respected (``import mlx`` only inside
  ``edge0/backends/mlx/``);
* no obvious secrets (API keys / private keys) in the tree;
* README/NOTICE references resolve.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

_EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", ".agents",
                 ".codex"}
_EXCLUDE_PARTS = {"egg-info"}
_EXCLUDE_SUFFIXES = {".pyc", ".so", ".dylib", ".bin", ".safetensors", ".npz",
                     ".png", ".jpg", ".jpeg", ".gif", ".ico"}

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

# ```` after ``mlx(?:_lm)?`` matches ``import mlx``, ``import mlx.core``,
# ``from mlx_lm.models...`` and ``from mlx_lm import ...`` alike, while a
# hypothetical ``mlxfoo`` package still does not match.  The previous
# pattern required a space or dot right after ``mlx`` and therefore let
# ``mlx_lm`` imports through.
_MLX_IMPORT = re.compile(r"^\s*(?:from|import)\s+mlx(?:_lm)?", re.MULTILINE)

# Files outside ``backends/mlx`` that currently import MLX packages.  Each
# entry is a known boundary violation tracked for removal; delete the entry
# once the import is gone so the guard starts enforcing it again.
_MLX_IMPORT_KNOWN_VIOLATIONS = {
    # ``_gather_sort`` / ``_scatter_unsort`` come from
    # ``mlx_lm.models.switch_layers``; they should move into the framework
    # so the streaming layer is importable on other backends.
    "src/edge0/streaming/layer.py",
}

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
]


def _iter_text_files():
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
    for path in (ROOT / "src" / "edge0").rglob("*.py"):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        posix = rel.as_posix()
        if "backends/mlx" in posix or posix in _MLX_IMPORT_KNOWN_VIOLATIONS:
            continue
        if _MLX_IMPORT.search(text):
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
