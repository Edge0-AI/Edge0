"""The hipBLASLt tuning tables that ship inside the AMD engine packages, and the one rule that decides whether a start uses one.

A table (tools/hip/<arch>-hipblaslt-<version>.txt) lists, per dense GEMM shape of the prompt path, the hipBLASLt solution that was
fastest on that GPU architecture. Its solution ids are valid for one architecture and one hipBLASLt version only; the engine checks both
(src/prefill/gemm.cu) and falls back to plain hipBLAS for anything else, so a wrong table can never change a result, it only wastes the
speed-up. The table picks a GEMM algorithm: it does not change what the model is asked to compute.

Package side (packaging/stage_tuning.py, tools/hip/package_windows.py): ``tuning/<arch>-hipblaslt-<version>.txt`` inside the engine package,
one per architecture of the package that has a table for the hipBLASLt version the package bundles, and ``hipblaslt_version`` in its BUILD.json.
The files are listed (and hashed, and signed) in engine-component.json like every other file of the package.

Start side (serve/server.py with_component_engine, setup.py in the app): ``resolve`` reads the package, the GPUs in this PC and the switch,
and returns the table to use or the reason there is none. Nobody sets STRATA_HIPBLASLT_TUNING by hand:

* ``STRATA_HIPBLASLT_TABLE`` (or ``"hipblaslt_table"`` in the run config): ``off`` / false never uses the packaged table, ``on`` / true
  uses it when it matches, ``auto`` (unset) follows ``AUTO_USES_TABLE`` below;
* a ``STRATA_HIPBLASLT_TUNING`` the person set themselves (environment or run config) always wins and is not touched;
* a table for another hipBLASLt version than the package bundles is not used, and the log says which versions it has.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

SWITCH = "STRATA_HIPBLASLT_TABLE"            # on | auto | off
#: What "auto" (the switch unset) does. A table changes which hipBLASLt solution runs the prompt's dense GEMMs, so their floating-point sums
#: round differently: on an R9700 (gfx1201, Q2_0) the answers with a table are NOT bit-identical to plain hipBLAS, see phaseA-tests/notes-021-hip.md.
#: A default that changes bits needs the owner's decision, so the packages carry their tables and the switch ("on") uses them, but "auto" is
#: off until that decision. Flipping this one constant turns the tables on for every package that carries one.
AUTO_USES_TABLE = False
#: The owner's decision (2026-10-19): "auto" uses the table on these architectures only. gfx1151 (Strix Halo): a community A/B measured
#: 251 -> 529 prompt tok/s (2.11x) with the table; on gfx1201 the gain was ~1% and decode lost ~2% (notes-021-hip.md), so it stays opt-in there.
AUTO_ARCHS = ("gfx1151",)
ENGINE_VAR = "STRATA_HIPBLASLT_TUNING"       # what the engine reads: the path of one table
TABLE_DIR = "tuning"                         # inside an engine package
HEADER = "STRATA_HIPBLASLT_TUNING_V1"
OFF_WORDS = ("off", "0", "no", "false", "none", "disable", "disabled")
ON_WORDS = ("on", "1", "yes", "true", "force", "enable", "enabled")
_NAME = re.compile(r"^(gfx[0-9a-f]+)-hipblaslt-(\d+)\.txt$")


def table_name(arch: str, version: int) -> str:
    return f"{arch}-hipblaslt-{version}.txt"


def version_number(major: int, minor: int, patch: int) -> int:
    """hipblasLtGetVersion's number: 1.5.0 -> 100500, 1.2.2 -> 100202."""
    return major * 100000 + minor * 100 + patch


def version_from_header(rocm_root) -> int | None:
    """The hipBLASLt version of a ROCm tree, from include/hipblaslt/hipblaslt-version.h (what a package build has at hand); None when absent."""
    try:
        text = (Path(rocm_root) / "include" / "hipblaslt" / "hipblaslt-version.h").read_text(encoding="utf-8")
        parts = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", text).group(1)) for k in ("MAJOR", "MINOR", "PATCH")]
    except (OSError, AttributeError, ValueError):
        return None
    return version_number(*parts)


def header_ok(path: Path, arch: str, version: int) -> bool:
    """The file's own second line must name the same architecture and version as its file name (the engine checks it again)."""
    try:
        with open(path, encoding="utf-8") as f:
            head = [f.readline().strip(), f.readline().strip()]
    except OSError:
        return False
    return f"{HEADER} {arch} {version}" in head


def tables_in(folder) -> dict[tuple[str, int], Path]:
    """{(arch, version): path} for every table file in a folder."""
    found = {}
    try:
        entries = sorted(Path(folder).iterdir())
    except OSError:
        return found
    for path in entries:
        m = _NAME.match(path.name)
        if m and path.is_file():
            found[(m.group(1), int(m.group(2)))] = path
    return found


# ------------------------------------------------------------------------------------------------- package side

def stage_tables(stage, archs, version: int | None, source) -> list[str]:
    """Copy the tables for ``archs`` and hipBLASLt ``version`` from the repo folder ``source`` (tools/hip) into ``stage/tuning/``.
    Returns the file names copied. An architecture without a table for that version is left out (plain hipBLAS stays for it); a
    table whose header disagrees with its name is an error, because the engine would refuse it on every start."""
    import shutil
    stage, copied = Path(stage), []
    if version is None:
        return copied
    have = tables_in(source)
    for arch in archs:
        path = have.get((arch, version))
        if path is None:
            continue
        if not header_ok(path, arch, version):
            raise SystemExit(f"{path}: the second line does not say '{HEADER} {arch} {version}'")
        (stage / TABLE_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, stage / TABLE_DIR / path.name)
        copied.append(path.name)
    return copied


def missing_report(archs, version: int | None, source) -> dict[str, list[int]]:
    """For each architecture without a table for ``version``: the versions the repo does have for it (so a build can say it)."""
    have = tables_in(source)
    return {a: sorted(v for (arch, v) in have if arch == a) for a in archs if version is None or (a, version) not in have}


def check_stage(stage, archs) -> list[str]:
    """What is wrong with the tuning of a staged HIP package (empty list: fine): BUILD.json must say which hipBLASLt it carries, and every
    table in tuning/ must be named <arch>-hipblaslt-<that version>.txt for an architecture of the package with a header that says the same."""
    stage, problems = Path(stage), []
    version = package_version(stage)
    if version is None:
        problems.append("BUILD.json has no hipblaslt_version (packaging/stage_tuning.py writes it): a start could not match a tuning table to this hipBLASLt")
    folder = stage / TABLE_DIR
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            m = _NAME.match(path.name)
            if not m:
                problems.append(f"tuning/{path.name} is not named <arch>-hipblaslt-<version>.txt")
            elif m.group(1) not in archs:
                problems.append(f"tuning/{path.name} is for {m.group(1)}, which this package has no code for")
            elif version is not None and int(m.group(2)) != version:
                problems.append(f"tuning/{path.name} is for hipBLASLt {m.group(2)}, this package carries {version}")
            elif not header_ok(path, m.group(1), int(m.group(2))):
                problems.append(f"tuning/{path.name}: the header line does not say '{HEADER} {m.group(1)} {m.group(2)}'")
    return problems


# ------------------------------------------------------------------------------------------------- start side

@dataclass
class Decision:
    path: Path | None            # the table to hand the engine, or None
    message: str                 # one plain line for the log ("" when there is nothing to say)
    level: str = "info"          # info | warn


def mode(environ=None, cfg: dict | None = None) -> str:
    """"on", "off" or "auto": the environment switch first, then the run config's "hipblaslt_table" (true / false)."""
    environ = os.environ if environ is None else environ
    word = str(environ.get(SWITCH, "")).strip().lower()
    if word in OFF_WORDS:
        return "off"
    if word in ON_WORDS:
        return "on"
    if word in ("", "auto"):
        value = (cfg or {}).get("hipblaslt_table")
        return "off" if value is False else "on" if value is True else "auto"
    return "auto"


def switched_off(environ=None, cfg: dict | None = None) -> bool:
    return mode(environ, cfg) == "off"


def package_version(folder) -> int | None:
    """hipblaslt_version in the package's BUILD.json (an int, 100500), else None."""
    try:
        value = json.loads((Path(folder) / "BUILD.json").read_text(encoding="utf-8-sig")).get("hipblaslt_version")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def user_choice(environ=None, cfg: dict | None = None) -> str | None:
    """A STRATA_HIPBLASLT_TUNING the person set (environment first, then the run config's env): theirs to keep, even if the file is gone."""
    environ = os.environ if environ is None else environ
    for source in (environ, (cfg or {}).get("env") or {}):
        value = source.get(ENGINE_VAR)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve(folder, archs_present, *, environ=None, cfg: dict | None = None) -> Decision:
    """The table for an installed AMD engine package ``folder`` on a PC whose GPUs are ``archs_present`` (in the order the engine sees them).
    Never raises; an empty message means there is nothing worth a line (not an AMD package, or the person chose already)."""
    folder = Path(folder)
    if switched_off(environ, cfg):
        return Decision(None, f"hipBLASLt tuning table: off ({SWITCH}=off); the prompt's dense matrix products use plain hipBLAS")
    chosen = user_choice(environ, cfg)
    if chosen:
        return Decision(None, f"hipBLASLt tuning table: {chosen} (set by you; the engine checks it against this GPU and hipBLASLt)")
    tuning = folder / TABLE_DIR
    if not tuning.is_dir():
        return Decision(None, "")                      # a package without tables (CUDA, Metal, an older HIP package): nothing to say
    version = package_version(folder)
    archs = [a for a in archs_present if isinstance(a, str)]
    if version is None:
        return Decision(None, "hipBLASLt tuning table: not used, this engine package does not say which hipBLASLt it carries "
                              "(BUILD.json hipblaslt_version); the prompt's dense matrix products use plain hipBLAS", "warn")
    if not archs:
        return Decision(None, "hipBLASLt tuning table: not used, this PC's AMD GPU architecture could not be read; plain hipBLAS", "warn")
    tables = tables_in(tuning)
    for arch in dict.fromkeys(archs):
        path = tables.get((arch, version))
        if path is None:
            continue
        if not header_ok(path, arch, version):
            return Decision(None, f"hipBLASLt tuning table: {path.name} is not used, its header does not match its name; plain hipBLAS", "warn")
        if mode(environ, cfg) == "auto" and not (AUTO_USES_TABLE or arch in AUTO_ARCHS):
            return Decision(None, f"hipBLASLt tuning table: {path.name} is in this package and not used by default (faster prompts, but answers can differ "
                                  f"in the last digits); {SWITCH}=on uses it")
        return Decision(path, f"hipBLASLt tuning table: {path.name} (faster prompts; {SWITCH}=off turns it off)")
    other = sorted({v for (a, v) in tables if a in archs})
    if other:
        return Decision(None, f"hipBLASLt tuning table: not used, this engine carries hipBLASLt {version} and the tables for {', '.join(dict.fromkeys(archs))} "
                              f"are for {', '.join(str(v) for v in other)} (solution ids are valid for one version only); plain hipBLAS", "warn")
    return Decision(None, f"hipBLASLt tuning table: none for {', '.join(dict.fromkeys(archs))} with hipBLASLt {version}; plain hipBLAS")


def apply(cfg: dict, folder, archs_present=None, *, environ=None, say=print) -> dict:
    """``cfg`` for a start from engine package ``folder``: a copy with ``env[STRATA_HIPBLASLT_TUNING]`` set when a table applies. One log line
    through ``say`` for an AMD package (nothing for other backends)."""
    if cfg.get("backend") not in (None, "hip"):
        return cfg
    folder = Path(folder)
    if not (folder / TABLE_DIR).is_dir() and package_version(folder) is None:
        return cfg                                      # not a package with tables or a hipBLASLt version: nothing to decide
    if archs_present is None:
        archs_present = present_archs(cfg)
    decision = resolve(folder, archs_present, environ=environ, cfg=cfg)
    if decision.message:
        say(("[strata] " if decision.level == "info" else "[strata] note: ") + decision.message)
    if decision.path is None:
        return cfg
    out = dict(cfg)
    out["env"] = {**(cfg.get("env") or {}), ENGINE_VAR: str(decision.path)}
    return out


def present_archs(cfg: dict | None = None) -> list[str]:
    """The AMD architectures of the GPUs the engine will use, as ``strata_control.engine_component.hip_archs_present`` lists them; the run
    config's "gpu" picks one card of several when the list is per card (Linux)."""
    try:
        from strata_control import engine_component as ec
        archs = ec.hip_archs_present()
    except Exception:
        return []
    gpu = (cfg or {}).get("gpu")
    picked = gpu if isinstance(gpu, list) else [gpu] if isinstance(gpu, int) and not isinstance(gpu, bool) else []
    chosen = [archs[i] for i in picked if 0 <= i < len(archs)] if os.name != "nt" else []
    return chosen or archs
