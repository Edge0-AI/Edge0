"""Bounded, read-only discovery of setup-written local model run configs.

The setup installer writes strata-*.json beside setup.py. A configured model is
not automatically a compatible or loadable model: component presence and native
engine compatibility are reported separately.

Finding an existing install: a terminal archive is a new folder with no run configs in it. The places where Strata
installs keep run configs (the Terminal's own folder, the folders earlier installs remembered in the user's settings file,
and the desktop app's install folder, which the Terminal shares) are searched read-only by strata_control/discovery.py,
the same function the launch screens use; the first place that has any is the main one. Nothing is copied or moved:
the configs name their engine and model files by absolute path, so the old install and its data folder keep
serving the models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

from strata_control import discovery
from strata_control.doctor import inspect as inspect_components
from strata_control.gguf_preflight import inspect_metadata, real_model_file
from strata_control.profiles import ProfileError


ROOT = Path(__file__).resolve().parents[1]
# a dot is allowed inside the name (DeepSeek-V4.1-Flash: "deepseek-v4.1-flash-ist-das-2.00bpw"), never two in a row and never first
MODEL_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]|\.(?!\.)){0,127}\Z")
MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIGS = 128
MAX_IMPORT_SHARDS = 128
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z", re.I)
SUPPORTED_SELECTIONS = frozenset({"qwen/Q2_0", "qwen/IQ2_XS", "qwen/IQ3_XXS", "qwen/IQ3_S",
                                   "swift/IQ2_XS", "swift/IQ3_XXS", "swift/IQ3_S", "coder/IQ1_M",
                                   "unsloth/UD-IQ4_XS", "unsloth/UD-Q4_K_XL", "unsloth/UD-Q5_K_XL", "unsloth/UD-Q6_K_XL",
                                   "unsloth/Q8_0", "gyro/Gyro-S", "gyro/Gyro-M"})


def settings_path() -> Path:
    """The user's Strata settings file: the same place setup.py writes it (%APPDATA%/Strata/settings.json, or
    $XDG_CONFIG_HOME/strata/settings.json). The desktop app sets STRATA_SETTINGS_FILE to a file of its own (in its data folder), so it
    never reads an older command-line install's list of Strata folders."""
    return discovery.own_settings_file()


def remembered_installs() -> list[Path]:
    """Earlier Strata folders recorded in the settings file, newest first. Read-only; a missing or unreadable file
    is an empty list."""
    return discovery._installs_in(discovery._read_json(settings_path()))        # noqa: SLF001


def _has_configs(directory: Path) -> bool:
    try:
        return directory.is_dir() and bool(_paths(directory))
    except (ProfileError, OSError):
        return False


def resolve_install(override: str | None = None) -> tuple[Path, str]:
    """(the main install folder, how it was found): explicit (an argument or STRATA_INSTALL_DIR), here (this folder has run
    configs), remembered (an earlier install from the settings file), app (the desktop app's install folder, shared with the
    Terminal) or default (the app's folder, with no model in it yet)."""
    raw = override or os.environ.get("STRATA_INSTALL_DIR")
    if raw is not None:
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ProfileError("install directory must be an existing absolute directory", 5)
        return path.resolve(), "explicit"
    return discovery.find_install(here=ROOT)


def all_installs(override: str | None = None) -> list[tuple[Path, str]]:
    """Every folder with run configs, main one first: what `strata models list` shows and what the launch screens read."""
    directory, source = resolve_install(override)
    if source == "explicit":
        return [(directory, source)]
    found = [(p.path, p.source) for p in discovery.places(here=ROOT)]
    return found or [(directory, source)]


def install_dir(override: str | None = None) -> Path:
    return resolve_install(override)[0]


def _model_id(path: Path) -> str | None:
    if not path.name.startswith("strata-") or not path.name.endswith(".json") or \
            path.name.endswith(".shared-settings.json"):
        return None
    value = path.name[len("strata-"):-len(".json")]
    return value if MODEL_ID.fullmatch(value) else None


def _paths(directory: Path) -> list[tuple[str, Path]]:
    try:
        entries = [(name, path) for path in directory.iterdir() if (name := _model_id(path)) is not None]
    except OSError:
        raise ProfileError("cannot read installed model directory", 5) from None
    if len(entries) > MAX_CONFIGS:
        raise ProfileError("more than 128 model configs need inspection", 5)
    return sorted(entries, key=lambda item: item[0])


def _read(path: Path) -> tuple[bytes, dict]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProfileError("run config must not be a symlink", 5)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_CONFIG_BYTES:
                raise ProfileError("run config must be a regular file under 1 MiB", 5)
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ProfileError("run config exceeds 1 MiB", 5)
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("run config is unreadable or invalid JSON", 5) from None
    if not isinstance(value, dict) or not isinstance(value.get("exe"), str) or not value["exe"] or \
            not isinstance(value.get("tokenizer"), str) or not value["tokenizer"] or \
            not isinstance(value.get("args"), list) or \
            not all(isinstance(arg, str) for arg in value["args"]):
        raise ProfileError("run config lacks engine, tokenizer or string arguments", 5)
    return raw, value


def _short(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return "".join(char if char.isprintable() and char not in "\x1b\x7f" else " " for char in value)[:120]


def _arg(args: list[str], flag: str) -> str | None:
    try:
        index = args.index(flag)
        return args[index + 1] if index + 1 < len(args) else None
    except ValueError:
        return None


def _missing_parts(report: dict) -> list[dict]:
    """The required parts a model's run config names that are not there: {"part": the component, "file": its file or folder name,
    "status": "missing" | "invalid"}. The app names them so a person is not told only that "some files are missing" (#3478)."""
    found = []
    for item in report.get("components", []):
        if item.get("required") and item.get("status") in ("missing", "invalid"):
            where = item.get("path")
            found.append({"part": _short(item.get("name")), "file": _short(os.path.basename(str(where).rstrip("/\\"))) if where else None,
                          "status": item["status"]})
    return found[:12]


def _summary(model_id: str, path: Path, raw: bytes, cfg: dict, report: dict) -> dict:
    context = _arg(cfg["args"], "--max-context")
    context_tokens = int(context) if isinstance(context, str) and 1 <= len(context) <= 8 and \
        context.isascii() and context.isdecimal() and 1 <= int(context) <= 16_777_216 else None
    backend = cfg.get("backend")
    return {"id": model_id, "name": _short(cfg.get("model_name")) or model_id,
            "source_config": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "backend": _short(backend) if backend is not None else None,
            "context_tokens": context_tokens,
            "component_status": "present" if report["ok"] else "missing_or_invalid",
            "issue_codes": [item.get("code") for item in report["issues"]][:32],
            "missing": _missing_parts(report),
            "execution_compatibility": "unverified"}


def _inspect_one(model_id: str, path: Path) -> tuple[dict, dict]:
    raw, cfg = _read(path)
    try:
        report = inspect_components(path)
        after, _ = _read(path)
    except (OSError, ValueError):
        raise ProfileError("run config changed or became invalid during inspection", 6) from None
    if after != raw:
        raise ProfileError("run config changed during inspection; retry", 6)
    return _summary(model_id, path, raw, cfg, report), report


def list_installed(override: str | None = None) -> dict:
    directory, source = resolve_install(override)
    places = all_installs(override)
    models = []
    for place, place_source in places:
        try:
            entries = _paths(place) if place.is_dir() else []
        except ProfileError as exc:
            models.append({"id": place.name, "install_dir": str(place), "component_status": "invalid", "error": str(exc),
                           "execution_compatibility": "unverified"})
            continue
        for model_id, path in entries:
            if any(m.get("id") == model_id for m in models):
                continue                                   # the same model in a later place: the main one wins
            try:
                summary, _ = _inspect_one(model_id, path)
            except ProfileError as exc:
                summary = {"id": model_id, "source_config": str(path), "component_status": "invalid",
                           "error": str(exc), "execution_compatibility": "unverified"}
            summary["install_dir"] = str(place)
            summary["install_dir_source"] = place_source
            models.append(summary)
    return {"scope": "local_run_configs", "install_dir": str(directory), "install_dir_source": source,
            "install_dirs": [str(p) for p, _ in places],
            "models": models,
            "model_count": len(models), "discovery_limit": MAX_CONFIGS}


def inspect(model_id: str, override: str | None = None) -> dict:
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ProfileError("model ID must be a setup config name without a path")
    directory, path = None, None
    for place, _src in all_installs(override):
        try:
            choices = dict(_paths(place)) if place.is_dir() else {}
        except ProfileError:
            continue
        if model_id in choices:
            directory, path = place, choices[model_id]
            break
    if path is None:
        raise ProfileError(f"model {model_id!r} is not configured in this installation", 5)
    summary, report = _inspect_one(model_id, path)
    return {"scope": "local_run_config", "install_dir": str(directory), "model": summary,
            "components": report}


def _gguf_header(path: Path) -> dict:
    """The first bytes of one model file (a link to it is followed: Hugging Face cache snapshots and Ollama blobs are links)."""
    real = real_model_file(path)
    try:
        descriptor = os.open(real, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size < 8:
                raise ProfileError("GGUF shard is not a nonempty regular file", 5)
            header = stream.read(8)
    except OSError:
        raise ProfileError("GGUF shard is unreadable", 5) from None
    if header[:4] != b"GGUF":
        raise ProfileError("file does not begin with a GGUF header", 5)
    version = int.from_bytes(header[4:8], "little")
    if version not in (2, 3):
        raise ProfileError("GGUF header version is not recognized", 5)
    return {"name": path.name, "path": str(path), "bytes": details.st_size,
            "gguf_version": version}


def _candidate_selection(shards: list[Path]) -> dict | None:
    fingerprint_file = ROOT / "data" / "gguf_fingerprints.json"
    try:
        if fingerprint_file.stat().st_size > 128 * 1024:
            raise ProfileError("model fingerprint catalog exceeds 128 KiB", 5)
        catalog = json.loads(fingerprint_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("model fingerprint catalog is unavailable", 5) from None
    names = {item.name for item in shards}
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        expected = row.get("files") if isinstance(row, dict) else None
        if isinstance(expected, list) and expected and names == {Path(name).name for name in expected}:
            family, model = selection.split("/", 1)
            expected_experts = row.get("expert_bytes")
            return {"family": family, "model": model, "basis": "known_shard_names_only",
                    "expected_expert_types": sorted(expected_experts) if isinstance(expected_experts, dict) else [],
                    "expected_tensor_count": row.get("tensors") if type(row.get("tensors")) is int else None,
                    "setup_args_hint": ["--family", family, "--model", model,
                                        "--gguf-dir", str(shards[0].parent)]}
    return None


def _candidate_by_header(shards: list[Path], tensors: int, payload: int, experts: set[str]) -> dict | None:
    """A file set whose NAME setup does not know (renamed, or downloaded elsewhere) but whose header is exactly a known one: the same tensor
    count, the same payload size and the same expert formats as a row of the fingerprint table."""
    try:
        catalog = json.loads((ROOT / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        if not isinstance(row, dict) or row.get("tensors") != tensors or row.get("bytes") != payload:
            continue
        expected = row.get("expert_bytes")
        if not isinstance(expected, dict) or set(expected) != experts:
            continue
        family, model = selection.split("/", 1)
        return {"family": family, "model": model, "basis": "header_size_and_tensors",
                "expected_expert_types": sorted(expected), "expected_tensor_count": tensors,
                "setup_args_hint": ["--family", family, "--model", model, "--gguf-dir", str(shards[0].parent)]}
    return None


def _candidate_by_layout(shards: list[Path], tensors: int, payload: int, experts: set[str]) -> dict | None:
    """A qwen4exp file set that is no build Strata has tested (a fine-tune, a K-quant mix from another maker) but has the layout of one
    (#3651 #3777): the closest row by expert formats, then by size. Strata uses the files as that model and says so; it never refuses a
    working file for not being on the list (recommend, never force)."""
    try:
        catalog = json.loads((ROOT / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    best = None
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        if not isinstance(row, dict) or row.get("tensors") != tensors or not isinstance(row.get("expert_bytes"), dict):
            continue
        have = set(row["expert_bytes"])
        union = have | experts
        score = (len(have & experts) / len(union)) if union else 0.0
        key = (-score, abs(int(row.get("bytes") or 0) - payload), selection)
        if score > 0 and (best is None or key < best[0]):
            best = (key, selection, have)
    if best is None:
        return None
    family, model = best[1].split("/", 1)
    return {"family": family, "model": model, "basis": "same_layout_other_build",
            "expected_expert_types": sorted(best[2] | experts), "expected_tensor_count": tensors,
            "setup_args_hint": ["--family", family, "--model", model, "--gguf-dir", str(shards[0].parent)]}


def _already_set_up(shards: list[Path], candidate: dict | None) -> dict | None:
    """Is this model already installed (a run config of any known install names the file) and what was prepared beside it (pack, draft
    layer, picture reader)? Read-only; None when the file set is not one Strata knows."""
    if not shards or candidate is None:
        return None
    first = shards[0]
    cfg = discovery.run_config_for(first, discovery.places(here=ROOT))
    beside = discovery.prepared_beside(first)
    if cfg is None and not any(beside.get(k) for k in ("pack", "mtp", "mmproj")):
        return {"config": None, "data_dir": str(beside["data_dir"]) if beside.get("data_dir") else None, "pack": None, "mtp": None, "mmproj": None}
    return {"config": str(cfg) if cfg else None, **{k: (str(v) if v else None) for k, v in beside.items()}}


ISSUE_WORDS = {
    "not_on_this_mac": "Unsloth's UD-IQ1_M set doesn't run on a Mac yet; the UD-IQ1_S, 2.00 and 2.25 sets do",
    "not_on_this_spark": "Unsloth's 1-bit DeepSeek sets were not tested on a DGX Spark yet; the 2.00 and 2.25 sets were",
    "multiple_model_groups": "the folder holds more than one model",
    "incomplete_or_invalid_shard_set": "some parts of the model are missing",
    "unrecognized_or_unsupported_model_variant": "this is not a model size Strata knows",
    "unsupported_or_missing_architecture": "this model type is not supported by the Strata engine (add it as a Custom model API with llama-server instead)",
    "unsupported_or_incomplete_tokenizer_metadata": "the file has no usable tokenizer",
    "missing_or_unexpected_expert_quantization": "its quantization is not one Strata knows",
}
# the checker's own sentences about one file, in words
_REASON_WORDS = (
    ("not a nonempty regular file", "the file is empty or is not a file"),
    ("unreadable", "the file cannot be read (is it open in another program, or on a drive that is gone?)"),
    ("could not be read", "the file cannot be read (is it open in another program, or on a drive that is gone?)"),
    ("is missing", "the file is not there"),
    ("link to a file that is not there", "this is a shortcut to a file that is not there any more"),
    ("does not begin with a gguf header", "this is not a GGUF model file"),
    ("header version", "this GGUF is from a format Strata does not read"),
    ("requires gguf v3", "this GGUF is in an old format (version 2); Strata reads version 3"),
    ("metadata is truncated", "the file is cut short (a download or copy that did not finish)"),
    ("payload is shorter", "the file is shorter than its own header says (a download or copy that did not finish)"),
    ("exceeds the bounded inspection limit", "its header is larger than Strata reads"),
    ("too large to inspect", "its header is larger than Strata reads"),
    ("unsupported encoding or shape", "it stores weights in a format Strata does not read"),
    ("unsupported array type", "its header uses a type Strata does not read"),
    ("unsupported value type", "its header uses a type Strata does not read"),
    ("byte ranges overlap", "its header is damaged"),
    ("unsupported_architecture", "this model type is not supported by the Strata engine (add it as a Custom model API with llama-server instead)"),
    ("symlink", "this is a shortcut (symbolic link) that cannot be followed"),
)


def _not_here(model: str) -> str | None:
    from strata_control import dsv41_fit
    return dsv41_fit.not_here(model)


def plain_issue(code: str) -> str:
    """One problem from the GGUF preview, in words, with the file's name when it is about one file
    (`file.gguf: GGUF shard is not a nonempty regular file`)."""
    code = str(code)
    if code in ISSUE_WORDS:
        return ISSUE_WORDS[code]
    name, sep, reason = code.partition(": ")
    if not (sep and name.lower().endswith(".gguf")):
        name, reason = "", code
    low = reason.lower()
    words = next((w for key, w in _REASON_WORDS if key in low), reason.replace("_", " "))
    return f"{name}: {words}" if name else words


def missing_words(missing: list | None) -> str:
    """"x-00002-of-00002.gguf is missing" for the shards of a split model that are not in the folder (#3478); "" when none are known."""
    names = [str(n) for n in missing or []]
    if not names:
        return ""
    if len(names) == 1:
        return f"{names[0]} is missing"
    if len(names) == 2:
        return f"{names[0]} and {names[1]} are missing"
    return f"{names[0]}, {names[1]} and {len(names) - 2} more files are missing"


def problem_words(issues: list, limit: int = 3, missing: list | None = None) -> str:
    """The problems of a file set as ONE sentence part, each said once: the same reason for several files names the files together
    ("a.gguf and b.gguf: the file is cut short"), a reason that was already given is not given again, and a long list is cut with a count.
    A folder that holds several models does not also say "not a size Strata knows"."""
    codes = [str(x) for x in issues or []]
    named = missing_words(missing)
    if named:                                                # the generic "some parts are missing" is replaced by the files
        codes = [x for x in codes if x != "incomplete_or_invalid_shard_set"]
    if "multiple_model_groups" in codes:
        codes = [x for x in codes if x != "unrecognized_or_unsupported_model_variant"]
    by_reason: dict[str, list[str]] = {}
    for code in codes:
        words = plain_issue(code)
        name, sep, reason = words.partition(": ")
        if sep and name.lower().endswith(".gguf"):
            by_reason.setdefault(reason, []).append(name)
        else:
            by_reason.setdefault(words, [])
    parts = [named] if named else []
    for reason, names in by_reason.items():
        if not names:
            parts.append(reason)
        elif len(names) <= 2:
            parts.append(f"{' and '.join(names)}: {reason}")
        else:
            parts.append(f"{names[0]}, {names[1]} and {len(names) - 2} more files: {reason}")
    if len(parts) > limit:
        parts = parts[:limit] + [f"and {len(parts) - limit} more problem" + ("s" if len(parts) - limit > 1 else "")]
    return "; ".join(parts)


UNSUPPORTED_MODEL_TYPE = ("This model type isn't supported by the Strata engine. "
                          "Add it as a Custom model API (llama-server) instead.")
# The picture readers Strata ships a pinned copy of (setup downloads and checks one of these next to the model).
# (name, size in bytes) as setup.py pins them: setup would delete a file of that name with another size, so only an exact
# match is called "the official one".
PINNED_PROJECTORS = {"mmproj-qwen3.8-flash-next-bf16.gguf": 907543008, "mmproj-f16.gguf": 899283584}
OUR_PROJECTOR_TYPES = {"qwen3vl_merger"}


def _assess_projector(path: Path) -> dict:
    """What a picture-reader file (mmproj*.gguf) next to a model is, in words; never blocks the import."""
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    row = {"name": path.name, "status": "unreadable", "pinned": PINNED_PROJECTORS.get(path.name.lower()) == size}
    try:
        info = inspect_metadata(path)
    except ProfileError:
        row["plain"] = f"{path.name} could not be read as a picture reader, so it is not used."
        return row
    row["architecture"], row["projector_type"] = info["architecture"], info.get("projector_type")
    if info["architecture"] != "clip":
        row["status"] = "not_a_projector"
        row["plain"] = f"{path.name} is not a picture reader (mmproj) file, so it is not used."
    elif info.get("projector_type") not in OUR_PROJECTOR_TYPES:
        row["status"] = "unsupported"
        row["plain"] = (f"{path.name} reads pictures for a different kind of model. "
                        "Strata cannot use it; add that model as a Custom model API (llama-server) instead.")
    elif row["pinned"]:
        row["status"] = "ok"
        row["plain"] = f"{path.name} is the picture reader Strata uses for this model family; pictures will work."
    else:
        row["status"] = "other_name"
        row["plain"] = (f"{path.name} is a picture reader of the right kind but not the file Strata checks; "
                        "Strata uses the official one (mmproj-Qwen3.8-Flash-Next-BF16.gguf) and "
                        "downloads it when it is not in the folder.")
    return row


MAX_LISTED = 5000            # .gguf files looked at in one folder (the checker reads headers of the chosen set only)
MAX_TREE_DIRS = 300          # folders looked into when the picked folder holds no GGUF itself
TREE_DEPTH = 3
SKIP_DIRS = frozenset({"packs", "mtp", "engine", "node_modules", "windows", "program files", "program files (x86)",
                       "$recycle.bin", "system volume information", "appdata"})


def _list_gguf(folder: Path) -> tuple[list[Path], list[str], bool]:
    """(model files, picture-reader file names, cut) of one folder. A link to a file counts (it is followed when read); a name that
    only ends in .gguf but is a folder does not."""
    shards: list[Path] = []
    readers: list[str] = []
    cut = False
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                low = entry.name.lower()
                if not low.endswith(".gguf"):
                    continue
                try:
                    is_file = entry.is_file() or entry.is_symlink()
                except OSError:
                    is_file = False
                if not is_file:
                    continue
                if low.startswith("mmproj"):
                    readers.append(entry.name)
                    continue
                if len(shards) >= MAX_LISTED:
                    cut = True
                    break
                shards.append(Path(entry.path))
    except OSError:
        raise ProfileError("cannot inspect GGUF directory", 5) from None
    return shards, readers, cut


def _folders_with_models(root: Path) -> list[Path]:
    """The folders under `root` (a Strata data folder, a whole Hugging Face repository, a drive's model folder) that hold model files,
    breadth first, `TREE_DEPTH` levels deep. A folder with models is not searched further."""
    found: list[Path] = []
    seen: set[str] = set()
    queue = [(root, 0)]
    while queue and len(seen) < MAX_TREE_DIRS:
        folder, depth = queue.pop(0)
        try:
            key = os.path.normcase(os.path.realpath(folder))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        try:
            with os.scandir(folder) as scan:
                entries = sorted(scan, key=lambda e: e.name.lower())
        except OSError:
            continue
        if any(e.name.lower().endswith(".gguf") and not e.name.lower().startswith("mmproj") for e in entries):
            found.append(folder)
            continue
        if depth >= TREE_DEPTH:
            continue
        for e in entries:
            try:
                if e.is_dir() and not e.name.startswith(".") and e.name.lower() not in SKIP_DIRS:
                    queue.append((Path(e.path), depth + 1))
            except OSError:
                continue
    return found


def _split_groups(shards: list[Path]) -> dict:
    """Model files by set: (name before -0000i-of-0000N, N) -> files. A file with no shard number is a set of its own."""
    groups: dict = {}
    for item in shards:
        match = SHARD.fullmatch(item.name)
        key = (match.group(1), int(match.group(3))) if match else (item.name, 1)
        groups.setdefault(key, []).append(item)
    return groups


def _known(members: list[Path]) -> dict | None:
    try:
        return _candidate_selection(sorted(members, key=lambda item: item.name))
    except ProfileError:
        return None


def _holds(groups_by_place: list) -> list[dict]:
    """What a folder (or several) holds, set by set, for the message that asks the person to point at one."""
    rows = []
    for place, members_by_key in groups_by_place:
        for key in sorted(members_by_key):
            members = sorted(members_by_key[key], key=lambda item: item.name)
            sel = _known(members)
            rows.append({"file": members[0].name, "files": len(members), "folder": str(place),
                         "family": sel["family"] if sel else None, "model": sel["model"] if sel else None})
    return rows


def absolute(picked: str) -> str:
    """The picked path as an absolute path, links left alone (a snapshot file in the Hugging Face cache keeps its model name)."""
    return os.path.abspath(os.path.expanduser(str(picked)))


def import_target(picked: str, plan: dict) -> str:
    """What to hand on as the person's model files: the file when they picked a file (a folder of several sizes needs the name), else the
    folder the model files are really in (they may have picked the folder above it)."""
    picked = absolute(picked)
    try:
        if Path(picked).is_file():
            return picked
    except OSError:
        pass
    return plan.get("model_dir") or picked


def own_path_problem(picked: str) -> tuple[str, str] | None:
    """(what is wrong, what to do) when the path a person gave for their own model files is not there or holds no model file, in plain
    words; None when there is something to look at. Said before anything is installed or downloaded, with the path the person typed."""
    path = Path(absolute(picked))
    kind = "file" if path.name.lower().endswith(".gguf") else "folder"
    try:
        if not path.exists():
            if path.is_symlink():
                return (f"This {kind} is a shortcut to something that is not there any more: {path}.",
                        "Point it at the real folder, or at the .gguf file itself. Nothing was downloaded or changed.")
            return (f"This {kind} does not exist: {path}.",
                    "Check the spelling, and that the drive or folder is connected (in Docker or a virtual machine it has to be shared with it). "
                    "Nothing was downloaded or changed.")
        if path.is_dir():
            has_files = any(entry.name.lower().endswith(".gguf") for entry in os.scandir(path))
            if not has_files and not _folders_with_models(path):
                return (f"There are no .gguf model files in {path}, nor in the folders inside it.",
                        "Point it at the folder that holds the model's .gguf files (names like ...-00001-of-00002.gguf), or at one of the files. "
                        "Nothing was downloaded or changed.")
    except OSError:
        return None
    return None


def _dsv41_files(folder: Path) -> dict[str, int]:
    """{name: bytes} of the .gguf files in one folder (a link to a file is followed)."""
    found: dict[str, int] = {}
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if not entry.name.lower().endswith(".gguf"):
                    continue
                try:
                    found[entry.name] = entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return found


def _dsv41_plan(path: Path, details) -> dict | None:
    """The preview of a folder (or one file) of an official DeepSeek-V4.1-Flash set: the IST-DAS 1.75 / 2.00 / 2.25 shards, under their
    official names or the first release's, with or without the n-gram table (Engram). Recognised from the pinned names and sizes
    (strata_control/dsv41_sets.py); None when nothing here is a DeepSeek file, so every other preview is exactly what it was."""
    from strata_control import dsv41_sets
    picked = path if stat.S_ISREG(details.st_mode) else None
    if picked is not None and not picked.name.lower().endswith(".gguf"):
        return None
    if dsv41_sets.tables() is None:
        return None
    places = [path.parent] if picked is not None else [path]
    found: list[tuple[Path, dict]] = []
    for place in places:
        for row in dsv41_sets.match(_dsv41_files(place)):
            if picked is None or picked.name in row["present"] or picked.name == row["engram"]:
                found.append((place, row))
    if not found and picked is None:
        try:
            for place in _folders_with_models(path)[:8]:
                for row in dsv41_sets.match(_dsv41_files(place)):
                    found.append((place, row))
        except OSError:
            pass
    if not found:
        return None
    complete = [(place, row) for place, row in found if row["complete"]]
    issues: list[str] = []
    messages: list[str] = []
    folder_holds: list[dict] = []
    missing_shards: list[str] = []
    candidate = None
    place, row = (complete[0] if complete else found[0])
    if len(complete) > 1:
        issues.append("multiple_model_groups")
        folder_holds = [{"file": r["present"][0], "files": len(r["present"]), "folder": str(p), "family": dsv41_sets.FAMILY, "model": r["model"]}
                        for p, r in complete]
        names = [f"DeepSeek-V4.1-Flash {r['label']} ({r['present'][0]})" for _p, r in complete]
        messages.append(f"The folder holds {len(complete)} DeepSeek sets: " + "; ".join(names) +
                        ". Choose the .gguf file of the one you want.")
    elif not complete:
        issues.append("incomplete_or_invalid_shard_set")
        missing_shards = list(row["missing"]) + list(row["wrong_size"])
    else:
        candidate = {"family": dsv41_sets.FAMILY, "model": row["model"], "basis": "known_names_and_sizes", "expected_expert_types": [],
                     "expected_tensor_count": None,
                     "setup_args_hint": ["--family", dsv41_sets.FAMILY, "--model", row["model"], "--gguf-dir", str(place)]}
        line = _not_here(row["model"])
        if line:       # recommend, never force: say what does work and stop before anything is copied or packed (Mac: no Metal IQ1_S kernel; DGX Spark: never run)
            issues.append("not_on_this_mac" if sys.platform == "darwin" else "not_on_this_spark")
            candidate = None
            messages.append(line)
        messages.append(f"This is DeepSeek-V4.1-Flash, {row['label']}: all {len(row['present'])} model files are here"
                        + (", and the n-gram table." if row["engram"] else "."))
        if not row["engram"]:
            messages.append("The n-gram table (111 GB) is not in this folder: setup downloads it, or put engram-IQ4_NL.gguf beside the model files.")
        if row["legacy"]:
            messages.append("These are the file names of the first release; Strata uses them as they are.")
        if row.get("layout") == "six":
            messages.append("These are ISTA-DASLab's six-shard files of 27 October 2026 (the Engram is the sixth part); Strata uses them as they are.")
        if row.get("newer_first"):
            messages.append("The first part is the version ISTA-DASLab re-published on 26 October 2026 (a larger chat template); Strata uses it as it is.")
        messages.append("Strata checks the files once when it sets them up (about 5 minutes on a fast drive); nothing is copied or downloaded for them.")
    sizes = _dsv41_files(place)
    wanted = list(row["present"]) + ([row["engram"]] if row["engram"] else [])
    files = [{"name": n, "path": str(place / n), "bytes": sizes.get(n, 0), "gguf_version": None} for n in sorted(wanted)]
    shards = [place / n for n in sorted(row["present"])]
    already = None
    if candidate is not None and shards:
        try:
            already = _already_set_up(shards, candidate)
        except Exception:                               # noqa: BLE001 - only a convenience for the screen
            already = None
    return {"source": str(path), "model_dir": str(place), "scope": "local_gguf_candidate", "files": files, "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files), "metadata": None, "candidate_selection": candidate,
            "plain_issues": list(dict.fromkeys((missing_words(missing_shards) if (i == "incomplete_or_invalid_shard_set" and missing_shards)
                                                else plain_issue(i)) for i in issues)),
            "observed_expert_types": [], "observed_tensor_count": 0, "ignored_auxiliary_files": [], "projectors": [], "messages": messages,
            "already_set_up": already, "reads_pictures": False, "folder_holds": folder_holds, "missing_shards": missing_shards,
            "file_set_status": "plausible" if not issues else "invalid_or_incomplete", "issues": issues,
            "adoption_status": "requires_name_size_sha256_and_engine_checks", "execution_compatibility": "unverified", "would_write": False,
            "notes": ["The names and sizes are the pinned ones of the official sets; setup checks every SHA-256 once before it adopts them",
                      "No model is adopted or copied by this preview"]}


def plan_import(raw_path: str) -> dict:
    """Inspect a local candidate without copying, parsing tensors or adopting it.

    The picked path can be a .gguf file (any shard of a split set), the folder with the files, or a folder above it (a Strata data
    folder, a whole Hugging Face repository): the folder with the model is looked for, and when a folder holds several models
    the one Strata knows is used if it is the only one; with two it says which and asks for the file. The result's `model_dir`
    is the folder to hand to setup (--gguf-dir). Links to files and folders are followed."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ProfileError("model import path must be absolute")
    try:
        details = path.stat()
    except OSError:
        raise ProfileError("model import path does not exist (or is a link to something that is gone)", 5) from None
    official_deepseek = _dsv41_plan(path, details)         # IST-DAS sets under their official or first-release names
    if official_deepseek is not None:
        return official_deepseek
    notes: list[str] = []
    folder_holds: list[dict] = []
    issues: list[str] = []
    missing_shards: list[str] = []
    if stat.S_ISDIR(details.st_mode):
        shards, ignored_auxiliary, cut = _list_gguf(path)
        if cut:
            notes.append(f"This folder holds more than {MAX_LISTED} GGUF files; only the first {MAX_LISTED} were looked at.")
        if not shards:
            places = _folders_with_models(path)
            if not places:
                raise ProfileError("no GGUF files found at the model import path (or in the folders inside it)", 5)
            scanned = [(place, *_list_gguf(place)[:2]) for place in places]
            usable = [(place, files, readers, members) for place, files, readers in scanned
                      for _key, members in sorted(_split_groups(files).items(), key=lambda kv: kv[0]) if _known(members)]
            if len(usable) == 1:
                place, shards, ignored_auxiliary, members = usable[0]
                notes.append(f"The model files are in {place}.")
            else:
                place, shards, ignored_auxiliary = scanned[0]
                if len(usable) > 1 or len(scanned) > 1:
                    issues.append("multiple_model_groups")
                    folder_holds = _holds([(p, _split_groups(f)) for p, f, _r in scanned])
                else:
                    notes.append(f"The model files are in {place}.")
    elif stat.S_ISREG(details.st_mode) and path.name.lower().endswith(".gguf"):
        ignored_auxiliary = []
        try:                       # the picture reader that sits next to a chosen model file
            ignored_auxiliary = [c.name for c in path.parent.iterdir()
                                 if c.name.lower().startswith("mmproj") and c.name.lower().endswith(".gguf")][:8]
        except OSError:
            pass
        if path.name.lower().startswith("mmproj"):
            raise ProfileError("this is a picture reader (mmproj), not a model. Choose the model's own "
                               ".gguf file; Strata finds the reader next to it", 5)
        match = SHARD.fullmatch(path.name)
        if match:
            total = int(match.group(3))
            if not 1 <= total <= MAX_IMPORT_SHARDS:
                raise ProfileError("GGUF shard count must be in 1..128", 5)
            shards = [path.with_name(f"{match.group(1)}-{i:05d}-of-{total:05d}.gguf")
                      for i in range(1, total + 1)]
        else:
            shards = [path]
    else:
        raise ProfileError("model import needs a GGUF file or directory", 5)
    if not shards:
        raise ProfileError("no GGUF files found at the model import path", 5)
    shards.sort(key=lambda item: item.name)
    groups = _split_groups(shards)
    if len(groups) != 1 and "multiple_model_groups" not in issues:
        known_sets = [key for key, members in groups.items() if _known(members)]
        if len(known_sets) == 1:
            # one set in the folder is a model Strata knows: use it and say what else is there
            keep = known_sets[0]
            others = sorted(sorted(m, key=lambda x: x.name)[0].name for k, m in groups.items() if k != keep)
            shards = sorted(groups[keep], key=lambda item: item.name)
            notes.append("The folder also holds " + ", ".join(others[:6]) + (" and more" if len(others) > 6 else "") +
                         f"; Strata uses {shards[0].name} and leaves the others alone.")
            groups = {keep: shards}
        else:
            issues.append("multiple_model_groups")
            folder_holds = _holds([(shards[0].parent, groups)])
    if len(groups) == 1 and "multiple_model_groups" not in issues:
        (_, total), members = next(iter(groups.items()))
        found = {int(m.group(2)) for m in (SHARD.fullmatch(item.name) for item in members) if m} or {1}
        if total > MAX_IMPORT_SHARDS or found != set(range(1, total + 1)):
            issues.append("incomplete_or_invalid_shard_set")
            sample = next((SHARD.fullmatch(item.name) for item in members if SHARD.fullmatch(item.name)), None)
            if sample and total <= MAX_IMPORT_SHARDS:        # #3478: name the files that are not there
                missing_shards = [f"{sample.group(1)}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1) if i not in found]
    files = []
    for item in shards:
        if missing_shards and not item.exists():
            continue                                         # a shard that is named only to be reported missing has no header to read
        try:
            files.append(_gguf_header(item))
        except ProfileError as exc:
            issues.append(f"{item.name}: {exc}")
    metadata = None
    observed_experts: set[str] = set()
    observed_tensor_count = 0
    observed_payload = 0
    if files and not issues:
        for item in shards:
            try:
                inspected = inspect_metadata(item)
            except ProfileError as exc:
                issues.append(f"{item.name}: {exc}")
                continue
            if metadata is None:
                metadata = inspected
            observed_experts.update(inspected["expert_tensor_types"])
            observed_tensor_count += inspected["tensor_count_scanned"]
            observed_payload += int(inspected.get("payload_bytes_required") or 0)
            if inspected["architecture"] is not None and inspected["architecture"] != "qwen4exp":
                issues.append(f"{item.name}: unsupported_architecture")
    if metadata is not None:
        if metadata["architecture"] != "qwen4exp":
            issues.append("unsupported_or_missing_architecture")
        if metadata["tokenizer_model"] != "gpt2" or not metadata["tokenizer_tokens_present"] or \
                not metadata["tokenizer_merges_present"]:
            issues.append("unsupported_or_incomplete_tokenizer_metadata")
    model_dir = shards[0].parent
    projectors = [_assess_projector(model_dir / n) for n in sorted(ignored_auxiliary)[:8]]
    messages = list(notes)
    if folder_holds:
        several = len({h.get("folder") for h in folder_holds}) > 1
        names = [(f"{h['family']} {h['model']}" if h.get("model") else h["file"]) + (f" in {h['folder']}" if several else f" ({h['file']})")
                 for h in folder_holds[:8]]
        messages.append(f"The folder holds {len(folder_holds)} models: " + "; ".join(names) + (" and more" if len(folder_holds) > 8 else "") +
                        ". Choose the folder of the one you want, or type the path of its .gguf file.")
    if any(i.endswith("unsupported_architecture") or i == "unsupported_or_missing_architecture" for i in issues):
        messages.append(UNSUPPORTED_MODEL_TYPE)
    elif projectors:
        messages.extend(p["plain"] for p in projectors)
    candidate = _candidate_selection(shards)
    if candidate is None and metadata is not None and not issues:
        candidate = _candidate_by_header(shards, observed_tensor_count, observed_payload, observed_experts)
        if candidate is None:
            candidate = _candidate_by_layout(shards, observed_tensor_count, observed_payload, observed_experts)
            if candidate is not None:
                messages.append(f"These files are not exactly a build Strata has tested; they have the layout of {candidate['family']} "
                                f"{candidate['model']}, so Strata uses them as that model. Check a few answers before you rely on them.")
    if "multiple_model_groups" in issues:
        candidate = None                                  # nothing is chosen for the person
    if candidate is None and not {"multiple_model_groups", "incomplete_or_invalid_shard_set"} & set(issues):
        issues.append("unrecognized_or_unsupported_model_variant")
    elif candidate is not None and metadata is not None:
        expected = set(candidate["expected_expert_types"])
        if not observed_experts or observed_experts - expected:
            issues.append("missing_or_unexpected_expert_quantization")
    return {"source": str(path), "model_dir": str(model_dir), "scope": "local_gguf_candidate", "files": files,
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
            "metadata": metadata, "candidate_selection": candidate,
            "plain_issues": list(dict.fromkeys((missing_words(missing_shards) if (i == "incomplete_or_invalid_shard_set" and missing_shards) else plain_issue(i)) for i in issues)),
            "observed_expert_types": sorted(observed_experts),
            "observed_tensor_count": observed_tensor_count,
            "ignored_auxiliary_files": sorted(ignored_auxiliary),
            "projectors": projectors, "messages": messages, "already_set_up": _already_set_up(shards, candidate),
            "reads_pictures": any(p["status"] in ("ok", "other_name") for p in projectors) and
            not any(i.endswith("unsupported_architecture") for i in issues),
            "folder_holds": folder_holds, "missing_shards": missing_shards,
            "file_set_status": "plausible" if not issues else "invalid_or_incomplete",
            "issues": issues, "adoption_status": "requires_architecture_quantization_tokenizer_and_engine_checks",
            "execution_compatibility": "unverified", "would_write": False,
            "notes": ["A known shard name is only a selection hint; it does not prove provenance or file integrity",
                      "Tensor directory, model-pack, engine version and hardware checks remain before adoption",
                      "No model is adopted or copied by this preview"]}
