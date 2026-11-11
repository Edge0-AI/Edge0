"""The official DeepSeek-V4.1-Flash GGUF sets (IST-DAS 1.75 / 2.00 / 2.25 bpw), recognised by their pinned file names and sizes.

The Terminal's setup downloads them (setup.py MULTI_FILE_FAMILIES, DSV41_ISTA_FILES) and accepts the files the first release was downloaded as
(DSV41_LEGACY_FILES: model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf). The app's "I already have the model" preview used to know only the
Qwen fingerprints, so a folder with such a set ended in "the folder holds more than one model" (the Engram is a second file) or, for one shard,
"this model type is not supported by the Strata engine". This module is how Control says what a folder of these files is, from the names and
sizes alone: no tensor is read and no 40 GB file is hashed in a preview (setup checks the SHA-256 once, when the set is adopted).

The tables are setup.py's own (read from it, never copied), so a new size added there is recognised here.

Three kinds of files are the same set: the pinned four-shard files (and the first release's names), the four-shard files as ISTA-DASLab re-published
their shard 1 on 2026-10-26 (a larger chat template: DSV41_NEW_SHARD1, what the Hub's last four-shard commit 653a9874 serves), and, since 0.2.8.1, ISTA-DASLab's
six-shard layout of 2026-10-27 (DSV41_ISTA6_FILES).  setup downloads only the first; it takes the other two from files already here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TABLES: dict | None = None
FAMILY = "deepseek41"


def tables() -> dict | None:
    """{model: {"label", "shards": {name: bytes}, "legacy": {name: bytes}, "new1": {name: bytes}, "six": {name: bytes}}} for the official sets and
    {"engram": {name: bytes}}; None when setup.py cannot be read (the preview then behaves as before)."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES or None
    try:
        spec = importlib.util.spec_from_file_location("strata_setup_tables", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)            # type: ignore[union-attr]
        family = module.MULTI_FILE_FAMILIES[FAMILY]
        ista, legacy, renamed = module.DSV41_ISTA_FILES, module.DSV41_LEGACY_FILES, module.DSV41_LEGACY_NAMES
        new1, six = getattr(module, "DSV41_NEW_SHARD1", {}), getattr(module, "DSV41_ISTA6_FILES", {})
        result: dict = {"sets": {}, "engram": {}}
        for model, entry in family["models"].items():
            if entry.get("recipe") not in ("ista", "udlow"):
                continue                                # Unsloth's 508 GB set is not an "already have it" case of the app
            ista = {**ista, **getattr(module, "DSV41_UD_FILES", {})}
            shards = {name: ista[name][0] for name in entry["shards"]}
            old = {renamed[name]: legacy[renamed[name]][0] for name in entry["shards"] if name in renamed and renamed[name] in legacy}
            alt = entry.get("engram_shard")             # Unsloth's UD-IQ1 sets: shard 5 is the Engram (the same data as IST-DAS's file)
            result["sets"][model] = {"label": entry.get("label", model), "shards": shards, "legacy": old,
                                     "engram_alt": {alt: ista[alt][0]} if alt else {},
                                     "new1": {name: new1[name][0] for name in entry["shards"] if name in new1},
                                     "six": {name: six[name][0] for name in entry.get("shards6") or [] if name in six}}
        for name in (module.DSV41_ISTA_ENGRAM, renamed.get(module.DSV41_ISTA_ENGRAM)):
            if name and (name in ista or name in legacy):
                result["engram"][name] = (ista.get(name) or legacy[name])[0]
        _TABLES = result
    except Exception:                                   # noqa: BLE001 - a preview must never fail because of this
        _TABLES = {}
    return _TABLES or None


def match(files: dict[str, int]) -> list[dict]:
    """The official sets the files {name: bytes} of one folder belong to, best first: complete sets before partial ones.
    Each: {"model", "label", "complete", "present": [names], "missing": [names], "wrong_size": [names], "legacy": bool, "layout": "four"|"legacy"|"six",
    "engram": name|None}.  A set counts when at least one of its shards is there; names and sizes are the pinned ones (shard 1 of the four-shard layout also
    at the size of its 2026-10-26 re-publication).  A folder that holds the whole four-shard set and the whole six-shard set of one model is that
    model once (the pinned four-shard files are the ones used)."""
    known = tables()
    if not known:
        return []
    engram = next((name for name in known["engram"] if files.get(name) == known["engram"][name]), None)
    found = []
    for model, entry in known["sets"].items():
        engram = engram_any = next((name for name in known["engram"] if files.get(name) == known["engram"][name]), None)
        if engram is None:                             # an Unsloth set's own shard 5 is its Engram
            engram = next((n for n, size in entry.get("engram_alt", {}).items() if files.get(n) == size), None)
        for layout, pinned in (("four", entry["shards"]), ("legacy", entry["legacy"]), ("six", entry.get("six") or {})):
            if not pinned:
                continue
            present = [n for n in pinned if n in files]
            if not present:
                continue
            ok_sizes = {n: {pinned[n], *([entry["new1"][n]] if layout == "four" and n in entry.get("new1", {}) else [])} for n in pinned}
            wrong = [n for n in present if files[n] not in ok_sizes[n]]
            missing = [n for n in pinned if n not in files]
            found.append({"model": model, "label": entry["label"], "legacy": layout == "legacy", "layout": layout, "present": sorted(present),
                          "missing": missing, "wrong_size": wrong, "complete": not missing and not wrong, "engram": engram,
                          "newer_first": [n for n in present if n not in wrong and files[n] != pinned[n]],
                          "bytes": sum(files[n] for n in present if n not in wrong) + ((known["engram"].get(engram) or entry["engram_alt"].get(engram) or 0) if engram else 0)})
    pinned_whole = {row["model"] for row in found if row["complete"] and row["layout"] != "six"}
    found = [row for row in found if not (row["layout"] == "six" and row["complete"] and row["model"] in pinned_whole)]
    found.sort(key=lambda row: (not row["complete"], len(row["missing"]) + len(row["wrong_size"]), row["model"]))
    return found
