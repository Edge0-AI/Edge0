"""Which drafter a model uses to guess ahead: MTP, DFlash2 or none. One place, read by the Terminal and the app.

The choice is the person's, and it is saved in the model's run config as ``"drafting"``: ``auto`` (the default), ``mtp``, ``dflash2`` or
``off``. ``strata launch --drafting ...`` writes it, the app's Models screen writes it, and ``serve/server.py`` applies it at every start
(like ``"mtp": false``, strata_control/draft_layer.py), so a setup run that rewrites the engine arguments cannot undo it.

  mtp      the engine's default drafter, the model's own draft layer (no extra flags)
  dflash2  ``--draft dflash2 --draft-model <pack>``: the DFlash2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2) for Qwen3.8-Flash-Next
  off      no draft layer (the same as ``"mtp": false``: the engine still looks ahead in the text with ``--spec``)
  auto     the table below says which of mtp / dflash2 this card class gets

Recommend, never force: a choice that cannot be applied (DFlash2 for another model, no drafter file yet) does not stop the start. The
model starts with the MTP and ``Decision.note`` says why in plain words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "mtp", "dflash2", "off")
DEFAULT_MODE = "auto"
LABELS = {"auto": "Auto (recommended)", "mtp": "MTP", "dflash2": "DFlash2", "off": "Off"}
DRAFTERS = ("mtp", "dflash2", "off")

# --------------------------------------------------------------------------------------------------- the suggestion table
#
# "Suggest DFlash2 where the win is 100%": Auto picks DFlash2 only on a card class where it measured faster than the MTP on EVERY tested
# prompt (``faster_everywhere``). A class goes here when it was measured; a row with ``faster_everywhere`` False is MTP whatever the average
# says. Fill this in as measurements come in (and change nothing else: the Terminal and the app both read it, tests/terminal/test_drafting.py
# checks that a DFlash2 row has the evidence).
#
# Evidence so far (phaseA-tests/notes-dflash2-p1.md, -p02.md, -ports.md; DFlash2 against MTP, tokens per second, same prompts and flags):
#   strix_halo  Radeon 8060S (gfx1151): faster on chat, code and json, SLOWER on tool_call (-6%).
#   blackwell   RTX 5070 (sm_120, 12 GB): faster on code, SLOWER on chat (-1%).
#   rdna4       Radeon AI PRO R9700 (gfx1201): slower on every prompt.
#   ampere      RTX 3060 (sm_86, 12 GB): slower on every prompt.
#   other       not measured.


@dataclass(frozen=True)
class Suggestion:
    drafter: str                 # "mtp" | "dflash2"
    faster_everywhere: bool      # DFlash2 measured faster than the MTP on every tested prompt of this class
    evidence: str                # one line: what was measured


SUGGESTED: dict[str, Suggestion] = {
    "strix_halo": Suggestion("mtp", False, "gfx1151: faster on chat, code and json; slower on tool_call (-6%)"),
    "blackwell": Suggestion("mtp", False, "sm_120 (RTX 5070, 12 GB): faster on code; slower on chat (-1%)"),
    "rdna4": Suggestion("mtp", False, "gfx1201 (R9700): slower on every prompt"),
    "ampere": Suggestion("mtp", False, "sm_86 (RTX 3060, 12 GB): slower on every prompt"),
    "other": Suggestion("mtp", False, "not measured on this kind of card"),
}

#: GPU architecture (what nvidia-smi's compute capability and the kernel driver report) -> card class
ARCH_CLASS = (
    (re.compile(r"^gfx1151$"), "strix_halo"),
    (re.compile(r"^gfx120[01]$"), "rdna4"),
    (re.compile(r"^sm_12\d$"), "blackwell"),
    (re.compile(r"^sm_86$"), "ampere"),
)


def card_class(archs: Any) -> str:
    """The class of the first card whose architecture the table knows (``sm_120``, ``gfx1151``...); ``other`` for an unknown or empty list."""
    for arch in ([archs] if isinstance(archs, str) else list(archs or [])):
        for pattern, name in ARCH_CLASS:
            if pattern.match(str(arch).strip().lower()):
                return name
    return "other"


def suggest(klass: str) -> str:
    """What Auto picks on this card class: DFlash2 only where it was faster on every tested prompt, MTP otherwise."""
    row = SUGGESTED.get(klass) or SUGGESTED["other"]
    return "dflash2" if row.drafter == "dflash2" and row.faster_everywhere else "mtp"


# --------------------------------------------------------------------------------------------------- the drafter file
#
# incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), pinned to one revision. The safetensors file is 1.2 GB; tools/dflash2_pack.py turns it into the
# engine's runtime file (Q8 pack, about 0.76 GB). The sizes and hashes are the repository's own for that revision: a download is refused
# if either differs.

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LICENSE = "Apache-2.0"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
FILES = (
    ("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"),
)
DOWNLOAD_GB = 1.2            # what is fetched (the safetensors file)
PACK_GB = 0.76               # what stays on disk (the source file is removed after packing)
PACK_NAME = "qwen3.8-flash-next-dflash2.q8.pack"
VRAM_GB = 1.2                # what the drafter needs in graphics memory (the engine's own estimate)
FOLDER = "dflash2"           # under the model data folder, next to ``mtp``

#: model families (strata launch --family) the drafter was trained for and measured on: Qwen3.8-Flash-Next and its Gyro build.
SUPPORTED_FAMILIES = ("qwen", "gyro")
_SUPPORTED_NAME = re.compile(r"^qwen3\.8-flash-next(-gyro-[a-z0-9._-]+|-(q|iq|tq)\d[a-z0-9_.]*)?$", re.IGNORECASE)


def supported_model(model_name: str | None = None, family: str | None = None) -> bool:
    """DFlash2 is for Qwen3.8-Flash-Next only: by family when it is known (the Terminal), else by the run config's model name."""
    if family is not None:
        return family.lower() in SUPPORTED_FAMILIES
    return bool(model_name and _SUPPORTED_NAME.match(str(model_name)))


def pack_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / FOLDER / PACK_NAME


def pack_ready(data_dir: str | Path | None) -> bool:
    try:
        return bool(data_dir) and pack_path(data_dir).is_file() and pack_path(data_dir).stat().st_size > (1 << 20)
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------- a decision


@dataclass
class Decision:
    asked: str                   # what the person chose (auto | mtp | dflash2 | off)
    drafter: str                 # what the engine is started with: mtp | dflash2 | off
    note: str = ""               # plain words when the choice could not be applied; "" otherwise
    flags: tuple[str, ...] = ()  # engine flags added


def normalize(value: Any) -> str:
    text = str(value if value is not None else DEFAULT_MODE).strip().lower()
    return text if text in MODES else DEFAULT_MODE


#: Does the engine in this release run DFlash2 when it serves (the app and `strata launch`)? In f-023-int src/program/generate.cpp, `--draft dflash2`
#: is "for one card and the CLI run (not --serve, --batch or a layer split)": a served model prints "--draft dflash2 is off: ... the MTP drafts
#: instead". So today the choice cannot take effect when serving, and nothing is downloaded or packed for it (1.2 GB for no effect would be a bad
#: surprise); the start says so ("Drafting: MTP (DFlash2 isn't available in this mode yet)"). Flip this when the engine serves it; everything
#: else (flags, download, pack, fallback text) is in place and tested with it on.
ENGINE_SERVES_DFLASH2 = False

#: Why the MTP runs when something else was asked for: a closed set of short reasons, each one is a sentence in the interface catalog in all eight
#: languages (the line is "Drafting: MTP (<reason>)").
DIDNT_FIT = "DFlash2 didn't fit"
NOT_HERE = "DFlash2 isn't available in this mode"
NOT_SUPPORTED = "DFlash2 is for Qwen3.8-Flash-Next only"
NOT_DOWNLOADED = "the DFlash2 drafter isn't downloaded yet"
NOT_SERVED = NOT_HERE
REASONS = (DIDNT_FIT, NOT_HERE, NOT_SUPPORTED, NOT_DOWNLOADED)


def mtp_line(reason: str) -> str:
    return f"Drafting: MTP ({reason})"


def decide(mode: Any, *, klass: str, model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None) -> Decision:
    """The drafter a start uses. Never raises: a choice that cannot be applied becomes MTP with a note."""
    asked = normalize(mode)
    if asked == "off":
        return Decision(asked, "off")
    want = suggest(klass) if asked == "auto" else asked
    if want != "dflash2":
        return Decision(asked, "mtp")
    if not supported_model(model_name, family):
        return Decision(asked, "mtp", NOT_SUPPORTED if asked == "dflash2" else "")
    if not ENGINE_SERVES_DFLASH2:
        return Decision(asked, "mtp", NOT_SERVED if asked == "dflash2" else "")
    if not pack_ready(data_dir):
        return Decision(asked, "mtp", NOT_DOWNLOADED)
    return Decision(asked, "dflash2", flags=("--draft", "dflash2", "--draft-model", str(pack_path(data_dir))))


def needs_pack(mode: Any, *, klass: str, model_name: str | None = None, family: str | None = None, data_dir: str | Path | None = None) -> bool:
    """Would this choice use the DFlash2 drafter if its file were here, and is it not here yet? (Then it is fetched before the start.)"""
    asked = normalize(mode)
    want = suggest(klass) if asked == "auto" else asked
    return want == "dflash2" and ENGINE_SERVES_DFLASH2 and supported_model(model_name, family) and not pack_ready(data_dir)


DRAFT_FLAGS = {"--draft": True, "--draft-model": True}


def strip_draft_flags(args: list[Any]) -> list[str]:
    """The arguments without ``--draft`` / ``--draft-model`` (and their values)."""
    out: list[str] = []
    items = [str(a) for a in args]
    i = 0
    while i < len(items):
        if items[i] in DRAFT_FLAGS:
            i += 2 if i + 1 < len(items) and not items[i + 1].startswith("--") else 1
            continue
        out.append(items[i])
        i += 1
    return out


def data_dir_of(args: list[Any]) -> Path | None:
    """The model data folder an argument list belongs to: `--pack <data>/packs/<tag>`, else `--mtp <data>/mtp/rt`."""
    items = [str(a) for a in args]
    for flag, up in (("--pack", 2), ("--mtp", 2)):
        if flag in items and items.index(flag) + 1 < len(items):
            path = Path(items[items.index(flag) + 1])
            if flag == "--mtp" and path.name.isdigit():
                continue
            return path.parents[up - 1] if len(path.parents) >= up else None
    return None


def apply(cfg: dict[str, Any], *, klass: str, data_dir: str | Path | None = None, family: str | None = None) -> Decision:
    """Put the config's ``"drafting"`` choice on its engine arguments (in place). ``off`` is the draft layer's own switch (``"mtp": false``)."""
    from strata_control import draft_layer
    decision = decide(cfg.get("drafting"), klass=klass, model_name=cfg.get("model_name"), family=family, data_dir=data_dir)
    if cfg.get("drafting") is None and decision.drafter == "mtp":
        return decision                                       # nobody chose and Auto is the MTP: the arguments stay exactly as they are
    args = strip_draft_flags(cfg.get("args") or [])
    if decision.drafter == "off":
        cfg["mtp"] = False
        cfg["args"] = args
        draft_layer.apply(cfg)
    else:
        if cfg.get("drafting") is not None and normalize(cfg.get("drafting")) != "off" and cfg.get("mtp") is False:
            cfg["mtp"] = True                                 # an explicit drafter choice overrides an older "mtp": false
        cfg["args"] = args + list(decision.flags)
    return decision


# --------------------------------------------------------------------------------------------------- what the engine said

_ON = re.compile(r"DFlash2 drafter on\b")


def fallback_note(log_text: str) -> str:
    """From one engine start's log: ``Drafting: MTP (DFlash2 didn't fit)`` when the engine fell back from DFlash2, else ``""``. The engine
    names the reason in its own words; two are told apart: no room in graphics memory, and a mode this build does not support."""
    text = str(log_text or "")
    if _ON.search(text):
        return ""
    if "--draft dflash2" not in text:
        return ""
    low = text.lower()
    if "it is for one card and the cli run" in low or "needs a native pack" in low or "it needs --mtp" in low:
        reason = NOT_HERE
    else:
        reason = DIDNT_FIT
    return mtp_line(reason)


def summary_line(decision: Decision, engine_note: str = "") -> str:
    """The line the Terminal and the app show: what drafts."""
    if engine_note:
        return engine_note
    if decision.note:
        return mtp_line(decision.note)
    return f"Drafting: {LABELS.get(decision.drafter, decision.drafter) if decision.drafter != 'mtp' else 'MTP'}"


# --------------------------------------------------------------------------------------------------- what a start did

_pending_note = ""


def remember_note(note: str) -> None:
    """The server calls this when it builds the engine arguments: why the choice was not applied, for the engine that starts next."""
    global _pending_note
    _pending_note = note or ""


def take_note() -> str:
    global _pending_note
    note, _pending_note = _pending_note, ""
    return note


def engine_info(args: list[Any], log_text: str = "", note: str = "") -> dict[str, str]:
    """The facts of one engine start for /metrics and the apps: ``drafting`` (mtp | dflash2 | off) and ``drafting_note`` (plain words, or "").
    The engine's own log wins over the choice: if it says it fell back, that is what ran."""
    items = [str(a) for a in args]
    asked_dflash2 = any(items[i] == "--draft" and i + 1 < len(items) and items[i + 1] == "dflash2" for i in range(len(items)))
    fell_back = fallback_note(log_text) if asked_dflash2 else ""
    if asked_dflash2 and not fell_back:
        return {"drafting": "dflash2", "drafting_note": ""}
    drafter = "mtp" if "--mtp" in items else "off"
    text = fell_back or (mtp_line(note) if note and drafter == "mtp" else "")
    return {"drafting": drafter, "drafting_note": text}
