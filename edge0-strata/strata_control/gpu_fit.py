"""Which graphics cards a model should use: ONE place for the app, `strata launch` and setup.

The rule (owner decision):
1. First try the model on ONE card (the largest) plus system RAM, the normal hybrid mode.
2. Only when that does not fit, or would be too tight, offer to add the second card as a layer split.
3. Let the person choose: every option comes with plain numbers and a recommended one is pre-selected (recommend, never force).

The memory arithmetic is native-fit.ts `fitOn` (apps/host/src/lib/native-fit.ts) with the same constants (a test keeps them equal):
the experts are held first in the graphics memory beyond about DENSE_GB (the dense weights, buffers and a 32K cache), then in RAM beyond
about SYSTEM_ROOM_GB left for the system; whatever is left would be read from the SSD (`spill_gb`). A split adds the memory of the extra
cards but every extra card holds its own dense layers and prompt buffers, so each one costs SPLIT_CARD_OVERHEAD_GB of that (an estimate:
docs/MULTI_GPU.md says each card "holds the dense weights of its layers and its own prompt buffers").

Cards that count: not integrated (an iGPU shares the system RAM and is ignored), at least MIN_CARD_GB of graphics memory, flagged usable by
the caller's own driver checks (NVIDIA RTX 20 or newer, a supported AMD card, ...). A card needs SPLIT_MIN_VRAM_GB to share a model.
Pure functions, no I/O: setup.py, strata_cli/launch and tools/strata_mcp.py import it.
"""

from __future__ import annotations

from typing import Any

DENSE_GB = 5.0                 # = native-fit.ts DENSE_GB
SYSTEM_ROOM_GB = 10.0          # = native-fit.ts SYSTEM_ROOM_GB
DENSE_EXTRA_GB = 1.0           # = native-fit.ts DENSE_EXTRA_GB
SPLIT_CARD_OVERHEAD_GB = 2.0   # each extra card of a split: its own dense layers and prompt buffers
SPLIT_MIN_VRAM_GB = 8.0        # setup.py SPLIT_MIN_VRAM_GB: a card sharing a model needs this much
SPLIT_PROMPT_VRAM_GB = 12.0    # setup.py SPLIT_PROMPT_VRAM_GB (#448): a smaller second card caps the prompt chunk
MIN_CARD_GB = 4.0              # under this a card cannot hold the dense weights: ignored (an iGPU, a display card)
FITS_SPARE_RAM_GB = 4.0        # RAM left beyond the experts below which "fits" becomes "tight"
SPLIT_PROMPT_GAIN = "about 20% faster"   # setup.py's own words for prompts read on two cards (docs/MULTI_GPU.md)


def _round1(x: float) -> float:
    return round(float(x) * 10) / 10


# ---- one engine, one vendor -----------------------------------------------------------------------------------------------
# Every Strata engine package drives the cards of ONE vendor (CUDA: NVIDIA, HIP: AMD, SYCL: Intel, Metal: Apple). A model is never split
# across vendors, and two models at once, one per vendor, is not supported yet. So the cards offered for running and splitting are the
# ones the engine can drive; a better card of the other vendor is mentioned in one plain line (recommend, never force).
BACKEND_VENDOR = {"cuda": "nvidia", "hip": "amd", "sycl": "intel", "metal": "apple"}
VENDOR_NAME = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}
SECOND_MODEL_SENTENCE = ("Running two models at the same time, one on each make of graphics card, is not supported yet. "
                         "Strata runs one model at a time on the cards of one make.")


def vendor_of_backend(backend: str | None) -> str | None:
    return BACKEND_VENDOR.get(str(backend or "").lower())


def _vendor(card: dict) -> str | None:
    v = card.get("vendor")
    return str(v).lower() if v else None


def pick_vendor(cards: list[dict], backend: str | None = None) -> str | None:
    """The make of card the engine drives: the installed (or detected) engine's, else the only make among the cards that count; with more
    than one make, NVIDIA (the engine setup picks first), else the make of the card with the most memory. None: no card names a make."""
    named = vendor_of_backend(backend)
    if named:
        return named
    keep, _ = usable_cards(cards)
    makes = {_vendor(c) for c in keep if _vendor(c)}
    if len(makes) <= 1:
        return next(iter(makes), None)
    if "nvidia" in makes:
        return "nvidia"
    return _vendor(keep[0])


def for_vendor(cards: list[dict], vendor: str | None) -> tuple[list[dict], list[dict]]:
    """(the cards the engine of that make can drive, the cards of other makes). A card that names no make goes with the engine's."""
    if not vendor:
        return list(cards or []), []
    mine = [c for c in cards or [] if _vendor(c) in (None, vendor)]
    return mine, [c for c in cards or [] if _vendor(c) not in (None, vendor)]


def _short_name(c: dict, vendor: str) -> str:
    n = str(c.get("name") or "graphics card")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel "):
        if n.startswith(junk):
            n = n[len(junk):]
            break
    return f"{VENDOR_NAME.get(vendor, vendor.title())} {n}".strip()


def other_vendor_note(cards: list[dict], vendor: str | None) -> str | None:
    """One plain line when the other make's best card is bigger than the best card the engine can use, else None:
    "This computer also has an AMD R9700 (32 GB). The AMD build of Strata can use it - it fits bigger models." """
    if not vendor:
        return None
    mine, others = for_vendor(usable_cards(cards)[0], vendor)
    others = [c for c in others if _vendor(c) and not _shares_memory(c)]
    if not others:
        return None
    best = max(others, key=lambda c: float(c.get("vram_gb") or 0))
    mine_best = max((float(c.get("vram_gb") or 0) for c in mine), default=0.0)
    if float(best.get("vram_gb") or 0) <= mine_best:
        return None
    other = _vendor(best)
    return (f"This computer also has an {_short_name(best, other)} ({float(best.get('vram_gb') or 0):.0f} GB). "
            f"The {VENDOR_NAME.get(other, other.title())} build of Strata can use it — it fits bigger models.")


def memory_needed(variant: dict) -> float:
    """RAM + VRAM the model asks for (native-fit.ts memoryNeeded)."""
    if isinstance(variant.get("memory_gb"), (int, float)):
        return float(variant["memory_gb"])
    if variant.get("mem_total") and isinstance(variant.get("ram_needed_gb"), (int, float)):
        return float(variant["ram_needed_gb"])
    return _round1(float(variant.get("experts_gb") or variant.get("ram_needed_gb") or 0) + DENSE_EXTRA_GB)


def fit(memory_gb: float, ram_gb: float | None, vram_gb: float) -> dict[str, Any]:
    """native-fit.ts `fitOn` for a given memory need: where the weights go. Keys: vram_use_gb, ram_use_gb, spill_gb, spare_ram_gb, known."""
    vram = float(vram_gb or 0)
    if not ram_gb or ram_gb <= 0:
        return {"vram_use_gb": 0.0, "ram_use_gb": 0.0, "spill_gb": 0.0, "spare_ram_gb": 0.0, "known": False}
    experts = max(0.0, memory_gb - DENSE_EXTRA_GB)
    on_gpu = max(0.0, min(experts, vram - DENSE_GB))
    ram_room = max(0.0, float(ram_gb) - SYSTEM_ROOM_GB)
    spill = max(0.0, experts - on_gpu - ram_room)
    in_ram = max(0.0, experts - on_gpu - spill)
    return {"vram_use_gb": _round1(min(vram, DENSE_GB + on_gpu) if vram > 0 else 0.0),
            "ram_use_gb": _round1(in_ram + (DENSE_EXTRA_GB if vram <= 0 else 0)), "spill_gb": _round1(spill),
            "spare_ram_gb": _round1(max(0.0, ram_room - in_ram)), "known": True}


def verdict(f: dict) -> str:
    """fits / tight / spills: all in memory with room to spare / all in memory but little RAM left / part of it is read from the SSD."""
    if not f.get("known"):
        return "unknown"
    if f["spill_gb"] >= 1.0:
        return "spills"
    return "fits" if f["spare_ram_gb"] >= FITS_SPARE_RAM_GB or f["ram_use_gb"] <= 0 else "tight"


def _shares_memory(c: dict) -> bool:
    """An integrated chip or a unified-memory chip (Apple, Strix Halo): its memory is the system RAM."""
    return bool(c.get("integrated") or c.get("uma") or c.get("unified_memory") or c.get("unified_gb") or str(c.get("vendor")) == "apple")


def usable_cards(cards: list[dict]) -> tuple[list[dict], list[dict]]:
    """(cards that count, cards ignored with a plain reason in 'why_ignored'). Largest memory first, then the card number.

    A chip that shares the system memory is ignored when the computer has a card with memory of its own (it adds nothing to it); when it is
    the only device (a Mac, a Strix Halo PC) it is the one that counts. A card under MIN_CARD_GB cannot hold the dense weights."""
    flagged = [c for c in (cards or []) if c.get("usable") is not False]
    has_real = any(not _shares_memory(c) and float(c.get("vram_gb") or 0) >= MIN_CARD_GB for c in flagged)
    keep, ignored = [], []
    for c in cards or []:
        vram = float(c.get("vram_gb") or 0)
        why = None
        if c.get("usable") is False:
            why = c.get("problem") or "Strata cannot use this card"
        elif _shares_memory(c):
            if has_real:
                why = "it shares the system memory (an integrated graphics chip), so it adds nothing"
        elif vram < MIN_CARD_GB:
            why = f"only {vram:.0f} GB of graphics memory"
        if why:
            ignored.append({**c, "why_ignored": why})
        else:
            keep.append(c)
    keep.sort(key=lambda c: (-round(float(c.get("vram_gb") or c.get("unified_gb") or 0)), c.get("index", 0)))
    return keep, ignored


def _name(c: dict) -> str:
    n = str(c.get("name") or f"card {c.get('index')}")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "):
        n = n.replace(junk, "")
    return n + (f" (card {c.get('index')})" if c.get("_dup") else "")


def own_memory_gb(c: dict) -> float:
    """The graphics memory a card adds BESIDE the RAM.  A discrete card: its VRAM.  A chip that shares the system memory (Strix Halo, an APU): only
    its BIOS carve-out (`dedicated_gb`; 0 when the report does not give it) - its "shared" pool is the RAM itself, and counting it as VRAM as well as
    as RAM made a model look like it fit twice (setup.py low_ram_vram does the same)."""
    if c.get("uma") or c.get("integrated"):
        return float(c.get("dedicated_gb") or 0)
    return float(c.get("vram_gb") or 0)


def _option(key: str, cards: list[dict], memory_gb: float, ram_gb: float | None) -> dict[str, Any]:
    n = len(cards)
    total = sum(own_memory_gb(c) for c in cards)
    eff = total - SPLIT_CARD_OVERHEAD_GB * (n - 1)
    f = fit(memory_gb, ram_gb, eff)
    v = verdict(f)
    names = " + ".join(_name(c) for c in cards)
    label = f"{names} + RAM" if n == 1 else f"{names} + RAM (layer split)"
    have = (eff + max(0.0, float(ram_gb or 0) - SYSTEM_ROOM_GB))
    if v == "unknown":
        words = "memory of this computer unknown"
    elif v == "spills":
        words = f"does not fit in memory: about {f['spill_gb']:.0f} GB is read from the SSD while answering (slower)"
    elif v == "tight":
        words = f"fits, but tight: needs {memory_gb:.0f} GB of the {have:.0f} GB available"
    else:
        words = f"fits: needs {memory_gb:.0f} GB of the {have:.0f} GB available"
    notes = []
    if n >= 2:
        notes.append(f"prompts are read {SPLIT_PROMPT_GAIN}; the cards share the layers (docs/MULTI_GPU.md)")
        small = [c for c in cards[1:] if float(c.get("vram_gb") or 0) < SPLIT_PROMPT_VRAM_GB - 0.5]
        if small and float(cards[0].get("vram_gb") or 0) >= SPLIT_PROMPT_VRAM_GB - 0.5:
            notes.append(f"{_name(small[0])} is small: it can make prompts slower than the first card alone")
    return {"key": key, "cards": [c.get("index") for c in cards], "label": label, "verdict": v, "summary": f"{label}: {words}",
            "words": words, "vram_gb": _round1(total), "spill_gb": f["spill_gb"], "ram_use_gb": f["ram_use_gb"],
            "spare_ram_gb": f["spare_ram_gb"], "notes": notes, "recommended": False}


_RANK = {"fits": 0, "tight": 1, "unknown": 2, "spills": 3}


def options(memory_gb: float, ram_gb: float | None, cards: list[dict], vendor: str | None = None) -> dict[str, Any]:
    """The choices for this model on this computer: {"options": [...], "recommended": key, "ignored": [...], "cards": n counted}.

    Options (best known first): the largest card alone; with two or more cards that can share a model, the two largest as a layer split
    (and all of them when there are more than two); then every other card alone. Exactly one option is recommended:
    the single largest card when the model fits it (with RAM); otherwise the split if it fits better (a smaller verdict or less SSD use);
    otherwise the single largest card again (nothing better to offer)."""
    vendor = vendor or pick_vendor(cards)
    mine, others = for_vendor(cards, vendor)
    keep, ignored = usable_cards(mine)
    ignored = ignored + [{**c, "why_ignored": f"it needs the {VENDOR_NAME.get(_vendor(c), _vendor(c))} build of Strata, and this engine drives "
                                              f"{VENDOR_NAME.get(vendor, vendor)} cards"} for c in others]
    out: dict[str, Any] = {"options": [], "recommended": None, "ignored": ignored, "cards": len(keep), "vendor": vendor,
                           "other_vendor_note": other_vendor_note(cards, vendor)}
    if not keep:
        return out
    names = [str(c.get("name")) for c in keep]
    keep = [{**c, "_dup": names.count(str(c.get("name"))) > 1} for c in keep]       # two cards of one model: told apart by their number
    opts = [_option("single", [keep[0]], memory_gb, ram_gb)]
    share = [c for c in keep if float(c.get("vram_gb") or 0) >= SPLIT_MIN_VRAM_GB - 0.5]
    if len(share) >= 2:
        opts.append(_option("split", share[:2], memory_gb, ram_gb))
        if len(share) > 2:
            opts.append(_option("split-all", share, memory_gb, ram_gb))
    for c in keep[1:]:
        opts.append(_option(f"single-{c.get('index')}", [c], memory_gb, ram_gb))
    single = opts[0]
    best = single
    if single["verdict"] != "fits":
        for o in opts[1:]:
            if len(o["cards"]) > 1 and (_RANK[o["verdict"]], o["spill_gb"]) < (_RANK[best["verdict"]], best["spill_gb"]):
                best = o
    best["recommended"] = True
    out["options"], out["recommended"] = opts, best["key"]
    return out


def recommended_cards(memory_gb: float, ram_gb: float | None, cards: list[dict], vendor: str | None = None) -> list[int]:
    """The card numbers of the recommended option (empty: no usable card)."""
    res = options(memory_gb, ram_gb, cards, vendor)
    return next((o["cards"] for o in res["options"] if o["recommended"]), [])
