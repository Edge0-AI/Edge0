"""Who chose the graphics cards of an installed model: the person, or setup's own pick.

Setup used to write "gpus_asked": true for every single-card run config, also when setup had picked the card itself (a DeepSeek run on a PC
with several AMD cards: setup took the first card, called it asked, and every later launch handed that "choice" back, so four cards ran
on one).  New configs say who chose in "gpus_by" ("person" or "setup").  A config without it is judged here:

* roles ("gpu_roles") and card lists of two or more are the person's (setup never wrote a list unless the cards were named or all used);
* a single card is setup's pick when the model is DeepSeek on AMD, the PC has more than one card that counts, and the card is the one setup
  would have taken (the first of the usable cards, largest memory first).  A person who named another card (``--gpu 2``) keeps it;
  one who named exactly the card setup would have taken cannot be told from setup's own pick and is moved to all cards once, with a plain
  line saying how to keep one card (``--gpu N``, remembered from then on).
"""
from __future__ import annotations

ASKED_BY_PERSON = "person"
PICKED_BY_SETUP = "setup"


def _is_deepseek_amd(cfg: dict) -> bool:
    return str(cfg.get("arch") or "") == "deepseek41" and str(cfg.get("backend") or "") == "hip"


def single_card(cfg: dict) -> int | None:
    g = cfg.get("gpu")
    if isinstance(g, bool) or g is None or g == "":
        return None
    if isinstance(g, (list, tuple)):
        return int(g[0]) if len(g) == 1 and str(g[0]).strip().lstrip("-").isdigit() else None
    return int(str(g).strip()) if str(g).strip().lstrip("-").isdigit() else None


def setup_pick_index(usable: list[dict]) -> int | None:
    """The card setup takes when nothing is named: the usable cards, largest memory first, then the lower number."""
    if not usable:
        return None
    first = min(usable, key=lambda c: (-round(float(c.get("vram_gb") or 0)), int(c.get("index", 0))))
    return int(first.get("index", 0))


def chosen_by_person(cfg: dict | None, usable: list[dict] | None = None) -> bool:
    """True when the cards in `cfg` are the person's choice (keep them); False when setup picked them (they may be widened).
    `usable`: the cards that count on this PC (strata_control.gpu_fit.usable_cards), needed only to judge a legacy single AMD card."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("gpu_roles"):
        return True
    by = cfg.get("gpus_by")
    if by in (ASKED_BY_PERSON, PICKED_BY_SETUP):
        return by == ASKED_BY_PERSON
    if not cfg.get("gpus_asked"):
        return False
    return not looks_like_setups_pick(cfg, usable)


def looks_like_setups_pick(cfg: dict, usable: list[dict] | None) -> bool:
    """A config from setup 0.2.8 and earlier: DeepSeek on AMD, one card, on a PC where more than one card counts, and the card setup would have taken."""
    if not _is_deepseek_amd(cfg) or cfg.get("gpus_by") or cfg.get("gpu_roles"):
        return False
    one = single_card(cfg)
    if one is None or not usable or len(usable) < 2:
        return False
    return one == setup_pick_index(usable)


def widened(cfg: dict, cards: list[int]) -> dict:
    """`cfg` with setup's own pick of `cards` (two or more): the card list and the marker that setup chose them; nothing else changes."""
    out = dict(cfg)
    out["gpu"] = list(cards) if len(cards) > 1 else cards[0]
    out["gpus_asked"] = True                           # as setup writes it for every card list; "gpus_by" says who chose
    out["gpus_by"] = PICKED_BY_SETUP
    out.pop("hip_ordinal", None)
    return out


def is_deepseek_amd(cfg: dict | None) -> bool:
    return isinstance(cfg, dict) and _is_deepseek_amd(cfg)


def remember(path, cards: list[int]) -> dict | None:
    """The person named `cards` for a DeepSeek model on AMD (`strata launch --gpu N`, `--gpus ...`, `setup --gpu N`): saved in the installed
    config as theirs, so no later start widens or moves it.  Returns the new config, or None when the file is not a DeepSeek AMD config
    (everything else keeps its own rules) or cannot be written.  Nothing else in the file changes."""
    import json
    import os
    from pathlib import Path
    p = Path(path)
    try:
        cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        if not _is_deepseek_amd(cfg) or not cards:
            return None
        cfg["gpu"] = list(cards) if len(cards) > 1 else int(cards[0])
        cfg["gpus_asked"] = True
        cfg["gpus_by"] = ASKED_BY_PERSON
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        return cfg
    except (OSError, ValueError, TypeError):
        return None
