"""--spec 2 for a run config on exactly two Intel Arc B-series (Battlemage) cards.

On two B-series cards the SYCL engine runs the verify windows of the two cards overlapped (--pipeline-windows 2, on by
default there).  Its answers are those of a width-2 draft window whatever --spec says: on 2x Arc Pro B65 the pipelined
answers equal a serial --spec 2 in 100 of 100 agent requests, while a serial --spec 4 differs from them in 3 of 100 (an end
token against a digit at a near-tie; a serial --spec 3 differs from --spec 4 in the same way).  So setup and the launcher
write `--spec 2` for such a config: the same answers as the engine gives anyway, about 20% faster than the old
`--spec 4` serial default (chat 73.9 against 60.7 tok/s), and no surprise between a serial and a pipelined start.

Only the generated default changes.  Single cards, Arc A-series (Alchemist) cards, three or more cards, a config with no
draft layer, and any config where the person set --spec (in "user_args", or a value other than setup's own 4 in "args") keep
what they have.  The card generation is the PCI device id from sysfs, the same id the engine's architecture enum comes from
(strata::intel_gpu_gen in sycl/include/strata/sycl_queue.hpp), never the marketing name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import user_args

SETUP_DEFAULT_SPEC = "4"      # what setup.py and the launcher write for every model
TWO_CARD_SPEC = "2"
# Battlemage: BMG-G21 (0xE20x: B580, B570, Pro B50) and BMG-G31 (0xE22x: Pro B70, B65, B60), driven by xe.  Alchemist (0x56A0..)
# and anything else (an integrated GPU, i915) is not.  Same range as sycl_queue.hpp's name-less fallback.
BATTLEMAGE_ID_RANGE = (0xE200, 0xE2FF)


def is_battlemage(driver: str, devid: str) -> bool:
    """True for an Arc B-series card: the xe driver and a PCI device id in the Battlemage range."""
    if driver != "xe":
        return False
    try:
        n = int(str(devid).lower().replace("0x", ""), 16)
    except ValueError:
        return False
    return BATTLEMAGE_ID_RANGE[0] <= n <= BATTLEMAGE_ID_RANGE[1]


def sysfs_cards(drm: str | os.PathLike = "/sys/class/drm") -> list[dict]:
    """The Intel discrete cards in sysfs card order (the order setup's `--gpus` indexes and Level Zero follows on the tested
    machines): [{"driver": "xe", "devid": "e222"}, ...].  Vendor 0x8086 under xe or i915; an i915 card with an id that is not an
    Alchemist one is an integrated GPU and is left out, as in sycl/setup_intel.py."""
    out = []
    for card in sorted(Path(drm).glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            devid = (dev / "device").read_text().strip().lower().replace("0x", "")
            driver = os.path.basename(os.path.realpath(dev / "driver")) if (dev / "driver").exists() else ""
        except OSError:
            continue
        if vendor != "0x8086" or driver not in ("xe", "i915"):
            continue
        if driver == "i915" and not devid.startswith("56"):
            continue
        out.append({"index": len(out), "driver": driver, "devid": devid})
    return out


def exactly_two_battlemage(cards: list[dict]) -> bool:
    return len(cards) == 2 and all(is_battlemage(c.get("driver", ""), c.get("devid", "")) for c in cards)


def _spec_group(args: list[str]) -> int:
    """Index of the value after --spec in `args`, or -1."""
    for i, a in enumerate(args):
        if a == "--spec" and i + 1 < len(args):
            return i + 1
    return -1


def plan(args: Any, user: Any, cards: list[dict], mtp_off: bool = False) -> tuple[list[str], str]:
    """(new args, note): `args` with --spec 4 turned into --spec 2 when the run is on exactly two Battlemage cards, and ""
    when nothing changed.  `user` is the config's "user_args"; `cards` the cards the run uses (exactly two to qualify);
    `mtp_off` is the config's "mtp": false.  `args` is not changed in place."""
    out = [str(a) for a in args] if isinstance(args, list) else []
    if not exactly_two_battlemage(cards) or mtp_off:
        return out, ""
    if "--mtp" not in out:                                  # the pipeline needs the draft layer
        return out, ""
    if any(user_args.flag_of(g) == "--spec" for g in user_args.groups(user)):
        return out, ""                                      # the person's own --spec wins
    i = _spec_group(out)
    if i < 0 or out[i] != SETUP_DEFAULT_SPEC:               # absent, or a value somebody chose
        return out, ""
    out[i] = TWO_CARD_SPEC
    return out, ("Two Arc B-series cards: --spec 2 instead of 4. With both cards working on neighbouring windows the engine "
                 "answers as a 2-token window anyway, and --spec 2 is about 20% faster here. To keep your own value, put "
                 "\"--spec\" and its number in the config's \"user_args\".")


def apply_for_launch(cfg: dict, sysfs: list[dict] | None = None) -> str:
    """The launcher's run config: the cards in cfg["gpu"] (a list of two numbers after a layer split) looked up in sysfs, then
    `apply_to_config`.  Nothing happens for a single card, another backend or a card list that sysfs cannot place."""
    if cfg.get("backend") != "sycl":
        return ""
    gpu = cfg.get("gpu")
    if not isinstance(gpu, (list, tuple)) or len(gpu) != 2:
        return ""
    try:
        have = sysfs if sysfs is not None else sysfs_cards()
        cards = [have[int(i)] for i in gpu if 0 <= int(i) < len(have)]
    except (TypeError, ValueError):
        return ""
    return apply_to_config(cfg, cards) if len(cards) == 2 else ""


def apply_to_config(cfg: dict, cards: list[dict]) -> str:
    """`plan` on a run config (in place): changes cfg["args"], and the copy in cfg["args_written"] so a later setup run does not
    take the change for the person's own edit.  Returns the note, "" when nothing changed."""
    args, note = plan(cfg.get("args"), cfg.get("user_args"), cards, mtp_off=cfg.get("mtp") is False)
    if not note:
        return ""
    cfg["args"] = args
    written = cfg.get(user_args.WRITTEN_KEY)
    if isinstance(written, list):
        w = [str(a) for a in written]
        i = _spec_group(w)
        if i >= 0 and w[i] == SETUP_DEFAULT_SPEC:
            w[i] = TWO_CARD_SPEC
            cfg[user_args.WRITTEN_KEY] = w
    return note
