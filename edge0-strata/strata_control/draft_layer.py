"""The draft layer (MTP) switch of a model's run config.

``"mtp": false`` in a model's run config (strata-<model>.json) starts the engine without the draft layer.  The key is the
person's: setup keeps it when it writes the file again (setup.carry_over), the app's Models settings and `strata launch --mtp off`
write it, and every start applies it (serve/server.py engine_args), so a setup run that puts ``--mtp <folder>`` back into
the engine arguments cannot bring the draft layer back.

What the draft layer costs, in plain words: about 1 GB of graphics memory.  Its weights are the files in the model data
folder's ``mtp/rt`` (0.7 GB of experts and 0.1 GB of dense weights, 786 MiB), the engine sets aside another 218 MiB for its
draft head on a 12 GB card ("expert cache auto: ... +218 MiB for the draft head"), and its own conversation memory grows with
the context (with a pinned copy of that in RAM when the context streams).  The expert cache gets what is left, so every GB here
is a GB less of cached experts.  It buys speed (about twice as many words per second on a
card that holds the model) and costs that room in the expert cache, so on a small card or a PC with little RAM the saving can be
worth more than the speed.

Without it the engine still guesses ahead from the text already in the conversation (``--spec`` with prompt lookup), which is
why ``--spec`` stays and only the draft layer's own options go: ``--mtp``, ``--spec-min-p`` (the draft layer's confidence
threshold) and ``--batch-mtp`` (the draft layer per request).
"""

from __future__ import annotations

from typing import Any

#: engine options that exist only because of the draft layer: flag -> takes a value
MTP_FLAGS = {"--mtp": True, "--spec-min-p": True, "--batch-mtp": False}


def is_off(cfg: dict[str, Any] | None) -> bool:
    """True when the run config says the draft layer is off (only an explicit ``false``; absent or true: on)."""
    return isinstance(cfg, dict) and cfg.get("mtp") is False


def strip(args: list[Any]) -> tuple[list[str], list[str]]:
    """(the arguments without the draft layer's options, the flags that were removed)."""
    out: list[str] = []
    removed: list[str] = []
    items = [str(a) for a in args]
    i = 0
    while i < len(items):
        a = items[i]
        if a in MTP_FLAGS:
            removed.append(a)
            i += 2 if MTP_FLAGS[a] and i + 1 < len(items) and not items[i + 1].startswith("--") else 1
            continue
        out.append(a)
        i += 1
    return out, removed


def apply(cfg: dict[str, Any]) -> list[str]:
    """Takes the draft layer's options out of ``cfg["args"]`` when the config has it off; returns the flags removed."""
    if not is_off(cfg) or not isinstance(cfg.get("args"), list):
        return []
    cfg["args"], removed = strip(cfg["args"])
    return removed
