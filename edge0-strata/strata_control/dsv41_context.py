"""DeepSeek-V4.1-Flash's context sizes: 8K up to 1M (the model reads 1,048,576 positions natively).

Project Maya v1.1.28 (#442, @needmorevram, MIT) offered 512K and 1M in its setup and dashboard; this is the same rule in Strata.

- The choices are 8K, 32K, 64K, 128K, 256K, 512K and 1M, limited to the pack's own limit when it is known (``max_position_embeddings`` in the pack's
  ds4.txt, written by tools/dsv41/pack_gguf.py from the GGUF's context_length).  Without a pack file the Terminal and setup still offer up to 1M (the
  model's own length); the app offers up to 256K until the pack says more.
- Past 256K nothing was tested by us: the owner's decision is to offer it and let users try it.  The words for that are NOTE.
- A size past the pack's limit (or past 1,048,576 when no pack says) cannot run: that is an error, the only one.
- The engine takes whatever ``--max-context`` it is given (src/dsv41/ds4_model.cu allocates for it, no clamp); the rope settings come from the pack.

apps/host/server/dsv41-context.mjs says the same for the app; tests/terminal/test_dsv41_context.py compares the two.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

CHOICES: tuple[int, ...] = (8192, 32768, 65536, 131072, 262144, 524288, 1048576)
TESTED = 262144                      # the largest size that is not flagged as untested
NATIVE = 1048576                     # the model's own length: the most any pack can say
NOTE = ("Not tested yet. A context this long takes graphics memory from the model and reads long prompts slowly.")

_LIMIT = re.compile(r"^max_position_embeddings\s+(\d+)\s*$", re.MULTILINE)


def read_limit(pack: str | Path | None) -> int | None:
    """The pack's own limit from its ds4.txt; None when there is no pack, no file or no such line (the caller then behaves as before)."""
    if not pack:
        return None
    try:
        text = (Path(pack) / "ds4.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _LIMIT.search(text)
    if not m:
        return None
    value = int(m.group(1))
    return value if value >= 1024 else None


def limit_of_config(cfg: Mapping[str, Any] | None) -> int | None:
    """The limit of the pack a DeepSeek run config names (``--pack DIR``)."""
    args = [str(a) for a in (cfg or {}).get("args") or []]
    if "--pack" in args and args.index("--pack") + 1 < len(args):
        return read_limit(args[args.index("--pack") + 1])
    return None


def choices(limit: int | None = None, *, unknown_cap: int = NATIVE) -> list[int]:
    """The sizes to offer: up to the pack's limit when known, else up to ``unknown_cap`` (the Terminal and setup: the model's 1M; the app: TESTED)."""
    cap = limit if limit else unknown_cap
    return [c for c in CHOICES if c <= cap]


def problem(context: int, limit: int | None = None) -> str:
    """"" when the size can run; otherwise one plain sentence (the only refusal: a size the model cannot address)."""
    cap = limit or NATIVE
    if context < 1024:
        return f"A context of {context} tokens is too small to be useful; the smallest size offered is 8K."
    if context > cap:
        what = "this model's pack reads" if limit else "DeepSeek-V4.1-Flash reads"
        return f"{label(context)} is more than {what} ({label(cap)}). Choose {label(cap)} or less."
    return ""


def note(context: int | None) -> str:
    """The plain words past 256K, "" otherwise."""
    return NOTE if context and context > TESTED else ""


def label(tokens: int) -> str:
    if tokens % NATIVE == 0:
        return f"{tokens // NATIVE}M"
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return f"{tokens:,} tokens"


def labels(sizes: Iterable[int]) -> list[str]:
    return [label(c) for c in sizes]
