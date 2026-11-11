"""The oldest NVIDIA driver the CUDA 12 engine package runs on, taken from the package's own metadata (#4247, #4188).

Setup used to say "the CUDA 12 engine works from driver 528 (525 on Linux)", a number copied from NVIDIA's minor-version-compatibility table,
while the signed feed's cuda12 packages (built with CUDA 12.9) carry `min_driver` 575.51 / 576.02 and `strata update` refuses them below
that: a PC on driver 556 was told its engine would work and then told the feed had no engine for it. The floor is now read from
the package that would run:

1. an installed cuda12 package of this platform (the components folder, setup's own engine-feed folder): its manifest's `min_driver`;
2. else the signed feed (best effort, a few seconds, never an error): the newest cuda12 package of this platform;
3. else FALLBACK, the figure the shipped packages carry today (packaging/ENGINE-COMPONENT.md).

Nothing here refuses anything: the callers say the number, and say where it came from.
"""
from __future__ import annotations

import threading
from pathlib import Path

#: What the cuda12 packages carry (packaging/ENGINE-COMPONENT.md: min_driver nvidia 575.51). Only used when no package or feed can be read.
FALLBACK = "575.51"
VARIANT = "cuda12"
FEED_SECONDS = 10.0


def major(version: str | None) -> int:
    """The leading number of a driver version ("576.02" -> 576); 0 when there is none."""
    try:
        return int(str(version or "").strip().split(".")[0])
    except ValueError:
        return 0


def _key(version: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(version).replace("-", ".").split("."))


def _floor_of(packages, platform: str, variant: str):
    """The `min_driver` (nvidia) of the newest package of `variant` for `platform`: items with .version, .platform, .backend, .variant, .min_driver."""
    best = None
    for item in packages:
        if item.backend != "cuda" or item.variant != variant or (platform and item.platform != platform):
            continue
        need = (item.min_driver or {}).get("nvidia")
        if not need or not major(need):
            continue
        if best is None or _key(item.version) > _key(best.version):
            best = item
    return None if best is None else str(best.min_driver["nvidia"])


def installed(roots, platform: str, variant: str = VARIANT) -> str | None:
    """The floor of a cuda12 package already on this PC, from its manifest (no network)."""
    from strata_control import engine_component as ec
    found = []
    for root in roots or ():
        try:
            components, _problems = ec.scan(Path(root))
        except Exception:                                # noqa: BLE001 - a folder that cannot be read has no answer
            continue
        found.extend(components)
    return _floor_of(found, platform, variant)


def from_feed(source: str, platform: str, variant: str = VARIANT, *, seconds: float = FEED_SECONDS) -> str | None:
    """The floor the signed feed gives a cuda12 package of this platform; None when the feed cannot be read in `seconds` (offline, no key,
    a feed with no such package). Runs the read on a thread so a hung connection never holds setup up."""
    answer: list = []

    def read() -> None:
        try:
            from strata_control import feed as feedmod
            from strata_control import signing
            loaded = feedmod.fetch(source, signing.load_trusted_keys())
            answer.append(_floor_of(loaded.engine, platform, variant))
        except Exception:                                # noqa: BLE001 - the floor is a convenience, never a reason to stop
            answer.append(None)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(seconds)
    return answer[0] if answer else None


def cuda12_floor(platform: str, *, roots=(), feed_source: str | None = None) -> tuple[str, str]:
    """(floor, where it came from): "576.02", "as the signed update feed says"."""
    got = installed(roots, platform)
    if got:
        return got, "as the installed engine package says"
    if feed_source:
        got = from_feed(feed_source, platform)
        if got:
            return got, "as the signed update feed says"
    return FALLBACK, "the figure the published CUDA 12 packages carry"
