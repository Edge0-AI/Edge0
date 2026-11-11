"""The replica serving profile (opt-in): on two or more GPUs, one engine per card behind the server instead of one layer split.

Measured on 4 x Radeon AI PRO R9700 (Qwen3.8-Flash-Next IQ3_S, 1k-token prompts, 128 tokens out, total tokens per second;
docs/BATCHING.md, "Replicas as a serving profile"):

* 2 cards: 2 engines of one card each reach 1.8 to 1.9 times what a layer split of the two cards does with `parallel` 8 or
  16 (116.6 against 63.8 tok/s at 8 requests, 126.9 against 65.7 at 16);
* 4 cards: 4 engines reach 1.8 to 1.9 times the shipped four-card profile (190 against 107 at 8, 220 against 116 at 16).

Nothing is lost for one user: a request alone runs on one card at the same speed (13 ms a token on one card, 13 ms on four).

The price is host RAM: every engine keeps its own copy of the experts.  The profile therefore turns on the shared expert
arena (serve/replicas.py: one memory object holds the experts for all engines, measured 61 GB for 2 engines and 66 GB for 4
instead of 109 and 211 GB) and chooses the number of engines the RAM allows.  Without the shared arena (Windows) it needs the
private copies to fit.

Recommend, never force: this profile is opt-in (`setup.py --replica-profile`, STRATA_REPLICA_PROFILE=1, or
"replica_profile": true in the run config; "replica_profile": false or STRATA_REPLICA_PROFILE=0 turn it off), and what the
person chose (the config's `replicas`, `--parallel`, `--layer-split`, `--peer-device`, an environment variable) always wins.
It only touches the measured model class; every other computer gets exactly what it got before.

Pure functions, no I/O (the callers pass the cards, the RAM and the run config).
"""

from __future__ import annotations

import os
import sys

# the experts' size in RAM per quant (setup.py MODELS "arena_gb"): the shared arena holds this once
ARENA_GB = {"iq3_s": 50.3, "iq3_xxs": 42.9, "q2_0": 34.0}
PER_REPLICA_GB = 8.0        # measured: about 7 GB of private RAM per engine beside the arena (2 engines: 61 GB in all, 47 of them shared)
HEADROOM_GB = 12.0          # what the OS, the server and the page cache keep
MIN_CARD_GB = 24.0          # measured on 32 GB cards; a smaller card is not claimed (an A4000 16 GB is measured separately)
MIN_REPLICAS = 2
SWITCHES = {"STRATA_QSA_SLOTS": "1", "STRATA_BATCH_SELF_COMMIT": "1", "STRATA_SLOT_COPY_DIRECT": "1",
            "STRATA_CKPT_POOL": "1", "STRATA_CKPT_POOL_READY": "4", "STRATA_SLOT_CKPT_MOVE": "1",
            "STRATA_SLOT_COPY_BATCHSYNC": "1"}
OFF_VALUES = ("0", "off", "no", "false")
ON_VALUES = ("1", "on", "yes", "true")


def requested(cfg: dict | None = None, environ=None) -> bool:
    """Opt-in, and an explicit no wins: config "replica_profile" true/false, else STRATA_REPLICA_PROFILE."""
    environ = os.environ if environ is None else environ
    if cfg and cfg.get("replica_profile") is not None:
        return cfg.get("replica_profile") is True
    return str(environ.get("STRATA_REPLICA_PROFILE", "")).strip().lower() in ON_VALUES


def quant_of(cfg: dict) -> str | None:
    """The measured model: Qwen3.8-Flash-Next at a quant whose experts size is known."""
    text = " ".join([str(cfg.get("model_name") or "")] + [str(a) for a in cfg.get("args") or []]).lower().replace("-", "_").replace(".", "_")
    if "qwen3_8" not in text:
        return None
    for q in ARENA_GB:
        if q in text:
            return q
    return None


def shared_arena_possible(platform: str | None = None) -> bool:
    return (sys.platform if platform is None else platform).startswith(("linux", "win"))


def replicas_for(n_cards: int, arena_gb: float, ram_gb: float, shared: bool) -> int:
    """How many one-card engines the RAM allows (at most one per card)."""
    if shared:
        room = ram_gb - HEADROOM_GB - arena_gb
        k = int(room // PER_REPLICA_GB) if room > 0 else 0
    else:
        k = int((ram_gb - HEADROOM_GB) // (arena_gb + PER_REPLICA_GB))
    return max(0, min(n_cards, k))


def applies(cards: list[dict], cfg: dict, ram_gb: float, *, platform: str | None = None, environ=None) -> tuple[bool, str]:
    """(True, "") when the profile would be applied; else (False, why not) - the reason is for the log, '' when nothing was asked."""
    if not requested(cfg, environ):
        return False, ""
    args = [str(a) for a in cfg.get("args") or []]
    if cfg.get("replicas") or cfg.get("arch") == "deepseek41" or "--peer-device" in args:
        return False, "replica profile: the config already names replicas, a peer device or another model family"
    if str(cfg.get("layer_split") or "auto") != "auto" or "--layer-split" in args:
        return False, "replica profile: you chose a layer split; kept"
    if len(cards) < MIN_REPLICAS:
        return False, "replica profile: needs two or more GPUs"
    q = quant_of(cfg)
    if q is None:
        return False, "replica profile: measured only for Qwen3.8-Flash-Next IQ3_S, IQ3_XXS and Q2_0"
    if any(float(c.get("vram_gb") or 0.0) < MIN_CARD_GB for c in cards):
        return False, f"replica profile: a card has under {MIN_CARD_GB:.0f} GB of VRAM (measured on 32 GB cards)"
    shared = shared_arena_possible(platform)
    k = replicas_for(len(cards), ARENA_GB[q], ram_gb, shared)
    if k < MIN_REPLICAS:
        need = HEADROOM_GB + ARENA_GB[q] + MIN_REPLICAS * PER_REPLICA_GB if shared else HEADROOM_GB + MIN_REPLICAS * (ARENA_GB[q] + PER_REPLICA_GB)
        return False, (f"replica profile: {ram_gb:.0f} GB of RAM; two engines need about {need:.0f} GB"
                       f"{'' if shared else ' (this OS cannot share the experts between engines)'}; the layer split stays")
    return True, ""


def apply(cards: list[dict], cfg: dict, ram_gb: float, *, parallel_given: bool = False, platform: str | None = None,
          environ=None) -> list[str]:
    """Fill the profile into `cfg` (in place) where the person has not chosen; returns the plain lines to show."""
    ok, why = applies(cards, cfg, ram_gb, platform=platform, environ=environ)
    if not ok:
        return [why] if why else []
    environ = os.environ if environ is None else environ
    q = quant_of(cfg)
    shared = shared_arena_possible(platform)
    k = replicas_for(len(cards), ARENA_GB[q], ram_gb, shared)
    gpus = cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [c.get("index", i) for i, c in enumerate(cards)]
    chosen = list(gpus)[:k]
    cfg["gpu"] = chosen
    cfg["replicas"] = [{"gpus": [g]} for g in chosen]
    if shared:
        cfg.setdefault("replica_shared_arena", True)
    env = dict(cfg.get("env") or {})
    for key, val in SWITCHES.items():
        if key not in environ:                              # an environment variable of the person's own wins
            env.setdefault(key, val)
    cfg["env"] = env
    notes = [f"{k} engines of one card each instead of one layer split of {len(cards)}: requests spread over them "
             f"(about 1.8 times the throughput at 8 to 16 requests at once; one request alone is as fast as before)"]
    if k < len(cards):
        notes.append(f"{len(cards) - k} card(s) left out: the RAM ({ram_gb:.0f} GB) holds {k} engines")
    notes.append("the experts are in RAM once for all engines (shared expert arena)" if shared else
                 "each engine keeps its own copy of the experts in RAM (this OS cannot share them)")
    if not parallel_given and "parallel" not in cfg and not {"--batch", "--slots"} & set(str(a) for a in cfg.get("args") or []):
        cfg["parallel"] = 8 if k == 2 else 4
        notes.append(f"{cfg['parallel']} requests at once on each engine (--parallel N changes it)")
    return notes
