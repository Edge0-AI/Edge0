"""The serving profile of a four-card AMD box: 4 x Radeon AI PRO R9700 (32 GB, gfx1201) running Qwen3.8-Flash-Next IQ3_S.

Measured on that box (docs/BATCHING.md, "Four R9700 cards: the serving profile"): the cards hold the whole model, so what limits
them is how many requests decode together and how the four stages are fed, not memory.  The profile sets

* 32 slots in 4 pipelined groups (one group per card), instead of the launcher's 4 requests at a time;
* the layer split 13,26,39: the output head costs about four layers on the last card, so it gets 9 layers and the others 13;
* a 2048-token prompt chunk: a smaller chunk reads long prompts faster beside the decodes of the other slots;
* the three batch switches STRATA_QSA_SLOTS, STRATA_BATCH_SELF_COMMIT and STRATA_SLOT_COPY_DIRECT, which change how the work is
  scheduled and copied, not what is computed (tools/batch_test.py: every slot decodes the tokens it decodes alone).

Recommend, never force: a setting the person made (the config, `--parallel`, `--layer-split`, an advanced flag, an environment
variable) always wins, the profile only fills what is not set, and `STRATA_FOUR_CARD_PROFILE=0` (or "four_card_profile": false in
the run config) turns it off.  Only this hardware class is touched; every other computer gets exactly what it got before.

Pure functions, no I/O (the callers pass the cards and the run config).
"""

from __future__ import annotations

import os

PARALLEL = 32                     # slots (requests decoding together)
GROUPS = 4                        # pipelined groups: one per card
LAYER_SPLIT = "13,26,39"          # Qwen3.8-Flash-Next has 48 layers: 13 / 13 / 13 / 9 (the output head sits on the last card)
PREFILL_CHUNK = "2048"
BATCH_ENV = {"STRATA_QSA_SLOTS": "1", "STRATA_BATCH_SELF_COMMIT": "1", "STRATA_SLOT_COPY_DIRECT": "1"}
PROMPT_CACHE_OFF = False          # --prompt-cache 0 is not part of the default: see docs/BATCHING.md
MIN_CARD_GB = 28.0                # a 32 GB card reports a little less than its name
CARDS = 4
OFF_VALUES = ("0", "off", "no", "false")


def disabled(cfg: dict | None = None) -> bool:
    """The person turned the profile off: STRATA_FOUR_CARD_PROFILE=0 in the environment, or "four_card_profile": false in the config."""
    if str(os.environ.get("STRATA_FOUR_CARD_PROFILE", "")).strip().lower() in OFF_VALUES:
        return True
    return bool(cfg) and cfg.get("four_card_profile") is False


def card_class(cards: list[dict]) -> bool:
    """Exactly four AMD cards of one kind, each with at least MIN_CARD_GB: the Radeon AI PRO R9700 (the 32 GB gfx12 card this was
    measured on)."""
    if len(cards) != CARDS:
        return False
    names = {str(c.get("name") or "") for c in cards}
    if len(names) != 1 or "R9700" not in next(iter(names)).upper().replace(" ", ""):
        return False
    return all(str(c.get("vendor") or "amd").lower() == "amd" and float(c.get("vram_gb") or 0.0) >= MIN_CARD_GB for c in cards)


def model_ok(cfg: dict) -> bool:
    """The measured model: Qwen3.8-Flash-Next at IQ3_S (48 layers; the pack that fits whole on four 32 GB cards)."""
    text = " ".join([str(cfg.get("model_name") or "")] + [str(a) for a in cfg.get("args") or []]).lower().replace("-", "_").replace(".", "_")
    return "qwen3_8" in text and "iq3_s" in text


def applies(cards: list[dict], cfg: dict) -> bool:
    args = [str(a) for a in cfg.get("args") or []]
    return (not disabled(cfg)) and card_class(cards) and cfg.get("backend", "hip") == "hip" and model_ok(cfg) \
        and "--peer-device" not in args and not cfg.get("replicas") and cfg.get("arch") != "deepseek41"


def apply(cards: list[dict], cfg: dict, *, parallel_given: bool = False) -> list[str]:
    """Fill the profile into the run config `cfg` (in place) where the person has not chosen; returns the plain lines to show.
    `parallel_given`: the person named a number of requests (--parallel): then only the split, the chunk and the switches are
    filled (the engine groups the slots itself)."""
    if not applies(cards, cfg):
        return []
    args = [str(a) for a in cfg.get("args") or []]
    env = dict(cfg.get("env") or {})
    notes: list[str] = []
    if str(cfg.get("layer_split") or "auto") == "auto" and "--layer-split" not in args:
        cfg["layer_split"] = LAYER_SPLIT
        notes.append(f"layer split {LAYER_SPLIT}: the last card also holds the output head, so it takes fewer layers")
    if "--prefill" not in args:
        args.extend(["--prefill", PREFILL_CHUNK])
    elif args[args.index("--prefill") + 1:args.index("--prefill") + 2] == ["auto"]:   # "auto" is what setup writes, not a choice
        args[args.index("--prefill") + 1] = PREFILL_CHUNK
    if PROMPT_CACHE_OFF and "--prompt-cache" not in args:
        args.extend(["--prompt-cache", "0"])
    for k, v in BATCH_ENV.items():
        if k not in os.environ:                                  # an environment variable of the person's own wins
            env.setdefault(k, v)
    if not parallel_given and "parallel" not in cfg and not {"--batch", "--slots"} & set(args):
        cfg["parallel"] = PARALLEL
        if "--batch-groups" not in args:
            args.extend(["--batch-groups", str(GROUPS)])
        notes.append(f"{PARALLEL} requests at once in {GROUPS} groups, one group per card (--parallel N changes it)")
    cfg["args"] = args
    cfg["env"] = env
    return notes
