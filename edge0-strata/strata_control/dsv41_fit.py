"""Where DeepSeek-V4.1-Flash goes on a computer: the real numbers of the model and its engine, in one place.

The model is 40 layers of 384 routed experts (top 6) beside dense attention weights.  Unsloth's UD-Q8_K_XL (the
file Strata runs, pinned in setup.py) is 508.4 GB in nine shards:

  routed experts, MXFP4      288.8 GB   15,360 experts of 18.8 MB (40 layers x 384)
  dense weights, BF16 / F32   16.8 GB   attention, router, shared experts, the head (the embedding table stays on the CPU)
  the n-gram table (Engram)  202.8 GB   MXFP8, shard 8: read from the SSD, a few rows a token, never held in memory

so the memory the model needs is the experts and the dense weights, 305.6 GB, with the n-gram table left out.  strata-dsv41
(src/dsv41) puts the dense weights and as many experts as the cards hold in graphics memory, one block of layers on each
card, and computes the rest of the experts on the CPU, reading them in place from the model's files (the page cache).

Measured (r9700, 4 x Radeon AI PRO R9700 32 GB, 251 GB RAM, 2026-10-20, f-dsv41-port 01389205e): per card 3.4-4.7 GB of dense
weights, 28.5-29.8 GB of experts (1,505-1,575 of its 3,840), ~0.6 GB free; 6,197 experts (116.5 GB) in graphics memory and
9,163 (172.3 GB) on the CPU lane; --tp 2: prompts 143-181 tokens a second (3K prompts), answers 25-31 tokens a second.  Pure functions, no I/O:
setup.py, tools/strata_mcp.py and strata_cli/launch import it.  Recommend, never force: nothing here refuses a computer.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from strata_control import gpu_fit

GB = 1e9
GIB = 2 ** 30
MAIN_BYTES = 508_376_763_197          # the nine shards of UD-Q8_K_XL (setup.py DSV41_FILES)
ENGRAM_BYTES = 202_758_032_592        # shard 8: the n-gram table
EXPERT_BYTES = 288_777_830_400        # MXFP4: 543,581,798,400 values x 17/32 bytes
LAYERS = 40
EXPERTS_PER_LAYER = 384
EXPERTS = LAYERS * EXPERTS_PER_LAYER  # 15,360
EXPERT_GB = EXPERT_BYTES / GB / EXPERTS     # 18.8 MB each
WEIGHTS_GB = (MAIN_BYTES - ENGRAM_BYTES) / GB            # 305.6
EXPERTS_GB = EXPERT_BYTES / GB                           # 288.8
DENSE_GB = WEIGHTS_GB - EXPERTS_GB                       # 16.8
DENSE_ON_GPU_GB = 15.5                # measured: 3.72 + 3.69 + 3.40 + 4.70 (the embedding table, ~1.3 GB, stays on the CPU)
ENGRAM_GB = ENGRAM_BYTES / GB                            # 202.8
DOWNLOAD_GB = MAIN_BYTES / GB                            # 508.4
RESERVE_GB = 1.15                     # per card: the 600 MiB the engine leaves free (--reserve-mib) and ~0.5 GB of its buffers
                                      # (measured: dense weights + experts + free = 33.8 of an R9700's 34.4 GB)
REST_RAM_GB = 9.0                     # RAM the engine and the system keep besides the experts ("beside 9 GB for the rest")
SPARE_FITS_GB = 4.0                   # RAM left beyond the CPU lane's experts below which "fits" becomes "tight"
DISK_HEADROOM_GB = 6.0                # the pack folder (hash tables, tokenizer) and room to work
SMALLEST_CARD_GB = 8.0                # a card under this adds little: the dense weights alone are ~4 GB of it per card of four
MEASURED = ("4 x Radeon AI PRO R9700 (32 GB) with 251 GB of RAM: prompts of 3,000 tokens read at 143-181 tokens a second, answers "
            "at 25-31 tokens a second (--tp 2); one answer at a time")


# The two official IST-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF, setup.py DSV41_ISTA_FILES): 4 shards each (the experts as IQ1_M / IQ2 / IQ3 types, the
# dense weights Q8_0 / Q5_K), plus ONE Engram (110.6 GB, IQ4_NL, shared by both sets, stays on the SSD) and the DSpark drafter (8.0 GB).
# Measured 2026-10-23 (f-dsv41-rel): quality against the original model (final files, 4,995 positions: ppl +15.5% for 2.00 [code +8.6%, agent +5.3%], +7.3% for 2.25 [code +3.2%, agent +2.2%]
# on 2.25); speed on the owner PC (RTX 5070 12 GB, 64 GB RAM, 3.6 GB/s NVMe): answers ~4.7 tokens a second, prompts 121 tokens a second at 4K and
# 231 tokens a second at 8K; RTX 3060 12 GB with 46 GB of RAM (the engine held to 36 GB): answers ~7.2 tokens a second on 2.00; 4 x Radeon AI PRO
# R9700: answers ~31 tokens a second, prompts 330-373 tokens a second at 8K.
ISTA_ENGRAM_BYTES = 110_595_290_816
ISTA_DSPARK_BYTES = 7_976_914_240
# (the six-shard layout of the same sets, setup.py DSV41_ISTA6_FILES, is 74 KB larger: 2.00 138,238,967,104, 2.25 155,225,898,368, 1.75 120,721,194,336 bytes; the figures below are the pinned four-shard ones)
ISTA_META_BYTES = 5_780_589          # the first shard of Unsloth's UD-Q8_K_XL: only the metadata --engram-meta reads
ISTA_SETS = {
    "IST-DAS-2.00bpw": {"title": "IST-DAS 2.00 bpw", "set_bytes": 138_238_893_056, "experts_gb": 132.07, "profile": "profile-2.00.bin",
                        "ppl": "15.5%", "ram_ok_gb": 40.0, "ram_great_gb": 160.0, "disk_total": 257.0,
                        "quality": "about 15.5% above the original model's perplexity (code and agent work about 5-9% above)"},
    # IQ1_M (1.75 bpw): 120,721,120,256 bytes of shards; experts = the shards minus the 6.16 GB of dense weights.  RAM class from the experts' size against
    # the 2.00 set's measured 40 GB (not measured on a computer yet); quality: the H100's 4,995-position score (KL 0.217, ppl 3.80, +21%; code +8.9%, agent +10.2%, English prose +53%).
    "IST-DAS-1.75bpw": {"title": "IST-DAS 1.75 bpw", "set_bytes": 120_721_120_256, "experts_gb": 114.56, "profile": "profile-2.00.bin",
                        "ppl": "21%", "ram_ok_gb": 34.0, "disk_total": 239.0,
                        "quality": "about 21% above the original model's perplexity: the smallest, fine for code and agent work, noticeably weaker "
                                   "on plain prose; pick 2.00 if it fits"},
    "IST-DAS-2.25bpw": {"title": "IST-DAS 2.25 bpw", "set_bytes": 155_225_824_256, "experts_gb": 149.06, "profile": "profile-2.25.bin",
                        "ppl": "7.3%", "ram_ok_gb": 44.0, "ram_great_gb": 176.0, "disk_total": 274.0,
                        "quality": "about 7.3% above the original model's perplexity (code and agent work about 2-3% above), "
                                   "the better of the two, 17 GB more disk"},
}
# Unsloth's UD-IQ1_S / UD-IQ1_M (2026-10-30, unsloth/DeepSeek-V4.1-Flash-1017-GGUF; setup.py DSV41_UD_FILES): shard 1 = metadata, shards 2-4 = the model, shard 5 = the Engram
# (the same IQ4_NL data as IST-DAS's file, so it is shared with them).  set_bytes = shards 1-4; "meta_bytes": 0 (their own shard 1 is the metadata).  The dense weights are
# a little bigger than IST-DAS's (attention in Q6_K): 7.2 and 7.6 GB.  Quality is not measured on our side yet.
ISTA_SETS["UD-IQ1_S"] = {"title": "Unsloth UD-IQ1_S", "set_bytes": 122_384_317_828, "experts_gb": 115.23, "profile": "profile-2.00.bin", "meta_bytes": 0,
                         "ppl": "?", "ram_ok_gb": 36.0, "disk_total": 241.0,
                         "quality": "Unsloth's smallest dynamic version, the size of IST-DAS 1.75 bpw; not measured here yet"}
ISTA_SETS["UD-IQ1_M"] = {"title": "Unsloth UD-IQ1_M", "set_bytes": 139_809_056_132, "experts_gb": 132.21, "profile": "profile-2.00.bin", "meta_bytes": 0,
                         "ppl": "?", "ram_ok_gb": 40.0, "disk_total": 258.0,
                         "quality": "Unsloth's next dynamic version, the size of IST-DAS 2.00 bpw; not measured here yet"}
ISTA_DENSE_GB = 6.16                  # the dense weights in the shards (attention, router, shared experts, the head; the embedding stays on the CPU)
ISTA_DENSE_ON_GPU_GB = 4.5            # what a card holds of them with a 12 GB card (measured on the 5070 / 3060: the engine places the rest itself)
ISTA_MEASURED = ("an RTX 5070 (12 GB) with 64 GB of RAM and a 3.6 GB/s NVMe drive: answers about 4.7 tokens a second, a 4,000-token prompt "
                 "read at 121 tokens a second; an RTX 3060 (12 GB) with 46 GB of RAM: about 7.2 tokens a second; four Radeon AI PRO R9700 cards: "
                 "about 31 tokens a second")
# A unified-memory APU (Ryzen AI Max "Strix Halo", gfx1151, Windows): the graphics chip and the CPU share one pool of RAM, so the
# engine's expert tiers are one pool too.  The measured 64 GB box: 2.00 bpw, Faster (5 experts), DSpark auto, Adrenalin 26.10.1.
UMA_RESERVE_GB = 12.0                 # kept for Windows, the desktop and the engine's buffers (the engine's own RAM store margin)
UMA_MEASURED = ("a Ryzen AI Max+ 395 (Radeon 8060S, 128 GB unified: a 64 GB BIOS carve-out plus 64 GB seen by Windows) on Windows: answers "
                "14-16 tokens a second on code in the default Faster mode (5 experts) and 10-11 on chat; a 4,000-token prompt 77-81 tokens a "
                "second, 8,000 tokens 108-111")
# A DGX Spark (GB10, Linux, 128 GB unified, no BIOS carve-out): the same one-pool plan.  Measured with the 2.00 bpw set, six experts per word.
UMA_MEASURED_GB10 = ("an NVIDIA DGX Spark (GB10, 128 GB unified) on Linux with six experts per word: about 17 tokens a second on a first short "
                     "chat and 20-23 once warm; a 1,700-token prompt read at 213-237 tokens a second, a 500-token one at 76")


def variant(model: str | None) -> dict | None:
    """The official-set entry for a model name (case-insensitive), None for UD-Q8_K_XL and anything else."""
    if not model:
        return None
    return next((v for k, v in ISTA_SETS.items() if k.lower() == str(model).lower()), None)


#: The AMD cards the DeepSeek engine package holds code for (packaging/build_hip_linux.sh: strata-dsv41 ships in the rx9070 build, gfx1200 and
#: gfx1201).  setup passes the installed package's own list when it has one.
DSV41_HIP_ARCHS = ("gfx1200", "gfx1201")


def _on_windows() -> bool:
    return os.name == "nt"


def run_cards(cards: list[dict] | None, vendor: str | None = None, *, windows: bool | None = None,
              covered=None, named: bool = False) -> tuple[list[dict], list[dict]]:
    """The cards one DeepSeek run uses when the person named none: (cards, left out with 'why_ignored'), in card order.  The ONE rule that
    `strata launch --plan`, setup and the app share, so the plan says what the run does.
    Every card that counts (gpu_fit.usable_cards: no integrated chip beside a card with memory of its own, no card under 8 GB).  AMD:
    on Windows one card only (several AMD cards sharing a model is Linux-only, setup.py), and only cards the engine package holds code for
    (`covered`, default DSV41_HIP_ARCHS): a PC with an RX 9070 and an RX 7900 uses the RX 9070 and says why the other is left out.
    `named`: the person named these cards: they are used as named (setup warns about one the package has no code for)."""
    mine = gpu_fit.for_vendor(list(cards or []), vendor)[0] if vendor else list(cards or [])
    keep, ignored = gpu_fit.usable_cards(mine)
    small = [c for c in keep if float(c.get("vram_gb") or 0) < SMALLEST_CARD_GB]
    keep = [c for c in keep if float(c.get("vram_gb") or 0) >= SMALLEST_CARD_GB]
    ignored = list(ignored) + [{**c, "why_ignored": f"only {float(c.get('vram_gb') or 0):.0f} GB of graphics memory"} for c in small]
    if not named and len(keep) > 1 and all(str(c.get("vendor") or "").lower() == "nvidia" for c in keep):
        # NVIDIA: setup's role rule (gpu_roles.recommend): a card of 8 GB or less starts Off beside a bigger one, so the plan counts what the run uses
        # (an AMD card of 8 GB counts in both the plan and the run)
        from strata_control import gpu_roles
        if any(round(float(c.get("vram_gb") or 0)) > gpu_roles.OFF_AT_OR_BELOW_GB for c in keep):
            off = [c for c in keep if round(float(c.get("vram_gb") or 0)) <= gpu_roles.OFF_AT_OR_BELOW_GB]
            ignored += [{**c, "why_ignored": f"a card with {gpu_roles.OFF_AT_OR_BELOW_GB:.0f} GB or less starts Off beside a bigger card (setup does the same)"}
                        for c in off]
            keep = [c for c in keep if c not in off]
    amd = not named and bool(keep) and all(str(c.get("vendor") or vendor or "").lower() == "amd" for c in keep)
    if amd:
        win = _on_windows() if windows is None else windows
        # the list of architectures: the package's own (`covered`), else the Linux package's (gfx1200, gfx1201).  Windows ships a different
        # DeepSeek package (hip-rdna3), so without a list of its own nothing is filtered there.
        archs = tuple(covered) if covered else (() if win else DSV41_HIP_ARCHS)
        if archs:
            known = [c for c in keep if c.get("arch")]
            ok = [c for c in keep if not c.get("arch") or c["arch"] in archs]
            if known and ok and len(ok) < len(keep):
                ignored += [{**c, "why_ignored": f"the DeepSeek engine has no code for {c['arch']} (it runs on {', '.join(archs)})"}
                            for c in keep if c not in ok]
                keep = ok
        if win and len(keep) > 1:
            # the card setup takes (setup.py amd_rank): a card with memory of its own before an APU, then the most memory, then the lower number
            first = min(keep, key=lambda c: (bool(c.get("uma") or c.get("integrated")), -round(float(c.get("vram_gb") or 0)), c.get("index", 0)))
            ignored += [{**c, "why_ignored": "on Windows a model runs on one AMD card"} for c in keep if c is not first]
            keep = [first]
    keep = sorted(keep, key=lambda c: c.get("index", 0))
    return keep, ignored



def uma_card(cards: list[dict] | None) -> dict | None:
    """The integrated Radeon a run uses, when the graphics memory is the PC's RAM (Ryzen AI Max, gfx1151): setup marks it `uma` (and
    `integrated`); a discrete card never is.  None on every other computer."""
    return next((c for c in cards or [] if c.get("uma") or c.get("integrated")), None)


def uma_total_gb(card: dict, ram_gb: float | None) -> float:
    """The one pool of a unified-memory PC, counted once: the RAM Windows sees plus the BIOS carve-out the GPU owns (the "shared"
    part of the GPU's figure is that same RAM).  On the owner's Strix Halo: 64 GB + a 64 GB carve-out = 128 GB.  Without the carve-out
    in the card's data, the GPU's figure less its shared part, or the larger of the two, is the best that can be said."""
    ram = float(ram_gb or 0.0)
    dedicated = float(card.get("dedicated_gb") or 0.0)
    if dedicated > 0:
        return ram + dedicated
    vram = float(card.get("vram_gb") or 0.0)
    shared = float(card.get("shared_gb") or 0.0)
    if vram > 0 and shared > 0:
        return ram + max(0.0, vram - shared)
    return max(ram, vram)


def variant_key(model: str | None) -> str | None:
    return next((k for k in ISTA_SETS if k.lower() == str(model or "").lower()), None)


def ista_download_bytes(model: str, have_engram: bool = False) -> int:
    v = variant(model)
    return v["set_bytes"] + ISTA_DSPARK_BYTES + v.get("meta_bytes", ISTA_META_BYTES) + (0 if have_engram else ISTA_ENGRAM_BYTES)


def recommend(ram_gb: float | None, disk_free_gb: float | None = None) -> str:
    """The official set that suits this computer: 2.00 bpw, unless its memory (under the measured 40 GB of RAM) or its disk (under 2.00's need but
    enough for the smaller one) make the 1.75 bpw set the better start.  Only a recommendation: every set can still be chosen."""
    big, small = "IST-DAS-2.00bpw", "IST-DAS-1.75bpw"
    if ram_gb is not None and float(ram_gb) < ISTA_SETS[big]["ram_ok_gb"]:
        return small
    if disk_free_gb is not None:
        need_big = ista_download_bytes(big) / GB + DISK_HEADROOM_GB
        need_small = ista_download_bytes(small) / GB + DISK_HEADROOM_GB
        if float(disk_free_gb) < need_big and float(disk_free_gb) >= need_small:
            return small
    return big


def _round1(x: float) -> float:
    return round(float(x) * 10) / 10


def _cap_gb(vram_gb: float) -> float:
    """Decimal GB a card of `vram_gb` (as the catalogs state it, GiB: 32 for an R9700) has."""
    return float(vram_gb) * GIB / GB


def split_layers(weights: list[float], layers: int = LAYERS) -> list[tuple[int, int]]:
    """The layer ranges (first, last) of consecutive cards, in proportion to `weights` (each card at least one layer).
    The engine balances its own split; on cards of one size this is what it picked (0-9, 10-19, 20-29, 30-39)."""
    n = len(weights)
    if n <= 1:
        return [(0, layers - 1)]
    total = float(sum(weights)) or 1.0
    raw = [w / total * layers for w in weights]
    counts = [max(1, int(r)) for r in raw]
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    k = 0
    while sum(counts) < layers:
        counts[order[k % n]] += 1
        k += 1
    while sum(counts) > layers:
        i = max(range(n), key=lambda j: counts[j])
        counts[i] -= 1
    out, first = [], 0
    for c in counts:
        out.append((first, first + c - 1))
        first += c
    return out


def tp_for(n_cards: int, cards: list[dict] | None = None) -> int:
    """--tp for this many cards.  Pairs of cards sharing each layer's matrices (--tp 2, the engine's default for two or more cards) were
    measured 2x faster for prompts than a plain layer split on the first rig; on four AMD Radeon AI PRO R9700 cards the plain layer split
    (--tp 1) is faster for both: 8K prompt 370 against 196 tokens a second and 32 against 29 for answers (2026-10-23, r9700), so the IST-DAS plans give four
    or more AMD cards 1 (cards passed); the Unsloth plan is unmeasured and keeps 2.  Two and three cards keep 2 (two R9700: prompts 220 against 140, answers slower; not changed).
    (The engine's own check refuses 4 - 576 is not a multiple of 128 - and drops to 2.)"""
    if n_cards >= 4 and cards and all(str(c.get("vendor") or "").lower() == "amd" for c in cards):
        return 1
    return 2 if n_cards >= 2 else 1


def _tp_args(n_cards: int, tp: int) -> list[str]:
    """The engine's --tp when it is not the engine's own pick: it takes 2 with two or more cards when told nothing, so 1 is written out."""
    return ["--tp", str(tp)] if tp > 1 or n_cards >= 2 else []


def _plan_ista(model: str, cards: list[dict] | None, ram_gb: float | None, disk_free_gb: float | None, context: int,
               vendor: str | None, named: bool = False) -> dict[str, Any]:
    """plan() for an official IST-DAS set (ISTA_SETS): the same keys, the measured speeds in plain words.  RAM classes: with RAM for every
    expert the CPU lane keeps ("fits"), with 40 GB or more the engine's RAM store keeps the hot experts and reads the rest from the SSD
    ("tight": measured 4.7 tokens a second on 64 GB, 7.2 on 46 GB), under 40 GB it is untested and slower ("spills").  Never a refusal."""
    v = variant(model)
    keep, ignored = run_cards(cards, vendor, named=named)
    n = len(keep)
    expert_gb = v["experts_gb"] / EXPERTS
    set_gb = v["set_bytes"] / GB
    download = ista_download_bytes(model) / GB
    out: dict[str, Any] = {"variant": model, "cards": [], "experts_total": EXPERTS, "experts_in_vram": 0, "vram_experts_gb": 0.0,
                           "cpu_experts_gb": v["experts_gb"], "weights_gb": _round1(set_gb), "experts_gb": _round1(v["experts_gb"]),
                           "dense_gb": _round1(ISTA_DENSE_GB), "engram_gb": _round1(ISTA_ENGRAM_BYTES / GB), "download_gb": _round1(download),
                           "tp": tp_for(n, keep), "args": [], "measured": ISTA_MEASURED, "lines": [], "warnings": [], "ram_gb": ram_gb,
                           "context": context, "disk_total_gb": v["disk_total"]}
    need_disk = download + DISK_HEADROOM_GB
    out["disk_need_gb"] = _round1(need_disk)
    out["disk_ok"] = None if disk_free_gb is None else disk_free_gb >= need_disk
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash, {v['title']}: a {download:.0f} GB download ({set_gb:.0f} GB of model, {ISTA_ENGRAM_BYTES / GB:.0f} GB "
                 f"n-gram table, {ISTA_DSPARK_BYTES / GB:.0f} GB drafter). Quality: {v['quality']}.")
    lines.append("The n-gram table stays on the SSD and is read a few rows for each word: use an NVMe drive for the model files (a slow "
                 "disk makes every word slower). The same table serves every IST-DAS set, so installing another one downloads it once.")
    if n == 0:
        out["verdict"] = "no_card"
        warns.append("No graphics card Strata can use was found: this model needs one or more cards (8 GB or more; 12 GB is what it was "
                     "measured with).")
    else:
        per_dense = ISTA_DENSE_ON_GPU_GB / n
        cap = [max(0.0, _cap_gb(float(c.get("vram_gb") or 0)) - per_dense - RESERVE_GB) for c in keep]
        vram_exp = min(v["experts_gb"], sum(cap))
        ranges = split_layers([float(c.get("vram_gb") or 0) for c in keep])
        in_vram = 0
        for c, room, (a, b) in zip(keep, cap, ranges):
            gb = min(room, vram_exp * (room / sum(cap))) if sum(cap) else 0.0
            ne = min(int(gb / expert_gb), (b - a + 1) * EXPERTS_PER_LAYER)
            in_vram += ne
            out["cards"].append({"index": c.get("index"), "name": c.get("name"), "vram_gb": float(c.get("vram_gb") or 0), "layers": [a, b],
                                 "dense_gb": _round1(per_dense), "experts_gb": _round1(ne * expert_gb), "experts": ne})
        out["experts_in_vram"] = in_vram
        out["vram_experts_gb"] = _round1(in_vram * expert_gb)
        out["cpu_experts_gb"] = _round1(v["experts_gb"] - in_vram * expert_gb)
        out["tp"] = tp_for(n, keep)
        out["args"] = _tp_args(n, out["tp"])
        names = " + ".join(f"{c['name'] or 'card ' + str(c['index'])} ({c['vram_gb']:.0f} GB)" for c in out["cards"])
        lines.append(f"Graphics memory: {names}. About {in_vram:,} of the {EXPERTS:,} experts ({out['vram_experts_gb']:.0f} GB) stay on "
                     f"the card{'s' if n > 1 else ''}; the other {EXPERTS - in_vram:,} ({out['cpu_experts_gb']:.0f} GB) are computed on the "
                     "CPU from RAM, and the ones RAM cannot hold are read from the SSD when a word needs them.")
        if n >= 2 and out["tp"] == 2:
            lines.append("Two cards share each layer's matrices (--tp 2): measured about twice as fast at reading prompts.")
        if ram_gb:
            all_cached = out["cpu_experts_gb"] + REST_RAM_GB
            out["ram_needed_gb"] = _round1(all_cached)
            out["spill_gb"] = _round1(max(0.0, all_cached - float(ram_gb)))
            out["spare_ram_gb"] = _round1(max(0.0, float(ram_gb) - all_cached))
            if float(ram_gb) >= all_cached:
                out["verdict"] = "fits"
                lines.append(f"That fits this computer: {ram_gb:.0f} GB of RAM keeps all {out['cpu_experts_gb']:.0f} GB of the CPU's experts cached.")
            elif float(ram_gb) >= v["ram_ok_gb"]:
                out["verdict"] = "tight"
                lines.append(f"This computer has {ram_gb:.0f} GB of RAM: Strata keeps the experts you use most in RAM (it sizes this itself) and "
                             "reads the rest from the SSD. Measured: about 4.7 tokens a second on 64 GB of RAM and a 12 GB card, about 7 on "
                             "46 GB with a faster drive. More RAM or a faster NVMe drive makes it quicker.")
            else:
                out["verdict"] = "spills"
                warns.append(f"This computer has {ram_gb:.0f} GB of RAM and this model was measured with 46 GB and 64 GB. With less, most "
                             "experts come from the SSD while it answers: it works but is slow (a word can take seconds). More RAM fixes "
                             "it. It is still allowed.")
        else:
            out["verdict"] = "unknown"
        short = [c for c in keep if _cap_gb(float(c.get("vram_gb") or 0)) < per_dense + RESERVE_GB + 1.0]
        if short:
            warns.append(f"The dense weights need ~{per_dense:.0f} GB of graphics memory and {short[0].get('name') or 'a card'} "
                         f"({float(short[0].get('vram_gb') or 0):.0f} GB) has hardly that: the engine may stop at the start. It is still allowed.")
        if ignored:
            lines.append("Not used: " + "; ".join(f"{c.get('name') or 'a card'} ({c.get('why_ignored')})" for c in ignored) + ".")
        lines.append("Measured on " + ISTA_MEASURED + ". Other computers are slower or faster by what their cards, RAM and drive can do.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the files go. "
                     "Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def _plan_uma(key: str, model: str, card: dict, ram_gb: float | None, disk_free_gb: float | None, context: int) -> dict[str, Any]:
    """plan() for an official IST-DAS set on a unified-memory APU (Ryzen AI Max, gfx1151, Windows): the graphics memory is the RAM,
    counted once.  The engine holds what fits beside the dense weights and the reserve and reads the rest from the SSD; the numbers in
    UMA_MEASURED are the measured 64 GB box.  Never a refusal."""
    v = ISTA_SETS[key]
    total = uma_total_gb(card, ram_gb)
    expert_gb = v["experts_gb"] / EXPERTS
    set_gb = v["set_bytes"] / GB
    download = ista_download_bytes(model) / GB
    room = max(0.0, total - ISTA_DENSE_GB - UMA_RESERVE_GB)
    resident = min(float(v["experts_gb"]), room)
    streamed = max(0.0, float(v["experts_gb"]) - room)
    in_mem = int(resident / expert_gb)
    need_disk = download + DISK_HEADROOM_GB
    out: dict[str, Any] = {
        "variant": model, "cards": [{"index": card.get("index"), "name": card.get("name"), "vram_gb": _round1(total),
                                     "layers": [0, LAYERS - 1], "dense_gb": _round1(ISTA_DENSE_GB),
                                     "experts_gb": _round1(resident), "experts": in_mem}],
        "experts_total": EXPERTS, "experts_in_vram": in_mem, "vram_experts_gb": _round1(resident), "cpu_experts_gb": 0.0,
        "weights_gb": _round1(set_gb), "experts_gb": _round1(v["experts_gb"]), "dense_gb": _round1(ISTA_DENSE_GB),
        "engram_gb": _round1(ISTA_ENGRAM_BYTES / GB), "download_gb": _round1(download), "tp": 1, "args": [],
        "measured": UMA_MEASURED, "lines": [], "warnings": [], "ram_gb": ram_gb, "context": context,
        "disk_total_gb": v["disk_total"], "disk_need_gb": _round1(need_disk),
        "disk_ok": None if disk_free_gb is None else disk_free_gb >= need_disk,
        "uma": True, "streamed_gb": _round1(streamed), "ram_needed_gb": _round1(total), "spill_gb": _round1(streamed),
        "spare_ram_gb": _round1(max(0.0, total - ISTA_DENSE_GB - v["experts_gb"] - UMA_RESERVE_GB)),
    }
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash, {v['title']}: a {download:.0f} GB download ({set_gb:.0f} GB of model, "
                 f"{ISTA_ENGRAM_BYTES / GB:.0f} GB n-gram table, {ISTA_DSPARK_BYTES / GB:.0f} GB drafter). Quality: {v['quality']}.")
    lines.append("The n-gram table stays on the SSD and is read a few rows for each word: use an NVMe drive for the model files (a slow "
                 "disk makes every word slower). The same table serves every IST-DAS set, so installing another one downloads it once.")
    name = card.get("name") or "the built-in Radeon"
    if float(card.get("dedicated_gb") or 0.0) > 0 or "GB10" not in str(card.get("name") or ""):
        lines.append(f"{name} shares this PC's memory: the RAM and the GPU's BIOS carve-out are one pool of {total:.0f} GB. About "
                     f"{resident:.0f} GB of the {float(v['experts_gb']):.0f} GB of experts stay in it beside {ISTA_DENSE_GB:.1f} GB of "
                     f"dense weights and {UMA_RESERVE_GB:.0f} GB kept for Windows; the other {streamed:.0f} GB are read from the SSD while it answers.")
        lines.append(f"Measured on this kind of PC: {UMA_MEASURED}.")
    else:       # a DGX Spark's GB10: one memory for the CPU and the GPU, nothing set aside in a BIOS, Linux
        lines.append(f"{name} shares this computer's memory: the CPU and the GPU are one pool of {total:.0f} GB. About "
                     f"{resident:.0f} GB of the {float(v['experts_gb']):.0f} GB of experts stay in it beside {ISTA_DENSE_GB:.1f} GB of "
                     f"dense weights and {UMA_RESERVE_GB:.0f} GB kept for the system; the other {streamed:.0f} GB are read from the SSD while it answers.")
        lines.append(f"Measured on this kind of computer: {UMA_MEASURED_GB10}.")
        out["measured"] = UMA_MEASURED_GB10
    if streamed > 0.5:
        lines.append("Keep the model files on the NVMe drive and the machine otherwise idle while it answers: every word reads experts from it.")
    if streamed <= 0.5:
        out["verdict"] = "fits"
    elif total >= float(v.get("ram_ok_gb") or 0):
        out["verdict"] = "tight"
        if total < 48:
            warns.append(f"This PC has {total:.0f} GB of memory shared with the graphics chip and the model was measured on 128 GB: with less, "
                         "more experts come from the SSD and each word takes longer. It still runs.")
    else:
        out["verdict"] = "spills"
        warns.append(f"This PC has {total:.0f} GB of memory shared with the graphics chip and the model was measured on 128 GB: most of its "
                     "experts are read from the SSD while it answers, so it is slow (a word can take seconds). The smaller 1.75 bpw set "
                     "suits this computer better. It is still allowed.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the files go. "
                     "Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def plan(cards: list[dict] | None, ram_gb: float | None, disk_free_gb: float | None = None, context: int = 8192,
         vendor: str | None = None, model: str | None = None, named: bool = False) -> dict[str, Any]:
    """Where the model goes on this computer.  `cards`: the graphics cards ({"index", "name", "vram_gb", ...}, as gpu_fit takes them),
    `ram_gb` the RAM, `disk_free_gb` the free space where the files go (None: not checked).  `model`: an official IST-DAS set
    (ISTA_SETS: "IST-DAS-2.00bpw" / "IST-DAS-2.25bpw" / "IST-DAS-1.75bpw") gets its own numbers; None or "UD-Q8_K_XL" is Unsloth's 508 GB set.

    Keys: cards (per card: index, name, vram_gb, layers, dense_gb, experts_gb, experts), experts_in_vram, experts_total, vram_experts_gb,
    cpu_experts_gb, ram_needed_gb, spill_gb, spare_ram_gb, verdict (fits / tight / spills / unknown / no_card), tp, args,
    download_gb, engram_gb, disk_need_gb, disk_ok, lines (plain words, one idea each), warnings (the risks, plainly), measured.
    On a unified-memory APU (Ryzen AI Max, gfx1151) the plan counts one pool: also `uma: True` and `streamed_gb` (the experts
    read from the SSD), tp 1 and no args (nothing to split between cards)."""
    if variant(model):
        mac = mac_card(cards)
        if mac is not None and variant_key(model) in MAC_SET_PACK:
            return _plan_mac(variant_key(model), model, mac, ram_gb, disk_free_gb, context)
        mine = gpu_fit.for_vendor(list(cards or []), vendor)[0] if vendor else list(cards or [])
        card = uma_card(gpu_fit.usable_cards(mine)[0])   # an APU beside a card with memory of its own does not count (setup leaves it out too)
        if card is not None:
            return _plan_uma(variant_key(model), model, card, ram_gb, disk_free_gb, context)
        return _plan_ista(variant_key(model), cards, ram_gb, disk_free_gb, context, vendor, named)
    keep, ignored = run_cards(cards, vendor, named=named)                 # the engine takes the layers in card order
    n = len(keep)
    out: dict[str, Any] = {"cards": [], "experts_total": EXPERTS, "experts_in_vram": 0, "vram_experts_gb": 0.0,
                           "cpu_experts_gb": EXPERTS_GB, "weights_gb": _round1(WEIGHTS_GB), "experts_gb": _round1(EXPERTS_GB),
                           "dense_gb": _round1(DENSE_GB), "engram_gb": _round1(ENGRAM_GB), "download_gb": _round1(DOWNLOAD_GB),
                           "tp": tp_for(n), "args": [], "measured": MEASURED, "lines": [], "warnings": [],
                           "ram_gb": ram_gb, "context": context}
    need_disk = DOWNLOAD_GB + DISK_HEADROOM_GB
    out["disk_need_gb"] = _round1(need_disk)
    out["disk_ok"] = None if disk_free_gb is None else disk_free_gb >= need_disk
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash (UD-Q8_K_XL): a {DOWNLOAD_GB:.0f} GB download. In memory it needs {WEIGHTS_GB:.0f} GB: "
                 f"{EXPERTS_GB:.0f} GB of experts ({EXPERTS:,} of {EXPERT_GB * 1000:.1f} MB) and {DENSE_GB:.0f} GB of attention "
                 "and other dense weights.")
    lines.append(f"Its n-gram table ({ENGRAM_GB:.0f} GB of the download) stays on the SSD and is read a few rows for each word "
                 "it writes: it is not counted in RAM, but a slow disk (SATA, a network drive, a USB disk) makes every word "
                 "slower. Use an NVMe drive for the model files.")
    if n == 0:
        out["verdict"] = "no_card"
        warns.append("No graphics card Strata can use was found: this model needs one or more cards (it was run on AMD Radeon "
                     "AI PRO R9700 cards).")
    else:
        per_dense = DENSE_ON_GPU_GB / n
        cap = []
        for c in keep:
            room = _cap_gb(float(c.get("vram_gb") or 0)) - per_dense - RESERVE_GB
            cap.append(max(0.0, room))
        vram_exp = min(EXPERTS_GB, sum(cap))
        share = [x / sum(cap) if sum(cap) else 1.0 / n for x in cap]
        ranges = split_layers([float(c.get("vram_gb") or 0) for c in keep])
        in_vram = 0
        for c, room, (a, b), s in zip(keep, cap, ranges, share):
            gb = min(room, vram_exp * s) if sum(cap) else 0.0
            ne = int(gb / EXPERT_GB)
            ne = min(ne, (b - a + 1) * EXPERTS_PER_LAYER)
            in_vram += ne
            out["cards"].append({"index": c.get("index"), "name": c.get("name"), "vram_gb": float(c.get("vram_gb") or 0),
                                 "layers": [a, b], "dense_gb": _round1(per_dense), "experts_gb": _round1(ne * EXPERT_GB),
                                 "experts": ne})
        out["experts_in_vram"] = in_vram
        out["vram_experts_gb"] = _round1(in_vram * EXPERT_GB)
        out["cpu_experts_gb"] = _round1(EXPERTS_GB - in_vram * EXPERT_GB)
        out["tp"] = tp_for(n)
        out["args"] = ["--tp", str(out["tp"])] if out["tp"] > 1 else []
        if ram_gb:
            needed = out["cpu_experts_gb"] + REST_RAM_GB
            spare = float(ram_gb) - needed
            out["ram_needed_gb"] = _round1(needed)
            out["spare_ram_gb"] = _round1(max(0.0, spare))
            out["spill_gb"] = _round1(max(0.0, -spare))
            out["verdict"] = "spills" if spare < 0 else "tight" if spare < SPARE_FITS_GB else "fits"
        else:
            out["verdict"] = "unknown"
        # the placement, in words
        short = [c for c in keep if _cap_gb(float(c.get("vram_gb") or 0)) < per_dense + RESERVE_GB + 1.0]
        if short:
            warns.append(f"The dense weights need ~{per_dense:.0f} GB of graphics memory on {'each card' if n > 1 else 'the card'} "
                         f"and {short[0].get('name') or 'a card'} ({float(short[0].get('vram_gb') or 0):.0f} GB) has "
                         "hardly that: the engine may stop at the start (out of graphics memory). More or bigger cards fix it; "
                         "it is still allowed.")
        if n == 1:
            c = out["cards"][0]
            lines.append(f"One card ({c['name'] or 'card ' + str(c['index'])}, {c['vram_gb']:.0f} GB): the dense weights ({c['dense_gb']:.0f} GB) "
                         f"and {c['experts']:,} of the {EXPERTS:,} experts ({c['experts_gb']:.0f} GB) go in its memory, the other "
                         f"{EXPERTS - c['experts']:,} experts ({out['cpu_experts_gb']:.0f} GB) are computed on the CPU from RAM.")
        else:
            blocks = "  |  ".join(f"card {c['index']}: layers {c['layers'][0]}-{c['layers'][1]}" for c in out["cards"])
            lines.append(f"{n} cards, one block of layers on each ({blocks}); the engine balances the split. On each card "
                         f"~{out['cards'][0]['dense_gb']:.1f} GB of dense weights and ~{out['cards'][0]['experts_gb']:.0f} GB of experts "
                         f"(about {out['cards'][0]['experts']:,}); in all {in_vram:,} of {EXPERTS:,} experts ({out['vram_experts_gb']:.0f} GB) "
                         f"are in graphics memory.")
            lines.append("Two cards share each layer's matrices (--tp 2): measured about twice as fast at reading prompts as a plain "
                         "split of the layers.")
        lines.append(f"The other {EXPERTS - in_vram:,} experts ({out['cpu_experts_gb']:.0f} GB) are computed on the CPU, read in place "
                     "from the model's files: they need that much free RAM to stay cached "
                     f"({out.get('ram_needed_gb', out['cpu_experts_gb'] + REST_RAM_GB):.0f} GB with {REST_RAM_GB:.0f} GB for the system and the engine).")
        if out["verdict"] == "spills":
            warns.append(f"This computer has {ram_gb:.0f} GB of RAM and {out['ram_needed_gb']:.0f} GB would keep the CPU's share of the experts "
                         f"cached: about {out['spill_gb']:.0f} GB of them are read from the SSD while the model answers, which is slow "
                         "(a word can take seconds). More RAM or more graphics memory fixes it. It is still allowed.")
        elif out["verdict"] == "tight":
            warns.append(f"RAM is tight: {out['spare_ram_gb']:.0f} GB are left beside the experts; close other programs.")
        elif out["verdict"] == "fits":
            lines.append(f"That fits this computer ({ram_gb:.0f} GB of RAM, {out['spare_ram_gb']:.0f} GB to spare).")
        if n >= 2 and any(abs(float(c.get("vram_gb") or 0) - float(keep[0].get("vram_gb") or 0)) > 1 for c in keep):
            lines.append("The cards are not the same size: the layers are shared in proportion to their memory.")
        if ignored:
            lines.append("Not used: " + "; ".join(f"{c.get('name') or 'a card'} ({c.get('why_ignored')})" for c in ignored) + ".")
        lines.append("Measured on " + MEASURED + ". Other computers are slower or faster by what their cards and RAM can hold; "
                     "this engine has been run on AMD cards.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the "
                     "files go. Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def _option(key: str, cards: list[dict], ram_gb: float | None, ctx: int, vendor: str | None, model: str | None = None) -> dict[str, Any]:
    p = plan(cards, ram_gb, None, ctx, vendor, model)
    names = " + ".join(gpu_fit._name(c) for c in cards)
    label = f"{names} + RAM" if len(cards) == 1 else f"{names} + RAM (layers shared, --tp {p['tp']})"
    words = {"fits": f"fits: {p['experts_in_vram']:,} of {p['experts_total']:,} experts in graphics memory, the rest computed on the CPU "
                     f"({p.get('ram_needed_gb', 0):.0f} GB of RAM)",
             "tight": f"fits, but tight: RAM left over is under {SPARE_FITS_GB:.0f} GB",
             "spills": f"does not fit in memory: about {p.get('spill_gb', 0):.0f} GB of the experts are read from the SSD while answering (slow)",
             "unknown": "memory of this computer unknown", "no_card": "no usable card"}[p["verdict"]]
    return {"key": key, "cards": [c.get("index") for c in cards], "label": label, "verdict": p["verdict"], "summary": f"{label}: {words}",
            "words": words, "vram_gb": _round1(sum(float(c.get("vram_gb") or 0) for c in cards)), "spill_gb": p.get("spill_gb", 0.0),
            "ram_use_gb": p.get("cpu_experts_gb", 0.0), "spare_ram_gb": p.get("spare_ram_gb", 0.0),
            "notes": [l for l in p["lines"] if l.startswith(("One card", f"{len(cards)} cards", "Two cards share"))],
            "recommended": False, "plan": p}


def options(cards: list[dict], ram_gb: float | None, vendor: str | None = None, context: int = 8192, model: str | None = None) -> dict[str, Any]:
    """gpu_fit.options for this model: every card that counts (recommended: the model is bigger than any one card, so each card adds
    experts that stay off the CPU), and the largest card alone.  Same keys as gpu_fit.options."""
    keep, ignored = run_cards(cards, vendor)          # the same cards `strata launch --plan` and setup use (Windows: one AMD card; the package's own archs)
    out: dict[str, Any] = {"options": [], "recommended": None, "ignored": ignored, "cards": len(keep), "vendor": vendor,
                           "other_vendor_note": gpu_fit.other_vendor_note(list(cards or []), vendor)}
    if not keep:
        return out
    opts = [_option("all", sorted(keep, key=lambda c: c.get("index", 0)), ram_gb, context, vendor, model)]
    if len(keep) > 1:
        largest = min(keep, key=lambda c: (-round(float(c.get("vram_gb") or c.get("unified_gb") or 0)), c.get("index", 0)))
        opts.append(_option("single", [largest], ram_gb, context, vendor, model))
    opts[0]["recommended"] = True
    out["options"], out["recommended"] = opts, "all"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Apple Silicon: one pool of unified memory, the routed experts are file-backed and stay resident up to the engine's cap.
# The numbers are the three GSQ-RCO packs (the research plan's table, measured from the public allocations) and the
# engine's own cap rule (src/dsv41/ds4_model.cu metal_expert_cap, metal/src/engine/arena_budget.h).

MAC_PACKS: dict[str, dict[str, Any]] = {
    "IQ1_M": {"name": "1.75 bpw (IQ1_M)", "routed_gb": 114.56, "backbone_gb": 120.72, "bpw": 1.75},
    "IQ2_XXS": {"name": "2.00 bpw (IQ2_XXS), the default for most users", "routed_gb": 132.07, "backbone_gb": 138.24, "bpw": 2.00},
    "IQ2_XS": {"name": "2.25 bpw (IQ2_XS), the quality one", "routed_gb": 149.06, "backbone_gb": 155.23, "bpw": 2.25},
    "UD-IQ1_S": {"name": "Unsloth UD-IQ1_S (1-bit experts)", "routed_gb": 115.23, "backbone_gb": 122.38, "bpw": 1.75},
    "UD-IQ1_M": {"name": "Unsloth UD-IQ1_M (1-bit experts)", "routed_gb": 132.21, "backbone_gb": 139.81, "bpw": 2.00},
}
MAC_ENGRAM_GB = 110.60          # the same Engram table for all three, read row by row off the SSD
MAC_ENGINE_FOOTPRINT_GB = 12.0  # the process' anonymous memory after the load (Metal buffers, host buffers, the KV cache)
MAC_DEVICE_SHARE = 0.75         # recommendedMaxWorkingSetSize as a share of the RAM (107.5 of 128 GB on the M5 Max)
MAC_FILECACHE_FLOOR_GB = 37.2   # vm.vm_page_filecache_min on the 128 GB M5 Max
MAC_HEADROOM_GB = 12.0          # beside that floor: macOS itself and one more program
MAC_MEASURED = ("M5 Max, 128 GB, 2.25 bpw: 13-15 tokens/s for a short answer (10-12 after a long prompt), a 3,000-token "
                "prompt at about 295 tokens/s, 32,000 at about 720; the engine kept about 68 GB of the experts resident "
                "and read about 110 MB a token from the SSD")


# What each pack did on that Mac (measured on the 0.2.7.1 build, 2026-10-27, product flags, frozen placement).
MAC_MEASURED_PACK = {
    "IQ2_XXS": ("M5 Max, 128 GB, 2.00 bpw: 15-16 tokens/s for a short answer (14-15 after a long prompt), a 3,000-token prompt at about 240 "
                "tokens/s, 32,000 at about 610; the engine kept about 69 GB of the experts resident and read about 48 MB a token from the SSD"),
    "IQ2_XS": MAC_MEASURED,
    "UD-IQ1_S": ("M5 Max, 128 GB, Unsloth UD-IQ1_S: 15-18 tokens/s for a short answer (16-18 after a 3,000-token prompt), a 3,000-token prompt at "
                 "about 230-310 tokens/s, 14,000 at about 520; the engine kept about 68 GB of the experts resident and the process used about 79 GB"),
    "UD-IQ1_M": "Not run on a Mac yet; UD-IQ1_S was. It keeps fewer experts in memory, so expect it to be slower.",
}


def mac_expert_cap_gb(ram_gb: float) -> float:
    """The expert budget the engine will pick on a Mac of `ram_gb` (GiB, as the rest of the app reports memory), in
    decimal GB: the RAM less what macOS keeps, less the process' own footprint.  Mirrors metal_expert_cap() with its
    big-Mac rule (>= 64 GB: 40% left, at least the file-cache floor + 12 GB).  128 GiB gives the 68-69 GB measured on
    the M5 Max."""
    gib = 1.073741824
    p = float(ram_gb) * gib
    if p <= 0:
        return 0.0
    keep = max(7.0, 0.15 * p)
    # the file-cache floor (vm.vm_page_filecache_min) was read on the 128 GiB M5 Max (37.2 GB) and on a 32 GiB M4 mini (11.1 GB, a third
    # of its RAM): it grows with the memory, so a Mac between the two gets it by that share, an estimate (only the 128 GiB one was run)
    floor = MAC_FILECACHE_FLOOR_GB * min(1.0, p / (128.0 * gib))
    leave = max(keep, 0.40 * p, floor + MAC_HEADROOM_GB) if p >= 64.0 else keep
    cap = min(p - leave, MAC_DEVICE_SHARE * p)
    return max(0.0, cap - MAC_ENGINE_FOOTPRINT_GB - 1.0)


def mac_plan(ram_gb: float, quant: str = "IQ2_XS", disk_free_gb: float | None = None) -> dict[str, Any]:
    """Where the DeepSeek pack goes on an Apple Silicon Mac.  Keys: quant, name, routed_gb, dense_gb, resident_gb,
    streaming_gb, engram_gb, disk_need_gb, disk_ok, ram_gb, verdict (fits / streams / tight / small), lines, warnings,
    words, measured.  `fits` means every routed expert stays in memory; `streams` is the normal case and is still
    allowed (the engine reads the rest from the SSD in place)."""
    if quant not in MAC_PACKS:
        raise ValueError(f"unknown Mac pack {quant!r}; one of {sorted(MAC_PACKS)}")
    pack = MAC_PACKS[quant]
    routed = float(pack["routed_gb"])
    dense = round(float(pack["backbone_gb"]) - routed, 2)
    cap = mac_expert_cap_gb(ram_gb or 0.0)
    resident = min(routed, cap)
    streaming = max(0.0, routed - resident)
    disk_need = float(pack["backbone_gb"]) + MAC_ENGRAM_GB + 6.0     # the shards, the Engram table, the pack folder
    out: dict[str, Any] = {"quant": quant, "name": pack["name"], "routed_gb": round(routed, 1), "dense_gb": dense,
                           "resident_gb": round(resident, 1), "streaming_gb": round(streaming, 1),
                           "engram_gb": MAC_ENGRAM_GB, "disk_need_gb": round(disk_need, 1),
                           "disk_ok": None if disk_free_gb is None else float(disk_free_gb) >= disk_need,
                           "ram_gb": ram_gb, "measured": MAC_MEASURED, "lines": [], "warnings": []}
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash {pack['name']}: {routed:.0f} GB of routed experts and {dense:.1f} GB of dense "
                 "weights, plus a 111 GB Engram table that stays on the SSD and is read a few rows for each word.")
    if ram_gb and ram_gb >= 96 and streaming <= 0.5:
        lines.append(f"This Mac ({ram_gb:.0f} GB): the engine will keep all {routed:.0f} GB of the experts in memory.")
    elif ram_gb and ram_gb >= 96:
        lines.append(f"This Mac ({ram_gb:.0f} GB): the engine will keep about {resident:.0f} GB of the experts in memory "
                     f"and read the other {streaming:.0f} GB from the SSD while it answers.")
    elif ram_gb:
        lines.append(f"This Mac ({ram_gb:.0f} GB) can hold about {resident:.0f} GB of the experts; the other "
                     f"{streaming:.0f} GB stream from the SSD, which is slow on a small Mac.")
    if cap <= 0:
        out["verdict"] = "small"
        warns.append("Too little memory to hold a useful part of the model: 64 GB or more is wanted.")
    elif streaming == 0:
        out["verdict"] = "fits"
        lines.append("Every routed expert stays in memory: the fastest case.")
    elif resident < 0.25 * routed:
        out["verdict"] = "small"
        warns.append(f"Only about {resident:.0f} GB of {routed:.0f} GB of experts fit in memory (under a quarter): "
                     "most of the model is read from the SSD for every word, so answers are much slower than on a Mac with 128 GB "
                     "(nothing this small was measured). It still runs; a 64 GB or bigger Mac is the better fit.")
    else:
        out["verdict"] = "streams"
        lines.append("The model streams: answers are slower than on a machine that holds all the experts, and they still "
                     "work. An NVMe SSD is wanted, not a USB or network drive.")
    if disk_free_gb is not None and disk_free_gb < disk_need:
        warns.append(f"The files need about {disk_need:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free.")
    lines.append("Measured on " + MAC_MEASURED + ".")
    out["words"] = " ".join(lines)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The Mac in the launch flow: plan() answers for an Apple Silicon Mac in the shape the catalog, `strata launch` and the app read
# for any other computer, from mac_plan's numbers.  The official sets that were run on a Mac (2.00 and 2.25 bpw, 2026-10-25);
# the 1.75 bpw set shares the 2.00 set's IQ1_M kernel but was never run there, so it is not offered on a Mac.


MAC_SET_PACK = {"IST-DAS-2.00bpw": "IQ2_XXS", "IST-DAS-2.25bpw": "IQ2_XS", "UD-IQ1_S": "UD-IQ1_S", "UD-IQ1_M": "UD-IQ1_M"}
# Unsloth's UD-IQ1_S was run on an M5 Max (f-039-mac-dsu).  UD-IQ1_M has the same kernels and was not run there; the owner chose to offer it anyway ("they
# will tell if it works"), with a plain "not run yet" line in its plan.  The gate stays as a mechanism, empty: put a name here to keep a set off a Mac
# (the model lists, setup, the launch and the "I already have the model" check all read it through not_here()).
MAC_NOT_YET = ()
MAC_NOT_YET_LINE = "Unsloth's UD-IQ1_M set doesn't run on a Mac yet; the UD-IQ1_S, 2.00 and 2.25 sets do."
# The same two sets were kept off a DGX Spark (linux-aarch64, GB10) until a Spark run passed: both passed there on 2026-10-30 (real `strata launch --model-path`,
# 10 of 10 answers, tool call correct), so the tuple is empty.  Put a name back here to hide a set from the Spark again.
AARCH64_NOT_YET = ()
AARCH64_NOT_YET_LINE = "Unsloth's 1-bit DeepSeek sets were not tested on a DGX Spark yet; the 2.00 and 2.25 sets were."


def not_here(model: str | None) -> str | None:
    """The plain line when `model` is not offered on THIS computer (a Mac: MAC_NOT_YET; linux-aarch64, the DGX Spark: AARCH64_NOT_YET), else None.
    Read at call time from sys.platform and platform.machine(), so one rule serves the model lists, setup, `strata launch` and the app's file check."""
    if sys.platform == "darwin" and model in MAC_NOT_YET:
        return MAC_NOT_YET_LINE
    if sys.platform.startswith("linux") and platform.machine().lower() in ("aarch64", "arm64") and model in AARCH64_NOT_YET:
        return AARCH64_NOT_YET_LINE
    return None
MAC_SMALL_RAM_GB = 64.0     # below this the measured M5 Max numbers do not apply (nothing smaller than 32 GB was ever started)


def mac_card(cards: list[dict] | None) -> dict | None:
    """The Apple Silicon "graphics card" of the hardware report (vendor apple, its unified memory in `unified_gb`), None on any other computer."""
    return next((c for c in cards or [] if str(c.get("vendor")) == "apple" or float(c.get("unified_gb") or 0) > 0), None)


def _plan_mac(key: str, model: str, card: dict, ram_gb: float | None, disk_free_gb: float | None, context: int) -> dict[str, Any]:
    """plan() for an official IST-DAS set on an Apple Silicon Mac.  One pool of unified memory: the engine keeps the experts it can
    beside the dense weights resident (mac_expert_cap_gb) and reads the rest from the SSD in place.  Never a refusal: a smaller Mac
    gets a plain warning, not a block."""
    v = ISTA_SETS[key]
    ram = float(ram_gb or card.get("unified_gb") or 0.0)
    mp = mac_plan(ram, MAC_SET_PACK[key], disk_free_gb)
    expert_gb = v["experts_gb"] / EXPERTS
    resident = float(mp["resident_gb"])
    streamed = float(mp["streaming_gb"])
    in_mem = int(resident / expert_gb)
    download = ista_download_bytes(model) / GB
    need_disk = download + DISK_HEADROOM_GB
    verdict = {"fits": "fits", "streams": "tight", "small": "spills"}[mp["verdict"]]
    out: dict[str, Any] = {
        "variant": model, "mac": True, "mac_verdict": mp["verdict"],
        "cards": [{"index": card.get("index", 0), "name": card.get("name") or "Apple Silicon", "vram_gb": _round1(ram),
                   "layers": [0, LAYERS - 1], "dense_gb": _round1(ISTA_DENSE_GB), "experts_gb": _round1(resident), "experts": in_mem}],
        "experts_total": EXPERTS, "experts_in_vram": in_mem, "vram_experts_gb": _round1(resident), "cpu_experts_gb": 0.0,
        "weights_gb": _round1(v["set_bytes"] / GB), "experts_gb": _round1(v["experts_gb"]), "dense_gb": _round1(ISTA_DENSE_GB),
        "engram_gb": _round1(ISTA_ENGRAM_BYTES / GB), "download_gb": _round1(download), "tp": 1, "args": [],
        "measured": MAC_MEASURED, "lines": [], "warnings": [], "ram_gb": ram_gb, "context": context,
        "disk_total_gb": v["disk_total"], "disk_need_gb": _round1(need_disk),
        "disk_ok": None if disk_free_gb is None else disk_free_gb >= need_disk,
        "streamed_gb": _round1(streamed), "ram_needed_gb": _round1(resident + ISTA_DENSE_GB + MAC_ENGINE_FOOTPRINT_GB),
        "spill_gb": _round1(streamed), "spare_ram_gb": _round1(max(0.0, ram - resident - ISTA_DENSE_GB - MAC_ENGINE_FOOTPRINT_GB)),
        "verdict": verdict,
    }
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash, {v['title']}: a {download:.0f} GB download ({v['set_bytes'] / GB:.0f} GB of model, "
                 f"{ISTA_ENGRAM_BYTES / GB:.0f} GB n-gram table, {ISTA_DSPARK_BYTES / GB:.0f} GB drafter). Quality: {v['quality']}.")
    lines.extend(l for l in mp["lines"][1:] if not l.startswith("Measured on "))
    measured = MAC_MEASURED_PACK[MAC_SET_PACK[key]]
    lines.append(measured if measured.startswith("Not run") else "Measured on " + measured + ".")
    out["measured"] = measured
    warns.extend(mp["warnings"])
    if ram and ram < MAC_SMALL_RAM_GB and mp["verdict"] != "small":
        warns.append(f"This Mac has {ram:.0f} GB of memory and the set was measured on one with 128 GB: with less, more experts come from "
                     "the SSD for every word, so answers are slower. It still runs.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the files go. "
                     "Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out
