#!/usr/bin/env python3
"""Decode bounds of a quant from its real per-layer formats (tools/quant_formats.py --json output).

    python tools/quant_bound.py FMT.json [...] --ram-gbs 120 [--vram-gbs 640] [--hit 0.3] [--tokens-per-pass 1]

Per token: the routed experts (10 of 512 per layer: each layer's gate + up + down bytes / 512 x 10), the shared expert
and the dense side (attention, SSM, hyper-connections, router, head: read once per pass on the GPU).  The bound for a
pass of P tokens (a verify window) with a VRAM hit fraction h of the routed experts:
    t_cpu = (1 - h) x routed bytes x U(P) / RAM bandwidth      (the CPU pool reads every missed expert once per pass)
    t_gpu = (h x routed x U(P) + dense) / VRAM bandwidth
    tok/s <= P / max(t_cpu, t_gpu)
U(P) is the distinct experts a pass of P tokens reads per expert a single token reads (1 for P = 1; measured from
the engine's window logs where known, else --union).  The CPU line alone (h = 0) is the floor of the all-on-CPU case.
"""
from __future__ import annotations

import argparse
import json
import pathlib

TOPK, N_EXPERT = 10, 512


def bytes_of(js: dict) -> dict:
    roles = js["roles"]
    tot = lambda role: sum(v["bytes"] for v in roles.get(role, {}).values())
    routed_all = tot("routed gate") + tot("routed up") + tot("routed down")
    per_token = routed_all / N_EXPERT * TOPK
    dense = sum(tot(r) for r in ("shared expert", "router", "attention", "SSM", "hyper-connections", "indexer",
                                 "PLE key/value", "output head"))
    return {"routed_all": routed_all, "routed_per_token": per_token, "dense_per_pass": dense,
            "expert_bytes": routed_all / 48 / N_EXPERT}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fmt", nargs="+")
    ap.add_argument("--ram-gbs", type=float, required=True, help="measured RAM read bandwidth, GB/s")
    ap.add_argument("--vram-gbs", type=float, default=0, help="GPU memory bandwidth, GB/s (0: no GPU line)")
    ap.add_argument("--hit", type=float, default=0.0, help="VRAM hit fraction of the routed experts")
    ap.add_argument("--tokens-per-pass", type=float, default=1.0)
    ap.add_argument("--union", type=float, default=1.0, help="distinct experts per pass / experts per token")
    a = ap.parse_args()
    print(f"{'quant':12s} {'expert MB':>9s} {'routed MB/tok':>13s} {'dense MB':>9s} {'CPU-only tok/s':>15s} "
          f"{'bound tok/s':>11s}")
    for f in a.fmt:
        b = bytes_of(json.loads(pathlib.Path(f).read_text()))
        r = b["routed_per_token"] * a.union
        t_cpu = (1 - a.hit) * r / (a.ram_gbs * 1e9)
        t_gpu = (a.hit * r + b["dense_per_pass"]) / (a.vram_gbs * 1e9) if a.vram_gbs else 0.0
        cpu_only = 1 / (b["routed_per_token"] / (a.ram_gbs * 1e9))
        bound = a.tokens_per_pass / max(t_cpu, t_gpu) if max(t_cpu, t_gpu) > 0 else float("inf")
        print(f"{pathlib.Path(f).stem:12s} {b['expert_bytes'] / 1e6:9.2f} {b['routed_per_token'] / 1e6:13.1f} "
              f"{b['dense_per_pass'] / 1e6:9.1f} {cpu_only:15.1f} {bound:11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
