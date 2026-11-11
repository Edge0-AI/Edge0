#!/usr/bin/env python3
"""The 24-prompt identity gate against a RECORDED set of answers: start the server of CONFIG, send the 24 greedy prompts of
tools/dgx_quality24.py, and compare every answer byte for byte with the answers in REFERENCE.json (a file tools/dgx_quality24.py wrote:
its rows hold "a" and "b"; --side picks which one is the reference, "a" by default).

    python tools/dgx_gate_ref.py CONFIG REFERENCE.json OUT.json [--side a|b] [--port 8091]

Exit code 0 = every answer identical. This is the gate a build on a DGX Spark has to pass against the answers the 2026-10-26 handover
recorded (bench/quality24-fork.json for IQ3_S, bench/quality24-nvfp4-fork.json for NVFP4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgx_quality24 as q  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--side", choices=("a", "b"), default="a")
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    ref = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    rows = ref["rows"]
    if [r["prompt"] for r in rows] != q.PROMPTS:
        print("the reference was made with other prompts than tools/dgx_quality24.py has now; the gate is not comparable", file=sys.stderr)
        return 2
    mine = q.run_config(a.config, a.port, "ref")
    out = [{"prompt": r["prompt"], "reference": r[a.side], "answer": m, "identical": r[a.side] == m} for r, m in zip(rows, mine)]
    same = sum(1 for r in out if r["identical"])
    Path(a.out).write_text(json.dumps({"identical": same, "total": len(out), "reference": a.reference, "side": a.side, "rows": out}, indent=1),
                           encoding="utf-8")
    print(f"{same} of {len(out)} answers identical to {Path(a.reference).name} (side {a.side})")
    for i, r in enumerate(out):
        if not r["identical"]:
            print(f"  differs at prompt {i}: {r['prompt'][:60]!r}")
            print(f"    reference: {r['reference'][:90]!r}")
            print(f"    this run : {r['answer'][:90]!r}")
    return 0 if same == len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
