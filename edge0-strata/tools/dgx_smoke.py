#!/usr/bin/env python3
"""The drop's acceptance gate in one command: setup detection, one benchmark cell, the identity gate and a needle, each
with a band that the numbers on a DGX Spark should land in. Exit code 0 = all passed.

    python tools/dgx_smoke.py [--config strata-iq3_s.json] [--tokens 12000] [--data-dir ~/Strata-data]

Bands are deliberately loose (they must survive page-cache state and drift); a failure means something is wrong, not that
the machine is 5% slower than the number in the documents.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WORK = Path(tempfile.mkdtemp(prefix="dgx-smoke-"))      # this run's own scratch folder (never a fixed /tmp name another run could share)


def sh(cmd: str) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def check_setup(data_dir: str) -> tuple[bool, str]:
    rc, out = sh(f"{PY} setup.py --check --yes --data-dir '{data_dir}' 2>&1 | head -8")
    low = out.lower()
    ok = "unified memory" in low and "arm64" in low
    gpu = next((line.strip() for line in out.splitlines() if "GPU:" in line), "")
    return ok, gpu or (out.strip().splitlines()[0] if out.strip() else "no output")


def bench(config: str, cell: str, out_dir: str) -> dict | None:
    rc, out = sh(f"{PY} tools/dgx_bench.py --config {config} --tag smoke --cells {cell} --rounds 1 --port 8093 "
                 f"--out {out_dir} 2>&1")
    m = re.search(r"prompt_n\s+(\d+)\s+prompt\s+([0-9.]+)\s+tok/s\s+decode\s+([0-9.]+)", out)
    if not m:
        return None
    return {"prompt_n": int(m.group(1)), "prompt": float(m.group(2)), "decode": float(m.group(3))}


def identity(config: str) -> tuple[bool, str]:
    base = json.loads((ROOT / config).read_text())
    off = dict(base)
    off["args"] = list(base["args"])
    env = dict(base.get("env") or {})
    env["STRATA_PF_FUSED"] = "0"
    off["env"] = env
    off_path = WORK / "nofused.json"
    off_path.write_text(json.dumps(off, indent=1))
    rc, out = sh(f"{PY} tools/dgx_quality.py {off_path} {config} {WORK / 'quality.json'} 2>&1")
    m = re.search(r"(\d+) of (\d+) answers identical", out)
    if not m:
        return False, out.strip()[-200:]
    return m.group(1) == m.group(2), out.strip().splitlines()[-1]


def needle(config: str, tokens: int) -> tuple[bool, str]:
    rc, out = sh(f"{PY} tools/dgx_needle.py --config {config} --tokens {tokens} --runs 2 2>&1")
    m = re.search(r"(\d+) of (\d+) recalled", out)
    if not m:
        return False, out.strip()[-200:]
    return m.group(1) == m.group(2), out.strip().splitlines()[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="strata-iq3_s.json")
    ap.add_argument("--cell", default="4096:0:256")
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--data-dir", default=os.environ.get("STRATA_DATA_DIR") or str(Path.home() / "Strata-data"),
                    help="the data folder setup uses (default: $STRATA_DATA_DIR or ~/Strata-data)")
    a = ap.parse_args()
    results = []
    t0 = time.time()

    ok, detail = check_setup(a.data_dir)
    results.append(("setup detection", ok, detail))

    b = bench(a.config, a.cell, str(WORK / "bench"))
    if b is None:
        results.append(("bench cell", False, "no parsable cell"))
    else:
        ok = b["prompt"] > 800 and b["decode"] > 30
        results.append(("bench cell", ok, f"{b['prompt']:.0f} prompt / {b['decode']:.1f} decode tok/s"))

    ok, detail = identity(a.config)
    results.append(("identity, fused vs MMQ", ok, detail))

    ok, detail = needle(a.config, a.tokens)
    results.append(("needle, recall", ok, detail))

    print()
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)} of {len(results)} passed in {time.time() - t0:.0f} s"
          + (f"; failed: {', '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
