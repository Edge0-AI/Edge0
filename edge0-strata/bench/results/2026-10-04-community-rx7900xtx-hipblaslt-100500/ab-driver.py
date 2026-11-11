#!/usr/bin/env python3
"""hipblaslt1005_driver.py - unconfound the 128K prefill comparison in PR #745.

T4 (record-autotune-128k-pr-20261004.json) measured the ROCm 10.2 nightly arm without a
matching hipBLASLt tuning table (its 100500-era library refuses the gfx1100-100100 table),
so its prefill number carried the #505 table-version fallback, not the toolchain.  This
driver A/Bs the SAME nightly engine on the SAME workload with and without a freshly tuned
gfx1100-hipblaslt-100500 table (tuned on this host against the nightly SDK's own
libhipblaslt by build-hip-0138/tune_hipblaslt under LD_LIBRARY_PATH).

Methodology = rates_amd.py: one-shot `strata generate`, greedy, 256 generated tokens,
prefill_tok_s = prompt_tokens / GPU_timeline_s, decode from the engine's own decode line,
median of 3 clean cells per arm (STALL_FRAC flagging, flagged cells never published).
One 1k cell on the table arm is the decode sanity check (decode is O(1) in context).

Cells run one at a time inside a systemd-run --user scope with MemoryMax (the desktop
protection), gated on MemAvailable, and each cell first checks no stale engine process
holds the GPU (the P1 KFD-pid check).
"""
import json
import os
import re
import subprocess
import sys
import time

TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT = f"/home/zacch/projects/strata-amd/artifacts/rocm/results/record-hipblaslt-100500-{TS}.json"
LOGDIR = "/home/zacch/projects/strata-amd/artifacts/rocm/logs"

SDK = "/home/zacch/strata-rocm-experiments/20261003-231142/sdk"
EXE = "/home/zacch/strata-rocm-experiments/20261003-231142/build-nightly/strata"
TABLE = "/home/zacch/strata-rocm-experiments/20261003-231142/gfx1100-hipblaslt-100500.txt"

# the nightly candidate's exact serving args (serve-nightly-8082.json) - the config the
# PR's r10 numbers and runs.json describe.
COMMON = [
    "--pack", "/home/zacch/projects/strata-amd/packs/iq3_s",
    "--native", "/home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/zacch/models/flash-next-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--mtp", "/mnt/LINUXWINSHARE/Backup/models/mtp/rt",
    "--expert-profile", "/home/zacch/projects/strata-amd/Strata/data/expert-profile.bin",
    "--expert-cache", "auto", "--spec", "3",
    "--kv", "int8", "--kv-resident", "32768", "--prefill", "8192",
    # without an explicit reserve the cache auto-sizes against ~all free VRAM and the later
    # cublasCreate fails ALLOC (cell 1 of run 5: engine fully loaded, then status 2).  3072
    # matches serve-nightly-8082.json exactly.
    "--vram-reserve-mib", "3072",
]
ARM_ENV = {
    "nightly-100500": {"STRATA_HIPBLASLT_TUNING": TABLE},
    "nightly-notable": {},
}
BASE_ENV = {
    "LD_LIBRARY_PATH": f"{SDK}/lib",
    "STRATA_PREFILL_RING": "96",
    "STRATA_GR_V3": "1",
    "STRATA_SELECT_WMMA": "1",
    "STRATA_PREFILL_TIMING": "1",
    "HIP_VISIBLE_DEVICES": "0",
    "HOME": "/home/zacch",
}

TOKENS = "/home/zacch/projects/strata-amd/artifacts/rocm/workloads/qsa-profile-%d.tokens"
PLAN = [
    # (arm, tier, prompt tokens, token file, --max-context)
    ("nightly-100500", "1k",   1024,   TOKENS % 1024,   9216),
    ("nightly-100500", "128k", 131072, TOKENS % 131072, 139264),
    ("nightly-notable", "128k", 131072, TOKENS % 131072, 139264),
]
PER_ARM = 3          # clean cells per (arm, tier); the 1k sanity tier needs only 1
SANITY = {("nightly-100500", "1k")}
STALL_FRAC = 0.4

PRE_RE = re.compile(r"prefill timing: (\d+) tokens, GPU timeline ([\d.]+) ms, wall ([\d.]+) ms")
DEC_RE = re.compile(r"decode\s+(\d+) tokens? in ([\d.]+)\s*(ms|s)\s*->\s*([\d.]+) tok/s", re.I)
SPEC_RE = re.compile(r"spec accept[^%]*?([\d.]+)%")
TTFT_RE = re.compile(r"ttft[^0-9]*([\d.]+)\s*s", re.I)
LT_ENABLED_RE = re.compile(r"hipBLASLt tuning enabled \((\d+) rows, (\S+), version (\d+)\)")
LT_REFUSED_RE = re.compile(r"prefill gemm: (.*mismatch.*|.*cannot open.*|.*invalid.*); using hipBLASEx")


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def stale_engine_holders():
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    hits = []
    for l in out.splitlines():
        if ("build-nightly/strata" in l or "build-hip" in l and "/strata" in l) \
                and "ps -eo" not in l and "appimage" not in l.lower():
            hits.append(l.strip())
    return hits


def vram_used_mib():
    try:
        return int(open("/sys/class/drm/card1/device/mem_info_vram_used").read()) // 1048576
    except OSError:
        return 0


def wait_headroom(need_gb=58, timeout_s=1800):
    # 58 GB RAM + < 2600 MiB VRAM, not the campaigns' 66/70-RAM-only gate: those assumed the
    # 51 GB-resident 8081 co-running and nothing else touching VRAM.  The resident is stopped
    # for this run; the desktop is protected by the per-cell MemoryMax/MemoryHigh scope.
    # VRAM must be gated per cell because a previous cell's pages drain asynchronously
    # (a shrunken expert cache silently poisons every later cell otherwise), and RAM because
    # this engine refuses rc=1 below its own startup demand.
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if mem_available_gb() >= need_gb and vram_used_mib() < 2600:
            return True
        time.sleep(10)
    return False


def run_cell(arm, tier, tokfile, ctx, seq):
    """One one-shot generate inside a MemoryMax scope; returns the parsed cell dict."""
    log_path = f"{LOGDIR}/t1005-raw-{arm}-{tier}-{seq}.log"
    env = dict(BASE_ENV)
    env.update(ARM_ENV[arm])
    envstr = " ".join(f"{k}={json.dumps(str(v))}" if "(" in str(v) or " " in str(v) else f"{k}={v}"
                      for k, v in env.items())
    cmd = (" ".join(["--tokens-file", tokfile, "--max-new", "256", "--max-context", str(ctx)] + COMMON))
    unit = f"t1005-{seq}"
    # --wait is load-bearing: without it systemd-run only SUBMITS the unit (rc 0, no engine
    # output in our log - cell 1 of the first run proved it).  --pipe is equally load-bearing:
    # without it the unit's stdout/stderr go to the journal and a failed cell's error is
    # invisible (cell 1 of the third run: rc=1 with only systemd's summary in the log).
    scope = ["systemd-run", "--user", "--collect", "--wait", "--pipe", f"--unit={unit}",
             "-p", "MemoryMax=66G", "-p", "MemoryHigh=61G", "-p", "Restart=no",
             "/usr/sbin/bash", "-c",
             f"cd /home/zacch/projects/strata-amd && exec env {envstr} {EXE} {cmd}"]
    with open(log_path, "w") as lf:
        t0 = time.time()
        p = subprocess.run(scope, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
        wall = time.time() - t0
    out = open(log_path, errors="replace").read()
    cell = {"arm": arm, "tier": tier, "exit": p.returncode, "wall_s": round(wall, 1),
            "raw_log": log_path,
            "conditions": {"loadavg": open("/proc/loadavg").read().split()[:3],
                           "mem_available_gb": round(mem_available_gb(), 1)}}
    m = LT_ENABLED_RE.search(out)
    if m:
        cell["lt_table"] = {"rows": int(m.group(1)), "arch": m.group(2), "version": int(m.group(3))}
    else:
        r = LT_REFUSED_RE.search(out)
        cell["lt_table"] = {"engaged": False, "refusal": r.group(1) if r else "no tuning line found"}
    m = PRE_RE.search(out)
    if m:
        cell["prompt_tokens"] = int(m.group(1))
        gpu_s = float(m.group(2)) / 1000.0
        cell["prefill_gpu_s"] = round(gpu_s, 4)
        cell["prefill_tok_s"] = round(cell["prompt_tokens"] / gpu_s, 2) if gpu_s else None
    m = DEC_RE.search(out)
    if m:
        cell["decoded"] = int(m.group(1))
        cell["decode_wall_s"] = round(float(m.group(2)) / (1000.0 if m.group(3).lower() == "ms" else 1.0), 3)
        cell["decode_tok_s"] = round(float(m.group(4)), 2)
    m = SPEC_RE.search(out)
    if m:
        cell["spec_accept"] = round(float(m.group(1)) / 100.0, 3)
    m = TTFT_RE.search(out)
    if m:
        cell["ttft_s"] = round(float(m.group(1)), 2)
    return cell


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 2)


def main():
    if not os.path.exists(TABLE):
        sys.exit(f"table missing: {TABLE} - tune it first (see record-hipblaslt-100500)")
    state = {"record": "hipblaslt-100500-unconfound", "at": TS,
             "pr": "https://github.com/Niko1221/Strata/pull/745",
             "method": ("rates_amd.py methodology on the PR #745 nightly candidate (ROCm 10.2 "
                        "nightly SDK + rocWMMA, spec 3): one-shot strata generate, greedy, 256 "
                        "generated tokens; prefill_tok_s = prompt_tokens / GPU_timeline_s; median "
                        "of 3 clean cells per arm (STALL_FRAC=0.4 flagging, flagged cells never "
                        "published); arms differ ONLY in STRATA_HIPBLASLT_TUNING "
                        "(gfx1100-hipblaslt-100500.txt tuned on this host vs unset)."),
             "engine": EXE, "sdk": SDK, "table": TABLE, "table_provenance": {
                 "tuner": "/home/zacch/projects/strata/build-hip-0138/tune_hipblaslt under "
                          "LD_LIBRARY_PATH=<sdk>/lib",
                 "shapes": "artifacts/rocm/tables/hipblaslt-shapes-canon.json (16 shapes x "
                           "{4096,8192} buckets, union of shipped gfx1100 tables + upstream "
                           "gfx1201-100500 set)",
                 "tuner_log": "artifacts/rocm/logs/tune-hipblaslt-100500.log"},
             "cells": [], "verdict": None}
    if os.path.exists(OUT):
        state["cells"] = json.load(open(OUT)).get("cells", [])

    def clean_cells(arm, tier):
        cs = [c for c in state["cells"] if c["arm"] == arm and c["tier"] == tier
              and c["exit"] == 0 and c.get("decode_tok_s") and c.get("prefill_tok_s")
              and not c.get("stall_flag")]
        return cs

    def flag_stalls():
        for arm, tier, _, _, _ in PLAN:
            need = 1 if (arm, tier) in SANITY else PER_ARM
            cs = [c for c in state["cells"] if c["arm"] == arm and c["tier"] == tier
                  and c["exit"] == 0 and c.get("decode_tok_s") and c.get("prefill_tok_s")]
            bd = max((c["decode_tok_s"] for c in cs), default=0.0)
            bp = max((c["prefill_tok_s"] for c in cs), default=0.0)
            for c in cs:
                d = bool(bd and c["decode_tok_s"] < STALL_FRAC * bd)
                pp = bool(bp and c["prefill_tok_s"] < STALL_FRAC * bp)
                c["stall_flag"] = bool(d or pp)
                c["stall_why"] = "decode" if d and pp else "prefill" if pp else "decode" if d else ""

    seq = len(state["cells"])
    consecutive_fail = 0
    for arm, tier, _, tokfile, ctx in PLAN:
        while len(clean_cells(arm, tier)) < (1 if (arm, tier) in SANITY else PER_ARM):
            if consecutive_fail >= 3:
                state["verdict"] = "aborted: 3 consecutive cell failures"
                break
            if not wait_headroom():
                state["verdict"] = "incomplete: MemAvailable gate timed out"
                break
            stale = stale_engine_holders()
            if stale:
                print("stale engine holders, waiting:", stale, flush=True)
                time.sleep(20)
                continue
            flag_stalls()
            seq += 1
            print(f"cell {seq}: {arm} {tier} (vram={vram_used_mib()}MiB "
                  f"avail={mem_available_gb():.0f}GB) ...", flush=True)
            cell = run_cell(arm, tier, tokfile, ctx, seq)
            state["cells"].append(cell)
            flag_stalls()
            json.dump(state, open(OUT, "w"), indent=1)
            print(f"  rc={cell['exit']} lt={cell['lt_table']} prefill={cell.get('prefill_tok_s')} "
                  f"decode={cell.get('decode_tok_s')}", flush=True)
            consecutive_fail = consecutive_fail + 1 if (cell["exit"] != 0 or
                                                        not cell.get("decode_tok_s")) else 0
        else:
            continue
        break

    # verdict: medians per arm at 128k, table effect on the nightly stack
    flag_stalls()
    verdict = {"128k": {}, "1k_sanity": None}
    for arm in ARM_ENV:
        cs = clean_cells(arm, "128k")
        if cs:
            verdict["128k"][arm] = {
                "n": len(cs),
                "prefill_tok_s_median": median([c["prefill_tok_s"] for c in cs]),
                "prefill_range": [min(c["prefill_tok_s"] for c in cs), max(c["prefill_tok_s"] for c in cs)],
                "decode_tok_s_median": median([c["decode_tok_s"] for c in cs]),
                "decode_range": [min(c["decode_tok_s"] for c in cs), max(c["decode_tok_s"] for c in cs)],
            }
    sanity = clean_cells("nightly-100500", "1k")
    if sanity:
        verdict["1k_sanity"] = {"decode_tok_s": sanity[0]["decode_tok_s"],
                                "prefill_tok_s": sanity[0]["prefill_tok_s"]}
    a, b = verdict["128k"].get("nightly-100500"), verdict["128k"].get("nightly-notable")
    if a and b and a["prefill_tok_s_median"] and b["prefill_tok_s_median"]:
        verdict["prefill_table_effect_pct"] = round(
            100.0 * (a["prefill_tok_s_median"] / b["prefill_tok_s_median"] - 1.0), 2)
        verdict["conclusion"] = (
            "prefill gain from the 100500 table on the nightly stack; the T4 r10 deficit was the "
            "table-version fallback" if verdict["prefill_table_effect_pct"] > 10.0 else
            "prefill NOT recovered by the table - the T4 r10 deficit is NOT purely the "
            "table-version effect; report as measured")
    state["verdict"] = verdict
    json.dump(state, open(OUT, "w"), indent=1)
    print(json.dumps(verdict, indent=1), flush=True)


if __name__ == "__main__":
    main()
