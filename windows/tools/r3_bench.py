#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r3_bench.py — throughput measurement on the resident tier (48 GB, warm cache).

Each run: 3 reps — long warmup case (a 512-token warmup rep is discarded first) →
256-token greedy decode; parse prompt/eval tok/s from llama-completion; 5s idle
between reps. Absolute values are only meaningful within the same machine and
window when compared against the reference — cross-window comparison is always
suspect (measure on the same day / same process, or treat numbers as shape-only).

Usage: python tools/r3_bench.py --tier 35b [--reps 3]
"""
import argparse, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_bin_dir():
    """Engine dir — same contract as the app (paths.rs::bin_dir): EDGE0_BIN_DIR override,
    else monorepo depot build <repo>/../wt/win/..., else bootstrap <repo>/.edge0/wt/win/..."""
    env = os.environ.get("EDGE0_BIN_DIR")
    if env:
        return env
    for c in (os.path.join(ROOT, "..", "wt", "win", "build-vk", "bin", "Release"),
              os.path.join(ROOT, ".edge0", "wt", "win", "build-vk", "bin", "Release")):
        if os.path.isfile(os.path.join(c, "llama-completion.exe")):
            return c
    return os.path.join(ROOT, "..", "wt", "win", "build-vk", "bin", "Release")


SERVER_BIN = resolve_bin_dir()


def models_root(tier):
    """Where the GGUF lives — repo-local models/ wins when present (dev layout), else the
    app's model home (EDGE0_HOME, default ~/.edge0) where the download+convert step lands it."""
    home = os.environ.get("EDGE0_HOME") or os.path.join(os.path.expanduser("~"), ".edge0")
    roots = [os.path.join(ROOT, "models"), os.path.join(home, "models")]
    for r in roots:
        if os.path.isfile(os.path.join(r, f"edge0-{tier}-gguf", f"edge0-{tier}.gguf")):
            return r
    raise SystemExit(f"no edge0-{tier}.gguf under {roots[0]} or {roots[1]} "
                     "— load/convert once in the app, or run tools/convert_mlx_to_gguf.py first")


PROMPT = ("混合专家模型（MoE）通过稀疏激活把多个专家网络与门控路由组合起来：" * 12)
PAT = {
    "prompt": re.compile(r"prompt eval time =.*?,\s*([\d.]+) tokens per second"),
    "eval": re.compile(r"(?<!prompt )eval time =.*?,\s*([\d.]+) tokens per second"),
    "total_ms": re.compile(r"total time =\s*([\d.]+) ms"),
    "load_ms": re.compile(r"load time =\s*([\d.]+) ms"),
    "prompt_ms": re.compile(r"prompt eval time =\s*([\d.]+) ms"),
}


def one(gguf, lora, n, tier):
    cmd = [os.path.join(SERVER_BIN, "llama-completion.exe"), "-m", gguf,
           "-ngl", "99", "-cmoe", "--temp", "0", "-n", str(n), "--seed", "7",
           "-no-cnv", "--no-display-prompt", "-p", PROMPT]
    if lora:
        cmd[3:3] = ["--lora", lora]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf8", errors="replace")
    err = r.stderr or ""
    out = {"wall_s": round(time.time() - t0, 1)}
    for k, pat in PAT.items():
        m = pat.search(err)
        if m:
            out[k] = float(m.group(1))
    ev = re.search(r"n_ctx = (\d+)", err)
    out["n_ctx"] = int(ev.group(1)) if ev else None
    if "error" in err.lower() and "eval time" not in err:
        out["error"] = [l for l in err.splitlines() if "error" in l.lower()][:2]
    if "total_ms" in out and "prompt_ms" in out:
        dec_ms = out["total_ms"] - out["prompt_ms"]        # total = generation end-to-end, excludes load (verified)
        out["decode_tps_derived"] = round(n * 1000 / dec_ms, 2) if dec_ms > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["35b", "8b"], required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    exe = os.path.join(SERVER_BIN, "llama-completion.exe")
    if not os.path.isfile(exe):
        raise SystemExit(f"llama-completion.exe not found in {SERVER_BIN} "
                         "— build the engine (scripts/vendor-build.ps1) or set EDGE0_BIN_DIR")
    md = models_root(a.tier)
    gguf = os.path.join(md, f"edge0-{a.tier}-gguf", f"edge0-{a.tier}.gguf")
    lora = os.path.join(md, f"edge0-{a.tier}",
                        f"lora_edge0_{a.tier}-gguf.gguf")
    rows = []
    warm = one(gguf, lora, 64, a.tier)          # warmup (page cache + mmap fault-in), discarded
    print(f"warm(discard): {warm}", flush=True)
    for i in range(a.reps):
        time.sleep(5)
        m = one(gguf, lora, 256, a.tier)
        m["rep"] = i
        rows.append(m)
        print(f"rep{i}: {m}", flush=True)
    ok = [r for r in rows if "eval" in r]
    evs = sorted(r["eval"] for r in ok)
    prs = sorted(r["prompt"] for r in ok if "prompt" in r)
    summ = dict(tier=a.tier, reps=len(ok), decode_tps_med=evs[len(evs) // 2] if evs else None,
                decode_tps_min=evs[0] if evs else None, decode_tps_max=evs[-1] if evs else None,
                prefill_tps_med=prs[len(prs) // 2] if prs else None, rows=rows,
                note="48 GB tier, -cmoe -ngl99 with LoRA; compare against the same-machine reference")
    outp = os.path.join(ROOT, "benchmarks", "r3", f"bench_{a.tier}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(summ, open(outp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("BENCH", json.dumps({k: v for k, v in summ.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
