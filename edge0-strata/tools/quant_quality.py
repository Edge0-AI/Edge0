#!/usr/bin/env python3
"""The quant quality yardstick (MODEL-UPDATE.md, Phase 0): teacher-forced top-1 agreement and KL of a quant against
Q8_0, on one fixed text set (data/quant_quality/v1: code, docs, English, other languages, an agent transcript).

    python tools/quant_quality.py ids --tokenizer DIR [--set data/quant_quality/v1] [--out set.ids.json]
        tokenizes the set (each segment cut to --seg-tokens, default 1000) - the ids file is committed with the set
        (data/quant_quality/v1/set.ids.json), so a box without the tokenizer scores the same positions
    python tools/quant_quality.py run --config CFG.json --out RUN.tsv [--ids set.ids.json] [--env K=V ...] [--gpu-experts]
        one `strata --serve` engine (the config's exe and args), every segment one request whose tokens are all read
        through the decode windows (--short-read), where STRATA_LOGPOS / STRATA_LOGPOS_TOPK=64 write, per position,
        the log-probability of the text's next token and the 64 most likely tokens.  Fixed: the experts on the CPU
        (one VRAM slot, --expert-cache 1: the verify windows need a profile-filled tier, so the config must have
        --expert-profile; --gpu-experts: 1500 slots), no expert adaptation, no PCIe share, no suffix drafts, no prompt
        cache, FP16 K/V, STRATA_ROUTE_TAIL_SKIP=0, STRATA_PREFILL_CPU_SHARE=0.  RUN.tsv + RUN.tsv.seg.json (line ranges).
    python tools/quant_quality.py compare REF.tsv RUN.tsv [...]   (REF = the Q8_0 run)
        per segment and overall: top-1 agreement (same argmax), mean / p99 KL(REF || RUN) over REF's 64 most likely
        tokens (both renormalized to that set; a token RUN does not list gets RUN's 64th log-probability), and the
        perplexity of the text under each.

The RUN.tsv rows are the engine's STRATA_LOGPOS format: pos, target id, logp(target), argmax id, logp(argmax),
argmax==target, logp(<|im_end|>), logp(target without <|im_end|>), then 64 `id:logp` fields (natural log).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SET_DIR = HERE.parent / "data" / "quant_quality" / "v1"
SEGMENTS = ["code", "docs", "english", "multilingual", "agent"]
TOPK = 64
FIXED = ["--expert-cache", "1", "--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0",
         "--prompt-cache", "0", "--kv", "fp16"]
FIXED_ENV = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_PCIE_FRAC_DEFAULT": "old"}
# config args the yardstick replaces (value-taking) or drops (flags)
DROP_VALUE = {"--expert-cache", "--adapt-every", "--pcie-frac", "--suffix-draft", "--prompt-cache", "--max-context",
              "--prefill", "--prompt-cache-every", "--short-read", "--conversation-cache-mib", "--eos-ids",
              "--kv", "--kv-resident"}   # (--resident-budget-gib stays: a PC whose RAM does not hold the experts)
DROP_FLAG = {"--serve", "--vision", "--resident-experts", "--resident-cpu-experts"}


# ------------------------------------------------------------------------------------------------ ids
def cmd_ids(a) -> int:
    sys.path.insert(0, str(HERE))
    import strata_tokenizer as ST  # noqa: E402
    tp = pathlib.Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    segs = []
    for name in SEGMENTS:
        text = (pathlib.Path(a.set) / f"{name}.txt").read_text(encoding="utf-8")
        ids = tok.encode(text, parse_special=(name == "agent"))
        if len(ids) < a.seg_tokens:
            raise SystemExit(f"{name}: only {len(ids)} tokens (< {a.seg_tokens})")
        segs.append({"name": name, "ids": ids[:a.seg_tokens]})
        print(f"{name}: {len(ids)} tokens, kept {a.seg_tokens}")
    out = pathlib.Path(a.out or (pathlib.Path(a.set) / "set.ids.json"))
    out.write_text(json.dumps({"version": pathlib.Path(a.set).name, "seg_tokens": a.seg_tokens, "segments": segs}))
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------------------------------------ run
def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


def engine_args(cfg: dict, gpu_experts, seg_tokens: int) -> list[str]:
    base, out, i = list(cfg["args"]), [], 0
    while i < len(base):
        x = base[i]
        if x in DROP_FLAG:
            i += 1
            continue
        if x in DROP_VALUE:
            i += 2
            continue
        out.append(x)
        i += 1
    fixed = list(FIXED)
    if gpu_experts:
        fixed[1] = gpu_experts if isinstance(gpu_experts, str) else "1500"
    return [cfg["exe"], "--serve", *out, *fixed, "--max-context", str(seg_tokens + 512),
            "--short-read", str(seg_tokens + 64), "--prefill", "512"]


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    out = pathlib.Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_LOGPOS"] = str(out)
    env["STRATA_LOGPOS_TOPK"] = str(TOPK)
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    args = engine_args(cfg, a.gpu_experts, st["seg_tokens"])
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(str(out) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")
    segs, info = [], []

    def lines() -> int:
        return sum(1 for _ in open(out)) if out.exists() else 0

    for s in st["segments"]:
        n0, t0 = lines(), time.time()
        p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"])) + "\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            if line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE") or "nonfinite" in done:
            p.kill()
            raise SystemExit(f"{s['name']}: the engine answered {done!r}; see {out}.stderr.txt")
        n1 = lines()
        segs.append({"name": s["name"], "first_line": n0, "end_line": n1})
        info.append(f"{s['name']}: {n1 - n0} positions, {time.time() - t0:.1f} s")
        print(info[-1], flush=True)
        if n1 - n0 < len(s["ids"]) // 2:
            p.kill()
            raise SystemExit(f"{s['name']}: only {n1 - n0} positions were scored (the prompt took the batched path?)")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    err = pathlib.Path(str(out) + ".stderr.txt").read_text(errors="replace")
    cache = [l for l in err.splitlines() if "expert cache" in l][:3]
    pathlib.Path(str(out) + ".seg.json").write_text(json.dumps({"segments": segs, "config": a.config,
                                                                 "args": args, "env": a.env or [],
                                                                 "expert_cache_lines": cache}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ prompt path
PROMPT_LENS = (256, 512, 768, 1000)


def cmd_prompt(a) -> int:
    """The batched prompt path's answer: every segment's first 256 / 512 / 768 / 1000 tokens as a prompt (one request
    each, no prompt cache), and the logits after it (STRATA_DUMP_FIRST_LOGITS: float32, the whole vocabulary)."""
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("first.*"):
        f.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_DUMP_FIRST_LOGITS"] = str(out / "first")
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    args = engine_args(cfg, a.gpu_experts, max(PROMPT_LENS))
    k = args.index("--short-read")
    del args[k:k + 2]                      # the default (64): these prompts take the batched prompt path
    k = args.index("--prefill")
    args[k + 1] = "auto"
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(out / "stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}/stderr.txt")
    index = []
    for s in st["segments"]:
        for n in PROMPT_LENS:
            p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"][:n])) + "\n")
            p.stdin.flush()
            done = None
            for line in p.stdout:
                if line.startswith(("DONE", "ERR")):
                    done = line.strip()
                    break
            if done is None or not done.startswith("DONE") or "nonfinite" in done:
                p.kill()
                raise SystemExit(f"{s['name']} {n}: the engine answered {done!r}")
            index.append({"segment": s["name"], "tokens": n})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    (out / "index.json").write_text(json.dumps({"requests": index, "args": args, "env": a.env or []}, indent=1))
    print(f"{len(index)} prompts")
    return 0


def cmd_pcompare(a) -> int:
    import numpy as np
    ref = pathlib.Path(a.ref)
    idx = json.loads((ref / "index.json").read_text())["requests"]

    def logp(d, i):
        x = np.fromfile(d / f"first.{i}", dtype=np.float32).astype(np.float64)
        x -= x.max()
        return x - np.log(np.exp(x).sum())

    print(f"reference: {ref}")
    for run in a.runs:
        r = pathlib.Path(run)
        kls, agree = [], 0
        for i in range(len(idx)):
            if not (r / f"first.{i}").exists():
                continue
            lr, la = logp(ref, i), logp(r, i)
            pr = np.exp(lr)
            kls.append(float((pr * (lr - la)).sum()))
            agree += int(lr.argmax() == la.argmax())
        kls.sort()
        n = len(kls)
        print(f"{r.name:40s} prompts {n:3d}  top-1 {100 * agree / max(n, 1):6.2f}%  KL mean {sum(kls) / max(n, 1):.5f}  "
              f"median {kls[n // 2] if n else float('nan'):.5f}  max {kls[-1] if n else float('nan'):.5f}")
    return 0


# ------------------------------------------------------------------------------------------------ compare
def read_run(path: str) -> dict:
    rows = open(path).read().splitlines()
    segs = json.load(open(path + ".seg.json"))["segments"]
    out = {}
    for s in segs:
        d = {}
        for line in rows[s["first_line"]:s["end_line"]]:
            f = line.split("\t")
            if len(f) < 8 + TOPK:
                continue
            d[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]), [(int(x), float(y)) for x, y in
                                                                (z.split(":") for z in f[8:8 + TOPK])])
        out[s["name"]] = d
    return out


def compare(ref: dict, run: dict) -> dict:
    res, allk, tot = {}, [], {"n": 0, "agree": 0, "nll_r": 0.0, "nll_a": 0.0, "kl": 0.0}
    for name, rr in ref.items():
        ra = run.get(name, {})
        pos = sorted(set(rr) & set(ra))
        n = agree = 0
        kls, nll_r, nll_a = [], 0.0, 0.0
        for q in pos:
            tr, lr, topr, kr = rr[q]
            ta, la, topa, ka = ra[q]
            if tr != ta:
                raise SystemExit(f"{name} pos {q}: target {tr} != {ta} (not the same text)")
            n += 1
            agree += topr == topa
            nll_r -= lr
            nll_a -= la
            da = dict(ka)
            floor = min(b for _, b in ka)
            pr = [math.exp(b) for _, b in kr]
            pa = [math.exp(da.get(i, floor)) for i, _ in kr]
            sr, sa = sum(pr), sum(pa)
            kls.append(sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0))
        allk += kls
        if n:
            res[name] = {"positions": n, "top1": agree / n, "kl": sum(kls) / n,
                         "ppl_ref": math.exp(nll_r / n), "ppl": math.exp(nll_a / n)}
        tot["n"] += n
        tot["agree"] += agree
        tot["nll_r"] += nll_r
        tot["nll_a"] += nll_a
        tot["kl"] += sum(kls)
    n = max(tot["n"], 1)
    allk.sort()
    res["all"] = {"positions": tot["n"], "top1": tot["agree"] / n, "kl": tot["kl"] / n,
                  "kl_p99": allk[int(0.99 * (len(allk) - 1))] if allk else float("nan"),
                  "ppl_ref": math.exp(tot["nll_r"] / n), "ppl": math.exp(tot["nll_a"] / n)}
    return res


def cmd_compare(a) -> int:
    ref = read_run(a.ref)
    print(f"reference: {a.ref}")
    print(f"{'run':40s} {'segment':13s} {'pos':>5s} {'top1':>7s} {'KL':>8s} {'KL p99':>8s} {'PPL':>7s} {'PPL ref':>7s}")
    allres = {}
    for path in a.runs:
        r = compare(ref, read_run(path))
        allres[path] = r
        for seg, v in r.items():
            if seg != "all" and not a.segments:
                continue
            print(f"{pathlib.Path(path).name[:40]:40s} {seg:13s} {v['positions']:5d} {100 * v['top1']:6.2f}% "
                  f"{v['kl']:8.5f} {v.get('kl_p99', float('nan')):8.4f} {v['ppl']:7.3f} {v['ppl_ref']:7.3f}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(allres, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--set", default=str(SET_DIR))
    p.add_argument("--seg-tokens", type=int, default=1000)
    p.add_argument("--out")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--env", action="append")
    p.add_argument("--gpu-experts", nargs="?", const=True, default=False,
                   help="--expert-cache 1500 (or the N given, e.g. auto) instead of one slot (the experts on the GPU kernels)")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap", help="a command prefix for the engine, e.g. 'systemd-run --user --scope -p MemoryMax=200G'")
    p = sub.add_parser("prompt", help="the batched prompt path's next-token logits after 20 prompts (a directory)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--env", action="append")
    p.add_argument("--gpu-experts", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--segments", action="store_true")
    p.add_argument("--json")
    p = sub.add_parser("pcompare", help="full-vocabulary KL and top-1 of `prompt` directories against a reference one")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    a = ap.parse_args()
    return {"ids": cmd_ids, "run": cmd_run, "prompt": cmd_prompt, "compare": cmd_compare,
            "pcompare": cmd_pcompare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
