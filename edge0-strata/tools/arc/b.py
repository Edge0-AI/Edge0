#!/usr/bin/env python3
"""b.py (ARC-W): one engine load on the Arc B65 box, a list of solo requests over the engine's line protocol.

  b.py --engine E --tag T --plan "short:128:2,p1k:8:3,p4k:8:2,p16k:8:2" [--env K=V ...] [--extra "--spec 2 ..."]

plan item = case:max_new:repeat (cases are the token-id files in /mnt/models/strata-test/prompts).  Output: out/<tag>.json
with per-request prompt tok/s, decode tok/s and the tokens; one summary line per request on stderr.  A new first token per
repeat keeps the prompt from being served out of the conversation cache (the case then differs by 1 token, same for every arm).
"""
import argparse, json, os, queue, shlex, subprocess, sys, threading, time

ROOT = "/mnt/models/strata-test"
W = f"{ROOT}/arc037"
M = "/mnt/models/llama/Qwen3.8-Flash-Next-GSQ-RCO-ISTA-DASLab"
S1 = f"{M}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
S2 = f"{M}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--plan", default="short:128:2,p1k:8:3,p4k:8:2,p16k:8:2")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--extra", default="--spec 2 --spec-min-p 0.5 --pipeline-windows 2 --mtp-draft-vocab " + f"{W}/src/data/draft_vocab_en.bin")
    ap.add_argument("--layer-split", default="auto")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    os.makedirs(f"{W}/out", exist_ok=True)
    env = dict(os.environ)
    env.update({"ONEAPI_DEVICE_SELECTOR": "level_zero:*", "SYCL_CACHE_PERSISTENT": "0", "ZES_ENABLE_SYSMAN": "1",
                "STRATA_VERIFY_NO_HOST": "1"})
    if a.gpus:
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:" + a.gpus
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    args = ["--serve", "--pack", f"{ROOT}/data/packs/iq3_s", "--native", S1, "--ple-gguf", S2,
            "--expert-profile", f"{W}/src/data/expert-profile-sycl.bin", "--expert-cache", "auto", "--prefill", "auto",
            "--mtp", f"{ROOT}/data/mtp/rt", "--max-context", str(a.max_context), "--kv", "int8", "--ple-io", "ram",
            "--stream-experts", "--vram-reserve-mib", "1024", "--layer-split", a.layer_split] + shlex.split(a.extra)
    cmdline = "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec " + " ".join(shlex.quote(x) for x in [a.engine] + args)
    log = open(f"{W}/out/{a.tag}.err", "w")
    p = subprocess.Popen(["bash", "-c", cmdline], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: ([q.put(l.rstrip("\n")) for l in p.stdout], q.put("__EOF__")), daemon=True).start()

    def prompt(case):
        return [int(x) for x in open(f"{ROOT}/prompts/{case}.txt").read().split()]

    def gen(ids, n):
        p.stdin.write(f"GEN {n} " + ",".join(map(str, ids)) + "\n"); p.stdin.flush()
        toks, t0 = [], time.time()
        while True:
            try:
                l = q.get(timeout=1.0)
            except queue.Empty:
                if p.poll() is not None: raise RuntimeError("engine exited %s" % p.returncode)
                if time.time() - t0 > a.timeout: raise TimeoutError("request")
                continue
            if l == "__EOF__": raise RuntimeError("engine closed stdout")
            if l.startswith("T "): toks.append(int(l.split()[1]))
            elif l.startswith("DONE"): return l.split(), toks
            elif l.startswith("ERR"): raise RuntimeError(l)

    res = []
    try:
        d, _ = gen(prompt("short"), 8)   # warm-up
        for item in a.plan.split(","):
            case, n, rep = item.split(":")
            for r in range(int(rep)):
                ids = [1000 + r] + prompt(case)
                d, toks = gen(ids, int(n))
                g, pt, pms, dms = int(d[1]), int(d[2]), float(d[3]), float(d[4])
                rec = {"case": case, "rep": r, "prompt_tokens": pt, "prompt_ms": pms, "pp": pt / pms * 1000 if pms else 0,
                       "generated": g, "decode_ms": dms, "tg": g / dms * 1000 if dms else 0, "tokens": toks}
                res.append(rec)
                print(f"[{a.tag}] {case}#{r} pt={pt} pp={rec['pp']:.1f} gen={g} tg={rec['tg']:.1f}", file=sys.stderr, flush=True)
    finally:
        try: p.stdin.write("QUIT\n"); p.stdin.flush()
        except Exception: pass
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: p.kill()
        log.close()
        json.dump({"tag": a.tag, "engine": a.engine, "env": a.env, "extra": a.extra, "results": res}, open(f"{W}/out/{a.tag}.json", "w"))


main()
