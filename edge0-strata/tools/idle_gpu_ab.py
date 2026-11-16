"""2507-0: how idle are the GPUs, and does an existing switch close the gap? Interleaved A/B of whole engine runs
(fresh engine per arm run, rotating arm order), with a per-card utilisation sampler (nvidia-smi, 5 samples/s) over every
decode, prompt and batch window. Derived from the 0.2.1 split A/B harness; measurement only, no engine change.

  python split_ab.py single CONFIG OUTDIR --arm name=EXE!ARGS!K=V,K=V ... [--pairs 6] [--new 300] [--warm 2] [--pps 4096,20480] [--extra "..."]
      per run: WARM x 64-token requests, then story/code/facts greedy requests of --new tokens, then one prompt read per --pps length
  python split_ab.py batch CONFIG OUTDIR --batch N --arm ... [--pairs 5] [--new 200]
      per run: engine with --batch N, one warm batch (48 tokens), then --passes batch passes of N prompts (BGEN), aggregate tok/s
  python split_ab.py --summary OUTDIR [--base name]

Results: OUTDIR/results.jsonl, one line per arm run.
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time


class Sampler:
    """nvidia-smi utilisation/power per card, time stamped on arrival; window(t0, t1) -> per-card means."""

    def __init__(self):
        self.rows = []
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw", "--format=csv,noheader,nounits",
                                      "-lms", "200"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                i, u, w = [x.strip() for x in line.split(",")]
                self.rows.append((time.time(), int(i), float(u), float(w)))
            except ValueError:
                pass

    def window(self, t0, t1):
        out = {}
        for t, i, u, w in list(self.rows):
            if t0 <= t <= t1:
                d = out.setdefault(i, [[], []])
                d[0].append(u)
                d[1].append(w)
        return {str(i): {"util": round(sum(d[0]) / len(d[0]), 1), "power_w": round(sum(d[1]) / len(d[1]), 1), "n": len(d[0])}
                for i, d in sorted(out.items())}

    def stop(self):
        self.proc.kill()


SMP = None
from pathlib import Path


def args_of(name):
    return [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == name]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def summary(out, base=None):
    R = [json.loads(l) for l in open(out / "results.jsonl")]
    names = []
    for r in R:
        if r["arm"] not in names:
            names.append(r["arm"])
    base = base or names[0]
    keys = []
    for r in R:
        for x in r["rows"]:
            k = (x["kind"], "tok_s" if x.get("tok_s") is not None else "pp_tok_s")
            if k not in keys and x["kind"] != "batch":
                keys.append(k)
    if any(x["kind"] == "batch" for r in R for x in r["rows"]):
        keys += [("batch", "tok_s"), ("batch", "steady"), ("batch", "ttft")]

    def per_round(arm, kind, key):
        d = {}
        for r in R:
            if r["arm"] == arm:
                v = [x.get(key) for x in r["rows"] if x["kind"] == kind]
                d[r["round"]] = med(v)
        return d

    print("arm | metric | median | vs base | pairs faster (higher) | rounds")
    for kind, key in keys:
        for nm in names:
            a, b = per_round(base, kind, key), per_round(nm, kind, key)
            rounds = [i for i in sorted(set(a) & set(b)) if a[i] == a[i] and b[i] == b[i]]
            if not rounds:
                continue
            ma, mb = med([a[i] for i in rounds]), med([b[i] for i in rounds])
            fb = sum(1 for i in rounds if (b[i] < a[i] if key == "ttft" else b[i] > a[i]))
            print(f"{nm} | {kind}:{key} | {mb:.2f} | {mb / ma:.3f} | {fb}/{len(rounds)} | {len(rounds)}")
    print("\narm | window | mean util per card % (median over runs) | mean power W per card | idle% (mean over cards)")
    kinds = []
    for r in R:
        for x in r["rows"]:
            if x["kind"] not in kinds:
                kinds.append(x["kind"])
    for nm in names:
        for kind in kinds:
            us, ws = {}, {}
            for r in R:
                if r["arm"] != nm:
                    continue
                for x in r["rows"]:
                    if x["kind"] == kind:
                        for gi, g in (x.get("gpu") or {}).items():
                            us.setdefault(gi, []).append(g["util"])
                            ws.setdefault(gi, []).append(g["power_w"])
            if us:
                mu = {gi: med(v) for gi, v in us.items()}
                print(f"{nm} | {kind} | {', '.join(f'gpu{gi} {v:.0f}' for gi, v in sorted(mu.items()))} | "
                      f"{', '.join(f'{med(v):.0f}' for gi, v in sorted(ws.items()))} | {100 - sum(mu.values()) / len(mu):.0f}")
    print("\nanswer hashes per window kind (distinct values over all runs of all arms; 1 = byte-identical)")
    for kind in kinds:
        hs = {}
        for r in R:
            for x in r["rows"]:
                if x["kind"] == kind and x.get("sha"):
                    hs.setdefault(x["sha"], set()).add(r["arm"])
        print(f"{kind}: {len(hs)} distinct" + ("" if len(hs) == 1 else " -> " + "; ".join(f"{h}:{','.join(sorted(a))}" for h, a in hs.items())))
    for nm in names:
        sl = sorted({r.get("slots") for r in R if r["arm"] == nm}, key=str)
        print(f"{nm}: expert-cache slots {sl}")


if "--summary" in sys.argv:
    summary(Path(arg("--summary", ".")), arg("--base", None))
    sys.exit(0)

MODE = sys.argv[1]
REPO = Path(os.environ["STRATA_REPO"])
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
import strata_tokenizer as ST  # noqa: E402
from serve.server import child_env  # noqa: E402

SMP = Sampler()
CFG, OUT = sys.argv[2], Path(sys.argv[3])
PAIRS = int(arg("--pairs", "5"))
NEW = int(arg("--new", "300" if MODE == "single" else "200"))
WARM = int(arg("--warm", "2"))
PPS = [int(x) for x in arg("--pps", "0").split(",") if int(x)]
BATCH = int(arg("--batch", "2"))
PASSES = int(arg("--passes", "2"))
extra = arg("--extra", "").split()
ARMS = []
for a in args_of("--arm"):
    nm, rest = a.split("=", 1)
    exe, xargs, envs = (rest.split("!") + ["", ""])[:3]
    ARMS.append((nm, exe, xargs.split(), dict(kv.split("=", 1) for kv in envs.split(",") if kv)))
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(CFG, encoding="utf-8-sig"))
tp = Path(cfg["tokenizer"])
vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((tp / "token_type.json").read_text()))
HEAD = tok.encode("<|im_start|>user\n", parse_special=True)
TAIL = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
KINDS = ["story", "code", "facts"]
ASK = {
    "story": "Write a long story about a lighthouse keeper who finds a message in a bottle. Make it detailed.",
    "code": "Write a Python program that implements an LRU cache class with get and put, plus unit tests. Explain each part.",
    "facts": "Give me twenty well-known facts about the planets of the solar system, one per line, with the key numbers for each.",
}
QUESTIONS = [
    "Explain in detail how a B-tree works.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Tell the history of the calculation of pi, from Archimedes to today.",
]
_DOC = []


def doc_ids(n, start=30000):
    if not _DOC:
        txt = ""
        for f in sorted((REPO / "docs").glob("*.md")):
            txt += f.read_text(encoding="utf-8", errors="replace") + "\n"
        txt = txt * 8
        _DOC.extend(tok.encode(txt[start:], parse_special=False))
    return _DOC[:n]


def ids_for(name, n=0):
    if name == "pp":
        return HEAD + tok.encode("Summarise:\n\n", parse_special=False) + doc_ids(n, 5000) + tok.encode("\n\nOne line.", parse_special=False) + TAIL
    if name.startswith("q"):
        return HEAD + tok.encode(QUESTIONS[int(name[1:])], parse_special=False) + TAIL
    return HEAD + tok.encode(ASK[name], parse_special=False) + TAIL


def clean_env(env_extra):
    env = child_env(cfg)
    for k in list(env):
        if k.startswith("STRATA_"):
            env.pop(k, None)
    env["STRATA_DECODE_TIMING"] = "1"
    env.update(env_extra)
    return env


def finish(p, log):
    try:
        p.stdin.close()
    except Exception:
        pass
    try:
        p.wait(120)
    except Exception:
        p.kill()
    txt = open(log, errors="replace").read()
    slots = re.findall(r"expert cache (\d+) slots", txt)
    notes = [l for l in txt.splitlines() if re.search(r"batch_groups|pipeline.windows|SPLIT_OVERLAP|handoff", l, re.I)][:4]
    return (slots[0] if slots else None), notes


def start(exe, args, env, log):
    lf = open(log, "w")
    p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=lf, text=True, bufsize=1, env=env)
    for line in p.stdout:
        if line.startswith("READY"):
            return p
    raise SystemExit(f"engine ended before READY: {log}")


def gen(p, ids, n):
    p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n")
    p.stdin.flush()
    got, info = [], ""
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            info = line
            break
    return got, info


def kv_of(info):
    f = info.split()
    if len(f) < 5 or f[0] != "DONE":
        return {}
    return {"generated": f[1], "prompt_tokens": f[2], "prompt_ms": f[3], "decode_ms": f[4]}


def run_single(label, exe, xargs, env_extra):
    log = OUT / f"{label}.log"
    p = start(exe, list(cfg["args"]) + extra + xargs, clean_env(env_extra), str(log))
    for i in range(WARM):
        gen(p, ids_for(KINDS[i % len(KINDS)]), 64)
    rows = []
    for name in KINDS:
        t0 = time.time()
        out, info = gen(p, ids_for(name), NEW)
        t1 = time.time()
        d = kv_of(info)
        g, dms = d.get("generated"), d.get("decode_ms")
        rows.append({"kind": name, "gen": len(out), "tok_s": (len(out) / float(dms) * 1000.0) if dms else None, "ids": out, "info": info,
                     "sha": hashlib.sha256(",".join(map(str, out)).encode()).hexdigest()[:16], "gpu": SMP.window(t0, t1)})
    for n in PPS:
        ids = ids_for("pp", n)
        t0 = time.time()
        out, info = gen(p, ids, 8)
        t1 = time.time()
        d = kv_of(info)
        pms = d.get("prompt_ms")
        rows.append({"kind": f"pp{n // 1024}k", "pp_tokens": len(ids), "pp_tok_s": (len(ids) / float(pms) * 1000.0) if pms else None,
                     "ids": out, "info": info, "tok_s": None, "gpu": SMP.window(t0, t1),
                     "sha": hashlib.sha256(",".join(map(str, out)).encode()).hexdigest()[:16]})
    slots, notes = finish(p, log)
    return rows, slots, notes


def batch_pass(p, n_new, n):
    prompts = [ids_for(f"q{i % len(QUESTIONS)}") for i in range(n)]
    got = {i: [] for i in range(n)}
    first = {}
    sent = {}
    done = {}
    stamp = []
    t0 = time.time()
    for i, ids in enumerate(prompts):
        sent[i] = time.time()
        p.stdin.write(f"BGEN {i} {n_new} {','.join(map(str, ids))}\n")
        p.stdin.flush()
        for line in p.stdout:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
                first.setdefault(i, time.time())
                stamp.append(time.time())
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                first.setdefault(int(s), time.time())
                stamp.append(time.time())
            elif line.startswith("BDONE "):
                done[int(line.split()[1])] = line
            elif line.startswith("ERR"):
                raise SystemExit("batch ERR: " + line)
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_adm = time.time()
    while len(done) < n:
        line = p.stdout.readline()
        if not line:
            raise SystemExit("engine ended in batch")
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            first.setdefault(int(s), time.time())
            stamp.append(time.time())
        elif line.startswith("BDONE "):
            done[int(line.split()[1])] = line
        elif line.startswith("ERR"):
            raise SystemExit("batch ERR: " + line)
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    late = sum(1 for s in stamp if s > t_adm)
    return {"kind": "batch", "tokens": total, "tok_s": total / (t_end - t0), "steady": late / max(t_end - t_adm, 1e-9),
            "gpu": SMP.window(t_adm, t_end), "sha": hashlib.sha256(json.dumps([got[i] for i in range(n)]).encode()).hexdigest()[:16],
            "ttft": med([first[i] - sent[i] for i in first]), "ids": [got[i] for i in range(n)], "secs": t_end - t0}


def run_batch(label, exe, xargs, env_extra):
    log = OUT / f"{label}.log"
    p = start(exe, list(cfg["args"]) + ["--batch", str(BATCH)] + extra + xargs, clean_env(env_extra), str(log))
    batch_pass(p, 48, BATCH)
    rows = [batch_pass(p, NEW, BATCH) for _ in range(PASSES)]
    for r in rows:
        r["pass"] = rows.index(r)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    slots, notes = finish(p, log)
    return rows, slots, notes


res = OUT / "results.jsonl"
ROUND0 = int(arg("--round0", "0"))
only = arg("--only", None)
for rnd in range(ROUND0, ROUND0 + (1 if only else PAIRS)):
    order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
    for nm, exe, xargs, e in order:
        if only and nm not in only.split(","):
            continue
        fn = run_single if MODE == "single" else run_batch
        rows, slots, notes = fn(f"{nm}_{rnd}", exe, xargs, e)
        with open(res, "a") as f:
            f.write(json.dumps({"round": rnd, "arm": nm, "slots": slots, "notes": notes, "rows": rows}) + "\n")
        if MODE == "single":
            m = med([r["tok_s"] for r in rows if r.get("tok_s")])
            pp = [round(r.get("pp_tok_s") or 0) for r in rows if r.get("pp_tok_s")]
            print(f"round {rnd} {nm}: decode median {m:.2f} tok/s, pp {pp} slots {slots} {notes[:1]}", flush=True)
        else:
            print(f"round {rnd} {nm}: agg {[round(r['tok_s'], 1) for r in rows]} steady {[round(r['steady'], 1) for r in rows]} "
                  f"ttft {[round(r['ttft'], 2) for r in rows]} slots {slots} {notes[:2]}", flush=True)
