#!/usr/bin/env python3
"""Split serving (serve/split.py): the exactness check and the speed comparison, run on a test box with the real engine.

  python3 tools/split_serve_check.py exact CONFIG.json OUT.json [--tokens N]
  python3 tools/split_serve_check.py bench CONFIG.json OUT.json --arm layer|single|split [--reps 3]

Run it from the repository root (it imports serve.server and serve.split).  CONFIG.json:

  {"exe": "/path/strata", "cwd": "/path/repo", "args": ["--serve", "--pack", ..., "--max-context", "40960", ...],
   "env": {...}, "prompts": "/path/prompts", "dir": "/path/work",
   "prefill_gpu": 0, "decode_gpu": 1, "arena": "/path/arena.bin" (optional), "min_prompt_tokens": 2048}

`args` holds no --layer-split, --shared-expert-arena or card choice: the tool adds them for each arm.
The prompts folder has pp_<N>.txt files of comma-separated token ids (the bench harness's).

exact: a fresh engine reads a prompt and answers (the reference, twice, to show the engine's own run-to-run noise);
then the prefill engine reads the same prompt, saves the session, a fresh decode engine restores it and answers.  Compared:
the state fingerprint (STRATA_STATE_HASH) of the prefill engine and of the decode engine right after the restore, then the
answer's token ids and the fingerprint after the answer.

bench: one arm per call (run it in its own job so the engines start fresh):
  layer  one engine over both cards (--layer-split auto), the normal two-card setup
  single one engine on the decode card
  split  prefill engine on one card, decode engine on the other, through SplitEngine (the production path)
Workloads: 8K and 32K prompt + 256 tokens, a 3-turn chat with short follow-ups, a 3-turn chat with long follow-ups.
"""
import argparse
import copy
import json
import os
import re
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve import server as srv          # noqa: E402
from serve import split as split_mod     # noqa: E402


def read_ids(path):
    return [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]


def session_dir(cfg):
    """Where the session files go: cfg["session_dir"], else /dev/shm when it has room (as the server chooses), else cfg["dir"]."""
    d = cfg.get("session_dir")
    if not d and os.path.isdir("/dev/shm"):
        import shutil
        if shutil.disk_usage("/dev/shm").free > (6 << 30):
            d = "/dev/shm/strata-split-check"
    d = d or cfg["dir"]
    os.makedirs(d, exist_ok=True)
    return d


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def make_engine(cfg, gpus, tag, extra_args=(), arena=None, min_free_gib=8.0, state_hash=False):
    free = mem_available_gib()
    if free < min_free_gib:
        raise SystemExit(f"only {free:.1f} GiB of RAM available (< {min_free_gib}): not starting an engine")
    c = copy.deepcopy(cfg)
    c["gpu"] = gpus
    c["args"] = list(c["args"]) + list(extra_args) + (["--shared-expert-arena", arena] if arena else [])
    c["log"] = os.path.join(cfg["dir"], f"engine-{tag}.log")
    open(c["log"], "w").close()
    env = srv.child_env(c)
    if state_hash:                  # the fingerprint after every request costs time proportional to the prompt: exact mode only
        env["STRATA_STATE_HASH"] = "1"
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    t0 = time.time()
    e = srv.StrataEngine(c["exe"], srv.engine_args(c), cwd=c.get("cwd"), log=c["log"], env=env)
    e.silence_s = 0
    e.tag, e.load_s = tag, time.time() - t0
    return e


def log_tail(e, start):
    try:
        with open(e.log_path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size(e):
    try:
        return os.path.getsize(e.log_path)
    except OSError:
        return 0


def state_hashes(text):
    """The STATE_HASH lines of an engine log excerpt, as dicts."""
    out = []
    for m in re.finditer(r"STATE_HASH L=(\d+) (.*)", text):
        d = {"L": int(m.group(1))}
        d.update(dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv))
        out.append(d)
    return out


def run_gen(engine, ids, max_new):
    """One request through engine.generate (a StrataEngine or a SplitEngine).  -> dict of tokens and times."""
    cancel = threading.Event()
    toks, t_first = [], None
    t0 = time.perf_counter()
    for t in engine.generate(ids, max_new, {}, cancel):
        if t is None:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        toks.append(t)
    t_end = time.perf_counter()
    last = dict(engine.last or {})
    dec = (len(toks) - 1) / (t_end - t_first) if t_first and len(toks) > 1 and t_end > t_first else None
    return {"tokens": toks, "ttft_s": (t_first - t0) if t_first else None, "wall_s": t_end - t0,
            "dec_tok_s": dec, "last": last}


def close_all(*engines):
    for e in engines:
        try:
            e.close()
        except Exception as x:       # noqa: BLE001
            print("close:", x, flush=True)


def cmd_exact(a, cfg):
    ids = read_ids(os.path.join(cfg["prompts"], f"pp_{a.prompt}.txt"))[:a.prompt]
    arena = cfg.get("arena")
    P, Dg = cfg["prefill_gpu"], cfg["decode_gpu"]
    res = {"prompt_tokens": len(ids), "new_tokens": a.tokens, "runs": {}}
    # 1. the reference twice: a fresh engine on the decode card each time
    for k in (1, 2):
        e = make_engine(cfg, Dg, f"ref{k}", arena=arena, state_hash=True)
        pos = log_size(e)
        r = run_gen(e, ids, a.tokens)
        h = state_hashes(log_tail(e, pos))
        res["runs"][f"ref{k}"] = {"tokens": r["tokens"], "hash_end": h[-1] if h else None, "last": r["last"],
                                  "wall_s": r["wall_s"]}
        print(f"ref{k}: {len(r['tokens'])} tokens, wall {r['wall_s']:.1f} s", flush=True)
        close_all(e)
    # 2. the split: prefill engine, save, fresh decode engine, restore, answer
    pe = make_engine(cfg, P, "prefill", arena=arena, state_hash=True)
    pos = log_size(pe)
    cancel = threading.Event()
    for _ in pe.generate(ids, 0, {}, cancel, prefill_only=True):
        pass
    hp = state_hashes(log_tail(pe, pos))
    path = os.path.join(session_dir(cfg), "exact.session")
    saved = pe.session_file("save", path)
    print(f"prefill engine: held {saved['tokens']} tokens, saved {saved['bytes'] / 2**20:.0f} MiB in {saved['ms']:.0f} ms",
          flush=True)
    de = make_engine(cfg, Dg, "decode", arena=arena, state_hash=True)
    restored = de.session_file("restore", path)
    print(f"decode engine: restored {restored['tokens']} tokens in {restored['ms']:.0f} ms", flush=True)
    pos = log_size(de)
    for _ in de.generate(ids, 0, {}, cancel, prefill_only=True):    # nothing to read: only the state fingerprint
        pass
    hd = state_hashes(log_tail(de, pos))
    pos = log_size(de)
    r = run_gen(de, ids, a.tokens)
    h = state_hashes(log_tail(de, pos))
    res["runs"]["split"] = {"tokens": r["tokens"], "hash_end": h[-1] if h else None, "last": r["last"],
                            "wall_s": r["wall_s"], "hash_prefill_engine": hp[-1] if hp else None,
                            "hash_decode_after_restore": hd[-1] if hd else None, "saved": saved, "restored": restored}
    print(f"split: {len(r['tokens'])} tokens, wall {r['wall_s']:.1f} s", flush=True)
    close_all(pe, de)
    try:
        os.unlink(path)
    except OSError:
        pass
    # verdicts
    ref1, ref2, sp = (res["runs"][k] for k in ("ref1", "ref2", "split"))

    def first_diff(x, y):
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return i
        return None if len(x) == len(y) else min(len(x), len(y))
    v = {
        "restore_round_trip_state_equal": sp["hash_prefill_engine"] == sp["hash_decode_after_restore"] and
                                          sp["hash_prefill_engine"] is not None,
        "ref1_vs_ref2_first_diff": first_diff(ref1["tokens"], ref2["tokens"]),
        "ref1_vs_split_first_diff": first_diff(ref1["tokens"], sp["tokens"]),
        "ref2_vs_split_first_diff": first_diff(ref2["tokens"], sp["tokens"]),
        "end_state_ref1_eq_ref2": ref1["hash_end"] == ref2["hash_end"],
        "end_state_ref1_eq_split": ref1["hash_end"] == sp["hash_end"],
    }
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(v, indent=1), flush=True)
    # the check passes when the split run equals the reference wherever the reference equals itself
    same_ref = v["ref1_vs_ref2_first_diff"] is None
    ok = v["restore_round_trip_state_equal"] and (
        (v["ref1_vs_split_first_diff"] is None and v["end_state_ref1_eq_split"]) if same_ref
        else v["ref1_vs_split_first_diff"] is None or v["ref2_vs_split_first_diff"] is None)
    print("EXACT: PASS" if ok else "EXACT: FAIL", flush=True)
    return 0 if ok else 1


WORKLOADS = [
    ("p8k", {"prompt": 8192, "gen": 256}),
    ("p32k", {"prompt": 32768, "gen": 256}),
    ("chat_short", {"prompt": 4096, "gen": 128, "turns": 3, "follow": 200}),
    ("chat_long", {"prompt": 8192, "gen": 128, "turns": 3, "follow": 6000}),
]


def follow_ids(cfg, n, turn):
    """n token ids for a follow-up message: a slice of the 16K prompt (different for each turn)."""
    pool = read_ids(os.path.join(cfg["prompts"], "pp_16384.txt"))
    start = 1000 + turn * 7000
    return pool[start:start + n]


def cmd_bench(a, cfg):
    arm = a.arm
    P, Dg = cfg["prefill_gpu"], cfg["decode_gpu"]
    arena = cfg.get("arena")
    t_load = time.time()
    if arm == "layer":
        eng = make_engine(cfg, [min(P, Dg), max(P, Dg)], "layer")
        closers = [eng]
    elif arm == "single":
        eng = make_engine(cfg, Dg, "single")
        closers = [eng]
    elif arm == "single_shared":                 # one card, the arena in the shared file: what the split's engines read from
        eng = make_engine(cfg, Dg, "single_shared", arena=arena)
        closers = [eng]
    elif arm == "split":
        spec = split_mod.split_spec({"split_serve": {"prefill_gpu": P, "decode_gpu": Dg,
                                                    "min_prompt_tokens": cfg.get("min_prompt_tokens", 2048)}})
        pe = make_engine(cfg, P, "prefill", extra_args=cfg.get("prefill_args", []), arena=arena)
        de = make_engine(cfg, Dg, "decode", extra_args=cfg.get("decode_args", []), arena=arena)
        eng = split_mod.SplitEngine(pe, de, min_tokens=spec["min_prompt_tokens"], work_dir=session_dir(cfg), say=print)
        closers = [pe, de]
    else:
        raise SystemExit("--arm layer|single|split")
    load_s = time.time() - t_load
    print(f"[{arm}] loaded in {load_s:.0f} s", flush=True)
    rows = []
    try:
        warm = read_ids(os.path.join(cfg["prompts"], "dec_story.txt"))
        for _ in range(3):                                       # warm-up: graphs, the adaptive tier (discarded)
            run_gen(eng, warm, 100)
        if arm == "split":
            wp = read_ids(os.path.join(cfg["prompts"], "pp_4096.txt"))
            wp[0] = 2999                                     # its own prompt: no workload below finds it in a cache
            run_gen(eng, wp, 16)                             # one hand-over: warms the file path
        for rep in range(a.reps):
            for widx, (name, w) in enumerate(WORKLOADS):
                if a.only and name not in a.only:
                    continue
                ids = read_ids(os.path.join(cfg["prompts"], f"pp_{w['prompt']}.txt"))[:w["prompt"]]
                ids[0] = 3000 + rep * 16 + widx              # a prompt of its own: a cold start in every arm, every rep
                if "turns" not in w:
                    # a fresh conversation each time: a tiny unrelated request first, so no session is continued
                    run_gen(eng, warm[:40], 1)
                    r = run_gen(eng, ids, w["gen"])
                    turns = [r]
                else:
                    run_gen(eng, warm[:40], 1)
                    turns, conv = [], list(ids)
                    for t in range(w["turns"]):
                        r = run_gen(eng, conv, w["gen"])
                        turns.append(r)
                        conv = conv + r["tokens"] + follow_ids(cfg, w["follow"], t)
                row = {"arm": arm, "rep": rep, "workload": name, "load_s": round(load_s, 1), "turns": []}
                for r in turns:
                    last = r["last"]
                    row["turns"].append({
                        "prompt": last.get("prompt_tokens"), "generated": len(r["tokens"]),
                        "ttft_s": round(r["ttft_s"], 3) if r["ttft_s"] else None, "wall_s": round(r["wall_s"], 3),
                        "dec_tok_s": round(r["dec_tok_s"], 2) if r["dec_tok_s"] else None,
                        "reused": last.get("reused"), "prompt_ms": last.get("prompt_ms"),
                        "split": {k: last[k] for k in last if k.startswith("split_")} or None,
                        "finish": last.get("finish"), "tokens": r["tokens"][:8]})
                row["total_wall_s"] = round(sum(r["wall_s"] for r in turns), 3)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("arm", "rep", "workload", "total_wall_s")}), flush=True)
                for t in row["turns"]:
                    print("   ", json.dumps({k: t[k] for k in ("prompt", "generated", "ttft_s", "wall_s", "dec_tok_s",
                                                             "reused", "split")}), flush=True)
    finally:
        out = {"arm": arm, "rows": rows, "cfg": {k: v for k, v in cfg.items() if k != "env"},
               "split_view": eng.split_view() if hasattr(eng, "split_view") else None}
        json.dump(out, open(a.out, "w"), indent=1)
        close_all(*closers)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "bench"])
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=4096)
    ap.add_argument("--arm", default="split", choices=["layer", "single", "single_shared", "split"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(cfg["dir"], exist_ok=True)
    return cmd_exact(a, cfg) if a.mode == "exact" else cmd_bench(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
