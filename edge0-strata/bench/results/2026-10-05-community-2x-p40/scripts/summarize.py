#!/usr/bin/env python3
"""summarize.py <raw-dir-or-files...>  — per-arm table: accuracy / completeness / reliability / speed, + cross-arm agreement.

Reads the *.jsonl records written by bench.py. If an arm has several files or reps, they are pooled (rep count shown).
Speed uses server-reported timings when present (llama-server/Strata `timings`, Ollama eval_duration), else client-side.
"""
import glob, json, os, statistics as st, sys, difflib, collections

def load(paths):
    recs = []
    for p in paths:
        for f in (glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else [p]):
            for line in open(f):
                line = line.strip()
                if line:
                    r = json.loads(line); r["_file"] = os.path.basename(f); recs.append(r)
    return recs

def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None

def fmt(x, nd=1):
    return "-" if x is None else ("%.*f" % (nd, x))

def decode_tps(r):
    res = r.get("res") or {}
    t = res.get("server_timings") or {}
    if t.get("predicted_per_second"):
        return t["predicted_per_second"], "srv"
    n, tf, tt = res.get("completion_tokens"), res.get("t_first"), res.get("t_total")
    if n and n > 1 and tf is not None and tt and tt > tf:
        return (n - 1) / (tt - tf), "cli"
    return None, None

def prefill_tps(r):
    res = r.get("res") or {}
    t = res.get("server_timings") or {}
    if t.get("prompt_per_second"):
        return t["prompt_per_second"]
    return None

def main(paths):
    recs = load(paths)
    arms = collections.defaultdict(list)
    for r in recs:
        arms[r["arm"]].append(r)
    cats = ["reasoning", "code", "tools", "complete", "longctx"]
    print("## Quality (pass rate = passed/total across pooled reps; errors count as fails)\n")
    print("| arm | think | " + " | ".join(cats) + " | truncated(length) | errors |")
    print("|---|---|" + "---|" * (len(cats) + 2))
    for a, rs in sorted(arms.items()):
        row = []
        for c in cats:
            xs = [r for r in rs if r["category"] == c]
            if not xs: row.append("-"); continue
            ok = sum(1 for r in xs if r["passed"]); row.append("%d/%d (%.0f%%)" % (ok, len(xs), 100 * ok / len(xs)))
        gen = [r for r in rs if r["category"] not in ("speed",)]
        trunc = sum(1 for r in gen if (r.get("res") or {}).get("finish_reason") == "length")
        errs = sum(1 for r in rs if r.get("error"))
        print("| %s | %s | %s | %d | %d |" % (a, rs[0].get("think"), " | ".join(row), trunc, errs))
    print("\n## Long-context recall fraction (mean of 3 needles) by size\n")
    sizes = sorted({r["id"] for r in recs if r["category"] == "longctx"}, key=lambda s: int(s.split("-")[1][:-1]))
    if sizes:
        print("| arm | " + " | ".join(sizes) + " |"); print("|---|" + "---|" * len(sizes))
        for a, rs in sorted(arms.items()):
            cells = []
            for s in sizes:
                xs = [(r.get("grade") or {}).get("fraction") for r in rs if r["id"] == s and r.get("res")]
                cells.append(fmt(100 * st.mean(xs), 0) + "%" if xs else "-")
            print("| %s | %s |" % (a, " | ".join(cells)))
    print("\n## Speed (median over reps; srv = server-reported, cli = client-side)\n")
    print("| arm | decode tok/s (all gen tasks) | decode tok/s (speed-short) | prefill tok/s by prompt size | TTFT short (s) | first-request total (s) |")
    print("|---|---|---|---|---|---|")
    for a, rs in sorted(arms.items()):
        dec = [decode_tps(r)[0] for r in rs if r.get("res")]
        short = [decode_tps(r)[0] for r in rs if r["id"] == "speed-short" and r.get("res")]
        pf = collections.defaultdict(list)
        for r in rs:
            if r["id"].startswith("speed-prefill") or r["id"].startswith("longctx"):
                p = prefill_tps(r)
                if p: pf[r["id"].replace("speed-prefill-", "").replace("longctx-", "lc-")].append(p)
        pfs = ", ".join("%s:%s" % (k, fmt(med(v), 0)) for k, v in sorted(pf.items(), key=lambda kv: (len(kv[0]), kv[0])))
        ttft = med([(r.get("res") or {}).get("t_first") for r in rs if r["id"] == "speed-short"])
        first = (sorted(rs, key=lambda r: r["ts"])[0].get("res") or {}).get("t_total")
        print("| %s | %s | %s | %s | %s | %s |" % (a, fmt(med(dec)), fmt(med(short)), pfs or "-", fmt(ttft, 2), fmt(first)))
    # agreement: same task id, text-level, between arms (think off, rep 0)
    print("\n## Cross-arm output agreement (greedy, same prompts; similarity = difflib ratio of final text; exact = identical)\n")
    texts = {a: {r["id"]: (r.get("res") or {}).get("text", "") for r in rs if r.get("res") and r["rep"] == 0 and r["category"] != "speed"} for a, rs in arms.items()}
    names = sorted(texts)
    if len(names) > 1:
        print("| arm A | arm B | tasks | exact | mean similarity |"); print("|---|---|---|---|---|")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ids = sorted(set(texts[names[i]]) & set(texts[names[j]]))
                if not ids: continue
                ex = sum(texts[names[i]][t] == texts[names[j]][t] for t in ids)
                sim = st.mean(difflib.SequenceMatcher(None, texts[names[i]][t], texts[names[j]][t]).ratio() for t in ids)
                print("| %s | %s | %d | %d | %.3f |" % (names[i], names[j], len(ids), ex, sim))
    # within-arm determinism across reps
    print("\n## Within-arm repeatability (reps of identical greedy requests)\n")
    for a, rs in sorted(arms.items()):
        by = collections.defaultdict(list)
        for r in rs:
            if r.get("res") and r["category"] != "speed": by[r["id"]].append(r["res"]["text"])
        multi = {k: v for k, v in by.items() if len(v) > 1}
        if multi:
            same = sum(len(set(v)) == 1 for v in multi.values())
            print("- %s: %d/%d tasks gave identical text on every rep" % (a, same, len(multi)))

if __name__ == "__main__":
    main(sys.argv[1:] or ["results/raw"])
