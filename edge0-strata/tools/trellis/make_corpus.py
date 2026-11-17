"""make_corpus.py - the calibration set and the held-out evaluation set for the Gyro swap experiment.

Writes two ids files in the format of tools/quant_quality.py (run / compare): `cal.ids.json` and `eval.ids.json`, every
segment SEG tokens.  The two sets share no document: Wikipedia articles come from two different seeds with their titles
checked against each other, code from two disjoint directory lists, agent transcripts from two seeds of a generator.

  python3 tools/trellis/make_corpus.py --tokenizer packs/gyro-s/tokenizer --out DIR [--seg 512] [--cal 320] [--eval 128]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import strata_tokenizer as ST  # noqa: E402

LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "pt", "it", "ar", "ko", "tr", "pl", "nl", "hi", "vi"]


def load_tokenizer(path):
    tp = pathlib.Path(path)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def wiki_article(lang: str, seed: int):
    url = ("https://%s.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0&grnlimit=1"
           "&prop=extracts&explaintext=1&redirects=1&grnfilterredir=nonredirects" % lang)
    req = urllib.request.Request(url + "&_=%d" % seed, headers={"User-Agent": "strata-research/1.0 (research corpus; contact via GitHub Niko1221)"})
    d = {}
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(3 * (attempt + 1))
    for p in d.get("query", {}).get("pages", {}).values():
        return p.get("title", ""), p.get("extract", "")
    return "", ""


def wiki_docs(langs, n_each, seed):
    jobs = [(l, seed * 100000 + i * 17 + k) for k, l in enumerate(langs) for i in range(n_each)]
    out = {}
    with cf.ThreadPoolExecutor(2) as ex:
        for (l, s), res in zip(jobs, ex.map(lambda j: wiki_article(*j), jobs)):
            title, text = res
            if len(text) > 1500:
                out.setdefault(l, []).append((title, text))
    return out


def code_files(dirs, exts, limit_bytes=40000):
    files = []
    for d in dirs:
        for p in sorted(pathlib.Path(d).rglob("*")):
            if p.is_file() and p.suffix in exts and 800 < p.stat().st_size < 400000:
                files.append(p)
    return files


def agent_transcript(rnd: random.Random) -> str:
    tools = [("read_file", {"path": "src/%s.py"}), ("run_shell", {"command": "grep -rn %s ."}),
             ("search", {"query": "%s", "limit": 5}), ("write_file", {"path": "notes/%s.md", "content": "..."}),
             ("http_get", {"url": "https://example.org/%s"}), ("list_dir", {"path": "%s"})]
    words = ["parser", "cache", "router", "scheduler", "tokenizer", "config", "retry", "metrics", "auth", "index",
             "queue", "loader", "report", "build", "deploy", "schema"]
    turns = ["<|im_start|>system\nYou are a coding agent. Tools: " +
             json.dumps([{"name": t[0], "parameters": t[1]} for t in tools]) + "<|im_end|>"]
    turns.append("<|im_start|>user\n%s<|im_end|>" % rnd.choice([
        "Find where the %s is configured and fix the retry bug." % rnd.choice(words),
        "Summarize how %s works in this repository and list the files involved." % rnd.choice(words),
        "Add a unit test for the %s module and run it." % rnd.choice(words)]))
    for _ in range(rnd.randint(3, 6)):
        name, args = rnd.choice(tools)
        w = rnd.choice(words)
        a = {k: (v % w if isinstance(v, str) and "%s" in v else v) for k, v in args.items()}
        turns.append("<|im_start|>assistant\nI will check the %s first.\n<tool_call>\n%s\n</tool_call><|im_end|>" % (
            w, json.dumps({"name": name, "arguments": a})))
        turns.append("<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>" % json.dumps(
            {"ok": True, "result": "%s: %d matches in %d files" % (w, rnd.randint(1, 40), rnd.randint(1, 9))}))
    turns.append("<|im_start|>assistant\nDone. The %s change is in place and the tests pass.<|im_end|>" % rnd.choice(words))
    return "\n".join(turns)


def windows(tok, text, seg, parse_special=False):
    ids = tok.encode(text, parse_special=parse_special)
    return [ids[i:i + seg] for i in range(0, len(ids) - seg + 1, seg)]


def build(tok, split: str, n: int, seg: int, rnd: random.Random, seed: int, code_dirs, md_dirs):
    quota = {"english": 0.25, "multilingual": 0.25, "code": 0.25, "agent": 0.125, "docs": 0.125}
    segs = []
    # prose / multilingual from Wikipedia (the seed decides the articles; the two splits use different seeds)
    wl = wiki_docs(["en"], int(n * 0.25 / 2) + 6, seed)
    ml = wiki_docs([l for l in LANGS if l != "en"], max(2, int(n * 0.25 / 15 / 2) + 2), seed + 1)
    titles = set()
    for kind, docs in (("english", [d for v in wl.values() for d in v]),
                       ("multilingual", [d for v in ml.values() for d in v])):
        pool = []
        for title, text in docs:
            if title in titles:
                continue
            titles.add(title)
            pool += windows(tok, text, seg)
        rnd.shuffle(pool)
        segs += [{"name": "%s_%03d" % (kind, i), "ids": w} for i, w in enumerate(pool[:int(n * quota[kind])])]
    # code
    pool = []
    for p in code_files(code_dirs, {".py", ".cpp", ".h", ".hpp", ".c", ".cu", ".js", ".sh", ".cmake", ".rs", ".go"}):
        try:
            pool += windows(tok, p.read_text(encoding="utf-8", errors="ignore")[:60000], seg)[:2]
        except Exception:
            pass
    rnd.shuffle(pool)
    segs += [{"name": "code_%03d" % i, "ids": w} for i, w in enumerate(pool[:int(n * quota["code"])])]
    # docs (markdown / text of the repositories)
    pool = []
    for d in md_dirs:
        for p in sorted(pathlib.Path(d).rglob("*.md")):
            if p.stat().st_size > 3000:
                pool += windows(tok, p.read_text(encoding="utf-8", errors="ignore")[:50000], seg)[:2]
    rnd.shuffle(pool)
    segs += [{"name": "docs_%03d" % i, "ids": w} for i, w in enumerate(pool[:int(n * quota["docs"])])]
    # agent transcripts (tool calls)
    pool = []
    while len(pool) < int(n * quota["agent"]) + 4:
        t = "\n".join(agent_transcript(rnd) for _ in range(3))
        pool += windows(tok, t, seg, parse_special=True)
    segs += [{"name": "agent_%03d" % i, "ids": w} for i, w in enumerate(pool[:int(n * quota["agent"])])]
    print(split, {k: sum(1 for s in segs if s["name"].startswith(k)) for k in quota}, "total", len(segs))
    return {"version": "gyro-swap-" + split, "seg_tokens": seg, "segments": segs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--cal", type=int, default=320)
    ap.add_argument("--eval", type=int, default=128)
    ap.add_argument("--cal-code", nargs="+", default=["/data/llama.cpp/src", "/data/llama.cpp/tools", "/data/llama.cpp/gguf-py"])
    ap.add_argument("--eval-code", nargs="+", default=["/data/llama.cpp/ggml", "/data/llama.cpp/examples", "/data/llama.cpp/tests"])
    ap.add_argument("--cal-md", nargs="+", default=["/data/llama.cpp/docs"])
    ap.add_argument("--eval-md", nargs="+", default=["/data/llama.cpp/tools", "/data/llama.cpp/examples"])
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = build(tok, "cal", a.cal, a.seg, random.Random(11), 1, a.cal_code, a.cal_md)
    ev = build(tok, "eval", a.eval, a.seg, random.Random(22), 2, a.eval_code, a.eval_md)
    # no document in common: compare the first 64 tokens of every segment
    heads = {tuple(s["ids"][:64]) for s in cal["segments"]}
    ev["segments"] = [s for s in ev["segments"] if tuple(s["ids"][:64]) not in heads]
    (out / "cal.ids.json").write_text(json.dumps(cal))
    (out / "eval.ids.json").write_text(json.dumps(ev))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
