"""A scripted research run against a running Strata server: one long document, then many different questions.

Reads the document once, then asks each question and reports, per question, the time to the first streamed token and
the whole request, and the run's total.  Greedy (temperature 0), so two runs of the same file give the same answers and
a change in how the server keeps the document (the shared prefix below) can be compared answer by answer.

    python tools/research_run.py --url http://127.0.0.1:8080 --doc paper.txt --questions questions.txt
    python tools/research_run.py --doc paper.txt --questions q.txt --prefix off --json base.json
    python tools/research_run.py --doc paper.txt --questions q.txt --prefix on  --json fork.json
    python tools/research_run.py --compare base.json fork.json

--layout split  (the default): the document is the first message, the question the second.
--layout joined: one message, "document + question" (a client that cannot split them).
--prefix on: the request carries "strata_prefix" naming the document as the shared prefix (docs/DETAILS.md): the
server reads it once and every question starts from it.  off: the field is left out.
--parallel N: N questions at a time (the server needs "parallel": N for them to run together).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path


def request_body(doc: str, question: str, layout: str, prefix: bool, max_tokens: int, model: str | None) -> dict:
    if layout == "joined":
        body = {"messages": [{"role": "user", "content": doc + "\n\nQuestion: " + question + "\nAnswer:"}]}
        if prefix:
            body["strata_prefix"] = {"message": 0, "chars": len(doc) + 2}
    else:
        body = {"messages": [{"role": "user", "content": doc}, {"role": "user", "content": question}]}
        if prefix:
            body["strata_prefix"] = {"messages": 1}
    body.update(temperature=0, max_tokens=max_tokens, stream=True)
    if model:
        body["model"] = model
    return body


def ask(url: str, body: dict, headers: dict, timeout: float) -> dict:
    t0 = time.time()
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    first, parts = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:])
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece and first is None:
                    first = time.time() - t0
                parts.append(piece)
    return {"ttft_s": round(first or 0.0, 3), "wall_s": round(time.time() - t0, 2), "text": "".join(parts)}


def run(args) -> dict:
    doc = Path(args.doc).read_text(encoding="utf-8", errors="replace")
    if args.limit_chars:
        doc = doc[:args.limit_chars]
    questions = [x.strip() for x in Path(args.questions).read_text(encoding="utf-8").splitlines() if x.strip()]
    questions = questions[:args.n] if args.n else questions
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    rows: list[dict | None] = [None] * len(questions)
    started = time.time()
    for k in range(0, len(questions), max(1, args.parallel)):
        group = list(range(k, min(k + max(1, args.parallel), len(questions))))

        def one(i):
            try:
                rows[i] = {"q": i, **ask(args.url, request_body(doc, questions[i], args.layout, args.prefix == "on",
                                                                  args.max_tokens, args.model), headers, args.timeout)}
            except Exception as e:  # noqa: BLE001 - a failed question is reported, the run goes on
                rows[i] = {"q": i, "error": repr(e)}
        threads = [threading.Thread(target=one, args=(i,)) for i in group]
        [t.start() for t in threads]
        [t.join() for t in threads]
        for i in group:
            r = rows[i]
            print(f"q{i:02d}  first token {r.get('ttft_s', '-')} s  total {r.get('wall_s', '-')} s  "
                  f"{r.get('error') or repr(r['text'][:60])}", flush=True)
    out = {"url": args.url, "layout": args.layout, "prefix": args.prefix, "doc_chars": len(doc), "questions": rows,
           "total_s": round(time.time() - started, 1)}
    later = [r["ttft_s"] for r in rows[1:] if r and "ttft_s" in r]
    if later:
        out["first_token_median_after_first_s"] = round(statistics.median(later), 3)
    print(f"total {out['total_s']} s" + (f", median first token after the first question {out['first_token_median_after_first_s']} s"
                                         if later else ""), flush=True)
    return out


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a_path, b_path))
    same = diff = 0
    for x, y in zip(a["questions"], b["questions"]):
        if "text" in x and "text" in y:
            if x["text"] == y["text"]:
                same += 1
            else:
                diff += 1
                print(f"q{x['q']:02d}: the answers differ")
    print(f"{same} answers identical, {diff} differ; total {a['total_s']} s -> {b['total_s']} s; median first token "
          f"{a.get('first_token_median_after_first_s')} s -> {b.get('first_token_median_after_first_s')} s")
    return 1 if diff else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--doc")
    ap.add_argument("--questions", help="a text file, one question per line")
    ap.add_argument("--n", type=int, default=0, help="only the first N questions")
    ap.add_argument("--layout", choices=("split", "joined"), default="split")
    ap.add_argument("--prefix", choices=("on", "off"), default="on")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit-chars", type=int, default=0, help="use only the first N characters of the document")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--json", help="write the per-question results here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two saved runs' answers and times")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.doc or not args.questions:
        ap.error("--doc and --questions are needed")
    out = run(args)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 1 if any("error" in r for r in out["questions"]) else 0


if __name__ == "__main__":
    sys.exit(main())
