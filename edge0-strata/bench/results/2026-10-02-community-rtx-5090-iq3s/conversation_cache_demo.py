"""Conversation-cache demo: two long agent-style conversations interleaved through the live server.

The engine's `--conversation-cache-mib` defaults to 0 (off): every follow-up re-reads the whole
conversation. With 32 GiB / 8 slots (our production config), a follow-up restores the parked K/V
instead. This script runs five requests — doc A, doc B, follow-up on A, follow-up on B, second
follow-up on A — and reports, per request, the reused tokens (`usage.prompt_tokens_details.cached_tokens`),
the freshly read tokens and the wall time, plus the engine's own `restored/parked` log lines.

    python bench/results/2026-10-02-community-rtx-5090-iq3s/conversation_cache_demo.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from needle_bench import haystack, CHARS_PER_TOKEN  # noqa: E402

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next-iq3_s"
LOG = ROOT / "strata-iq3_s.log"
DOC_TOKENS = 60_000


def ask(messages) -> dict:
    body = {"model": MODEL, "max_tokens": 128, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.load(r)
    u = out["usage"]
    return {"prompt_tokens": u["prompt_tokens"], "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "completion_tokens": u["completion_tokens"], "seconds": round(time.time() - t0, 2),
            "answer": (out["choices"][0]["message"].get("content") or "")[:120]}


def main() -> int:
    base = int(time.time()) % 20_000_000          # a fresh text window per run, printed for reproducibility
    print(f"haystack window base: {base}")
    n = int(DOC_TOKENS * CHARS_PER_TOKEN)
    doc_a = haystack(n + base)[base:]
    doc_b = haystack(n + base + 5_000_003)[base + 5_000_003:]
    q = "\n\nIn three sentences: what kinds of files make up the text above?"
    fu1 = "\n\nAnd which of those file types would I edit to change compiler flags?"
    fu2 = "\n\nGive one concrete example of a line you would change there."

    conv_a, conv_b, rows = [], [], []
    steps = ["A1 fresh doc A", "B1 fresh doc B", "A2 follow-up on A", "B2 follow-up on B",
             "A3 follow-up on A again"]
    for name in steps:
        conv = conv_a if name.startswith("A") else conv_b
        if name.startswith("A1"):
            conv.append(("user", doc_a + q))
        elif name.startswith("B1"):
            conv.append(("user", doc_b + q))
        else:
            conv.append(("user", fu1 if name[1] == "2" else fu2))
        msgs = list(conv)
        mark = len(LOG.read_text(encoding="utf-8", errors="replace"))
        r = ask([{"role": m[0], "content": m[1]} for m in msgs])
        conv.append(("assistant", r["answer"]))
        new = [l for l in re.findall(r"strata serve: (?:conversation cache:|prompt) [^\n]*", LOG.read_text(encoding="utf-8", errors="replace")[mark:])]
        r["step"], r["engine_lines"] = name, new
        rows.append(r)
        print(f"{name:24} prompt {r['prompt_tokens']:>7,}  reused {r['cached_tokens']:>7,}  "
              f"fresh {r['prompt_tokens'] - r['cached_tokens']:>6,}  {r['seconds']:>6.2f} s", flush=True)
    Path(__file__).with_name("demo.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote demo.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
