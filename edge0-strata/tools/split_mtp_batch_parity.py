#!/usr/bin/env python3
"""Live parity check for STRATA_SPLIT_MTP_BATCH (needs a running engine; no model on the build box).

  split_mtp_batch_parity.py dump OUT.json [--url URL] [--key-file FILE] [--long-words N]
  split_mtp_batch_parity.py compare A.json B.json

`dump` sends 5 fixed greedy prompts x 256 tokens and one long prompt (default ~25k tokens) x 256 tokens to a
server started WITHOUT the env (A) or WITH STRATA_SPLIT_MTP_BATCH=1 (B), and stores the texts.  `compare` exits 0
when every text is identical, 1 otherwise, printing PASS/FAIL lines.  The drafter's K/V is not bit-identical
(Q8_1 x Q8_0 MMQ instead of mmvq), so only the target's tokens are compared here; read `drafts accepted X of Y`
from the engine log for each arm (expect within +-3 points).
"""
import json
import sys
import urllib.request

PROMPTS = [
    "Write a 200 word story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a binary search tree stays balanced, with a short Python example.",
    "List the planets of the solar system with one fact each.",
    "Translate to French and then explain the grammar: 'The quick brown fox jumps over the lazy dog.'",
    "Write a C function that reverses a singly linked list and explain its complexity.",
]


def haystack(n_entries):
    w = ["amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor", "iris", "juniper"]
    return "\n".join(
        f"Entry {i}: the {w[(i * 3) % 10]} station {(i * 7) % 997} logged a reading of {(i * 31) % 1009}."
        for i in range(n_entries)
    )


def ask(url, key, content, n):
    body = json.dumps({"model": "qwen3.8-flash-next-ud-q4_k_xl", "messages": [{"role": "user", "content": content}],
                       "max_tokens": n, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, body, hdr), timeout=3600))
    return d["choices"][0]["message"]["content"], d["usage"]


def main(argv):
    if len(argv) >= 3 and argv[0] == "compare":
        a, b = (json.load(open(p)) for p in argv[1:3])
        bad = 0
        for k in a:
            same = a[k]["text"] == b.get(k, {}).get("text")
            print(("PASS" if same else "FAIL"), k, f"({a[k]['completion_tokens']} vs {b.get(k, {}).get('completion_tokens')} tokens)")
            bad += not same
        print("PARITY", "OK" if not bad else f"BROKEN ({bad} differ)")
        return 1 if bad else 0
    if len(argv) >= 2 and argv[0] == "dump":
        url = "http://127.0.0.1:8000/v1/chat/completions"
        key = None
        words = 1100
        i = 2
        while i < len(argv):
            if argv[i] == "--url":
                url = argv[i + 1]
            elif argv[i] == "--key-file":
                key = open(argv[i + 1]).read().strip()
            elif argv[i] == "--long-words":
                words = int(argv[i + 1])
            i += 2
        out = {}
        for n, p in enumerate(PROMPTS):
            t, u = ask(url, key, p, 256)
            out[f"short{n}"] = {"text": t, "completion_tokens": u["completion_tokens"], "prompt_tokens": u["prompt_tokens"]}
        t, u = ask(url, key, "Summarise the log below in three sentences.\n" + haystack(words), 256)
        out["long"] = {"text": t, "completion_tokens": u["completion_tokens"], "prompt_tokens": u["prompt_tokens"]}
        json.dump(out, open(argv[1], "w"), indent=1)
        print({k: v["prompt_tokens"] for k, v in out.items()})
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
