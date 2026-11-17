"""#3493: does an agent-Harness-style run reuse its prefix with the server's DEFAULT conversation cache?

    python tools/harness_reuse_probe.py http://127.0.0.1:8080 [--turns 4] [--main-tokens 6000] [--key KEY]

An agent Harness keeps a long MAIN conversation (system prompt, tools, a growing list of tool results) and, between its
turns, sends shorter side conversations (a sub-agent, a summary) with their own system prompts.  Every turn of either one
extends the previous request of the same conversation.  With the conversation cache off, the engine holds one prompt, so
each switch to the other conversation read the whole prompt again (the report: Reused = 0 on 12 of 12 requests).

The probe sends MAIN1 SUB1 MAIN2 SUB2 ... and reads the server's /metrics before and after each request: the tokens the
request took from the cache ("reused") out of its prompt.  Exit code 0 when every follow-up turn (all but the first of
each conversation) reused more than half of its prompt, 1 otherwise.  Needs an idle server; sampling is greedy, 8 new tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def post(url, key, body, timeout=600):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def metrics(url, key):
    req = urllib.request.Request(url + "/metrics", headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def filler(tag, n_words):
    """Deterministic text of about n_words words that differs per tag (so two conversations share no prefix)."""
    words = ["%s%d" % (tag, i % 977) for i in range(n_words)]
    return " ".join(words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--main-tokens", type=int, default=6000, help="about this many tokens in the main conversation's first request")
    ap.add_argument("--sub-tokens", type=int, default=1500)
    ap.add_argument("--step-tokens", type=int, default=500, help="what every turn adds (a tool result)")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    model = (json.load(urllib.request.urlopen(url + "/v1/models", timeout=20))["data"][0]["id"]
             if not a.key else json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer " + a.key}), timeout=20))["data"][0]["id"])
    conv = {"MAIN": [{"role": "system", "content": "You are the main coding agent. " + filler("m", a.main_tokens // 2)}],
            "SUB": [{"role": "system", "content": "You are a sub-agent that summarizes. " + filler("s", a.sub_tokens // 2)}]}
    rows, ok = [], True
    for turn in range(a.turns):
        for name, step in (("MAIN", a.step_tokens), ("SUB", a.step_tokens // 2)):
            conv[name].append({"role": "user", "content": "tool result %d: %s Reply with one word." % (turn, filler(name[0].lower() + "t%d" % turn, step // 2))})
            before = metrics(a.url.rstrip("/"), a.key).get("conversation_cache") or {}
            t0 = time.time()
            reply = post(url, a.key, {"model": model, "messages": conv[name], "max_tokens": 8, "temperature": 0,
                                      "chat_template_kwargs": {"enable_thinking": False}})
            took = time.time() - t0
            after = metrics(url, a.key).get("conversation_cache") or {}
            prompt = (reply.get("usage") or {}).get("prompt_tokens", 0)
            reused = int(after.get("reused_tokens", 0)) - int(before.get("reused_tokens", 0))
            conv[name].append({"role": "assistant", "content": (reply["choices"][0]["message"].get("content") or "ok")})
            first = turn == 0
            good = first or reused * 2 > prompt
            ok = ok and good
            rows.append((name, turn + 1, prompt, reused, took, good))
            print("%-4s turn %d: prompt %6d tokens, reused %6d (%3.0f%%), %5.1f s%s" % (
                name, turn + 1, prompt, reused, 100.0 * reused / max(prompt, 1), took, "" if good else "   <-- not reused"), flush=True)
    follow = [r for r in rows if r[1] > 1]
    total_prompt = sum(r[2] for r in follow)
    total_reused = sum(r[3] for r in follow)
    print("follow-up turns: %d, reused %d of %d prompt tokens (%.0f%%); conversation cache: %s" % (
        len(follow), total_reused, total_prompt, 100.0 * total_reused / max(total_prompt, 1),
        json.dumps({k: v for k, v in (metrics(url, a.key).get("conversation_cache") or {}).items() if k in ("enabled", "budget_mib", "slots", "parked", "requests_reused")})))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
