#!/usr/bin/env python3
"""Stop, edit and resend, over and over (#879, #606): every answer must be real text, never a repeated token.

The reports are an all-logits NaN (one token repeated, "!!!!") after a request was stopped and the next one, an edit
or a resend of the same prompt, started while the engine was still winding the first down.  This drives that loop
against a running server: stream an answer, read a few lines, close the connection (the client's Stop), wait 0 to
~0.4 s (sometimes not at all, sometimes a second request already running), send the edited prompt, and check the
answer.  Run the engine with STRATA_DBG_NAN=1 to get the first non-finite layer named in its log.

  STRATA_KEY=... python3 tools/stop_resend_test.py http://127.0.0.1:8090 [rounds=40] [seed=1]

Exit 0 when every answer was real text, 1 otherwise.
"""
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
KEY = os.environ.get("STRATA_KEY", "")
URL = BASE + "/v1/chat/completions"

LONG = ["Describe the weather in Montpellier in great detail, at length.",
        "Write a long story about a lighthouse keeper.",
        "Explain how a bicycle works, in as much detail as you can."]
SHORT = [("Reply with exactly one word: the color of the sky on a clear day.", "blue"),
         ("Reply with exactly one word: the opposite of cold.", "hot|warm"),
         ("What is 2 plus 3? Reply with the number only.", "5|five")]


def req(content, stream, max_tokens):
    body = json.dumps({"model": "m", "max_tokens": max_tokens, "temperature": 0, "stream": stream,
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    return urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                           "Authorization": f"Bearer {KEY}"})


def stop_after(content, lines):
    r = urllib.request.urlopen(req(content, True, 400), timeout=600)
    for _ in range(lines):
        r.readline()
    r.close()


def answer(content, max_tokens=40):
    with urllib.request.urlopen(req(content, False, max_tokens), timeout=600) as r:
        m = json.load(r)["choices"][0]["message"]
    return ((m.get("content") or "") + " " + (m.get("reasoning_content") or m.get("reasoning") or "")).strip()


def garbage(text):
    """Empty, one character or one token repeated (the NaN signature: '!!!!!!', 'the the the the')."""
    t = text.strip()
    if len(t) < 1:
        return True
    if len(t) >= 8 and len(set(t.replace(" ", ""))) <= 2:
        return True
    words = re.findall(r"\S+", t)
    return len(words) >= 6 and len(set(words)) == 1


def main():
    rnd = random.Random(SEED)
    bad = 0
    for i in range(ROUNDS):
        long_p = rnd.choice(LONG)
        q, want = rnd.choice(SHORT)
        mode = rnd.choice(("stop-resend", "stop-edit", "overlap"))
        bg = None
        try:
            if mode == "overlap":                  # the next request is already running while the first one is stopped
                bg = threading.Thread(target=lambda: answer(rnd.choice(LONG), 60))
                bg.start()
                time.sleep(rnd.uniform(0, 0.3))
            stop_after(long_p, rnd.randint(1, 8))
            time.sleep(rnd.choice((0, 0, 0.05, 0.2, 0.4)))
            text = answer(long_p if mode == "stop-resend" else q, 40 if mode == "stop-resend" else 30)
        except Exception as e:                      # a refused or dropped request is a failure of its own
            print(f"{i:3d} {mode}: ERROR {e!r}", flush=True)
            bad += 1
            continue
        finally:
            if bg:
                bg.join()
        ok = not garbage(text) and (mode == "stop-resend" or re.search(want, text, re.I) is not None)
        bad += not ok
        print(f"{i:3d} {mode}: {'ok' if ok else 'BAD'} {text[:70]!r}", flush=True)
    print(f"{ROUNDS} rounds, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
