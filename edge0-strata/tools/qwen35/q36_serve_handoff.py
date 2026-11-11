"""q36_serve_handoff.py PORT [--n 300] : a request that runs alone (the solo path, with MTP drafts), another one arrives while it
runs (the server stops it and continues it in a batch slot from its held state), the second one ends (the first is alone again and
goes back to the solo path): the first request's text must be the text of the same request run entirely alone.  Greedy, thinking
off, every expert in VRAM (in the hybrid mode a different window mix can round differently: see docs/QWEN36.md).

  python tools/qwen35/q36_serve_handoff.py 8199 [--n 300] [--delay 1.0]
"""
import argparse, json, sys, threading, time, urllib.request

A = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle and follows it across the sea."
B = "List the first twenty prime numbers and explain how you found them."


def ask(port, text, n, out, key):
    body = {"model": "x", "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    out[key] = (r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t0)


ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--delay", type=float, default=1.0)
a = ap.parse_args()
alone = {}
ask(a.port, A, a.n, alone, "A")
ask(a.port, B, 60, alone, "B")
for rnd in range(2):
    mix = {}
    ta = threading.Thread(target=ask, args=(a.port, A, a.n, mix, "A"))
    tb = threading.Thread(target=lambda: (time.sleep(a.delay), ask(a.port, B, 60, mix, "B")))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    ea, eb = mix["A"][0] == alone["A"][0], mix["B"][0] == alone["B"][0]
    print("round %d: A %d tokens in %.1fs, B %d tokens in %.1fs; A equals its run alone: %s; B equals its run alone: %s"
          % (rnd, mix["A"][1], mix["A"][2], mix["B"][1], mix["B"][2], ea, eb), flush=True)
    if not (ea and eb):
        x, y = mix["A"][0], alone["A"][0]
        k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        print("  A first differs from its run alone at character %d of %d: %r vs %r" % (k, len(y), x[max(0, k - 20):k + 30],
                                                                                    y[max(0, k - 20):k + 30]))
