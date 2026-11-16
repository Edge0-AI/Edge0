"""q36_serve_par.py PORT [--n 128] [--conc 4] [--img PATH] : N concurrent chat requests through the server (parallel: N), against
the same requests one after the other (the solo path).  Greedy, thinking off: the texts must be equal.  Prints the
aggregate rate of the concurrent round.  With --img: half of the requests carry the picture (and ask about it)."""
import argparse, base64, json, sys, threading, time, urllib.request

PROMPTS = json.load(open(sys.argv[0].rsplit("/", 1)[0] + "/parity_prompts.json")) if False else [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Implement an LRU cache in Python with get and put in O(1), and add three unit tests.",
    "What is 17 * 23 + 144 / 12? Show the steps.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Translate into German: The train to Munich leaves at half past seven, platform four.",
    "List five differences between TCP and UDP.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Write a JSON object describing a book with title, author, year and three tags.",
]


def ask(port, content, n, out, i, stream=False):
    body = {"model": "x", "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                       json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
        out[i] = (r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t0)
    except Exception as e:  # noqa: BLE001
        out[i] = ("ERROR %s" % e, 0, time.time() - t0)


ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--img", default=None)
a = ap.parse_args()
contents = []
for i in range(a.conc):
    p = PROMPTS[i % len(PROMPTS)]
    if a.img and i % 2 == 1:
        b64 = base64.b64encode(open(a.img, "rb").read()).decode()
        contents.append([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                         {"type": "text", "text": "What text is written in the image? Answer with the text only."}])
    else:
        contents.append(p)
solo = [None] * a.conc
for i, c in enumerate(contents):
    ask(a.port, c, a.n, solo, i)
print("one after the other: %d requests, %d tokens, %.1fs" % (a.conc, sum(x[1] for x in solo), sum(x[2] for x in solo)), flush=True)
for rnd in range(2):
    conc = [None] * a.conc
    ts = [threading.Thread(target=ask, args=(a.port, c, a.n, conc, i)) for i, c in enumerate(contents)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    tot = sum(x[1] for x in conc)
    same = [conc[i][0] == solo[i][0] for i in range(a.conc)]
    print("concurrent round %d: %d requests, %d tokens in %.1fs = %.1f tok/s aggregate; texts equal to the solo run: %s"
          % (rnd, a.conc, tot, wall, tot / wall, same), flush=True)
    for i, ok in enumerate(same):
        if not ok:
            print("  request %d: %r vs %r" % (i, conc[i][0][:80], solo[i][0][:80]))
sys.exit(0)
