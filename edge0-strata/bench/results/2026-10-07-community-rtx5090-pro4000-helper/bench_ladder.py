import json, random, string, time, urllib.request, sys

URL = "http://127.0.0.1:8081/v1/chat/completions"
corpus = ""
for p in [r"C:\Strata\docs\DETAILS.md", r"C:\Strata\serve\server.py", r"C:\Strata\src\core\verify.cpp"]:
    try:
        corpus += open(p, encoding="utf-8", errors="ignore").read() + "\n"
    except OSError:
        pass
task = "\n\nExplain what this code and documentation do and propose three refactorings.\n"

def run(prompt):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 256, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r.get("timings", {})
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    return {"prompt": r["usage"]["prompt_tokens"], "gen": r["usage"]["completion_tokens"],
            "prompt_tok_s": round(t.get("prompt_per_second", 0), 1),
            "decode_tok_s": round(t.get("predicted_per_second", 0), 1),
            "accept": round(100 * da / dn, 1) if dn else None}

for chars, label in [(14000, "4k"), (112000, "32k"), (448000, "128k")]:
    nonce = "".join(random.choices(string.ascii_letters, k=24))
    prompt = nonce + "\n" + (corpus * (chars // len(corpus) + 1))[:chars] + task
    run(prompt)  # warm-up
    for i in range(3):
        nonce = "".join(random.choices(string.ascii_letters, k=24))
        prompt = nonce + "\n" + (corpus * (chars // len(corpus) + 1))[:chars] + task
        res = run(prompt)
        print(json.dumps({"len": label, "run": i, **res}), flush=True)
