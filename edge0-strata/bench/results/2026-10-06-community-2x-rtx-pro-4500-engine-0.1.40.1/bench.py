#!/usr/bin/env python3
"""Community benchmark: laufender Server (Port 8080), 3 Laeufe je Laenge, 256 Token, greedy, Prompt ohne Cache (Nonce vorn)."""
import json, re, time, random, pathlib, statistics, urllib.request, sys
ROOT = pathlib.Path.home() / "ai/Strata"
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LOG = pathlib.Path(sys.argv[2])
MODEL = "swift-1.5-iq3_xxs"
files = []
for d, g in [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"), ("third_party/llama.cpp/docs", "*.md")]:
    files += sorted((ROOT / d).rglob(g))
corp = "".join(f"\n### {f.name}\n" + f.read_text(errors="ignore") for f in files)
print("Korpus Zeichen:", len(corp), "Dateien:", len(files))
TASK = "\n\nExplain what this code does and propose three concrete refactorings with code."
def prompt(tok):
    return "Here is source code:\n" + corp[: int(tok * 3.2)] + TASK
LENGTHS = {"1k": 1000, "4k": 4000, "32k": 32000, "128k": 125000, "256k": 245000}
def ask(text):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{random.random()}]\n" + text}], "max_tokens": 256, "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
pat = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
res = []
ask("Say hi.")
for name, tok in LENGTHS.items():
    text = prompt(tok)
    ask(text)  # Aufwaermung
    for i in range(3):
        before = LOG.read_text()
        ask(text); time.sleep(1)
        m = pat.findall(LOG.read_text()[len(before):])
        if not m: print("kein Logtreffer", name, i); continue
        pt, reused, read, pms, ptps, gen, gms, gtps, acc, tot = m[-1]
        res.append(dict(length=name, run=i + 1, prompt_tokens=int(pt), reused=int(reused), prompt_tok_s=float(ptps), generated=int(gen), output_tok_s=float(gtps), drafts_accepted=int(acc), drafts_total=int(tot)))
        print(res[-1], flush=True)
    json.dump(res, open(OUT / "runs.json", "w"), indent=1)
lines = ["| Length | Prompt tokens | Prompt tok/s (median, range) | Output tok/s (median, range) | Draft accept |", "| --- | ---: | ---: | ---: | ---: |"]
for name in LENGTHS:
    r = [x for x in res if x["length"] == name]
    if not r: continue
    p = [x["prompt_tok_s"] for x in r]; o = [x["output_tok_s"] for x in r]
    lines.append(f"| {name} | {r[0]['prompt_tokens']} | {statistics.median(p):.0f} ({min(p):.0f}-{max(p):.0f}) | {statistics.median(o):.1f} ({min(o):.1f}-{max(o):.1f}) | {sum(x['drafts_accepted'] for x in r) / max(1, sum(x['drafts_total'] for x in r)):.0%} |")
open(OUT / "table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
