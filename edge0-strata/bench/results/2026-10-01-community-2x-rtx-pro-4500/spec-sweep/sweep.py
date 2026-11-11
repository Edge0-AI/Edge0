#!/usr/bin/env python3
"""Spec-Sweep fuer Strata: pro N den Server neu starten, 3 Laeufe je Prompt, 256 Token, greedy.
Liest tok/s aus dem Serverlog ("generated in X ms")."""
import json, os, re, signal, subprocess, sys, time, urllib.request, random, pathlib, statistics
ROOT = pathlib.Path.home() / "ai/Strata"
OUT = pathlib.Path.home() / "messungen/2026-10-01-strata-spec-sweep"
BASE = json.load(open(ROOT / "strata-swift-iq3_xxs.json.bak-20261001-vorsweep"))
NS = [int(x) for x in sys.argv[1].split(",")]
RUNS = 3
def corpus(tokens):
    files = sorted((ROOT/"src").rglob("*.cpp")) + sorted((ROOT/"serve").glob("*.py")) + sorted((ROOT/"docs").glob("*.md"))
    txt = ""
    for f in files:
        txt += f"\n### {f.name}\n" + f.read_text(errors="ignore")
        if len(txt) > tokens * 3.2: break
    return txt[: int(tokens * 3.2)]
PROMPTS = {
  "kurz-code": ("Write a Python class LRUCache with get/put, type hints and unit tests using pytest.", 0),
  "4k-code": ("Here is source code:\n" + corpus(4000) + "\n\nExplain what this code does and propose three concrete refactorings with code.", 4000),
  "32k-code": ("Here is source code:\n" + corpus(32000) + "\n\nExplain what this code does and propose three concrete refactorings with code.", 32000),
}
def stop():
    me = os.getpid()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for d in os.listdir("/proc"):
            if d.isdigit() and int(d) != me:
                try: cmd = open(f"/proc/{d}/cmdline").read().replace(chr(0), " ")
                except OSError: continue
                if "serve/server.py --engine strata" in cmd or "engine/strata" in cmd and "python3 sweep" not in cmd:
                    try: os.kill(int(d), sig)
                    except OSError: pass
        time.sleep(8)
    for _ in range(60):
        port = subprocess.run("ss -ltn | grep -q ':8080 '", shell=True).returncode == 0
        mem = max(int(x) for x in subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.split())
        if not port and mem < 1000: return
        time.sleep(3)
    raise SystemExit("Port/GPU wurde nicht frei")
def start(n):
    cfg = json.loads(json.dumps(BASE))
    a = cfg["args"]; a[a.index("--spec")+1] = str(n)
    cfg["log"] = str(OUT / f"server-spec{n}.log")
    p = OUT / f"cfg-spec{n}.json"; json.dump(cfg, open(p, "w"), indent=1)
    proc = subprocess.Popen(["setsid", str(ROOT/".venv/bin/python"), str(ROOT/"serve/server.py"), "--engine", "strata", "--config", str(p), "--port", "8080", "--open"],
        cwd=ROOT, stdout=open(OUT/f"stdout-spec{n}.txt","w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(3)
        if proc.poll() is not None: raise SystemExit(f"Server fuer spec={n} sofort beendet, siehe stdout-spec{n}.txt")
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3).read(); return cfg["log"]
        except Exception: pass
    raise SystemExit(f"Server fuer spec={n} nicht hochgekommen")
def ask(text, nonce):
    body = {"model": BASE["model_name"], "messages": [{"role":"user","content": f"[{nonce}]\n"+text}],
            "max_tokens": 256, "temperature": 0, "stream": False}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
res = []
for n in NS:
    stop(); log = start(n)
    for name, (text, _) in PROMPTS.items():
        ask(text, f"warm{n}{name}{random.random()}")   # Aufwaermung, zaehlt nicht
        for i in range(RUNS):
            before = open(log).read()
            ask(text, f"r{n}{name}{i}{random.random()}")
            time.sleep(1)
            new = open(log).read()[len(before):]
            m = re.findall(r"prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", new)
            if not m: print("kein Logtreffer", n, name, i); continue
            pt, gen, ms, tps, acc, tot = m[-1]
            res.append(dict(spec=n, prompt=name, run=i, prompt_tokens=int(pt), generated=int(gen), tok_s=float(tps), accepted=int(acc), drafted=int(tot)))
            print(n, name, i, tps, f"{acc}/{tot}", flush=True)
    json.dump(res, open(OUT/"sweep.json","w"), indent=1)
print("\nspec  prompt      median  min-max   accept")
for n in NS:
    for name in PROMPTS:
        r = [x for x in res if x["spec"]==n and x["prompt"]==name]
        if r:
            t = [x["tok_s"] for x in r]
            print(f"{n:>4}  {name:<10} {statistics.median(t):7.1f}  {min(t):.0f}-{max(t):.0f}   {sum(x['accepted'] for x in r)/max(1,sum(x['drafted'] for x in r)):.0%}")

stop()
