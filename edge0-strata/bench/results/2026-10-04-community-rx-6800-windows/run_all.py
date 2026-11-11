"""bench-local/run_all.py - the full local benchmark: quality suites, speed per configuration, the longest contexts.

Runs detached; restarts the Strata server with other settings between phases and restores 128K / 2048 MiB at the
end.  Progress: run_all.log; numbers: run_all.json.
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import requests

HERE = Path(__file__).parent
ROOT = HERE.parent
PY = sys.executable
BASE = "http://127.0.0.1:8080"
CFG = ROOT / "strata-iq2_xs.json"
LOG = HERE / "run_all.log"
RES: dict = {"speed": [], "longctx": [], "vision": [], "config": [], "suites": []}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    (HERE / "run_all.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the server
def stop_server():
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if p.info["name"] in ("strata.exe", "strata-vision.exe") or ("serve" in cl and "server.py" in cl):
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(5)


def health():
    try:
        return requests.get(BASE + "/health", timeout=5).json()
    except Exception:
        return {}


def start_server():
    out = open(ROOT / "server.out", "w")
    err = open(ROOT / "server.err", "w")
    proc = subprocess.Popen([PY, "serve/server.py", "--engine", "strata", "--config", str(CFG), "--port", "8080"],
                            cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    t = time.time()
    while time.time() - t < 1200:
        if health().get("loaded"):
            return round(time.time() - t)
        if proc.poll() is not None:
            return None
        time.sleep(3)
    return None


def engine_facts():
    text = (ROOT / "strata-iq2_xs.log").read_text(encoding="utf-8", errors="replace")[-200000:]
    slots = re.findall(r"expert cache (\d+) slots, ([\d.]+) GiB", text)
    free = re.findall(r"(\d+) MiB of VRAM free with everything loaded", text)
    return {"expert_slots": int(slots[-1][0]) if slots else None, "cache_gib": float(slots[-1][1]) if slots else None,
            "vram_free_mib": int(free[-1]) if free else None}


def configure(ctx, reserve, vision_tokens=None):
    stop_server()
    p = subprocess.run([PY, "setup.py", "--setup", "--yes", "--family", "qwen", "--model", "IQ2_XS", "--context",
                        str(ctx), "--vram-reserve-mib", str(reserve), "--vision", "cpu", "--data-dir",
                        r"D:\Strata-data", "--no-start"], cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    notes = [ln.strip()[:220] for ln in p.stdout.splitlines() if "[!]" in ln or "context" in ln.lower() or "rope" in ln.lower()]
    if vision_tokens:
        cfg = json.loads(CFG.read_text(encoding="utf-8"))
        cfg["vision"]["max_tokens"] = vision_tokens
        CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    secs = start_server()
    h = health()
    rec = {"asked_context": ctx, "reserve_mib": reserve, "vision_tokens": vision_tokens, "start_s": secs,
           "max_context": h.get("max_context"), "setup_exit": p.returncode, "notes": notes[-8:], **engine_facts()}
    RES["config"].append(rec)
    log(f"CONFIG ctx {ctx} reserve {reserve}: started in {secs} s, max_context {h.get('max_context')}, {engine_facts()}")
    for n in notes[-8:]:
        log("   setup: " + n)
    return secs is not None


# ------------------------------------------------------------------------------------------------ requests
def ask(content, max_tokens=300, effort="none", timeout=7200):
    t = time.time()
    try:
        r = requests.post(BASE + "/v1/chat/completions", timeout=timeout, json={
            "model": "strata", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning_effort": effort, "temperature": 0}).json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    if "choices" not in r:
        return {"error": json.dumps(r)[:300]}
    tm = r.get("timings", {})
    return {"content": r["choices"][0]["message"].get("content") or "", "wall": round(time.time() - t, 1),
            "prompt_n": tm.get("prompt_n"), "prompt_s": round(tm.get("prompt_ms", 0) / 1000, 1),
            "prompt_tok_s": tm.get("prompt_per_second"), "out_n": tm.get("predicted_n"),
            "tok_s": tm.get("predicted_per_second")}


def speed(label):
    dec = [ask(f"[{uuid.uuid4().hex[:8]}] Explain in detail how a hash map works: hashing, collisions, resizing, "
               "complexity. About 300 words.", 420) for _ in range(3)]
    src = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
    reads = {}
    for name, kb in (("2K", 8), ("8K", 30), ("32K", 125)):
        mc = health().get("max_context") or 0
        if kb * 280 > mc - 2000:      # ~280 tokens per KB of this file
            continue
        r = ask(f"Run {uuid.uuid4().hex}.\n\n" + src[:kb * 1000] + "\n\nIn one sentence: what is this code?", 60)
        reads[name] = {k: r.get(k) for k in ("prompt_n", "prompt_s", "prompt_tok_s", "tok_s", "error")}
    ok = [d["tok_s"] for d in dec if d.get("tok_s")]
    rec = {"label": label, "decode_tok_s": ok, "decode_median": statistics.median(ok) if ok else None, "read": reads,
           **engine_facts()}
    RES["speed"].append(rec)
    log(f"SPEED {label}: decode {ok} tok/s; read " +
        ", ".join(f"{k} {v.get('prompt_n')} tok at {v.get('prompt_tok_s')} tok/s" for k, v in reads.items()))


# ------------------------------------------------------------------------------------------------ long context
WORDS = ("river mountain engine garden window bridge market signal harbor library forest station valley mirror "
         "ladder castle meadow tunnel beacon anchor copper marble velvet thunder lantern compass orchard glacier "
         "pepper cotton ribbon basket candle feather hammer island jacket kettle lemon magnet needle ocean pillow "
         "quartz rocket saddle timber violet walnut yarn zipper").split()
COLORS = ["red", "green", "blue", "amber", "violet"]


def filler(rng, n_chars):
    out, size, i = [], 0, 0
    while size < n_chars:
        i += 1
        s = f"Section {i}. " + " ".join(
            f"The {rng.choice(WORDS)} near the {rng.choice(WORDS)} was moved to the {rng.choice(WORDS)} on day "
            f"{rng.randint(1, 365)}." for _ in range(6)) + "\n"
        out.append(s)
        size += len(s)
    return out


def longctx(target_tokens, label):
    rng = random.Random(target_tokens)
    cal = ask("".join(filler(rng, 40000)) + "\nReply OK.", 5)
    if cal.get("error"):
        log(f"LONGCTX {label}: calibration failed {cal['error']}")
        return
    per_tok = 40000 / cal["prompt_n"]
    paras = filler(rng, int(target_tokens * per_tok))
    codes = {c: str(rng.randint(100000, 999999)) for c in COLORS}
    for c, depth in zip(COLORS, (0.1, 0.3, 0.5, 0.7, 0.9)):
        paras.insert(int(len(paras) * depth), f"Important: the secret code for {c} is {codes[c]}.\n")
    q = ("\nQuestion: the text above contains five secret codes, one each for red, green, blue, amber and violet. "
         "Reply with only JSON like {\"red\": \"...\", \"green\": \"...\", \"blue\": \"...\", \"amber\": \"...\", "
         "\"violet\": \"...\"}.")
    r = ask(f"Document {uuid.uuid4().hex}:\n\n" + "".join(paras) + q, 200, timeout=14400)
    found = {c: (codes[c] in r.get("content", "")) for c in COLORS}
    rec = {"label": label, "target": target_tokens, "found": sum(found.values()), "by_depth": found,
           **{k: r.get(k) for k in ("prompt_n", "prompt_s", "prompt_tok_s", "tok_s", "wall", "error")},
           "answer": r.get("content", "")[:200]}
    RES["longctx"].append(rec)
    log(f"LONGCTX {label}: {r.get('prompt_n')} tok read in {r.get('prompt_s')} s ({r.get('prompt_tok_s')} tok/s), "
        f"found {sum(found.values())}/5 {found}, decode {r.get('tok_s')} tok/s {r.get('error', '')}")
    if not r.get("error"):          # a follow-up in the same conversation is not possible through ask(); a new short
        r2 = ask("What is 12 * 12? Number only.", 20)   # request shows the server still answers after the long one
        log(f"   after it, a short request: {r2.get('content', r2.get('error'))!r} in {r2.get('wall')} s")


# ------------------------------------------------------------------------------------------------ pictures
def vision(label):
    from PIL import Image, ImageDraw, ImageFont
    for name, size, px, rows in (("1600x900 16px", (1600, 900), 16, 40), ("3840x2160 16px", (3840, 2160), 16, 95),
                                 ("3840x2160 28px", (3840, 2160), 28, 60)):
        rng = random.Random(px + size[0])
        font = ImageFont.truetype("consola.ttf", px)
        lines = [f"{i:02d}  cfg.{rng.choice(['port', 'retries', 'timeout', 'workers', 'limit', 'depth'])}_{i} = "
                 f"{rng.randint(1000, 9999)}" for i in range(1, rows + 1)]
        img = Image.new("RGB", size, (30, 30, 30))
        d = ImageDraw.Draw(img)
        for i, s in enumerate(lines):
            d.text((40, 20 + i * int(px * 1.35)), s, font=font, fill=(220, 220, 220))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        hits, walls, ptoks = 0, [], None
        picks = [3, rows // 3, rows // 2, (2 * rows) // 3, rows - 2]
        for ln in picks:
            r = ask([{"type": "text", "text": f"This is a screenshot of code. Copy line {ln:02d} exactly."},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}], 80)
            want = lines[ln - 1].split("  ", 1)[1]
            hits += want in r.get("content", "")
            walls.append(r.get("wall"))
            ptoks = r.get("prompt_n") or ptoks
        RES["vision"].append({"label": label, "image": name, "exact": hits, "of": len(picks), "prompt_tokens": ptoks,
                              "wall": walls})
        log(f"VISION {label} {name}: {hits}/{len(picks)} lines exact, {ptoks} prompt tokens, {walls} s")


def suite(script, tag, args=()):
    t = time.time()
    with (HERE / f"{tag}.log").open("w", encoding="utf-8") as f:
        p = subprocess.run([PY, str(HERE / script), *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    RES["suites"].append({"tag": tag, "exit": p.returncode, "minutes": round((time.time() - t) / 60, 1)})
    log(f"SUITE {tag} finished in {(time.time() - t) / 60:.1f} min (exit {p.returncode})")


def main():
    log("=== run_all started")
    try:
        if not health().get("loaded"):
            configure(131072, 2048)
        # 1. the everyday configuration: 128K, 2048 MiB kept free
        # (the 128K speed test and the round-1 suites were measured before; only what is still missing runs here)
        vision("300 image tokens")
        suite("strata_eval2.py", "eval2-qa", ["qa"])
        suite("strata_eval2.py", "eval2-hard", ["hard"])
        longctx(118000, "128K config")
        # 2. the upper speed limit: a small context and the smallest VRAM reserve
        if configure(8192, 700):
            speed("8K reserve 700")
        # 3. the trained maximum, and pictures with more tokens
        if configure(262144, 2048, vision_tokens=1024):
            speed("256K reserve 2048")
            vision("1024 image tokens")
            longctx(248000, "256K config")
        # 4. past the trained range (experimental rope scaling)
        if configure(524288, 2048):
            mc = health().get("max_context") or 0
            speed(f"{mc // 1024}K (asked 512K) reserve 2048")
            if mc > 270000:
                longctx(mc - 14000, f"{mc // 1024}K config")
    except Exception as e:
        log(f"CRASHED {type(e).__name__}: {e}")
    finally:
        configure(131072, 2048)
        log("=== run_all finished; server restored to 128K / 2048 MiB")


if __name__ == "__main__":
    main()
