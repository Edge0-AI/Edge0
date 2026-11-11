#!/usr/bin/env python3
"""Clean long-context decode comparison: same 47K-token prompt, but a request that generates ~200 tokens in both cases, so
the decode rate is not dominated by per-request overhead. Compares --kv-resident 32768 against 65536.

    python dgx_kvresident2.py OUT.json [--tokens 47000]
"""
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = Path.home() / "workspace" / "strata-spark"
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel").split()


def build(tokens, rng):
    body = [rng.choice(WORDS) for _ in range(int(tokens / 1.35))]
    return ("Read the following log carefully.\n\n" + " ".join(body) +
            "\n\nWrite a detailed 250-word analysis of the vocabulary and tone of the log above. Do not stop early.")


def run(cfg_name, tag, port=8091):
    cfg = json.loads((root / cfg_name).read_text())
    cfg["port"] = port
    cfg["log"] = f"/tmp/kvres2-{tag}.log"
    tmp = Path(f"/tmp/kvres2-{tag}.json")
    tmp.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(tmp), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < 1800:
            if proc.poll() is not None:
                raise SystemExit(f"server exited {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(3)
        prompt = build(int(sys.argv[2]) if len(sys.argv) > 2 else 47000, random.Random(7))
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        t = out.get("timings") or {}
        return {"kv_resident": cfg["args"][cfg["args"].index("--kv-resident") + 1],
                "prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second"),
                "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second")}
    finally:
        proc.terminate()
        try:
            proc.wait(180)
        except subprocess.TimeoutExpired:
            proc.kill()


rows = [run("strata-iq3_s.json", "a"), run("strata-iq3s-kv64k.json", "b")]
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
for r in rows:
    print(json.dumps(r))
