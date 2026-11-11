#!/usr/bin/env python3
"""Does the engine hold a context past the installed 64K? Copy the shipped config, raise the context to 128K, and send a
100K-token prompt with a hidden code. Success = an answer that recalls it; failure is recorded as a limit.

    python dgx_128k.py CONFIG OUT.json [--tokens 100000] [--ctx 131072]
"""
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel "
         "lattice mirror nectar oasis pepper quarry raven saddle tundra umber vessel willow xenon yellow zephyr").split()


def build(tokens, code, rng):
    body = [rng.choice(WORDS) for _ in range(int(tokens / 1.35))]
    at = rng.randrange(len(body) // 3, 2 * len(body) // 3)
    body[at:at] = [f"The access code for the harbor gate is {code}."]
    return ("Read the following log carefully, then answer the question at the end.\n\n" + " ".join(body) +
            "\n\nQuestion: what is the access code for the harbor gate? Answer with the code only.")


def main():
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 131072
    port = 8091
    cfg = json.loads(Path(cfg_path).read_text())
    a = cfg["args"]
    if "--max-context" in a:
        a[a.index("--max-context") + 1] = str(ctx)
    if "--kv-resident" in a:
        a[a.index("--kv-resident") + 1] = "32768"
    cfg["port"] = port
    cfg["log"] = "/tmp/long-128k.log"
    tmp = Path("/tmp/long-128k.json")
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
        print(f"ready in {time.time() - t0:.0f} s", flush=True)
        rng = random.Random(11)
        code = "271828"
        prompt = build(tokens, code, rng)
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=5400) as r:
            out = json.loads(r.read())
        wall = time.time() - t1
        t = out.get("timings") or {}
        answer = out["choices"][0]["message"].get("content", "")
        row = {"tokens": tokens, "ctx": ctx, "prompt_n": t.get("prompt_n"),
               "prompt_per_second": t.get("prompt_per_second"),
               "predicted_per_second": t.get("predicted_per_second"), "wall_s": round(wall, 1),
               "answer": answer[:80], "recalled": code in answer}
        Path(out_path).write_text(json.dumps(row, indent=1))
        print(json.dumps(row))
    finally:
        proc.terminate()
        try:
            proc.wait(180)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
