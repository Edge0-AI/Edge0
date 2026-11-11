#!/usr/bin/env python3
"""Is decode at 56K limited by KV streaming? Raise --kv-resident to the whole context if it fits, and re-measure the same
56K prompt. Writes one row; compare it with the 31.0 tok/s of the shipped 32,768-cell setting.

    python dgx_kvresident.py CONFIG OUT.json [--tokens 56000]
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
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel").split()


def build(tokens, code, rng):
    body = [rng.choice(WORDS) for _ in range(int(tokens / 1.35))]
    at = rng.randrange(len(body) // 3, 2 * len(body) // 3)
    body[at:at] = [f"The access code for the harbor gate is {code}."]
    return ("Read the following log carefully, then answer the question at the end.\n\n" + " ".join(body) +
            "\n\nQuestion: what is the access code for the harbor gate? Answer with the code only.")


def main():
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 56000
    port = 8091
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["port"] = port
    cfg["log"] = "/tmp/kvresident.log"
    tmp = Path("/tmp/kvresident.json")
    tmp.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(tmp), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < 1800:
            if proc.poll() is not None:
                raise SystemExit(f"server exited {proc.returncode} (see /tmp/kvresident.log)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(3)
        print(f"ready in {time.time() - t0:.0f} s", flush=True)
        rng = random.Random(7)
        code = "481516"
        prompt = build(tokens, code, rng)
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        wall = time.time() - t1
        t = out.get("timings") or {}
        answer = out["choices"][0]["message"].get("content", "")
        row = {"tokens": tokens, "kv_resident": cfg["args"][cfg["args"].index("--kv-resident") + 1],
               "prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second"),
               "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
               "wall_s": round(wall, 1), "recalled": code in answer}
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
