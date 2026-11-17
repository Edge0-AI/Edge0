"""par_bench.py - aggregate decode throughput of N requests at once: strata-q36 (--batch) and llama.cpp (llama-server -np N),
same GGUF, same prompt ids, greedy, 128 new tokens each.  One JSON line per (engine, N, round).

  python q36_par_bench.py strata --exe E --args "--model M --max-context 8192 --batch 8 --mtp 0" --ns 1,2,4,8 --rounds 3 --out F
  python q36_par_bench.py llama --server BIN --model M --ns 1,2,4,8 --rounds 3 --out F [--port 8197]

The strata wall time runs from the first BGEN to the last BDONE (admissions included); the llama one from the first request
sent to the last answer (prompt reading included); tok/s = tokens produced / wall.  `tail_tok_s` (strata): the window phase
only, after the last admission."""
from pathlib import Path
import argparse, json, os, shlex, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

ORDER = [0, 1, 2, 3, 5, 8, 0, 1]   # the prompts that run the full 128 tokens, repeated to fill 8 slots


def strata(a):
    prompts = load_prompts(a.prompts)
    eng = Eng(a.exe, shlex.split(a.args), log=a.log)
    out = open(a.out, "a")
    for n in [int(x) for x in a.ns.split(",")]:
        for rnd in range(a.rounds):
            if n == 1 and not eng.info.get("batch_slots"):   # an engine without slots: the request alone (GEN)
                toks, done, ttft, wall = eng.solo(prompts[ORDER[0]], a.n)
                row = {"engine": "strata", "n": 1, "round": rnd, "tokens": len(toks), "wall": wall,
                       "tok_s": len(toks) / wall, "tail_tok_s": len(toks) / max(1e-9, wall - (ttft or 0))}
            else:
                reqs = [{"ids": prompts[ORDER[i]], "n": a.n, "slot": i} for i in range(n)]
                r = eng.batch(reqs)
                tot = sum(len(t) for t in r["tokens"])
                row = {"engine": "strata", "n": n, "round": rnd, "tokens": tot, "wall": r["wall"], "tok_s": tot / r["wall"],
                       "tail_tok_s": r["tail_tok_s"]}
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")
            out.flush()
    eng.close()


def post(port, body):
    return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"}), timeout=900).read())


def llama(a):
    prompts = load_prompts(a.prompts)
    out = open(a.out, "a")
    for n in [int(x) for x in a.ns.split(",")]:
        cmd = [a.server, "-m", a.model, "-ngl", "99", "-fa", "on", "-c", str(8192 * n), "-np", str(n), "--port", str(a.port),
               "--no-webui", "-t", "16"] + shlex.split(a.llama_args)
        log = open(a.log or os.devnull, "a")
        p = subprocess.Popen(cmd, stdout=log, stderr=log)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(1)
            for rnd in range(a.rounds + 1):   # round 0: the warm-up
                res = [None] * n

                def one(i):
                    res[i] = post(a.port, {"prompt": prompts[ORDER[i]], "n_predict": a.n, "temperature": 0, "top_k": 1,
                                           "cache_prompt": False, "id_slot": i})
                ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
                t0 = time.time()
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                wall = time.time() - t0
                tot = sum(r["tokens_predicted"] for r in res)
                if rnd == 0:
                    continue
                row = {"engine": "llama", "n": n, "round": rnd - 1, "tokens": tot, "wall": wall, "tok_s": tot / wall}
                print(json.dumps(row), flush=True)
                out.write(json.dumps(row) + "\n")
                out.flush()
        finally:
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["strata", "llama"])
ap.add_argument("--exe")
ap.add_argument("--args", default="")
ap.add_argument("--server")
ap.add_argument("--model")
ap.add_argument("--llama-args", default="")
ap.add_argument("--ns", default="1,2,4,8")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"))
ap.add_argument("--port", type=int, default=8197)
ap.add_argument("--out", required=True)
ap.add_argument("--log", default=None)
a = ap.parse_args()
(strata if a.mode == "strata" else llama)(a)
