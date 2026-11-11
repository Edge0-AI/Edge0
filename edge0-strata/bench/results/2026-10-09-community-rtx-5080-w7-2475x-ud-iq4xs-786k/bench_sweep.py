"""bench_sweep.py - sweep 4K / 32K / 128K per il report community (docs/COMMUNITY_BENCHMARKS.md).

Adattato da bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3/benchmark.py: prompt sintetici (testo ripetuto)
con un marcatore di revisione diverso per corsa, cosi' nessuna corsa riusa il prefisso di un'altra; cap 256 token,
temperature 0, thinking spento, richiesta non-streaming. I tempi sono quelli del server (campo `timings`), TTFT =
tempo totale meno la generazione. In piu' rispetto all'originale: contatore delle richieste (guard.py) e il
conteggio reale dei token del prompt dal server (`prompt_n`), non quello mirato.

    server quotidiano acceso, Jan scollegato
    python bench_sweep.py --label sweep-0.1.41
Scrive C:\\strata-lab\\results\\bench-5080\\<label>.json (riepilogo + tutte le corse).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from guard import report_extra, requests_total

OUT = Path(r"C:\strata-lab\results\bench-5080")
LENGTHS = (4096, 32768, 128000)
RUNS = 3
GEN = 256

BASE = ("Sparse mixture-of-experts routing sends each token through a small subset of "
        "feed-forward experts. The gate projects hidden states into per-expert scores, applies a "
        "top-k mask and renormalizes the surviving weights. Recurrent delta layers maintain linear "
        "state across the sequence while sparse attention indexes only a subset of keys. "
        "Quantized weights trade numerical precision for memory bandwidth at every matmul. ")


def make(n_tokens: int, variant: int) -> str:
    # lunghezza nel marcatore: altrimenti la corsa 32K riusa dalla cache il prefisso della 4K con lo stesso variant
    s = BASE + (" Document revision marker %d-%d. " % (n_tokens, variant))
    # tokens per ripetizione: misurato dal server sul primo sweep, 0.78-0.80 token per 4 caratteri (3192 token per
    # 4096 mirati); la stima "4 caratteri = 1 token" dello script Titan lasciava i prompt corti del 20%
    per = len(s) / 4 * 0.79
    reps = max(1, round(n_tokens / per))
    # domanda che riempie il cap di 256 token: un riassunto libero si fermava a 78-95 token ("finish stop") e il
    # decode era misurato su troppo pochi token
    return s * reps + ("\n\nWrite a detailed essay of at least 600 words on the ideas in the text above: "
                       "explain each mechanism, its trade-offs and how they interact.")


def req(url: str, prompt: str, timeout: float) -> dict:
    body = {"model": "strata", "max_tokens": GEN, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    r = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as f:
        d = json.loads(f.read())
    wall = time.time() - t0
    tm = d.get("timings", {})
    u = d.get("usage", {})
    return {"prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "prompt_ms": tm.get("prompt_ms"),
            "prompt_tps": tm.get("prompt_per_second"), "gen_n": tm.get("predicted_n"),
            "gen_ms": tm.get("predicted_ms"), "gen_tps": tm.get("predicted_per_second"),
            "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
            "ttft_s": round(wall - (tm.get("predicted_ms", 0) or 0) / 1000, 2), "total_s": round(wall, 2),
            "usage_prompt": u.get("prompt_tokens"), "usage_completion": u.get("completion_tokens"),
            "finish": d["choices"][0].get("finish_reason")}


def stats(xs: list[float], nd: int) -> dict:
    return {"runs": [round(x, nd) for x in xs], "median": round(statistics.median(xs), nd),
            "min": round(min(xs), nd), "max": round(max(xs), nd)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS), help="token mirati, separati da virgola")
    ap.add_argument("--runs", type=int, default=RUNS)
    a = ap.parse_args()
    lengths = tuple(int(x) for x in a.lengths.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.label}.json"
    if out.exists():
        print(f"{out} esiste gia': scegli un'altra etichetta")
        return 1
    before = requests_total(a.url)
    results: dict[str, list[dict]] = {}
    for length in lengths:
        runs = []
        for v in range(a.runs):
            r = req(a.url, make(length, v), a.timeout)
            r["variant"] = v
            runs.append(r)
            print(f"{length:>6} corsa {v + 1}: prompt_n {r['prompt_n']} (cache_n {r['cache_n']}) "
                  f"prompt {r['prompt_tps']:.1f} tok/s  decode {r['gen_tps']:.2f} tok/s su {r['gen_n']} token  "
                  f"ttft {r['ttft_s']} s  finish {r['finish']}", flush=True)
        results[str(length)] = runs
    after = requests_total(a.url)
    clean = report_extra(after - before - a.runs * len(lengths), "lo sweep", before, after)
    summary = {k: {"prompt_n": [x["prompt_n"] for x in v], "cache_n": [x["cache_n"] for x in v],
                   "gen_n": [x["gen_n"] for x in v],
                   "prompt_tps": stats([x["prompt_tps"] for x in v], 1),
                   "decode_tps": stats([x["gen_tps"] for x in v], 2),
                   "ttft_s": stats([x["ttft_s"] for x in v], 2)} for k, v in results.items()}
    if not clean:
        print("misura SPORCA: non salvo")
        return 1
    json.dump({"label": a.label, "gen_cap": GEN, "temperature": 0, "thinking": False,
               "summary": summary, "runs": results}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"salvato in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
