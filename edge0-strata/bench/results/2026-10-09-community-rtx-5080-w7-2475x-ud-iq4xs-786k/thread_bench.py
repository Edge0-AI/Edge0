"""thread_bench.py - velocita' di generazione a contesto lungo su un thread VERO di Jan.

Gemello di long_bench.py: invece di un file di testo manda i messaggi di un thread di Jan (messages.jsonl), in
ordine, piu' una domanda fissa in fondo che chiede almeno 600 parole, cosi' ogni corsa genera esattamente
--max-tokens token. Testo dei messaggi: le parti `text`; le chiamate ai tool scritte come testo nel messaggio
dell'assistente; i ragionamenti vecchi tolti (il template non li rimanda comunque).

    1. run-thread.bat sp05     (oppure sp06, sp07) e aspetta "ready"
    2. python thread_bench.py --label thread-sp05-0.1.36 --runs 9
    3. chiudi il server, avvia l'altro braccio, ripeti
    4. python perm_test.py thread-sp05-0.1.36 thread-sp06-0.1.36

La prima richiesta paga la lettura a freddo (campione singolo, non entra nelle corse). Le corse riusano il prefisso
in cache. Contatore della porta (guard.py): una richiesta estranea rende la misura SPORCA e il file non si salva.
Il risultato va in C:\\strata-lab\\results\\pw-<label>.json, stesso schema di pw_bench / long_bench.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from guard import report_extra, requests_total

THREAD = r"C:\strata-lab\jan-threads\nasdaqpredictor-50c5608c-2026-10-02\messages.jsonl"
QUESTION = ("Riassumi in dettaglio tutto il lavoro svolto in questa conversazione: per ogni fase indica cosa e' "
            "stato fatto, quali file e quali risultati numerici sono stati prodotti e quali decisioni sono state "
            "prese. Scrivi almeno 600 parole.")


def part_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and "value" in x:
        return str(x["value"])
    return json.dumps(x, ensure_ascii=False)


def thread_messages(path: str) -> list[dict]:
    msgs = []
    for line in open(path, encoding="utf-8"):
        m = json.loads(line)
        pieces = []
        for c in m["content"]:
            if c["type"] == "text":
                pieces.append(part_text(c["text"]))
            elif c["type"] == "tool_call":
                pieces.append(f"[tool {c.get('tool_name')}]\n{part_text(c.get('input'))}\n"
                              f"[risultato]\n{part_text(c.get('output'))}")
        msgs.append({"role": m["role"], "content": "\n\n".join(p for p in pieces if p)})
    return msgs


def one(url: str, messages: list[dict], max_tokens: int, timeout: float) -> dict:
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages + [{"role": "user", "content": QUESTION}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    t = out["timings"]
    text = out["choices"][0]["message"].get("content") or ""
    return {"tok_s": t["predicted_per_second"], "n": t["predicted_n"], "wall_s": round(wall, 1),
            "cache_n": t["cache_n"], "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
            "prompt_per_second": t["prompt_per_second"],
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "finish": out["choices"][0].get("finish_reason"), "text_head": text[:80]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="es. thread-sp05-0.1.36")
    ap.add_argument("--thread", default=THREAD)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()

    out = Path(r"C:\strata-lab\results") / f"pw-{a.label}.json"
    if out.exists():
        print(f"{out} esiste gia': scegli un'altra etichetta")
        return 1
    messages = thread_messages(a.thread)
    print(f"thread: {a.thread}, {len(messages)} messaggi, {sum(len(m['content']) for m in messages)} caratteri",
          flush=True)
    reqs_before = requests_total(a.url)

    print("lettura a freddo (campione singolo, scalda la cache del prompt) ...", flush=True)
    cold = one(a.url, messages, a.max_tokens, a.timeout)
    print(f"  prompt {cold['prompt_n']} token in {cold['prompt_ms'] / 1000:.1f} s "
          f"({cold['prompt_per_second']} tok/s), cache_n {cold['cache_n']}; "
          f"generazione {cold['tok_s']} tok/s su {cold['n']} token", flush=True)

    runs = []
    for i in range(a.runs):
        r = one(a.url, messages, a.max_tokens, a.timeout)
        runs.append(r)
        acc = f"{r['draft_n_accepted'] / r['draft_n']:.3f}" if r["draft_n"] else "-"
        print(f"corsa {i + 1}: {r['tok_s']} tok/s, {r['n']} token, {r['wall_s']} s, "
              f"cache_n {r['cache_n']}, prompt_n {r['prompt_n']}, bozze accettate {acc}", flush=True)

    reqs_after = requests_total(a.url)
    clean = report_extra(reqs_after - reqs_before - 1 - a.runs, "il banco", reqs_before, reqs_after)
    ns = {r["n"] for r in runs}
    if len(ns) != 1:
        print(f"ATTENZIONE: le corse non hanno generato lo stesso numero di token ({sorted(ns)}): "
              "il confronto non e' alla pari")
    if any(r["cache_n"] == 0 for r in runs):
        print("ATTENZIONE: qualche corsa ha riletto il prompt da zero (cache_n 0): non e' il regime voluto")
    speeds = [r["tok_s"] for r in runs]
    acc_all = [r["draft_n_accepted"] / r["draft_n"] for r in runs if r["draft_n"]]
    print(f"\n{a.label}: mediana {statistics.median(speeds)} tok/s, media {statistics.mean(speeds):.2f}, "
          f"min {min(speeds)}, max {max(speeds)}"
          + (f"; bozze accettate media {statistics.mean(acc_all):.3f}" if acc_all else ""))
    if not clean:
        print("NON salvato: misura sporca.")
        return 1
    out.write_text(json.dumps({"label": a.label, "max_tokens": a.max_tokens, "thread": a.thread,
                               "question": QUESTION, "prefill_freddo": cold, "runs": runs}, indent=1))
    print(f"salvato in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
