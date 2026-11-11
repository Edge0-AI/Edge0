#!/usr/bin/env python3
"""Замер параллельности Strata: сколько запросов одновременно и что с скоростью.

Нужен для проверки `"parallel": N` / `--batch N` / `--batch-mtp` (docs/BATCHING.md):
штатный strata-bench.py измеряет один запрос, а batching меняет именно картину
нескольких клиентов. Здесь два режима:

  solo  — по одному запросу (опора: что должно остаться на одном клиенте)
  conc  — N запросов одновременно (2, 3, 4): что получает каждый и что — сервер в целом

Каждый запрос содержит случайную метку -> кэш префиксов не переиспользуется.
Скорость считаем из таймингов движка (поле timings в ответе без стриминга),
TTFT и ожидание в очереди — из стриминга. Это два прохода на каждый размер:
движковые тайминги и клиентские задержки в одном ответе не появляются.

usage: strata-conc-bench.py [--levels 1,2,3,4] [--prompt-tokens 8000] [--max-tokens 256]
                            [--repeats 2] [--json out.json] [--label TEXT]
"""
import argparse
import datetime
import json
import os
import random
import re
import statistics
import string
import subprocess
import sys
import threading
import time
import urllib.request

REPO = os.environ.get("STRATA_REPO", "/home/dgbox/Strata")
CFG = os.environ.get("STRATA_CFG", REPO + "/strata-200k.json")
PORT = int(os.environ.get("STRATA_PORT", "8080"))
BASE = f"http://127.0.0.1:{PORT}"

FILLERS = [
    "В документе описывается модуль разбора конфигурации: он читает список секций, "
    "проверяет обязательные поля и собирает структуру настроек, которую дальше использует планировщик.",
    "Далее разбирается обработка ошибок: сбой на одной секции не должен останавливать остальные, "
    "поэтому парсер собирает список замечаний и возвращает его вместе с частично прочитанными настройками.",
    "Отдельно описан кэш: ключом служит хэш секции, запись живёт до изменения исходного файла, "
    "а при несовпадении версии запись отбрасывается и секция читается заново.",
]


def cfg_value(key):
    try:
        with open(CFG, encoding="utf-8") as f:
            c = json.load(f)
    except Exception:
        return None
    return c.get(key)


def api_key():
    return cfg_value("api_key") or ""


def server_cmdline():
    """Аргументы движка, с которыми он сейчас запущен (для провенанса)."""
    try:
        out = subprocess.run(["pgrep", "-f", "engine/strata --serve"], capture_output=True, text=True).stdout.split()
        for pid in out[:1]:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
            return " ".join(x for x in cmd if x)
    except Exception:
        pass
    return ""


def engine_version():
    try:
        req = urllib.request.Request(BASE + "/props", headers={"Authorization": f"Bearer {api_key()}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r).get("build_info")
    except Exception:
        return None


def batch_slots():
    """Сколько слотов движок реально открыл (concurrency.serving из /v1/status)."""
    for path in ("/v1/status", "/metrics"):
        try:
            req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {api_key()}"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
        except Exception:
            continue
        for k in ("concurrency", "live"):
            v = d.get(k)
            if isinstance(v, dict):
                for kk in ("serving", "slots", "batch_slots"):
                    if isinstance(v.get(kk), int):
                        return v[kk]
                if isinstance(v.get("slots"), dict):
                    return len(v["slots"])
    return None


def make_prompt(tokens, ratio):
    """Промпт нужной длины со случайной меткой.

    ratio — токенов на символ (у этой модели на русском ~0.28). Значит нужное число
    символов = tokens / ratio. Фактическую длину в токенах видно по prompt_n в ответе.
    """
    marker = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    body = []
    n = 0
    while n < tokens / ratio:
        body.append(f"[{marker}-{len(body)}] " + random.choice(FILLERS))
        n += len(body[-1])
    text = "\n".join(body)
    return text, marker


def post(path, payload, stream=False, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key()}"})
    return urllib.request.urlopen(req, timeout=timeout)


def one_request(prompt_tokens, max_tokens, ratio, stream_mode):
    """Один запрос. Возвращает dict с движковыми таймингами и (если стрим) TTFT/wall."""
    text, marker = make_prompt(prompt_tokens, ratio)
    payload = {"messages": [{"role": "user", "content": text + "\nКратко перескажи, что здесь описано."}],
               "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none"}
    t0 = time.time()
    ttft = None
    if stream_mode:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        resp = post("/v1/chat/completions", payload, stream=True)
        gen = 0
        timings = None
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            if d.get("timings"):
                timings = d["timings"]
            if d.get("usage"):
                gen = d["usage"].get("completion_tokens", gen)
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and ttft is None:
                    ttft = time.time() - t0
                gen += len(piece)
        wall = time.time() - t0
        return {"ok": True, "ttft_s": ttft, "wall_s": wall, "gen_n": gen,
                "prompt_s": (timings or {}).get("prompt_per_second"),
                "gen_s": (timings or {}).get("predicted_per_second"),
                "prompt_n": (timings or {}).get("prompt_n")}
    resp = post("/v1/chat/completions", payload)
    d = json.loads(resp.read().decode())
    wall = time.time() - t0
    t = d.get("timings") or {}
    return {"ok": True, "wall_s": wall, "ttft_s": None,
            "gen_n": d.get("usage", {}).get("completion_tokens"),
            "prompt_n": t.get("prompt_n"), "prompt_s": t.get("prompt_per_second"),
            "gen_s": t.get("predicted_per_second"),
            "draft_ok": t.get("draft_n_accepted"), "draft_n": t.get("draft_n")}


def run_level(n, prompt_tokens, max_tokens, ratio, repeats):
    """n одновременных запросов; repeats повторов уровня."""
    rows = []
    for _ in range(repeats):
        results = [None] * n
        errs = [None] * n

        def worker(i):
            try:
                results[i] = one_request(prompt_tokens, max_tokens, ratio, stream_mode=True)
            except Exception as e:
                errs[i] = f"{type(e).__name__}: {e}"

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall_all = time.time() - t0
        rows.append({"n": n, "wall_all_s": wall_all,
                     "clients": [r for r in results if r], "errors": [e for e in errs if e]})
    return rows


def med(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def summarize(level_rows):
    out = []
    for row in level_rows:
        cl = row["clients"]
        gen = [c.get("gen_n") or 0 for c in cl]
        ttft = [c.get("ttft_s") for c in cl]
        decode = [(c["gen_n"] / (c["wall_s"] - c["ttft_s"])) for c in cl
                  if c.get("gen_n") and c.get("ttft_s") is not None and c["wall_s"] > c["ttft_s"]]
        out.append({"n": row["n"], "clients_ok": len(cl), "errors": row["errors"],
                    "prompt_n": med([c.get("prompt_n") for c in cl]),
                    "wall_all_s": round(row["wall_all_s"], 2),
                    "gen_per_client": med(gen), "gen_total": sum(gen),
                    "ttft_med_s": round(med(ttft) or 0, 2),
                    "ttft_max_s": round(max([t for t in ttft if t is not None] or [0]), 2),
                    "decode_client_med": round(med(decode) or 0, 1),
                    "throughput_total": round(sum(gen) / row["wall_all_s"], 1) if row["wall_all_s"] else 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warm-tokens", type=int, default=2000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ratio = float(os.environ.get("STRATA_TOKENS_PER_CHAR", "0.28"))  # токенов на символ (русский текст)
    ver = engine_version()
    slots = batch_slots()
    print(f"сервер: {BASE}, движок: {ver}, слотов движка (concurrency.serving): {slots}")
    print(f"аргументы движка: {server_cmdline()[:400]}")
    print(f"прогрев ({a.warm_tokens} токенов)...")
    one_request(a.warm_tokens, 64, ratio, stream_mode=False)

    data = {"label": a.label, "engine": ver, "batch_slots": slots,
            "cmdline": server_cmdline(), "config": CFG,
            "prompt_tokens": a.prompt_tokens, "max_tokens": a.max_tokens,
            "repeats": a.repeats, "levels": [], "ts": datetime.datetime.now().isoformat()}
    for n in [int(x) for x in a.levels.split(",")]:
        print(f"\n=== уровень {n} одновременных запросов ({a.repeats} повтора) ===")
        rows = run_level(n, a.prompt_tokens, a.max_tokens, ratio, a.repeats)
        summ = summarize(rows)
        for s in summ:
            print(f"  n={s['n']} ок={s['clients_ok']} wall={s['wall_all_s']}с "
                  f"токенов на клиента={s['gen_per_client']} всего={s['gen_total']} "
                  f"TTFT мед={s['ttft_med_s']}с макс={s['ttft_max_s']}с "
                  f"генерация на клиента={s['decode_client_med']} ток/с "
                  f"итоговая скорость={s['throughput_total']} ток/с")
            if s["errors"]:
                print("   ошибки:", s["errors"])
        data["levels"].append({"n": n, "runs": rows, "summary": summ})
        time.sleep(3)

    out = a.json or f"/home/dgbox/logs/conc-{datetime.date.today()}-{re.sub(r'[^a-z0-9]+', '-', a.label.lower()).strip('-')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
