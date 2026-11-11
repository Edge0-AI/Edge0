#!/usr/bin/env python3
"""#1012 на живом сервере: запрос ждёт движок, движок умирает во время чтения промпта.

Что проверяем (из релиза v0.1.40.1): раньше запрос, дожидавшийся перезапуска мёртвого движка,
висел навсегда или падал с `list.remove(x)`, а длинный промпт, который читался в момент смерти
движка, завершался через ~300 с. Теперь тот же запрос либо продолжается с новым движком, либо
заканчивается сразу с 503 («не отправлен» -> клиент может повторить).

Ход: отправить длинный запрос -> через N секунд убить процесс движка -> смотреть, чем и когда
закончится запрос. Сервер поднимает движок сам.

usage: probe_restart_waiters.py [--kill-after 15] [--out results_restart.json]
"""
import json
import pathlib
import subprocess
import threading
import time
import urllib.error
import urllib.request

REPO = "/home/dgbox/Strata"
CFG = REPO + "/strata-200k.json"
BASE = "http://127.0.0.1:8080"
cfg = json.load(open(CFG, encoding="utf-8"))
KEY, MODEL = cfg["api_key"], cfg.get("model_name", "strata")

FILLER = ("Разбирается модуль разбора конфигурации: он читает список секций, проверяет обязательные поля и "
          "собирает структуру настроек, которую дальше использует планировщик. Сбой на одной секции не должен "
          "останавливать остальные, поэтому парсер собирает список замечаний. ")


def big_prompt(tokens: int) -> str:
    # ~0.28 токена на символ для русского текста
    return (FILLER * (int(tokens / 0.28 / len(FILLER)) + 1))[:int(tokens / 0.28)]


def send(rec, label, tokens):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user",
                        "content": big_prompt(tokens) + "\n\nОтветь одним словом."}],
                        "max_tokens": 32, "stream": True, "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    out = {"label": label, "prompt_tokens_asked": tokens, "started_at": round(t0, 3),
           "first_delta_s": None, "end_s": None, "http_status": None, "outcome": None,
           "deltas": 0}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out["http_status"] = r.status
            for line in r:
                s = line.decode(errors="replace").strip()
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    if (ch.get("delta") or {}).get("content"):
                        out["deltas"] += 1
                        if out["first_delta_s"] is None:
                            out["first_delta_s"] = round(time.time() - t0, 2)
        out["outcome"] = "done"
    except urllib.error.HTTPError as e:
        out["outcome"] = f"HTTPError {e.code}"
        out["http_status"] = e.code
        try:
            out["error_body"] = e.read().decode()[:300]
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        out["outcome"] = repr(e)[:200]
    out["end_s"] = round(time.time() - t0, 2)
    print(f"[{label}] status={out['http_status']} outcome={out['outcome']} "
          f"first_delta={out['first_delta_s']} end={out['end_s']} deltas={out['deltas']}")
    rec.append(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-after", type=float, default=15.0)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results_restart.json"))
    a = ap.parse_args()

    before = subprocess.run(["pgrep", "-f", REPO + "/engine/strata"], capture_output=True, text=True).stdout.split()
    recs = []
    threads = [threading.Thread(target=send, args=(recs, f"waiter-{i}", 60000)) for i in range(1, 4)]
    for t in threads:
        t.start()
    time.sleep(a.kill_after)
    killed = subprocess.run(["pkill", "-f", REPO + "/engine/strata"], capture_output=True, text=True)
    kill_at = round(time.time(), 3)
    print(f"движок убит через {a.kill_after} с: pkill rc={killed.returncode} (pid до: {before})")
    for t in threads:
        t.join(timeout=420)

    health = None
    for _ in range(80):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + "/health"), timeout=5) as r:
                h = json.loads(r.read().decode())
            if h.get("loaded"):
                health = {"recovered_after_kill_s": round(time.time() - kill_at, 1), "health": h}
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)

    out = {"kill_after_s": a.kill_after, "engine_pids_before": before, "killed_at": kill_at,
           "requests": recs, "server_recovered": health,
           "log_tail": subprocess.run(["bash", "-c",
                                      f"tail -n 200 {REPO}/strata-200k.log | grep -iE 'engine|restart|503|died|wait' | tail -15"],
                                     capture_output=True, text=True).stdout}
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано:", a.out)


if __name__ == "__main__":
    main()
