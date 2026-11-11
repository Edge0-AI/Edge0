#!/usr/bin/env python3
"""Проба правил 0.1.40.1 про `<tool_call>`, написанный моделью.

Правила из релиза v0.1.40.1 (#804, #1058):
  - вызов в размышлении становится вызовом только если инструменты объявлены и имя среди них;
  - он начинается с начала строки, вне код-фенса (``` или ~~~) и вне inline-кода;
  - закрытый `</think>` присутствует, и после него до конца хода или `</think>` только пробелы или другие вызовы;
  - ход завершился нормально (оборванный по max_tokens, отменённый или ошибочный оставляет текст в размышлении);
  - `<tool_call>` внутри код-фенса или inline-кода в ВИДИМОМ ответе — текст, никогда не вызов.

Скрипт прогоняет кейсы на живом сервере и пишет, что произошло: был ли реальный tool_calls,
остался ли текст вызова в content / reasoning_content, и что показал счётчик
/metrics totals.tool_calls_from_reasoning.

usage: probe_toolcall.py [--out results.json]
"""
import argparse
import json
import pathlib
import re
import subprocess
import time
import urllib.request

REPO = "/home/dgbox/Strata"
CFG = REPO + "/strata-200k.json"
BASE = "http://127.0.0.1:8080"

cfg = json.load(open(CFG, encoding="utf-8"))
KEY = cfg["api_key"]
MODEL = cfg.get("model_name", "strata")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Погода в городе.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]
CALL = '{"name": "get_weather", "arguments": {"city": "\u041c\u043e\u0441\u043a\u0432\u0430"}}'


def post(payload, stream=False):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
    if not stream:
        return json.loads(raw)
    # стриминг: собираем дельты и финальный chunk с tool_calls
    content, reasoning, tool_calls, finish = [], [], [], None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if d.get("tool_calls"):
                tool_calls.extend(d["tool_calls"])
    return {"choices": [{"message": {"content": "".join(content),
                                     "reasoning_content": "".join(reasoning),
                                     "tool_calls": tool_calls},
                         "finish_reason": finish}], "streamed": True}


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def run(case):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": case["prompt"]}],
               "tools": TOOLS, "max_tokens": case.get("max_tokens", 700)}
    if case.get("reasoning") is not None:
        payload["reasoning_effort"] = case["reasoning"]
    if case.get("stream"):
        payload["stream"] = True
    t0 = time.time()
    resp = post(payload, stream=bool(case.get("stream")))
    ch = resp["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    calls = msg.get("tool_calls") or []
    where = []
    if "tool_call" in content:
        where.append("content")
    if "tool_call" in reasoning:
        where.append("reasoning_content")
    return {
        "id": case["id"],
        "expectation": case["expect"],
        "stream": bool(case.get("stream")),
        "max_tokens": case.get("max_tokens", 700),
        "reasoning_effort": case.get("reasoning", "config default"),
        "wall_s": round(time.time() - t0, 2),
        "finish_reason": ch.get("finish_reason"),
        "tool_calls_returned": len(calls),
        "tool_calls": calls[:2],
        "tool_call_text_left_in": where,
        "fence_in_content": bool(re.search(r"```|~~~", content)),
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "call_text_in_inline_code": bool(re.search(r"`[^`]*tool_call[^`]*`", content + reasoning)),
        "call_text_in_fence": bool(re.search(r"```[^\x60]*tool_call[^\x60]*```", content + reasoning, re.S)),
        "content_head": content[:2000],
        "reasoning_head": reasoning[:2000],
    }


CASES = [
    {
        "id": "1-fenced-example-in-answer",
        "expect": "NO call: the call is only an example inside a code fence",
        "prompt": "Объясни одной фразой, как выглядит вызов инструмента. Приведи ровно один пример "
                  "в код-блоке:\n\n```\n<tool_call>" + CALL + "\n</tool_call>\n\n"
                  "Сам ничего не вызывай, инструменты не используй.",
    },
    {
        "id": "2-inline-code-in-answer",
        "expect": "NO call: the call is only inline code in the visible answer (no thinking at all)",
        "reasoning": "none",
        "prompt": "Перепиши ровно одну строку, ничего не меняя и ничего не вызывая: "
                  + "`<tool_call>" + CALL + "`",
    },
    {
        "id": "3-fenced-example-in-thinking",
        "expect": "NO call: inside the thinking the example is fenced",
        "reasoning": "high",
        "prompt": "Обдумай ответ шаг за шагом. В размышлении сравни два формата записи вызова: "
                  "обычный `<tool_call>" + CALL + "` и такой же внутри код-блока. Реальный вызов "
                  "инструмента не делай — нужен только разбор.",
    },
    {
        "id": "4-turn-cut-by-max-tokens",
        "expect": "NO call: the turn is cut by max_tokens, so a call written in the thinking stays text",
        "max_tokens": 60,
        "reasoning": "high",
        "prompt": "Рассуждай подробно и длинно о том, как ты бы вызвал инструмент для погоды в Москве, "
                  "и запиши этот вызов в размышлении отдельной строкой: <tool_call>" + CALL +
                  " Продолжай рассуждать как можно дольше.",
    },
    {
        "id": "5-real-call-top-level-after-sentence",
        "expect": "CALL: a real call at the top level of the answer, on the same line after a sentence",
        "reasoning": "none",
        "prompt": "Какая погода в Москве? Начни ответ с фразы «Сейчас проверю.» и сразу после неё, "
                  "на той же строке, вызови инструмент get_weather для города Москва.",
    },
    {
        "id": "8-fenced-example-in-answer-no-thinking",
        "expect": "NO call: fenced example in the visible answer, thinking off",
        "reasoning": "none",
        "prompt": "Приведи ровно один пример вызова инструмента в код-блоке и больше ничего не пиши:\n\n"
                  "```\n<tool_call>" + CALL + "\n</tool_call>\n```",
    },
    {
        "id": "6-plain-real-call",
        "expect": "CALL: the ordinary path must keep working",
        "prompt": "Узнай погоду в Москве с помощью инструмента.",
    },
    {
        "id": "7-streaming-split-fence",
        "expect": "NO call: streaming, the fence is split across chunks",
        "stream": True,
        "prompt": "Ответь коротко: приведи пример вызова инструмента внутри код-блока, разбив его "
                  "на несколько строк:\n\n```\n<tool_call>\n" + CALL + "\n" + "</tool_call>\n```\n"
                  "Ничего не вызывай.",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results.json"))
    args = ap.parse_args()

    before = metrics().get("totals", {}).get("tool_calls_from_reasoning", 0)
    out = {"engine_version_check": subprocess.run(
        ["python3", "-c", "import json;print(json.load(open('%s/engine/BUILD.json'))['version'])" % REPO],
        capture_output=True, text=True).stdout.strip(),
        "git": subprocess.run(["git", "-C", REPO, "describe", "--tags"],
                              capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool_calls_from_reasoning_before": before, "cases": []}
    for case in CASES:
        try:
            rec = run(case)
        except Exception as exc:  # noqa: BLE001
            rec = {"id": case["id"], "expectation": case["expect"], "error": repr(exc)}
        out["cases"].append(rec)
        print(f"{rec['id']}: calls={rec.get('tool_calls_returned')} "
              f"left_in={rec.get('tool_call_text_left_in')} finish={rec.get('finish_reason')} "
              f"| expect: {rec['expectation']}")
    after = metrics().get("totals", {}).get("tool_calls_from_reasoning", 0)
    out["tool_calls_from_reasoning_after"] = after
    print(f"totals.tool_calls_from_reasoning: {before} -> {after}")
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано:", args.out)


if __name__ == "__main__":
    main()
