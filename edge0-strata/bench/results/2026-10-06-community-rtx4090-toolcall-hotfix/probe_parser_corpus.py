#!/usr/bin/env python3
"""Повтор корпуса `serve/fixtures/rcall_specimens.json` на этой машине (0.1.40.1, #804 / #1058).

Корпус автора: 37 образцов — 16 должны дать вызов инструмента, 21 должны остаться текстом.
Прогоняем каждый образец через тот же `serve.frontend.OutputParser`, что использует сервер,
шестью ширинами подачи (1, 2, 3, 7, случайная 1-9, целиком) и при двух режимах
`stream_tools` — ровно как в `serve/test_reasoning_rescue.py`, но с результатом по каждому
образцу в JSON, а не «зелёно/красно».

usage: probe_parser_corpus.py [--out corpus_results.json]
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path("/home/dgbox/Strata")
sys.path.insert(0, str(ROOT))
from serve.frontend import OutputParser  # noqa: E402

SPECIMENS = json.loads((ROOT / "serve/fixtures/rcall_specimens.json").read_text(encoding="utf-8"))["specimens"]
WIDTHS = (1, 2, 3, 7, 0, 100_000)


def parse(text, width, tools, finish, stream_tools):
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools)
    evs, rnd, i = [], random.Random(7), 0
    while i < len(text):
        k = width or rnd.randint(1, 9)
        evs += p.feed(text[i:i + k])
        i += k
    return evs + p.finish(finish)


def calls(evs):
    return [e.call.name for e in evs if e.kind == "tool_call"]


def reasoning(evs):
    return "".join(e.text for e in evs if e.kind == "reasoning")


def main():
    out = {"corpus": "serve/fixtures/rcall_specimens.json", "n_specimens": len(SPECIMENS),
           "widths": list(WIDTHS), "cases": []}
    bad = 0
    for sp in SPECIMENS:
        tools = [{"name": n, "parameters": {"properties": {}}} for n in sp["tools"]]
        seen, fails = set(), []
        for width in WIDTHS:
            for st in (True, False):
                evs = parse(sp["input"], width, tools, sp["finish"], st)
                got, want = calls(evs), sp["calls"]
                if got != want:
                    fails.append({"width": width, "stream_tools": st, "got": got, "want": want})
                seen.add((reasoning(evs), tuple(got)))
        ok = not fails and len(seen) == 1
        bad += 0 if ok else 1
        out["cases"].append({"name": sp["name"], "expect_calls": sp["calls"],
                             "runs": len(WIDTHS) * 2, "mismatches": fails,
                             "streamed_matches_whole": len(seen) == 1, "pass": ok})
    out["totals"] = {"specimens": len(SPECIMENS),
                     "expected_to_act": sum(bool(s["calls"]) for s in SPECIMENS),
                     "expected_to_stay_text": sum(not s["calls"] for s in SPECIMENS),
                     "failed": bad}
    path = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else pathlib.Path(__file__).parent / "corpus_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"образцов {len(SPECIMENS)}, должны дать вызов: {out['totals']['expected_to_act']}, "
          f"должны остаться текстом: {out['totals']['expected_to_stay_text']}, провалились: {bad}")
    for c in out["cases"]:
        if not c["pass"]:
            print("ПРОВАЛ:", c["name"], c["mismatches"][:2])
    print("записано:", path)


if __name__ == "__main__":
    main()
