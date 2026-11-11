"""bench-local/strata_eval.py - a small capability check of a running Strata server (http://127.0.0.1:8080).

Sections: coding tasks with hidden tests, reasoning puzzles per thinking level, tool calling, JSON output,
conversation follow-up (prefix reuse) and a Ukrainian check.  Writes results.json next to this file.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
HERE = Path(__file__).parent
OUT = {"coding": [], "reasoning": [], "tools": [], "json": [], "followup": [], "ukrainian": []}


def chat(messages, effort="none", max_tokens=4000, **extra):
    t = time.time()
    r = requests.post(URL, json={"model": "strata", "messages": messages, "max_tokens": max_tokens,
                                 "reasoning_effort": effort, "temperature": 0, **extra}, timeout=1800).json()
    wall = time.time() - t
    if "choices" not in r:
        return {"error": json.dumps(r)[:400], "wall": wall}
    m, tm = r["choices"][0]["message"], r.get("timings", {})
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls"), "finish": r["choices"][0].get("finish_reason"), "wall": round(wall, 1),
            "out_tokens": tm.get("predicted_n"), "tok_s": tm.get("predicted_per_second"),
            "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "prompt_s": round(tm.get("prompt_ms", 0) / 1000, 1)}


# ------------------------------------------------------------------------------------------------ coding
CODING = [
    ("merge_intervals", "Write `merge_intervals(intervals)`: takes a list of [start, end] pairs (any order), merges "
     "overlapping or touching ones and returns them sorted as a list of lists.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\n"
     "assert merge_intervals([[5,7],[1,2],[2,3],[6,6]]) == [[1,3],[5,7]]\n"),
    ("lru_cache", "Write a class `LRUCache(capacity)` with `get(key)` (returns -1 when missing) and `put(key, value)`; "
     "both O(1); the least recently used key is evicted when over capacity.",
     "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1) == 1; c.put(3,3); assert c.get(2) == -1\n"
     "c.put(4,4); assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4\n"
     "c.put(3,30); c.put(5,5); assert c.get(4) == -1 and c.get(3) == 30\n"),
    ("parse_duration", "Write `parse_duration(s)`: parses strings like '1d2h30m15s', '90m', '45s' (units d, h, m, s, "
     "each at most once, in that order, any subset) and returns total seconds as int. Raise ValueError for anything "
     "else (empty string, unknown unit, wrong order, repeated unit, negative numbers, spaces).",
     "assert parse_duration('1d2h30m15s') == 95415\nassert parse_duration('90m') == 5400\n"
     "assert parse_duration('45s') == 45\nassert parse_duration('2h5s') == 7205\n"
     "for bad in ['', '5x', '30m1h', '1h1h', '-5m', '1h 5m', 'h', '10']:\n"
     "    try:\n        parse_duration(bad); raise AssertionError(bad)\n    except ValueError:\n        pass\n"),
    ("top_k_frequent", "Write `top_k_frequent(words, k)`: the k most frequent words, most frequent first; ties broken "
     "alphabetically.",
     "assert top_k_frequent(['i','love','leetcode','i','love','coding'], 2) == ['i','love']\n"
     "assert top_k_frequent(['b','a','c','a','b','c'], 2) == ['a','b']\n"
     "assert top_k_frequent(['x'], 1) == ['x']\n"),
    ("eval_expr", "Write `eval_expr(s)`: evaluates an arithmetic expression string with + - * /, parentheses, unary "
     "minus, integers and decimals, and spaces. Standard precedence, / is true division. Do not use eval or exec. "
     "Return a float.",
     "assert abs(eval_expr('1 + 2 * 3') - 7) < 1e-9\nassert abs(eval_expr('(1 + 2) * 3') - 9) < 1e-9\n"
     "assert abs(eval_expr('-3 + 5') - 2) < 1e-9\nassert abs(eval_expr('2 * -(3 + 1) / 4') + 2) < 1e-9\n"
     "assert abs(eval_expr('10 / 4 - 0.5') - 2) < 1e-9\nassert abs(eval_expr('2 - 3 - 4') + 5) < 1e-9\n"
     "assert abs(eval_expr('((2))*(3+(4*5))') - 46) < 1e-9\n"),
    ("int_to_roman", "Write `int_to_roman(n)` for 1 <= n <= 3999 and `roman_to_int(s)` as its inverse.",
     "assert int_to_roman(1994) == 'MCMXCIV' and int_to_roman(3999) == 'MMMCMXCIX' and int_to_roman(4) == 'IV'\n"
     "assert all(roman_to_int(int_to_roman(i)) == i for i in range(1, 4000))\n"),
    ("dijkstra", "Write `dijkstra(graph, src)`: graph is a dict {node: {neighbor: weight}} with non-negative weights "
     "(directed). Return a dict of shortest distances from src to every node that appears in the graph (as a key or "
     "as a neighbor); unreachable nodes get float('inf').",
     "g = {'a': {'b': 1, 'c': 4}, 'b': {'c': 2, 'd': 6}, 'c': {'d': 3}, 'e': {'a': 1}}\n"
     "d = dijkstra(g, 'a')\nassert d == {'a': 0, 'b': 1, 'c': 3, 'd': 6, 'e': float('inf')}, d\n"),
    ("bsearch_fix", "This function should return the index of the LEFTMOST occurrence of x in the sorted list a, or "
     "-1. It has bugs. Return the fixed function, same name.\n\n```python\ndef bsearch(a, x):\n    lo, hi = 0, len(a)\n"
     "    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] <= x:\n            lo = mid\n        else:\n"
     "            hi = mid - 1\n    return lo if a[lo] == x else -1\n```",
     "assert bsearch([1,2,2,2,3], 2) == 1\nassert bsearch([], 1) == -1\nassert bsearch([1,3,5], 4) == -1\n"
     "assert bsearch([1,3,5], 5) == 2\nassert bsearch([2,2,2], 2) == 0\nassert bsearch([1,3,5], 0) == -1\n"
     "assert bsearch([1,3,5], 9) == -1\n"),
    ("flatten_json", "Write `flatten_json(obj, sep='.')`: flattens nested dicts and lists into one dict; list items "
     "use their index as the key part. Empty dicts and empty lists are kept as values.",
     "assert flatten_json({'a': {'b': 1, 'c': [10, {'d': 2}]}, 'e': 3}) == {'a.b': 1, 'a.c.0': 10, 'a.c.1.d': 2, 'e': 3}\n"
     "assert flatten_json({'a': {}, 'b': []}) == {'a': {}, 'b': []}\n"
     "assert flatten_json({'x': {'y': {'z': None}}}, sep='/') == {'x/y/z': None}\n"),
    ("lcs", "Write `lcs(a, b)`: returns one longest common subsequence of two strings (as a string).",
     "def ok(a, b, n):\n    r = lcs(a, b)\n    def sub(s, t):\n        it = iter(t)\n        return all(ch in it for ch in s)\n"
     "    assert len(r) == n and sub(r, a) and sub(r, b), r\n"
     "ok('ABCBDAB', 'BDCABA', 4); ok('', 'abc', 0); ok('abc', 'abc', 3); ok('AGGTAB', 'GXTXAYB', 4)\n"),
    ("rate_limiter", "Write a class `RateLimiter(max_calls, period)` with `allow(now)`: returns True and records the "
     "call if fewer than max_calls were allowed in the half-open window (now - period, now], else False. `now` is a "
     "float in seconds and never decreases.",
     "r = RateLimiter(2, 10)\nassert r.allow(0) and r.allow(1) and not r.allow(2)\nassert not r.allow(9.9)\n"
     "assert r.allow(10)\nassert not r.allow(10.5)\nassert r.allow(11) and not r.allow(11)\n"),
    ("csv_parse", "Write `parse_csv_line(line)`: splits one CSV line into fields. Fields may be quoted with double "
     "quotes; quoted fields may contain commas and doubled quotes (\"\") meaning one quote. No csv module.",
     "assert parse_csv_line('a,b,c') == ['a','b','c']\nassert parse_csv_line('a,\"b,c\",d') == ['a','b,c','d']\n"
     "assert parse_csv_line('\"he said \"\"hi\"\"\",x') == ['he said \"hi\"','x']\n"
     "assert parse_csv_line('a,,') == ['a','','']\nassert parse_csv_line('') == ['']\n"),
]


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(code, tests):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code + "\n\n" + tests + "\nprint('PASS')\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=20)
        return "PASS" in p.stdout, (p.stderr.strip().splitlines() or [""])[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def coding():
    for effort in ("none", "medium"):
        for name, task, tests in CODING:
            r = chat([{"role": "user", "content": task + "\n\nPython 3, standard library only. Reply with one "
                       "```python code block containing the complete solution."}], effort, 6000)
            ok, err = (False, r["error"]) if "error" in r else run_tests(extract_code(r["content"]), tests)
            OUT["coding"].append({"task": name, "effort": effort, "pass": ok, "err": err,
                                  **{k: r.get(k) for k in ("wall", "out_tokens", "tok_s", "finish")}})
            print(f"coding {effort:6} {name:16} {'PASS' if ok else 'FAIL'} {r.get('wall')} s {r.get('out_tokens')} tok  {err}", flush=True)


# ------------------------------------------------------------------------------------------------ reasoning
PUZZLES = [
    ("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?", "0.05"),
    ("Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?", "1"),
    ("If 5 machines take 5 minutes to make 5 widgets, how many minutes do 100 machines take to make 100 widgets?", "5"),
    ("Today is Wednesday. What day of the week is it 100 days from now? Answer with the English day name.", "friday"),
    ("Скільки разів літера 'р' зустрічається у слові 'прапорщик'? Відповідь числом.", "2"),
    ("Compute 487 * 263. Answer with the number.", "128081"),
    ("A farmer has chickens and rabbits: 35 heads and 94 legs. How many rabbits?", "12"),
    ("I have 3 apples. Yesterday I ate 2 apples. How many apples do I have now?", "3"),
    ("Three boxes: one has two gold coins, one two silver, one a gold and a silver. You pick a box at random and "
     "draw a gold coin. What is the probability the other coin in that box is gold? Answer as a fraction a/b.", "2/3"),
    ("A snail climbs a 10 m wall: up 3 m each day, slides down 2 m each night. On which day does it reach the top? "
     "Answer with the number.", "8"),
    ("Alice is taller than Bob. Bob is taller than Carol. Dave is shorter than Carol. Eve is taller than Alice. "
     "Who is the second tallest? Answer with the name.", "alice"),
    ("What is the smallest positive integer divisible by each of 1 through 10? Answer with the number.", "2520"),
]


def reasoning():
    for effort in ("none", "low", "high"):
        for q, want in PUZZLES:
            r = chat([{"role": "user", "content": q + "\nEnd your reply with a line 'ANSWER: <answer>'."}], effort, 8000)
            m = re.findall(r"ANSWER:\s*(.+)", r.get("content", ""))
            got = m[-1].strip().strip(".*` ").lower().replace(",", "").replace(" ", "").lstrip("$") if m else ""
            ok = bool(re.match(re.escape(want) + r"(?![0-9/.])", got))
            OUT["reasoning"].append({"q": q[:50], "effort": effort, "pass": ok, "got": got[:30],
                                     **{k: r.get(k) for k in ("wall", "out_tokens", "tok_s")}})
            print(f"reason {effort:5} {'PASS' if ok else 'FAIL'} {r.get('wall')} s {r.get('out_tokens')} tok  {q[:40]!r} -> {got[:20]!r}", flush=True)


# ------------------------------------------------------------------------------------------------ tools
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                    "max_results": {"type": "integer"}}, "required": ["query"]}}},
]
TOOL_CASES = [
    ("Open src/main.py and tell me what the entry function is called.", "read_file", "path", "main.py"),
    ("How many Python files are in this repository? Use the shell.", "run_command", "command", ""),
    ("What is the latest stable version of Rust? Look it up.", "web_search", "query", "rust"),
    ("Знайди в інтернеті, яка зараз остання версія Python.", "web_search", "query", "python"),
    ("What is 2 + 2?", None, None, None),
]


def tools():
    for q, want, arg, needle in TOOL_CASES:
        r = chat([{"role": "user", "content": q}], "none", 1500, tools=TOOLS)
        calls = r.get("tool_calls") or []
        name = calls[0]["function"]["name"] if calls else None
        try:
            args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
            valid = True
        except ValueError:
            args, valid = {}, False
        ok = name == want and valid and (want is None or (arg in args and needle.lower() in str(args[arg]).lower()))
        OUT["tools"].append({"q": q[:50], "want": want, "got": name, "args": args, "pass": ok, "wall": r.get("wall")})
        print(f"tools {'PASS' if ok else 'FAIL'} want {want} got {name} {args}", flush=True)
    # a two-step round trip: the call, its result, then the answer
    msgs = [{"role": "user", "content": "Read config.toml and tell me which port the server listens on."}]
    r = chat(msgs, "none", 1500, tools=TOOLS)
    calls = r.get("tool_calls") or []
    ok = False
    if calls:
        msgs += [{"role": "assistant", "content": r["content"], "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"),
                  "content": "[server]\nhost = \"0.0.0.0\"\nport = 9137\nworkers = 4\n"}]
        r2 = chat(msgs, "none", 1500, tools=TOOLS)
        ok = "9137" in r2.get("content", "")
    OUT["tools"].append({"q": "round trip (read_file -> answer)", "pass": ok})
    print(f"tools round trip {'PASS' if ok else 'FAIL'}", flush=True)


# ------------------------------------------------------------------------------------------------ JSON
def json_out():
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
              "languages": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "year", "languages"],
              "additionalProperties": False}
    text = ("Guido van Rossum started Python in 1989 and released it in 1991. He also worked with C and ABC. "
            "Extract: the person's name, the release year, and the languages mentioned.")
    for label, extra in (("json_schema", {"response_format": {"type": "json_schema", "json_schema":
                                          {"name": "person", "schema": schema, "strict": True}}}),
                         ("prompt only", {})):
        r = chat([{"role": "user", "content": text + ("" if extra else " Reply with JSON only: "
                                                       "{\"name\": str, \"year\": int, \"languages\": [str]}")}],
                 "none", 600, **extra)
        try:
            d = json.loads(re.sub(r"^```(?:json)?|```$", "", r.get("content", "").strip(), flags=re.M).strip())
            ok = d.get("year") == 1991 and "rossum" in d.get("name", "").lower() and len(d.get("languages", [])) == 3
        except ValueError:
            d, ok = r.get("content", r.get("error", ""))[:200], False
        OUT["json"].append({"mode": label, "pass": ok, "got": d})
        print(f"json {label} {'PASS' if ok else 'FAIL'} {d}", flush=True)


# ------------------------------------------------------------------------------------------------ follow-up
def followup():
    src = (HERE.parent / "serve" / "mcp.py").read_text(encoding="utf-8")[:30000]
    msgs = [{"role": "user", "content": "Here is a Python file:\n\n" + src + "\n\nName the main class in one line."}]
    r1 = chat(msgs, "none", 200)
    msgs += [{"role": "assistant", "content": r1.get("content", "")},
             {"role": "user", "content": "Which transports does it implement? One line."}]
    r2 = chat(msgs, "none", 200)
    for label, r in (("first turn", r1), ("follow-up", r2)):
        OUT["followup"].append({"turn": label, **{k: r.get(k) for k in ("prompt_n", "cache_n", "prompt_s", "wall",
                                                                         "tok_s", "content")}})
        print(f"followup {label}: read {r.get('prompt_n')} tok in {r.get('prompt_s')} s, reused {r.get('cache_n')}, "
              f"wall {r.get('wall')} s | {r.get('content', '')[:120]!r}", flush=True)


# ------------------------------------------------------------------------------------------------ Ukrainian
def ukrainian():
    qs = ["Поясни трьома реченнями, чим процес відрізняється від потоку в операційній системі.",
          "Переклади українською: 'The quick brown fox jumps over the lazy dog, but the dog had already filed a "
          "bug report about it.'",
          "Напиши commit message українською для зміни: додано перевірку заголовка Host у HTTP-сервері проти DNS "
          "rebinding. Один рядок заголовка і два речення опису."]
    for q in qs:
        r = chat([{"role": "user", "content": q}], "none", 500)
        OUT["ukrainian"].append({"q": q, "a": r.get("content"), "tok_s": r.get("tok_s"), "wall": r.get("wall")})
        print(f"uk {r.get('tok_s')} tok/s | {r.get('content', '')[:300]!r}", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:] or ["tools", "json", "followup", "ukrainian", "coding", "reasoning"]
    for name in want:
        {"coding": coding, "reasoning": reasoning, "tools": tools, "json": json_out, "followup": followup,
         "ukrainian": ukrainian}[name]()
        (HERE / "results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", flush=True)
