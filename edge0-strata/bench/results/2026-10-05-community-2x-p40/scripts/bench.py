#!/usr/bin/env python3
"""bench.py - engine-agnostic benchmark: speed + accuracy + completeness + reliability.

One client for Strata / llama-server (OpenAI /v1/chat/completions, streaming) and Ollama (/api/chat, streaming).
Everything is graded against ground truth computed or executed here - never model self-report:
  reasoning   40 programmatic questions (answers computed in Python, fixed seed)
  code        12 functions graded by executing hidden tests in a subprocess
  tools       8 tool-call cases (name + args exact, plus restraint cases that must NOT call a tool)
  complete    6 completeness tasks (exact lists / required sections / valid JSON) + finish_reason
  longctx     needle recall at several prompt sizes (3 facts at 20/50/80 % depth)
  speed       fixed-size prompts: decode tok/s, prefill tok/s, TTFT (client-side AND server-reported)
Every record keeps the raw output, finish_reason, token counts and the effective request, so a score can be re-derived.

--selftest runs the graders against an oracle (must be 100 %) and a garbage engine (must be 0 %) - run it before trusting numbers.
Stdlib only (runs inside a no-network namespace as an unprivileged user).
"""
import argparse, http.client, json, os, random, re, subprocess, sys, tempfile, time, datetime, urllib.parse

SEED = 1234

# --------------------------------------------------------------------------- engines
class Result(dict):
    pass

def _post_stream(url, payload, timeout):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    conn.request("POST", u.path, body=json.dumps(payload), headers={"Content-Type": "application/json", "Host": "127.0.0.1:%d" % u.port})
    resp = conn.getresponse()
    if resp.status != 200:
        body = resp.read(2000).decode("utf-8", "replace")
        conn.close()
        raise RuntimeError("HTTP %d: %s" % (resp.status, body))
    return conn, resp

def call_openai(base, model, messages, max_tokens, think, tools, timeout):
    payload = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True},
               "temperature": 0, "top_k": 1, "top_p": 1, "seed": 1, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": bool(think)}}
    if not think:
        payload["reasoning_effort"] = "none"   # Strata honours this; llama-server ignores unknown fields
    if tools:
        payload["tools"] = tools
    t0 = time.time(); first = None
    text, reasoning, finish, usage, timings = [], [], None, None, None
    tcalls = {}
    conn, resp = _post_stream(base + "/v1/chat/completions", payload, timeout)
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ch = json.loads(data)
            if ch.get("usage"):
                usage = ch["usage"]
            if ch.get("timings"):
                timings = ch["timings"]
            for c in ch.get("choices", []):
                d = c.get("delta", {})
                if d.get("content"):
                    first = first or time.time(); text.append(d["content"])
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    first = first or time.time(); reasoning.append(rc)
                for tc in d.get("tool_calls") or []:
                    first = first or time.time()
                    slot = tcalls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function", {})
                    slot["name"] += f.get("name") or ""
                    slot["arguments"] += f.get("arguments") or ""
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    finally:
        conn.close()
    t1 = time.time()
    calls = []
    for k in sorted(tcalls):
        try:
            args = json.loads(tcalls[k]["arguments"] or "{}")
        except Exception:
            args = {"_unparsed": tcalls[k]["arguments"]}
        calls.append({"name": tcalls[k]["name"], "arguments": args})
    ct = (usage or {}).get("completion_tokens")
    return Result(text="".join(text), reasoning="".join(reasoning), finish_reason=finish, tool_calls=calls,
                  prompt_tokens=(usage or {}).get("prompt_tokens"), completion_tokens=ct,
                  t_total=t1 - t0, t_first=(first - t0) if first else None, server_timings=timings, error=None)

def call_ollama(base, model, messages, max_tokens, think, tools, timeout, num_ctx):
    payload = {"model": model, "messages": messages, "stream": True, "think": bool(think),
               "options": {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": 0, "top_k": 1, "top_p": 1,
                           "seed": 1, "repeat_penalty": 1.0}}
    if tools:
        payload["tools"] = tools
    t0 = time.time(); first = None
    text, reasoning, final, calls = [], [], {}, []
    conn, resp = _post_stream(base + "/api/chat", payload, timeout)
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            ch = json.loads(line)
            m = ch.get("message", {})
            if m.get("content"):
                first = first or time.time(); text.append(m["content"])
            if m.get("thinking"):
                first = first or time.time(); reasoning.append(m["thinking"])
            for tc in m.get("tool_calls") or []:
                first = first or time.time()
                f = tc.get("function", {})
                calls.append({"name": f.get("name", ""), "arguments": f.get("arguments", {})})
            if ch.get("done"):
                final = ch
    finally:
        conn.close()
    t1 = time.time()
    ns = lambda k: (final.get(k) or 0) / 1e9
    st = {"prompt_n": final.get("prompt_eval_count"), "prompt_ms": ns("prompt_eval_duration") * 1000,
          "predicted_n": final.get("eval_count"), "predicted_ms": ns("eval_duration") * 1000,
          "load_ms": ns("load_duration") * 1000}
    if st["prompt_ms"]:
        st["prompt_per_second"] = (st["prompt_n"] or 0) / (st["prompt_ms"] / 1000)
    if st["predicted_ms"]:
        st["predicted_per_second"] = (st["predicted_n"] or 0) / (st["predicted_ms"] / 1000)
    return Result(text="".join(text), reasoning="".join(reasoning), finish_reason=final.get("done_reason"),
                  tool_calls=calls, prompt_tokens=final.get("prompt_eval_count"), completion_tokens=final.get("eval_count"),
                  t_total=t1 - t0, t_first=(first - t0) if first else None, server_timings=st, error=None)

class Oracle:
    """Engine stand-in for --selftest: returns the task's known-correct answer (or garbage)."""
    def __init__(self, good): self.good = good
    def __call__(self, task):
        if task.get("tools"):
            calls = task["oracle_calls"] if self.good else [{"name": "bogus", "arguments": {}}]
            return Result(text="", reasoning="", finish_reason="tool_calls" if calls else "stop", tool_calls=calls if self.good or True else [],
                          prompt_tokens=1, completion_tokens=1, t_total=0.01, t_first=0.01, server_timings=None, error=None)
        txt = task["oracle"] if self.good else "I do not know."
        return Result(text=txt, reasoning="", finish_reason="stop", tool_calls=[], prompt_tokens=1, completion_tokens=1,
                      t_total=0.01, t_first=0.01, server_timings=None, error=None)

# --------------------------------------------------------------------------- tasks: reasoning (programmatic)
def norm(s):
    s = s.strip().strip("`*\"'").rstrip(".").strip()
    s = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s
    return s.lower()

def last_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    return m[-1].strip() if m else None

def grade_answer(task, res):
    a = last_answer(res["text"])
    return a is not None and norm(a) == norm(task["expected"]), {"got": a, "expected": task["expected"]}

FMT = "\n\nThink briefly if needed, then give the final result on the last line formatted exactly as 'ANSWER: <answer>'."

def make_reasoning():
    import datetime as dt
    r = random.Random(SEED); out = []
    def add(q, exp):
        out.append({"id": "reason-%02d" % (len(out) + 1), "category": "reasoning", "messages": [{"role": "user", "content": q + FMT}],
                    "max_tokens": 1024, "expected": str(exp), "oracle": "ANSWER: %s" % exp, "grader": "answer"})
    for _ in range(8):
        a, b = r.randint(120, 9800), r.randint(12, 990); add("What is %d * %d?" % (a, b), a * b)
    for _ in range(5):
        a, b, c, d = r.randint(100, 999), r.randint(100, 999), r.randint(3, 40), r.randint(3, 40); add("Compute %d + %d - %d * %d." % (a, b, c, d), a + b - c * d)
    words = ["strawberry", "mississippi", "bookkeeper", "committee", "rhythm", "banana bandana", "assassin", "occurrence", "parallel lines", "successful"]
    for w in r.sample(words, 5):
        ch = r.choice(sorted(set(w.replace(" ", ""))))
        add("How many times does the letter '%s' appear in the text \"%s\"? (case-insensitive)" % (ch, w), w.lower().count(ch))
    for _ in range(5):
        d0 = dt.date(r.randint(2019, 2031), r.randint(1, 12), r.randint(1, 28)); n = r.randint(40, 900)
        add("What day of the week is it %d days after %s? Answer with the weekday name only." % (n, d0.isoformat()), (d0 + dt.timedelta(days=n)).strftime("%A"))
    for _ in range(2):
        n = r.randint(200, 4000); add("Convert the decimal number %d to binary (digits only, no prefix)." % n, bin(n)[2:])
    for _ in range(2):
        n = r.randint(3000, 60000); add("Convert the decimal number %d to hexadecimal (uppercase, no prefix)." % n, "%X" % n)
    for _ in range(4):
        xs = r.sample(range(10, 99), 8); add("Here is a list: %s. What is the third largest number in it?" % xs, sorted(xs)[-3])
    for _ in range(3):
        a, b, m = r.randint(3, 12), r.randint(5, 40), r.randint(7, 97); add("What is %d^%d mod %d?" % (a, b, m), pow(a, b, m))
    for _ in range(3):
        v, t = r.randint(30, 90), r.randint(2, 9); add("A train travels at %d km/h for %d hours and then 2 more hours at %d km/h. What total distance in km did it cover?" % (v, t, v + 10), v * t + 2 * (v + 10))
    for w in r.sample(["calculator", "photograph", "microscope"], 3):
        add("Write the word '%s' backwards (letters only)." % w, w[::-1])
    return out

# --------------------------------------------------------------------------- tasks: code (executed hidden tests)
CODE_TASKS = [
 ("is_balanced", "is_balanced(s: str) -> bool: True if every (), [], {} in s is correctly nested and closed; ignore other characters.",
  [(["([]{})"], True), (["([)]"], False), (["a(b)c]"], False), ([""], True), (["{[()()]}"], True), (["(("], False)]),
 ("roman_to_int", "roman_to_int(s: str) -> int: convert a valid Roman numeral (1..3999) to an integer.",
  [(["III"], 3), (["LVIII"], 58), (["MCMXCIV"], 1994), (["IV"], 4), (["MMMCMXCIX"], 3999)]),
 ("merge_intervals", "merge_intervals(iv: list) -> list: iv is a list of [start, end] pairs; return merged overlapping/touching intervals sorted by start, as lists.",
  [([[[1, 3], [2, 6], [8, 10], [15, 18]]], [[1, 6], [8, 10], [15, 18]]), ([[[1, 4], [4, 5]]], [[1, 5]]), ([[]], []), ([[[5, 7], [1, 2], [2, 3]]], [[1, 3], [5, 7]])]),
 ("lcs_length", "lcs_length(a: str, b: str) -> int: length of the longest common subsequence.",
  [(["abcde", "ace"], 3), (["abc", "def"], 0), (["", "abc"], 0), (["AGGTAB", "GXTXAYB"], 4)]),
 ("flatten", "flatten(x) -> list: flatten arbitrarily nested lists of ints into one flat list, preserving order.",
  [([[1, [2, [3, [4]], 5]]], [1, 2, 3, 4, 5]), ([[]], []), ([[[], [[]], 7]], [7])]),
 ("top_words", "top_words(text: str, k: int) -> list: the k most frequent words (lowercase, words are runs of letters a-z/A-Z) as a list of [word, count], sorted by count descending then word ascending.",
  [(["the cat and the hat and the bat", 2], [["the", 3], ["and", 2]]), (["A a b B c", 2], [["a", 2], ["b", 2]]), (["", 3], [])]),
 ("rotate_matrix", "rotate_matrix(m: list) -> list: rotate a square matrix (list of lists) 90 degrees clockwise and return a new matrix.",
  [([[[1, 2], [3, 4]]], [[3, 1], [4, 2]]), ([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[7, 4, 1], [8, 5, 2], [9, 6, 3]]), ([[[5]]], [[5]])]),
 ("valid_ipv4", "valid_ipv4(s: str) -> bool: True iff s is a valid dotted-quad IPv4 address: exactly 4 decimal parts 0-255, no leading zeros (except '0' itself), digits only.",
  [(["192.168.1.1"], True), (["256.1.1.1"], False), (["01.2.3.4"], False), (["1.2.3"], False), (["1.2.3.4.5"], False), (["0.0.0.0"], True), (["a.b.c.d"], False), (["1..2.3"], False)]),
 ("prime_factors", "prime_factors(n: int) -> list: prime factorization of n >= 2 as a sorted list with repetition.",
  [([12], [2, 2, 3]), ([97], [97]), ([360], [2, 2, 2, 3, 3, 5]), ([2 ** 10], [2] * 10), ([1001], [7, 11, 13])]),
 ("longest_palindrome", "longest_palindrome(s: str) -> str: the longest palindromic substring; on ties return the one that starts earliest; empty string for empty input.",
  [(["babad"], "bab"), (["cbbd"], "bb"), (["a"], "a"), ([""], ""), (["forgeeksskeegfor"], "geeksskeeg")]),
 ("eval_rpn", "eval_rpn(tokens: list) -> int: evaluate a Reverse Polish Notation expression given as a list of strings with + - * /; division truncates toward zero.",
  [([["2", "1", "+", "3", "*"]], 9), ([["4", "13", "5", "/", "+"]], 6), ([["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]], 22), ([["7", "-2", "/"]], -3)]),
 ("parse_duration", "parse_duration(s: str) -> int: convert strings like '1h30m15s', '45s', '2h', '10m5s' (any subset of h, m, s in that order) to total seconds; return 0 for an empty string.",
  [(["1h30m15s"], 5415), (["45s"], 45), (["2h"], 7200), (["10m5s"], 605), ([""], 0)]),
]

def make_code():
    out = []
    for name, spec, tests in CODE_TASKS:
        out.append({"id": "code-" + name, "category": "code", "max_tokens": 1500, "grader": "code", "fn": name, "tests": tests,
                    "messages": [{"role": "user", "content": "Write a Python 3 function with this signature and behaviour:\n\n%s\n\nReturn only one ```python code block containing the function (and any imports/helpers it needs). No example usage, no prints." % spec}],
                    "oracle": None})
    return out

REF_IMPL = {  # reference solutions used ONLY by --selftest to prove the hidden tests are correct and passable
 "is_balanced": "def is_balanced(s):\n    st=[];p={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in p:\n            if not st or st.pop()!=p[c]: return False\n    return not st",
 "roman_to_int": "def roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000};t=0\n    for i,c in enumerate(s):\n        t+= -v[c] if i+1<len(s) and v[c]<v[s[i+1]] else v[c]\n    return t",
 "merge_intervals": "def merge_intervals(iv):\n    r=[]\n    for a,b in sorted(iv):\n        if r and a<=r[-1][1]: r[-1][1]=max(r[-1][1],b)\n        else: r.append([a,b])\n    return r",
 "lcs_length": "def lcs_length(a,b):\n    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]\n    for i in range(len(a)):\n        for j in range(len(b)):\n            d[i+1][j+1]=d[i][j]+1 if a[i]==b[j] else max(d[i][j+1],d[i+1][j])\n    return d[-1][-1]",
 "flatten": "def flatten(x):\n    o=[]\n    for e in x:\n        o+= flatten(e) if isinstance(e,list) else [e]\n    return o",
 "top_words": "import re,collections\ndef top_words(text,k):\n    c=collections.Counter(w.lower() for w in re.findall(r'[A-Za-z]+',text))\n    return [[w,n] for w,n in sorted(c.items(),key=lambda t:(-t[1],t[0]))[:k]]",
 "rotate_matrix": "def rotate_matrix(m):\n    return [list(r) for r in zip(*m[::-1])]",
 "valid_ipv4": "def valid_ipv4(s):\n    p=s.split('.')\n    return len(p)==4 and all(x.isdigit() and x.isascii() and (x=='0' or not x.startswith('0')) and int(x)<=255 for x in p)",
 "prime_factors": "def prime_factors(n):\n    f=[];d=2\n    while d*d<=n:\n        while n%d==0: f.append(d);n//=d\n        d+=1\n    if n>1: f.append(n)\n    return f",
 "longest_palindrome": "def longest_palindrome(s):\n    b=''\n    for i in range(len(s)):\n        for j in range(i,len(s)):\n            t=s[i:j+1]\n            if t==t[::-1] and len(t)>len(b): b=t\n    return b",
 "eval_rpn": "def eval_rpn(tokens):\n    st=[]\n    for t in tokens:\n        if t in '+-*/' and len(t)==1:\n            b=st.pop();a=st.pop()\n            st.append(a+b if t=='+' else a-b if t=='-' else a*b if t=='*' else int(a/b))\n        else: st.append(int(t))\n    return st[0]",
 "parse_duration": "import re\ndef parse_duration(s):\n    m=re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?',s)\n    h,mi,se=(int(x) if x else 0 for x in m.groups())\n    return h*3600+mi*60+se",
}

def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None

def grade_code(task, res):
    code = extract_code(res["text"])
    if not code:
        return False, {"error": "no code block"}
    harness = code + "\n\nimport json,sys\n_T=json.loads(sys.stdin.read())\n_ok=0;_fail=[]\nfor a,e in _T:\n    try:\n        g=%s(*a)\n    except Exception as ex:\n        g='EXC '+type(ex).__name__\n    if g==e: _ok+=1\n    else: _fail.append([a,e,g])\nprint(json.dumps({'ok':_ok,'n':len(_T),'fail':_fail[:3]}))\n" % task["fn"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness); path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], input=json.dumps(task["tests"]), capture_output=True, text=True, timeout=15)
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"error": (p.stderr or "no output")[-300:]}
    except subprocess.TimeoutExpired:
        out = {"error": "timeout"}
    except Exception as ex:
        out = {"error": repr(ex)}
    finally:
        os.unlink(path)
    return out.get("ok") == out.get("n") and "error" not in out, out

# --------------------------------------------------------------------------- tasks: tool calling
TOOLS = [
 {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city", "unit"]}}},
 {"type": "function", "function": {"name": "set_timer", "description": "Set a countdown timer.", "parameters": {"type": "object", "properties": {"minutes": {"type": "integer"}, "label": {"type": "string"}}, "required": ["minutes", "label"]}}},
 {"type": "function", "function": {"name": "search_docs", "description": "Search the internal documentation.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query", "limit"]}}},
 {"type": "function", "function": {"name": "send_email", "description": "Send an email.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
]
TOOL_CASES = [
 ("What's the weather in Paris right now? I want Celsius.", ("get_weather", {"city": "paris", "unit": "c"})),
 ("Tell me the temperature in Denver in Fahrenheit.", ("get_weather", {"city": "denver", "unit": "f"})),
 ("Set a timer for 25 minutes labelled pasta.", ("set_timer", {"minutes": 25, "label": "pasta"})),
 ("Look up 'rotate api keys' in the docs, give me the top 5 results.", ("search_docs", {"query": "rotate api keys", "limit": 5})),
 ("Email bob@example.com with subject 'Lunch' and body 'Noon works for me.'", ("send_email", {"to": "bob@example.com", "subject": "lunch", "body": "noon works for me."})),
 ("Remind me in 90 minutes to call mom, use the label 'call mom'.", ("set_timer", {"minutes": 90, "label": "call mom"})),
 ("What is the capital of France?", None),
 ("Explain in one sentence what a hash table is.", None),
]

def make_tools():
    out = []
    for i, (q, exp) in enumerate(TOOL_CASES, 1):
        calls = [{"name": exp[0], "arguments": exp[1]}] if exp else []
        out.append({"id": "tool-%02d" % i, "category": "tools", "max_tokens": 512, "grader": "tool", "expected_call": exp, "tools": TOOLS,
                    "oracle_calls": calls, "oracle": None, "messages": [{"role": "system", "content": "You are a helpful assistant. Use a tool only when one is needed."}, {"role": "user", "content": q}]})
    return out

def _eqv(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b

def grade_tool(task, res):
    exp, calls = task["expected_call"], res["tool_calls"]
    if exp is None:
        return (len(calls) == 0), {"calls": calls}
    if not calls:
        return False, {"calls": calls, "note": "no tool call"}
    c = calls[0]
    ok = c["name"] == exp[0] and set(c["arguments"].keys()) == set(exp[1].keys()) and all(_eqv(c["arguments"].get(k), v) for k, v in exp[1].items())
    return ok, {"calls": calls}

# --------------------------------------------------------------------------- tasks: completeness
PRIMES = [p for p in range(2, 600) if all(p % d for d in range(2, int(p ** .5) + 1))][:100]
ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg"]
SECTIONS = ["Overview", "Installation", "Usage", "Configuration", "Troubleshooting", "License"]

def make_complete():
    def t(id, q, mt, kind, oracle, **kw):
        d = {"id": "complete-" + id, "category": "complete", "max_tokens": mt, "grader": "complete", "kind": kind, "oracle": oracle,
             "messages": [{"role": "user", "content": q}]}
        d.update(kw); return d
    js = json.dumps([{"id": i, "name": "item%d" % i, "squares": i * i} for i in range(1, 21)])
    return [
     t("primes100", "List the first 100 prime numbers, comma-separated, in ascending order. Output only the numbers.", 2048, "ints", ", ".join(map(str, PRIMES)), expected_ints=PRIMES),
     t("count150", "Write the integers from 1 to 150 separated by single spaces. Output only the numbers.", 2048, "ints", " ".join(map(str, range(1, 151))), expected_ints=list(range(1, 151))),
     t("alpha_rev", "Write the English alphabet in reverse order, uppercase letters separated by commas. Output only the letters.", 512, "letters", ", ".join(chr(c) for c in range(90, 64, -1)), expected_letters=[chr(c) for c in range(90, 64, -1)]),
     t("sections", "Write a project README in Markdown with exactly these H2 sections, in this order: %s. Every section must contain at least 3 full sentences." % ", ".join(SECTIONS), 2048, "sections",
       "\n".join("## %s\nThis is sentence one. This is sentence two. This is sentence three.\n" % s for s in SECTIONS)),
     t("json20", "Output a JSON array of exactly 20 objects. Object i (i from 1 to 20) has keys \"id\" (= i), \"name\" (= \"item<i>\") and \"squares\" (= i*i). Output only the JSON.", 2048, "json", "```json\n%s\n```" % js),
     t("elements", "Make a Markdown table of the first 12 chemical elements by atomic number with columns: Number, Symbol, Name.", 1024, "elements",
       "| Number | Symbol | Name |\n|---|---|---|\n" + "\n".join("| %d | %s | x |" % (i + 1, s) for i, s in enumerate(ELEMENTS))),
    ]

def grade_complete(task, res):
    txt, k = res["text"], task["kind"]
    fin = res["finish_reason"] in ("stop", "end_turn", None) and res["finish_reason"] != "length"
    info = {"finish_reason": res["finish_reason"]}
    if k == "ints":
        got = [int(x) for x in re.findall(r"\d+", txt)]
        ok = got == task["expected_ints"]; info.update(got_n=len(got), want_n=len(task["expected_ints"]))
    elif k == "letters":
        got = re.findall(r"\b([A-Z])\b", txt); ok = got == task["expected_letters"]; info.update(got_n=len(got))
    elif k == "sections":
        heads = re.findall(r"^##\s+(.+?)\s*$", txt, re.M); ok = heads == SECTIONS
        if ok:
            parts = re.split(r"^##\s+.+$", txt, flags=re.M)[1:]
            ok = len(parts) == 6 and all(len(re.findall(r"[.!?](\s|$)", p)) >= 3 for p in parts)
        info["headings"] = heads
    elif k == "json":
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S); body = m.group(1) if m else txt
        try:
            arr = json.loads(body); ok = isinstance(arr, list) and len(arr) == 20 and all(o.get("id") == i + 1 and o.get("name") == "item%d" % (i + 1) and o.get("squares") == (i + 1) ** 2 for i, o in enumerate(arr))
        except Exception as ex:
            ok = False; info["error"] = repr(ex)[:120]
    elif k == "elements":
        syms = [c.strip() for line in txt.splitlines() if line.strip().startswith("|") for c in [line.split("|")[2]] if len(line.split("|")) > 3]
        syms = [s for s in syms if s in ELEMENTS or re.fullmatch(r"[A-Z][a-z]?", s)]
        ok = syms[:12] == ELEMENTS and len(syms) == 12; info["symbols"] = syms
    else:
        ok = False
    info["finished"] = fin
    return bool(ok and fin), info

# --------------------------------------------------------------------------- tasks: long-context recall + speed prompts
NOUNS = ["harbor", "lantern", "orchard", "bridge", "archive", "glacier", "market", "workshop", "canal", "observatory", "foundry", "meadow"]
ADJ = ["quiet", "old", "crowded", "northern", "abandoned", "busy", "narrow", "famous", "remote", "sunlit"]
VERB = ["recorded", "measured", "reported", "counted", "inspected", "catalogued"]
CITY = ["Aldermoor", "Brindle", "Casterly", "Dunmere", "Eastwick", "Fairholt", "Greyvale", "Hollin"]

def filler(n_chars, rnd):
    parts, size = [], 0
    while size < n_chars:
        s = "The %s %s of %s %s %d %s during week %d of the survey. " % (rnd.choice(ADJ), rnd.choice(NOUNS), rnd.choice(CITY), rnd.choice(VERB), rnd.randint(11, 989), rnd.choice(NOUNS) + "s", rnd.randint(1, 52))
        parts.append(s); size += len(s)
    return parts

def make_longctx(sizes_k):
    out = []
    for k in sizes_k:
        rnd = random.Random(SEED + k)
        parts = filler(int(k * 1024 * 3.6), rnd)  # ~3.6 chars/token for this filler -> approx k thousand tokens
        names = rnd.sample(["Orion", "Vesper", "Calder", "Mistral", "Juniper", "Tamsin", "Quillon", "Bracken"], 3)
        codes = ["%06d" % rnd.randint(100000, 999999) for _ in range(3)]
        for frac, nm, cd in zip((0.2, 0.5, 0.8), names, codes):
            parts.insert(int(len(parts) * frac), "NOTE: The access code for vault %s is %s. " % (nm, cd))
        doc = "".join(parts)
        q = "Below is a long survey log. Read it, then answer the question at the end.\n\n%s\n\nQuestion: What are the access codes for vaults %s, %s and %s? Reply exactly in the form '%s=<code>, %s=<code>, %s=<code>'." % (doc, names[0], names[1], names[2], names[0], names[1], names[2])
        out.append({"id": "longctx-%dk" % k, "category": "longctx", "max_tokens": 200, "grader": "longctx", "names": names, "codes": codes,
                    "messages": [{"role": "user", "content": q}], "oracle": ", ".join("%s=%s" % p for p in zip(names, codes))})
    return out

def grade_longctx(task, res):
    hits = [bool(re.search(r"%s\s*=\s*%s" % (n, c), res["text"])) for n, c in zip(task["names"], task["codes"])]
    return all(hits), {"hits": hits, "fraction": sum(hits) / 3}

def make_speed(prompt_ks, gen_tokens):
    out = [{"id": "speed-short", "category": "speed", "max_tokens": gen_tokens, "grader": "none", "oracle": "x",
            "messages": [{"role": "user", "content": "Write a detailed explanation of how a refrigerator works, in plain prose, at least 400 words."}]}]
    for k in prompt_ks:
        rnd = random.Random(SEED + 100 + k)
        doc = "".join(filler(int(k * 1024 * 3.6), rnd))
        out.append({"id": "speed-prefill-%dk" % k, "category": "speed", "max_tokens": 64, "grader": "none", "oracle": "x",
                    "messages": [{"role": "user", "content": "Summarize this log in two sentences.\n\n" + doc}]})
    return out

GRADERS = {"answer": grade_answer, "code": grade_code, "tool": grade_tool, "complete": grade_complete, "longctx": grade_longctx, "none": lambda t, r: (None, {})}

# --------------------------------------------------------------------------- run
def build_tasks(args):
    cats = set(args.suite.split(","))
    tasks = []
    if "reasoning" in cats: tasks += make_reasoning()
    if "code" in cats: tasks += make_code()
    if "tools" in cats: tasks += make_tools()
    if "complete" in cats: tasks += make_complete()
    if "longctx" in cats: tasks += make_longctx([int(x) for x in args.longctx_k.split(",")])
    if "speed" in cats: tasks += make_speed([int(x) for x in args.speed_k.split(",")], args.gen_tokens)
    return tasks

def selftest(args):
    args.suite = "reasoning,code,tools,complete,longctx"; args.longctx_k = "1"
    tasks = build_tasks(args)
    bad = 0
    for t in tasks:
        if t["category"] == "code":
            oracle = Result(text="```python\n%s\n```" % REF_IMPL[t["fn"]], finish_reason="stop", tool_calls=[])
        elif t["category"] == "tools":
            oracle = Oracle(True)(t)
        else:
            oracle = Oracle(True)(t)
        good, _ = GRADERS[t["grader"]](t, oracle)
        garb = Oracle(False)(t)
        if t["category"] == "tools":   # garbage = a bogus call; restraint cases fail it too
            garb = Result(text="", finish_reason="tool_calls", tool_calls=[{"name": "bogus", "arguments": {}}])
        if t["category"] == "code": garb = Result(text="no code", finish_reason="stop", tool_calls=[])
        gg, _ = GRADERS[t["grader"]](t, garb)
        if not good or gg:
            bad += 1; print("SELFTEST FAIL %s oracle_pass=%s garbage_pass=%s" % (t["id"], good, gg))
    print("selftest: %d tasks, %d problems" % (len(tasks), bad))
    return 1 if bad else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["openai", "ollama"]); ap.add_argument("--base", help="e.g. http://127.0.0.1:8080")
    ap.add_argument("--model", default="m"); ap.add_argument("--arm", help="label for this arm, e.g. strata-iq2xs-p40x1")
    ap.add_argument("--suite", default="speed,reasoning,code,tools,complete,longctx")
    ap.add_argument("--think", choices=["off", "on"], default="off")
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--longctx-k", default="4,12,24"); ap.add_argument("--speed-k", default="4,16"); ap.add_argument("--gen-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=3600); ap.add_argument("--out", default="results/raw")
    ap.add_argument("--only", help="comma list of task id prefixes"); ap.add_argument("--meta", default="{}", help="JSON string recorded in every record")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args))
    tasks = build_tasks(args)
    if args.only:
        pre = tuple(args.only.split(",")); tasks = [t for t in tasks if t["id"].startswith(pre)]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s.%s.jsonl" % (args.arm, datetime.datetime.now().strftime("%Y%m%dT%H%M%S")))
    meta = json.loads(args.meta); think = args.think == "on"
    print("arm=%s engine=%s tasks=%d reps=%d think=%s -> %s" % (args.arm, args.engine, len(tasks), args.reps, args.think, path), flush=True)
    with open(path, "a") as fh:
        for rep in range(args.reps):
            for t in tasks:
                rec = {"arm": args.arm, "rep": rep, "id": t["id"], "category": t["category"], "think": args.think, "ts": datetime.datetime.now().isoformat(timespec="seconds"), "meta": meta}
                try:
                    if args.engine == "openai":
                        res = call_openai(args.base, args.model, t["messages"], t["max_tokens"], think, t.get("tools"), args.timeout)
                    else:
                        res = call_ollama(args.base, args.model, t["messages"], t["max_tokens"], think, t.get("tools"), args.timeout, args.num_ctx)
                    ok, info = GRADERS[t["grader"]](t, res)
                    rec.update(res=res, passed=ok, grade=info, error=None)
                except Exception as ex:
                    rec.update(res=None, passed=False if t["grader"] != "none" else None, grade={}, error="%s: %s" % (type(ex).__name__, ex))
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                r = rec["res"] or {}
                tp = (r.get("server_timings") or {}).get("predicted_per_second")
                print("%-24s pass=%-5s fin=%-10s ptok=%-6s ctok=%-5s t=%6.1fs srv_tps=%s %s" % (t["id"], rec["passed"], r.get("finish_reason"), r.get("prompt_tokens"), r.get("completion_tokens"), r.get("t_total") or 0, "%.1f" % tp if tp else "-", rec["error"] or ""), flush=True)
    print("done:", path)

if __name__ == "__main__":
    main()
