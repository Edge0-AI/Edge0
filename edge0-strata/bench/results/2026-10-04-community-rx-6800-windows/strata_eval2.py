"""bench-local/strata_eval2.py - round 2: harder coding, JavaScript, SQL, regex, a tool-using agent loop on a small
project, fact questions over a ~60K-token file, a long generation.  Writes results2.json next to this file."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strata_eval as E  # noqa: E402

HERE = Path(__file__).parent
OUT = {"hard": [], "js": [], "sql": [], "regex": [], "agent": [], "qa": [], "longgen": []}
ASK = "\n\nPython 3, standard library only. Reply with one ```python code block containing the complete solution."

HARD = [
    ("topo_sort", "Write `topo_sort(edges)`: edges is a list of (a, b) string pairs meaning a must come before b. "
     "Return all nodes in a valid order; among valid orders return the lexicographically smallest one. Raise "
     "ValueError if there is a cycle.",
     "assert topo_sort([('b','c'),('a','c'),('c','d')]) == ['a','b','c','d']\n"
     "assert topo_sort([('x','a'),('b','a')]) == ['b','x','a']\nassert topo_sort([]) == []\n"
     "try:\n    topo_sort([('a','b'),('b','a')]); raise AssertionError('cycle')\nexcept ValueError:\n    pass\n"),
    ("wildcard_match", "Write `wildcard_match(s, p)`: '?' matches exactly one character, '*' matches any sequence "
     "(including empty); the whole string must match.",
     "T = [('aa','a',False),('aa','*',True),('cb','?a',False),('adceb','*a*b',True),('acdcb','a*c?b',False),"
     "('','*',True),('','?',False),('abc','a**c',True),('a'*30+'b','*'+'a*'*10+'c',False)]\n"
     "for s, p, w in T:\n    assert wildcard_match(s, p) is w, (s, p)\n"),
    ("min_window", "Write `min_window(s, t)`: the shortest substring of s containing every character of t (with "
     "multiplicity); '' if none.",
     "assert min_window('ADOBECODEBANC','ABC') == 'BANC'\nassert min_window('a','a') == 'a'\n"
     "assert min_window('a','aa') == ''\nassert min_window('aab','aab') == 'aab'\n"),
    ("json_pointer", "Write `json_pointer(doc, pointer)` implementing RFC 6901 lookup: '' returns the whole document, "
     "'/a/0' descends into dicts and lists, '~1' decodes to '/', '~0' decodes to '~'. Raise KeyError for any path "
     "that does not exist (including a list index out of range).",
     "d = {'foo': ['bar', 'baz'], '': 0, 'a/b': 1, 'm~n': 8, ' ': 7}\n"
     "assert json_pointer(d, '') == d and json_pointer(d, '/foo') == ['bar','baz'] and json_pointer(d, '/foo/0') == 'bar'\n"
     "assert json_pointer(d, '/') == 0 and json_pointer(d, '/a~1b') == 1 and json_pointer(d, '/m~0n') == 8\n"
     "assert json_pointer(d, '/ ') == 7\n"
     "for bad in ['/foo/5', '/nope', '/foo/x']:\n    try:\n        json_pointer(d, bad); raise AssertionError(bad)\n"
     "    except KeyError:\n        pass\n"),
    ("nqueens", "Write `nqueens(n)`: the number of ways to place n non-attacking queens on an n x n board.",
     "assert [nqueens(n) for n in (1, 4, 6, 8)] == [1, 2, 4, 92]\n"),
    ("semver_sort", "Write `semver_sort(versions)`: sorts semantic version strings ascending by the SemVer 2.0.0 "
     "precedence rules, including pre-release identifiers (no build metadata).",
     "v = ['1.0.0','1.0.0-alpha','1.0.0-alpha.1','1.0.0-beta','1.0.0-rc.1','0.9.9','1.0.0-alpha.beta','1.0.0-beta.2',"
     "'1.0.0-beta.11','2.0.0']\n"
     "assert semver_sort(v) == ['0.9.9','1.0.0-alpha','1.0.0-alpha.1','1.0.0-alpha.beta','1.0.0-beta','1.0.0-beta.2',"
     "'1.0.0-beta.11','1.0.0-rc.1','1.0.0','2.0.0'], semver_sort(v)\n"),
    ("edit_distance", "Write `edit_distance(a, b)`: the Levenshtein distance.",
     "assert edit_distance('kitten','sitting') == 3 and edit_distance('','abc') == 3\n"
     "assert edit_distance('flaw','lawn') == 2 and edit_distance('intention','execution') == 5\n"),
    ("gather_limited", "Write `async def gather_limited(coro_fns, limit)`: coro_fns is a list of zero-argument "
     "callables that each return a coroutine. Run them with at most `limit` running at the same time and return their "
     "results in the input order.",
     "import asyncio\nasync def _t():\n    running = 0; peak = 0\n    async def job(i):\n        nonlocal running, peak\n"
     "        running += 1; peak = max(peak, running)\n        await asyncio.sleep(0.01 * (5 - i % 5))\n"
     "        running -= 1\n        return i * i\n"
     "    res = await gather_limited([lambda i=i: job(i) for i in range(12)], 3)\n"
     "    assert res == [i * i for i in range(12)], res\n    assert 2 <= peak <= 3, peak\nasyncio.run(_t())\n"),
    ("trie", "Write a class `Trie` with `insert(word)`, `search(word)` (exact word), `starts_with(prefix)` and "
     "`delete(word)`: delete removes the word and prunes nodes no other word uses; deleting a missing word does "
     "nothing.",
     "t = Trie(); t.insert('apple'); t.insert('app')\n"
     "assert t.search('app') and t.search('apple') and not t.search('appl') and t.starts_with('appl')\n"
     "t.delete('apple'); assert not t.search('apple') and t.search('app') and not t.starts_with('appl')\n"
     "t.delete('zzz'); t.delete('app'); assert not t.starts_with('a')\n"),
    ("bowling", "Write `bowling_score(rolls)`: the total score of a complete ten-pin bowling game given the list of "
     "pins knocked down per roll.",
     "assert bowling_score([10]*12) == 300 and bowling_score([9,1]*10+[9]) == 190 and bowling_score([0]*20) == 0\n"
     "assert bowling_score([3,4]*10) == 70\n"
     "assert bowling_score([10,7,3,9,0,10,0,8,8,2,0,6,10,10,10,8,1]) == 167\n"),
    ("palindrome", "Write `longest_palindrome(s)`: a longest palindromic substring of s.",
     "assert longest_palindrome('babad') in ('bab','aba') and longest_palindrome('cbbd') == 'bb'\n"
     "assert longest_palindrome('') == '' and longest_palindrome('a') == 'a'\n"
     "assert longest_palindrome('forgeeksskeegfor') == 'geeksskeeg'\n"),
    ("normpath", "Write `normalize(path)` for POSIX-style paths without using os.path or pathlib: collapse repeated "
     "slashes, resolve '.' and '..', drop a trailing slash. An absolute path never goes above '/'. A relative path "
     "keeps leading '..' parts it cannot resolve. An empty result is '.' for a relative path.",
     "T = {'/a/./b/../../c/': '/c', '/../': '/', '/home//foo/': '/home/foo', 'a/b/../c': 'a/c', '../a': '../a', "
     "'': '.', 'a/..': '.', 'a/../../b': '../b', '/': '/'}\n"
     "for k, w in T.items():\n    assert normalize(k) == w, (k, normalize(k))\n"),
]


def hard():
    import os
    for effort in os.environ.get("EFFORTS", "none,high").split(","):
        for name, task, tests in HARD:
            r = E.chat([{"role": "user", "content": task + ASK}], effort, 12000)
            ok, err = (False, r["error"]) if "error" in r else E.run_tests(E.extract_code(r["content"]), tests)
            OUT["hard"].append({"task": name, "effort": effort, "pass": ok, "err": err,
                                **{k: r.get(k) for k in ("wall", "out_tokens", "tok_s", "finish")}})
            print(f"hard {effort:5} {name:15} {'PASS' if ok else 'FAIL'} {r.get('wall')} s {r.get('out_tokens')} tok  {err}", flush=True)


JS = [
    ("groupBy", "Write `groupBy(arr, fn)`: returns an object mapping fn(item) to the array of items with that key, "
     "in input order.",
     "assert.deepStrictEqual(groupBy([1,2,3,4,5], x => x % 2 ? 'odd' : 'even'), {odd: [1,3,5], even: [2,4]});\n"
     "assert.deepStrictEqual(groupBy([], x => x), {});\n"),
    ("deepEqual", "Write `deepEqual(a, b)`: structural equality for primitives, arrays, plain objects and Date "
     "objects; NaN equals NaN; key order does not matter.",
     "assert.ok(deepEqual({a: [1, {b: NaN}], c: new Date(5)}, {c: new Date(5), a: [1, {b: NaN}]}));\n"
     "assert.ok(!deepEqual({a: 1}, {a: 1, b: undefined}));\nassert.ok(!deepEqual([1, 2], [1, 2, 3]));\n"
     "assert.ok(!deepEqual(new Date(1), new Date(2)));\nassert.ok(deepEqual(null, null) && !deepEqual(null, {}));\n"
     "assert.ok(!deepEqual([1], {0: 1}));\n"),
    ("parseQuery", "Write `parseQuery(str)`: parses a URL query string (an optional leading '?') into an object; "
     "values are percent-decoded and '+' means space; a key that repeats becomes an array of its values; a key "
     "without '=' gets ''.",
     "assert.deepStrictEqual(parseQuery('?a=1&b=2&a=3&c=%20x+y&d'), {a: ['1','3'], b: '2', c: ' x y', d: ''});\n"
     "assert.deepStrictEqual(parseQuery(''), {});\n"),
    ("chunk", "Write `chunk(arr, n)`: splits an array into arrays of length n (the last may be shorter); throws "
     "RangeError when n < 1.",
     "assert.deepStrictEqual(chunk([1,2,3,4,5], 2), [[1,2],[3,4],[5]]);\nassert.deepStrictEqual(chunk([], 3), []);\n"
     "assert.throws(() => chunk([1], 0), RangeError);\n"),
]


def js():
    node = shutil.which("node")
    for name, task, tests in JS:
        r = E.chat([{"role": "user", "content": task + "\n\nPlain JavaScript for Node.js, no modules or exports: "
                     "define the function at the top level. Reply with one ```javascript code block."}], "none", 4000)
        m = re.findall(r"```(?:javascript|js)?\s*\n(.*?)```", r.get("content", ""), re.S)
        code = max(m, key=len) if m else r.get("content", "")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code + "\nconst assert = require('assert');\n" + tests + "console.log('PASS');\n")
        p = subprocess.run([node, f.name], capture_output=True, text=True, timeout=20)
        ok = "PASS" in p.stdout
        OUT["js"].append({"task": name, "pass": ok, "err": (p.stderr.strip().splitlines() or [""])[0][:160], "wall": r.get("wall")})
        print(f"js {name:11} {'PASS' if ok else 'FAIL'} {r.get('wall')} s", flush=True)
        Path(f.name).unlink(missing_ok=True)


SCHEMA = """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, created TEXT);
INSERT INTO customers VALUES (1,'Ann'),(2,'Bob'),(3,'Cid'),(4,'Dee');
INSERT INTO orders VALUES (1,1,50,'2026-01-01'),(2,1,70,'2026-02-01'),(3,2,200,'2026-01-15'),(4,3,20,'2026-03-01'),
(5,3,30,'2026-01-20');"""
SQL = [
    ("Names of customers who have no orders, sorted by name. One column.", [("Dee",)]),
    ("The two customers with the highest total order amount: name and total, highest first.",
     [("Bob", 200.0), ("Ann", 120.0)]),
    ("For each customer who has orders: name and the amount of their most recent order (by created), sorted by name.",
     [("Ann", 70.0), ("Bob", 200.0), ("Cid", 20.0)]),
]


def sql():
    for q, want in SQL:
        r = E.chat([{"role": "user", "content": "SQLite schema:\n" + SCHEMA.split("INSERT")[0] + "\n" + q +
                     "\nReply with one ```sql code block holding a single SELECT statement."}], "none", 1500)
        m = re.findall(r"```(?:sql)?\s*\n(.*?)```", r.get("content", ""), re.S)
        stmt = (m[-1] if m else r.get("content", "")).strip()
        db = sqlite3.connect(":memory:")
        db.executescript(SCHEMA)
        try:
            got = [tuple(float(x) if isinstance(x, (int, float)) else x for x in row) for row in db.execute(stmt)]
            ok, err = got == want, ""
        except sqlite3.Error as e:
            got, ok, err = None, False, str(e)[:120]
        OUT["sql"].append({"q": q[:60], "pass": ok, "got": got, "err": err})
        print(f"sql {'PASS' if ok else 'FAIL'} {q[:50]!r} {got} {err}", flush=True)


REGEX = [
    ("an IPv4 address in dotted decimal, each part 0-255 (no leading zeros except '0' itself)",
     ["0.0.0.0", "255.255.255.255", "192.168.1.1", "10.0.0.1"],
     ["256.1.1.1", "1.1.1", "1.1.1.1.1", "a.b.c.d", "1.1.1.", "999.0.0.1", "01.1.1.1"]),
    ("a CSS hex color: '#' followed by exactly 3 or exactly 6 hex digits",
     ["#fff", "#FFFFFF", "#a1B2c3"], ["fff", "#ffff", "#ggg", "#12345", "#1234567"]),
    ("a date YYYY-MM-DD with month 01-12 and day 01-31 (no per-month day check)",
     ["2026-10-03", "1999-01-31", "2000-12-01"],
     ["2026-13-01", "2026-00-10", "2026-10-32", "2026-1-1", "26-10-03", "2026-10-00"]),
]


def regex():
    for what, good, bad in REGEX:
        r = E.chat([{"role": "user", "content": f"Give a Python regular expression for re.fullmatch that matches {what}. "
                     "Reply with only the pattern on one line inside single backticks."}], "none", 800)
        m = re.findall(r"`([^`\n]+)`", r.get("content", ""))
        pat = m[-1] if m else r.get("content", "").strip()
        try:
            ok = all(re.fullmatch(pat, g) for g in good) and not any(re.fullmatch(pat, b) for b in bad)
        except re.error:
            ok = False
        OUT["regex"].append({"what": what[:40], "pass": ok, "pattern": pat})
        print(f"regex {'PASS' if ok else 'FAIL'} {what[:35]!r} {pat}", flush=True)


# ------------------------------------------------------------------------------------------------ agent loop
PRICING_BUG = '''def discount(total):
    """10% off for totals of 100 or more, 20% off for 500 or more."""
    if total > 500:
        return total * 0.2
    if total > 100:
        return total * 0.1
    return 0.0


def tax(amount, rate=0.2):
    return round(amount * rate, 2)
'''
CART = '''from pricing import discount, tax


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty=1):
        self.items.append((name, price, qty))

    def subtotal(self):
        return sum(%s for _, price, qty in self.items)

    def total(self):
        sub = self.subtotal()
        net = sub - discount(sub)
        return round(net + tax(net), 2)
'''
TESTS = '''import unittest
from cart import Cart


class T(unittest.TestCase):
    def test_qty(self):
        c = Cart(); c.add('pen', 2.5, 4); self.assertEqual(c.subtotal(), 10.0)

    def test_no_discount(self):
        c = Cart(); c.add('book', 50); self.assertEqual(c.total(), 60.0)

    def test_discount_at_100(self):
        c = Cart(); c.add('lamp', 100); self.assertEqual(c.total(), 108.0)

    def test_discount_at_500(self):
        c = Cart(); c.add('tv', 250, 2); self.assertEqual(c.total(), 480.0)


if __name__ == '__main__':
    unittest.main()
'''
HIDDEN_B = '''from cart import Cart
import coupons
c = Cart(); c.add('a', 10); c.add('b', 5); c.add('a', 1); assert c.remove('a') == 2 and c.subtotal() == 5
c = Cart(); c.add('book', 50); c.apply_coupon('SAVE5'); assert c.total() == 54.0, c.total()
c.apply_coupon('SAVE20'); assert c.total() == 36.0, c.total()
try:
    c.apply_coupon('NOPE'); raise AssertionError('unknown code')
except ValueError:
    pass
c = Cart(); c.add('x', 3); c.apply_coupon('SAVE20'); assert c.total() == 0.0
assert coupons.COUPONS['SAVE5'] == 5.0
print('PASS')
'''
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List the project's files.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a project file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (replace) a project file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's unit tests.",
     "parameters": {"type": "object", "properties": {}}}},
]


def run_py(root, args):
    p = subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=30)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def agent_run(label, files, task, effort, check):
    root = Path(tempfile.mkdtemp(prefix="strata-agent-"))
    for n, c in files.items():
        (root / n).write_text(c, encoding="utf-8")
    msgs = [{"role": "system", "content": "You are a coding agent working in a small Python project. Use the tools "
             "to inspect and change files and to run the tests. When the task is done, reply with DONE."},
            {"role": "user", "content": task}]
    steps = calls = 0
    total_wall = 0.0
    for steps in range(1, 26):
        r = E.chat(msgs, effort, 6000, tools=AGENT_TOOLS)
        total_wall += r.get("wall", 0)
        if "error" in r:
            break
        tc = r.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": r["content"], **({"tool_calls": tc} if tc else {})})
        if not tc:
            break
        for c in tc:
            calls += 1
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                a = None
            name = c["function"]["name"]
            if a is None:
                res = "error: arguments are not valid JSON"
            elif name == "list_files":
                res = "\n".join(sorted(p.name for p in root.iterdir() if p.is_file()))
            elif name in ("read_file", "write_file"):
                p = (root / str(a.get("path", ""))).resolve()
                if root.resolve() not in p.parents:
                    res = "error: path outside the project"
                elif name == "read_file":
                    res = p.read_text(encoding="utf-8") if p.exists() else "error: no such file"
                else:
                    p.write_text(str(a.get("content", "")), encoding="utf-8")
                    res = "written"
            elif name == "run_tests":
                res = run_py(root, ["-m", "unittest", "-q"])[1]
            else:
                res = "error: unknown tool"
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "call"), "content": res})
    ok, note = check(root)
    shutil.rmtree(root, ignore_errors=True)
    OUT["agent"].append({"scenario": label, "effort": effort, "pass": ok, "note": note, "turns": steps,
                         "tool_calls": calls, "wall": round(total_wall, 1)})
    print(f"agent {label:8} {effort:6} {'PASS' if ok else 'FAIL'} {steps} turns, {calls} tool calls, {total_wall:.0f} s  {note}", flush=True)


def agent():
    def check_a(root):
        rc, out = run_py(root, ["-m", "unittest", "-q"])
        same = (root / "test_cart.py").read_text(encoding="utf-8") == TESTS
        return rc == 0 and same, ("" if rc == 0 else out.strip().splitlines()[-1][:120]) + ("" if same else " tests modified")

    def check_b(root):
        (root / "_hidden.py").write_text(HIDDEN_B, encoding="utf-8")
        rc1, out1 = run_py(root, ["_hidden.py"])
        rc2, out2 = run_py(root, ["-m", "unittest", "-q"])
        added = (root / "test_cart.py").read_text(encoding="utf-8").count("def test_") > 4
        note = ("" if rc1 == 0 else "hidden: " + out1.strip().splitlines()[-1][:100]) + \
               ("" if rc2 == 0 else " own tests fail") + ("" if added else " no tests added")
        return rc1 == 0 and rc2 == 0 and added, note

    bug = {"pricing.py": PRICING_BUG, "cart.py": CART % "price", "test_cart.py": TESTS}
    good = {"pricing.py": PRICING_BUG.replace("> 500", ">= 500").replace("> 100", ">= 100"),
            "cart.py": CART % "price * qty", "test_cart.py": TESTS}
    feat = ("Add two things to Cart. 1) `remove(name)`: removes every line with that name and returns how many lines "
            "were removed. 2) `apply_coupon(code)`: coupon codes live in a new module coupons.py as "
            "COUPONS = {'SAVE5': 5.0, 'SAVE20': 20.0}; a valid coupon subtracts its value from the net amount (after "
            "the discount, before tax), never below 0; an unknown code raises ValueError; only one coupon applies, "
            "the latest one. Add tests for both to test_cart.py and make all tests pass.")
    for effort in ("none", "medium"):
        agent_run("bugfix", bug, "The unit tests fail. Find and fix the bugs in the code. Do not change the tests.",
                  effort, check_a)
        agent_run("feature", good, feat, effort, check_b)


# ------------------------------------------------------------------------------------------------ long-file QA
def qa():
    src = (HERE.parent / "setup.py").read_text(encoding="utf-8")
    msgs = [{"role": "user", "content": "Here is a Python file, setup.py:\n\n" + src +
             "\n\nI will ask short questions about it. Answer each in one short line. First: what is the value of "
             "the constant DESKTOP_RESERVE_MIB?"}]
    qs = [(None, "3072"),
          ("What is the default value of ROCM_VERSION (when the environment variable is not set)?", "7.10.0a20251120"),
          ("What is the value of WIN_HIP_ASSET?", "strata-windows-x64-hip.zip"),
          ("What is the name of the function that compiles the CPU image encoder beside the HIP engine?", "build_vision_cpu"),
          ("What is the value of the MMPROJ constant?", "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
          ("Is there a function named install_vulkan_sdk in the file? Answer yes or no.", "no")]
    for q, want in qs:
        if q:
            msgs.append({"role": "user", "content": q})
        r = E.chat(msgs, "none", 300)
        a = r.get("content", r.get("error", ""))
        msgs.append({"role": "assistant", "content": a})
        ok = want.lower() in a.lower() and not (want == "no" and "yes" in a.lower())
        OUT["qa"].append({"want": want, "got": a[:160], "pass": ok, **{k: r.get(k) for k in ("prompt_n", "cache_n",
                                                                                               "prompt_s", "tok_s")}})
        print(f"qa {'PASS' if ok else 'FAIL'} read {r.get('prompt_n')} tok in {r.get('prompt_s')} s (reused "
              f"{r.get('cache_n')}), {r.get('tok_s')} tok/s | want {want!r} got {a[:90]!r}", flush=True)


def longgen():
    r = E.chat([{"role": "user", "content": "Write a complete single-file Python command-line todo app, todo.py, "
                 "using argparse and a todo.json file in the current directory. Commands: `add <text>`, `list`, "
                 "`done <id>`, `remove <id>`, `clear-done`, `stats`. Ids are stable integers. `list` prints one line "
                 "per item with its id, a [x] or [ ] mark and the text, and supports `--all` / `--pending` / `--done` "
                 "filters. Handle a missing or corrupt todo.json. Include docstrings and type hints. Reply with one "
                 "```python code block."}], "none", 8000)
    root = Path(tempfile.mkdtemp(prefix="strata-todo-"))
    (root / "todo.py").write_text(E.extract_code(r.get("content", "")), encoding="utf-8")
    rc1, _ = run_py(root, ["todo.py", "add", "buy milk"])
    rc2, _ = run_py(root, ["todo.py", "add", "write report"])
    rc3, _ = run_py(root, ["todo.py", "done", "1"])
    rc4, out = run_py(root, ["todo.py", "list", "--all"])
    ok = rc1 == rc2 == rc3 == rc4 == 0 and "buy milk" in out and "write report" in out and "[x]" in out and "[ ]" in out
    shutil.rmtree(root, ignore_errors=True)
    OUT["longgen"].append({"pass": ok, **{k: r.get(k) for k in ("wall", "out_tokens", "tok_s", "finish")}, "list": out[:200]})
    print(f"longgen {'PASS' if ok else 'FAIL'} {r.get('out_tokens')} tok in {r.get('wall')} s ({r.get('tok_s')} tok/s) | {out[:120]!r}", flush=True)


if __name__ == "__main__":
    funcs = {"hard": hard, "js": js, "sql": sql, "regex": regex, "agent": agent, "qa": qa, "longgen": longgen}
    for name in sys.argv[1:] or ["js", "sql", "regex", "longgen", "agent", "qa", "hard"]:
        try:
            funcs[name]()
        except Exception as e:  # one broken section must not lose the others
            print(f"{name} CRASHED {type(e).__name__}: {e}", flush=True)
        # RESULTS2: another file name, so that a run for another model or engine keeps the earlier results
        (HERE / __import__("os").environ.get("RESULTS2", "results2.json")).write_text(
            json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", flush=True)
