#!/usr/bin/env python3
"""One command that measures this Strata install the way it is used, and writes a report you can share.

    BENCH.bat                      (Windows)        ./bench.sh        (Linux)        python tools/strata_bench.py

It starts the server with your existing settings (the newest strata-*.json of this folder, or --config FILE), or uses
the one already running (it checks the config's port, or --url http://host:port), and sends a fixed suite through the
real HTTP API, three runs each, medians reported:

  1. chat       three short questions (about 100 tokens out): decode tok/s and time to first token
  2. 4K prompt  a prompt of about 4,096 tokens: prompt tok/s and time to first token (fresh every run: no cache help)
  3. 16K prompt the same at about 16,384 tokens (skipped when the context is smaller)
  4. agent      a coding-agent style session: system prompt + four tools, then five turns, each adding about 1,500 words
                of a file to the conversation and continuing from the model's own earlier replies - how many prompt
                tokens the conversation cache reused, how many had to be read again, time to first token per turn
  5. --clients N   N requests at once (the server needs --batch for them to run together): total tok/s, p50/p95 latency
  6. --long 32K    the same prompt test at about 32,768 tokens

Speeds are the engine's own timings (the `timings` of the answer: prompt_per_second, predicted_per_second); time to first
token and latency are measured by this client.  Thinking is off for the suite (--think turns it on), so the answers are
short and the runs comparable.  The default run takes about 3-10 minutes after the model has loaded (the slower the card, the longer).

The result goes to bench/results/<date>-community-<gpu>-<model>/ : README.md in the format of
docs/COMMUNITY_BENCHMARKS.md, results.json (every request) and an engine log excerpt.  Home folders, user and host names,
IP addresses and API keys are replaced by placeholders before anything is written; read it once before you share it.
`--compare <older results folder>` prints the two side by side.  Standard library only.
This is the quick speed check, not the standard benchmark: `strata bench run` follows docs/BENCHMARK_PROTOCOL.md (fixed prompts, thinking on,
the model card sampling, twelve sanity checks) and writes results other people can compare with.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_engine import PROMPTS, long_prompt          # noqa: E402  the same three short questions and word list
from agent_turns_bench import SYSTEM, TOOLS         # noqa: E402  the same agent system prompt and tool list
import serve_load                                   # noqa: E402  the burst client for --clients

SCHEMA = 1
CHAT_MAX_TOKENS = 100
LONG_MAX_TOKENS = 24
AGENT_MAX_TOKENS = 64
AGENT_TURNS = 5
AGENT_WORDS = 1500
AGENT_MIN_CONTEXT = 10240      # the last turn of the session holds about 10K tokens
BURST_MAX_TOKENS = 128


# --------------------------------------------------------------------------------------------------------- redaction
def redactor(extra: list[str] = (), key: str = ""):
    """-> f(text) -> text with this machine's home folders, user and host names, the Strata folder, IP addresses and the
    API key replaced.  Applied to everything that is written or printed from the server's or the machine's own text."""
    subs: list[tuple[re.Pattern, str, int]] = []

    def add(value, label, word=False):
        if value and len(str(value)) >= 4:
            v = str(value)
            for form in sorted({v, v.replace("\\", "/"), v.replace("/", "\\")}, key=len, reverse=True):
                pat = re.escape(form)
                subs.append((re.compile(r"(?<![\w-])" + pat + r"(?![\w-])" if word else pat, re.I), label, len(form)))

    add(ROOT, "<strata>")
    add(Path.home(), "<home>")
    add(tempfile.gettempdir(), "<tmp>")
    try:
        add(getpass.getuser(), "<user>", True)
    except Exception:                                   # noqa: BLE001
        pass
    add(socket.gethostname(), "<host>", True)
    add(platform.node(), "<host>", True)
    for x in extra:
        add(x, "<private>", True)
    add(key, "<key>")
    subs.sort(key=lambda s: -s[2])      # by the text that is replaced: the word wrappers would make a short user name outrank the folder holding it
    path_user = re.compile(r"(?i)([A-Za-z]:)?([\\/])(Users|home)([\\/])(?!<)[^\\/\s\"':;,]+")
    ip4 = re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|169\.254)\.\d{1,3}\.\d{1,3}\b")
    bearer = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")

    def f(text):
        if not isinstance(text, str):
            return text
        for pat, label, _ in subs:
            text = pat.sub(label, text)
        text = path_user.sub(lambda m: f"{m.group(1) or ''}{m.group(2)}{m.group(3)}{m.group(4)}<user>", text)
        text = bearer.sub(r"\1<key>", text)
        return ip4.sub("<ip>", text)
    return f


def redact_obj(obj, red):
    if isinstance(obj, dict):
        return {red(k) if isinstance(k, str) else k: redact_obj(v, red) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, red) for v in obj]
    return red(obj) if isinstance(obj, str) else obj


# ---------------------------------------------------------------------------------------------------------- HTTP
def http_json(base: str, path: str, key: str = "", timeout: float = 20):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def try_json(base, path, key="", timeout=20):
    try:
        return http_json(base, path, key, timeout)
    except Exception:                                   # noqa: BLE001
        return None


def chat(base: str, key: str, messages: list, max_tokens: int, think: bool = False, tools=None, timeout: float = 3600) -> dict:
    """One streamed chat request -> {first, total, text, usage, timings, err}.  `first` is the seconds to the first
    non-empty content or reasoning piece (keep-alives and empty deltas ignored)."""
    body = {"model": "strata", "stream": True, "max_tokens": max_tokens, "temperature": 0, "messages": messages}
    if tools:
        body["tools"] = tools
    if not think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    first, text, usage, timings, err, pieces = None, [], None, None, None, 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                if j.get("error"):
                    err = str(j["error"])[:300]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                        pieces += 1
                        text.append(dl.get("content") or "")
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("timings"):
                    timings = j["timings"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:                              # noqa: BLE001
        err = repr(e)[:300]
    total = time.perf_counter() - t0
    if os.environ.get("STRATA_BENCH_ANSWERS"):
        try:
            with open(os.environ["STRATA_BENCH_ANSWERS"], "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"messages": messages[-1:], "text": "".join(text), "usage": usage,
                                     "err": err}) + "\n")
        except Exception:                               # noqa: BLE001
            pass
    return {"first": first, "total": total, "text": "".join(text), "usage": usage, "timings": timings, "err": err,
            "pieces": pieces}


def digest(r: dict) -> dict:
    """The numbers of one request, the engine's own where it reported them."""
    u, t = r.get("usage") or {}, r.get("timings") or {}
    gen = u.get("completion_tokens") or r.get("pieces") or 0
    prompt_tokens = u.get("prompt_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = t.get("cache_n")
    read = t.get("prompt_n")
    if read is None and prompt_tokens is not None and cached is not None:
        read = prompt_tokens - cached
    dec, dsrc = t.get("predicted_per_second"), "engine"
    if dec is None and r.get("first") is not None and gen > 1 and r["total"] > r["first"]:
        dec, dsrc = round((gen - 1) / (r["total"] - r["first"]), 1), "client"
    pp, psrc = t.get("prompt_per_second"), "engine"
    if pp is None:
        pp, psrc = None, None
    return {"ok": r.get("err") is None and r.get("first") is not None,
            "err": r.get("err"), "ttft_s": None if r.get("first") is None else round(r["first"], 3),
            "total_s": round(r["total"], 3), "prompt_tokens": prompt_tokens, "reused_tokens": cached,
            "read_tokens": read, "generated_tokens": gen, "prompt_tps": pp, "prompt_tps_source": psrc,
            "decode_tps": dec, "decode_tps_source": dsrc if dec is not None else None,
            "prompt_ms": t.get("prompt_ms"), "decode_ms": t.get("predicted_ms"),
            "drafts_offered": t.get("draft_n"), "drafts_accepted": t.get("draft_n_accepted")}


# ----------------------------------------------------------------------------------------------------- statistics
def stat(values) -> dict | None:
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {"median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3), "n": len(v)}


def fmt(s, digits=1, unit="") -> str:
    if not s:
        return "not measured"
    d = digits
    return f"{s['median']:.{d}f}{unit} ({s['min']:.{d}f}-{s['max']:.{d}f})"


def pctl(values, p):
    return serve_load.pct(values, p)


# ---------------------------------------------------------------------------------------------------- the suite
class Suite:
    def __init__(self, base, key, ctx, runs, think, log=print, want_clients=0, want_long=False, quick=False):
        self.base, self.key, self.ctx, self.runs, self.think, self.log = base, key, ctx, runs, think, log
        self.clients, self.want_long = want_clients, want_long
        self.quick = quick
        self.nonce = str(int(time.time()))
        self.tokens_per_word, self.overhead = 1.1, 40
        self.out: dict = {"chat": {}, "prompts": {}, "agent": {}, "clients": {}, "skipped": {}, "failures": []}

    def tag(self, *parts) -> str:
        return "-".join([*map(str, parts), self.nonce])

    def fail(self, where, d):
        self.out["failures"].append({"where": where, "error": d.get("err") or "no token came back"})
        self.log(f"    FAILED: {d.get('err') or 'no token came back'}")

    def calibrate(self):
        """Words that make a prompt of a wanted size: two tiny requests (200 and 800 words) give the tokens per word as a
        slope, so the template's fixed part does not skew it."""
        pts = []
        for words in (200, 800):
            r = digest(chat(self.base, self.key, [{"role": "user", "content": f"[{self.tag('cal', words)}]\n" + long_prompt(words)}], 1))
            pts.append(r.get("prompt_tokens"))
        slope = (pts[1] - pts[0]) / 600 if all(pts) and pts[1] > pts[0] else None
        self.tokens_per_word = slope or 1.1
        self.overhead = max(0, (pts[0] or 0) - 200 * self.tokens_per_word) if slope else 40
        self.log(f"  calibrated: about {self.tokens_per_word:.2f} tokens per word")

    def prompt_for(self, tokens: int, run: int) -> list:
        words = max(50, int((tokens - self.overhead) / self.tokens_per_word))
        return [{"role": "user", "content": f"[bench {self.tag(run, tokens, 'p')}]\n" + long_prompt(words)}]

    def warm_up(self):
        self.log("warming up (not measured) ...")
        chat(self.base, self.key, [{"role": "user", "content": PROMPTS[0]}], 24, self.think)
        self.calibrate()
        chat(self.base, self.key, self.prompt_for(1024, -1), 8, self.think)

    def run_chat(self):
        self.log(f"chat: {len(PROMPTS)} short questions x {self.runs} runs, up to {CHAT_MAX_TOKENS} new tokens")
        rows = []
        for run in range(self.runs):
            for i, p in enumerate(PROMPTS):
                d = digest(chat(self.base, self.key, [{"role": "user", "content": f"[{self.tag(run, i, 'c')}] {p}"}],
                                CHAT_MAX_TOKENS, self.think))
                d.update(run=run, prompt=i)
                rows.append(d)
                if not d["ok"]:
                    self.fail("chat", d)
                else:
                    self.log(f"  run {run + 1} q{i + 1}: decode {d['decode_tps']} tok/s, first token {d['ttft_s']} s, "
                             f"{d['generated_tokens']} tokens")
        good = [r for r in rows if r["ok"]]
        self.out["chat"] = {"requests": rows, "decode_tps": stat(r["decode_tps"] for r in good),
                            "ttft_s": stat(r["ttft_s"] for r in good), "generated_tokens": stat(r["generated_tokens"] for r in good),
                            "prompt_tokens": stat(r["prompt_tokens"] for r in good)}

    def run_prompt(self, tokens: int, label: str):
        if self.ctx and tokens + LONG_MAX_TOKENS + 256 > self.ctx:
            self.out["skipped"][label] = f"the context is {self.ctx} tokens"
            self.log(f"{label} prompt: skipped (the context is {self.ctx} tokens)")
            return
        self.log(f"{label} prompt: {self.runs} fresh prompts of about {tokens} tokens, {LONG_MAX_TOKENS} new tokens")
        rows = []
        for run in range(self.runs):
            d = digest(chat(self.base, self.key, self.prompt_for(tokens, run), LONG_MAX_TOKENS, self.think))
            d["run"] = run
            rows.append(d)
            if not d["ok"]:
                self.fail(label, d)
            else:
                self.log(f"  run {run + 1}: {d['prompt_tokens']} tokens, prompt {d['prompt_tps']} tok/s "
                         f"({d['reused_tokens'] or 0} reused), first token {d['ttft_s']} s")
        good = [r for r in rows if r["ok"]]
        self.out["prompts"][label] = {"target_tokens": tokens, "requests": rows,
                                      "prompt_tps": stat(r["prompt_tps"] for r in good),
                                      "ttft_s": stat(r["ttft_s"] for r in good),
                                      "prompt_tokens": stat(r["prompt_tokens"] for r in good),
                                      "reused_tokens": stat(r["reused_tokens"] or 0 for r in good)}

    def agent_files(self) -> list[tuple[str, str]]:
        """Five pieces of text a coding agent would read: this folder's own documents when they exist (nothing personal),
        else generated filler of the same size."""
        names = []
        for rel in ("README.md", "docs/HOW_IT_WORKS.md", "docs/DETAILS.md", "docs/INSTALL.md", "docs/MODELS.md", "docs/COMMUNITY_BENCHMARKS.md"):
            p = ROOT / rel
            if p.is_file():
                words = p.read_text(encoding="utf-8", errors="replace").split()
                if len(words) >= AGENT_WORDS // 2:
                    names.append((rel, " ".join(words[:AGENT_WORDS])))
        i = 0
        while len(names) < AGENT_TURNS:
            names.append((f"generated/part{i}.txt", long_prompt(AGENT_WORDS)))
            i += 1
        return names[:AGENT_TURNS]

    def agent_session(self, run: int) -> list[dict]:
        files = self.agent_files()
        msgs = [{"role": "system", "content": f"{SYSTEM}\n(session {self.tag(run, 'a')})"},
                {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                            "one or two sentences what it is for. Do not call tools."}]
        turns = []
        for t in range(AGENT_TURNS + 1):
            if t > 0:
                rel, body = files[t - 1]
                msgs.append({"role": "user", "content": f"File {rel}:\n{body}\n\nWhat is this file for? Two sentences."})
            r = chat(self.base, self.key, msgs, AGENT_MAX_TOKENS, self.think, tools=TOOLS)
            d = digest(r)
            d.update(run=run, turn=t)
            turns.append(d)
            if not d["ok"]:
                self.fail("agent", d)
                break
            msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
        return turns

    def run_agent(self):
        if self.ctx and self.ctx < AGENT_MIN_CONTEXT:
            self.out["skipped"]["agent"] = f"context {self.ctx} is below ~10K"
            self.log(f"agent: skipped: context {self.ctx} is below ~10K")
            return
        self.log(f"agent: a session of {AGENT_TURNS + 1} turns with four tools, x {self.runs} runs")
        sessions = []
        for run in range(self.runs):
            turns = self.agent_session(run)
            sessions.append(turns)
            for d in turns:
                if d["ok"]:
                    self.log(f"  run {run + 1} turn {d['turn']}: prompt {d['prompt_tokens']}, reused {d['reused_tokens']}, "
                             f"read {d['read_tokens']}, first token {d['ttft_s']} s")
        follow = [[d for d in s if d["ok"] and d["turn"] > 0] for s in sessions]
        hit = [sum(d["reused_tokens"] or 0 for d in s) / max(1, sum(d["prompt_tokens"] or 0 for d in s)) * 100
               for s in follow if s]
        # earlier conversation read again: what the previous turn held (prompt + its reply) minus what this turn reused
        reread, fresh = [], []
        for s in sessions:
            s = [d for d in s if d["ok"]]
            reread.append(sum(max(0, (p["prompt_tokens"] or 0) + (p["generated_tokens"] or 0) - (d["reused_tokens"] or 0))
                              for p, d in zip(s, s[1:])))
            fresh.append(sum(d["read_tokens"] or 0 for d in s[1:]) - reread[-1])
        self.out["agent"] = {"sessions": sessions, "hit_pct": stat(hit), "reread_tokens": stat(reread), "new_tokens": stat(fresh),
                             "turn_ttft_s": stat(d["ttft_s"] for s in follow for d in s),
                             "last_turn_prompt_tokens": stat(s[-1]["prompt_tokens"] for s in follow if s),
                             "turn_decode_tps": stat(d["decode_tps"] for s in follow for d in s),
                             "first_turn_ttft_s": stat(s[0]["ttft_s"] for s in sessions if s and s[0]["ok"])}

    def run_clients(self, n: int):
        self.log(f"concurrency: {n} requests at once x {self.runs} rounds, {BURST_MAX_TOKENS} new tokens each")
        rows = []
        for rd in range(self.runs):
            res: dict = {}
            th = []
            t0 = time.perf_counter()
            for i in range(n):
                t = threading.Thread(target=serve_load.one, args=(self.base, self.key, "strata", i, 400, BURST_MAX_TOKENS, res, {},
                                                                 self.think, f" {self.tag('b', n, rd)}"))
                t.start()
                th.append(t)
            for t in th:
                t.join()
            wall = time.perf_counter() - t0
            toks = sum(r["tokens"] for r in res.values())
            errs = [r["err"] for r in res.values() if r["err"]]
            rows.append({"round": rd, "wall_s": round(wall, 2), "tokens": toks, "total_tps": round(toks / wall, 2),
                         "first_s": [r["first"] for r in res.values() if r["first"] is not None],
                         "latency_s": [r["total"] for r in res.values()], "errors": errs[:3], "error_count": len(errs)})
            self.log(f"  round {rd + 1}: {toks} tokens in {wall:.1f} s = {toks / wall:.1f} tok/s total"
                     + (f"  ERRORS {errs[:1]}" if errs else ""))
        firsts = [x for r in rows for x in r["first_s"]]
        lats = [x for r in rows for x in r["latency_s"]]
        self.out["clients"] = {"clients": n, "rounds": rows, "total_tps": stat(r["total_tps"] for r in rows),
                               "ttft_p50_s": round(pctl(firsts, .5), 3) if firsts else None,
                               "ttft_p95_s": round(pctl(firsts, .95), 3) if firsts else None,
                               "latency_p50_s": round(pctl(lats, .5), 3) if lats else None,
                               "latency_p95_s": round(pctl(lats, .95), 3) if lats else None,
                               "errors": sum(r["error_count"] for r in rows)}

    def run(self):
        self.warm_up()
        self.run_chat()
        self.run_prompt(4096, "4K")
        self.run_prompt(16384, "16K")
        self.run_agent()
        if self.clients:
            self.run_clients(self.clients)
        if self.want_long:
            self.run_prompt(32768, "32K")
        return self.out


def summary_metrics(out: dict) -> dict:
    """The flat {name: stat} the --compare table and the README table read."""
    m = {"chat decode tok/s": out["chat"].get("decode_tps"), "chat first token (s)": out["chat"].get("ttft_s")}
    for k, v in out["prompts"].items():
        m[f"{k} prompt tok/s"] = v.get("prompt_tps")
        m[f"{k} first token (s)"] = v.get("ttft_s")
    a = out["agent"]
    if a:
        m["agent cache hit %"] = a.get("hit_pct")
        m["agent re-read tokens/session"] = a.get("reread_tokens")
        m["agent turn first token (s)"] = a.get("turn_ttft_s")
        m["agent decode tok/s"] = a.get("turn_decode_tps")
    c = out["clients"]
    if c:
        m[f"{c['clients']} clients total tok/s"] = c.get("total_tps")
    return m


# -------------------------------------------------------------------------------------------------- the machine
def run_cmd(cmd, timeout=15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def total_ram_gib() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2 ** 20, 1)
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong)] + [
                    (n, ctypes.c_ulonglong) for n in ("a", "b", "c", "d", "e")]
            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.total / 2 ** 30, 1)
        except Exception:                               # noqa: BLE001
            pass
    return None


def cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if os.name == "nt":
        out = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if out.strip():
            return out.strip()
    return platform.processor() or None


def gpu_info_local() -> dict:
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit,power.draw,pcie.link.gen.current,pcie.link.width.current,pstate,temperature.gpu",
                   "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            g = {"name": p[0], "vram_mib": int(float(p[1])) if p[1].replace(".", "").isdigit() else None, "driver": p[2]}
            if len(p) >= 9:
                g.update({"power_limit_w": p[3], "power_draw_w": p[4], "pcie": f"gen{p[5]} x{p[6]}",
                          "pstate": p[7], "temp_c": p[8]})
            gpus.append(g)
    if not gpus:
        out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                gpus.append({"name": p[0], "vram_mib": int(float(p[1])) if p[1].replace(".", "").isdigit() else None, "driver": p[2]})
    info = {"gpus": gpus}
    head = run_cmd(["nvidia-smi"])
    m = re.search(r"CUDA(?: UMD)? Version:\s*([\d.]+)", head)
    if m:
        info["cuda"] = m.group(1)
    if not gpus:
        rocm = run_cmd(["rocm-smi", "--showproductname"])
        names = re.findall(r"Card [Ss]eries:\s*(.+)", rocm) or re.findall(r"Card [Mm]odel:\s*(.+)", rocm)
        info["gpus"] = [{"name": n.strip(), "vram_mib": None, "driver": None} for n in names]
        v = run_cmd(["rocm-smi", "--showdriverversion"])
        m2 = re.search(r"Driver version:\s*(\S+)", v)
        if m2:
            info["rocm_driver"] = m2.group(1)
    if not info["gpus"] and os.name != "nt":
        # ROCm 7 and newer ship `amd-smi` and no `rocm-smi`, so a Radeon wrote the server's vendor stub ("AMD
        # Radeon", `...-community-unknown-...`) and `GPU driver: not read`. `static --asic` gives the marketing
        # name, `static --driver` the kernel driver's version, the PCI address only comes from `amd-smi list` and
        # the VRAM size is the amdgpu driver's sysfs file for that address (same probe as strata_cli/bench_facts.py).
        def amd_json(args):
            try:
                return json.loads(run_cmd(["amd-smi", "static"] + args + ["--json"]).strip() or "{}").get("gpu_data") or []
            except ValueError:                               # noqa: BLE001 - a report without the card's name is a report
                return []
        try:
            bdf = {e.get("gpu"): (e.get("bdf") or "") for e in json.loads(run_cmd(["amd-smi", "list", "--json"]).strip() or "[]")}
        except ValueError:
            bdf = {}
        drv = {e.get("gpu"): ((e.get("driver") or {}).get("version") or "") for e in amd_json(["--driver"])}
        cards = []
        for e in amd_json(["--asic"]):
            name = ((e.get("asic") or {}).get("market_name") or "").strip()
            if not name:
                continue
            d = drv.get(e.get("gpu"), "")
            d = d if d.upper() not in ("", "N/A") else None
            if d:
                info["rocm_driver"] = info.get("rocm_driver") or d
            vram = None
            for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
                try:
                    if bdf.get(e.get("gpu")) and os.path.basename(os.path.realpath(dev)).lower() == bdf[e["gpu"]].lower():
                        vram = int((dev / "mem_info_vram_total").read_text(encoding="utf-8").strip()) >> 20
                except (OSError, ValueError):
                    pass
            cards.append({"name": name, "vram_mib": vram, "driver": d})
        info["gpus"] = cards
    if not info["gpus"] and os.name == "nt":
        # #1587: Windows with only an AMD card has neither nvidia-smi nor rocm-smi; the display class in the registry
        # names the card, its VRAM and its driver (the run was labelled "CPU only" before)
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from serve.telemetry import windows_amd_cards
            info["gpus"] = [{"name": c["name"], "vram_mib": (c["vram_bytes"] >> 20) if c["vram_bytes"] else None,
                             "driver": c["driver"]} for c in windows_amd_cards()]
        except Exception:   # noqa: BLE001 - a report without the card's name is still a report
            pass
    return info


def storage_info() -> str | None:
    if os.name == "nt":
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "Get-PhysicalDisk | Select-Object -First 4 FriendlyName,MediaType,BusType | "
                       "ForEach-Object { $_.FriendlyName + ' ' + $_.MediaType + '/' + $_.BusType }"])
        names = [x.strip() for x in out.splitlines() if x.strip() and "FriendlyName" not in x]
        return "; ".join(names) or None
    if sys.platform == "darwin":
        out = run_cmd(["system_profiler", "SPNVMeDataType", "SPSerialATADataType"])
        names = re.findall(r"Model:\s*(.+)", out)
        if names:
            return "; ".join(n.strip() for n in names)
        return None
    out = run_cmd(["lsblk", "-d", "-o", "NAME,MODEL,TRAN", "-n", "-e", "7"])   # -e 7: skip loop devices (snap mounts)
    if out.strip():
        return "; ".join(x.strip() for x in out.splitlines() if x.strip()) or None
    return None


def engine_build(cfg: dict | None) -> dict | None:
    """BUILD.json next to the engine exe: the toolkit/archs it was compiled with, unlike nvidia-smi's driver ceiling."""
    exe = (cfg or {}).get("exe")
    if not exe:
        return None
    try:
        b = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {k: b[k] for k in ("version", "source", "cuda", "rocm", "archs", "ptx", "portable") if b.get(k) is not None} or None


QUANT = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_[A-Z0-9_]+|Q\d_K(?:_[A-Z]+)?|IQ\d_?[A-Z]*|BF16|F16|Q\d))(?=[-_.]|$)")
SECRET_KEY = re.compile(r"(?i)key|token|secret|password|auth")


def base_if_path(v):
    if isinstance(v, str) and (("/" in v or "\\" in v) or re.search(r"\.(gguf|bin|json|pack|txt|jinja|so|dll|exe)$", v, re.I)):
        return re.split(r"[\\/]", v.rstrip("\\/"))[-1]
    return v


def config_view(cfg: dict) -> dict:
    """The settings worth reporting, with paths cut to file names and secrets dropped."""
    out = {}
    for k, v in cfg.items():
        if SECRET_KEY.search(k) or k in ("log", "cwd", "env"):
            continue
        if k == "exe":
            out[k] = base_if_path(v)
        elif k == "args" and isinstance(v, list):
            out[k] = [base_if_path(x) if isinstance(x, str) else x for x in v]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = base_if_path(v)
        elif isinstance(v, (list, dict)):
            out[k] = json.loads(json.dumps(v))
    env = cfg.get("env")
    if isinstance(env, dict):
        out["env"] = {k: base_if_path(str(v)) for k, v in env.items() if k.startswith(("STRATA_", "HIP_", "CUDA_", "ROCR_", "ONEAPI_", "SYCL_")) and not SECRET_KEY.search(k)}
    return out


def launch_line(view: dict) -> str:
    args = " ".join(f'"{a}"' if " " in str(a) else str(a) for a in view.get("args", []))
    return f"{view.get('exe', 'strata')} {args}".strip()


def detect_quant(view: dict) -> str | None:
    names = [a for a in view.get("args", []) if isinstance(a, str) and re.search(r"\.(gguf|pack)$|^[\w.-]+$", a)]
    names += [str(view.get("model_name") or "")]
    for n in names:
        m = QUANT.search(n)
        if m:
            return m.group(1).upper()
    return None


def project_version() -> str | None:
    try:
        # UTF-8 as setup.py reads it, whatever the locale's code page (#2792)
        m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
        return m.group(1) if m else None
    except OSError:
        return None


def git_commit() -> str | None:
    if not (ROOT / ".git").exists():
        return None
    out = run_cmd(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"]).strip()
    return out or None


def gpu_selection(cfg: dict | None, gpu_arg: str | None = None) -> list[int]:
    """#1985: the GPU numbers (as nvidia-smi numbers them) the engine is told to use: the bench's --gpu when it starts
    the server, else the config's "gpu" (one card, a list or "0,2"), else a "--gpu" in the config's args.  [] when
    nothing says."""
    def parse(v):
        if v is None or v == "":
            return []
        items = v if isinstance(v, (list, tuple)) else str(v).split(",")
        out = []
        for x in items:
            x = str(x).strip()
            if x.isdigit():
                out.append(int(x))
        return out
    sel = parse(gpu_arg) or parse((cfg or {}).get("gpu"))
    if not sel:
        args = [str(x) for x in ((cfg or {}).get("args") or [])]
        if "--gpu" in args and args.index("--gpu") + 1 < len(args):
            sel = parse(args[args.index("--gpu") + 1])
    return sel


def collect_machine(base: str, key: str, cfg: dict | None, gpu_arg: str | None = None) -> dict:
    status = try_json(base, "/v1/status", key) or {}
    metrics = try_json(base, "/metrics", key, 30) or {}
    props = try_json(base, "/props", key) or {}
    models = try_json(base, "/v1/models", key) or {}
    static = metrics.get("hardware_static") or {}
    eng = metrics.get("engine") or {}
    local = gpu_info_local()
    view = config_view(cfg) if cfg else {}
    # #1985: only the GPUs the engine uses - the config's / --gpu selection; with none given the engine runs on one card
    # (the one the server reports), not on every card the PC has
    used = local["gpus"]
    sel = gpu_selection(cfg, gpu_arg)
    if sel:
        used = [local["gpus"][i] for i in sel if 0 <= i < len(local["gpus"])] or used
    elif len(used) > 1 and static.get("gpu_name"):
        named = [g for g in used if g["name"] == static["gpu_name"]]
        used = named[:1] or used[:1]
    gpu_names = [g["name"] for g in used] or ([static["gpu_name"]] if static.get("gpu_name") else [])
    m = status.get("machine") or {}
    vram_mib = (m.get("gpu") or {}).get("total_mib") or None
    # #2001: the server names the cards it reads (" + " between them).  A name nvidia-smi does not list means the
    # engine runs on another vendor's card (an Intel Arc beside a GeForce): the server's names are the right ones, and
    # so is its VRAM total - the PC's NVIDIA cards were not used.
    served = [n for n in (static.get("gpu_name") or "").split(" + ") if n and n != "?"]
    # the engine's own stub when the driver names no card is the vendor alone ("AMD Radeon"), not a card name:
    # taking it as one would throw away the real name amd-smi read (same guard as strata_cli/bench_facts.py)
    served = [n for n in served if not re.fullmatch(r"(?i)\s*(nvidia|amd|intel|advanced micro devices)"
                                                    r"([ -](geforce|radeon|graphics|arc))?\s*", n)]
    if served and not all(n in {g["name"] for g in local["gpus"]} for n in served):
        gpu_names, used = served, []
        local = {**local, "cuda": None}            # the PC's CUDA belongs to cards the engine did not use
    elif used and used[0].get("vram_mib"):
        vram_mib = sum(g.get("vram_mib") or 0 for g in used)
    n_ctx = None
    for d in models.get("data") or []:
        n_ctx = (d.get("meta") or {}).get("n_ctx") or n_ctx
    n_ctx = n_ctx or (props.get("default_generation_settings") or {}).get("n_ctx")
    return {
        "gpu": gpu_names, "gpu_vram_mib": vram_mib, "gpu_driver": (used[0].get("driver") if used else local.get("rocm_driver")),
        "gpu_detail": [{k: v for k, v in g.items() if k not in ("name", "vram_mib", "driver")} for g in used] or None,
        "cuda": local.get("cuda"), "engine_build": engine_build(cfg),
        "cpu": static.get("cpu_name") or cpu_name(), "cpu_cores": static.get("cores"),
        "storage": storage_info(),
        "cpu_threads": static.get("threads") or os.cpu_count(), "ram_gib": total_ram_gib(),
        "os": f"{platform.system()} {'11' if int(platform.version().split('.')[-1] or 0) >= 22000 else platform.release()} ({platform.version()})" if os.name == "nt" else f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "strata_version": project_version(), "strata_commit": git_commit(),
        "engine_version": (eng.get("version") or status.get("engine") or str(props.get("build_info", "")).replace("Strata ", "") or None),
        "model": view.get("model_name") or status.get("model") or props.get("model_alias"),
        "quant": detect_quant(view) if view else None,
        "context": n_ctx or eng.get("max_context"), "parallel_slots": props.get("total_slots"),
        "config": view, "launch": launch_line(view) if view else None,
        "engine_info": {k: v for k, v in eng.items() if isinstance(v, (str, int, float, bool, dict)) and k not in ("model_path",)},
        "vram_after": (status.get("vram") or {}),
    }


# --------------------------------------------------------------------------------------------------- the server
def find_configs(folder: Path = ROOT) -> list[Path]:
    out = []
    for p in folder.glob("strata-*.json"):
        if p.name.endswith(".shared-settings.json"):
            continue
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(c, dict) and c.get("exe") and isinstance(c.get("args"), list):
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def is_strata(base: str, key: str) -> bool:
    h = try_json(base, "/health", key, 5)
    return bool(h and h.get("service") == "strata")


def without_profile_save(cfg: dict) -> tuple[dict, bool]:
    """#2088: the config without `--expert-profile-save FILE` (and its value) - the engine writes that profile when the
    server quits, and a benchmark's short prompts must not replace the user's calibrated one.  -> (config, removed)."""
    args = list(cfg.get("args") or [])
    out, i, removed = [], 0, False
    while i < len(args):
        if args[i] == "--expert-profile-save":
            i, removed = i + 2, True
        elif isinstance(args[i], str) and args[i].startswith("--expert-profile-save="):
            i, removed = i + 1, True
        else:
            out.append(args[i])
            i += 1
    return ({**cfg, "args": out} if removed else cfg), removed


def start_server(cfg_path: Path, port: int, gpu: str | None, log_path: Path):
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        cfg = None
    if isinstance(cfg, dict):
        safe, removed = without_profile_save(cfg)
        if removed:                       # a copy beside the log: the user's config file is not touched
            print("Note: the bench starts this model without --expert-profile-save, so its short prompts do not "
                  "overwrite your saved expert profile.", file=sys.stderr)
            tmp = log_path.with_suffix(".config.json")
            tmp.write_text(json.dumps(safe), encoding="utf-8")
            cfg_path = tmp
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path), "--port", str(port)]
    if gpu:
        cmd += ["--gpu", gpu]
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)


def wait_ready(base: str, key: str, proc, timeout: float, log=print) -> bool:
    t0, last = time.time(), 0
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        h = try_json(base, "/health", key, 5)
        if h and h.get("service") == "strata" and h.get("loaded"):
            return True
        if time.time() - last > 30:
            log(f"  still loading ({int(time.time() - t0)} s) ...")
            last = time.time()
        time.sleep(3)
    return False


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill()


def log_excerpt(path: Path, limit: int = 80) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    keep = re.compile(r"(?i)expert cache|slots|vram|ready|resident|kv |context|backend|gpu|prefill|mtp|spec|batch|version|hit")
    out = [x for x in lines if keep.search(x) and "[strata] done:" not in x]
    return out[:limit]


# --------------------------------------------------------------------------------------------------- the report
def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:limit].strip("-") or "unknown"


def gpu_slug(machine: dict) -> str:
    names = machine.get("gpu") or []
    if not names:
        return "cpu"
    # #2062: the cards one by one when they differ (rtx-5070-ti-plus-rtx-a4000), "2x-" only for identical ones
    groups: list[list] = []                              # [name, count], in the order the cards are listed
    for name in names:
        same = next((g for g in groups if g[0] == name), None)
        if same:
            same[1] += 1
        else:
            groups.append([name, 1])
    parts = []
    for name, count in groups:
        n = re.sub(r"(?i)\b(nvidia|geforce|amd|radeon|intel|graphics|corporation)\b", " ", name)
        parts.append(slug(n, 30) if count == 1 else f"{count}x-{slug(n, 30)}")
    return "-plus-".join(parts)[:80].strip("-")


def model_label(machine: dict) -> str:
    model, q = machine.get("model") or "model", machine.get("quant")
    return model if not q or slug(q) in slug(model, 200) else f"{model}-{q}"


def results_dir_name(machine: dict, date: str) -> str:
    return f"{date}-community-{gpu_slug(machine)}-{slug(model_label(machine), 40)}"


def unique_dir(parent: Path, name: str) -> Path:
    p, n = parent / name, 2
    while p.exists():
        p = parent / f"{name}-{n}"
        n += 1
    return p


def row(label, prompt_tokens, reused, gen, runs, pp, dec, ttft):
    return f"| {label} | {prompt_tokens} | {reused} | {gen} | {runs} | {pp} | {dec} | {ttft} |"


def results_table(out: dict) -> list[str]:
    lines = ["| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | "
             "Decode tok/s median and range | TTFT seconds median and range |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    c = out["chat"]
    if c:
        lines.append(row("short chat", fmt(c.get("prompt_tokens"), 0), "0", fmt(c.get("generated_tokens"), 0),
                         (c.get("ttft_s") or {}).get("n", 0), "not measured", fmt(c.get("decode_tps")), fmt(c.get("ttft_s"), 2)))
    for k, v in out["prompts"].items():
        lines.append(row(f"{k} prompt, fresh", fmt(v.get("prompt_tokens"), 0), fmt(v.get("reused_tokens"), 0), str(LONG_MAX_TOKENS) + " cap",
                         (v.get("ttft_s") or {}).get("n", 0), fmt(v.get("prompt_tps")), "not measured", fmt(v.get("ttft_s"), 2)))
    a = out["agent"]
    if a:
        n = len([s for s in a["sessions"] if len(s) > 1])
        lines.append(row("agent session, follow-up turns", fmt(a.get("last_turn_prompt_tokens"), 0) + " at the last turn",
                         "see below", f"up to {AGENT_MAX_TOKENS}", n, "see below", fmt(a.get("turn_decode_tps")),
                         fmt(a.get("turn_ttft_s"), 2)))
    return lines


def render_readme(res: dict) -> str:
    mc, out, p = res["machine"], res["results"], res["params"]
    gpu = " + ".join(mc.get("gpu") or []) or "CPU only"
    vram = f"{mc['gpu_vram_mib'] / 1024:.0f} GiB VRAM" if mc.get("gpu_vram_mib") else "VRAM not read"
    model = (model_label(mc) if mc.get("model") else mc.get("quant")) or "model not read"
    L = [f"# Community benchmark on {gpu}", "",
         f"Measured on {res['date']} with `tools/strata_bench.py` (suite {SCHEMA}, {p['runs']} runs per test, medians with ranges). "
         f"Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no "
         f"correctness check was run.", "",
         "## Hardware and software", "",
         f"- GPU and VRAM: {gpu}, {vram}",
         f"- CPU: {mc.get('cpu') or 'not read'} ({mc.get('cpu_cores') or '?'} cores, {mc.get('cpu_threads') or '?'} threads); "
         f"installed RAM: {mc.get('ram_gib') or '?'} GiB; storage: {mc.get('storage') or 'not read'}",
         f"- OS: {mc.get('os')}; GPU driver: {mc.get('gpu_driver') or 'not read'}; "
         f"CUDA (driver supports up to): {mc.get('cuda') or 'not read'}"
         + (f"; engine built for CUDA {(mc.get('engine_build') or {}).get('cuda')}" if (mc.get('engine_build') or {}).get('cuda') else ""),
         f"- Strata: project {mc.get('strata_version') or '?'}, commit {mc.get('strata_commit') or 'not a git checkout'}; "
         f"engine {mc.get('engine_version') or 'not read'}"]
    for g in (mc.get("gpu_detail") or []):
        if g:
            L.append("- GPU detail (after the run, near idle): " + ", ".join(f"{k}={v}" for k, v in g.items() if v is not None))
    L += ["- Background workloads: not recorded (GPU power limit/draw read above)", "",
         "## Model and configuration", "",
         f"- Model and quantization: {model}",
         f"- Context: {mc.get('context') or 'not read'} tokens; parallel slots: {mc.get('parallel_slots') or 1}",
         f"- Settings: the install's own configuration, unchanged{' (' + mc['config']['backend'] + ' backend)' if mc.get('config', {}).get('backend') else ''}; thinking "
         f"{'on' if p['think'] else 'off'} for the suite; temperature 0", ""]
    if mc.get("launch"):
        L += ["```text", mc["launch"], "```", ""]
    env = (mc.get("config") or {}).get("env")
    if env:
        L += ["Environment of the config: " + ", ".join(f"`{k}={v}`" for k, v in env.items()), ""]
    L += ["## Method", "",
          f"Run by `{p['command']}` against the HTTP API of a Strata server "
          f"({'started by the benchmark from the install configuration' if p['started_server'] else 'already running'}); model loading is not "
          f"timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation "
          f"cache cannot help the fresh-prompt tests.", "",
          f"- short chat: the three questions of `tools/ab_engine.py`, up to {CHAT_MAX_TOKENS} new tokens, {p['runs']} runs",
          f"- 4K / 16K{' / 32K' if 'prompt' in res['results'] and '32K' in res['results']['prompts'] else ''} prompts: generated word lists sized by a calibration request, "
          f"{LONG_MAX_TOKENS} new tokens, {p['runs']} fresh prompts each",
          f"- agent session: system prompt and four tools, then {AGENT_TURNS} turns that each add about {AGENT_WORDS} words of a project file "
          f"and continue from the model's own earlier answers (the way agent clients resend a conversation), {AGENT_MAX_TOKENS} new tokens per turn, "
          f"{p['runs']} sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)",
          "- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, "
          "latency over all requests", "",
          "## Results", ""]
    L += results_table(out)
    L += ["", "Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token."]
    ag = out["agent"]
    if ag and ag.get("sessions"):
        L += ["", "### Agent session (conversation cache)", "",
              f"Cache hit over the follow-up turns: {fmt(ag.get('hit_pct'), 1, ' %')}; earlier-turn tokens read again per session: "
              f"{fmt(ag.get('reread_tokens'), 0)} (earlier turns that the cache did not hold; the {fmt(ag.get('new_tokens'), 0)} new tokens of the files are read once); first turn (nothing cached) first token: {fmt(ag.get('first_turn_ttft_s'), 2, ' s')}.", "",
              "| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for d in ag["sessions"][0]:
            if d["ok"]:
                L.append(f"| {d['turn']} | {d['prompt_tokens']} | {d['reused_tokens']} | {d['read_tokens']} | {d['prompt_tps'] or '-'} | {d['ttft_s']} | {d['decode_tps'] or '-'} |")
    cl = out["clients"]
    if cl:
        L += ["", f"### Concurrency: {cl['clients']} clients", "",
              f"Total {fmt(cl.get('total_tps'), 1, ' tok/s')}; first token p50 {cl.get('ttft_p50_s')} s, p95 {cl.get('ttft_p95_s')} s; "
              f"request latency p50 {cl.get('latency_p50_s')} s, p95 {cl.get('latency_p95_s')} s; errors {cl.get('errors')}. "
              f"Server parallel slots: {mc.get('parallel_slots') or 1} (requests beyond that wait their turn)."]
    if out["skipped"] or out["failures"]:
        L += ["", "### Skipped and failed", ""]
        L += [f"- skipped {k}: {v}" for k, v in out["skipped"].items()]
        L += [f"- failed {f['where']}: {f['error']}" for f in out["failures"]]
    L += ["", "Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).", "",
          "## Correctness and limitations", "",
          "Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. "
          "Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. "
          f"Total time of the suite: {res['elapsed_s']:.0f} s.", ""]
    return "\n".join(L)


def compare_table(old: dict, new: dict) -> str:
    a, b = summary_metrics(old["results"]), summary_metrics(new["results"])
    rows = [f"{'':32s} {'old':>12s} {'new':>12s} {'change':>9s}"]
    for k in dict.fromkeys([*a, *b]):
        x, y = (a.get(k) or {}).get("median"), (b.get(k) or {}).get("median")
        if x is None and y is None:
            continue
        ch = f"{(y - x) / x * 100:+.1f}%" if x and y is not None else ""
        rows.append(f"{k:32s} {('-' if x is None else format(x, '.2f')):>12s} {('-' if y is None else format(y, '.2f')):>12s} {ch:>9s}")
    ho, hn = old["machine"], new["machine"]
    notes = [f"  {n}: {ho.get(f)} -> {hn.get(f)}" for n, f in (("GPU", "gpu"), ("engine", "engine_version"), ("model", "model"), ("quant", "quant"),
                                                              ("context", "context")) if ho.get(f) != hn.get(f)]
    return "\n".join(rows + (["", "differences:"] + notes if notes else []))


def load_results(path: Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def short_table(res: dict) -> str:
    m = summary_metrics(res["results"])
    lines = []
    for k, v in m.items():
        if v:
            lines.append(f"  {k:32s} {v['median']:>10.2f}   ({v['min']:.2f} - {v['max']:.2f}, n={v['n']})")
    return "\n".join(lines)


def write_results(res: dict, parent: Path, red, log_lines: list[str]) -> Path:
    d = unique_dir(parent, results_dir_name(res["machine"], res["date"]))
    d.mkdir(parents=True)
    clean = redact_obj(res, red)
    (d / "results.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False), encoding="utf-8")
    (d / "README.md").write_text(red(render_readme(res)), encoding="utf-8")
    if log_lines:
        (d / "engine-log-excerpt.txt").write_text(red("\n".join(log_lines)) + "\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------------------------------------- main
def server_base(host, port) -> str:
    """The address to talk to for a locally started server. The config's `host` is the interface the server
    binds - a specific IP, or a wildcard that is not connectable as written - so it cannot be polled as-is.
    An explicit `--url` bypasses this; there the caller knows the address."""
    host = host or "127.0.0.1"
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)     # wildcard binds accept loopback
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{authority}:{port}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure this Strata install and write a shareable report (see the top of this file).")
    ap.add_argument("--config", help="the model config to run (default: the newest strata-*.json of this folder)")
    ap.add_argument("--url", help="use the server already running at this address (http://127.0.0.1:8080) instead of starting one")
    ap.add_argument("--port", type=int, help="the port to start the server on (default: the config's)")
    ap.add_argument("--gpu", help="start the server on this GPU only (as the server's --gpu)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_KEY", ""), help="the server's API key, if it has one (or STRATA_KEY)")
    ap.add_argument("--runs", type=int, default=3, help="runs per test (default 3)")
    ap.add_argument("--clients", type=int, default=0, metavar="N", help="also send N requests at once")
    ap.add_argument("--long", action="store_true", help="also a prompt of about 32,768 tokens")
    ap.add_argument("--think", action="store_true", help="leave thinking on (the default suite turns it off)")
    ap.add_argument("--out", help="where to write the results folder (default: bench/results)")
    ap.add_argument("--compare", metavar="DIR", help="a results folder (or its results.json) to put beside this run")
    ap.add_argument("--compare-only", metavar="NEW", help="print --compare of two existing results folders and stop")
    ap.add_argument("--ready-timeout", type=float, default=1800, help="seconds to wait for the model to load")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    say = (lambda *x, **k: None) if a.quiet else (lambda *x, **k: print(*x, **k, flush=True))

    if a.compare_only:
        if not a.compare:
            ap.error("--compare-only NEW needs --compare OLD")
        print(compare_table(load_results(Path(a.compare)), load_results(Path(a.compare_only))))
        return 0

    cfg_path, cfg, proc, log_path = None, None, None, None
    if a.config or not a.url:
        if a.config:
            cfg_path = Path(a.config)
            if not cfg_path.is_absolute() and not cfg_path.exists():
                cfg_path = ROOT / a.config
        else:
            found = find_configs()
            if found:
                cfg_path = found[0]
                if len(found) > 1:
                    say(f"Several models are installed; using {cfg_path.name} (the one used last). Pick another with --config FILE.")
        if cfg_path is None or not cfg_path.is_file():
            print("No Strata model is set up in this folder: run START-HERE.bat (Windows) or ./setup.sh (Linux) first, "
                  "or give --url of a running server.", file=sys.stderr)
            return 2
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    key = a.api_key or ((cfg or {}).get("api_key") or "")
    if a.url:
        base = a.url.rstrip("/")
    else:
        port = a.port or int(cfg.get("port", 8080))
        base = server_base(cfg.get("host"), port)
    red = redactor(key=key)
    started = False
    try:
        if is_strata(base, key):
            say(f"Using the Strata server already running at {base}.")
        elif a.url:
            print(f"No Strata server answers at {base}.", file=sys.stderr)
            return 2
        else:
            if try_json(base, "/health", key, 3) is not None:
                print(f"Port {port} is used by something that is not Strata: pass --port N.", file=sys.stderr)
                return 2
            log_path = Path(tempfile.gettempdir()) / f"strata-bench-{os.getpid()}.log"
            say(f"Starting {cfg_path.name} on port {port} (loading takes 1-3 minutes; the PC may be slow meanwhile) ...")
            proc = start_server(cfg_path, port, a.gpu, log_path)
            started = True
            if not wait_ready(base, key, proc, a.ready_timeout, say):
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:] if log_path.exists() else []
                print("The server did not become ready. Last log lines:\n" + red("\n".join(tail)), file=sys.stderr)
                return 1
        t0 = time.time()
        pre = collect_machine(base, key, cfg, a.gpu)
        suite = Suite(base, key, pre.get("context"), max(1, a.runs), a.think, log=say, want_clients=a.clients, want_long=a.long)
        slots = pre.get("parallel_slots") or 1
        if a.clients and a.clients > slots:
            say(f"Note: the server runs {slots} request(s) at a time (--batch), so {a.clients} clients will partly wait their turn.")
        say("Running the suite (about 3-10 minutes) ...")
        out = suite.run()
        machine = collect_machine(base, key, cfg, a.gpu)      # again: VRAM and the cache after the work
        if not machine.get("gpu"):
            machine["gpu"] = pre.get("gpu")
        res = {"schema": SCHEMA, "tool": "tools/strata_bench.py", "date": datetime.date.today().isoformat(),
               "params": {"runs": suite.runs, "think": a.think, "clients": a.clients, "long": a.long, "started_server": started,
                          "command": "python tools/strata_bench.py" + "".join(
                              f" {x}" for x in (f"--clients {a.clients}" if a.clients else "", "--long" if a.long else "",
                                                "--think" if a.think else "", f"--runs {a.runs}" if a.runs != 3 else "") if x)},
               "elapsed_s": time.time() - t0, "machine": machine, "results": out}
        parent = Path(a.out) if a.out else ROOT / "bench" / "results"
        d = write_results(res, parent, red, log_excerpt(log_path) if log_path else [])
        say("\nResults (median, range over the runs):")
        say(red(short_table(res)))
        if out["failures"] or out["skipped"]:
            say(f"  ({len(out['failures'])} failed request(s), {len(out['skipped'])} skipped test(s): see the report)")
        if a.compare:
            say("\nCompared with " + a.compare + ":")
            say(red(compare_table(load_results(Path(a.compare)), redact_obj(res, lambda s: s))))
        try:
            shown = d.relative_to(ROOT)
        except ValueError:
            shown = d
        say(f"\nReport: {shown / 'README.md'}\nShare it: open a pull request that adds the folder {shown} (see docs/COMMUNITY_BENCHMARKS.md), "
            f"or paste README.md into a GitHub issue. Home folders, user and host names and IP addresses are already replaced; "
            f"read it once before you share it.")
        say("For results other people can compare with, run the standard benchmark: strata bench run (docs/BENCHMARK_PROTOCOL.md).")
        return 1 if out["failures"] and not any(out[k] for k in ("chat", "prompts", "agent")) else 0
    finally:
        if started:
            stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
