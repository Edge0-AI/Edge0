"""Decode A/B on coding-agent turns over this repository: arms restarted per round, per-turn ratios.

    python tools/agent_turns_bench.py strata-iq2_xs.json base: hcq8:STRATA_HC_Q8=2 [--rounds 2]

The workload is five simulated coding-agent sessions over this repository's own files (README, serve/, setup.py, the
expert cache, the hyper-connection kernels, the tokenizer): a system prompt, four tools, a task (three in English, two
in French), then tool calls whose results are the real files, at most 2,000 lines per read as agents' read tools
return them.  Every point right after a tool result is one turn the model continues - 23 turns from ~2.5K to ~100K
tokens with the repository as of 0.1.41, consecutive turns sharing their prefix as in a real session.  Nothing
personal and nothing downloaded: anyone with this checkout gets the same requests.

Arms are given as ab_engine.py takes them (NAME:key=value,...; `exe` replaces the engine, `xargs` appends
';'-separated engine arguments, any other key goes to the engine's environment).  The rounds mirror the arm order
(A B, B A, ...), each arm on a fresh server.  Sampling is the agent's (temperature 1, top-p 0.95, top-k 20, a fixed
seed), 256 new tokens per turn.  The result is the server's own decode tok/s per turn and, per arm, the median of the
per-turn ratios to the first arm with the number of turns it won: a whole-session median hides which turns moved.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_engine import wait_ready  # noqa: E402

SYSTEM = ("You are an autonomous coding agent working in the user's repository. You can read files, search the code, "
          "edit files and run commands with the tools below. Work step by step: read the code you need before changing "
          "it, make small and correct edits, keep the existing style, and explain briefly what you did. Never invent "
          "file contents: read them. When the task is done, give a short summary of the changes and how to test them.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file with a new one.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

# (name, task, [(tool, argument, what the assistant says before the call)])
SESSIONS = [
    ("server-option",
     "Add a `--max-queue N` option to the HTTP server that rejects new chat requests with HTTP 503 when N requests are "
     "already waiting, document it in the README's settings section, and add a test.",
     [("read_file", "README.md", "I'll start with the README to see how settings are documented."),
      ("grep", "add_argument", "Now let me find where the server parses its options."),
      ("read_file", "serve/server.py", "The options are in serve/server.py; reading it."),
      ("read_file", "serve/frontend.py", "Requests go through the frontend; let me read it too."),
      ("read_file", "docs/DETAILS.md", "Checking DETAILS.md for the existing API documentation style.")]),
    ("expert-cache",
     "Explain precisely how the expert cache decides which experts stay in VRAM during decode, then propose and "
     "implement a small improvement to its eviction policy with a measurement plan.",
     [("read_file", "docs/HOW_IT_WORKS.md", "Let me read the design overview first."),
      ("read_file", "include/strata/core/expert_cache.hpp", "Now the cache's interface."),
      ("read_file", "src/core/expert_cache.cpp", "And its implementation."),
      ("grep", "adapt_every", "Searching for the adaptive swap logic."),
      ("read_file", "src/core/expert_source.cpp", "The dispatch that consults the cache is in expert_source.cpp.")]),
    ("installer-fr",
     "Dans l'installateur, la détection de la RAM disponible se trompe parfois sous Windows quand un autre programme "
     "réserve beaucoup de mémoire. Trouve où elle est faite, corrige-la, ajoute un test et explique la correction en "
     "français.",
     [("read_file", "setup.py", "Je commence par lire l'installateur."),
      ("grep", "def .*ram", "Je cherche les fonctions qui mesurent la RAM."),
      ("read_file", "docs/INSTALL.md", "Je vérifie ce que la documentation promet sur la RAM."),
      ("read_file", "docs/MODELS.md", "Le choix du modèle dépend de la RAM ; je lis cette page aussi.")]),
    ("cuda-kernel",
     "The hyper-connection read kernels are latency bound on small batches. Read them, explain where the time goes, "
     "and write a faster variant for 1-4 tokens that keeps the same results bit for bit.",
     [("read_file", "include/strata/kernels/fused_gr.hpp", "First the kernel interface."),
      ("read_file", "src/kernels/cuda/fused_gr.cu", "Now the kernels themselves."),
      ("grep", "fused_gr_read_multi", "Where are they called from?"),
      ("read_file", "src/core/verify.cpp", "The verify window calls them; reading it.")]),
    ("tokenizer-fr",
     "Écris une suite de tests pytest pour le tokenizer Python (encodage, décodage, tokens spéciaux, texte multilingue, "
     "cas limites), puis corrige les bugs que les tests révèlent. Réponds en français.",
     [("read_file", "tools/strata_tokenizer.py", "Je lis d'abord le tokenizer."),
      ("grep", "strata_tokenizer", "Je cherche qui l'utilise."),
      ("read_file", "serve/frontend.py", "Le frontend l'appelle ; je le lis pour voir les usages réels."),
      ("read_file", "serve/chat_template.jinja", "Je regarde aussi le modèle de conversation."),
      ("read_file", "chat.py", "Et le client en ligne de commande.")]),
]


def tool_result(tool: str, arg: str) -> str:
    if tool == "read_file":
        lines = (ROOT / arg).read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(lines[:2000])
        return text + (f"\n... (truncated: lines 1-2000 of {len(lines)})" if len(lines) > 2000 else "")
    if tool == "grep":
        r = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", "-I", "-E", arg], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        lines = r.stdout.splitlines()
        return "\n".join(lines[:200]) + (f"\n... ({len(lines) - 200} more lines)" if len(lines) > 200 else "")
    raise ValueError(tool)


def turns() -> list[tuple[str, list[dict]]]:
    """[(tag, messages)]: one entry per point right after a tool result."""
    out = []
    for name, task, steps in SESSIONS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        for i, (tool, arg, before) in enumerate(steps):
            cid = f"call_{name}_{i}"
            args = {"path": arg} if tool == "read_file" else {"pattern": arg}
            msgs.append({"role": "assistant", "content": before,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": tool, "arguments": json.dumps(args)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(tool, arg)})
            out.append((f"{name}.{i}", [dict(m) for m in msgs]))
    return out


def post(port: int, messages: list[dict], max_tokens: int, seed: int) -> dict:
    body = {"messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "seed": seed, "reasoning_effort": "high"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=0, help="only the first N turns (0: all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log", default="agent_turns_bench.jsonl")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text())
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    work = turns()
    if a.max_turns:
        work = work[:a.max_turns]
    out = open(a.log, "a")
    res: dict = {n: {} for n, _ in arms}
    for rnd in range(a.rounds):
        for name, env in (arms if rnd % 2 == 0 else arms[::-1]):
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(Path(tempfile.gettempdir()) / f"agent_turns_{name}_r{rnd}.log")
            cpath = Path(tempfile.gettempdir()) / f"agent_turns_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=dict(os.environ, **e),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                t0 = time.time()
                wait_ready(a.port, proc)
                print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                for tag, msgs in work:
                    r = post(a.port, msgs, a.max_new, a.seed)
                    t = r.get("timings", {})
                    row = {"round": rnd, "arm": name, "turn": tag, "decode": t.get("predicted_per_second"),
                           "prompt_n": t.get("prompt_n"), "prompt": t.get("prompt_per_second"),
                           "n": r.get("usage", {}).get("completion_tokens")}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    res[name].setdefault(tag, []).append(row)
                    print(f"  {tag:16s} decode {row['decode']} tok/s  prompt {row['prompt_n']} tokens", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    first = arms[0][0]

    def per_turn(name: str, tag: str) -> float:
        v = [x["decode"] for x in res[name].get(tag, []) if x["decode"]]
        return statistics.mean(v) if v else float("nan")

    print(f"\nper-turn decode ratio to {first} (mean over rounds per turn, then the median over turns):")
    for name, _ in arms:
        med = statistics.median([per_turn(name, t) for t, _ in work])
        if name == first:
            print(f"  {name:10s} median decode {med:6.1f} tok/s")
            continue
        r = [per_turn(name, t) / per_turn(first, t) for t, _ in work]
        print(f"  {name:10s} median decode {med:6.1f} tok/s   x{statistics.median(r):.3f} (x{min(r):.2f} to "
              f"x{max(r):.2f}), {sum(x > 1 for x in r)}/{len(r)} turns faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
