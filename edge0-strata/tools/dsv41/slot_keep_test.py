#!/usr/bin/env python3
"""tools/dsv41/slot_keep_test.py - the kept conversations of strata-dsv41 (STRATA_DS4_SLOT_KEEP=1): a prompt that does
not continue the conversation in the model takes the kept one it does continue back, so an agent and its sub-agents
that take turns on one model read only their new tokens.

    python tools/dsv41/slot_keep_test.py --exe build/strata-dsv41 --pack /data/pack2.00 --ids-dir /tmp/slot-ids
           [--engine-args "--vram-experts 0 --max-context 20000"] [--turns 3] [--steps ref,old,pin,on,restart,cap]

The prompts are real text (this repo's docs), tokenized through the pack's tokenizer (tools/strata_tokenizer.py, needs
`regex`) the first time and kept as .ids files in --ids-dir, so a box without `regex` can run the test from a folder
made elsewhere (--make-only writes just the folder).  Random ids draw near-tie logits, where two greedy runs differ and
nothing can be compared; real text does not.

Runs of the engine's `--serve` protocol, greedy, spec=0 (the state's position is then exactly the answer's).  Two agents,
A (parent) and B (a sub-agent), each with its own conversation behind the SAME system prompt and tools, take turns
A1 B1 A2 B2 ... (the server marks the shared start with pin=N on every request):
  ref      STRATA_DS4_REUSE=0: every request is a fresh read of its whole prompt.  Its answers are the reference every
           other run must print, token for token.
  old      the engine as it was (no pin, no slots): each turn reads its whole prompt again, the other agent having
           taken the model.  Timings only.
  pin      the shared start marked, no slots: B goes back to the start A's state holds instead of reading it again.
  on       the shared start marked and the slots on: a turn that continues a kept conversation reads only its new tokens.
  restart  a new process on the folder `on` left (QUIT wrote the conversations there): each agent's next turn reads only
           its new tokens.
  cap      a folder capped below three conversations' size: it stays under the cap and the newest is kept.

Exit 0: every check passed.  1: a check failed (the reason is printed).  2: the engine did not start or answer, or the
prompts could not be made.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
META_MAGIC = 0x324154454D345344   # "DS4META2" (little endian): a slot's .meta, as src/dsv41/ds4_slotmeta.hpp writes it


# ------------------------------------------------------------------------------------------------- prompts
def corpus_text() -> str:
    """Real text for the prompts: the repo's README and docs, in a fixed order."""
    parts = []
    for p in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n\n".join(parts)


def text_tokens(tok, text: str, salt: str, n: int) -> list[int]:
    """`n` tokens of `text`, opened by `salt` (so two prompts differ from their first words), repeated if too short."""
    out: list[int] = []
    i = 0
    text = text[:max(4000, n * 7)]   # (a few chars a token; the loop below repeats it if that was too short)
    while len(out) < n:
        out += tok.encode(f"{salt} {i}.\n{text}\n", parse_special=False)
        i += 1
    return out[:n]


def make_tokenizer(pack: str):
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST   # noqa: E402
    t = Path(pack) / "tokenizer"
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((t / "token_type.json").read_text(encoding="utf-8")))


def prompts(ids_dir: Path, pack: str, n_sys: int, n_task: int, n_turn: int, turns: int) -> dict[str, list[int]]:
    """sys (the shared system prompt + tools), taskA, taskB, and turnA1.. turnB<turns> (each new user/tool text)."""
    want = {"sys": n_sys, "taskA": n_task, "taskB": n_task, **{f"turn{x}{i}": n_turn for x in "AB" for i in range(1, turns + 1)}}
    ids_dir.mkdir(parents=True, exist_ok=True)
    have: dict[str, list[int]] = {}
    tok = None
    text = None
    for name, n in want.items():
        f = ids_dir / f"{name}.ids"
        ids: list[int] | None = None
        if f.exists():
            ids = [int(x) for x in f.read_text().split()]
            if len(ids) != n:
                ids = None
        if ids is None:
            if tok is None:
                tok = make_tokenizer(pack)
                text = corpus_text()
            salt = {"sys": "System prompt and tools.", "taskA": "Task for agent A."}.get(name, f"Message {name}.")
            if name == "taskB":
                salt = "Task for agent B."
            # (the system prompt is cut from the corpus' start, the others from later in it, so they differ)
            body = text if name == "sys" else text[len(text) // 3:]
            ids = text_tokens(tok, body, salt, n)
            f.write_text(" ".join(str(x) for x in ids))
        have[name] = ids
    return have


# ------------------------------------------------------------------------------------------------- the .meta file
def meta_tokens(path: str) -> tuple[list[int], list[int]] | None:
    """A slot's .meta, as the engine writes it: the magic, its identity (length, bytes), the state's size, its tokens,
    its checkpoints (count, positions), the magic again.  None: not one of the engine's .meta files, or damaged."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        m0, il = struct.unpack_from("<QQ", data, 0)
        if m0 != META_MAGIC:
            return None
        _size, nt = struct.unpack_from("<QQ", data, 16 + il)
        toks = list(struct.unpack_from("<%di" % nt, data, 32 + il))
        (nc,) = struct.unpack_from("<Q", data, 32 + il + 4 * nt)
        cks = list(struct.unpack_from("<%dq" % nc, data, 40 + il + 4 * nt))
        (m1,) = struct.unpack_from("<Q", data, 40 + il + 4 * nt + 8 * nc)
    except (OSError, struct.error):
        return None
    return (toks, cks) if m1 == META_MAGIC else None


# ------------------------------------------------------------------------------------------------- the engine
class Engine:
    """One `strata-dsv41 --serve` process: its READY line, then `GEN` in, `T` / `DONE` out (a reader thread, so it
    runs the same on Windows)."""

    def __init__(self, exe: str, args: list[str], env: dict):
        self.exe, self.args, self.env = exe, args, env
        self.notes: list[str] = []
        self.lines: queue.Queue = queue.Queue()
        self.proc = None
        self.err_path = ""
        self.ready = ""

    def _pump(self) -> None:
        for raw in iter(self.proc.stdout.readline, b""):
            self.lines.put(raw.decode("utf-8", "replace").strip())
        self.lines.put(None)

    def _line(self, end: float) -> str:
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise RuntimeError("the engine did not answer in time")
            try:
                line = self.lines.get(timeout=min(left, 5.0))
            except queue.Empty:
                if self.proc.poll() is not None and self.lines.empty():
                    raise RuntimeError(f"the engine ended (exit {self.proc.returncode})")
                continue
            if line is None:
                raise RuntimeError("the engine's output ended")
            if line.startswith("NOTE "):
                self.notes.append(line[5:].strip())
            return line

    def start(self, timeout: float) -> None:
        env = dict(os.environ)
        env.update(self.env)
        fd, self.err_path = tempfile.mkstemp(prefix="ds4-slots-err-")
        err = os.fdopen(fd, "wb")
        self.proc = subprocess.Popen([self.exe, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     env=env, bufsize=0)
        err.close()
        threading.Thread(target=self._pump, daemon=True).start()
        end = time.monotonic() + timeout
        while True:
            line = self._line(end)
            if line.startswith("READY "):
                self.ready = line
                return
            if line.startswith("ERR "):
                raise RuntimeError("the engine refused to start: " + line)

    def request(self, prompt: list[int], max_new: int, pin: int = 0, timeout: float = 3600.0) -> dict:
        keys = f" pin={pin}" if pin else ""
        self.proc.stdin.write(f"GEN {max_new} spec=0{keys} {','.join(str(t) for t in prompt)}\n".encode())
        self.proc.stdin.flush()
        out = {"tokens": [], "reused": None, "prompt_len": len(prompt), "finish": None, "s": 0.0, "ttft_ms": 0.0}
        t0 = time.monotonic()
        end = t0 + timeout
        while True:
            line = self._line(end)
            if line.startswith("T "):
                out["tokens"].append(int(line[2:]))
            elif line.startswith("ERR "):
                out["error"] = line[4:]
                return out
            elif line.startswith("DONE "):
                f = line.split()
                out["generated"], out["prompt"], out["finish"] = int(f[1]), int(f[2]), f[5]
                out["ttft_ms"] = float(f[3])    # (request start to the first token's logits: the restore + the read)
                out["decode_ms"] = float(f[4])
                out["reused"] = int(f[8])
                out["s"] = time.monotonic() - t0
                return out

    def quit(self, timeout: float = 300.0) -> None:
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            if line.startswith("NOTE "):
                self.notes.append(line[5:].strip())
        try:
            self.proc.wait(timeout=max(1.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise RuntimeError("the engine did not end after QUIT")

    def stderr_lines(self) -> list[str]:
        if not self.err_path:
            return []
        try:
            with open(self.err_path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            os.unlink(self.err_path)
        except OSError:
            return []
        return lines


def run(exe: str, pack: str, engine_args: list[str], slots: str, keep: bool, steps: list[tuple[str, object]],
        max_new: int, pin: int = 0, reuse: bool = True, slot_gb: float | None = None, load_timeout: float = 3600.0):
    """One engine process; each step is (name, a function of the answers so far that gives its prompt).  `pin`: the
    shared start's length sent with every request (what the server sends); `reuse` False: every request is a fresh
    read of its whole prompt (STRATA_DS4_REUSE=0)."""
    env = {"STRATA_DS4_SLOT_KEEP": "1" if keep else "0", "STRATA_DS4_SLOT_DIR": slots, "STRATA_DS4_SLOT_MIN": "256",
           "STRATA_DS4_REUSE": "1" if reuse else "0"}
    if slot_gb is not None:
        env["STRATA_DS4_SLOT_GB"] = str(slot_gb)
    args = ["--pack", pack, "--dspark", "off", "--threads", "on", "--vision", "off", "--expert-profile", "none",
            *engine_args, "--serve"]
    eng = Engine(exe, args, env)
    t0 = time.monotonic()
    answers: dict[str, dict] = {}
    try:
        eng.start(load_timeout)
        load_s = time.monotonic() - t0
        for name, make in steps:
            prompt = make(answers)
            answers[name] = eng.request(prompt, max_new, pin=pin)
            if "error" in answers[name]:
                raise RuntimeError(f"{name}: {answers[name]['error']}")
    finally:
        if eng.proc is not None:
            try:
                eng.quit()
            except RuntimeError as e:
                print("   ", e, flush=True)
        err = eng.stderr_lines()
    for line in err:
        if line.startswith("ds4 slots:") and "failed" in line:
            raise RuntimeError("the engine: " + line)
    return answers, eng.notes, load_s


def main() -> int:
    ap = argparse.ArgumentParser(description="the kept conversations: two agents taking turns on one model")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--exe", default=str(ROOT / "build" / "strata-dsv41"))
    ap.add_argument("--ids-dir", default="", help="the prompts' .ids files (made from the docs when missing)")
    ap.add_argument("--make-only", action="store_true", help="write the .ids files and stop")
    ap.add_argument("--engine-args", default="", help="more engine flags, e.g. '--vram-experts 0 --max-context 20000'")
    ap.add_argument("--len-sys", type=int, default=6000, help="the shared system prompt + tools (tokens)")
    ap.add_argument("--len-task", type=int, default=300, help="each agent's own first message (tokens)")
    ap.add_argument("--len-turn", type=int, default=150, help="each later message (tokens)")
    ap.add_argument("--turns", type=int, default=3, help="turns per agent")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--steps", default="ref,old,pin,on,restart,cap",
                    help="ref: fresh reads (the reference), old: the engine as it was, pin: the shared start only, "
                         "on: slots + shared start, restart: a new process on the folder, cap: the folder's cap")
    ap.add_argument("--keep-dir", default="", help="keep the slot folders here (default: a temp dir, removed)")
    a_ = ap.parse_args()
    steps = set(a_.steps.split(","))
    ids_dir = Path(a_.ids_dir or tempfile.mkdtemp(prefix="slot-ids-"))
    try:
        P = prompts(ids_dir, a_.pack, a_.len_sys, a_.len_task, a_.len_turn, a_.turns + 1)
    except (OSError, ImportError, ValueError) as e:
        print(f"the prompts cannot be made: {e}", flush=True)
        return 2
    if a_.make_only:
        print(f"prompts written to {ids_dir}", flush=True)
        return 0
    if not os.path.exists(a_.exe):
        print(f"{a_.exe}: no engine - build it first", flush=True)
        return 2
    engine_args = shlex.split(a_.engine_args)
    N, max_new = a_.turns, a_.max_new
    PIN = len(P["sys"])
    ok = True

    def check(what: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
        ok = ok and cond

    # the turns A1 B1 A2 B2 ...; an agent's prompt is its previous prompt + the answer it got + the new message
    # (the answers of `ref` when given, so every arm sends the same prompts byte for byte)
    def build_steps(ref: dict | None, upto: int = N, start: int = 1, agents: str = "AB"):
        out = []
        for i in range(start, upto + 1):
            for x in agents:
                def make(ans, x=x, i=i):
                    src = ref if ref is not None else ans
                    prompt = P["sys"] + P["task" + x]
                    for j in range(1, i):
                        prompt = prompt + src[f"{x}{j}"]["tokens"] + P[f"turn{x}{j}"]
                    return prompt
                out.append((f"{x}{i}", make))
        return out

    def table(ans: dict, upto: int = N) -> tuple[float, float]:
        for i in range(1, upto + 1):
            for x in "AB":
                r = ans[f"{x}{i}"]
                print(f"        {x}{i}: prompt {r['prompt_len']:>6}  first token {r['ttft_ms'] / 1000:6.2f} s  "
                      f"reused {r['reused']:>6}", flush=True)
        wall = sum(ans[f"{x}{i}"]["s"] for i in range(1, upto + 1) for x in "AB")
        tt = sum(ans[f"{x}{i}"]["ttft_ms"] for i in range(1, upto + 1) for x in "AB") / 1000
        print(f"        {2 * upto} turns: wall {wall:.1f} s, time to first token (sum) {tt:.1f} s", flush=True)
        return wall, tt

    def same_tokens(name: str, ans: dict, ref: dict, upto: int = N) -> bool:
        bad = [f"{x}{i}" for i in range(1, upto + 1) for x in "AB" if ans[f"{x}{i}"]["tokens"] != ref[f"{x}{i}"]["tokens"]]
        for k in bad:
            print(f"        {k}: {ans[k]['tokens'][:8]} vs the reference's {ref[k]['tokens'][:8]}", flush=True)
        return not bad

    tmp = tempfile.mkdtemp(prefix="ds4-slots-") if not a_.keep_dir else a_.keep_dir
    os.makedirs(tmp, exist_ok=True)
    dir_on = os.path.join(tmp, "on")
    ref = None
    sums: dict[str, tuple[float, float]] = {}
    try:
        print(f"prompts: system+tools {len(P['sys'])} tok, task {len(P['taskA'])}, turn {len(P['turnA1'])}; "
              f"{N} turns per agent, {max_new} new tokens each, pin={PIN}", flush=True)
        need_ref = bool(steps & {"ref", "pin", "on", "restart", "cap"})
        if need_ref:
            print("ref: A1 B1 A2 B2 ... every request a fresh read of its whole prompt (STRATA_DS4_REUSE=0), shared "
                  "start marked", flush=True)
            ref, _, load0 = run(a_.exe, a_.pack, engine_args, os.path.join(tmp, "ref"), False,
                                build_steps(None, N + 1), max_new, pin=PIN, reuse=False)
            print(f"        load {load0:.0f} s", flush=True)
            sums["ref"] = table(ref)
            check("ref: nothing is reused", all(ref[f"{x}{i}"]["reused"] == 0 for i in range(1, N + 1) for x in "AB"))

        if "old" in steps:
            print("old: the engine as it was (no shared start, no slots): the other agent takes the model every turn",
                  flush=True)
            old, _, _ = run(a_.exe, a_.pack, engine_args, os.path.join(tmp, "old"), False,
                            build_steps(ref, N) if ref else build_steps(None, N), max_new)
            sums["old"] = table(old)
            if ref is not None:
                print(f"        tokens {'the same as' if same_tokens('old', old, ref) else 'DIFFER from'} the reference's "
                      "(informational: no shared-start split here)", flush=True)

        if "pin" in steps:
            print("pin: the shared start marked, no slots: a sub-agent goes back to the start the other left", flush=True)
            pn, _, _ = run(a_.exe, a_.pack, engine_args, os.path.join(tmp, "pin"), False, build_steps(ref, N), max_new,
                           pin=PIN)
            sums["pin"] = table(pn)
            check(f"pin: the sub-agent's first turn reads only its own task (reused {pn['B1']['reused']} of {PIN})",
                  pn["B1"]["reused"] == PIN)
            check(f"pin: every turn prints the reference's tokens ({2 * N} x {max_new})", same_tokens("pin", pn, ref))

        if "on" in steps:
            print("on: slots on (RAM) and the shared start marked", flush=True)
            on, notes, _ = run(a_.exe, a_.pack, engine_args, dir_on, True, build_steps(ref, N), max_new, pin=PIN)
            sums["on"] = table(on)
            check(f"on: the sub-agent's first turn reads only its own task (reused {on['B1']['reused']} of {PIN})",
                  on["B1"]["reused"] == PIN)
            check(f"on: every turn prints the reference's tokens ({2 * N} x {max_new})", same_tokens("on", on, ref))
            # a turn after the first continues a kept conversation: the whole earlier prompt + answer minus its last
            # token is reused (the state ends one token short of the answer's last one)
            full = True
            for i in range(2, N + 1):
                for x in "AB":
                    want = on[f"{x}{i}"]["prompt_len"] - len(P[f"turn{x}{i - 1}"]) - 1
                    got = on[f"{x}{i}"]["reused"]
                    if got != want:
                        full = False
                        print(f"        {x}{i}: reused {got}, expected {want}", flush=True)
            check("on: every later turn reuses its whole earlier conversation", full)
            if "old" in sums:
                check(f"on: faster to the first token than the old engine ({sums['on'][1]:.1f} s vs {sums['old'][1]:.1f} s)",
                      sums["on"][1] < sums["old"][1])
                print(f"        wall x{sums['old'][0] / max(sums['on'][0], 1e-9):.2f}, first token x"
                      f"{sums['old'][1] / max(sums['on'][1], 1e-9):.2f} against the old engine", flush=True)

        if "restart" in steps:
            bins = [f for f in os.listdir(dir_on) if f.startswith("slot-") and f.endswith(".bin")] if os.path.isdir(dir_on) else []
            check(f"restart: the folder holds both agents' conversations ({len(bins)} .bin)", len(bins) == 2)
            print("restart: each agent's next turn, a new process on the folder", flush=True)
            rs, notes2, _ = run(a_.exe, a_.pack, engine_args, dir_on, True,
                                [(f"{x}{N + 1}", build_steps(ref, N + 1, N + 1, x)[0][1]) for x in "AB"], max_new,
                                pin=PIN)
            for x in "AB":
                r = rs[f"{x}{N + 1}"]
                print(f"        {x}{N + 1}: prompt {r['prompt_len']:>6}  first token {r['ttft_ms'] / 1000:6.2f} s  reused {r['reused']:>6}",
                      flush=True)
                want = r["prompt_len"] - len(P[f"turn{x}{N}"]) - 1
                check(f"restart: {x}{N + 1} reuses its whole conversation ({r['reused']} of {want})", r["reused"] == want)
                check(f"restart: {x}{N + 1} prints the reference's tokens", r["tokens"] == ref[f"{x}{N + 1}"]["tokens"])

        if "cap" in steps:
            # three conversations (A, B and an unrelated C) cannot fit under A+B's bytes: the least recently used goes
            ca = os.path.join(tmp, "cap")
            seen = {}
            if os.path.isdir(dir_on):
                for fn in os.listdir(dir_on):
                    if fn.endswith(".bin"):
                        seen[fn] = os.path.getsize(os.path.join(dir_on, fn))
            if len(seen) == 2:
                cap_gb = sum(seen.values()) / 2**30 * 1.02
                c_prompt = P["taskB"] + P["taskA"] + P["sys"][:2000]
                print(f"cap: A, B, C into a folder capped at {cap_gb:.3f} GB (the bytes of A and B)", flush=True)
                run(a_.exe, a_.pack, engine_args, ca, True,
                    [("A1", build_steps(ref, 1)[0][1]), ("B1", build_steps(ref, 1)[1][1]),
                     ("C1", lambda _: c_prompt)], max_new, pin=PIN, slot_gb=cap_gb)
                total = sum(os.path.getsize(os.path.join(ca, f)) for f in os.listdir(ca) if f.endswith(".bin"))
                held = []
                a_start, b_start = P["sys"] + P["taskA"], P["sys"] + P["taskB"]
                for fn in sorted(os.listdir(ca)):
                    if fn.endswith(".meta"):
                        mt = meta_tokens(os.path.join(ca, fn))
                        if mt:
                            t = mt[0]
                            held.append("C" if t[:len(c_prompt)] == c_prompt else "A" if t[:len(a_start)] == a_start
                                        else "B" if t[:len(b_start)] == b_start else "?")
                print(f"        the folder holds {sorted(held)} ({total / 2**20:.1f} MiB of {cap_gb * 1024:.1f} MiB)", flush=True)
                check("cap: the folder stays under its cap", total <= cap_gb * 2**30)
                check("cap: the newest conversation (C) is kept", "C" in held)
                check("cap: the least recently used (A) is gone", "A" not in held)
            else:
                print("        (no two conversations in the loop's folder: the cap check needs the on step)", flush=True)
    except (RuntimeError, OSError) as e:
        print(f"FAIL: {e}", flush=True)
        ok = False
    finally:
        if not a_.keep_dir:
            shutil.rmtree(tmp, ignore_errors=True)
    if sums:
        print("summary (wall s, sum of first-token s): " + "  ".join(f"{k} {v[0]:.1f}/{v[1]:.1f}" for k, v in sums.items()),
              flush=True)
    print("PASS: the kept conversations read on from their state" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
