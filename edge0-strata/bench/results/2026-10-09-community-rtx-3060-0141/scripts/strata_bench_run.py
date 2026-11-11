"""Bounded local benchmark for the approved four-pack plan."""
import argparse
import csv
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import psutil

ROOT = Path(r"<workspace>\Strata")
OUT = ROOT.parent / "results/2026-10-09-rtx3060-0141"
SCRIPT = Path(__file__).resolve()
MODELS = ["q2_0", "iq2_xs", "iq3_xxs", "iq3_s"]
PORT = 18080
URL = f"http://127.0.0.1:{PORT}"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def get(path, timeout=3):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.load(r)


def kill_tree(p):
    try:
        children = psutil.Process(p.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if p.poll() is None:
        p.terminate()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def worker(source, dest, cancel=False):
    """Only public benchmark data crosses this unauthenticated loopback socket."""
    spec = json.loads(Path(source).read_text(encoding="utf-8"))
    body = spec["request"]
    chunks, texts, reason, calls = [], [], [], {}
    started = time.perf_counter()
    first = None
    finish = None
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                chunks.append(chunk)
                if chunk.get("error"):
                    raise RuntimeError(json.dumps(chunk["error"]))
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thinking = delta.get("reasoning_content") or ""
                    if text or thinking or delta.get("tool_calls"):
                        if first is None:
                            first = time.perf_counter() - started
                    texts.append(text)
                    reason.append(thinking)
                    for call in delta.get("tool_calls") or []:
                        i = call.get("index", 0)
                        item = calls.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if call.get("id"):
                            item["id"] = call["id"]
                        for key in ("name", "arguments"):
                            item["function"][key] += (call.get("function") or {}).get(key) or ""
                    finish = choice.get("finish_reason") or finish
                if cancel and first is not None:
                    break
        result = {"ok": True, "client_ttft_s": first, "client_elapsed_s": time.perf_counter() - started,
                  "text": "".join(texts), "reasoning": "".join(reason), "tool_calls": list(calls.values()),
                  "finish_reason": finish, "cancelled": cancel,
                  "usage": next((c["usage"] for c in reversed(chunks) if c.get("usage")), {}), "chunks": chunks}
    except urllib.error.HTTPError as exc:
        result = {"ok": False, "error_type": "HTTPError", "http_status": exc.code,
                  "error": exc.read(4096).decode("utf-8", errors="replace"),
                  "client_elapsed_s": time.perf_counter() - started}
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc),
                  "client_elapsed_s": time.perf_counter() - started}
    write(Path(dest), result)


class CaseError(Exception):
    pass


class Monitor:
    def __init__(self, runner):
        self.runner = runner
        self.done = threading.Event()
        self.low_since = None
        self.memory_stop = False
        self.last = {}
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        class PI(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in
                ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
                 "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                 ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]
        with (OUT / "telemetry.jsonl").open("a", encoding="utf-8") as f:
            while not self.done.is_set():
                t0 = time.monotonic()
                mem = psutil.virtual_memory()
                pi = PI()
                pi.cb = ctypes.sizeof(pi)
                ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), ctypes.sizeof(pi))
                commit = pi.CommitTotal / pi.CommitLimit if ok and pi.CommitLimit else None
                row = {"epoch": time.time(), "phase": self.runner.phase, "model": self.runner.model,
                       "case": self.runner.current, "ram_available_bytes": mem.available,
                       "ram_used_bytes": mem.used, "commit_fraction": commit,
                       "commit_total_bytes": pi.CommitTotal * pi.PageSize if ok else None,
                       "commit_limit_bytes": pi.CommitLimit * pi.PageSize if ok else None}
                rss = 0
                if self.runner.server and self.runner.server.poll() is None:
                    try:
                        proc = psutil.Process(self.runner.server.pid)
                        for p in [proc] + proc.children(recursive=True):
                            try:
                                rss += p.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                row["server_tree_rss_bytes"] = rss
                try:
                    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, creationflags=FLAGS)
                    if q.returncode == 0:
                        vals = q.stdout.strip().split(",")
                        keys = ["vram_mib", "vram_total_mib", "gpu_util_percent", "temperature_c", "graphics_mhz", "memory_mhz",
                                "gpu_power_w", "gpu_power_limit_w", "pcie_gen", "pcie_width"]
                        row.update({k: float(v.strip()) if v.strip().replace(".", "", 1).isdigit() else None for k, v in zip(keys, vals)})
                except (OSError, subprocess.TimeoutExpired):
                    row["gpu_sensor_unavailable"] = True
                io = psutil.disk_io_counters()
                if io:
                    row.update(disk_read_bytes=io.read_bytes, disk_write_bytes=io.write_bytes)
                low = mem.available < 3 * 1024**3 or (commit is not None and commit >= .90)
                if low and self.runner.server:
                    self.low_since = self.low_since or time.monotonic()
                    if time.monotonic() - self.low_since >= 5:
                        self.memory_stop = True
                else:
                    self.low_since = None
                self.last = row
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.done.wait(max(.05, 1 - (time.monotonic() - t0)))


class Runner:
    def __init__(self):
        self.clock = json.loads((OUT / "run-clock.json").read_text(encoding="utf-8"))
        self.start_epoch = self.clock["start_epoch"]
        self.hard = time.monotonic() + self.clock["measurement_stop_epoch"] - time.time()
        self.deadline = self.hard
        self.server = None
        self.log = None
        self.model = None
        self.phase = "preparation"
        self.current = None
        self.ctx = 0
        self.rows = []
        self.gates = {}
        self.monitor = Monitor(self)
        self.server_number = 0
        self.child_jobs = []
        self.tokenizer = None
        self.template = None
        self.filler = None
        self.report_data = {}
        self.active_worker = None
        self.weights_mode = "native"

    def event(self, kind, **data):
        row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "elapsed_s": round(time.time() - self.start_epoch, 1),
               "phase": self.phase, "model": self.model, "event": kind, **data}
        with (OUT / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write(OUT / "status.json", {**row, "completed_requests": len(self.rows), "server_pid": self.server.pid if self.server else None})
        print(json.dumps({k: row[k] for k in row if k not in ("engine", "metrics")}, ensure_ascii=False), flush=True)

    def budget(self, phase, seconds):
        self.phase = phase
        self.deadline = min(self.hard, time.monotonic() + seconds)
        self.event("phase_start", budget_seconds=seconds)

    def left(self):
        return max(0, self.deadline - time.monotonic())

    def stop(self):
        if self.server:
            kill_tree(self.server)
            self.server = None
        if self.log:
            self.log.close()
            self.log = None
        self.ctx = 0
        self.monitor.memory_stop = False
        self.monitor.low_since = None

    def idle(self, timeout=15):
        until = min(self.deadline, time.monotonic() + timeout)
        while time.monotonic() < until:
            try:
                m = get("/metrics")
                if m.get("live", {}).get("state") == "idle" and not m.get("live", {}).get("running", 0):
                    return m
            except (OSError, ValueError):
                pass
            if not self.server or self.server.poll() is not None:
                break
            time.sleep(.15)
        return None

    def config(self, model, ctx, parallel=1, mtp=True, cache=False):
        c = json.loads((ROOT / f"strata-{model}.json").read_text(encoding="utf-8-sig"))
        safe = {k: c[k] for k in ("exe", "cwd", "tokenizer", "args", "lib_dirs", "gpu") if k in c}
        args = list(safe["args"])
        if any("key" in a.lower() or "token" in a.lower() and a.startswith("--api") for a in args):
            raise CaseError("Unexpected credential argument in engine configuration")
        def setarg(key, value):
            if key in args:
                i = args.index(key)
                args[i+1] = str(value)
            else:
                args.extend([key, str(value)])
        setarg("--max-context", ctx)
        setarg("--kv", "int8")
        setarg("--kv-resident", min(ctx, 32768))
        setarg("--expert-cache", "auto")
        setarg("--prefill", "auto")
        setarg("--spec", 4)
        setarg("--spec-min-p", .5)
        if not mtp and "--mtp" in args:
            i = args.index("--mtp")
            del args[i:i+2]
        if cache:
            setarg("--conversation-cache-mib", 512)
            setarg("--conversation-cache-slots", 4)
        safe.update(args=args, model_name="strata", port=PORT, host="127.0.0.1", open_browser=False,
                    parallel=parallel, fit_max_tokens=False, lazy_load=False,
                    log=str(OUT / "servers" / f"{self.server_number:03d}-{model}-{ctx}-engine.log"))
        return safe

    def start(self, model, ctx=131072, parallel=1, mtp=True, cache=False, cpu_off=False):
        self.stop()
        if self.left() < 20:
            raise CaseError("insufficient phase time to start")
        self.model, self.ctx = model, ctx
        self.server_number += 1
        (OUT / "servers").mkdir(parents=True, exist_ok=True)
        c = self.config(model, ctx, parallel, mtp, cache)
        self.weights_mode = "resident" if "--resident-experts" in c["args"] else "mmap" if "--mmap-experts" in c["args"] else "native"
        path = OUT / "configs" / f"{self.server_number:03d}-{model}-{ctx}.json"
        write(path, c)
        self.log = (OUT / "servers" / f"{self.server_number:03d}-{model}-{ctx}-server.log").open("w", encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        if cpu_off:
            env["STRATA_PREFILL_CPU_SHARE"] = "0"
        else:
            env.pop("STRATA_PREFILL_CPU_SHARE", None)
        env.update(getattr(self, "engine_environment", {}))
        write(path.with_suffix(".environment.json"), {**getattr(self, "engine_environment", {}),
              **({"STRATA_PREFILL_CPU_SHARE": "0"} if cpu_off else {})})
        t0 = time.monotonic()
        self.server = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config", str(path),
            "--host", "127.0.0.1", "--port", str(PORT)], cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, creationflags=FLAGS)
        self.event("server_start", context=ctx, parallel_requested=parallel, mtp=mtp, cpu_off=cpu_off)
        until = min(self.deadline, t0 + 300)
        while time.monotonic() < until:
            if self.server.poll() is not None:
                self.event("server_exit", returncode=self.server.returncode)
                self.stop()
                raise CaseError("server exited during load")
            if self.monitor.memory_stop:
                self.event("startup_memory_stop", available_bytes=psutil.virtual_memory().available)
                self.stop()
                raise CaseError("startup memory threshold")
            try:
                health = get("/health")
                if health.get("loaded"):
                    if health.get("max_context") != ctx:
                        raise CaseError(f"context mismatch: requested {ctx}, actual {health.get('max_context')}")
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.5)
        else:
            self.stop()
            raise CaseError("startup deadline")
        self.event("server_ready", context=ctx, load_seconds=round(time.monotonic()-t0, 3))
        write(OUT / "servers" / f"{self.server_number:03d}-initial-metrics.json", get("/metrics"))
        self.perform(f"warmup-{self.server_number}", self.make(512, "warmup", 0), 512, group="warmup", max_tokens=128)
        if ctx == 262144 and psutil.virtual_memory().available < 3 * 1024**3:
            self.stop()
            raise CaseError("262K post-start free RAM below 3 GiB")

    def init_tokenizer(self):
        sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
        from strata_tokenizer import Tokenizer
        from serve.frontend import ChatTemplate, openai_to_messages
        self.openai_to_messages = openai_to_messages
        directory = ROOT.parent / "Strata-data/packs/iq2_xs/tokenizer"
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, number in vocab.items():
            tokens[number] = token
        self.tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
                                   json.loads((directory / "token_type.json").read_text(encoding="utf-8")))
        self.template = ChatTemplate(directory / "chat_template.jinja")
        code = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97)+1} + {i}) % 100003" for i in range(18000))
        self.filler = self.tokenizer.encode(code)
        self.event("tokenizer_ready", filler_tokens=len(self.filler))

    def body(self, text, maximum=256):
        return {"model": "strata", "messages": [{"role": "user", "content": text}], "temperature": 0,
                "reasoning_effort": "none", "max_tokens": maximum, "stream": True,
                "stream_options": {"include_usage": True}}

    def count(self, body):
        messages, tools, kwargs = self.openai_to_messages(body)
        return len(self.tokenizer.encode(self.template.render(messages, tools, **kwargs), parse_special=True))

    def make(self, target, language="code", rep=0, depth=50):
        name = f"{target}-{language}-{rep}-{depth}"
        cached = OUT / "prompts" / (name + ".json")
        if cached.exists():
            item = json.loads(cached.read_text(encoding="utf-8"))
            return item["request"]["messages"][0]["content"]
        prefix = f"Unique benchmark identifier {name}.\n"
        expected = f"amber-quartz-{target}-{rep}-{depth}"
        ending = "\nExplain the module in detail, covering testing, arithmetic, complexity and naming. Write at least 600 words."
        needle = f"\n# The code word to remember is {expected}.\n"
        if target >= 32768:
            ending = f"\nFirst print the remembered code word exactly. Then explain the module in detail in at least 600 words."
        else:
            needle = ""
        if language == "ja":
            base = self.tokenizer.encode("春の図書館では、地域の資料を整理し、利用者が必要な情報を探せるようにしています。貸出記録と蔵書目録は別々に管理します。\n" * 200)
            ending = "\nこの文章の業務について、手順、注意点、改善案を日本語で詳しく説明してください。600字以上書いてください。"
        elif language == "en":
            base = self.tokenizer.encode("A public library maintains a catalogue and a separate record of loans. Clear procedures help readers find reliable information.\n" * 300)
        else:
            base = self.filler
        overhead = self.count(self.body(prefix + needle + ending))
        n = max(1, target - overhead)
        best = None
        for _ in range(8):
            if n > len(base):
                raise CaseError("filler too short")
            if needle:
                cut = int(n * depth / 100)
                content = prefix + self.tokenizer.decode(base[:cut]) + needle + self.tokenizer.decode(base[cut:n]) + ending
            else:
                content = prefix + self.tokenizer.decode(base[:n]) + ending
            actual = self.count(self.body(content))
            if actual <= target and (best is None or actual > best[0]):
                best = actual, content
            if actual == target:
                break
            n += target - actual
        if best is None or best[0] < target - 20:
            raise CaseError("could not fit token target")
        write(cached, {"actual_tokens": best[0], "expected_code_word": expected if needle else None,
                       "depth_percent": depth, "request": self.body(best[1])})
        return best[1]

    def perform(self, label, text=None, target=None, group="speed", max_tokens=256, body=None,
                expected=None, cancel=False):
        if self.left() < 3:
            raise CaseError("phase deadline")
        if not self.server or self.server.poll() is not None:
            raise CaseError("no live benchmark server")
        body = body or self.body(text, max_tokens)
        local_count = self.count(body)
        base = OUT / "requests" / f"{len(self.rows):05d}-{self.model}-{label}"
        write(base.with_suffix(".json"), {"request": body, "expected_prompt_tokens": local_count})
        timeout = 900 if local_count > 131072 else 600 if local_count > 65536 else 180
        until = min(self.deadline, time.monotonic() + timeout)
        dest = base.with_suffix(".response.json")
        args = [sys.executable, str(SCRIPT), "--worker", str(base.with_suffix(".json")), str(dest)]
        if cancel:
            args.append("--cancel")
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=FLAGS)
        self.active_worker = p
        self.current = label
        self.event("request_start", label=label, group=group, local_prompt_tokens=local_count)
        failure = None
        while p.poll() is None:
            if self.monitor.memory_stop:
                failure = "memory_threshold"
                kill_tree(p)
                break
            if time.monotonic() >= until:
                failure = "deadline"
                kill_tree(p)
                break
            if self.server.poll() is not None:
                failure = "server_exited"
                kill_tree(p)
                break
            time.sleep(.1)
        self.active_worker = None
        row = {"model": self.model, "context": self.ctx, "label": label, "group": group, "target": target,
               "weights_mode": self.weights_mode,
               "engine_environment": dict(getattr(self, "engine_environment", {})),
               "local_prompt_tokens": local_count, "request_file": str(base.with_suffix(".json").relative_to(OUT)),
               "epoch": time.time(), "server_number": self.server_number}
        if failure:
            row.update(ok=False, error_type=failure)
        elif dest.exists():
            row.update(json.loads(dest.read_text(encoding="utf-8")))
        else:
            row.update(ok=False, error_type="worker_no_result", worker_returncode=p.returncode)
        metrics = self.idle()
        if metrics:
            hist = metrics.get("requests") or []
            row["engine"] = hist[0] if hist else {}
            row["engine_info"] = metrics.get("engine", {})
            write(base.with_suffix(".metrics.json"), metrics)
        else:
            row["server_idle_confirmed"] = False
            self.stop()
        eng = row.get("engine", {})
        if row.get("ok") and not cancel:
            actual = eng.get("prompt_total") or eng.get("prompt_tokens") or row.get("usage", {}).get("prompt_tokens")
            row["token_count_matches"] = actual == local_count
            if actual != local_count:
                row.update(ok=False, error_type="token_count_mismatch")
            if not (row.get("text") or row.get("tool_calls")):
                row.update(ok=False, error_type="empty_output")
        if expected is not None:
            row["expected"] = expected
            row["correct"] = expected in (row.get("text") or "")
        row.pop("chunks", None)
        self.rows.append(row)
        with (OUT / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.current = None
        self.event("request_done", label=label, group=group, ok=row.get("ok"), error_type=row.get("error_type"),
                   seconds=round(row.get("client_elapsed_s", 0), 2), correct=row.get("correct"), reused=eng.get("reused"))
        if failure == "memory_threshold":
            self.stop()
            raise CaseError("memory threshold")
        if failure:
            raise CaseError(failure)
        if not row.get("ok"):
            raise CaseError(row.get("error_type", "request failed"))
        return row

    def gate262(self, model):
        free = psutil.virtual_memory().available
        needed = (262144 - 131072) * 13 * 1056 + 3 * 1024**3
        allowed = self.monitor.last.get("commit_fraction") is not None and free >= needed
        self.gates[model] = {"available_bytes_before_switch": free, "required_available_bytes": needed,
                             "allowed": allowed, "status": "eligible" if allowed else "precheck_stopped"}
        write(OUT / "262k-gates.json", self.gates)
        self.event("262k_gate", **self.gates[model])
        return allowed

    def safe_block(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except CaseError as exc:
            self.event("case_block_stopped", reason=str(exc))
            self.stop()
            return None

    def speed_block(self, model, rep, lengths):
        self.start(model)
        for target in lengths:
            if self.left() < 5:
                break
            self.perform(f"speed-{target}-r{rep}", self.make(target, rep=rep), target,
                expected=f"amber-quartz-{target}-{rep}-50" if target >= 32768 else None)
        if rep == 1:
            allowed = self.gate262(model)
        else:
            allowed = self.gates.get(model, {}).get("allowed", False)
        if allowed and self.left() > 20:
            try:
                self.start(model, 262144)
                self.perform(f"speed-260000-r{rep}", self.make(260000, rep=rep), 260000,
                             expected=f"amber-quartz-260000-{rep}-50")
                self.gates[model]["status"] = "completed"
                if rep == 1:
                    self.perform("bridge-4096", self.make(4096, "bridge", 0), 4096, group="bridge")
            except CaseError as exc:
                self.gates[model].update(status="runtime_stopped", reason=str(exc), allowed=False)
                raise
            finally:
                write(OUT / "262k-gates.json", self.gates)

    def speed(self):
        order = list(MODELS)
        random.Random(141).shuffle(order)
        for model in order:
            self.safe_block(self.speed_block, model, 1, [4096, 128000, 128, 512, 1024, 32768, 64000])
        for rep in (2, 3):
            for model in order[rep-1:] + order[:rep-1]:
                if self.left() < 20:
                    return
                self.safe_block(self.speed_block, model, rep, [128000])
        for rep in (2, 3):
            for model in reversed(order) if rep == 2 else order:
                if self.left() < 20:
                    return
                def short():
                    self.start(model)
                    for target in [128, 512, 1024, 4096, 32768, 64000]:
                        self.perform(f"speed-{target}-r{rep}", self.make(target, rep=rep), target,
                            expected=f"amber-quartz-{target}-{rep}-50" if target >= 32768 else None)
                self.safe_block(short)
        for model in order:
            if self.left() < 30:
                return
            def languages():
                self.start(model)
                for lang in ("ja", "en"):
                    for rep in (1, 2, 3):
                        self.perform(f"language-{lang}-{rep}", self.make(4096, lang, rep), 4096, group="language")
            self.safe_block(languages)

    def recall(self):
        # Complete every model's 10% position before attempting the 90% position.
        for depth in (10, 90):
            for model in MODELS:
                if self.left() < 20:
                    return
                def run():
                    self.start(model)
                    self.perform(f"recall-128000-d{depth}", self.make(128000, "code", 9, depth), 128000,
                                 group="recall", max_tokens=40, expected=f"amber-quartz-128000-9-{depth}")
                self.safe_block(run)
        for model in MODELS[:2]:
            if self.left() < 20 or not self.gates.get(model, {}).get("allowed"):
                continue
            def run262():
                self.start(model, 262144)
                for depth in (10, 90):
                    self.perform(f"recall-260000-d{depth}", self.make(260000, "code", 9, depth), 260000,
                                 group="recall", max_tokens=40, expected=f"amber-quartz-260000-9-{depth}")
            self.safe_block(run262)

    def quality(self):
        ready = OUT / "eval-ready.json"
        if ready.exists() and json.loads(ready.read_text(encoding="utf-8")).get("ready"):
            from evalplus.sanitize import sanitize
            dataset = json.loads((OUT / "humaneval-subset.json").read_text(encoding="utf-8"))
            for model in MODELS:
                until = min(self.deadline, time.monotonic() + 525)
                def generate():
                    self.start(model)
                    (OUT / "code").mkdir(exist_ok=True)
                    with (OUT / "code" / f"{model}-samples.jsonl").open("a", encoding="utf-8") as f:
                        for tid in dataset["ids"]:
                            if time.monotonic() >= until:
                                break
                            old = self.deadline
                            self.deadline = min(old, until)
                            try:
                                p = dataset["problems"][tid]
                                row = self.perform("code-" + tid.replace("/", "-"),
                                    "Complete this Python function. Output valid Python code only, no explanations.\n\n" + p["prompt"],
                                    group="code", max_tokens=1024)
                                solution = sanitize(row["text"], p["entry_point"])
                                f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
                                f.flush()
                            finally:
                                self.deadline = old
                self.safe_block(generate)
            self.stop()
            self.evaluate(min(720, self.left()))
        else:
            self.event("code_evaluation_unavailable")
        # All assertions are fixed before any answers are seen.
        cases = [
            ("17×19を計算し、整数だけを答えてください。", "323"),
            ("123から58を引き、整数だけを答えてください。", "65"),
            ("文字列abcdeを逆順にし、その文字列だけを答えてください。", "edcba"),
            ("[8, 3, 5, 1]を昇順に並べ、JSON配列だけを答えてください。", [1, 3, 5, 8]),
            ("JSONだけで答えてください。キーnameの値を東京、キーcountの値を3にしてください。", {"name": "東京", "count": 3}),
            ("りんごが7個あり3個食べました。残りの個数を整数だけで答えてください。", "4"),
            ("次の文章に含まれる都市名だけを答えてください：私は昨日、大阪で友人と会いました。", "大阪"),
            ("A=5、B=9です。A+Bの結果を整数だけで答えてください。", "14"),
            ("東京、大阪、京都の3つをその順でJSON配列として答えてください。", ["東京", "大阪", "京都"]),
            ("英単語catを日本語のひらがなに訳し、訳語だけを答えてください。", "ねこ"),
            ("次の文字列の英小文字aの個数を整数だけで答えてください：banana", "3"),
            ("次のJSONのvalueだけを整数で答えてください：{\"value\":42,\"other\":7}", "42")]
        write(OUT / "japanese-cases.json", cases)
        tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                           "required": ["a", "b"], "additionalProperties": False}}}]
        pairs = [(2, 3), (17, 19), (-4, 9), (100, 23), (0, 7), (81, -12), (999, 1), (14, 28)]
        write(OUT / "tool-cases.json", {"schema": tools, "pairs": pairs})
        for model in MODELS:
            if self.left() < 20:
                break
            def small():
                self.start(model)
                for i, (prompt, expected) in enumerate(cases):
                    row = self.perform(f"japanese-{i}", prompt, group="japanese", max_tokens=128)
                    text = row["text"].strip().strip("`").strip()
                    try:
                        actual = json.loads(text) if isinstance(expected, (list, dict)) else text
                    except ValueError:
                        actual = None
                    self.event("japanese_grade", index=i, passed=actual == expected)
                    with (OUT / "grades.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"model": model, "group": "japanese", "index": i, "pass": actual == expected}) + "\n")
                for i, (a, b) in enumerate(pairs):
                    body = self.body(f"Use the add tool to add {a} and {b}. Do not compute it yourself.", 128)
                    body["tools"] = tools
                    row = self.perform(f"tool-{i}", body=body, group="tool")
                    calls = row["tool_calls"]
                    call = calls[0] if len(calls) == 1 else None
                    try:
                        valid = bool(call and call["function"]["name"] == "add" and json.loads(call["function"]["arguments"]) == {"a": a, "b": b})
                    except ValueError:
                        valid = False
                    if valid:
                        follow = self.body("", 128)
                        follow["tools"] = tools
                        follow["messages"] = body["messages"] + [{"role": "assistant", "content": row["text"], "tool_calls": calls},
                            {"role": "tool", "tool_call_id": call["id"], "content": str(a+b)},
                            {"role": "user", "content": "Reply with only the integer result returned by the tool."}]
                        reply = self.perform(f"tool-followup-{i}", body=follow, group="tool_followup")
                        valid = reply["text"].strip() == str(a+b)
                    with (OUT / "grades.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"model": model, "group": "tool", "index": i, "pass": valid}) + "\n")
            self.safe_block(small)

    def evaluate(self, seconds):
        if seconds < 5:
            return
        self.stop()
        log = (OUT / "eval.log").open("a", encoding="utf-8")
        p = subprocess.Popen(["wsl.exe", "-d", "Ubuntu", "--", "<home>/strata-bench-0141/.venv/bin/python",
            "<workspace>/tmp/strata_eval_subset.py", "--out",
            "<workspace>/results/2026-10-09-rtx3060-0141", "--limit-seconds", str(max(1, seconds-15))],
            stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
        try:
            p.wait(timeout=seconds)
            self.event("code_eval_exit", returncode=p.returncode)
        except subprocess.TimeoutExpired:
            kill_tree(p)
            # Stop only this benchmark's WSL evaluator, never the distribution or other sessions.
            subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "pkill", "-f", "^<home>/strata-bench-0141/.venv/bin/python <workspace>/tmp/strata_eval_subset.py --out"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=FLAGS, timeout=15)
            self.event("code_eval_deadline")
        finally:
            log.close()

    def features(self):
        for model in MODELS:
            if self.left() < 20:
                return
            def cache():
                self.start(model, cache=True)
                for rep in (1, 2, 3):
                    for pattern in ("continue", "restore"):
                        a = self.body(self.make(4096, "code", 20+rep+(10 if pattern == "restore" else 0)), 64)
                        r = self.perform(f"cache-{pattern}-{rep}-a", body=a, group="cache")
                        if pattern == "restore":
                            self.perform(f"cache-{pattern}-{rep}-b", self.make(4096, "code", 60+rep), 4096, group="cache", max_tokens=64)
                        a["messages"] += [{"role": "assistant", "content": r["text"]},
                                          {"role": "user", "content": "Continue with two concrete testing examples."}]
                        self.perform(f"cache-{pattern}-{rep}-return", body=a, group="cache", max_tokens=64)
            self.safe_block(cache)
        for model in MODELS:
            for off in (False, True):
                if self.left() < 20:
                    return
                def cpu():
                    self.start(model, cpu_off=off)
                    for target in (512, 1024):
                        for rep in (1, 2, 3):
                            self.perform(f"cpu-{'off' if off else 'default'}-{target}-{rep}", self.make(target, rep=40+rep), target, group="cpu")
                self.safe_block(cpu)
        for model in ("iq2_xs", "iq3_s"):
            for mtp in (True, False):
                if self.left() < 20:
                    return
                def draft():
                    self.start(model, mtp=mtp)
                    for rep in (1, 2, 3):
                        self.perform(f"mtp-{'on' if mtp else 'off'}-{rep}", self.make(4096, rep=50+rep), 4096, group="mtp", max_tokens=1024)
                self.safe_block(draft)

    def soak(self):
        self.start("iq2_xs")
        began = time.monotonic()
        i = 0
        while self.left() > 5:
            if i % 4 == 0:
                self.perform(f"soak-check-{i}", "Calculate 17 multiplied by 19. Reply only with the integer result.", group="soak", max_tokens=16, expected="323")
            else:
                target = (512, 32768, 128000)[(i-1) % 3]
                self.perform(f"soak-{i}", self.make(target, rep=100+i), target, group="soak", max_tokens=512)
            i += 1
        self.event("soak_complete", observed_seconds=time.monotonic()-began, requests=i)

    def concurrency(self):
        for model in MODELS:
            for parallel in (1, 2):
                if self.left() < 20:
                    return
                def burst():
                    self.start(model, 8192, parallel=parallel)
                    for clients in (1, 2, 4):
                        jobs = []
                        started = time.monotonic()
                        for client in range(clients):
                            base = OUT / "concurrency" / f"{model}-p{parallel}-c{clients}-i{client}"
                            write(base.with_suffix(".json"), {"request": self.body(self.make(512, rep=200+client), 128)})
                            dest = base.with_suffix(".response.json")
                            p = subprocess.Popen([sys.executable, str(SCRIPT), "--worker", str(base.with_suffix(".json")), str(dest)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=FLAGS)
                            self.child_jobs.append(p)
                            jobs.append((p, dest))
                        until = min(self.deadline, started+180)
                        while any(p.poll() is None for p, _ in jobs) and time.monotonic() < until and not self.monitor.memory_stop:
                            time.sleep(.1)
                        answers = []
                        for p, dest in jobs:
                            if p.poll() is None:
                                kill_tree(p)
                            if dest.exists():
                                answers.append(json.loads(dest.read_text(encoding="utf-8")))
                            self.child_jobs.remove(p)
                        metrics = self.idle()
                        write(OUT / "concurrency" / f"{model}-p{parallel}-c{clients}-burst.json",
                              {"model": model, "parallel_requested": parallel, "clients": clients,
                               "seconds": time.monotonic()-started, "answers": answers, "metrics": metrics})
                        self.event("burst_done", parallel_requested=parallel, clients=clients, results=len(answers))
                    if parallel == 1:
                        try:
                            self.perform("boundary-oversize", self.make(9000, rep=201), group="boundary", max_tokens=32)
                        except CaseError:
                            pass
                        if self.server:
                            self.perform("cancel", "Write a very long explanation of Python testing, at least 2000 words.", group="boundary", max_tokens=1024, cancel=True)
                            self.perform("after-cancel", "What is 17 times 19? Reply only with the integer.", group="boundary", max_tokens=16, expected="323")
                self.safe_block(burst)

    def inventory(self):
        write(OUT / "environment.json", {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "engine_build": json.loads((ROOT / "engine/BUILD.json").read_text(encoding="utf-8")), "python": sys.version,
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(False), "ram_bytes": psutil.virtual_memory().total,
            "disk_free_bytes": psutil.disk_usage(str(ROOT)).free, "start_clock": self.clock,
            "background_gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,power.limit,pcie.link.gen.max,pcie.link.width.max",
                "--format=csv"], text=True, creationflags=FLAGS)})
        artifacts = []
        seen = {}
        for model in MODELS:
            config = json.loads((ROOT / f"strata-{model}.json").read_text(encoding="utf-8-sig"))
            for p in (ROOT.parent / "Strata-data/models" / model.upper()).glob("*.gguf"):
                stat = p.stat()
                key = (stat.st_dev, stat.st_ino, stat.st_size)
                if key not in seen:
                    self.event("hash_start", file=p.name, bytes=stat.st_size)
                    h = hashlib.sha256()
                    with p.open("rb") as f:
                        for block in iter(lambda: f.read(16*1024*1024), b""):
                            if self.left() <= 0:
                                raise CaseError("preparation deadline during model hashes")
                            h.update(block)
                    seen[key] = h.hexdigest()
                artifacts.append({"model": model, "file": str(p), "bytes": stat.st_size, "sha256": seen[key]})
            directory = Path(config["tokenizer"])
            hashes = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")}
            artifacts.append({"model": model, "tokenizer_hashes": hashes})
        write(OUT / "model-manifest.json", artifacts)
        tokenizers = [r["tokenizer_hashes"] for r in artifacts if "tokenizer_hashes" in r]
        if any(t != tokenizers[0] for t in tokenizers):
            raise CaseError("the four model tokenizers differ")

    def report(self):
        self.phase = "report"
        self.stop()
        self.monitor.done.set()
        self.monitor.thread.join(timeout=5)
        groups = {}
        for row in self.rows:
            if row.get("group") not in ("speed", "language", "bridge", "recall"):
                continue
            key = (row["model"], row["group"], row.get("target"), row["context"])
            groups.setdefault(key, []).append(row)
        summaries = []
        for key, rows in sorted(groups.items(), key=str):
            model, group, target, context = key
            valid = [r for r in rows if r.get("ok") and not r.get("engine", {}).get("reused") and r.get("token_count_matches")]
            item = {"model": model, "group": group, "target": target, "context": context, "attempts": len(rows), "valid_fresh": len(valid),
                    "recall_pass": sum(r.get("correct", False) for r in rows if r.get("ok")),
                    "recall_trials": sum("correct" in r and r.get("ok", False) for r in rows)}
            for metric in ("prefill_tok_s", "decode_tok_s", "ttft_s", "total_s"):
                values = []
                for row in valid:
                    e = row.get("engine", {})
                    if metric == "prefill_tok_s":
                        value = (e.get("prompt_tokens", 0) - (e.get("reused") or 0))/(e["prompt_ms"]/1000) if e.get("prompt_ms") else None
                    elif metric == "decode_tok_s":
                        value = e["engine_generated"]/(e["decode_ms"]/1000) if e.get("decode_ms") and e.get("engine_generated") else None
                    else:
                        value = row.get("client_ttft_s" if metric == "ttft_s" else "client_elapsed_s")
                    if value is not None:
                        values.append(value)
                item[metric] = statistics.median(values) if values else None
                item[metric+"_min"] = min(values) if values else None
                item[metric+"_max"] = max(values) if values else None
            summaries.append(item)
        write(OUT / "summary.json", {"groups": summaries, "gates": self.gates, "requests": len(self.rows),
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "elapsed_minutes": (time.time()-self.start_epoch)/60})
        if summaries:
            with (OUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        lines = ["# RTX 3060 12 GB／64 GiB RAM：Strata 0.1.41 ベンチマーク", "", "計測中の自動集計。最終報告時に失敗・品質評価・資源ピークを確認する。", "",
                 "| モデル | 入力目標 | 有効な新規入力測定 | 入力 tok/s 中央値 | 生成 tok/s 中央値 | TTFT 秒 中央値 | 中央検索 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s in summaries:
            if s["group"] != "speed":
                continue
            def fmt(x):
                return "未測定" if x is None else f"{x:.2f}"
            lines.append(f"| {s['model']} | {s['target']} | {s['valid_fresh']} | {fmt(s['prefill_tok_s'])} | {fmt(s['decode_tok_s'])} | {fmt(s['ttft_s'])} | {s['recall_pass']}/{s['recall_trials']} |")
        lines += ["", "## 約262Kの実行判断", "", "判断の詳細は262k-gates.jsonを参照。事前停止は動作不可能やOOMを意味しない。", "",
                  "## 測定条件", "", "元のQwen3.8-Flash-Nextの4量子化。Windows推論、INT8 KV、常駐KV 32768、auto cache/prefill、MTP、reasoning off、temperature 0。", "",
                  "主測定は検索付き合成コード入力。公式表と同じ入力ではなく、GPU・エンジンも異なる。128K以下の最大コンテキストは131072、約262Kは262144。", "",
                  "生データはrequests.jsonl、監視はtelemetry.jsonl、起動ログはservers/、秘密を含まない実設定はconfigs/。", "",
                  "コード評価は先頭40問の部分集合。全164問のHumanEval+スコアではない。GPU電力はPC全体の消費電力ではない。"]
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.event("run_finished", requests=len(self.rows), elapsed_minutes=round((time.time()-self.start_epoch)/60, 2))

    def run(self):
        # A second monitor can kill this process tree if HTTP, GPU tools or native code hang.
        supervisor = subprocess.Popen([sys.executable, str(SCRIPT), "--watchdog", str(os.getpid()), str(self.clock["measurement_stop_epoch"]+300)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=FLAGS)
        try:
            self.deadline = min(self.hard, time.monotonic() + self.start_epoch + 115*60 - time.time())
            while not (OUT / "models-ready.json").exists():
                if self.left() <= 0:
                    raise CaseError("additional models not ready by preparation deadline")
                time.sleep(1)
            self.init_tokenizer()
            self.inventory()
            # Preparation smoke tests also establish counts and startup timings.
            for model in MODELS:
                self.start(model)
                self.perform("preparation-4096", self.make(4096, rep=0), 4096, group="preparation")
            self.stop()
            # Carry unused preparation time into the highest-priority measurement.
            preparation_credit = max(0, self.start_epoch + 90*60 - time.time())
            self.budget("speed", 190*60 + preparation_credit)
            self.speed()
            self.stop()
            self.budget("recall", 55*60)
            self.recall()
            self.stop()
            self.budget("quality", 55*60)
            self.quality()
            self.stop()
            self.budget("features", 30*60)
            self.features()
            self.stop()
            self.budget("soak", 30*60)
            self.safe_block(self.soak)
            self.stop()
            self.budget("concurrency", 15*60)
            self.concurrency()
            self.stop()
            self.budget("reserve", min(25*60, self.hard-time.monotonic()))
            # Prioritize completing generated code evaluation, then missing key speed runs.
            if (OUT / "code").exists():
                self.evaluate(min(720, self.left()))
            for model in MODELS:
                if self.left() < 30:
                    break
                missing = [t for t in (128000, 32768, 64000, 4096, 1024, 512, 128)
                    if sum(r.get("ok", False) for r in self.rows if r.get("group") == "speed" and r["model"] == model and r.get("target") == t) < 3]
                if missing:
                    def fill_missing():
                        self.start(model)
                        for target in missing:
                            completed = sum(r.get("ok", False) for r in self.rows if r.get("group") == "speed" and r["model"] == model and r.get("target") == target)
                            for rep in range(completed+1, 4):
                                self.perform(f"reserve-{target}-r{rep}", self.make(target, rep=rep), target,
                                    expected=f"amber-quartz-{target}-{rep}-50" if target >= 32768 else None)
                    self.safe_block(fill_missing)
            self.report()
        except Exception as exc:
            self.event("run_error", error_type=type(exc).__name__, reason=str(exc))
            self.report()
            raise
        finally:
            if self.active_worker and self.active_worker.poll() is None:
                kill_tree(self.active_worker)
            for job in self.child_jobs:
                kill_tree(job)
            self.stop()
            self.monitor.done.set()
            if supervisor.poll() is None:
                supervisor.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--cancel", action="store_true")
    ap.add_argument("--watchdog", nargs=2)
    a = ap.parse_args()
    if a.worker:
        return worker(*a.worker, cancel=a.cancel)
    if a.watchdog:
        pid, until = int(a.watchdog[0]), float(a.watchdog[1])
        while time.time() < until:
            if not psutil.pid_exists(pid):
                return
            time.sleep(1)
        try:
            p = psutil.Process(pid)
            for child in reversed(p.children(recursive=True)):
                if child.pid != os.getpid():
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            p.kill()
        except psutil.NoSuchProcess:
            pass
        return
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError("benchmark port is already occupied; existing service preserved")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.pid").open("w") as f:
        f.write(str(os.getpid()))
    Runner().run()


if __name__ == "__main__":
    main()
