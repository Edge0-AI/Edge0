"""Running one engine configuration and measuring it, for the tuner.

A *session* is what the tuner needs from the outside world: start an engine with these arguments and environment, run the
fixed request set, report numbers, stop. The real session (`WorkerSession`) starts Strata's own worker (``serve.server``)
on a **copy** of the user's run configuration and talks to it over HTTP, so the engine, the queueing and the prompt
handling are exactly the ones the user gets. The numbers come from the engine's own clock (the ``timings`` block of the
reply), not from this process's stopwatch. The stub session in the tests implements the same two methods.

Never invents a number: an engine that does not start, answers nothing or answers too little raises `StartFailed` or
returns a result whose ``status`` says why, and the tuner records that as the candidate's result.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from strata_control import loopback, tune_machine

# --------------------------------------------------------------------------- the contract


class StartFailed(RuntimeError):
    """The engine did not come up. ``status``: FAIL_BOOT | OOM | UNSUPPORTED | TIMEOUT."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class RequestSet:
    """The fixed work every run does, so two configurations are compared on the same requests."""

    warmups: int = 1
    decodes: int = 3
    decode_tokens: int = 200
    pp_tokens: int = 4096
    long_tokens: int = 0              # 0 = no long prompt


@dataclass
class RunResult:
    status: str = "PASS"              # PASS | FAIL_BOOT | OOM | UNSUPPORTED | TIMEOUT | NO_TOKENS
    reason: str = ""
    load_s: float | None = None
    decode_tok_s: float | None = None          # median of the measured decodes
    decode_runs: list[float] = field(default_factory=list)
    pp_tok_s: float | None = None              # the warm prompt (the second one; the first is cold)
    pp_cold_tok_s: float | None = None
    ttft_s: float | None = None
    pp_long_tok_s: float | None = None
    rss_peak_mib: float | None = None
    vram_used_mib: float | None = None         # this run's total graphics memory in use (all cards, absolute)
    vram_added_mib: float | None = None        # ... minus what was in use before the engine started
    min_avail_mib: float | None = None         # lowest free system RAM seen while the engine ran
    facts: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Engine(Protocol):
    facts: dict[str, Any]

    def measure(self, spec: RequestSet) -> RunResult: ...

    def decode_once(self, tune: dict[str, Any] | None = None, tokens: int = 200) -> float | None: ...


class Session(Protocol):
    def engine(self, args: list[str], env: dict[str, str | None], pre_start: Callable[[], None] | None = None) -> contextlib.AbstractContextManager[Engine]: ...

    def system(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- the engine log

_FACTS = [
    ("slots", re.compile(r"expert cache (\d+) slots"), int),
    ("workers", re.compile(r"(\d+) (?:expert-)?pool workers"), int),
    ("cpu_threads", re.compile(r"(\d+) CPU threads"), int),
    ("pcie_gbs", re.compile(r"PCIe probe: ([\d.]+) GB/s"), float),
    ("pcie_frac", re.compile(r"-> pcie_frac ([\d.]+)"), float),
    ("chunk", re.compile(r"prompt chunk auto: (\d+) tokens"), int),
    ("avail_gib", re.compile(r"available ([\d.]+) GiB \(MemAvailable"), float),
    ("loaded_gib", re.compile(r"loaded ([\d.]+) GiB at"), float),
]


def parse_facts(text: str) -> dict[str, Any]:
    """What the engine's own log says about the setup it chose (the last value of each fact wins)."""
    facts: dict[str, Any] = {}
    for key, rx, cast in _FACTS:
        found = rx.findall(text)
        if found:
            try:
                facts[key] = cast(found[-1])
            except ValueError:
                pass
    if "workers" not in facts and "cpu_threads" in facts:
        facts["workers"] = facts["cpu_threads"]
    low = text.lower()
    # the engine's low-RAM resident mode (--resident-experts) says so in its log; "resident experts" alone is the GPU cache's own wording
    facts["resident_ram"] = bool(re.search(r"resident ram mode|resident complement|resident [\d.]+ gib, pinned|low-ram mode", low))
    facts["hybrid_experts"] = "hybrid experts" in low or "hybrid:" in low
    facts["unknown_argument"] = bool(re.search(r"unknown argument|unrecognized (?:argument|option)", low))
    return facts


_CUDA_BUSY = re.compile(r"failed to initialize cuda|no cuda-capable device|cuda-capable device\(s\) is/are busy|"
                        r"cuda driver version is insufficient|hipinit.{0,40}(?:fail|error)|no hip-capable device", re.I)


def failure_kind(text: str, returncode: int | None) -> tuple[str, str]:
    """Classify an engine that ended before it was ready: (status, one plain line). The engine's informational
    `memory guard: limit ... available ...` line is not a failure and never decides the status (#3502)."""
    low = text.lower()
    lines = [ln.strip() for ln in text.splitlines()
             if re.search(r"unknown argument|unrecognized|invalid value|out of memory|cannot|failed|error|refus|killed", ln, re.I)
             and not re.search(r"memory guard: (?:limit|the available)", ln, re.I)]
    reason = " | ".join(lines[:3])[:300]
    if re.search(r"unknown argument|unrecognized (argument|option)", low):
        return "UNSUPPORTED", reason or "the engine does not accept one of the settings"
    if _CUDA_BUSY.search(text):
        return "FAIL_BOOT", ("the graphics card could not be opened by the measuring engine; it is probably in use by the "
                             "loaded model. Unload the model and tune again (" + reason + ")")[:400]
    if re.search(r"out of memory|memory guard: (?:refus|not enough|too)|cannot allocate|std::bad_alloc|\boom\b|(?:^|\n)killed(?:\r?\n|$)", low) or returncode in (-9, 137, 3221226505):
        return "OOM", reason or "the engine ran out of memory"
    return "FAIL_BOOT", reason or f"the engine stopped while starting (exit {returncode})"


# --------------------------------------------------------------------------- prompts

_SENTENCES = (
    "The harbour town woke slowly, with fishing boats knocking against the wooden piers.",
    "A small team measured how long each step of the process took and wrote the numbers in a notebook.",
    "Rain fell on the old railway station while a clerk sorted letters by hand.",
    "The recipe asked for two cups of flour, a pinch of salt and patience.",
    "In the library, a quiet reader turned pages while the clock in the hall struck three.",
    "Engineers compared the two bridges and noted how each handled wind and weight.",
    "The orchard produced more apples than the village could eat, so the surplus went to the city market.",
    "She packed a map, a flask of tea and a spare pair of boots for the long walk north.",
    "Each morning the baker lit the oven before dawn and left the shop door unlocked for the milkman.",
    "A careful gardener keeps a record of when each seed is planted and when it first sprouts.",
    "The museum's new wing displayed instruments that once guided ships across open water.",
    "Traffic slowed near the tunnel entrance, and the radio suggested taking the coast road instead.",
    "Two neighbours argued kindly over whose tomato plants had grown taller that summer.",
    "The orchestra tuned together, a single note spreading through the hall until the conductor raised a hand.",
    "He wrote down every question he had about the machine before opening its case.",
    "A river of lanterns floated past the bridge as the festival reached its final evening.",
)


def make_prompt(tokens: int, salt: str, chars_per_token: float = 4.4) -> str:
    """A prompt of about ``tokens`` tokens that no earlier run has seen: its first line is the salt, so a cached prefix is
    never reused, and the rest is shuffled prose."""
    rng = random.Random(salt)
    target = int(tokens * chars_per_token)
    parts = [f"Reference {salt}."]
    size = len(parts[0])
    while size < target:
        s = rng.choice(_SENTENCES)
        parts.append(s)
        size += len(s) + 1
    parts.append("Now answer in one short sentence: what is the main subject of the text above?")
    return " ".join(parts)


DECODE_PROMPTS = (
    "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Keep going without stopping.",
    "Explain step by step how a bicycle works, covering the frame, wheels, gears, chain and brakes in depth.",
)


# --------------------------------------------------------------------------- a flock for the box locks

class FileLock:
    """An exclusive advisory lock on a file, held around each engine run (the lent boxes' /data/gpu.lock rule)."""

    def __init__(self, path: str | os.PathLike[str], wait_s: float = 5400.0, log: Callable[[str], None] = lambda s: None) -> None:
        self.path = str(path)
        self.wait_s = wait_s
        self.log = log
        self._fh = None

    def __enter__(self) -> "FileLock":
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        if os.name != "nt" and threading.current_thread() is threading.main_thread():
            return self._blocking()
        return self._polling()

    def _blocking(self) -> "FileLock":
        """A blocking flock(2), queued in the kernel like `flock -x` (so the other streams' waiters are served in order and
        a polling tuner is not starved). The wait limit is a timer, not a loop."""
        import fcntl
        import signal

        def on_alarm(signum, frame):  # noqa: ANN001
            raise TimeoutError

        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self
        except OSError:
            self.log(f"waiting for the lock {self.path}")
        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, self.wait_s)
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            return self
        except TimeoutError:
            self._fh.close()
            raise StartFailed("TIMEOUT", f"the lock {self.path} was not free within {int(self.wait_s)} s") from None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def _polling(self) -> "FileLock":
        deadline = time.time() + self.wait_s
        told = False
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() > deadline:
                    self._fh.close()
                    raise StartFailed("TIMEOUT", f"the lock {self.path} was not free within {int(self.wait_s)} s")
                if not told:
                    self.log(f"waiting for the lock {self.path}")
                    told = True
                time.sleep(2.0)

    def __exit__(self, *exc: object) -> None:
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            if self._fh:
                self._fh.close()


# --------------------------------------------------------------------------- the real session

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WorkerEngine:
    """One running worker and the requests that measure it."""

    def __init__(self, session: "WorkerSession", proc: subprocess.Popen, port: int, key: str, log_files: list[Path],
                 load_s: float, vram_before: int | None, log: Callable[[str], None]) -> None:
        self.s = session
        self.proc = proc
        self.port = port
        self.key = key
        self.log_files = log_files
        self.load_s = load_s
        self.vram_before = vram_before
        self.log = log
        self.model = "strata-local"
        self.min_avail: float | None = None
        self.facts: dict[str, Any] = parse_facts(self.log_text())
        self.salt_n = 0
        self._model_id()

    # ---- helpers
    def log_text(self) -> str:
        out = []
        for f in self.log_files:
            try:
                out.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return "\n".join(out)

    def _model_id(self) -> None:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/models", headers=self._h())
            with loopback.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            self.model = ((data.get("data") or [{}])[0]).get("id") or self.model
        except Exception:  # noqa: BLE001
            pass

    def _h(self) -> dict[str, str]:
        return {"content-type": "application/json", "authorization": f"Bearer {self.key}"}

    def _sample_mem(self) -> None:
        a = tune_machine.memory().get("available_mib")
        if a is not None:
            self.min_avail = a if self.min_avail is None else min(self.min_avail, a)

    def chat(self, prompt: str, max_tokens: int, tune: dict[str, Any] | None = None, timeout: float = 1800.0) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "stream": False, "max_tokens": max_tokens, "temperature": 0.0,
                                "messages": [{"role": "user", "content": prompt}], "strata_checkpoint": False}
        if tune:
            body["strata_tune"] = tune
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=self._h(), method="POST")
        try:
            with loopback.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise StartFailed("NO_TOKENS", f"the worker refused the request ({e.code}): {e.read()[:200]!r}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            code = self.proc.poll()
            status, why = failure_kind(self.log_text()[-4000:], code) if code is not None else ("TIMEOUT", str(e))
            raise StartFailed(status, f"the engine did not answer: {why}") from None
        self._sample_mem()
        return data

    # ---- measurements
    def decode_once(self, tune: dict[str, Any] | None = None, tokens: int = 200) -> float | None:
        for prompt in DECODE_PROMPTS:
            data = self.chat(prompt, tokens, tune)
            t = data.get("timings") or {}
            n, rate = t.get("predicted_n"), t.get("predicted_per_second")
            if n and rate and n >= max(32, tokens // 2):
                return float(rate)
        return None

    def prompt_once(self, tokens: int) -> dict[str, float | None]:
        self.salt_n += 1
        salt = secrets.token_hex(6)
        text = make_prompt(tokens, salt, self.s.chars_per_token)
        data = self.chat(text, 8)
        t = data.get("timings") or {}
        usage = data.get("usage") or {}
        pt = usage.get("prompt_tokens")
        if isinstance(pt, int) and pt > 64 and not self.s.calibrated:
            self.s.chars_per_token = max(2.0, min(8.0, self.s.chars_per_token * tokens / pt))
            self.s.calibrated = True
        rate = t.get("prompt_per_second")
        # a prompt that was partly cached is not a prompt-speed measurement
        if t.get("cache_n") and t.get("prompt_n") and t["cache_n"] > 0.1 * (t["prompt_n"] + t["cache_n"]):
            rate = None
        ms = t.get("prompt_ms")
        return {"rate": float(rate) if rate else None, "ttft_s": float(ms) / 1000.0 if ms else None,
                "tokens": pt if isinstance(pt, int) else None}

    def measure(self, spec: RequestSet) -> RunResult:
        t0 = time.time()
        res = RunResult(load_s=self.load_s, facts=dict(self.facts))
        for _ in range(spec.warmups):
            self.decode_once(None, 64)
        runs: list[float] = []
        for _ in range(spec.decodes):
            r = self.decode_once(None, spec.decode_tokens)
            if r:
                runs.append(r)
        res.decode_runs = [round(x, 3) for x in runs]
        if not runs:
            res.status, res.reason = "NO_TOKENS", "the engine produced no measurable decode"
        else:
            res.decode_tok_s = round(statistics.median(runs), 3)
        if spec.pp_tokens:
            cold = self.prompt_once(spec.pp_tokens)
            warm = self.prompt_once(spec.pp_tokens)
            res.pp_cold_tok_s = cold["rate"]
            res.pp_tok_s, res.ttft_s = warm["rate"], warm["ttft_s"]
        if spec.long_tokens:
            res.pp_long_tok_s = self.prompt_once(spec.long_tokens)["rate"]
        res.rss_peak_mib = tune_engine_peak(self.proc.pid)
        used = tune_machine.gpu_used_mib()
        if used is not None:
            res.vram_used_mib = used
            if self.vram_before is not None:
                res.vram_added_mib = used - self.vram_before
        res.min_avail_mib = self.min_avail
        res.facts = dict(parse_facts(self.log_text()))
        res.wall_s = round(time.time() - t0, 1)
        return res


def tune_engine_peak(pid: int) -> float | None:
    try:
        from strata_control.tune_engine import process_tree_peak_mib
        return process_tree_peak_mib(pid)
    except Exception:  # noqa: BLE001
        return None


class WorkerSession:
    """Starts ``serve.server`` on a copy of the run configuration for each configuration the tuner asks for."""

    def __init__(self, *, base_config: dict[str, Any], python: str | os.PathLike[str] | None = None,
                 serve_dir: str | os.PathLike[str], workdir: str | os.PathLike[str] | None = None,
                 boot_timeout_s: float = 900.0, lock: Callable[[], contextlib.AbstractContextManager] | None = None,
                 log: Callable[[str], None] = lambda s: None, extra_env: dict[str, str] | None = None) -> None:
        self.base = json.loads(json.dumps(base_config))
        self.python = str(python or sys.executable)
        self.serve_dir = Path(serve_dir)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="strata-tune-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.boot_timeout_s = boot_timeout_s
        self.lock = lock
        self.log = log
        self.extra_env = dict(extra_env or {})
        self.chars_per_token = 4.4
        self.calibrated = False
        self.runs = 0

    def system(self) -> dict[str, Any]:
        return {"memory": tune_machine.memory(), "gpus": tune_machine.gpus(), "cpu": tune_machine.cpu()}

    @contextlib.contextmanager
    def engine(self, args: list[str], env: dict[str, str | None], pre_start: Callable[[], None] | None = None) -> Iterator[WorkerEngine]:
        with contextlib.ExitStack() as stack:
            if self.lock is not None:
                stack.enter_context(self.lock())
            if pre_start is not None:
                pre_start()                      # the memory and quiet checks run once the lock is ours, right before the start
            self.runs += 1
            n = self.runs
            cfg = json.loads(json.dumps(self.base))
            cfg["args"] = list(args)
            cenv = dict(cfg.get("env") or {})
            for k, v in env.items():
                if v is None:
                    cenv.pop(k, None)
                else:
                    cenv[k] = str(v)
            cfg["env"] = cenv
            key = secrets.token_hex(16)
            port = _free_port()
            engine_log = self.workdir / f"engine-{n}.log"
            worker_log = self.workdir / f"worker-{n}.log"
            engine_log.write_text("")
            cfg.update({"api_key": key, "port": port, "host": "127.0.0.1", "open_browser": False, "log": str(engine_log)})
            cfg_path = self.workdir / f"run-{n}.json"
            cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            spawn_env = dict(os.environ)
            spawn_env.update(self.extra_env)
            for k, v in env.items():
                if v is None:
                    spawn_env.pop(k, None)
            spawn_env["PYTHONUNBUFFERED"] = "1"
            spawn_env["STRATA_NO_PRESET"] = "1"          # measure the configuration as given, not a saved tuning on top
            before = tune_machine.gpu_used_mib()
            wl = open(worker_log, "w")
            argv = [self.python, "-m", "serve.server", "--engine", "strata", "--config", str(cfg_path),
                    "--host", "127.0.0.1", "--port", str(port)]
            t0 = time.time()
            try:
                from serve import lifeline
                proc = lifeline.popen(argv, "worker", cwd=str(self.serve_dir), env=spawn_env, stdout=wl,
                                      stderr=subprocess.STDOUT)
            except OSError as e:
                wl.close()
                raise StartFailed("FAIL_BOOT", f"the worker could not be started: {e}") from None
            try:
                deadline = t0 + self.boot_timeout_s
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        text = (engine_log.read_text(errors="replace") if engine_log.exists() else "") + "\n" + \
                               worker_log.read_text(errors="replace")
                        status, why = failure_kind(text, proc.returncode)
                        raise StartFailed(status, why)
                    try:
                        req = urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                                     headers={"authorization": f"Bearer {key}"})
                        with loopback.urlopen(req, timeout=2) as r:
                            if r.status == 200:
                                ready = True
                                break
                    except (urllib.error.URLError, OSError):
                        time.sleep(0.5)
                if not ready:
                    raise StartFailed("TIMEOUT", f"the engine was not ready within {int(self.boot_timeout_s)} s")
                eng = WorkerEngine(self, proc, port, key, [engine_log, worker_log], time.time() - t0, before, self.log)
                yield eng
            finally:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=30)
                finally:
                    wl.close()
                    # the engine gives its graphics memory back a moment after it exits
                    time.sleep(3.0)
