"""Measure one engine configuration, for the tuning job.

The tuning job (`strata_control/tuning.py`) searches configurations and decides from numbers. This
module is where those numbers come from: it boots the engine with one candidate's flags, sends the
declared workload, reads the engine's own output, and returns measurements.

It is deliberately separate from the job, and it is deliberately paranoid:

* **It never invents a number.** If the engine does not start, produces no tokens, or finishes for a
  reason other than a stop or a length, this raises with the engine's own words. The job records
  that as a failed candidate — which is a real result — rather than as a slow one.
* **It measures what the engine reports, not what the wall clock suggests.** The engine prints its
  own prompt and decode milliseconds, so a measurement does not include this process's scheduling.
  Client-observed timings are recorded separately and labelled, because they answer a different
  question.
* **It runs the declared workload every time**, with a fixed seed, so two candidates are compared on
  the same work. A "faster" configuration that quietly did less is the classic tuning failure.

The production engine protocol this reads (see `src/program/generate.cpp`):

    READY <max_context> stop
    PP <processed> <total> <ms> [<tok_s>]      prompt progress
    T <token_id>                               one generated token
    DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts_acc> <drafts_off> ...

`finish` is `stop`, `length`, `cancel` or `nonfinite`. A `cancel` or `nonfinite` finish is not a
measurement, and neither is a run whose `produced` is zero.
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: A workload is fixed for the whole sweep: the same prompt, the same number of tokens to generate,
#: the same sampling. Changing any of these between candidates invalidates the comparison.
DEFAULT_WORKLOAD = {
    "prompt_tokens": 512,
    "max_new": 128,
    "seed": 1,
    "temperature": 0.0,      # greedy, so the work is deterministic
}


class MeasurementError(RuntimeError):
    """The candidate could not be measured. The message is the engine's own reason."""


@dataclass
class EngineMeasurement:
    """One candidate's measurements, with the units and the source of each.

    `source` distinguishes what the engine reported from what this process observed, because they
    are not interchangeable: the engine's `prompt_ms`/`decode_ms` exclude process startup and
    scheduling, while `client_ttft_ms` includes everything a user would wait for.
    """

    tokens_per_s: float
    #: Peak resident memory of the measured process tree, in MiB. Tuning that gets faster by
    #: spilling or by thrashing is not an improvement, and the checklist asks for memory to travel
    #: with the speed number so the trade is visible rather than hidden.
    peak_memory_mib: float | None = None
    prompt_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    ttft_ms: float | None = None            # engine-reported: prompt_ms + the first decode step
    client_ttft_ms: float | None = None     # observed here, includes process scheduling
    produced: int = 0
    prompt_tokens: int = 0
    finish: str = ""
    memory_mib: float | None = None
    source: str = "engine-serve-protocol"
    #: `protocol` measures the engine's own serve protocol; `api` drives the real HTTP API the agent
    #: uses. They answer different questions, so the report must not present one as the other.
    mode: str = "protocol"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, float]:
        """The dict the job's `measure` callable must return.

        Only finite numbers are emitted, so a missing measurement stays missing rather than becoming
        a zero that a comparison might treat as a very slow result.
        """
        out: dict[str, float] = {"tokens_per_s": self.tokens_per_s}
        for key in ("prompt_tokens_per_s", "decode_tokens_per_s", "ttft_ms", "client_ttft_ms",
                    "memory_mib", "peak_memory_mib"):
            value = getattr(self, key)
            if isinstance(value, (int, float)):
                out[key] = float(value)
        return out


# --------------------------------------------------------------------------- parsing

_READY = re.compile(r"^READY\s+(\d+)\s+(\S+)\s*$")
_PP = re.compile(r"^PP\s+(\d+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?\s*$")
_DONE = re.compile(r"^DONE\s+(.*)$")
_TOKEN = re.compile(r"^T\s+(\d+)\s*$")


def parse_done(line: str) -> dict[str, Any]:
    """Parse a `DONE` line.

    The line's tail is variable — the engine appends lookup-chain counters only when that option is
    on — so the leading fields are read positionally and the rest is kept, never required.
    """
    m = _DONE.match(line.strip())
    if not m:
        raise MeasurementError(f"not a DONE line: {line[:120]!r}")
    parts = m.group(1).split()
    if len(parts) < 5:
        raise MeasurementError(f"a DONE line with fewer than 5 fields: {line[:120]!r}")
    try:
        produced = int(parts[0])
        prompt_tokens = int(parts[1])
        prompt_ms = float(parts[2])
        decode_ms = float(parts[3])
    except ValueError as e:
        raise MeasurementError(f"a DONE line with an unreadable number: {line[:120]!r}") from e
    return {
        "produced": produced,
        "prompt_tokens": prompt_tokens,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "finish": parts[4],
        "drafts_accepted": _int(parts[5]) if len(parts) > 5 else None,
        "drafts_offered": _int(parts[6]) if len(parts) > 6 else None,
        "tail": parts[7:],
    }


def _int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def parse_ready(line: str) -> int:
    m = _READY.match(line.strip())
    if not m:
        raise MeasurementError(f"not a READY line: {line[:120]!r}")
    return int(m.group(1))


def parse_pp(line: str) -> dict[str, float]:
    m = _PP.match(line.strip())
    if not m:
        raise MeasurementError(f"not a PP line: {line[:120]!r}")
    out = {"processed": float(m.group(1)), "total": float(m.group(2)), "ms": float(m.group(3))}
    if m.group(4):
        out["tok_s"] = float(m.group(4))
    return out


def measurement_from_protocol(done: dict[str, Any], *, tokens_seen: int,
                              client_ttft_ms: float | None = None,
                              ready_ms: float | None = None) -> EngineMeasurement:
    """Turn the engine's own report into measurements.

    Rejects a run that did not actually generate: a `cancel`, a `nonfinite`, a zero-token run and a
    disagreement between the token count and the DONE line are all errors, not slow results.
    """
    finish = str(done.get("finish", ""))
    produced = int(done.get("produced", 0))
    if finish in ("cancel", "nonfinite"):
        raise MeasurementError(f"the engine finished with '{finish}', which is not a measurement")
    if produced <= 0:
        raise MeasurementError("the engine produced no tokens")
    if tokens_seen and tokens_seen != produced:
        # The stream and the summary disagree, so neither can be trusted for a rate.
        raise MeasurementError(
            f"the engine reported {produced} tokens but streamed {tokens_seen}; refusing to score it")

    decode_ms = float(done.get("decode_ms") or 0.0)
    prompt_ms = float(done.get("prompt_ms") or 0.0)
    prompt_tokens = int(done.get("prompt_tokens") or 0)

    decode_per_s = (produced * 1000.0 / decode_ms) if decode_ms > 0 else None
    prompt_per_s = (prompt_tokens * 1000.0 / prompt_ms) if prompt_ms > 0 and prompt_tokens else None

    # What "tokens per second" means for the objective: the whole request's decoding rate. Prompt
    # processing is reported separately because a candidate can move one without the other, and
    # averaging them would hide exactly the trade the tuner is searching for.
    total_ms = prompt_ms + decode_ms
    overall = (produced * 1000.0 / total_ms) if total_ms > 0 else (decode_per_s or 0.0)

    # The engine's own TTFT: reading the prompt plus the first decode step.
    per_token_ms = (decode_ms / produced) if produced else None
    ttft = (prompt_ms + per_token_ms) if per_token_ms is not None else (prompt_ms or None)

    return EngineMeasurement(
        tokens_per_s=overall,
        prompt_tokens_per_s=prompt_per_s,
        decode_tokens_per_s=decode_per_s,
        ttft_ms=ttft,
        client_ttft_ms=client_ttft_ms,
        produced=produced,
        prompt_tokens=prompt_tokens,
        finish=finish,
        raw=dict(done),
    )


# --------------------------------------------------------------------------- running

class LineReader:
    """Read lines from a pipe without ever blocking the caller.

    `readline()` on a pipe has no timeout, so a `while time.monotonic() < deadline` loop around it
    does not bound anything: an engine that starts and then says nothing hangs the measurement
    forever. That is a real defect, not a hypothetical — it made the boot-timeout path unreachable
    and hung this module's own tests.

    A daemon reader thread moves the blocking read off the caller's thread, and the caller polls a
    queue with a deadline it actually enforces.
    """

    def __init__(self, stream, name: str = "engine") -> None:
        import queue
        import threading
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._stream = stream
        self.name = name
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=f"strata-{name}-reader")
        self._thread.start()

    def _pump(self) -> None:
        try:
            for line in self._stream:
                self._queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            # A sentinel so a reader knows the stream ended rather than that it is merely quiet.
            self._queue.put(None)

    def read_line(self, deadline: float) -> str | None:
        """One line, or None if the deadline passed or the stream ended.

        Raises `EOFError` when the stream has ended, so a caller can distinguish "nothing yet" from
        "the engine is gone".
        """
        import queue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = self._queue.get(timeout=remaining)
        except queue.Empty:
            return None
        if line is None:
            raise EOFError(f"the {self.name} stream ended")
        return line

    def drain_available(self) -> list[str]:
        """Everything already buffered, without waiting."""
        import queue
        out: list[str] = []
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return out
            if line is None:
                return out
            out.append(line)



def _windows_tree_peak_mib(pid: int) -> float | None:
    """The same measurement on Windows: peak working set summed over the process and its children.

    ``ps`` does not exist there, so the process list comes from a Toolhelp snapshot and each
    process's peak from ``GetProcessMemoryInfo``. A process that cannot be opened (another
    session, already gone) is skipped; if none can be read the answer is absent, not zero.
    """
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]

    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)       # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        return None
    rows: list[tuple[int, int]] = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            rows.append((entry.th32ProcessID, entry.th32ParentProcessID))
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    if pid not in {child for child, _ in rows}:
        return None
    wanted = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in rows:
            if parent in wanted and child not in wanted:
                wanted.add(child)
                changed = True
    total = 0
    for member in wanted:
        handle = kernel.OpenProcess(0x1000 | 0x0010, False, member)   # QUERY_LIMITED_INFORMATION | VM_READ
        if not handle:
            handle = kernel.OpenProcess(0x0400 | 0x0010, False, member)  # QUERY_INFORMATION | VM_READ
        if not handle:
            continue
        try:
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(MemoryCounters)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                total += counters.PeakWorkingSetSize
        finally:
            kernel.CloseHandle(handle)
    return round(total / (1024.0 * 1024.0), 1) if total else None


def process_tree_peak_mib(pid: int) -> float | None:
    """Peak resident memory of a process and its children, in MiB.

    The engine is one process today, but a backend may spawn workers, so the tree is summed rather
    than the root alone. `ps` is used because it reports each process's own peak (`rss` is current,
    `vsz` is address space); when it is unavailable the memory is simply absent, which is the honest
    answer and not a zero.
    """
    if os.name == "nt":
        return _windows_tree_peak_mib(pid)
    try:
        out = subprocess.run(["ps", "-o", "rss=,pid=,ppid=", "-ax"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows: list[tuple[int, int, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), int(parts[0])))
        except ValueError:
            continue
    wanted = {pid}
    changed = True
    while changed:                      # transitive closure over ppid
        changed = False
        for child, parent, _ in rows:
            if parent in wanted and child not in wanted:
                wanted.add(child)
                changed = True
    total = sum(rss for child, _, rss in rows if child in wanted)
    return round(total / 1024.0, 1) if total else None


class EngineHarness:
    """Boot the engine once per candidate and measure the declared workload.

    The engine is a subprocess speaking line protocol over stdout, exactly as `serve/server.py`
    drives it. One harness is created per candidate: a configuration that changes the expert cache
    or the KV layout cannot be assumed to take effect in a running engine.
    """

    def __init__(self, engine: Path, args: dict[str, str], *, workload: dict[str, Any] | None = None,
                 cwd: Path | None = None, env: dict[str, str] | None = None,
                 boot_timeout_s: float = 600.0, request_timeout_s: float = 600.0) -> None:
        self.engine = Path(engine)
        self.args = dict(args)
        self.workload = {**DEFAULT_WORKLOAD, **(workload or {})}
        self.cwd = Path(cwd) if cwd else Path.cwd()
        if env is None:
            from strata_control.tuning import engine_env
            env = engine_env(self.engine)       # an engine package's own folder first on the library path
        self.env = env
        self.boot_timeout_s = boot_timeout_s
        self.request_timeout_s = request_timeout_s
        self._proc: subprocess.Popen | None = None
        self._reader: LineReader | None = None
        self.last_measurement: EngineMeasurement | None = None
        self._ready_ms: float | None = None
        self._context: int | None = None

    def _argv(self) -> list[str]:
        from strata_control.tuning import engine_command, flag_argv
        argv = [*engine_command(self.engine), "--serve"]
        for flag, value in self.args.items():
            argv.extend(flag_argv(flag, value))
        return argv

    def __enter__(self) -> "EngineHarness":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if not self.engine.exists():
            raise MeasurementError(f"the engine binary is not at {self.engine}")
        started = time.monotonic()
        try:
            from serve import lifeline       # the engine ends with this process, however this process ends
            self._proc = lifeline.popen(
                self._argv(), "engine", cwd=str(self.cwd), env=self.env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", bufsize=1)
        except OSError as e:
            raise MeasurementError(f"the engine could not be started: {e}") from e

        self._reader = LineReader(self._proc.stdout, "engine")
        deadline = time.monotonic() + self.boot_timeout_s
        while True:
            if time.monotonic() >= deadline:
                if self._proc.poll() is not None:
                    raise MeasurementError(
                        f"the engine exited with code {self._exit_code()} before it was ready"
                        + self._stderr_tail())
                raise MeasurementError(
                    f"the engine did not report READY within {self.boot_timeout_s:.0f}s")
            try:
                line = self._reader.read_line(deadline)
            except EOFError:
                raise MeasurementError(
                    f"the engine exited with code {self._exit_code()} before it was ready"
                    + self._stderr_tail()) from None
            if line is None:
                # Nothing yet. Only a *finished* process means failure; a slow boot does not.
                if self._proc.poll() is not None:
                    raise MeasurementError(
                        f"the engine exited with code {self._exit_code()} before it was ready"
                        + self._stderr_tail())
                continue
            line = line.strip()
            if line.startswith("READY"):
                self._context = parse_ready(line)
                self._ready_ms = (time.monotonic() - started) * 1000.0
                return
            if line.startswith("ERR"):
                raise MeasurementError(f"the engine refused to start: {line}")

    def _exit_code(self) -> int | None:
        """The engine's exit code, reaped if necessary.

        `poll()` returns None until the process has been waited on, and a reader thread can observe
        end-of-stream *before* that happens. Reporting "exited with code None" is useless in exactly
        the case where the code matters most, so this waits briefly for the status.
        """
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return None
        return self._proc.returncode

    def _stderr_tail(self, limit: int = 400) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            # Only meaningful once the process has ended; a running engine's stderr is still open.
            if self._proc.poll() is None:
                return ""
            text = self._proc.stderr.read() or ""
            return f"; stderr: {text[-limit:].strip()}"
        except OSError:
            return ""

    def measure(self) -> EngineMeasurement:
        """Send the workload and read one result. Raises `MeasurementError` on anything unusable."""
        if self._proc is None or self._proc.poll() is not None:
            raise MeasurementError("the engine is not running")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise MeasurementError("the engine has no input/output pipe")

        tokens = self._prompt_tokens()
        flags = (f" temperature={self.workload['temperature']} seed={self.workload['seed']}")
        request = f"GEN {int(self.workload['max_new'])}{flags} {','.join(str(t) for t in tokens)}\n"

        client_start = time.monotonic()
        try:
            self._proc.stdin.write(request)
            self._proc.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            raise MeasurementError(f"the engine stopped accepting work: {e}" + self._stderr_tail()) from e

        deadline = time.monotonic() + self.request_timeout_s
        seen = 0
        client_ttft: float | None = None
        done: dict[str, Any] | None = None
        while True:
            if time.monotonic() >= deadline:
                break
            try:
                raw = self._reader.read_line(deadline)
            except EOFError:
                raise MeasurementError(
                    f"the engine exited with code {self._exit_code()} during the request"
                    + self._stderr_tail()) from None
            if raw is None:
                if self._proc.poll() is not None:
                    raise MeasurementError(
                        f"the engine exited with code {self._exit_code()} during the request"
                        + self._stderr_tail())
                continue
            line = raw.strip()
            if _TOKEN.match(line):
                seen += 1
                if client_ttft is None:
                    client_ttft = (time.monotonic() - client_start) * 1000.0
            elif line.startswith("DONE"):
                done = parse_done(line)
                break
            elif line.startswith("ERR"):
                raise MeasurementError(f"the engine reported an error: {line}")
        if done is None:
            raise MeasurementError(
                f"the engine produced no DONE line within {self.request_timeout_s:.0f}s")

        result = measurement_from_protocol(done, tokens_seen=seen, client_ttft_ms=client_ttft,
                                          ready_ms=self._ready_ms)
        if self._proc is not None:
            result.peak_memory_mib = process_tree_peak_mib(self._proc.pid)
        result.mode = "protocol"
        self.last_measurement = result
        return result

    def _prompt_tokens(self) -> Sequence[int]:
        """The workload's token ids.

        A real sweep substitutes the tokenizer's ids here. The default is a fixed, deterministic
        sequence of the declared length: it exercises the same code path for every candidate, which
        is what makes the comparison valid, and it cannot be accused of silently changing the work.
        """
        n = max(1, int(self.workload["prompt_tokens"]))
        return [1000 + (i % 97) for i in range(n)]

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except (OSError, ValueError):
            try:
                self._proc.kill()
            except OSError:
                pass
        finally:
            for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self._proc = None


def make_measure(engine: Path, workload: dict[str, Any] | None = None,
                 cwd: Path | None = None) -> "callable":
    """Build the `measure` callable the tuning job expects.

    The returned function boots the engine with the candidate's args, measures once, and returns the
    metrics. It raises on anything unusable, which the job records as that candidate's status.
    """
    def measure(candidate) -> dict[str, float]:
        harness = EngineHarness(engine, candidate.args if hasattr(candidate, "args") else candidate,
                                workload=workload, cwd=cwd)
        try:
            harness.start()
            return harness.measure().to_metrics()
        finally:
            harness.close()

    return measure
