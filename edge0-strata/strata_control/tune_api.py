"""Measure a configuration through the API the agent actually uses.

`tune_engine.EngineHarness` measures the engine's own serve protocol, which is fast and precise and
answers "how fast can this configuration decode". It does **not** answer "is this configuration
better for a user", because the user's requests arrive over HTTP, go through the worker's queueing,
prompt handling and streaming, and it is that whole path whose latency matters.

Checklist T04 asks for the full API/Harness workload and for memory, so this module provides it. The
two modes are deliberately separate rather than one replacing the other:

* ``protocol`` is cheap enough to run for every candidate in a sweep;
* ``api`` is the confirmation, run against the finalists and against the exported configuration.

Reporting one as the other would be the same class of error as a fixture result standing in for real
inference, so every measurement carries the mode that produced it and the report shows the mode.

What this measures, per request: time to first token (from the first streamed content frame, which is
what a user waits for), end-to-end duration, token counts taken from the server's own usage block
rather than counted here, and the peak resident memory of the server process tree. The prompt and the
expectation are supplied by the caller, and the response is checked against the expectation, because
a configuration that answers faster and wrongly is not an improvement.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata_control import loopback
from strata_control.tune_engine import EngineMeasurement, MeasurementError, process_tree_peak_mib


class ApiHarness:
    """Start the worker with one configuration and measure a real chat completion.

    The server is started as a child process and stopped afterwards: a configuration that changes the
    KV layout or the expert cache cannot be assumed to take effect in a running server, and reusing
    one would silently measure the previous candidate.
    """

    def __init__(self, *, python: Path | str, serve_dir: Path, config_path: Path,
                 port: int = 8123, host: str = "127.0.0.1", boot_timeout_s: float = 900.0,
                 request_timeout_s: float = 900.0, env: dict[str, str] | None = None,
                 engine: str = "strata") -> None:
        self.python = str(python)
        # `strata` is the real engine; `mock` is the scripted one, and a measurement taken against it
        # is a test of this harness and never a performance result.
        self.engine = engine
        self.serve_dir = Path(serve_dir)
        self.config_path = Path(config_path)
        self.port = int(port)
        self.host = host
        self.boot_timeout_s = boot_timeout_s
        self.request_timeout_s = request_timeout_s
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._log = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _argv(self) -> list[str]:
        # The worker's own spelling. `--profile` is the terminal launcher's flag, not this one, and
        # passing it made the worker exit 2 with a usage message before it ever bound a port.
        return [self.python, "-m", "serve.server", "--engine", self.engine,
                "--config", str(self.config_path),
                "--host", self.host, "--port", str(self.port)]

    def start(self) -> None:
        import tempfile
        self._log = tempfile.NamedTemporaryFile(prefix="strata-api-harness-", suffix=".log",
                                                delete=False)
        try:
            from serve import lifeline       # the worker (and the engine under it) end with this process
            self._proc = lifeline.popen(self._argv(), "worker", cwd=str(self.serve_dir), env=self.env,
                                        stdout=self._log, stderr=subprocess.STDOUT)
        except OSError as e:
            raise MeasurementError(f"the worker could not be started: {e}") from e

        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise MeasurementError(
                    f"the worker exited with code {self._proc.returncode} before becoming ready"
                    + self._log_tail())
            try:
                with loopback.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.2)
        raise MeasurementError(f"the worker did not answer /health within {self.boot_timeout_s:.0f}s"
                               + self._log_tail())

    def _log_tail(self, limit: int = 500) -> str:
        if self._log is None:
            return ""
        try:
            text = Path(self._log.name).read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return ""
            # The engine prints its reason first and a usage dump after it, so the last 500 characters of a
            # rejected flag are the middle of the usage text. Name the reason line(s) before the tail.
            reasons = [line.strip() for line in text.splitlines()
                       if re.search(r"unknown argument|unrecognized|invalid value|error|refus|is required|cannot|failed",
                                    line, re.IGNORECASE)][:3]
            lead = f"; reason: {' | '.join(reasons)[:300]}" if reasons else ""
            return f"{lead}; log: {text[-limit:].strip()}"
        except OSError:
            return ""

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass

    def __enter__(self) -> "ApiHarness":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ measuring

    def measure(self, *, prompt: str, max_new: int, expect: str | None = None,
                temperature: float = 0.0, seed: int = 1,
                model: str = "strata-local") -> EngineMeasurement:
        """One streamed chat completion, measured end to end.

        `expect` is a regular expression the answer must match for the run to count as correct. A
        configuration that is fast and wrong must not be able to win a sweep, so a mismatch raises
        rather than being recorded as a fast result.
        """
        body = json.dumps({
            "model": model, "stream": True, "max_tokens": max_new,
            "temperature": temperature, "seed": seed,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions", data=body,
            headers={"content-type": "application/json", "authorization": "Bearer strata-local"},
            method="POST")

        started = time.monotonic()
        # Reasoning models stream their thinking into `reasoning_content` and the answer into
        # `content`. Measuring only `content` reports "no content" for a reasoning model and, worse,
        # moves the first-token time to whenever the thinking finished -- so a configuration that
        # thinks longer would look faster. Both are output, and the first of either is the wait.
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        first_token_at: float | None = None
        usage: dict[str, Any] = {}
        finish: str | None = None
        try:
            with loopback.urlopen(request, timeout=self.request_timeout_s) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        frame = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if frame.get("error"):
                        raise MeasurementError(
                            f"the worker reported an error: {frame['error'].get('message')}")
                    if frame.get("usage"):
                        usage = frame["usage"]
                    choice = (frame.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                        else:
                            reasoning_parts.append(delta["reasoning_content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise MeasurementError(f"the worker refused the request ({e.code}): {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise MeasurementError(f"the request failed: {e}" + self._log_tail()) from None

        duration_ms = (time.monotonic() - started) * 1000.0
        answer = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        if not answer and not reasoning:
            raise MeasurementError("the worker streamed no content and no reasoning")
        # The expectation is checked against the answer when there is one, and against the reasoning
        # otherwise: an engine that only streams reasoning is still producing checkable output.
        checked = answer or reasoning
        if expect is not None and not re.search(expect, checked, re.IGNORECASE | re.DOTALL):
            raise MeasurementError(
                f"the output did not match the expectation {expect!r}, so this configuration is not "
                f"correct: {checked[:200]!r}")
        if not answer:
            raise MeasurementError(
                "the worker streamed reasoning but no answer, so its rate is not a generation rate: "
                f"{reasoning[:120]!r}")

        out_tokens = usage.get("completion_tokens")
        if not isinstance(out_tokens, int) or out_tokens <= 0:
            # The server reports usage; counting frames here would be an estimate, and reporting an
            # estimate as a token count is how chunk counts get mistaken for tokens.
            raise MeasurementError("the worker reported no completion_tokens, so the rate is unknown")
        prompt_tokens = usage.get("prompt_tokens")
        decode_ms = duration_ms - ((first_token_at - started) * 1000.0 if first_token_at else 0.0)
        decode_ms = max(decode_ms, 0.001)

        parts = usage.get("prompt_tokens_details") or {}
        measurement = EngineMeasurement(
            tokens_per_s=round(out_tokens * 1000.0 / decode_ms, 4),
            decode_tokens_per_s=round(out_tokens * 1000.0 / decode_ms, 4),
            prompt_tokens_per_s=(round(prompt_tokens * 1000.0 / max(
                (first_token_at - started) * 1000.0, 0.001), 4)
                if isinstance(prompt_tokens, int) and prompt_tokens and first_token_at else None),
            ttft_ms=round((first_token_at - started) * 1000.0, 2) if first_token_at else None,
            client_ttft_ms=round((first_token_at - started) * 1000.0, 2) if first_token_at else None,
            produced=out_tokens,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else 0,
            finish=finish or "unknown",
            source="worker-http-api",
            mode="api",
            raw={"usage": usage, "cached_tokens": parts.get("cached_tokens"),
                 "duration_ms": round(duration_ms, 2), "answer_chars": len(answer),
                 "reasoning_chars": len(reasoning)})
        if self._proc is not None:
            measurement.peak_memory_mib = process_tree_peak_mib(self._proc.pid)
        return measurement


def make_api_measure(*, python: Path | str, serve_dir: Path, config_path: Path, prompt: str,
                     max_new: int, expect: str | None = None, port: int = 8123,
                     boot_timeout_s: float = 900.0,
                     request_timeout_s: float = 900.0):
    """Build a `measure` callable for the tuning job that uses the HTTP API.

    Unlike the protocol harness, the configuration is already in a file, so the candidate's flags are
    written to a *copy* of it and the worker is pointed at the copy. The caller's file is never
    touched: a sweep must not be able to change the active configuration even by accident.
    """
    import tempfile

    from strata_control.tune_apply import apply_args_to_config, args_to_argv

    base = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def measure(candidate) -> dict[str, float]:
        args = getattr(candidate, "args", candidate)
        candidate_cfg = apply_args_to_config(base, args_to_argv(args) if isinstance(args, dict) else list(args))
        with tempfile.NamedTemporaryFile("w", prefix="strata-tune-candidate-", suffix=".json",
                                         delete=False, encoding="utf-8") as fh:
            json.dump(candidate_cfg, fh, indent=1)
            path = Path(fh.name)
        harness = ApiHarness(python=python, serve_dir=serve_dir, config_path=path, port=port,
                             boot_timeout_s=boot_timeout_s, request_timeout_s=request_timeout_s)
        try:
            harness.start()
            return harness.measure(prompt=prompt, max_new=max_new, expect=expect).to_metrics()
        finally:
            harness.close()
            try:
                path.unlink()
            except OSError:
                pass

    return measure
