"""Explicit, local-file serving benchmark with privacy-safe evidence.

This measures the API as a client sees it. Engine batching and GPU readings are
sampled from telemetry; missing signals stay unknown. It does not tune a model.
"""

from __future__ import annotations

try:
    from strata_control.private_files import private_mkdir as _private_mkdir
except ImportError:                                     # a thin bundle (monitor only) ships without strata_control
    def _private_mkdir(path, *, parents=False, exist_ok=True):
        path.mkdir(mode=0o700, parents=parents, exist_ok=exist_ok)
        return path
import hashlib
import http.client
import json
import math
import os
import re
import socket
import stat
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from strata_cli.client import ClientError, Endpoint, _NoRedirect
from strata_control.profiles import ProfileError
from strata_control import atomic as _atomic_mod
from strata_control import posix_owner


SCHEMA = 1
MAX_EVENT_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_SAMPLES_PER_PHASE = 500
SHORT_PROMPT = "Answer in one sentence: what does a GPU do during model inference?"
LONG_PROMPT = ("Explain how a serving system handles simultaneous short and long requests, "
               "including queue admission, prefill, decode slots, batching and cancellation. " * 12)
CANCEL_PROMPT = ("Cancellation benchmark: write a detailed explanation of batching and parallel inference "
                 "with at least 200 words. Discuss admission, prompt reading, decoding and recovery.")
SUITES = {"serving-smoke": ("warmup", "serial", "mixed"),
          "parallel-mixed-v1": ("warmup", "serial", "mixed", "mixed"),
          "parallel-overload-v1": ("warmup", "overload", "recovery"),
          "parallel-cancel-v1": ("warmup", "cancel_active", "cancel_queued", "recovery"),
          "parallel-prefill-cancel-v1": ("warmup", "cancel_prefill", "recovery")}


class OperatorCancellation:
    """Stop benchmark-owned streams after an owner-local cancel request appears."""

    def __init__(self, directory: Path):
        self.path = directory / "cancel.request"
        self.event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        self._watcher = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(0.1):
            if self.path.exists():
                self.cancel()
                return

    def cancel(self) -> None:
        with self._lock:
            self.event.set()
            sockets = tuple(self._sockets)
        for owned in sockets:
            try:
                owned.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    @contextmanager
    def track(self, response):
        # urllib's HTTPResponse owns the socket while reading its buffered stream.
        stream = getattr(response, "fp", None)
        raw = getattr(stream, "raw", None)
        owned = getattr(raw, "_sock", None)
        if not isinstance(owned, socket.socket):
            owned = None
        if owned is not None:
            with self._lock:
                self._sockets.add(owned)
                cancelled = self.event.is_set()
            if cancelled:
                try:
                    owned.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        try:
            yield
        finally:
            if owned is not None:
                with self._lock:
                    self._sockets.discard(owned)

    def __enter__(self):
        self._watcher.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._watcher.join(timeout=1)


def _planned_phase_kinds(suite: str, repeat: int) -> list[str]:
    single = {"warmup", "recovery", "cancel_active", "cancel_queued", "cancel_prefill"}
    return [kind for kind in SUITES[suite]
            for _ in range(1 if kind in single else repeat)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 4)


def _sample(snapshot: dict) -> dict:
    work = snapshot.get("work") or {}
    rates = snapshot.get("rates") or {}
    return {"at": time.time(), "instance_id": (snapshot.get("instance") or {}).get("id"),
            "active": work.get("active"), "queued": work.get("queued"),
            "waiting": work.get("waiting"), "occupied_slots": work.get("occupied_slots"),
            "slots_effective": work.get("slots_effective"), "groups": work.get("groups"),
            "admission_reserved": work.get("admission_reserved"),
            "admission_rejected": work.get("admission_rejected"),
            "admission_limit": work.get("admission_limit"), "max_queue": work.get("max_queue"),
            "queue_timeout_s": work.get("queue_timeout_s"), "queue_timeouts": work.get("queue_timeouts"),
            "client_disconnect_cancels": work.get("client_disconnect_cancels"),
            "answer_tok_s": rates.get("answer_tok_s"),
            "devices": [{"index": d.get("index"), "mem_used": d.get("mem_used"),
                         "mem_total": d.get("mem_total"), "util": d.get("util")}
                        for d in snapshot.get("devices", [])[:16]]}


def _environment(snapshot: dict) -> dict:
    """Bounded reported setup for comparisons, not a hardware attestation."""
    work = snapshot.get("work") or {}
    model = snapshot.get("model") or {}
    topology = snapshot.get("topology") or {}
    devices = []
    for row in snapshot.get("devices", [])[:16]:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        devices.append({"index": row.get("index"),
                        "name": name[:120] if isinstance(name, str) and name.isprintable() else None,
                        "mem_total": row.get("mem_total")})
    return {"model_context": model.get("context"),
            "slots_effective": work.get("slots_effective"), "groups": work.get("groups"),
            "max_queue": work.get("max_queue"), "queue_timeout_s": work.get("queue_timeout_s"),
            "selected_devices": (topology.get("selected_devices") or [])[:16],
            "stages": (topology.get("stages") or [])[:16], "devices": devices,
            "source": "server_telemetry; driver, engine binary and model artifact hashes not attested"}


def _managed_match(profile_identity: dict, endpoint_root: str, snapshot: dict) -> dict:
    """Check a local owner host without persisting its control token or launch MAC."""
    from strata_control import runtime

    name, revision = profile_identity.get("profile"), profile_identity.get("revision")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or \
            not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{16}", revision):
        return {"matched": False, "reason": "saved_profile_required"}
    try:
        owner = runtime.status(name)
        plan = runtime.plan_restart(name)
    except (OSError, ProfileError, ValueError):
        return {"matched": False, "reason": "owner_unavailable"}
    instance = (snapshot.get("instance") or {}).get("id")
    if (not owner.get("ok") or owner.get("state") != "ready" or
            owner.get("launch_attested") is not True or owner.get("api_identity_verified") is not True or
            owner.get("generation_ready") is not True):
        return {"matched": False, "reason": "owner_not_verified"}
    if (not plan.get("ok") or plan.get("changed") is not False):
        return {"matched": False, "reason": "profile_or_source_changed"}
    if (not isinstance(instance, str) or owner.get("instance_id") != instance or
            plan.get("current_instance_id") != instance):
        return {"matched": False, "reason": "instance_mismatch"}
    if (owner.get("profile_revision") != revision or
            plan.get("current_profile_revision") != revision or
            plan.get("candidate_profile_revision") != revision):
        return {"matched": False, "reason": "profile_revision_mismatch"}
    if owner.get("api_url") != endpoint_root:
        return {"matched": False, "reason": "endpoint_mismatch"}
    return {"matched": True, "reason": "managed_launch_checked", "instance_id": instance,
            "profile_revision": revision}


def _event_result(event: dict, first_ns: int | None,
                  completion_tokens: int | None) -> tuple[int | None, int | None, str | None]:
    if "error" in event:
        error = event["error"]
        code = (error.get("code") or error.get("type")) if isinstance(error, dict) else "stream_error"
        return first_ns, completion_tokens, str(code)[:80]
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
            first_ns = first_ns or time.monotonic_ns()
    usage = event.get("usage")
    if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int:
        completion_tokens = usage["completion_tokens"]
    return first_ns, completion_tokens, None


def _request(root: str, credential: str | None, prompt: str, max_tokens: int,
             timeout: float, label: str, start_gate: threading.Event | None = None,
             operator: OperatorCancellation | None = None) -> dict:
    """Send one streamed request. Save measurements, never prompt or answer text."""
    payload = {"messages": [{"role": "user", "content": prompt}],
               "stream": True, "max_tokens": max_tokens, "temperature": 0}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": "Strata-Bench/0.1"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    request = urllib.request.Request(root + "/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    if start_gate is not None:
        start_gate.wait()
    start_ns = time.monotonic_ns()
    first_ns = None
    tokens = None
    received = 0
    done = False
    status = None
    error = None
    try:
        if operator is not None and operator.event.is_set():
            raise InterruptedError("operator cancelled")
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            with operator.track(response) if operator is not None else nullcontext():
                status = response.status
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise ValueError("response was not an event stream")
                while True:
                    line = response.readline(MAX_EVENT_BYTES + 1)
                    if not line:
                        break
                    received += len(line)
                    if len(line) > MAX_EVENT_BYTES or received > MAX_RESPONSE_BYTES:
                        raise ValueError("stream exceeded the benchmark size limit")
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError("stream event was not an object")
                    first_ns, tokens, error = _event_result(event, first_ns, tokens)
                    if error:
                        break
                if not done and not error:
                    error = "stream_ended_without_done"
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = "authentication_failed" if status in (401, 403) else f"http_{status}"
        if status == 429:
            try:
                raw = exc.read(4097)
                if len(raw) <= 4096:
                    body = json.loads(raw)
                    code = (body.get("error") or {}).get("code")
                    if code in ("admission_full", "queue_wait_timeout"):
                        error = code
            except (OSError, ValueError, AttributeError):
                pass
        exc.close()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
        error = "connection_" + type(exc).__name__
    except (UnicodeError, ValueError) as exc:
        error = "invalid_stream_" + type(exc).__name__
    if operator is not None and operator.event.is_set():
        error = "operator_cancelled"
        done = False
    end_ns = time.monotonic_ns()
    result = {"label": label, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_characters": len(prompt), "max_tokens": max_tokens,
            "status": status, "ok": status == 200 and done and error is None,
            "error_code": error, "latency_s": round((end_ns - start_ns) / 1e9, 4),
            "ttft_s": round((first_ns - start_ns) / 1e9, 4) if first_ns else None,
            "completion_tokens": tokens}
    if start_gate is not None:
        result["_launch_ns"] = start_ns
    return result


def _cancellable_request(root: str, credential: str | None, prompt: str, timeout: float,
                         first_content: bool, opened: threading.Event,
                         ready_to_close: threading.Event, close_now: threading.Event,
                         result: dict, operator: OperatorCancellation | None = None) -> None:
    """Hold one streamed client socket until the benchmark deliberately closes it."""
    payload = {"messages": [{"role": "user", "content": prompt}],
               "stream": True, "max_tokens": 512, "temperature": 0}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": "Strata-Bench/0.1"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    request = urllib.request.Request(root + "/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started = time.monotonic_ns()
    result.update({"label": "cancel_target", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "prompt_characters": len(prompt), "max_tokens": 512, "status": None,
                   "client_closed": False, "first_content_seen": False, "error_code": None})
    try:
        if operator is not None and operator.event.is_set():
            raise InterruptedError("operator cancelled")
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            with operator.track(response) if operator is not None else nullcontext():
                result["status"] = response.status
                opened.set()
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    result["error_code"] = "not_event_stream"
                    return
                if first_content:
                    received = 0
                    while received <= MAX_RESPONSE_BYTES:
                        line = response.readline(MAX_EVENT_BYTES + 1)
                        if not line:
                            result["error_code"] = "ended_before_cancel"
                            return
                        received += len(line)
                        if len(line) > MAX_EVENT_BYTES:
                            result["error_code"] = "stream_too_large"
                            return
                        if line.startswith(b"data: "):
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                result["error_code"] = "ended_before_cancel"
                                return
                            event = json.loads(data)
                            first_ns, _, error = _event_result(event, None, None)
                            if error:
                                result["error_code"] = error
                                return
                            if first_ns is not None:
                                result["first_content_seen"] = True
                                break
                    if not result["first_content_seen"]:
                        result["error_code"] = "stream_too_large"
                        return
                ready_to_close.set()
                if not close_now.wait(timeout):
                    result["error_code"] = "client_close_signal_timeout"
                    return
                result["client_closed"] = True
    except urllib.error.HTTPError as error:
        result["status"] = error.code
        result["error_code"] = "authentication_failed" if error.code in (401, 403) else f"http_{error.code}"
        error.close()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError,
            http.client.HTTPException, UnicodeError, ValueError, TypeError, AttributeError) as error:
        result["error_code"] = type(error).__name__[:80]
    finally:
        if operator is not None and operator.event.is_set():
            result["error_code"] = "operator_cancelled"
        opened.set()
        ready_to_close.set()
        result["latency_s"] = round((time.monotonic_ns() - started) / 1e9, 4)


def _cancel_phase(endpoint: Endpoint, telemetry: Endpoint, *, kind: str,
                  slots: int, timeout: float,
                  operator: OperatorCancellation | None = None) -> dict:
    samples: list[dict] = []
    sample_errors = 0
    samples_omitted = 0
    peaks: dict[str, int | float] = {}
    beginning = time.monotonic()

    def capture() -> dict | None:
        nonlocal sample_errors, samples_omitted
        try:
            snapshot = telemetry.snapshot()
        except (ClientError, ValueError):
            sample_errors += 1
            return None
        row = _sample(snapshot)
        for key in ("active", "queued", "waiting", "occupied_slots", "groups"):
            value = row.get(key)
            if type(value) in (int, float):
                peaks[key] = max(peaks.get(key, value), value)
        if len(samples) < 120:
            samples.append(row)
        else:
            samples_omitted += 1
        return snapshot

    def until(check, seconds: float) -> dict | None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if operator is not None and operator.event.is_set():
                return None
            snapshot = capture()
            if snapshot is not None and check(snapshot):
                return snapshot
            time.sleep(0.05)
        return None

    def wait_signal(signal: threading.Event, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not signal.is_set() and time.monotonic() < deadline:
            if operator is not None and operator.event.is_set():
                return
            signal.wait(min(0.2, max(0, deadline - time.monotonic())))

    before = capture()
    if before is None:
        raise ProfileError("cancellation benchmark could not read telemetry", 5)
    work_before = before.get("work") or {}
    before_ids = {row.get("id") for row in work_before.get("requests") or []}
    base_cancel = work_before.get("client_disconnect_cancels")
    base_timeout = work_before.get("queue_timeouts")
    if type(base_cancel) is not int or type(base_timeout) is not int:
        raise ProfileError("cancellation benchmark lost required telemetry counters", 5)
    opened, ready_to_close, close_now = threading.Event(), threading.Event(), threading.Event()
    target: dict = {}
    companion_rows: list[dict] = []
    seen_state = False
    observed_cancel = False
    settled = False
    target_id = None
    companion_ok = True
    worker = None
    companions = None
    companions_needed = slots if kind == "cancel_queued" else 1 if kind == "cancel_prefill" else 0
    prompt_read_seen = False
    try:
        if companions_needed:
            companions = ThreadPoolExecutor(max_workers=companions_needed)
            futures = [companions.submit(_request, endpoint.root, endpoint.credential,
                                         CANCEL_PROMPT, 512, timeout, "companion", None, operator)
                       for _ in range(companions_needed)]
            running = until(lambda snap: sum(row.get("state") not in ("queued", "waiting")
                                             for row in (snap.get("work") or {}).get("requests") or []
                                             if row.get("id") not in before_ids) >= companions_needed,
                            min(timeout, 20))
            if running is None:
                target["error_code"] = "companion_slots_not_active"
            else:
                before_ids = {row.get("id") for row in (running.get("work") or {}).get("requests") or []}
        if "error_code" not in target:
            worker = threading.Thread(target=_cancellable_request,
                                      args=(endpoint.root, endpoint.credential,
                                            SHORT_PROMPT if kind == "cancel_queued" else
                                            LONG_PROMPT if kind == "cancel_prefill" else CANCEL_PROMPT,
                                            timeout, kind == "cancel_active",
                                            opened, ready_to_close, close_now, target, operator), daemon=True)
            worker.start()
            wait_signal(opened, min(timeout, 20))
            wait_signal(ready_to_close, min(timeout, 20))
            desired = ({"queued", "waiting"} if kind == "cancel_queued" else
                       {"reading"} if kind == "cancel_prefill" else None)
            found = until(lambda snap: any(row.get("id") not in before_ids and
                                           (desired is None or row.get("state") in desired) and
                                           (kind != "cancel_prefill" or
                                            type(row.get("prompt_read")) is int and
                                            type(row.get("prompt_tokens")) is int and
                                            0 < row["prompt_read"] < row["prompt_tokens"])
                                           for row in (snap.get("work") or {}).get("requests") or []),
                          min(timeout, 5.0))
            if found is not None:
                row = next(row for row in (found.get("work") or {}).get("requests") or []
                           if row.get("id") not in before_ids and
                           (desired is None or row.get("state") in desired) and
                           (kind != "cancel_prefill" or
                            type(row.get("prompt_read")) is int and
                            type(row.get("prompt_tokens")) is int and
                            0 < row["prompt_read"] < row["prompt_tokens"]))
                target_id = row["id"]
                target["request_id"] = target_id
                seen_state = True
                if kind == "cancel_prefill":
                    target["prompt_read_at_close"] = row["prompt_read"]
                    target["prompt_tokens_at_close"] = row["prompt_tokens"]
                    prompt_read_seen = True
            close_now.set()
            worker.join(timeout=min(timeout, 10))
            observed = until(lambda snap: type((snap.get("work") or {}).get("client_disconnect_cancels")) is int
                             and (snap["work"]["client_disconnect_cancels"] >= base_cancel + 1)
                             and (target_id is None or all(row.get("id") != target_id
                                                           for row in snap["work"].get("requests") or [])), 10.0)
            observed_cancel = observed is not None
        if companions is not None:
            for future in as_completed(futures):
                companion_rows.append(future.result())
            companion_ok = all(row["ok"] for row in companion_rows)
        settled = until(lambda snap: (snap.get("work") or {}).get("admission_reserved") == 0 and
                        (snap.get("work") or {}).get("active") == 0 and
                        (snap.get("work") or {}).get("occupied_slots") in (0, None), 10.0) is not None
    finally:
        close_now.set()
        if worker is not None:
            worker.join(timeout=1)
        if companions is not None:
            companions.shutdown(wait=True)
    after = capture() or {}
    work_after = after.get("work") or {}
    cancel_delta = (work_after.get("client_disconnect_cancels") - base_cancel
                    if type(work_after.get("client_disconnect_cancels")) is int and type(base_cancel) is int
                    else None)
    timeout_delta = (work_after.get("queue_timeouts") - base_timeout
                     if type(work_after.get("queue_timeouts")) is int and type(base_timeout) is int else None)
    passed = (seen_state and (kind != "cancel_prefill" or prompt_read_seen) and
              target.get("client_closed") is True and observed_cancel and settled and
              companion_ok and timeout_delta == 0 and target.get("error_code") is None and
              (kind != "cancel_active" or target.get("first_content_seen")))
    elapsed = time.monotonic() - beginning
    return {"kind": kind, "requested": 1 + companions_needed,
            "client_workers": 1 + companions_needed, "passed": passed,
            "duration_s": round(elapsed, 4), "requests": [target, *companion_rows],
            "telemetry_samples": samples, "telemetry_sample_errors": sample_errors,
            "telemetry_samples_omitted": samples_omitted,
            "summary": {"successful": sum(row["ok"] for row in companion_rows),
                        "failed": 0 if passed else 1, "expected_overload": 0,
                        "target_state_seen": seen_state, "client_closed": target.get("client_closed") is True,
                        "prompt_read_seen": prompt_read_seen,
                        "first_content_seen": target.get("first_content_seen") is True,
                        "client_disconnect_delta": cancel_delta, "queue_timeouts_delta": timeout_delta,
                        "companion_successful": sum(row["ok"] for row in companion_rows),
                        "settled": settled,
                        "p50_latency_s": _percentile([row["latency_s"] for row in companion_rows if row["ok"]], 0.5),
                        "p95_latency_s": _percentile([row["latency_s"] for row in companion_rows if row["ok"]], 0.95),
                        "p50_ttft_s": None, "p95_ttft_s": None, "output_tokens": None, "output_tok_s": None,
                        "peak_active": peaks.get("active"), "peak_queued": peaks.get("queued"),
                        "peak_waiting": peaks.get("waiting"),
                        "peak_occupied_slots": peaks.get("occupied_slots"), "peak_groups": peaks.get("groups"),
                        "peak_admission_rejected": None}}


def _phase(endpoint: Endpoint, telemetry: Endpoint, *, kind: str, count: int,
           concurrency: int, timeout: float,
           operator: OperatorCancellation | None = None) -> dict:
    samples = []
    sample_errors = 0
    samples_omitted = 0
    sample_cap = min(MAX_SAMPLES_PER_PHASE, 120) if kind == "overload" else MAX_SAMPLES_PER_PHASE
    peaks = {}
    first_sample = None
    last_sample = None
    stop = threading.Event()

    def record(snapshot):
        nonlocal samples_omitted, first_sample, last_sample
        sample = _sample(snapshot)
        if first_sample is None:
            first_sample = sample
        last_sample = sample
        for key in ("active", "queued", "waiting", "occupied_slots", "groups", "admission_rejected"):
            value = sample.get(key)
            if type(value) in (int, float):
                peaks[key] = max(peaks.get(key, value), value)
        if len(samples) < sample_cap:
            samples.append(sample)
        else:
            samples_omitted += 1

    def poll():
        nonlocal sample_errors
        while not stop.is_set() and (operator is None or not operator.event.is_set()):
            try:
                record(telemetry.snapshot())
            except (ClientError, ValueError):
                sample_errors += 1
            stop.wait(0.15)

    try:
        record(telemetry.snapshot())
    except (ClientError, ValueError):
        sample_errors += 1
    worker = threading.Thread(target=poll, daemon=True)
    worker.start()
    start = time.monotonic()
    requests = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            start_gate = threading.Event() if kind == "overload" else None
            try:
                for index in range(count):
                    long = kind in ("mixed", "overload") and index % 2 == 1
                    prompt = LONG_PROMPT if long else SHORT_PROMPT
                    limit = 256 if kind == "overload" else 96 if long else 32
                    futures.append(pool.submit(_request, endpoint.root, endpoint.credential,
                                               prompt, limit, timeout, "long" if long else "short",
                                               start_gate, operator))
            finally:
                if start_gate is not None:
                    start_gate.set()
            for future in as_completed(futures):
                requests.append(future.result())
    finally:
        stop.set()
        worker.join(timeout=min(telemetry.timeout + 1, 31))
    elapsed = time.monotonic() - start
    try:
        record(telemetry.snapshot())
    except (ClientError, ValueError):
        sample_errors += 1
    launches = [row["_launch_ns"] for row in requests if "_launch_ns" in row]
    if launches:
        for row in requests:
            # The monotonic clock has meaning only inside this run; expose phase-relative offsets.
            row["launch_offset_s"] = round((row.pop("_launch_ns") / 1e9) - start, 4)
    launch_spread = round((max(launches) - min(launches)) / 1e9, 4) if launches else None
    latencies = [r["latency_s"] for r in requests if r["ok"]]
    ttfts = [r["ttft_s"] for r in requests if r["ok"] and r["ttft_s"] is not None]
    tokens = [r["completion_tokens"] for r in requests if r["ok"]]
    token_count_complete = (bool(tokens) and all(type(value) is int for value in tokens) and
                            (kind == "overload" or len(tokens) == len(requests)))
    output_tokens = sum(tokens) if token_count_complete else None
    rejected = sum(r["error_code"] == "admission_full" for r in requests)
    timed_out_count = sum(r["error_code"] == "queue_wait_timeout" for r in requests)
    expected_overload = rejected + timed_out_count
    unexpected_failures = sum(not r["ok"] and r["error_code"] not in
                              ("admission_full", "queue_wait_timeout") for r in requests)
    passed = (sum(bool(r["ok"]) for r in requests) > 0 and expected_overload > 0 and
              unexpected_failures == 0) if kind == "overload" else all(r["ok"] for r in requests)
    def counter_delta(key):
        if (first_sample is None or last_sample is None or
                first_sample.get("instance_id") != last_sample.get("instance_id") or
                type(first_sample.get(key)) is not int or type(last_sample.get(key)) is not int):
            return None
        return last_sample[key] - first_sample[key] if last_sample[key] >= first_sample[key] else None
    return {"kind": kind, "requested": count, "client_workers": concurrency,
            "passed": passed,
            "duration_s": round(elapsed, 4), "requests": requests, "telemetry_samples": samples,
            "telemetry_sample_errors": sample_errors, "telemetry_samples_omitted": samples_omitted,
            "summary": {"successful": sum(bool(r["ok"]) for r in requests),
                        "launch_spread_s": launch_spread,
                        "failed": unexpected_failures if kind == "overload" else sum(not r["ok"] for r in requests),
                        "expected_overload": expected_overload if kind == "overload" else 0,
                        "admission_full": rejected if kind == "overload" else 0,
                        "queue_wait_timeout": timed_out_count if kind == "overload" else 0,
                        "admission_rejected_delta": counter_delta("admission_rejected"),
                        "queue_timeouts_delta": counter_delta("queue_timeouts"),
                        "p50_latency_s": _percentile(latencies, 0.5),
                        "p95_latency_s": _percentile(latencies, 0.95),
                        "p50_ttft_s": _percentile(ttfts, 0.5),
                        "p95_ttft_s": _percentile(ttfts, 0.95),
                        "output_tokens": output_tokens,
                        "output_tok_s": round(output_tokens / elapsed, 3)
                        if output_tokens is not None and elapsed > 0 else None,
                        "peak_active": peaks.get("active"), "peak_queued": peaks.get("queued"),
                        "peak_waiting": peaks.get("waiting"),
                        "peak_occupied_slots": peaks.get("occupied_slots"), "peak_groups": peaks.get("groups"),
                        "peak_admission_rejected": peaks.get("admission_rejected")}}


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _progress_write(path: Path, record: dict, *, first: bool = False) -> None:
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > 4096:
        raise ProfileError("benchmark progress record exceeded 4 KiB", 6)
    if first:
        _write_new(path, encoded)
        return
    temporary = path.with_name(".progress-" + uuid.uuid4().hex + ".tmp")
    try:
        _write_new(temporary, encoded)
        _atomic_mod.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def progress(path: Path) -> dict:
    """Read only bounded phase evidence; a running record does not prove liveness."""
    target = (path / "progress.json" if path.is_dir() else
              path.with_name("progress.json") if path.name == "results.json" else path)
    try:
        details = target.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > 4096 or \
                (os.name == "posix" and (posix_owner.owner_only_wrong(target, details))):
            raise ProfileError(posix_owner.owner_only_text("benchmark progress must be an owner-only regular file under 4 KiB", target), 5)
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("cannot read benchmark progress", 5) from None
    phases = record.get("planned_phases") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or set(record) != {"schema_version", "kind", "suite", "instance_id",
                                                "started_at", "updated_at", "state", "planned_phases",
                                                "completed_phases", "current_phase", "last_completed",
                                                "successful"} or
            type(record.get("schema_version")) is not int or record["schema_version"] != 1 or
            record.get("kind") != "strata.serving.benchmark.progress" or
            record.get("suite") not in SUITES or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("instance_id"))) or
            not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00",
                             str(record.get("started_at"))) or
            not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00",
                             str(record.get("updated_at"))) or
            record.get("state") not in ("running", "report_saved", "cancelled") or
            not isinstance(phases, list) or not 1 <= len(phases) <= 21 or
            any(phase not in ("warmup", "serial", "mixed", "overload", "recovery",
                              "cancel_active", "cancel_queued", "cancel_prefill") for phase in phases) or
            type(record.get("completed_phases")) is not int or
            not 0 <= record["completed_phases"] <= len(phases) or
            record.get("current_phase") not in (None, *phases) or
            record.get("last_completed") not in (None, *phases) or
            (record["state"] == "running" and record.get("successful") is not None) or
            (record["state"] in ("report_saved", "cancelled") and type(record.get("successful")) is not bool) or
            (record["state"] == "cancelled" and record.get("successful") is not False)):
        raise ProfileError("benchmark progress has an unsupported shape", 5)
    if record["state"] in ("report_saved", "cancelled"):
        report = load(target.parent)
        if (report["suite"] != record["suite"] or report["instance_id"] != record["instance_id"] or
                report["successful"] != record["successful"] or
                report.get("planned_phases") != record["planned_phases"] or
                len(report["phases"]) != record["completed_phases"] or
                bool(report.get("cancelled")) != (record["state"] == "cancelled")):
            raise ProfileError("benchmark progress and report do not match", 5)
    return record


def run(endpoint: Endpoint, telemetry: Endpoint, out: Path, *, suite: str = "serving-smoke",
        concurrency: int = 4, repeat: int = 1, request_timeout: float = 120,
        profile_identity: dict | None = None, condition: str | None = None,
        require_managed_match: bool = False) -> dict:
    if suite not in SUITES:
        raise ProfileError("unknown serving benchmark suite")
    maximum = 64 if suite == "parallel-overload-v1" else 16
    if not 1 <= concurrency <= maximum or not 1 <= repeat <= 5 or not 1 <= request_timeout <= 600:
        raise ProfileError(f"concurrency must be 1..{maximum}, repeat 1..5 and request timeout 1..600 seconds")
    if condition is not None and not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", condition):
        raise ProfileError("benchmark condition must be a lowercase name up to 40 characters")
    if out.exists() or out.is_symlink():
        raise ProfileError("benchmark output directory already exists", 2)
    before = telemetry.snapshot()
    instance = (before.get("instance") or {}).get("id")
    if not instance or (before.get("model") or {}).get("loaded") is not True:
        raise ProfileError("benchmark needs a loaded Strata instance with versioned telemetry", 5)
    managed_before = None
    if require_managed_match:
        managed_before = _managed_match(profile_identity or {}, endpoint.root, before)
        if not managed_before["matched"]:
            raise ProfileError("managed benchmark preflight failed: " + managed_before["reason"], 5)
    if suite == "parallel-overload-v1":
        work = before.get("work") or {}
        limit = work.get("admission_limit")
        deadline = work.get("queue_timeout_s")
        if (type(limit) is not int or limit < 1 or type(deadline) not in (int, float) or
                not math.isfinite(deadline) or deadline <= 0):
            raise ProfileError("overload suite requires a server with max_queue and queue_timeout configured", 5)
        if concurrency <= limit:
            raise ProfileError(f"overload concurrency must exceed the admission limit ({limit})", 2)
    slots = None
    if suite in ("parallel-cancel-v1", "parallel-prefill-cancel-v1"):
        work = before.get("work") or {}
        slots = work.get("slots_effective")
        if (type(slots) is not int or not 1 <= slots <= 16 or
                type(work.get("client_disconnect_cancels")) is not int):
            raise ProfileError("cancellation suite requires 1..16 effective slots and disconnect telemetry", 5)
        if suite == "parallel-cancel-v1":
            if (type(work.get("max_queue")) is not int or work["max_queue"] < 1 or
                    type(work.get("queue_timeout_s")) not in (int, float) or
                    not math.isfinite(work["queue_timeout_s"]) or work["queue_timeout_s"] < 2):
                raise ProfileError("cancellation suite requires max_queue >= 1 and "
                                   "queue_timeout >= 2 seconds", 5)
            if concurrency < slots:
                raise ProfileError(f"cancellation concurrency must cover all {slots} effective slots", 2)
        elif slots < 2 or concurrency < 2:
            raise ProfileError("prefill cancellation suite requires at least two effective slots and concurrency >= 2", 5)
    _private_mkdir(out, parents=False, exist_ok=False)
    phases = []
    started = _utc_now()
    planned_phases = _planned_phase_kinds(suite, repeat)
    progress_record = {"schema_version": 1, "kind": "strata.serving.benchmark.progress",
                       "suite": suite, "instance_id": instance, "started_at": started,
                       "updated_at": started, "state": "running", "planned_phases": planned_phases,
                       "completed_phases": 0, "current_phase": None,
                       "last_completed": None, "successful": None}
    progress_path = out / "progress.json"
    _progress_write(progress_path, progress_record, first=True)

    def mark(*, current: str | None = None, completed: str | None = None) -> None:
        if completed is not None:
            progress_record["completed_phases"] = len(phases)
            progress_record["last_completed"] = completed
        progress_record["current_phase"] = current
        progress_record["updated_at"] = _utc_now()
        _progress_write(progress_path, progress_record)

    operator = OperatorCancellation(out)
    with operator:
        try:
            for kind in SUITES[suite]:
                if operator.event.is_set():
                    break
                if kind in ("cancel_active", "cancel_queued", "cancel_prefill"):
                    mark(current=kind)
                    phase = _cancel_phase(endpoint, telemetry, kind=kind, slots=slots,
                                          timeout=request_timeout, operator=operator)
                    if operator.event.is_set():
                        break
                    phases.append(phase)
                    mark(completed=kind)
                    continue
                for _ in range(1 if kind in ("warmup", "recovery") else repeat):
                    if operator.event.is_set():
                        break
                    mark(current=kind)
                    count = 1 if kind in ("warmup", "recovery") else 2 if kind == "serial" else concurrency
                    workers = concurrency if kind in ("mixed", "overload") else 1
                    phase = _phase(endpoint, telemetry, kind=kind, count=count,
                                   concurrency=workers, timeout=request_timeout, operator=operator)
                    if operator.event.is_set():
                        break
                    phases.append(phase)
                    mark(completed=kind)
                    if not phases[-1]["passed"] and kind != "overload":
                        break
                if operator.event.is_set() or (phases and not phases[-1]["passed"] and kind != "overload"):
                    break
        finally:
            try:
                after = telemetry.snapshot()
            except (ClientError, ValueError):
                after = {}
            cancelled = operator.event.is_set() or operator.path.exists()
            success = (not cancelled and len(phases) == len(planned_phases)
                       and all(p["passed"] for p in phases)
                       and (after.get("instance") or {}).get("id") == instance)
            managed_after = (_managed_match(profile_identity or {}, endpoint.root, after)
                             if require_managed_match else None)
            if require_managed_match and (not managed_after["matched"] or
                                          managed_after.get("instance_id") != managed_before.get("instance_id")):
                success = False
            report = {"schema_version": SCHEMA, "kind": "strata.serving.benchmark",
                      "run_id": uuid.uuid4().hex, "suite": suite, "started_at": started,
                      "finished_at": _utc_now(), "server_root": endpoint.root,
                      "condition": condition, "environment": _environment(before),
                      "profile_identity": profile_identity,
                      "profile_running_match": "managed_launch_checked" if require_managed_match and success else "unverified",
                      "profile_running_check": {"before": managed_before, "after": managed_after}
                      if require_managed_match else None,
                      "instance_id": instance, "model_id": (before.get("model") or {}).get("id"),
                      "planned_phases": planned_phases,
                      "before": _sample(before), "after": _sample(after), "phases": phases,
                      "successful": success, "cancelled": cancelled,
                      "limitations": ["client-observed timings; no synchronized backend clock",
                                      "missing telemetry samples are not zero",
                                      "no engine batch-window or stage mapping measurement",
                                      "condition is operator-supplied; monitor attachment is not verified",
                                      "environment fields come from server telemetry, not hardware attestation",
                                      "managed launch is checked only at the beginning and end; post-start changes are not continuously verified"
                                      if require_managed_match else
                                      "selected local profile is not authenticated as the running instance's config",
                                      "driver and model artifact hashes not verified by this suite"] +
                                     (["client-disconnect cancellation is inferred from a server counter and "
                                       "request disappearance; run without unrelated clients"]
                                      if suite in ("parallel-cancel-v1", "parallel-prefill-cancel-v1") else [])}
            encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if len(encoded) > MAX_REPORT_BYTES:
                raise ProfileError("benchmark report exceeded 4 MiB", 6)
            _write_new(out / "results.json", encoded)
            progress_record.update({"state": "cancelled" if cancelled else "report_saved", "successful": success,
                                    "current_phase": None, "updated_at": _utc_now()})
            _progress_write(progress_path, progress_record)
    return report


def load(path: Path) -> dict:
    target = path / "results.json" if path.is_dir() else path
    try:
        if not stat.S_ISREG(target.lstat().st_mode) or target.stat().st_size > MAX_REPORT_BYTES:
            raise ProfileError("benchmark report must be a regular file under 4 MiB", 5)
        result = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("cannot read benchmark report", 5) from None
    planned = result.get("planned_phases") if isinstance(result, dict) else None
    if not isinstance(result, dict) or type(result.get("schema_version")) is not int or \
            result.get("schema_version") != SCHEMA or result.get("kind") != "strata.serving.benchmark" or \
            not isinstance(result.get("phases"), list) or len(result["phases"]) > 21 or \
            not re.fullmatch(r"[0-9a-f]{32}", str(result.get("run_id", ""))) or \
            not isinstance(result.get("suite"), str) or result["suite"] not in SUITES or \
            (result.get("condition") is not None and
             (not isinstance(result["condition"], str) or
              not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", result["condition"]))) or \
            not re.fullmatch(r"[0-9a-f]{32}", str(result.get("instance_id", ""))) or \
            type(result.get("successful")) is not bool or \
            ("cancelled" in result and type(result["cancelled"]) is not bool) or \
            (result.get("cancelled") is True and result["successful"] is not False) or \
            (result.get("model_id") is not None and
             (not isinstance(result["model_id"], str) or len(result["model_id"]) > 120 or
              not result["model_id"].isprintable())) or \
            not all(isinstance(phase, dict) and isinstance(phase.get("summary"), dict) and
                    phase.get("kind") in ("warmup", "serial", "mixed", "overload", "recovery",
                                          "cancel_active", "cancel_queued", "cancel_prefill") and
                    type(phase.get("requested")) is int and
                    type(phase.get("client_workers")) is int and
                    type(phase["summary"].get("successful")) is int and
                    type(phase["summary"].get("failed")) is int and
                    ("passed" not in phase or type(phase["passed"]) is bool) and
                    (phase.get("kind") not in ("overload", "cancel_active", "cancel_queued", "cancel_prefill") or
                     type(phase.get("passed")) is bool) and
                    (phase.get("kind") not in ("cancel_active", "cancel_queued", "cancel_prefill") or
                     ("client_disconnect_delta" in phase["summary"] and
                      (phase["summary"]["client_disconnect_delta"] is None or
                       type(phase["summary"]["client_disconnect_delta"]) is int) and
                      type(phase["summary"].get("target_state_seen")) is bool and
                      (phase.get("kind") != "cancel_prefill" or
                       type(phase["summary"].get("prompt_read_seen")) is bool) and
                      type(phase["summary"].get("settled")) is bool)) and
                    all(key in phase["summary"] for key in ("p95_latency_s", "output_tok_s",
                                                            "peak_occupied_slots"))
                    for phase in result["phases"]):
        raise ProfileError("unsupported benchmark report", 5)
    kinds = [phase["kind"] for phase in result["phases"]]
    valid_plans = [_planned_phase_kinds(result["suite"], repeat) for repeat in range(1, 6)]
    if (planned is not None and
            (not isinstance(planned, list) or planned not in valid_plans or
             kinds != planned[:len(kinds)] or
             (result["successful"] and len(kinds) != len(planned)))) or \
            (result["successful"] and
             (kinds not in valid_plans or any(phase.get("passed") is False
                                              for phase in result["phases"]))):
        raise ProfileError("unsupported benchmark report", 5)
    return result


def summary(report: dict) -> dict:
    fields = ("successful", "failed", "expected_overload", "admission_full", "queue_wait_timeout",
              "launch_spread_s",
              "admission_rejected_delta",
              "queue_timeouts_delta", "client_disconnect_delta", "target_state_seen", "client_closed",
              "first_content_seen", "companion_successful", "settled",
              "p50_latency_s", "p95_latency_s", "p50_ttft_s", "p95_ttft_s",
              "output_tokens", "output_tok_s", "peak_active", "peak_queued", "peak_waiting",
              "peak_occupied_slots", "peak_groups", "peak_admission_rejected")
    rows = []
    for phase in report["phases"]:
        rows.append({"kind": phase["kind"], "requested": phase["requested"],
                     "passed": phase.get("passed", phase["summary"]["failed"] == 0),
                     "client_workers": phase["client_workers"],
                     **{key: phase["summary"].get(key) for key in fields}})
    return {"run_id": report["run_id"], "suite": report["suite"],
            "condition": report.get("condition"),
            "profile_running_match": report.get("profile_running_match", "unverified"),
            "instance_id": report["instance_id"], "model_id": report["model_id"],
            "successful": report["successful"], "cancelled": report.get("cancelled", False),
            "phases": rows,
            "planned_phase_count": len(report["planned_phases"])
            if isinstance(report.get("planned_phases"), list) else None,
            "completed_phase_count": len(rows)}


def _workload_signature(phase: dict) -> tuple:
    rows = phase.get("requests")
    if not isinstance(rows, list) or len(rows) > 64:
        raise ProfileError("benchmark comparison needs bounded request evidence", 5)
    signatures = []
    for row in rows:
        if not isinstance(row, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("prompt_sha256", ""))) or \
                type(row.get("max_tokens")) is not int or not 1 <= row["max_tokens"] <= 65536 or \
                not isinstance(row.get("label"), str) or not re.fullmatch(r"[a-z_]{1,40}", row["label"]):
            raise ProfileError("benchmark comparison needs valid request fingerprints", 5)
        signatures.append((row["prompt_sha256"], row["max_tokens"], row["label"]))
    return (phase["kind"], phase["requested"], phase["client_workers"], tuple(sorted(signatures)))


def _metric(value: object) -> float | None:
    if type(value) not in (int, float) or not 0 <= value <= 1e9:
        return None
    return float(value) if math.isfinite(value) else None


def _change(baseline: object, candidate: object) -> dict:
    old, new = _metric(baseline), _metric(candidate)
    return {"baseline": old, "candidate": new,
            "delta": round(new - old, 4) if old is not None and new is not None else None,
            "delta_percent": round(100 * (new / old - 1), 2)
            if old is not None and new is not None and old > 0 else None}


def compare(baseline: dict, candidate: dict) -> dict:
    """Compare two saved runs without claiming hardware or monitor-state attestation."""
    if not baseline["successful"] or not candidate["successful"]:
        raise ProfileError("benchmark comparison requires two successful runs", 5)
    if baseline["run_id"] == candidate["run_id"]:
        raise ProfileError("benchmark comparison needs two distinct runs", 2)
    if (baseline["suite"] != candidate["suite"] or baseline["model_id"] != candidate["model_id"] or
            not baseline["model_id"]):
        raise ProfileError("benchmark suite and model ID must match", 5)
    old_phases, new_phases = baseline["phases"], candidate["phases"]
    if len(old_phases) != len(new_phases) or not old_phases:
        raise ProfileError("benchmark phase counts differ", 5)
    if any(_workload_signature(old) != _workload_signature(new)
           for old, new in zip(old_phases, new_phases)):
        raise ProfileError("benchmark request workload or concurrency differs", 5)

    warnings = []
    old_profile = baseline.get("profile_identity")
    new_profile = candidate.get("profile_identity")
    old_profile = old_profile if isinstance(old_profile, dict) else {}
    new_profile = new_profile if isinstance(new_profile, dict) else {}
    old_digest, new_digest = old_profile.get("source_sha256"), new_profile.get("source_sha256")
    if old_digest and new_digest:
        if old_digest != new_digest:
            raise ProfileError("benchmark source config digests differ", 5)
    else:
        warnings.append("source config identity missing from one or both reports")
    if (baseline.get("profile_running_match") != "managed_launch_checked" or
            candidate.get("profile_running_match") != "managed_launch_checked"):
        warnings.append("source config was not attested as the running server config")

    old_env = baseline.get("environment")
    new_env = candidate.get("environment")
    old_env = old_env if isinstance(old_env, dict) else {}
    new_env = new_env if isinstance(new_env, dict) else {}
    for key in ("model_context", "slots_effective", "groups", "max_queue", "queue_timeout_s",
                "selected_devices", "stages", "devices"):
        old, new = old_env.get(key), new_env.get(key)
        if old is None or new is None or old == [] or new == []:
            warnings.append(f"{key} identity is unavailable in one or both reports")
        elif old != new:
            raise ProfileError(f"benchmark reported environment differs: {key}", 5)
    warnings.append("driver, engine binary, model artifact and monitor attachment are not attested by these reports")

    metric_keys = ("p95_ttft_s", "p95_latency_s", "output_tok_s", "admission_full",
                   "queue_wait_timeout", "admission_rejected_delta", "queue_timeouts_delta",
                   "client_disconnect_delta", "companion_successful", "launch_spread_s",
                   "peak_waiting", "peak_queued")
    phases = []
    for index, (old, new) in enumerate(zip(old_phases, new_phases)):
        phases.append({"index": index, "kind": old["kind"], "requested": old["requested"],
                       "client_workers": old["client_workers"],
                       "duration_s": _change(old.get("duration_s"), new.get("duration_s")),
                       "metrics": {key: _change(old["summary"].get(key), new["summary"].get(key))
                                   for key in metric_keys}})
    primary_kind = next((kind for kind in ("mixed", "overload", "serial", "cancel_prefill",
                                           "cancel_active", "cancel_queued", "recovery")
                         if any(phase["kind"] == kind for phase in phases)), phases[0]["kind"])
    primary = [phase for phase in phases if phase["kind"] == primary_kind]
    if len(primary) < 3:
        warnings.append("fewer than three measured primary phases; repeat the workload for qualification")
    medians = {}
    for key in ("p95_ttft_s", "p95_latency_s", "output_tok_s"):
        old_values = [phase["metrics"][key]["baseline"] for phase in primary]
        new_values = [phase["metrics"][key]["candidate"] for phase in primary]
        medians[key] = _change(statistics.median(old_values) if all(x is not None for x in old_values) else None,
                               statistics.median(new_values) if all(x is not None for x in new_values) else None)
    return {"schema_version": 1, "kind": "strata.serving.benchmark_comparison",
            "suite": baseline["suite"], "model_id": baseline["model_id"],
            "baseline": {"run_id": baseline["run_id"], "condition": baseline.get("condition"),
                         "profile_running_match": baseline.get("profile_running_match", "unverified")},
            "candidate": {"run_id": candidate["run_id"], "condition": candidate.get("condition"),
                          "profile_running_match": candidate.get("profile_running_match", "unverified")},
            "workload_match": True, "qualification": "provisional",
            "primary_kind": primary_kind, "primary_phases": len(primary),
            "median_primary_metrics": medians, "phases": phases, "warnings": warnings}


def format_comparison(value: dict) -> str:
    def pair(metric: dict) -> str:
        return (f"{metric['baseline']:g} -> {metric['candidate']:g}"
                if metric["baseline"] is not None and metric["candidate"] is not None else "unavailable")

    lines = [f"{value['suite']} - provisional comparison ({value['primary_phases']} "
             f"{value['primary_kind']} phases)",
             f"Baseline {value['baseline']['condition'] or value['baseline']['run_id']}  "
             f"Candidate {value['candidate']['condition'] or value['candidate']['run_id']}",
             f"Managed launch: baseline {value['baseline']['profile_running_match']}, "
             f"candidate {value['candidate']['profile_running_match']}"]
    for label, key in (("p95 TTFT", "p95_ttft_s"), ("p95 latency", "p95_latency_s"),
                       ("output throughput", "output_tok_s")):
        metric = value["median_primary_metrics"][key]
        if metric["baseline"] is None or metric["candidate"] is None:
            lines.append(f"{label}: unavailable")
        else:
            unit = " tok/s" if key == "output_tok_s" else " s"
            change = (f"{metric['delta_percent']:+.2f}%" if metric["delta_percent"] is not None
                      else "percent unavailable")
            lines.append(f"{label}: {metric['baseline']:.4f}{unit} -> "
                         f"{metric['candidate']:.4f}{unit} ({change})")
    for phase in value["phases"]:
        metrics = phase["metrics"]
        if phase["kind"] == "overload":
            lines.append(f"overload {phase['index']}: admission full {pair(metrics['admission_full'])}, "
                         f"queue timeout {pair(metrics['queue_wait_timeout'])}, "
                         f"duration {pair(phase['duration_s'])} s")
        elif phase["kind"].startswith("cancel_"):
            lines.append(f"{phase['kind']}: disconnects {pair(metrics['client_disconnect_delta'])}, "
                         f"companions {pair(metrics['companion_successful'])}, "
                         f"duration {pair(phase['duration_s'])} s")
    lines.extend(f"Note: {warning}" for warning in value["warnings"])
    return "\n".join(lines)


def format_summary(value: dict) -> str:
    def shown(value):
        return "--" if value is None else value

    condition = f" [{value['condition']}]" if value.get("condition") else ""
    outcome = "cancelled" if value.get("cancelled") else "passed" if value["successful"] else "failed"
    lines = [f"{value['suite']}{condition} - {outcome}",
             f"Instance {value['instance_id']}  Model {value['model_id']}",
             f"Config match {value.get('profile_running_match', 'unverified')}"]
    for phase in value["phases"]:
        if phase["kind"] in ("cancel_active", "cancel_queued"):
            lines.append(f"{phase['kind']}: {'passed' if phase['passed'] else 'failed'}  "
                         f"companions {shown(phase.get('companion_successful'))}")
        else:
            lines.append(f"{phase['kind']}: {phase['successful']}/{phase['requested']} requests  "
                         f"p95 TTFT {shown(phase.get('p95_ttft_s'))} s  "
                         f"p95 latency {shown(phase.get('p95_latency_s'))} s  "
                         f"output {shown(phase.get('output_tok_s'))} tok/s")
        if phase["kind"] == "overload":
            lines.append(f"  launch spread {shown(phase.get('launch_spread_s'))} s  "
                         f"rejected {shown(phase.get('admission_full'))}  "
                         f"timed out {shown(phase.get('queue_wait_timeout'))}  "
                         f"unexpected failures {shown(phase.get('failed'))}  "
                         f"rejected delta {shown(phase.get('admission_rejected_delta'))}  "
                         f"timeout delta {shown(phase.get('queue_timeouts_delta'))}")
        if phase["kind"] in ("cancel_active", "cancel_queued"):
            lines.append(f"  target seen {shown(phase.get('target_state_seen'))}  "
                         f"client closed {shown(phase.get('client_closed'))}  "
                         f"disconnect delta {shown(phase.get('client_disconnect_delta'))}  "
                         f"companions completed {shown(phase.get('companion_successful'))}  "
                         f"settled {shown(phase.get('settled'))}")
        lines.append(f"  peak active {shown(phase.get('peak_active'))}  "
                     f"occupied {shown(phase.get('peak_occupied_slots'))}  "
                     f"engine waiting {shown(phase.get('peak_waiting'))}  "
                     f"queued {shown(phase.get('peak_queued'))}  groups {shown(phase.get('peak_groups'))}")
    return "\n".join(lines)


def _md_cell(value: object, *, limit: int = 120) -> str:
    """Bound an untrusted report label before placing it in a Markdown cell."""
    if value is None:
        return "unavailable"
    raw = str(value)
    raw = " ".join(part for part in raw.split() if part)
    raw = "".join(char for char in raw[:limit] if char.isprintable())
    for char in ("\\", "|", "*", "_", "[", "]", "<", ">", "`"):
        raw = raw.replace(char, "\\" + char)
    return raw or "unavailable"


def _md_number(value: object, *, digits: int = 3) -> str:
    measured = _metric(value)
    if measured is None:
        return "unavailable"
    return str(int(measured)) if measured.is_integer() else f"{measured:.{digits}f}"


def _md_time(value: object) -> str:
    return value if isinstance(value, str) and len(value) <= 40 and re.fullmatch(
        r"[0-9TZ:+.\-]+", value) else "unavailable"


def markdown_report(report: dict) -> str:
    """Render a shareable, bounded summary without endpoint, paths or request text."""
    compact = summary(report)
    incomplete = (compact["planned_phase_count"] is not None and
                  compact["completed_phase_count"] < compact["planned_phase_count"])
    result_label = ("Cancelled" if compact["cancelled"] else "Passed" if compact["successful"]
                    else "Incomplete" if incomplete else "Failed")
    rows = ["# Strata serving benchmark", "",
            "This report summarizes client-observed requests and server-reported telemetry. "
            "It is not a hardware or model-quality qualification.", "",
            "| Run | Value |", "| --- | --- |",
            f"| Result | {result_label} |",
            f"| Phases | {compact['completed_phase_count']} of {compact['planned_phase_count'] if compact['planned_phase_count'] is not None else 'unavailable'} |",
            f"| Suite | {_md_cell(compact['suite'])} |",
            f"| Run ID | {_md_cell(compact['run_id'])} |",
            f"| Started (UTC) | {_md_time(report.get('started_at'))} |",
            f"| Finished (UTC) | {_md_time(report.get('finished_at'))} |",
            f"| Operator condition | {_md_cell(compact.get('condition'))} |",
            f"| Model ID | {_md_cell(compact.get('model_id'))} |",
            f"| Instance ID | {_md_cell(compact.get('instance_id'))} |",
            f"| Managed launch check | {_md_cell(compact.get('profile_running_match'))} |",
            "", "## Measured phases", "",
            "| Phase | Clients | Succeeded | Failed | p95 TTFT (s) | p95 latency (s) | Output (tok/s) | Peak active | Peak queued | Peak slots | Peak groups | Result |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for phase in compact["phases"][:21]:
        rows.append("| " + " | ".join((
            _md_cell(phase.get("kind")), _md_number(phase.get("client_workers")),
            _md_number(phase.get("successful")), _md_number(phase.get("failed")),
            _md_number(phase.get("p95_ttft_s")), _md_number(phase.get("p95_latency_s")),
            _md_number(phase.get("output_tok_s")), _md_number(phase.get("peak_active")),
            _md_number(phase.get("peak_queued")), _md_number(phase.get("peak_occupied_slots")),
            _md_number(phase.get("peak_groups")), "Passed" if phase.get("passed") else "Failed",
        )) + " |")
    exceptional = [phase for phase in compact["phases"] if phase.get("kind") in
                   ("overload", "cancel_active", "cancel_queued", "cancel_prefill")]
    if exceptional:
        rows += ["", "## Admission and cancellation observations", "",
                 "| Phase | Expected overload | Admission full | Queue timeout | Disconnect delta | Companion completed | Settled |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for phase in exceptional:
            rows.append("| " + " | ".join((
                _md_cell(phase.get("kind")), _md_number(phase.get("expected_overload")),
                _md_number(phase.get("admission_full")), _md_number(phase.get("queue_wait_timeout")),
                _md_number(phase.get("client_disconnect_delta")),
                _md_number(phase.get("companion_successful")),
                "Yes" if phase.get("settled") is True else
                "No" if phase.get("settled") is False else "unavailable",
            )) + " |")
    environment = report.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    selected = environment.get("selected_devices")
    selected = ", ".join(_md_number(item) for item in selected[:16]) if isinstance(selected, list) else "unavailable"
    rows += ["", "## Server-reported environment", "",
             "| Field | Value |", "| --- | --- |",
             f"| Context tokens | {_md_number(environment.get('model_context'))} |",
             f"| Effective slots | {_md_number(environment.get('slots_effective'))} |",
             f"| Pipeline groups | {_md_number(environment.get('groups'))} |",
             f"| Maximum queue | {_md_number(environment.get('max_queue'))} |",
             f"| Selected device indices | {selected or 'unavailable'} |"]
    devices = environment.get("devices")
    if isinstance(devices, list):
        for device in devices[:16]:
            if isinstance(device, dict):
                rows.append(f"| Reported device {_md_number(device.get('index'))} | "
                            f"{_md_cell(device.get('name'))} |")
    rows += ["", "## Measurement limits", "",
             "- Missing readings appear as `unavailable`; they are not zero.",
             "- The condition label is operator supplied. Monitor attachment is not verified.",
             "- Device names and serving configuration come from server telemetry. Driver, topology, "
             "engine binary and model artifact hashes are not attested here.",
             "- A managed launch check, when present, covers the beginning and end of the run only.",
             "- Timings are observed by the client. This report does not measure GPU kernel occupancy "
             "or independently synchronize backend clocks.", ""]
    return "\n".join(rows)


def markdown_comparison(comparison: dict) -> str:
    """Render the workload-checked comparison with provisional status visible."""
    rows = ["# Strata serving benchmark comparison", "",
            "**Provisional comparison.** The saved runs passed workload checks; environment "
            "and monitor attachment still need independent qualification.", "",
            "| Run | Baseline | Candidate |", "| --- | --- | --- |",
            f"| Suite | {_md_cell(comparison.get('suite'))} | {_md_cell(comparison.get('suite'))} |",
            f"| Model ID | {_md_cell(comparison.get('model_id'))} | {_md_cell(comparison.get('model_id'))} |",
            f"| Run ID | {_md_cell(comparison['baseline'].get('run_id'))} | {_md_cell(comparison['candidate'].get('run_id'))} |",
            f"| Operator condition | {_md_cell(comparison['baseline'].get('condition'))} | {_md_cell(comparison['candidate'].get('condition'))} |",
            f"| Managed launch check | {_md_cell(comparison['baseline'].get('profile_running_match'))} | {_md_cell(comparison['candidate'].get('profile_running_match'))} |",
            "", f"Primary phase: {_md_cell(comparison.get('primary_kind'))} "
            f"({_md_number(comparison.get('primary_phases'))} measured phase(s)).", "",
            "| Median primary metric | Baseline | Candidate | Change |",
            "| --- | ---: | ---: | ---: |"]
    for label, key, unit in (("p95 TTFT", "p95_ttft_s", "s"),
                             ("p95 latency", "p95_latency_s", "s"),
                             ("Output throughput", "output_tok_s", "tok/s")):
        metric = comparison["median_primary_metrics"][key]
        change = metric.get("delta_percent")
        change_text = (f"{change:+.2f}%" if type(change) in (int, float) and
                       math.isfinite(change) and abs(change) <= 1e9 else "unavailable")
        rows.append(f"| {label} ({unit}) | {_md_number(metric.get('baseline'))} | "
                    f"{_md_number(metric.get('candidate'))} | {change_text} |")
    rows += ["", "## Per-phase results", "",
             "| Phase | Clients | Baseline p95 TTFT (s) | Candidate p95 TTFT (s) | Baseline output (tok/s) | Candidate output (tok/s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase in comparison["phases"][:21]:
        metrics = phase["metrics"]
        rows.append("| " + " | ".join((
            _md_cell(phase.get("kind")), _md_number(phase.get("client_workers")),
            _md_number(metrics["p95_ttft_s"].get("baseline")),
            _md_number(metrics["p95_ttft_s"].get("candidate")),
            _md_number(metrics["output_tok_s"].get("baseline")),
            _md_number(metrics["output_tok_s"].get("candidate")),
        )) + " |")
    rows += ["", "## Comparison limits", ""]
    rows.extend(f"- {_md_cell(warning, limit=240)}" for warning in comparison.get("warnings", [])[:32])
    rows += ["- A percentage change describes the observed result; it does not by itself "
             "establish an improvement or hardware qualification.", ""]
    return "\n".join(rows)


def write_markdown(path: Path, content: str) -> None:
    """Create a shareable report file without replacing an earlier result."""
    encoded = content.encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ProfileError("Markdown benchmark report exceeded 64 KiB", 6)
    if path.exists() or path.is_symlink():
        raise ProfileError("Markdown output file already exists", 2)
    _write_new(path, encoded)
