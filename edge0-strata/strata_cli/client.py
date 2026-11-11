"""Bounded, read-only transport and normalization for Strata telemetry."""

from __future__ import annotations

import json
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


MAX_BODY = 2 * 1024 * 1024


def complete_slots(work: dict) -> list[dict] | None:
    """Return the effective slot set only when its identity and states are sound."""
    slots, effective = work.get("slots"), work.get("slots_effective")
    if (not isinstance(slots, list) or not isinstance(effective, int) or
            isinstance(effective, bool) or effective <= 0 or len(slots) != effective):
        return None
    identities: set[int] = set()
    for row in slots:
        if not isinstance(row, dict):
            return None
        index, state = row.get("slot"), row.get("state")
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < effective or
                index in identities or state not in ("idle", "reading", "decoding")):
            return None
        identities.add(index)
    return slots


class ClientError(Exception):
    def __init__(self, message: str, exit_code: int = 3, *, http_status: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.http_status = http_status


def server_url(raw: str) -> str:
    """Require a server root and prevent credential/path ambiguity."""
    if not isinstance(raw, str):
        raise ValueError("URL must be an http(s) server root")
    u = urllib.parse.urlsplit(raw)
    if u.scheme not in ("http", "https") or not u.hostname or "@" in u.netloc:
        raise ValueError("URL must be an http(s) server root without embedded credentials")
    if u.path not in ("", "/") or u.query or u.fragment:
        raise ValueError("URL must be the server root, without a path, query or fragment")
    try:
        port = u.port
    except ValueError as e:
        raise ValueError("invalid server port") from e
    host = u.hostname.lower()
    local = host in ("localhost", "127.0.0.1", "::1")
    if u.scheme != "https" and not local:
        raise ValueError("remote monitoring requires HTTPS or a local SSH tunnel")
    authority = f"[{host}]" if ":" in host else host
    return f"{u.scheme}://{authority}{':' + str(port) if port is not None else ''}"


def url_prefix(value: str) -> str:
    """A saved reverse-proxy path, never an alternate origin or query."""
    if not isinstance(value, str) or len(value) > 256:
        raise ValueError("connection prefix must be a short URL path")
    if value in ("", "/"):
        return ""
    parts = value.split("/")
    if parts[0] or len(parts) < 2 or any(not re.fullmatch(r"[A-Za-z0-9._~-]+", part) or part in (".", "..")
                                               for part in parts[1:]):
        raise ValueError("connection prefix must use simple /path segments without traversal")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class Endpoint:
    root: str
    credential: str | None = None
    timeout: float = 3.0
    prefix: str = ""

    def __post_init__(self):
        object.__setattr__(self, "root", server_url(self.root))
        object.__setattr__(self, "prefix", url_prefix(self.prefix))
        if not 0 < self.timeout <= 30:
            raise ValueError("timeout must be between 0 and 30 seconds")

    @property
    def base_url(self) -> str:
        return self.root + self.prefix

    def get(self, path: str) -> dict[str, Any]:
        if path not in ("/v1/telemetry", "/v1/status", "/metrics", "/health"):
            raise ValueError("endpoint is not part of the read-only telemetry contract")
        headers = {"Accept": "application/json", "User-Agent": "Strata-Terminal/0.1"}
        if self.credential:
            headers["Authorization"] = "Bearer " + self.credential
        req = urllib.request.Request(self.base_url + path, headers=headers)
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    raise ClientError("server did not return JSON telemetry")
                body = response.read(MAX_BODY + 1)
                if len(body) > MAX_BODY:
                    raise ClientError("telemetry response exceeds the 2 MiB limit")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise ClientError("telemetry authentication or permission failed", 4, http_status=e.code) from e
            if e.code in (301, 302, 303, 307, 308):
                raise ClientError("telemetry endpoint redirected; use the final server root",
                                  http_status=e.code) from e
            raise ClientError(f"telemetry request failed: HTTP {e.code}", http_status=e.code) from e
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
            raise ClientError(f"cannot reach Strata server: {type(e).__name__}") from e
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError) as e:
            raise ClientError("server returned malformed JSON telemetry") from e
        if not isinstance(data, dict):
            raise ClientError("server returned a non-object telemetry response")
        return data

    def snapshot(self) -> dict[str, Any]:
        """Prefer the versioned endpoint; adapt the existing server if unavailable."""
        try:
            snapshot = self.get("/v1/telemetry")
        except ClientError as e:
            if e.http_status != 404:
                raise
        else:
            if snapshot.get("schema_version") != 1:
                raise ClientError("incompatible telemetry schema version", 5)
            return sanitize_telemetry(snapshot)
        try:
            metrics = self.get("/metrics")
        except ClientError as e:
            if e.http_status != 404:
                raise
            return sanitize_telemetry(normalize_status_only(self.get("/v1/status")))
        status = self.get("/v1/status")
        return sanitize_telemetry(normalize_legacy(metrics, status))


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _short(value: Any, limit: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    # Data is untrusted even if it came from the user's own model/server.
    return "".join(c if c.isprintable() and c not in "\x1b\x7f" else " " for c in value)[:limit]


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rows(value: Any, fields: tuple[str, ...], limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        out.append({key: _short(item.get(key)) if key in ("id", "request_id", "state", "name", "finish")
                    else _number(item.get(key)) for key in fields})
    return out


def _selected_devices(value: Any) -> list[int]:
    """Preserve the declared serving order without accepting ambiguous indices."""
    if not isinstance(value, list):
        return []
    selected: list[int] = []
    for index in value[:16]:
        if type(index) is int and 0 <= index <= 65535 and index not in selected:
            selected.append(index)
    return selected


def _replica_devices(value: Any) -> list[dict]:
    """Accept only a bounded, unambiguous configured card assignment."""
    if not isinstance(value, list):
        return []
    rows, seen_replicas, seen_devices = [], set(), set()
    for entry in value[:16]:
        if not isinstance(entry, dict):
            continue
        replica = entry.get("replica")
        raw = entry.get("devices")
        devices = _selected_devices(raw)
        if (type(replica) is not int or not 0 <= replica < 16 or replica in seen_replicas or
                not isinstance(raw, list) or not devices or len(devices) != len(raw) or
                any(device in seen_devices for device in devices)):
            continue
        rows.append({"replica": replica, "devices": devices})
        seen_replicas.add(replica)
        seen_devices.update(devices)
    return rows


def _replica_stages(value: Any, devices: list[dict]) -> list[dict]:
    """Keep child log ranges, resolving a card only inside that replica's configured set."""
    if not isinstance(value, list):
        return []
    cards = {row["replica"]: set(row["devices"]) for row in devices}
    rows, seen = [], set()
    for raw in value[:16]:
        if not isinstance(raw, dict):
            continue
        replica = raw.get("replica")
        if type(replica) is not int or not 0 <= replica < 16 or replica in seen:
            continue
        seen.add(replica)
        stages = _stage_rows(raw.get("stages"))
        allowed = cards.get(replica, set())
        for stage in stages:
            if stage["device_index"] not in allowed:
                stage["device_index"] = None
        rows.append({"replica": replica, "stages": stages,
                     "resolved": bool(stages) and all(stage["device_index"] is not None for stage in stages)})
    return rows


_ACTIVITY_EVENTS = frozenset(("queued", "admitted", "reading", "first_token", "finished",
                              "cancelled", "disconnected", "timeout", "error"))
_ACTIVITY_OUTCOMES = frozenset(("stop", "length", "cancel", "disconnect", "timeout", "error"))


def _activity(value: Any) -> dict:
    """Keep only the server's fixed-shape, bounded request transitions."""
    if not isinstance(value, dict) or type(value.get("last_seq")) is not int or value["last_seq"] < 0 \
            or not isinstance(value.get("events"), list):
        return {"available": False, "last_seq": None, "events": []}
    last_seq = value["last_seq"]
    events = []
    previous = 0
    for row in value["events"][-32:]:
        if not isinstance(row, dict):
            continue
        seq, at, request_id, event = (row.get(k) for k in ("seq", "at", "request_id", "event"))
        if (type(seq) is not int or seq <= previous or seq > last_seq or
                not isinstance(at, (int, float)) or isinstance(at, bool) or not math.isfinite(at) or
                not 0 <= at <= 4_102_444_800 or
                not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{12}", request_id) or
                not isinstance(event, str) or event not in _ACTIVITY_EVENTS):
            continue
        outcome = row.get("outcome")
        if outcome is not None and (not isinstance(outcome, str) or outcome not in _ACTIVITY_OUTCOMES):
            continue
        events.append({"seq": seq, "at": at, "request_id": request_id,
                       "event": event, "outcome": outcome})
        previous = seq
    return {"available": True, "last_seq": last_seq, "events": events}


def _stage_rows(value: Any) -> list[dict]:
    """Keep a bounded, contiguous stage plan; remote telemetry is untrusted."""
    if not isinstance(value, list) or not 2 <= len(value) <= 16:
        return []
    out, next_layer = [], 0
    for i, row in enumerate(value):
        if not isinstance(row, dict):
            return []
        stage, first, last = row.get("stage"), row.get("layer_start"), row.get("layer_end")
        ordinal, device = row.get("engine_ordinal"), row.get("device_index")
        if (not all(isinstance(v, int) and not isinstance(v, bool) for v in (stage, first, last, ordinal))
                or stage != i or first != next_layer or last < first or last > 65535
                or not 0 <= ordinal <= 15
                or (device is not None and (not isinstance(device, int) or isinstance(device, bool)
                                            or not 0 <= device <= 65535))):
            return []
        out.append({"stage": stage, "layer_start": first, "layer_end": last,
                    "engine_ordinal": ordinal, "device_index": device})
        next_layer = last + 1
    return out


def _batch_sample(work: dict) -> dict | None:
    """A remote average is trusted only with bounded, internally consistent engine counts."""
    windows, rows, active = (work.get(k) for k in
                             ("batch_sample_windows", "batch_sample_rows", "batch_sample_active_rows"))
    age = _number(work.get("batch_sample_age_s"))
    span = _number(work.get("batch_sample_span_s"))
    if (not all(type(v) is int for v in (windows, rows, active)) or
            not 1 <= windows <= 1_000_000_000 or not 1 <= active <= rows <= windows * 8 or
            age is None or not 0 <= age <= 86_400 or span is None or not 0 < span <= 86_400):
        return None
    return {"effective_rows_window": rows / windows, "batch_active_rows_window": active / windows,
            "batch_sample_windows": windows, "batch_sample_rows": rows,
            "batch_sample_active_rows": active, "batch_sample_span_s": span,
            "batch_sample_age_s": age}


def _replica_batches(value: Any) -> list[dict]:
    """Retain separate bounded engine samples; do not combine their unrelated windows."""
    if not isinstance(value, list):
        return []
    rows, seen = [], set()
    for raw in value[:16]:
        if not isinstance(raw, dict):
            continue
        replica = raw.get("replica")
        if type(replica) is not int or not 0 <= replica < 16 or replica in seen:
            continue
        seen.add(replica)
        rows.append({"replica": replica, **(_batch_sample(raw) or {})})
    return rows


def _stage_sample(work: dict, known_stages: int) -> dict | None:
    """Validate one bounded, internally consistent pipeline host-time sample."""
    elapsed = work.get("stage_sample_elapsed_us")
    busy = work.get("stage_in_flight_us")
    age = _number(work.get("stage_sample_age_s"))
    if type(elapsed) is not int or not 1 <= elapsed <= 86_400_000_000 or \
            not isinstance(busy, list) or not 2 <= len(busy) <= 16 or \
            (known_stages and len(busy) != known_stages) or \
            age is None or not 0 <= age <= 86_400 or \
            any(type(value) is not int or not 0 <= value <= elapsed for value in busy):
        return None
    return {"stage_in_flight": [
                {"stage": i, "percent": round(100.0 * value / elapsed, 1),
                 "in_flight_ms": round(value / 1000.0, 3)}
                for i, value in enumerate(busy)],
            "stage_sample_span_s": elapsed / 1_000_000.0,
            "stage_sample_age_s": age}


def _replica_stage_samples(value: Any, replica_stages: list[dict]) -> list[dict]:
    """Keep child host-span samples separate and check each against its own stage plan."""
    if not isinstance(value, list):
        return []
    counts = {row["replica"]: len(row["stages"]) for row in replica_stages}
    rows, seen = [], set()
    for raw in value[:16]:
        if not isinstance(raw, dict):
            continue
        replica = raw.get("replica")
        if type(replica) is not int or not 0 <= replica < 16 or replica in seen:
            continue
        seen.add(replica)
        rows.append({"replica": replica, **(_stage_sample(raw, counts.get(replica, 0)) or {})})
    return rows


def sanitize_telemetry(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept only bounded, privacy-safe v1 fields from a potentially remote server."""
    instance, model = _object(raw.get("instance")), _object(raw.get("model"))
    work, rates = _object(raw.get("work")), _object(raw.get("rates"))
    topology, ram = _object(raw.get("topology")), _object(raw.get("ram"))
    quality, totals = _object(raw.get("quality")), _object(raw.get("totals"))
    activity = _activity(raw.get("activity"))
    stages = _stage_rows(topology.get("stages"))
    replica_devices = _replica_devices(topology.get("replica_devices"))
    replica_stages = _replica_stages(topology.get("replica_stages"), replica_devices)
    batch = _batch_sample(work)
    replica_batches = _replica_batches(work.get("replica_batches"))
    replica_mode = (str(topology.get("mode") or "").startswith("replicas") or
                    bool(replica_batches) or bool(replica_stages))
    replica_stage_samples = (_replica_stage_samples(work.get("replica_stage_samples"), replica_stages)
                             if replica_mode else [])
    if replica_mode:
        batch = None
        stages = []
    stage_sample = None if replica_mode else _stage_sample(work, len(stages))
    safe_slots = _rows(work.get("slots"), ("slot", "group", "replica", "request_id", "state", "held_tokens",
                                               "prompt_tokens", "prompt_read", "generated", "elapsed_s", "tok_s"), 64)
    for row in safe_slots:
        replica = row.get("replica")
        row["replica"] = replica if type(replica) is int and 0 <= replica < 16 else None
    safe_requests = _rows(work.get("requests"), ("id", "state", "queued_at", "admitted_at", "slot",
                                                     "group", "replica", "prompt_tokens", "prompt_read",
                                                     "generated", "first_engine_token_s"), 50)
    for row in safe_requests:
        replica = row.get("replica")
        row["replica"] = replica if type(replica) is int and 0 <= replica < 16 else None
    reported_slots = complete_slots({"slots": safe_slots, "slots_effective": _number(work.get("slots_effective"))})
    occupied_slots = (sum(row["state"] != "idle" for row in reported_slots)
                      if reported_slots is not None else None)
    placement_resolved = topology.get("resolved") is True and bool(stages) and all(
        row["device_index"] is not None for row in stages)
    return {
        "schema_version": 1, "source": _short(raw.get("source")), "sampled_at": _number(raw.get("sampled_at")),
        "instance": {"id": _short(instance.get("id"), 64), "boot_time": _number(instance.get("boot_time")),
                     "server": _short(instance.get("server")),
                     "state": instance.get("state") if instance.get("state") in ("ready", "draining") else None},
        "model": {"id": _short(model.get("id")),
                  "loaded": model.get("loaded") if isinstance(model.get("loaded"), bool) else None,
                  "context": _number(model.get("context"))},
        "work": {**{k: _number(work.get(k)) for k in ("slots_requested", "slots_effective", "active", "queued",
                                                       "waiting", "outside_slots", "groups",
                                                       "requests_omitted", "max_queue", "admission_limit",
                                                       "admission_reserved", "admission_rejected", "oldest_queue_s",
                                                       "queue_timeout_s", "queue_timeouts",
                                                       "client_disconnect_cancels")},
                 **(batch or {"effective_rows_window": None, "batch_active_rows_window": None,
                              "batch_sample_windows": None, "batch_sample_rows": None,
                              "batch_sample_active_rows": None, "batch_sample_span_s": None,
                              "batch_sample_age_s": None}),
                 **(stage_sample or {"stage_in_flight": [], "stage_sample_span_s": None,
                                    "stage_sample_age_s": None}),
                 "occupied_slots": occupied_slots,
                 "replica_batches": replica_batches,
                 "replica_stage_samples": replica_stage_samples,
                 "slots": safe_slots,
                 "requests": safe_requests},
        "rates": {k: _number(rates.get(k)) for k in ("answer_tok_s", "answer_window_s", "prefill_tok_s",
                                                       "output_tokens_server_counter")},
        "topology": {"mode": _short(topology.get("mode")), "configured_split": _short(topology.get("configured_split")),
                     "selected_devices": _selected_devices(topology.get("selected_devices")),
                     "replica_devices": replica_devices, "replica_stages": replica_stages,
                     "resolved": placement_resolved,
                     "stages": stages, "source": _short(topology.get("source"), 40)},
        "devices": _rows(raw.get("devices"), ("index", "name", "mem_used", "mem_total", "util", "temp", "power"), 16),
        "ram": {"used": _number(ram.get("used")), "total": _number(ram.get("total"))},
        "recent": _rows(raw.get("recent"), ("id", "time", "duration_s", "finish", "prompt_tokens",
                                           "reused", "output_tokens", "prompt_ms", "decode_ms",
                                           "first_engine_token_s"), 50),
        "totals": {k: _number(totals.get(k)) for k in ("since", "requests", "prompt_tokens", "output_tokens")},
        "activity": activity,
        "quality": {**{k: _short(quality.get(k)) for k in ("request_correlation", "per_device")},
                    "telemetry": "unavailable" if raw.get("source") == "status-only" else "available",
                    "batch_window_rows": "measured" if batch else "unavailable",
                    "batch_window_source": "engine_bstat" if batch else None,
                    "stage_in_flight": "measured_host_span" if stage_sample else "unavailable",
                    "stage_in_flight_source": "engine_pstat_host_span" if stage_sample else None,
                    "replica_stage_in_flight": ("measured_host_span" if any(
                        row.get("stage_in_flight") for row in replica_stage_samples) else "unavailable"),
                    "replica_stage_in_flight_source": ("engine_pstat_host_span" if any(
                        row.get("stage_in_flight") for row in replica_stage_samples) else None),
                    "placement": "reported" if placement_resolved else "partial" if stages else "unavailable"},
    }


def normalize_status_only(status: dict[str, Any]) -> dict[str, Any]:
    """A reachable old server can still identify its model without metrics."""
    concurrency = _object(status.get("concurrency"))
    return {"schema_version": 1, "source": "status-only", "sampled_at": time.time(),
            "instance": {"boot_time": _number(status.get("started")),
                         "server": _short(status.get("service"))},
            "model": {"id": _short(status.get("model")),
                      "loaded": status.get("loaded") if isinstance(status.get("loaded"), bool) else None,
                      "context": _number(status.get("cache_max_tokens"))},
            "work": {"slots_effective": _number(concurrency.get("serving"))}}


def normalize_legacy(metrics: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Keep null distinct from zero; never infer group/stage work from GPU count."""
    live = metrics.get("live") if isinstance(metrics.get("live"), dict) else {}
    engine = metrics.get("engine") if isinstance(metrics.get("engine"), dict) else {}
    hardware = metrics.get("hardware") if isinstance(metrics.get("hardware"), dict) else {}
    static = metrics.get("hardware_static") if isinstance(metrics.get("hardware_static"), dict) else {}
    concurrency = status.get("concurrency") if isinstance(status.get("concurrency"), dict) else {}
    slots = live.get("slots") if isinstance(live.get("slots"), list) else []
    devices = hardware.get("gpus") if isinstance(hardware.get("gpus"), list) else []
    safe_slots = []
    for row in slots[:64]:
        if isinstance(row, dict):
            safe_slots.append({k: row.get(k) for k in ("slot", "group", "replica", "request_id", "state", "held_tokens",
                                                       "prompt_tokens", "prompt_read", "generated", "elapsed_s", "tok_s")})
    effective_slots = _number(concurrency.get("serving"))
    reported_slots = complete_slots({"slots": safe_slots, "slots_effective": effective_slots})
    safe_devices = []
    for row in devices[:16]:
        if isinstance(row, dict):
            safe_devices.append({"index": row.get("index"), "name": _short(row.get("name")),
                                 "mem_used": _number(row.get("mem_used")),
                                 "mem_total": _number(row.get("mem_total")),
                                 "util": _number(row.get("util")), "temp": _number(row.get("temp")),
                                 "power": _number(row.get("power"))})
    # Old single-GPU servers report only aggregate GPU fields. It is one device only
    # when the static sampler says so; never duplicate an aggregate into four rows.
    if not safe_devices and static.get("gpu_count") == 1 and static.get("gpu_name"):
        safe_devices = [{"index": 0, "name": _short(static.get("gpu_name")),
                         "mem_used": _number(hardware.get("gpu_mem_used")),
                         "mem_total": _number(hardware.get("gpu_mem_total")),
                         "util": _number(hardware.get("gpu_util")), "temp": _number(hardware.get("gpu_temp")),
                         "power": _number(hardware.get("gpu_power"))}]
    recent = []
    for row in (metrics.get("requests") if isinstance(metrics.get("requests"), list) else [])[:50]:
        if isinstance(row, dict):
            recent.append({k: row.get(k) for k in ("id", "time", "duration_s", "finish", "prompt_tokens",
                                                      "reused", "output_tokens", "prompt_ms", "decode_ms",
                                                      "first_engine_token_s")})
    return {"schema_version": 1, "source": "legacy-adapter", "sampled_at": _number(metrics.get("time")),
            "instance": {"boot_time": _number(status.get("started")), "server": _short(status.get("service"))},
            "model": {"id": _short(status.get("model")), "loaded": status.get("loaded") if isinstance(status.get("loaded"), bool) else None,
                      "context": _number(status.get("cache_max_tokens"))},
            "work": {"slots_requested": None, "slots_effective": effective_slots,
                     "active": _number(live.get("running")) if "running" in live else (1 if live.get("state") in ("reading", "generating") else 0),
                     "occupied_slots": (sum(row["state"] != "idle" for row in reported_slots)
                                        if reported_slots is not None else None),
                     "queued": _number(live.get("queued")), "waiting": _number(live.get("waiting")),
                     "outside_slots": _number(live.get("outside_slots")), "slots": safe_slots,
                     "groups": _number(engine.get("batch_groups")), "effective_rows_window": None,
                     "batch_active_rows_window": None, "batch_sample_windows": None,
                     "batch_sample_rows": None, "batch_sample_active_rows": None,
                     "batch_sample_span_s": None, "batch_sample_age_s": None,
                     "stage_in_flight": [], "stage_sample_span_s": None, "stage_sample_age_s": None,
                     "requests": [], "requests_omitted": None, "max_queue": None,
                     "admission_limit": None, "admission_reserved": None,
                     "admission_rejected": None, "oldest_queue_s": None,
                     "queue_timeout_s": None, "queue_timeouts": None,
                     "client_disconnect_cancels": None},
            "rates": {"answer_tok_s": _number(live.get("tok_s")),
                      "answer_window_s": _number(live.get("tok_s_window_s")),
                      "prefill_tok_s": _number(live.get("prefill_tok_s_mean"))},
            "topology": {"mode": None, "stages": []}, "devices": safe_devices,
            "ram": {"used": _number(hardware.get("ram_used")), "total": _number(hardware.get("ram_total"))},
            "recent": recent, "totals": {k: _number(v) for k, v in
                                          (metrics.get("totals") if isinstance(metrics.get("totals"), dict) else {}).items()
                                          if k in ("since", "requests", "prompt_tokens", "output_tokens")},
            "quality": {"request_correlation": "unavailable", "batch_window_rows": "unavailable",
                        "batch_window_source": None,
                        "stage_in_flight": "unavailable", "stage_in_flight_source": None,
                        "per_device": "measured" if safe_devices else "unavailable"}}
