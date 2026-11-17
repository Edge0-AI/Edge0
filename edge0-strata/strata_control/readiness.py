"""Bounded, local API verification for a managed server instance.

The ready file is only a hint that a port was bound.  This module asks that
same instance for telemetry, then performs one one-token generation when the
model is loaded.  Credentials stay in memory and no response text is saved.
"""

from __future__ import annotations

import http.client
import hmac
import ipaddress
import json
import os
import re
import socket
import time


class ReadinessError(Exception):
    pass


def _address(ready: dict, expected_host: str) -> tuple[str, int]:
    host = ready.get("bound_host")
    port = ready.get("bound_port")
    if host != expected_host or type(port) is not int or not 1 <= port <= 65535:
        raise ReadinessError("ready_endpoint_mismatch")
    if host in ("", "0.0.0.0"):
        return "127.0.0.1", port
    if host == "::":
        return "::1", port
    if host == "localhost":
        return host, port
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise ReadinessError("ready_endpoint_not_numeric") from None
    return host, port


def _credential(source_bytes: bytes) -> str | None:
    from serve.server import api_key_of, key_list

    cfg = json.loads(source_bytes.decode("utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("run config must be an object")
    raw = os.environ.get("STRATA_API_KEY") or cfg.get("api_key", "")
    if str(raw).strip().lower() == "none":               # "api_key": "none" = served without a key on purpose
        return None
    keys = key_list(api_key_of(raw))
    return keys[0] if keys else None


def _request(host: str, port: int, method: str, path: str, key: str | None,
             timeout: float, body: dict | None = None) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    connection = http.client.HTTPConnection(host, port, timeout=max(0.1, timeout))
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            raise ReadinessError("probe_response_too_large")
        if response.status != 200:
            raise ReadinessError(f"probe_http_{response.status}")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ReadinessError("probe_invalid_response")
        return result
    except (OSError, socket.timeout, http.client.HTTPException):
        raise ReadinessError("probe_transport_failed") from None
    except (UnicodeError, ValueError):
        raise ReadinessError("probe_invalid_response") from None
    finally:
        connection.close()


def verify(ready: dict, source_bytes: bytes, *, expected_host: str,
           expected_instance: str, expected_attestation: str,
           timeout: float, force_load: bool = False) -> dict:
    if not isinstance(expected_attestation, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_attestation):
        raise ReadinessError("probe_launch_attestation_unavailable")
    host, port = _address(ready, expected_host)
    display_host = f"[{host}]" if ":" in host else host
    api_url = f"http://{display_host}:{port}"
    deadline = time.monotonic() + timeout
    try:
        cfg = json.loads(source_bytes.decode("utf-8-sig"))
        key = _credential(source_bytes)
    except (UnicodeError, ValueError, TypeError):
        raise ReadinessError("probe_credential_invalid") from None
    if not isinstance(cfg, dict) or not isinstance(ready.get("model_id"), str) or \
            (cfg.get("model_name") is not None and ready["model_id"] != cfg["model_name"]):
        raise ReadinessError("probe_model_mismatch")
    telemetry = _request(host, port, "GET", "/v1/telemetry", key,
                         max(0.1, deadline - time.monotonic()))
    instance = telemetry.get("instance")
    model = telemetry.get("model")
    if not isinstance(instance, dict) or instance.get("id") != expected_instance:
        raise ReadinessError("probe_instance_mismatch")
    reported_attestation = instance.get("launch_attestation")
    if not isinstance(reported_attestation, str) or not hmac.compare_digest(
            reported_attestation, expected_attestation):
        raise ReadinessError("probe_launch_attestation_mismatch")
    if not isinstance(model, dict) or model.get("id") != ready.get("model_id"):
        raise ReadinessError("probe_model_mismatch")
    if ready.get("model_state") not in ("ready", "unloaded"):
        raise ReadinessError("probe_model_state_invalid")
    loaded = model.get("loaded")
    if type(loaded) is not bool:
        raise ReadinessError("probe_model_state_unavailable")
    if not loaded and not force_load:
        return {"api_identity_verified": True, "launch_attested": True, "generation_ready": False,
                "readiness_state": "unloaded", "model_state": "unloaded", "api_url": api_url}
    probe = {"model": ready.get("model_id"), "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 1, "temperature": 0, "stream": False}
    while True:
        try:
            result = _request(host, port, "POST", "/v1/chat/completions", key,
                              max(0.1, deadline - time.monotonic()), probe)
            break
        except ReadinessError as exc:
            if str(exc) not in ("probe_http_409", "probe_http_429", "probe_http_503") or \
                    time.monotonic() + 0.25 >= deadline:
                raise
            time.sleep(0.25)
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or \
            not isinstance(choices[0].get("message"), dict) or \
            not isinstance(choices[0].get("finish_reason"), str):
        raise ReadinessError("probe_generation_invalid_response")
    usage = result.get("usage")
    if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int or \
            usage["completion_tokens"] < 1:
        raise ReadinessError("probe_generation_no_tokens")
    return {"api_identity_verified": True, "launch_attested": True, "generation_ready": True,
            "readiness_state": "generation_ready", "model_state": "ready", "api_url": api_url}
