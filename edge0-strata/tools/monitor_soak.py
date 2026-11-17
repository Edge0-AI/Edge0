#!/usr/bin/env python3
"""Headless Monitor soak against a bounded synthetic telemetry server.

This is a client stability check, not model, GPU or serving performance evidence.
Run for 1800 seconds to exercise reconnects, instance changes and bounded state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from strata_cli.client import Endpoint


class SyntheticTelemetry:
    def __init__(self, *, exercise_ui: bool = False):
        self.lock = threading.Lock()
        self.instance = 0
        self.unavailable = False
        self.gets = 0
        self.failures = 0
        self.other_methods = 0
        self.bytes_sent = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.seq = 0
        self.exercise_ui = exercise_ui

    def snapshot(self) -> dict:
        with self.lock:
            self.seq += 1
            seq, instance = self.seq, self.instance
        now = time.time()
        events = [{"seq": number, "at": now, "request_id": f"{number:012x}", "event": "queued"}
                  for number in range(max(1, seq - 63), seq + 1)]
        devices = [{"index": i, "name": f"Synthetic card {i}"} for i in range(4)]
        topology = {}
        if self.exercise_ui:
            # Rotate an unrelated host card and intermittently lose or duplicate
            # a selected sensor. The Overview must retain only the four selected
            # card histories and mark current gaps unavailable.
            devices = [{"index": 100 + seq, "name": "Unselected host card", "util": 10}]
            devices += [{"index": i, "name": f"Selected card {i}",
                         "util": (40 + i * 11 + seq) % 100,
                         "mem_used": (12 + i) * 2**30, "mem_total": 24 * 2**30}
                        for i in range(4) if not (i == 3 and seq % 5 == 0)]
            if seq % 7 == 0:
                devices.append({"index": 1, "name": "Ambiguous card 1", "util": 99})
            topology = {"mode": "layer split", "selected_devices": [0, 1, 2, 3],
                        "resolved": False, "stages": []}
        return {"schema_version": 1, "source": "synthetic-soak", "sampled_at": now,
                "instance": {"id": f"synthetic-{instance}", "state": "ready", "boot_time": now - 60},
                "model": {"id": "synthetic-model", "loaded": True, "context": 4096},
                "rates": {"output_tokens_server_counter": seq * 12},
                "work": {"active": 0, "queued": seq % 5, "waiting": 0, "slots_requested": 8,
                         "slots_effective": 8, "groups": 4,
                         "batch_sample_windows": 4, "batch_sample_rows": 4 * (3 + seq % 4),
                         "batch_sample_active_rows": 4 * min(3, 2 + seq % 3),
                         "batch_sample_span_s": 1.0, "batch_sample_age_s": 0.1,
                         "requests": [{"id": f"r{i:02d}", "state": "queued"} for i in range(70)]},
                "topology": topology, "devices": devices,
                "activity": {"last_seq": seq, "events": events}}

    def counters(self) -> dict:
        with self.lock:
            return {"gets": self.gets, "http_failures": self.failures,
                    "other_methods": self.other_methods, "bytes_sent": self.bytes_sent,
                    "max_in_flight": self.max_in_flight, "instance": self.instance}


def handler_for(service: SyntheticTelemetry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            with service.lock:
                service.in_flight += 1
                service.max_in_flight = max(service.max_in_flight, service.in_flight)
                service.gets += 1
                unavailable = service.unavailable
            try:
                if self.path != "/v1/telemetry":
                    self.send_error(404)
                    return
                if unavailable:
                    with service.lock:
                        service.failures += 1
                    self.send_error(503)
                    return
                body = json.dumps(service.snapshot(), separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                with service.lock:
                    service.bytes_sent += len(body)
            finally:
                with service.lock:
                    service.in_flight -= 1

        def do_POST(self):
            with service.lock:
                service.other_methods += 1
            self.send_error(405)

    return Handler


async def soak(seconds: float, interval: float, *, exercise_ui: bool = False) -> dict:
    try:
        from strata_cli.monitor import MonitorApp
    except ImportError as exc:
        raise RuntimeError("the optional Textual monitor dependency is required") from exc
    service = SyntheticTelemetry(exercise_ui=exercise_ui)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = Endpoint(f"http://127.0.0.1:{server.server_port}", timeout=1)
    tracemalloc.start()
    started = time.monotonic()
    cpu_started = time.process_time()
    events: list[str] = []
    baseline = None
    max_output_rate_samples = 0
    max_batch_samples = 0
    max_pre_admit_samples = 0
    max_device_histories = 0
    max_device_samples = 0
    sensor_gap_observed = False
    max_vram_histories = 0
    max_vram_samples = 0
    vram_sensor_gap_observed = False
    resize_count = 0
    sizes_seen: set[tuple[int, int]] = set()
    resize_size_cycle = ((140, 44), (110, 30), (80, 24), (60, 16), (40, 12))
    resize_step = max(1.0, min(5.0, seconds / 20))
    next_resize = 1.0
    next_progress = 60.0
    outage_length = min(5.0, max(1.0, seconds / 20))
    try:
        app = MonitorApp(endpoint, None, interval=interval)
        observed_states = {app.connection_state}
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("2")
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    break
                if baseline is None and elapsed >= min(5.0, seconds / 5):
                    baseline = tracemalloc.get_traced_memory()[0]
                if elapsed >= seconds * 0.25 and "disconnect" not in events:
                    with service.lock:
                        service.unavailable = True
                    events.append("disconnect")
                if elapsed >= seconds * 0.25 + outage_length and "reconnect" not in events:
                    with service.lock:
                        service.unavailable = False
                    events.append("reconnect")
                if elapsed >= seconds * 0.5 and "restart" not in events:
                    with service.lock:
                        service.instance += 1
                        service.seq = 0
                    events.append("restart")
                if elapsed >= seconds * 0.75 and "second_disconnect" not in events:
                    with service.lock:
                        service.unavailable = True
                    events.append("second_disconnect")
                if elapsed >= seconds * 0.75 + outage_length and "second_reconnect" not in events:
                    with service.lock:
                        service.unavailable = False
                    events.append("second_reconnect")
                if elapsed >= next_progress:
                    print(json.dumps({"elapsed_s": round(elapsed, 1), **service.counters()}), flush=True)
                    next_progress += 60.0
                if exercise_ui and elapsed >= next_resize:
                    size = resize_size_cycle[resize_count % len(resize_size_cycle)]
                    await pilot.resize_terminal(*size)
                    sizes_seen.add(size)
                    resize_count += 1
                    next_resize += resize_step
                observed_states.add(app.connection_state)
                max_output_rate_samples = max(max_output_rate_samples, len(app.history.output_rate))
                max_batch_samples = max(max_batch_samples, len(app.history.batch_rows))
                max_pre_admit_samples = max(max_pre_admit_samples, len(app.history.pre_admit))
                max_device_histories = max(max_device_histories, len(app.history.device_util))
                max_device_samples = max(max_device_samples,
                                         *(len(rows) for rows in app.history.device_util.values()), 0)
                sensor_gap_observed |= any(rows and rows[-1] is None
                                           for rows in app.history.device_util.values())
                max_vram_histories = max(max_vram_histories, len(app.history.device_vram))
                max_vram_samples = max(max_vram_samples,
                                       *(len(rows) for rows in app.history.device_vram.values()), 0)
                vram_sensor_gap_observed |= any(rows and rows[-1] is None
                                                for rows in app.history.device_vram.values())
                await asyncio.sleep(min(0.5, max(0.01, seconds - elapsed)))
            await pilot.press("6")
            await pilot.pause(0.1)
            latest = app.latest
            final_instance = (latest.get("instance") or {}).get("id")
            retained_activity = len((latest.get("activity") or {}).get("events") or [])
            retained_requests = len((latest.get("work") or {}).get("requests") or [])
            output_rate_samples = list(app.history.output_rate)
            valid_output_rate_samples = sum(value is not None for value in output_rate_samples)
            valid_batch_samples = sum(value is not None for value in app.history.batch_rows)
            valid_pre_admit_samples = sum(value is not None for value in app.history.pre_admit)
            output_rate_instance = app.history.instance
            monitor_recovered = app.error is None and app.failures == 0
            await pilot.press("q")
        process_memory = tracemalloc.get_traced_memory()[0]
        counters = service.counters()
        # A fresh read after detach proves the monitor did not shut down the server.
        detached_server_readable = endpoint.snapshot()["instance"]["id"] == final_instance
        return {"schema_version": 1, "scope": "synthetic_headless_monitor_soak",
                "duration_s": round(time.monotonic() - started, 2), "poll_interval_s": interval,
                "events": events, "final_instance_id": final_instance,
                "monitor_states": sorted(observed_states),
                "retained_activity": retained_activity, "retained_requests": retained_requests,
                "output_rate_samples_retained": len(output_rate_samples),
                "output_rate_valid_samples": valid_output_rate_samples,
                "output_rate_history_max_samples": max_output_rate_samples,
                "batch_history_max_samples": max_batch_samples,
                "batch_history_valid_samples": valid_batch_samples,
                "pre_admit_history_max_samples": max_pre_admit_samples,
                "pre_admit_history_valid_samples": valid_pre_admit_samples,
                "output_rate_history_instance": list(output_rate_instance) if output_rate_instance else None,
                "exercise_ui": exercise_ui,
                "resize_count": resize_count,
                "terminal_sizes_seen": [list(size) for size in sorted(sizes_seen)],
                "device_history_max_cards": max_device_histories,
                "device_history_max_samples": max_device_samples,
                "device_sensor_gap_observed": sensor_gap_observed,
                "vram_history_max_cards": max_vram_histories,
                "vram_history_max_samples": max_vram_samples,
                "vram_sensor_gap_observed": vram_sensor_gap_observed,
                "traced_memory_baseline_bytes": baseline,
                "traced_memory_final_bytes": process_memory,
                "traced_memory_growth_bytes": process_memory - baseline if baseline is not None else None,
                "synthetic_memory_growth_budget_bytes": 20 * 1024 * 1024,
                "combined_process_cpu_s": round(time.process_time() - cpu_started, 3),
                "monitor_recovered": monitor_recovered,
                "detached_server_readable": detached_server_readable,
                "http": {**counters,
                         "gets_per_s": round(counters["gets"] / max(0.1, time.monotonic() - started), 3),
                         "bytes_per_s": round(counters["bytes_sent"] / max(0.1, time.monotonic() - started), 1)},
                "passed": (set(events) == {"disconnect", "reconnect", "restart",
                                          "second_disconnect", "second_reconnect"}
                           and final_instance == "synthetic-1" and retained_activity <= 32
                           and retained_requests <= 50 and 2 <= valid_output_rate_samples
                           and max_output_rate_samples <= app.history.limit
                           and 2 <= valid_batch_samples and max_batch_samples <= app.history.limit
                           and 2 <= valid_pre_admit_samples and max_pre_admit_samples <= app.history.limit
                           and output_rate_instance == ("id", final_instance)
                           and counters["other_methods"] == 0
                           and counters["max_in_flight"] <= 1 and counters["http_failures"] >= 2
                           and {"connecting", "connected", "stale"} <= observed_states
                           and monitor_recovered and detached_server_readable and baseline is not None
                           and process_memory - baseline <= 20 * 1024 * 1024
                           and (not exercise_ui or
                                (sizes_seen == set(resize_size_cycle) and resize_count >= 4
                                 and max_device_histories <= 4 and max_device_samples <= app.history.limit
                                 and sensor_gap_observed
                                 and set(app.history.device_util) == {0, 1, 2, 3}
                                 and max_vram_histories <= 4 and max_vram_samples <= app.history.limit
                                 and vram_sensor_gap_observed
                                 and set(app.history.device_vram) == {0, 1, 2, 3})))}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        tracemalloc.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless synthetic Strata Monitor soak")
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--exercise-ui", action="store_true",
                        help="resize across five terminal sizes and churn selected/host card sensors")
    parser.add_argument("--report", type=Path, required=True, help="new absolute JSON report path")
    args = parser.parse_args(argv)
    if not 8 <= args.seconds <= 3600 or not 0.5 <= args.interval <= 30:
        parser.error("--seconds must be 8..3600 and --interval must be 0.5..30")
    if args.exercise_ui and args.seconds < 30:
        parser.error("--exercise-ui requires at least 30 seconds")
    report_path = args.report.expanduser()
    if not report_path.is_absolute() or report_path.exists():
        parser.error("--report must be an absolute new file")
    result = asyncio.run(soak(args.seconds, args.interval, exercise_ui=args.exercise_ui))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(report_path), "passed": result["passed"]}), flush=True)
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
