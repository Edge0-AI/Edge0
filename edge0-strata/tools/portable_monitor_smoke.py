#!/usr/bin/env python3
"""Exercise one portable Monitor bundle with a local fake server.

Run this on each target OS/Python before release. It checks the client and the
local bundle lifecycle. --interactive also checks packaged Textual rendering.
Neither mode exercises the engine or GPU drivers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BUILDER = Path(__file__).resolve().parents[1] / "packaging" / "monitor_bundle.py"
INTERACTIVE_CODE = r'''
import asyncio
import io
import json
import sys

sys.path.insert(0, sys.argv[1])
from rich.console import Console
from textual.widgets import Static
from strata_cli.client import Endpoint
from strata_cli.monitor import MonitorApp
from strata_cli.dashboard import render_dashboard
from strata_cli.detail_dashboard import render_detail_dashboard
import strata_cli.monitor as monitor_module

async def main():
    source = str(monitor_module.__file__).replace("\\", "/")
    bundle_path = sys.argv[1].replace("\\", "/")
    if not source.startswith(bundle_path + "/"):
        raise RuntimeError("interactive Monitor was not imported from the bundle")
    app = MonitorApp(Endpoint(sys.argv[2]), None, interval=0.5,
                     theme="dark", color="never", glyphs="ascii")
    async with app.run_test(size=(110, 30)) as pilot:
        for _ in range(30):
            if app.connection_state == "connected":
                break
            await pilot.pause(0.1)
        if app.connection_state != "connected":
            raise RuntimeError("interactive Monitor did not connect")
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(
            app.query_one("#content", Static).content)
        shown = output.getvalue()
        if not all(title in shown for title in
                   ("SERVING", "PERFORMANCE", "BATCHING / PARALLEL",
                    "HARDWARE", "REQUESTS", "ACTIVITY")):
            raise RuntimeError("packaged one-screen Overview is incomplete")
        if (app.latest.get("model") or {}).get("id") != "portable-smoke-model":
            raise RuntimeError("interactive Monitor showed the wrong model")
        wide = io.StringIO()
        Console(file=wide, width=140, color_system=None).print(render_dashboard(
            app.latest, width=138, height=42, history=app.history,
            glyphs="ascii", color=False))
        if not all(title in wide.getvalue() for title in
                   ("THROUGHPUT", "GPUS", "CONTEXT / BATCHING", "PARALLEL SLOTS", "REQUESTS")):
            raise RuntimeError("packaged wide visual Overview is incomplete")
        for view, title in (("serving", "REQUEST FLOW"), ("requests", "IN FLIGHT"),
                            ("engine", "STAGES / HOST SPAN"), ("hardware", "GPU CARDS"),
                            ("logs", "RECENT EVENT TIMELINE")):
            detail = io.StringIO()
            Console(file=detail, width=140, color_system=None).print(render_detail_dashboard(
                app.latest, view, width=138, height=42, history=app.history,
                glyphs="ascii", color=False))
            if title not in detail.getvalue():
                raise RuntimeError("packaged visual " + view + " page is incomplete")
        await pilot.press("q")
    print(json.dumps({"connected": True, "overview_panels": 6,
                      "wide_visual": True, "visual_details": 5,
                      "imported_from_bundle": True, "detached": True}))

asyncio.run(main())
'''


def _run(*argv: str) -> dict:
    result = subprocess.run([sys.executable, "-I", "-S", *argv],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"isolated command failed ({result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("isolated command did not return one JSON record") from exc


def _server() -> tuple[ThreadingHTTPServer, list[str], threading.Event]:
    paths: list[str] = []
    malformed = threading.Event()

    class TelemetryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append("GET " + self.path)
            if self.path != "/v1/telemetry":
                self.send_error(404)
                return
            body = json.dumps({
                "schema_version": 1, "sampled_at": time.time(),
                "instance": {"id": "portable-smoke", "state": "ready"},
                "model": {"id": "portable-smoke-model", "loaded": True},
                "work": {"active": 0, "queued": 0, "waiting": 0,
                         "slots_requested": 1, "slots_effective": 1, "groups": 1,
                         "slots": [{"slot": 0, "state": "idle", "group": 0}]},
                "topology": {"mode": "single", "selected_devices": []},
                "rates": {"output_tokens_server_counter": 0},
                "devices": [], "quality": {"telemetry": "available"},
            }).encode() if not malformed.is_set() else b"[" * 100000 + b"0" + b"]" * 100000
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject(self) -> None:
            paths.append(self.command + " " + self.path)
            self.send_error(405)

        do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_TRACE = _reject

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), TelemetryHandler), paths, malformed


def _malformed_response(bundle: Path, url: str) -> None:
    result = subprocess.run([sys.executable, "-I", "-S", str(bundle),
                             "status", "--url", url, "--json"],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode != 3 or result.stderr:
        raise RuntimeError("deeply nested telemetry escaped the client error boundary")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("malformed telemetry did not return one JSON error") from exc
    if (response.get("ok") is not False or
            (response.get("error") or {}).get("message") != "server returned malformed JSON telemetry"):
        raise RuntimeError("deeply nested telemetry returned an unexpected error")


def _interactive(bundle: Path, url: str) -> dict:
    result = subprocess.run([sys.executable, "-I", "-c", INTERACTIVE_CODE, str(bundle), url],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError(f"packaged interactive Monitor failed ({result.returncode}): "
                           f"{result.stderr.strip()[-2000:]}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("interactive Monitor did not return one JSON record") from exc
    if report != {"connected": True, "overview_panels": 6, "wide_visual": True,
                  "visual_details": 5,
                  "imported_from_bundle": True, "detached": True}:
        raise RuntimeError("packaged interactive Monitor smoke was incomplete")
    return report


def smoke(bundle: Path, *, interactive: bool = False) -> dict:
    bundle = bundle.resolve(strict=True)
    manifest = _run(str(BUILDER), "verify", str(bundle))
    expected_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if manifest.get("sha256") != expected_sha:
        raise RuntimeError("bundle digest changed after verification")

    server, paths, malformed = _server()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        one_shot = _run(str(bundle), "monitor", "--url", url, "--once", "--json")
        if (one_shot.get("ok") is not True or
                (one_shot.get("snapshot") or {}).get("model", {}).get("id") != "portable-smoke-model"):
            raise RuntimeError("one-shot monitor returned the wrong model or failed")

        with tempfile.TemporaryDirectory(prefix="strata-portable-smoke-") as scratch:
            target = Path(scratch) / "monitor"
            preview = _run(str(BUILDER), "install", str(bundle), "--target", str(target), "--plan")
            if preview.get("sha256") != expected_sha or preview.get("would_install") is not True:
                raise RuntimeError("bundle install preview did not bind the candidate digest")
            installed = _run(str(BUILDER), "install", str(bundle), "--target", str(target),
                             "--if-candidate-sha256", expected_sha)
            if installed.get("installed") is not True:
                raise RuntimeError("bundle install did not complete")
            installed_bundle = target / bundle.name
            _run(str(BUILDER), "verify", str(installed_bundle))
            status = _run(str(installed_bundle), "status", "--url", url, "--json")
            if (status.get("ok") is not True or
                    (status.get("snapshot") or {}).get("instance", {}).get("id") != "portable-smoke"):
                raise RuntimeError("installed client did not reach the fake server")
            removal = _run(str(BUILDER), "uninstall", "--target", str(target), "--plan")
            if removal.get("sha256") != expected_sha or removal.get("would_remove") is not True:
                raise RuntimeError("bundle removal preview did not bind the installed digest")
            removed = _run(str(BUILDER), "uninstall", "--target", str(target),
                           "--if-active-sha256", expected_sha)
            if removed.get("removed") is not True or target.exists():
                raise RuntimeError("bundle uninstall left an installed directory")
        malformed.set()
        _malformed_response(bundle, url)
        malformed.clear()
        interactive_report = _interactive(bundle, url) if interactive else None
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    if (paths[:2] != ["GET /v1/telemetry", "GET /v1/telemetry"] or
            any(path not in ("GET /v1/telemetry", "GET /metrics") for path in paths) or     # the interactive Overview also reads /metrics (404 here: it keeps the earlier Overview)
            (interactive and len(paths) < 4) or (not interactive and len(paths) != 3)):
        raise RuntimeError(f"unexpected client HTTP operations: {paths}")
    report = {"schema_version": 1, "scope": "portable_monitor_stdlib_smoke",
            "bundle_sha256": expected_sha, "python": platform.python_version(),
            "platform": platform.platform(), "isolated_no_site_packages": True,
            "one_shot_monitor": True, "installed_status": True,
            "digest_pinned_install_uninstall": True,
            "deep_json_error": True,
            "http_gets": len(paths),
            "other_http_methods": 0, "passed": True}
    if interactive_report is not None:
        report["interactive"] = interactive_report
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True,
                    help="new absolute JSON report file")
    ap.add_argument("--interactive", action="store_true",
                    help="also load the packaged Textual Overview in a headless terminal")
    args = ap.parse_args()
    if not args.report.is_absolute() or args.report.exists():
        ap.error("--report must be a new absolute file path")
    report = smoke(args.bundle, interactive=args.interactive)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
