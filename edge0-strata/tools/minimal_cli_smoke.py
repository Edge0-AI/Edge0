#!/usr/bin/env python3
"""Check source CLI dispatch without optional terminal or harness packages.

This runs Python with -S against a loopback telemetry fixture. It checks lazy
client dispatch and early serving failure, not an actual engine or installation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from portable_monitor_smoke import _server


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run([sys.executable, "-S", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=20, check=False)


def _record(result: subprocess.CompletedProcess[str], expected_exit: int) -> dict:
    if result.returncode != expected_exit or result.stderr.strip():
        raise RuntimeError(f"CLI exit {result.returncode}, stderr {result.stderr.strip()!r}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CLI did not write exactly one JSON record") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("CLI JSON schema was missing")
    return value


def smoke() -> dict:
    probe = _run("-c", "import importlib.util,json,sys; print(json.dumps({"
                 "'no_site':sys.flags.no_site,'rich':importlib.util.find_spec('rich') is not None,"
                 "'textual':importlib.util.find_spec('textual') is not None}))")
    if probe.returncode or probe.stderr.strip():
        raise RuntimeError("could not establish the no-site-packages test environment")
    extras = json.loads(probe.stdout)
    if extras != {"no_site": 1, "rich": False, "textual": False}:
        raise RuntimeError("optional terminal packages are visible in the test environment")

    help_result = _run(str(ROOT / "strata.py"), "--help")
    if (help_result.returncode or help_result.stderr.strip() or
            not all(word in help_result.stdout for word in ("serve", "monitor", "code"))):
        raise RuntimeError("source CLI help failed without optional packages")

    server, paths, _ = _server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        status = _record(_run(str(ROOT / "strata.py"), "status", "--url", url, "--json"), 0)
        once = _record(_run(str(ROOT / "strata.py"), "monitor", "--url", url,
                            "--once", "--output", "json"), 0)
        for value in (status, once):
            if (value.get("ok") is not True or
                    (value.get("snapshot", {}).get("instance") or {}).get("id") != "portable-smoke" or
                    (value.get("snapshot", {}).get("model") or {}).get("id") != "portable-smoke-model"):
                raise RuntimeError("one-shot CLI showed the wrong telemetry fixture")
        if paths != ["GET /v1/telemetry", "GET /v1/telemetry"]:
            raise RuntimeError("one-shot commands sent unexpected HTTP requests")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    with tempfile.TemporaryDirectory(prefix="strata-minimal-cli-") as directory:
        missing = str(Path(directory) / "missing-config.json")
        # serve needs the API server's packages; without them strata.py stops in its
        # preflight (exit 5, a plain message, nothing installed) before reading the config.
        serve = _run(str(ROOT / "strata.py"), "serve", "--profile", missing,
                     "--offline", "--output", "jsonl")
        if (serve.returncode != 5 or serve.stdout.strip() or
                "missing packages the API server needs" not in serve.stderr or
                "never install anything" not in serve.stderr):
            raise RuntimeError(f"serve without server packages did not stop in the preflight: "
                               f"exit {serve.returncode}, stderr {serve.stderr.strip()!r}")
        code = _record(_run(str(ROOT / "strata.py"), "code", "run", "--prompt-file",
                            str(Path(directory) / "missing-prompt.txt"), "--policy", "review",
                            "--output", "jsonl"), 5)
        if (code.get("ok") is not False or
                (code.get("error") or {}).get("code") != 5 or
                "harness component" not in (code.get("error") or {}).get("message", "")):
            raise RuntimeError("missing harness did not return actionable exit 5")

    return {"schema_version": 1, "scope": "source_cli_without_site_packages",
            "platform": platform.platform(), "python": platform.python_version(),
            "optional_packages_visible": {"rich": False, "textual": False},
            "commands": ["--help", "status --json", "monitor --once --output json",
                         "serve --offline --output jsonl (server packages absent: preflight exit 5)",
                         "code run --output jsonl (missing harness)"],
            "telemetry_gets": len(paths), "non_gets": 0,
            "actual_engine_started": False, "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="new absolute JSON report path")
    args = parser.parse_args()
    report = args.report.expanduser()
    if not report.is_absolute() or report.exists():
        parser.error("--report must be an absolute new file")
    result = smoke()
    report.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(report), "passed": result["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
