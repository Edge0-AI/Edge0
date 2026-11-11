#!/usr/bin/env python3
"""Run the repository's unchanged tools/needle_bench.py against the live server and keep the engine's log lines.

The key is read at run time (STRATA_API_KEY or --key-config) and handed to needle_bench.py on its command line
(local process only, never saved). Waits for an idle server first; afterwards GET /metrics must have counted exactly
one finished request per needle case, or someone else's request ran in between and the output says so.

    python run_needles.py --key-config <server config.json> --lengths 32k,128k --depths 10,50,90 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import LogTail, Server, api_key, parse_engine, scrub_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-config")
    ap.add_argument("--engine-log", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--lengths", default="32k,128k")
    ap.add_argument("--depths", default="10,50,90")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args()
    key = api_key(a.key_config)
    server = Server(a.url, key)
    elog, slog = LogTail(a.engine_log), LogTail(a.server_log)
    server.wait_idle()
    before = server.totals()
    elog.mark(), slog.mark()
    t0 = time.time()
    out = a.out / "needles.json"
    cmd = [sys.executable, str(ROOT / "tools" / "needle_bench.py"), "--url", a.url, "--api-key", key,
           "--lengths", a.lengths, "--depths", a.depths, "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    time.sleep(1.5)
    after = server.totals()
    elines, slines = elog.since(), slog.since()
    parsed = parse_engine(elines)
    cases = len(a.lengths.split(",")) * len(a.depths.split(","))
    meta = {"command": "python tools/needle_bench.py --url http://127.0.0.1:8080 --api-key <redacted> "
                       f"--lengths {a.lengths} --depths {a.depths} --out needles.json",
            "epoch_start": t0, "epoch_end": time.time(), "exit_code": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr[-2000:], "totals_before": before, "totals_after": after,
            "foreign_overlap": after - before != cases or len(parsed) != cases,
            "engine": parsed, "engine_log": scrub_lines(elines), "server_log": scrub_lines(slines)}
    (a.out / "needles-run.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(proc.stdout, proc.stderr[-2000:], "foreign_overlap:", meta["foreign_overlap"], flush=True)


if __name__ == "__main__":
    main()
