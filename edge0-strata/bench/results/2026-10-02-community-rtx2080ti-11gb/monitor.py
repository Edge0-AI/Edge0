#!/usr/bin/env python3
"""Sample host RAM/swap and GPU memory once per second while the measurements run (one JSON line per sample).

    python monitor.py telemetry.jsonl &      # stop it with kill when done
"""
import json
import subprocess
import sys
import time
from pathlib import Path

FIELDS = "memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,pcie.link.gen.current"
with Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl").open("a") as output:
    while True:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree"):
                memory[key + "_KiB"] = int(value.strip().split()[0])
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=" + FIELDS, "--format=csv,noheader,nounits"],
                                      text=True).strip()
        ps = subprocess.run(["ps", "-C", "strata,strata-vision", "-o", "comm=,rss="], text=True,
                            capture_output=True).stdout.split("\n")
        rss = {f[0]: int(f[1]) for f in (l.split() for l in ps) if len(f) == 2}     # KiB, resident set
        output.write(json.dumps({"epoch_s": round(time.time(), 2), "memory": memory, "gpu": gpu,
                                 "rss_KiB": rss}) + "\n")
        output.flush()
        time.sleep(1)
