#!/usr/bin/env python3
"""Sample Strata's own /metrics once per second into JSONL.

Runs inside the container next to the benchmark. Every sample keeps the engine's
own view of memory (bytes), which is the only view available for VRAM inside the
container: nvidia-smi reports the whole GPU's allocation, not the container's.
"""
import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default="/data/bench/telemetry.jsonl")
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    started = time.time()
    with open(args.out, "w", encoding="utf-8") as sink:
        while time.time() - started < args.seconds:
            try:
                with urllib.request.urlopen(args.url + "/metrics", timeout=10) as response:
                    metrics = json.load(response)
                engine, hardware = metrics["engine"], metrics["hardware"]
                live = metrics.get("live", {})
                sink.write(json.dumps({
                    "t": round(time.time() - started, 3),
                    "gpu_mem_used": hardware["gpu_mem_used"],
                    "gpu_mem_total": hardware["gpu_mem_total"],
                    "gpu_util": hardware["gpu_util"],
                    "gpu_temp_c": hardware["gpu_temp"],
                    "gpu_power_w": hardware["gpu_power"],
                    "gpu_pcie_gen": hardware["gpu_pcie_gen"],
                    "gpu_pcie_width": hardware["gpu_pcie_width"],
                    "ram_used": hardware["ram_used"],
                    "ram_total": hardware["ram_total"],
                    "disk_read_mb": hardware.get("disk_read_mb"),
                    "vram_free_mib": engine["vram_free_mib"],
                    "expert_cache_mib": engine["expert_cache_mib"],
                    "state": live.get("state"),
                    "tok_s": live.get("tok_s"),
                }) + "\n")
                sink.flush()
            except Exception as error:                       # a sample may race a restart; keep going
                sink.write(json.dumps({"t": round(time.time() - started, 3), "error": str(error)}) + "\n")
                sink.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
