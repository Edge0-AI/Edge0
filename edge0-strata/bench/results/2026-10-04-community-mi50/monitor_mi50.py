"""Sample system RAM, swap and AMD GPU memory once per second during the trial.

MI50 / ROCm adaptation: GPU memory via rocm-smi instead of nvidia-smi, and the
GPU is the Vega 20 (MI50) at PCI 0000:07:00.0 — rocm-smi indexes it as GPU[0].
Runs on the HOST (needs rocm-smi), writes telemetry.jsonl.
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp", "--csv"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception as error:  # noqa: BLE001
        return f"rocm-smi error: {error}"


with Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl").open("a") as output:
    while True:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                memory[key + "_KiB"] = int(value.strip().split()[0])
        output.write(json.dumps({"epoch_s": time.time(), "memory": memory, "gpu": gpu_mem()}) + "\n")
        output.flush()
        time.sleep(1)
