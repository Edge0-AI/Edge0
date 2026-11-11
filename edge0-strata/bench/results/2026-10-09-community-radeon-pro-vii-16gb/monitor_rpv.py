"""Sample system RAM, swap and AMD GPU readings once per second during the trial.

Radeon Pro VII / ROCm 6.2 adaptation of
bench/results/2026-10-04-community-mi50/monitor_mi50.py:
GPU memory, edge/junction/memory temperature and socket power via rocm-smi
(ROCm 6.2 has no `--showpowercap`, so the power cap comes from the hwmon
`power1_cap` file). Runs on the HOST (needs rocm-smi), writes telemetry.jsonl.

    python3 monitor_rpv.py telemetry.jsonl
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_rocm_smi():
    """One rocm-smi call per sample: VRAM, temperatures, power, GPU use."""
    out = {}
    try:
        text = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showpower", "--showuse"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception as error:  # noqa: BLE001
        return {"error": str(error)}
    for line in text.splitlines():
        # rocm-smi 的行形如 "GPU[0]\t\t: Temperature (Sensor edge) (C): 37.0" —— 指标名在**最后一个**
        # 冒号之前，所以按最后一个冒号切（按第一个切会只拿到 "GPU[0]"）。
        if ":" not in line:
            continue
        name, _, value = line.rpartition(":")
        name, value = name.strip(), value.strip()
        try:
            if name.endswith("VRAM Total Memory (B)"):
                out["vram_total_bytes"] = int(value)
            elif name.endswith("VRAM Total Used Memory (B)"):
                out["vram_used_bytes"] = int(value)
            elif "(Sensor edge) (C)" in name:
                out["temp_edge_c"] = float(value)
            elif "(Sensor junction) (C)" in name:
                out["temp_junction_c"] = float(value)
            elif "(Sensor memory) (C)" in name:
                out["temp_memory_c"] = float(value)
            elif "Socket Graphics Package Power (W)" in name:
                out["power_w"] = float(value)
            elif "GPU use (%)" in name:
                out["gpu_use_pct"] = float(value)
        except ValueError:
            continue
    return out


def power_cap_uw():
    for path in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/power1_cap"):
        try:
            return int(path.read_text().strip()), str(path)
        except OSError:
            continue
    return None, None


cap, cap_path = power_cap_uw()
target = Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry.jsonl")
with target.open("a") as output:
    while True:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                memory[key + "_KiB"] = int(value.strip().split()[0])
        output.write(json.dumps({"epoch_s": time.time(), "memory": memory,
                                 "gpu": parse_rocm_smi(),
                                 "power_cap_uw": cap, "power_cap_path": cap_path}) + "\n")
        output.flush()
        time.sleep(1)
