#!/usr/bin/env python3
"""e00.py OUT.json [CUDA_HOME] - machine capability JSON: nvidia-smi links, e00_probe.cu results, RAM, NUMA, CPU (read-only)."""
import json, os, platform, subprocess, sys, tempfile

out = sys.argv[1]
ch = sys.argv[2] if len(sys.argv) > 2 else "/usr/local/cuda"
here = os.path.dirname(os.path.abspath(__file__))
w = tempfile.mkdtemp()


def sh(c, t=120):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return str(e)


subprocess.run([f"{ch}/bin/nvcc", "-O2", "-std=c++17", f"{here}/e00_probe.cu", "-o", f"{w}/p", "-lpthread"], check=True)
p = json.loads(sh(f"{w}/p", 300))
p["host"] = platform.node()
p["nvidia_smi_pcie"] = sh("nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,clocks.max.sm,display_active,memory.total,memory.used,driver_version --format=csv").splitlines()
p["nvidia_topo"] = sh("nvidia-smi topo -m").splitlines()
want = ("Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "NUMA node(s)", "L3 cache")
p["lscpu"] = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in sh("lscpu").splitlines() if ":" in l and l.split(":", 1)[0].strip() in want}
p["meminfo_gb"] = {l.split(":")[0]: int(l.split()[1]) // 1048576 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal")}
p["dimms"] = sh("sudo -n dmidecode -t memory 2>/dev/null | grep -E 'Speed:|Size: [0-9]' | sort | uniq -c").splitlines()
p["numa"] = sh("numactl -H 2>/dev/null | head -8").splitlines()
p["kernel"] = platform.release()
p["note"] = "read-only probe; IOMMU/ACS/clocks untouched"
json.dump(p, open(out, "w"), indent=1)
print("wrote", out)
