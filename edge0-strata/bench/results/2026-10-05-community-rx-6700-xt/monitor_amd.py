"""Sample system RAM, swap and the AMD GPU's memory, load, power and temperature once per second (sysfs).
The RTX 5090 report's monitor.py calls nvidia-smi; this reads the same quantities from amdgpu's sysfs files."""
import glob
import json
import sys
import time
from pathlib import Path

CARD = Path(sys.argv[2] if len(sys.argv) > 2 else '/sys/class/drm/card1/device')
HWMON = Path(glob.glob(str(CARD / 'hwmon' / 'hwmon*'))[0])


def number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


with Path(sys.argv[1] if len(sys.argv) > 1 else 'telemetry.jsonl').open('a') as output:
    while True:
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                memory[key + '_KiB'] = int(value.strip().split()[0])
        gpu = {'vram_used_bytes': number(CARD / 'mem_info_vram_used'),
               'vram_total_bytes': number(CARD / 'mem_info_vram_total'),
               'gpu_busy_percent': number(CARD / 'gpu_busy_percent'),
               'power_average_uW': number(HWMON / 'power1_average'),
               'temperature_mC': number(HWMON / 'temp1_input')}
        output.write(json.dumps({'epoch_s': time.time(), 'memory': memory, 'gpu': gpu}) + '\n')
        output.flush()
        time.sleep(1)
