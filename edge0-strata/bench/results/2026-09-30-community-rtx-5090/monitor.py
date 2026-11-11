"""Sample system RAM, swap and GPU memory once per second during the trial."""
import json
import subprocess
import sys
import time
from pathlib import Path

with Path(sys.argv[1] if len(sys.argv) > 1 else 'telemetry.jsonl').open('a') as output:
    while True:
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                memory[key + '_KiB'] = int(value.strip().split()[0])
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        output.write(json.dumps({'epoch_s': time.time(), 'memory': memory, 'gpu': gpu}) + '\n')
        output.flush()
        time.sleep(1)
