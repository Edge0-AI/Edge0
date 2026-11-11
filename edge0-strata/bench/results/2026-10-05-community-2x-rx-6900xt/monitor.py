"""Sample system RAM, swap and both AMD GPUs' memory, busy and power once per second (sysfs; the AMD form of
bench/results/2026-09-30-community-rtx-5090/monitor.py)."""
import glob, json, sys, time
from pathlib import Path
out = Path(sys.argv[1] if len(sys.argv) > 1 else 'telemetry.jsonl')
cards = sorted(glob.glob('/sys/class/drm/card[0-9]/device'))
def rd(p):
    try: return int(Path(p).read_text().split()[0])
    except Exception: return None
with out.open('a') as f:
    while True:
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                memory[key + '_KiB'] = int(value.strip().split()[0])
        gpus = []
        for d in cards:
            hw = glob.glob(d + '/hwmon/hwmon*/')
            hw = hw[0] if hw else ''
            gpus.append({'pci': Path(d).resolve().name, 'vram_used_MiB': (rd(d + '/mem_info_vram_used') or 0) // 1048576,
                         'gtt_used_MiB': (rd(d + '/mem_info_gtt_used') or 0) // 1048576, 'busy_percent': rd(d + '/gpu_busy_percent'),
                         'power_W': (rd(hw + 'power1_average') or rd(hw + 'power1_input') or 0) / 1e6 if hw else None,
                         'temp_C': (rd(hw + 'temp1_input') or 0) / 1000 if hw else None})
        f.write(json.dumps({'epoch_s': time.time(), 'memory': memory, 'gpus': gpus}) + '\n'); f.flush()
        time.sleep(1)
