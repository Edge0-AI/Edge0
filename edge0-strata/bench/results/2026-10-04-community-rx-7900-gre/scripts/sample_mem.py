#!/usr/bin/env python3
"""Sample VRAM and host RAM once a second into a CSV until told to stop.

Reports what the machine used while Strata served requests: VRAM from the amdgpu
sysfs counters of every card, host RAM as MemTotal - MemAvailable. Both are
whole-machine numbers, not the engine's own accounting.
"""
import argparse
import csv
import os
import time
from pathlib import Path


def cards():
    return sorted(Path('/sys/class/drm').glob('card*/device/mem_info_vram_total'))


def read(path):
    try:
        return int(path.read_text())
    except OSError:
        return -1


def meminfo():
    total = avail = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            total = int(line.split()[1])
        elif line.startswith('MemAvailable:'):
            avail = int(line.split()[1])
    return total, avail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--stop-file', type=Path)
    args = p.parse_args()
    paths = cards()
    total, _ = meminfo()
    end = time.monotonic() + args.seconds
    with args.out.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unix_s'] + [c.parent.name + '_vram_used' for c in paths] + ['ram_used_mib'])
        while time.monotonic() < end:
            if args.stop_file and args.stop_file.exists():
                break
            used = total - meminfo()[1]
            w.writerow([round(time.time(), 2)] + [read(c.with_name('mem_info_vram_used')) for c in paths] + [round(used / 1024, 1)])
            fh.flush()
            time.sleep(1.0)


if __name__ == '__main__':
    os.nice(10)
    main()
