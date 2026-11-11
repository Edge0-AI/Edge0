#!/usr/bin/env python3
"""The PCIe path of every GPU, read from sysfs (no root needed).

Walks from each display card up through its bridges to the root port and prints
every hop's negotiated and maximum link speed and width. The card's own
`current_link_*` attributes are printed too; on this PC they claim x16 Gen4 for
both cards while the upstream ports of one of them are all 8.0 GT/s x4, so the
bridge chain is the reading that agrees with the engine's own probe.

Also prints the DRM slot, subsystem id and rocm-smi's view of each card.
"""
import os
import subprocess
import sys
from pathlib import Path


def r(path, name):
    try:
        return Path(path, name).read_text().strip()
    except OSError:
        return '?'


def main():
    out = sys.stdout
    smi = subprocess.run(['rocm-smi', '--showbus', '--showid'],
                         capture_output=True, text=True).stdout
    for line in smi.splitlines():
        if 'PCI Bus' in line or 'Subsystem' in line or 'Device ID' in line:
            print(line, file=out)
    for dev in sorted(Path('/sys/bus/pci/devices').glob('*')):
        if not (dev / 'class').exists() or r(dev, 'class') != '0x030000':
            continue
        real = os.path.realpath(dev)
        # which DRM card is this endpoint?
        card = '?'
        for c in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
            if c.name.count('-') == 0 and os.path.realpath(c / 'device') == real:
                card = c.name
        print(f'\n=== {dev.name}  drm {card}  subsystem {r(dev, "device")}', file=out)
        print(f'    realpath {real}', file=out)
        p = real
        while p and p != '/sys/devices' and p != '/':
            name = os.path.basename(p)
            print(f'    {name:15} class={r(p, "class")} device={r(p, "device"):<7} '
                  f'current={r(p, "current_link_speed")} x{r(p, "current_link_width")}  '
                  f'max={r(p, "max_link_speed")} x{r(p, "max_link_width")}', file=out)
            p = os.path.dirname(p)


if __name__ == '__main__':
    main()
