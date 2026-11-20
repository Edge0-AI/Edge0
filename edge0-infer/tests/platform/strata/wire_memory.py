"""Wire (mlock) a given amount of memory so that macOS has that much less to give, then release it.

Usage: wire_memory.py --gib 128 [--min-free-gib 24] [--step-gib 8] [--max-seconds 3600] [--ready-file PATH]

Used to make a big Mac behave like a smaller one: the memory is anonymous, mapped in steps of --step-gib and wired
with mlock (which also fills the pages), so macOS can neither compress nor swap it. Before every step the memory
macOS can give without paging (free + inactive + speculative pages, from vm_stat) is read; a step that would take it
below --min-free-gib is not taken, and the wiring stops there.

Once wired, one line "READY wired_gib=... available_gib=..." is printed (and written to --ready-file). The memory is
released and the program exits on SIGINT or SIGTERM, when its parent process goes away, or after --max-seconds.
Exit codes: 0 after a release; 3 when not all of --gib could be wired (the memory that was wired is held all the
same, until one of the release conditions); 2 for bad arguments or when mmap / mlock fails at the first step.
Standard library only (ctypes), macOS only.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import signal
import subprocess
import sys
import time

GIB = 1 << 30
PROT_READ, PROT_WRITE = 1, 2
MAP_PRIVATE, MAP_ANON = 0x0002, 0x1000
MAP_FAILED = ctypes.c_void_p(-1).value

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def vm_stat():
    """vm_stat's counters in bytes, keyed by the line's name."""
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r'page size of (\d+) bytes', text).group(1))
    values = {}
    for line in text.splitlines()[1:]:
        match = re.match(r'(.+?):\s+(\d+)\.?$', line.strip())
        if match:
            values[match.group(1)] = int(match.group(2)) * page
    return values


def available_bytes():
    v = vm_stat()
    return v.get('Pages free', 0) + v.get('Pages inactive', 0) + v.get('Pages speculative', 0)


def log(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gib', type=float, required=True, help='GiB to wire')
    parser.add_argument('--min-free-gib', type=float, default=24.0,
                        help='never take a step that leaves macOS less than this to give (default 24)')
    parser.add_argument('--step-gib', type=float, default=8.0, help='GiB per step (default 8)')
    parser.add_argument('--max-seconds', type=float, default=3600.0, help='release after this long (default 3600)')
    parser.add_argument('--ready-file', help='also write the READY line to this file')
    args = parser.parse_args()
    if args.gib <= 0 or args.step_gib <= 0 or args.max_seconds <= 0:
        parser.error('--gib, --step-gib and --max-seconds must be positive')

    blocks = []
    stop = {'why': None}

    def on_signal(number, _frame):
        stop['why'] = signal.Signals(number).name
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def release(why):
        started = time.time()
        for address, size in blocks:
            libc.munlock(address, size)
            libc.munmap(address, size)
        log(f'RELEASED wired_gib={sum(s for _, s in blocks) / GIB:.2f} reason={why} '
            f'seconds={time.time() - started:.2f} available_gib={available_bytes() / GIB:.2f}')
        blocks.clear()

    parent = os.getppid()
    started = time.time()
    target = int(args.gib * GIB)
    step = int(args.step_gib * GIB)
    floor = args.min_free_gib * GIB
    wired = 0
    complete = True
    log(f'START target_gib={args.gib:.2f} min_free_gib={args.min_free_gib:.2f} available_gib={available_bytes() / GIB:.2f}')
    while wired < target and stop['why'] is None:
        size = min(step, target - wired)
        free_now = available_bytes()
        if free_now - size < floor:
            log(f'STOPPED wired_gib={wired / GIB:.2f}: one more step of {size / GIB:.2f} GiB would leave '
                f'{(free_now - size) / GIB:.2f} GiB, below {args.min_free_gib:.2f}')
            complete = False
            break
        address = libc.mmap(None, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        if address in (None, MAP_FAILED):
            log(f'ERROR mmap of {size / GIB:.2f} GiB failed: {os.strerror(ctypes.get_errno())}')
            complete = False
            break
        if libc.mlock(address, size) != 0:
            error = os.strerror(ctypes.get_errno())
            libc.munmap(address, size)
            log(f'ERROR mlock of {size / GIB:.2f} GiB failed after {wired / GIB:.2f} GiB: {error}')
            complete = False
            break
        blocks.append((address, size))
        wired += size
    if not blocks and stop['why'] is None:
        log('ERROR nothing could be wired')
        sys.exit(2)
    ready = f'READY wired_gib={wired / GIB:.2f} available_gib={available_bytes() / GIB:.2f} seconds={time.time() - started:.1f}'
    log(ready)
    if args.ready_file:
        with open(args.ready_file, 'w', encoding='utf-8') as out:
            out.write(ready + '\n')
    why = stop['why']
    while why is None:
        time.sleep(0.5)
        if stop['why']:
            why = stop['why']
        elif os.getppid() != parent:
            why = 'parent-exited'
        elif time.time() - started > args.max_seconds:
            why = 'max-seconds'
    release(why)
    sys.exit(0 if complete else 3)


if __name__ == '__main__':
    main()
