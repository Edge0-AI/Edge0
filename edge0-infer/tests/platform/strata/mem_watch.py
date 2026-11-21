"""Watch one process's memory and the system's paging while something runs.

Usage: mem_watch.py (--pid PID | --match TEXT) [--interval 2] [--out watch.json] [--max-seconds 7200]

Every --interval seconds it records the process's phys_footprint and resident size (proc_pid_rusage), the system's
Swapouts, Pageouts and compressor pages (vm_stat), the kernel's memory pressure level
(kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical) and memory_pressure's free percentage.
--match follows the newest process whose command line contains TEXT (looked up again when it goes away, e.g. an
engine restarted by its server). It stops on SIGINT / SIGTERM, after --max-seconds, or (with --pid) when the
process exits, and then prints and writes the summary: peak phys_footprint and resident size, the Swapouts and
Pageouts increase over the watch (in pages and MiB), the worst pressure level seen and whether it was critical.
Standard library only (ctypes), macOS only.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time

RUSAGE_INFO_V2 = 2


class RusageInfoV2(ctypes.Structure):
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups', 'ri_pageins', 'ri_wired_size',
        'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime', 'ri_proc_exit_abstime',
        'ri_child_user_time', 'ri_child_system_time', 'ri_child_pkg_idle_wkups', 'ri_child_interrupt_wkups',
        'ri_child_pageins', 'ri_child_elapsed_abstime', 'ri_diskio_bytesread', 'ri_diskio_byteswritten')]


libc = ctypes.CDLL(None, use_errno=True)
libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def process_memory(pid):
    """(phys_footprint, resident size) in bytes, or None when the process is gone."""
    info = RusageInfoV2()
    if libc.proc_pid_rusage(pid, RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint, info.ri_resident_size


def vm_counters():
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r'page size of (\d+) bytes', text).group(1))
    values = {'page': page}
    for line in text.splitlines()[1:]:
        match = re.match(r'(.+?):\s+(\d+)\.?$', line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def pressure():
    """(level, free percent)."""
    level = subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], capture_output=True, text=True).stdout
    free = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout
    match = re.search(r'free percentage:\s*(\d+)%', free)
    return int(level.strip() or 0), int(match.group(1)) if match else None


def find(text):
    """The newest process (largest pid) other than this one whose command line contains text."""
    out = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, command = line.strip().partition(' ')
        if text in command and int(pid) != os.getpid() and 'mem_watch.py' not in command:
            pids.append(int(pid))
    return max(pids) if pids else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pid', type=int)
    group.add_argument('--match', help='follow the newest process whose command line contains this text')
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--out', help='write the samples and the summary as JSON')
    parser.add_argument('--max-seconds', type=float, default=7200.0)
    args = parser.parse_args()

    stop = {'why': None}

    def on_signal(number, _frame):
        stop['why'] = signal.Signals(number).name
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    first = vm_counters()
    samples = []
    started = time.time()
    pid = args.pid
    while stop['why'] is None:
        if args.match and (pid is None or process_memory(pid) is None):
            pid = find(args.match)
        memory = process_memory(pid) if pid else None
        if memory is None and args.pid:
            stop['why'] = 'process-exited'
        vm = vm_counters()
        level, free = pressure()
        samples.append({'t': round(time.time() - started, 1), 'pid': pid,
                        'phys_footprint': memory[0] if memory else None, 'resident': memory[1] if memory else None,
                        'swapouts': vm.get('Swapouts', 0), 'pageouts': vm.get('Pageouts', 0),
                        'compressor_pages': vm.get('Pages occupied by compressor', 0),
                        'pressure_level': level, 'free_percent': free})
        if time.time() - started > args.max_seconds:
            stop['why'] = 'max-seconds'
            break
        deadline = time.time() + args.interval
        while stop['why'] is None and time.time() < deadline:
            time.sleep(0.1)
    last = vm_counters()
    page = first['page']
    footprints = [s['phys_footprint'] for s in samples if s['phys_footprint'] is not None]
    residents = [s['resident'] for s in samples if s['resident'] is not None]
    levels = [s['pressure_level'] for s in samples]
    frees = [s['free_percent'] for s in samples if s['free_percent'] is not None]
    swapouts = last.get('Swapouts', 0) - first.get('Swapouts', 0)
    pageouts = last.get('Pageouts', 0) - first.get('Pageouts', 0)
    summary = {
        'stopped': stop['why'], 'seconds': round(time.time() - started, 1), 'samples': len(samples),
        'peak_phys_footprint_gb': round(max(footprints) / 1e9, 3) if footprints else None,
        'peak_resident_gb': round(max(residents) / 1e9, 3) if residents else None,
        'swapouts_pages': swapouts, 'swapouts_mib': round(swapouts * page / (1 << 20), 1),
        'pageouts_pages': pageouts, 'pageouts_mib': round(pageouts * page / (1 << 20), 1),
        'compressor_pages_start': first.get('Pages occupied by compressor', 0),
        'compressor_pages_peak': max((s['compressor_pages'] for s in samples), default=0),
        'worst_pressure_level': max(levels) if levels else None, 'critical_seen': any(level >= 4 for level in levels),
        'least_free_percent': min(frees) if frees else None,
    }
    print('SUMMARY ' + json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as out:
            json.dump({'summary': summary, 'samples': samples}, out, indent=1)


if __name__ == '__main__':
    sys.exit(main())
