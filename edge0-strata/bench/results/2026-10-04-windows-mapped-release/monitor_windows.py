"""Record Windows host memory and physical-disk counters during a local experiment."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time

import psutil


class PerformanceInformation(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
            "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
            "KernelNonpaged", "PageSize",
        )
    ] + [(name, wintypes.DWORD) for name in (
        "HandleCount", "ProcessCount", "ThreadCount",
    )]


performance = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
performance.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
performance.restype = wintypes.BOOL


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set_bytes", "working_set_bytes", "peak_paged_pool_bytes",
            "paged_pool_bytes", "peak_nonpaged_pool_bytes", "nonpaged_pool_bytes",
            "commit_bytes", "peak_commit_bytes", "private_commit_bytes",
            "private_working_set_bytes",
        )
    ] + [("shared_commit_bytes", ctypes.c_ulonglong)]


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
process_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
process_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
process_memory.restype = wintypes.BOOL


def extended_memory(process):
    handle = kernel.OpenProcess(0x1000, False, process.pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        if not process.is_running():
            raise psutil.NoSuchProcess(process.pid)
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = ProcessMemoryCounters()
        info.cb = ctypes.sizeof(info)
        if not process_memory(handle, ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {name: getattr(info, name) for name, _ in info._fields_ if name != "cb"}
    finally:
        kernel.CloseHandle(handle)


def sample(server):
    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not performance(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    processes = []
    if server is not None:
        try:
            descendants = [server, *server.children(recursive=True)]
        except psutil.NoSuchProcess:
            descendants = []
        for process in descendants:
            try:
                with process.oneshot():
                    processes.append({
                        "pid": process.pid,
                        "name": process.name(),
                        "memory": process.memory_info()._asdict(),
                        "windows_memory": extended_memory(process),
                        "cpu_seconds": process.cpu_times()._asdict(),
                        "io_counters": process.io_counters()._asdict(),
                    })
            except psutil.NoSuchProcess:
                continue
    battery = psutil.sensors_battery()
    return {
        "epoch_s": time.time(),
        "memory": psutil.virtual_memory()._asdict(),
        "pagefile": psutil.swap_memory()._asdict(),
        "commit_bytes": info.CommitTotal * info.PageSize,
        "commit_limit_bytes": info.CommitLimit * info.PageSize,
        "commit_peak_bytes": info.CommitPeak * info.PageSize,
        "system_cache_bytes": info.SystemCache * info.PageSize,
        "physical_disks": {
            name: counters._asdict()
            for name, counters in psutil.disk_io_counters(perdisk=True).items()
        },
        "ac_power": battery.power_plugged if battery is not None else None,
        "processes": processes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, help="Optional bounded diagnostic run")
    args = parser.parse_args()
    if args.interval <= 0 or (args.samples is not None and args.samples < 1):
        parser.error("Interval and sample limit must be positive")
    if args.stop_file.exists():
        parser.error("The stop file already exists; choose a new experiment path")
    server = psutil.Process(args.server_pid) if args.server_pid is not None else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        count = 0
        while not args.stop_file.exists():
            started = time.monotonic()
            output.write(json.dumps(sample(server)) + "\n")
            output.flush()
            count += 1
            if args.samples is not None and count >= args.samples:
                break
            if server is not None and not server.is_running():
                break
            time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
