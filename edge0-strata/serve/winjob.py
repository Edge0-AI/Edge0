"""serve/winjob.py - on Windows, tie the processes Strata starts to the server's own lifetime.

Closing the console window, Task Manager or a crash end the server without running its cleanup, and the engine,
the vision encoder and the MCP servers kept running on their own. `contain(proc)` puts a child in a job object that
is set to kill everything in it when its last handle closes - and the only handle is this process's, which the OS
closes however the server ends. Processes the child starts later join the same job. The server itself stays out of
the job, so a browser that `--open` starts is not tied to it. Each child (and the server) is also opted out of Windows
power throttling, see `no_throttle` (#691). Elsewhere, and if the job cannot be made, a no-op.
"""
from __future__ import annotations

import os
import threading

_job = None
_lock = threading.Lock()

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)

    def _make_job():
        job = _k32.CreateJobObjectW(None, None)     # no security attributes: the handle is not inherited
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        return job

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    _k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE

    def no_throttle(handle) -> bool:
        """Opt a process out of Windows power throttling (EcoQoS).  Without it Windows 11 treats a process whose console
        window is minimized as background work and, on a hybrid CPU, moves its threads to the E-cores: generation
        slowed by about 20% while the server window was minimized (#691, i9-13980HX: 36.6 -> 43.2 tok/s)."""
        for mask in (0x1 | 0x4, 0x1):               # EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both off (Windows 11);
            info = _PowerThrottling(1, mask, 0)     # Windows 10 knows the first only: ERROR_INVALID_PARAMETER for both
            if _k32.SetProcessInformation(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):   # ProcessPowerThrottling
                return True
        return False

    no_throttle(_k32.GetCurrentProcess())           # the server itself (tokenizing, streaming) too
else:
    def no_throttle(handle) -> bool:
        return False


def terminate() -> bool:
    """End every process in the job now (the deadline kill of an owner that must not leave a tree behind). False when
    there is no job (nothing was contained, or not Windows)."""
    if os.name != "nt" or not _job:
        return False
    return bool(_k32.TerminateJobObject(_job, 1))


def contain(proc) -> bool:
    """Put a started subprocess.Popen in the kill-on-close job. True if it is in; False (harmlessly) otherwise."""
    global _job
    if os.name != "nt" or proc is None:
        return False
    with _lock:
        if _job is None:
            _job = _make_job() or 0
        if not _job:
            return False
    try:
        no_throttle(int(proc._handle))
        return bool(_k32.AssignProcessToJobObject(_job, int(proc._handle)))
    except (AttributeError, OSError, ValueError):
        return False
