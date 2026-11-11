"""What the tuner needs to know about this PC, read without ever raising.

Every probe here is optional: a missing tool or an unreadable file gives ``None`` (or an empty list), never an
exception, because the tuner must still run on a machine whose sensors it cannot read, only with fewer guards.
A guard that cannot read its input says so in the report ("the free graphics memory could not be read"); it does not
pretend the memory was free.

Stdlib only, like the rest of Strata Control.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MIB = 1024 * 1024


# --------------------------------------------------------------------------- memory

def memory() -> dict[str, int | None]:
    """System RAM in MiB: ``total`` and ``available`` (what a new process could use without swapping)."""
    try:
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {"total_mib": int(status.ullTotalPhys // MIB), "available_mib": int(status.ullAvailPhys // MIB)}
        elif sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                       timeout=5).stdout.strip()) // MIB
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", vm).group(1)) if re.search(r"page size of (\d+)", vm) else 16384
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative"):
                m = re.search(rf"{key}:\s+(\d+)", vm)
                pages += int(m.group(1)) if m else 0
            return {"total_mib": total, "available_mib": pages * page // MIB}
        else:
            info: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])
            return {"total_mib": info["MemTotal"] // 1024, "available_mib": info["MemAvailable"] // 1024}
    except Exception:  # noqa: BLE001 - a probe must not raise
        pass
    return {"total_mib": None, "available_mib": None}


# --------------------------------------------------------------------------- graphics cards

def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return None


def gpus() -> list[dict[str, Any]]:
    """Graphics cards with their memory: ``name``, ``vram_total_mib``, ``vram_used_mib`` (None when unreadable),
    ``driver``, ``vendor``. NVIDIA through nvidia-smi, AMD through the kernel's sysfs files; Intel and Apple report
    their name only where it is known, and leave the memory unreadable."""
    out: list[dict[str, Any]] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        text = _run([smi, "--query-gpu=index,name,memory.total,memory.used,driver_version",
                     "--format=csv,noheader,nounits"])
        for line in (text or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    out.append({"vendor": "nvidia", "index": int(parts[0]), "name": parts[1],
                                "vram_total_mib": int(float(parts[2])), "vram_used_mib": int(float(parts[3])),
                                "driver": parts[4]})
                except ValueError:
                    continue
    if not out and sys.platform.startswith("linux"):
        base = Path("/sys/class/drm")
        if base.is_dir():
            index = 0
            for card in sorted(base.glob("card[0-9]*")):
                dev = card / "device"
                try:
                    vendor = (dev / "vendor").read_text().strip()
                    total = int((dev / "mem_info_vram_total").read_text()) // MIB
                    used = int((dev / "mem_info_vram_used").read_text()) // MIB
                except (OSError, ValueError):
                    continue
                if vendor != "0x1002":
                    continue
                name = None
                try:
                    name = (dev / "product_name").read_text().strip()
                except OSError:
                    pass
                out.append({"vendor": "amd", "index": index, "name": name or f"AMD GPU {index}",
                            "vram_total_mib": total, "vram_used_mib": used, "driver": None})
                index += 1
    return out


def gpu_used_mib(indexes: list[int] | None = None) -> int | None:
    """Graphics memory in use now, summed over the cards the model runs on (all when ``indexes`` is None)."""
    cards = gpus()
    if not cards:
        return None
    vals = [c["vram_used_mib"] for c in cards if c.get("vram_used_mib") is not None
            and (not indexes or c["index"] in indexes)]
    return sum(vals) if vals else None


# --------------------------------------------------------------------------- processor

def cpu() -> dict[str, Any]:
    """``model``, ``logical`` and ``physical`` thread counts, and ``hybrid`` (performance plus efficiency cores)."""
    logical = os.cpu_count() or 1
    model = platform.processor() or platform.machine()
    physical: int | None = None
    hybrid = False
    try:
        if sys.platform.startswith("linux"):
            text = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", text)
            if m:
                model = m.group(1).strip()
            cores = set()
            for block in text.split("\n\n"):
                pid = re.search(r"physical id\s*:\s*(\d+)", block)
                cid = re.search(r"core id\s*:\s*(\d+)", block)
                if pid and cid:
                    cores.add((pid.group(1), cid.group(1)))
            physical = len(cores) or None
            hybrid = Path("/sys/devices/cpu_core").exists() and Path("/sys/devices/cpu_atom").exists()
        elif os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                    model = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip() or model
            except OSError:
                pass
            physical, hybrid = _windows_cores()
        elif sys.platform == "darwin":
            model = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or model).strip()
            phys = _run(["sysctl", "-n", "hw.physicalcpu"])
            physical = int(phys) if phys and phys.strip().isdigit() else None
            hybrid = False
    except Exception:  # noqa: BLE001
        pass
    return {"model": model, "logical": logical, "physical": physical, "hybrid": hybrid}


def _windows_cores() -> tuple[int | None, bool]:
    """(physical cores, hybrid) from GetLogicalProcessorInformationEx: hybrid when the cores have more than one
    efficiency class (performance and efficiency cores)."""
    try:
        import ctypes
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        k.GetLogicalProcessorInformationEx.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        size = wintypes.DWORD(0)
        k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))      # 0 = RelationProcessorCore
        buf = ctypes.create_string_buffer(size.value)
        if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return None, False
        raw = bytes(buf.raw[:size.value])
        classes, cores, offset = set(), 0, 0
        while offset + 10 <= len(raw):
            rel_size = int.from_bytes(raw[offset + 4:offset + 8], "little")
            if rel_size <= 0:
                break
            classes.add(raw[offset + 9])       # PROCESSOR_RELATIONSHIP.EfficiencyClass
            cores += 1
            offset += rel_size
        return (cores or None), len(classes) > 1
    except Exception:  # noqa: BLE001
        return None, False


_last_cpu: tuple[float, int, int] | None = None


def cpu_busy_percent(sample_s: float = 1.0) -> float | None:
    """How busy the whole CPU was over the last ``sample_s`` seconds (0-100), or None when unreadable."""
    def read() -> tuple[int, int] | None:
        try:
            if sys.platform.startswith("linux"):
                f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
                v = [int(x) for x in f]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                return sum(v), idle
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                    return None
                q = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
                return q(kern) + q(user), q(idle)
        except Exception:  # noqa: BLE001
            return None
        return None

    a = read()
    if a is None:
        return None
    time.sleep(sample_s)
    b = read()
    if b is None or b[0] == a[0]:
        return None
    return round(100.0 * (1.0 - (b[1] - a[1]) / (b[0] - a[0])), 1)


def other_engines(own_pids: set[int] | None = None) -> list[str]:
    """Command lines of Strata engines that this tune did not start (a second engine changes every number)."""
    own = own_pids or set()
    found: list[str] = []
    try:
        if os.name == "nt":
            text = _run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process -Filter \"Name='strata.exe'\" | "
                         "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"], timeout=20)
            if text and text.strip():
                data = json.loads(text)
                for row in data if isinstance(data, list) else [data]:
                    if int(row.get("ProcessId", 0)) not in own and "--serve" in (row.get("CommandLine") or ""):
                        found.append((row.get("CommandLine") or "")[:160])
        else:
            text = _run(["ps", "-eo", "pid=,args="]) or ""
            for line in text.splitlines():
                pid_s, _, args = line.strip().partition(" ")
                if pid_s.isdigit() and int(pid_s) not in own and re.search(r"(^|/)strata(\s|$)", args.split(" --serve")[0]) \
                        and " --serve" in args:
                    found.append(args[:160])
    except Exception:  # noqa: BLE001
        pass
    return found


# --------------------------------------------------------------------------- the key a preset is valid for

def _round_down(value: int | None, step: int) -> int | None:
    return None if value is None else value - value % step


def hardware_identity(*, pcie: str | None = None) -> dict[str, Any]:
    """Hardware in two parts. ``hard``: a change means the stored settings may not even start (graphics card,
    memory sizes, processor). ``soft``: a change deserves a short re-check (graphics driver)."""
    cards = gpus()
    mem = memory()
    c = cpu()
    return {
        "hard": {
            "gpus": [{"name": g["name"], "vram_mib": g["vram_total_mib"]} for g in cards],
            "cpu": c["model"], "logical": c["logical"], "physical": c["physical"], "hybrid": c["hybrid"],
            "ram_gib": _round_down((mem["total_mib"] or 0) // 1024 or None, 2),
            "pcie": pcie, "os": platform.system(),
        },
        "soft": {"driver": sorted({g["driver"] for g in cards if g.get("driver")}), "os_release": platform.release()},
    }


def engine_identity(exe: str | os.PathLike[str]) -> dict[str, Any]:
    """The engine build: version, backend, build id and executable SHA-256 from its manifest, plus the manifest's
    ``tuning_epoch`` (bumped when a kernel or default change can move a tuning result)."""
    folder = Path(exe).resolve().parent
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads((folder / "engine-component.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    sha = None
    for entry in manifest.get("files") or []:
        if entry.get("path") == manifest.get("entrypoint", Path(exe).name):
            sha = entry.get("sha256")
    return {"version": manifest.get("version"), "backend": manifest.get("backend"),
            "build_id": manifest.get("build_id"), "sha256": sha,
            "tuning_epoch": int(manifest.get("tuning_epoch") or 0), "path": str(exe)}


def model_identity(files: list[str | os.PathLike[str]]) -> dict[str, Any]:
    """The model file(s): size and a digest of the head, middle and tail of the first shard (the whole file would
    mean reading tens of gigabytes before every tune)."""
    from strata_control import tune_history
    out: list[dict[str, Any]] = []
    for f in files[:2]:
        ident = tune_history.file_identity(Path(f))
        out.append({"name": Path(f).name, "bytes": ident.get("bytes"), "digest": ident.get("digest"),
                    "method": ident.get("method")})
    first = out[0] if out else {}
    return {"files": out, "sha256_shard1": first.get("digest"), "bytes": first.get("bytes")}


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
