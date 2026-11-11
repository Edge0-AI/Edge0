"""serve/loader_diag.py - name the file a Windows engine could not load.

A Windows program that imports a DLL which is not there ends with exit code 0xC0000135 and prints nothing. The person only sees
"the engine stopped". This reads the engine's import table (the same reader as packaging/binary_deps.py, copied because the
packaging folder does not travel with Control) and lists the DLLs that are neither beside the engine, nor in the Windows
folders, nor part of the GPU driver, so the start error can say which file is missing.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

LOADER_FAILURE_CODES = (0xC0000135,)          # STATUS_DLL_NOT_FOUND

def pe_imports(path: Path) -> list[str]:
    """DLL names a PE file imports at load time (import table) plus delay-load imports."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("bad PE header")
    nsec, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    dd = opt + (112 if magic == 0x20B else 96)
    sections = []
    sec = opt + opt_size
    for i in range(nsec):
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, sec + i * 40 + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva: int) -> int:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError("RVA outside every section")

    def cstr(o: int) -> str:
        return data[o:data.index(b"\0", o)].decode("ascii", "replace")

    names: list[str] = []
    imp_rva, imp_size = struct.unpack_from("<II", data, dd + 8)
    if imp_rva and imp_size:
        o = off(imp_rva)
        while True:
            _, _, _, name_rva, _ = struct.unpack_from("<IIIII", data, o)
            if not name_rva:
                break
            names.append(cstr(off(name_rva)))
            o += 20
    delay_rva, delay_size = struct.unpack_from("<II", data, dd + 13 * 8)
    if delay_rva and delay_size:
        o = off(delay_rva)
        while True:
            attrs, name_rva = struct.unpack_from("<II", data, o)
            if not name_rva:
                break
            names.append(cstr(off(name_rva)))
            o += 32
    return names


def missing_files(exe, system_dirs=None) -> list[str]:
    """DLL names the engine (and the DLLs beside it) import that cannot be found: not in the engine's folder, not a Windows or
    driver library, and not in a Windows folder on this PC. Sorted, without duplicates; [] when nothing is missing or the file
    cannot be read (this is advice for an error message, it never raises)."""
    try:
        exe = Path(exe)
        folder = exe.parent
        here = {p.name.lower() for p in folder.iterdir() if p.is_file()}
        if system_dirs is None:
            root = os.environ.get("SystemRoot") or r"C:\Windows"
            system_dirs = [Path(root) / "System32", Path(root)]
        files = [exe] + [p for p in sorted(folder.glob("*.dll"))]
        missing: set[str] = set()
        for f in files:
            try:
                names = pe_imports(f)
            except (ValueError, OSError, struct.error):
                continue
            for name in names:
                low = name.lower()
                if low in here or low.startswith(("api-ms-win-", "ext-ms-win-")):
                    continue                    # beside the engine, or a virtual API-set name Windows resolves itself
                if any((d / name).exists() for d in system_dirs):
                    continue
                missing.add(name)               # not found anywhere the loader looks (a missing driver DLL is named too)
        return sorted(missing, key=str.lower)
    except OSError:
        return []


def missing_note(rc, exe) -> str:
    """The sentence for the start error, or "" when the exit status is not a missing-library one or nothing can be named."""
    if rc is None or (rc & 0xFFFFFFFF) not in LOADER_FAILURE_CODES or not exe:
        return ""
    names = missing_files(exe)
    if not names:
        return ""
    return (f"\n  The engine is missing a file it needs ({', '.join(names)}). Reinstall the engine in Settings > Engine, "
            f"or send us the log.")
