"""tools/hip/dll_closure.py - which DLLs a DLL needs, read from the PE import tables (no pefile, no objdump).

    python tools/hip/dll_closure.py DIR amdhip64_7.dll amd_comgr.dll      # prints the closure inside DIR

Why: the Windows HIP engine puts amdhip64_7.dll and amd_comgr.dll next to strata.exe so that Windows binds THEM and not
an AMD driver's copy in System32 (#468, #461).  But amdhip64_7.dll itself imports rocm_kpack.dll, which imports the
MSVC runtime; with only the two DLLs beside the exe the full-path load fails (error 126, ERROR_MOD_NOT_FOUND), Windows
falls back to System32's copy, and the first big prompt dies with hipErrorInvalidDeviceFunction (#461, maraa081).  So
what goes beside the exe is the DLLs' whole dependency closure, computed here from the files actually shipped rather
than from a list of names that goes stale with the next ROCm.

A name is followed only when the DLL exists in `search_dir` (the bundle's rocm/bin).  Everything else - kernel32,
user32, setupapi, the api-ms-win-* forwarders - is a Windows system DLL and is left alone.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def pe_imports(path) -> list[str]:
    """The DLL names in a PE file's import table and delay-import table, in file order.  [] for a file that is not a
    PE image (or is truncated): the caller treats it as a leaf."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    try:
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        nsec, = struct.unpack_from("<H", data, pe + 6)
        opt_size, = struct.unpack_from("<H", data, pe + 20)
        opt = pe + 24
        magic, = struct.unpack_from("<H", data, opt)
        dd = opt + (112 if magic == 0x20B else 96)          # PE32+ : 112, PE32 : 96
        secs = []
        sec0 = opt + opt_size
        for i in range(nsec):
            o = sec0 + 40 * i
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, o + 8)
            secs.append((va, max(vsize, rsize), raw))

        def off(rva):
            for va, size, raw in secs:
                if va <= rva < va + size:
                    return rva - va + raw
            return None

        def cstr(o):
            e = data.index(b"\0", o)
            return data[o:e].decode("ascii", "replace")

        names: list[str] = []
        imp_rva, = struct.unpack_from("<I", data, dd + 8 * 1)                 # directory 1: imports
        o = off(imp_rva) if imp_rva else None
        while o is not None:
            name_rva = struct.unpack_from("<I", data, o + 12)[0]
            if name_rva == 0:
                break
            n = off(name_rva)
            if n is not None:
                names.append(cstr(n))
            o += 20
        dly_rva, = struct.unpack_from("<I", data, dd + 8 * 13)                # directory 13: delay imports
        o = off(dly_rva) if dly_rva else None
        while o is not None:
            attrs, name_rva = struct.unpack_from("<II", data, o)
            if name_rva == 0:
                break
            n = off(name_rva)                                                 # (RVA form, attrs bit 0)
            if n is not None:
                names.append(cstr(n))
            o += 32
        return names
    except (struct.error, ValueError, IndexError):
        return []


def dll_closure(search_dir, roots, imports=pe_imports) -> list[str]:
    """`roots` (file names inside `search_dir`) plus every DLL in `search_dir` they import, followed recursively.
    The names are returned as spelled on disk, roots first, the rest in discovery order.  A root that is not in the
    directory is skipped; an import that is not in the directory (a Windows system DLL) is not followed."""
    d = Path(search_dir)
    on_disk = {p.name.lower(): p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".dll"} \
        if d.is_dir() else {}
    out: list[str] = []
    todo = [r for r in roots]
    seen: set[str] = set()
    while todo:
        name = todo.pop(0)
        low = name.lower()
        if low in seen or low not in on_disk:
            continue
        seen.add(low)
        real = on_disk[low]
        out.append(real)
        for dep in imports(d / real):
            if dep.lower() not in seen:
                todo.append(dep)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    for n in dll_closure(sys.argv[1], sys.argv[2:]):
        print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
