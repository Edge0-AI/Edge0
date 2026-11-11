"""#461: the DLLs the Windows HIP runtime needs beside strata.exe, from the PE import tables.

    python tools/test_dll_closure.py          (the real-zip test skips when the 0.1.40.2 HIP zip is not here)

Covers tools/hip/dll_closure.py and setup.hip_runtime_beside_exe on a fake engine folder.
"""
import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "hip"))
sys.path.insert(0, str(ROOT))

import dll_closure as dc  # noqa: E402
import setup  # noqa: E402


def fake_pe(imports, delay=()):
    """A minimal PE32+ image whose import table (and delay-import table) names `imports` (`delay`)."""
    text = bytearray()
    va = 0x1000

    def add_str(s):
        off = len(text)
        text.extend(s.encode() + b"\0")
        return va + off

    names = [add_str(n) for n in imports]
    dnames = [add_str(n) for n in delay]
    while len(text) % 8:
        text.append(0)
    imp_off = len(text)
    for rva in names:
        text.extend(struct.pack("<IIIII", 0, 0, 0, rva, 0))
    text.extend(b"\0" * 20)
    dly_off = len(text)
    for rva in dnames:
        text.extend(struct.pack("<IIIIIIII", 1, rva, 0, 0, 0, 0, 0, 0))
    text.extend(b"\0" * 32)
    opt_size = 112 + 16 * 8
    pe = 0x80
    hdr = bytearray(0x200)
    hdr[0:2] = b"MZ"
    struct.pack_into("<I", hdr, 0x3C, pe)
    hdr[pe:pe + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", hdr, pe + 4, 0x8664, 1, 0, 0, 0, opt_size, 0x2022)
    opt = pe + 24
    struct.pack_into("<H", hdr, opt, 0x20B)
    dd = opt + 112
    struct.pack_into("<II", hdr, dd + 8 * 1, va + imp_off, 20 * (len(names) + 1))
    struct.pack_into("<II", hdr, dd + 8 * 13, va + dly_off, 32 * (len(dnames) + 1))
    sec = opt + opt_size
    hdr[sec:sec + 8] = b".idata\0\0"
    struct.pack_into("<IIII", hdr, sec + 8, len(text), va, len(text), 0x200)
    return bytes(hdr) + bytes(text)


class Closure(unittest.TestCase):
    def test_reads_imports_and_delay_imports(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.dll"
            f.write_bytes(fake_pe(["KERNEL32.dll", "b.dll"], delay=["c.dll"]))
            self.assertEqual(dc.pe_imports(f), ["KERNEL32.dll", "b.dll", "c.dll"])

    def test_not_a_pe_is_a_leaf(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.dll"
            f.write_bytes(b"hip-3686")
            self.assertEqual(dc.pe_imports(f), [])
            self.assertEqual(dc.pe_imports(Path(d) / "missing.dll"), [])

    def test_follows_only_what_is_shipped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "AmdHip64_7.dll").write_bytes(fake_pe(["KERNEL32.dll", "ROCM_KPACK.dll", "SETUPAPI.dll"]))
            (d / "rocm_kpack.dll").write_bytes(fake_pe(["MSVCP140.dll", "VCRUNTIME140.dll", "kernel32.dll"]))
            (d / "msvcp140.dll").write_bytes(fake_pe(["VCRUNTIME140.dll", "api-ms-win-crt-heap-l1-1-0.dll"]))
            (d / "vcruntime140.dll").write_bytes(fake_pe(["KERNEL32.dll"]))
            (d / "amd_comgr.dll").write_bytes(fake_pe(["msvcp140.dll"]))
            (d / "hipblas.dll").write_bytes(fake_pe(["amdhip64_7.dll", "rocblas.dll"]))     # not a root: left out
            got = dc.dll_closure(d, ["AmdHip64_7.dll", "amd_comgr.dll", "not_there.dll"])
            self.assertEqual(sorted(x.lower() for x in got),
                             ["amd_comgr.dll", "amdhip64_7.dll", "msvcp140.dll", "rocm_kpack.dll", "vcruntime140.dll"])
            self.assertEqual(got[:2], ["AmdHip64_7.dll", "amd_comgr.dll"])               # roots first, as spelled

    def test_cycle_terminates(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.dll").write_bytes(fake_pe(["b.dll"]))
            (d / "b.dll").write_bytes(fake_pe(["a.dll"]))
            self.assertEqual(dc.dll_closure(d, ["a.dll"]), ["a.dll", "b.dll"])

    ZIP = ROOT.parent / "release" / "z04002" / "strata-windows-x64-hip.zip"

    @unittest.skipUnless(ZIP.exists(), "the 0.1.40.2 HIP zip is not here")
    def test_real_zip(self):
        """The reporter's finding: amdhip64_7.dll needs rocm_kpack.dll and the MSVC runtime, which the 0.1.40.2 setup
        did not put beside strata.exe."""
        with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(self.ZIP) as z:
            for n in z.namelist():
                if n.startswith("rocm/bin/") and n.endswith(".dll"):
                    z.extract(n, d)
            got = dc.dll_closure(Path(d) / "rocm" / "bin", ["amdhip64_7.dll", "amd_comgr.dll"])
            self.assertEqual(sorted(got), ["amd_comgr.dll", "amdhip64_7.dll", "msvcp140.dll", "rocm_kpack.dll",
                                           "vcruntime140.dll", "vcruntime140_1.dll"])
            self.assertNotIn("hipblas.dll", got)            # strata.exe loads that one from rocm/bin by PATH


class SetupCopy(unittest.TestCase):
    def test_copies_the_closure_not_the_rest(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            rb = eng / "rocm" / "bin"
            rb.mkdir(parents=True)
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "lib_dirs": ["rocm/bin"]}))
            (rb / "amdhip64_7.dll").write_bytes(fake_pe(["KERNEL32.dll", "rocm_kpack.dll"]))
            (rb / "rocm_kpack.dll").write_bytes(fake_pe(["MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]))
            for n in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
                (rb / n).write_bytes(fake_pe(["KERNEL32.dll"]))
            (rb / "amd_comgr.dll").write_bytes(fake_pe(["msvcp140.dll"]))
            (rb / "rocblas.dll").write_bytes(fake_pe(["amdhip64_7.dll"]))
            (rb / "hipblas.dll").write_bytes(fake_pe(["rocblas.dll"]))
            # a 0.1.40.2 install: the two runtime DLLs beside the exe, and nothing else
            (eng / "amdhip64_7.dll").write_bytes((rb / "amdhip64_7.dll").read_bytes())
            with mock.patch.object(setup, "warn", lambda *a: None):
                setup.hip_runtime_beside_exe(eng)
            beside = sorted(p.name for p in eng.glob("*.dll"))
            self.assertEqual(beside, ["amd_comgr.dll", "amdhip64_7.dll", "msvcp140.dll", "rocm_kpack.dll",
                                      "vcruntime140.dll", "vcruntime140_1.dll"])
            self.assertFalse((eng / "rocblas.dll").exists())
            self.assertFalse((eng / "hipblas.dll").exists())
            before = {p.name: p.stat().st_mtime_ns for p in eng.glob("*.dll")}
            with mock.patch.object(setup, "warn", lambda *a: None):
                setup.hip_runtime_beside_exe(eng)           # a second run changes nothing
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in eng.glob("*.dll")})

    def test_unreadable_helper_falls_back_to_the_runtime_pair(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "amdhip64_7.dll").write_bytes(b"x")
            (d / "amd_comgr.dll").write_bytes(b"y")
            (d / "rocm_kpack.dll").write_bytes(b"z")
            with mock.patch.object(setup, "ROOT", d):        # no tools/hip/dll_closure.py under this root
                self.assertEqual(setup.hip_runtime_closure(d), ["amd_comgr.dll", "amdhip64_7.dll"])


if __name__ == "__main__":
    unittest.main()
