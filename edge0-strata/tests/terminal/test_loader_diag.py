"""A Windows engine that cannot load a DLL ends with 0xC0000135 and no message. The start error must name the missing file
(tester report 2026-10-17: cublas64_13.dll was not in the engine component and the app blamed memory)."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import binary_deps  # noqa: E402
from serve import loader_diag  # noqa: E402
from serve.server import start_exit_note  # noqa: E402
from test_engine_deps import make_pe  # noqa: E402


class MissingFiles(unittest.TestCase):
    def folder(self, d, *, with_cublas):
        d = Path(d)
        (d / "strata.exe").write_bytes(make_pe(["cublas64_13.dll", "KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]))
        (d / "strata_kq.dll").write_bytes(make_pe(["KERNEL32.dll"]))
        if with_cublas:
            (d / "cublas64_13.dll").write_bytes(make_pe(["cublasLt64_13.dll", "KERNEL32.dll"]))
            (d / "cublasLt64_13.dll").write_bytes(make_pe(["KERNEL32.dll"]))
        return d / "strata.exe"

    def test_reader_matches_the_packaging_reader(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self.folder(d, with_cublas=True)
            self.assertEqual(loader_diag.pe_imports(exe), binary_deps.pe_imports(exe))

    def test_names_the_missing_dll(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=False)
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), ["cublas64_13.dll"])

    def test_transitive_import_is_named(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=True)
            (exe.parent / "cublasLt64_13.dll").unlink()
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), ["cublasLt64_13.dll"])

    def test_complete_folder_names_nothing(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=True)
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), [])

    def test_unreadable_file_never_raises(self):
        self.assertEqual(loader_diag.missing_files(Path("no-such-folder") / "strata.exe"), [])

    def test_start_note_only_for_the_loader_status(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self.folder(d, with_cublas=False)
            note = start_exit_note(0xC0000135, exe)
            self.assertIn("0xC0000135", note)
            self.assertIn("cublas64_13.dll", note)
            self.assertIn("Reinstall the engine in Settings > Engine", note)
            self.assertNotIn("cublas64_13.dll", start_exit_note(1, exe))
            self.assertNotIn("missing a file", start_exit_note(0xC0000135, None))


if __name__ == "__main__":
    unittest.main()
