"""No raw NUL byte in a source file.  A shell heredoc that loses the backslash of '\\0' leaves a literal 0x00 inside a
character constant: gcc only warns ("null character(s) preserved in literal"), MSVC stops with an error (0.1.43.1's
generate.cpp; expert_source.cpp and pinned.cu after it).  No binary, no GPU.

    python -m unittest tools.test_source_nul_bytes
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".hip", ".mm", ".metal", ".py", ".cmake",
                   ".txt", ".sh", ".bat", ".ps1", ".js", ".html", ".css", ".json", ".jinja", ".md", ".in"}
SOURCE_DIRS = ("src", "include", "sycl", "metal", "serve", "tools", "tests", "cmake", "bench", "docs")


def source_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *SOURCE_DIRS, "*.py", "*.txt", "*.sh", "*.bat"],
                             cwd=ROOT, capture_output=True, check=True).stdout
        names = [n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n]
        paths = [ROOT / n for n in names]
    except (OSError, subprocess.CalledProcessError):   # not a git checkout (a release zip): walk the tree instead
        paths = [p for d in SOURCE_DIRS if (ROOT / d).is_dir() for p in (ROOT / d).rglob("*")]
        paths += [p for p in ROOT.iterdir() if p.is_file()]
    return sorted({p for p in paths if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES})


def nul_lines(path: Path) -> list[int]:
    data = path.read_bytes()
    if b"\0" not in data:
        return []
    return [i + 1 for i, line in enumerate(data.split(b"\n")) if b"\0" in line]


class SourceNulBytes(unittest.TestCase):
    def test_no_source_file_holds_a_nul_byte(self):
        files = source_files()
        self.assertGreater(len(files), 50, "the source scan found almost nothing")
        bad = {str(p.relative_to(ROOT)): nul_lines(p) for p in files}
        bad = {k: v for k, v in bad.items() if v}
        self.assertEqual(bad, {}, "raw NUL bytes (write '\\0' instead) at file: [lines]")

    def test_the_check_sees_a_nul(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.cpp"
            p.write_bytes(b"int a;\nchar c = '\0';\n")
            self.assertEqual(nul_lines(p), [2])


if __name__ == "__main__":
    unittest.main()
