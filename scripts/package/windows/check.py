"""Checks of a Windows package tree, run by package.ps1.

  check.py paths ROOT NEEDLE...       no file under ROOT holds a NEEDLE (case-insensitive, ASCII or UTF-16LE)
  check.py machine ROOT 0xAA64        every .exe, .dll and .pyd under ROOT is built for that PE machine
  check.py modules SRC STAGE          every repo module that serve/server.py imports is in STAGE (both are
                                      edge0-strata folders); tools/dflash2_pack.py is imported by name at run time
"""
import modulefinder
import re
import struct
import sys
from pathlib import Path


def paths(root: Path, needles: list[str]) -> int:
    forms = set()
    for n in needles:
        for s in (n, n.replace("\\", "/"), n.replace("/", "\\")):
            forms.add(s.lower())
    pats = [re.compile(re.escape(s.encode(enc)), re.IGNORECASE) for s in forms for enc in ("ascii", "utf-16-le")]
    bad = 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        data = f.read_bytes()
        hits = sorted({m.group().decode("utf-16-le" if b"\0" in m.group() else "ascii")
                       for p in pats for m in p.finditer(data)})
        if hits:
            print(f"build path in {f.relative_to(root)}: {', '.join(hits)}")
            bad += 1
    return 1 if bad else 0


def machine(root: Path, want: int) -> int:
    bad = 0
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in (".exe", ".dll", ".pyd")):
        with f.open("rb") as h:
            head = h.read(4096)
        if head[:2] != b"MZ":
            print(f"not a PE file: {f.relative_to(root)}")
            bad += 1
            continue
        off = struct.unpack_from("<I", head, 0x3C)[0]
        got = struct.unpack_from("<H", head, off + 4)[0] if off + 6 <= len(head) else -1
        if got != want:
            print(f"PE machine 0x{got:04X}, want 0x{want:04X}: {f.relative_to(root)}")
            bad += 1
    return 1 if bad else 0


def repo_modules(top: Path) -> set[str]:
    finder = modulefinder.ModuleFinder(path=[str(top / "serve"), str(top), str(top / "tools")] + sys.path)
    finder.run_script(str(top / "serve" / "server.py"))
    out = {"tools/dflash2_pack.py"}
    for mod in finder.modules.values():
        if mod.__file__:
            try:
                out.add(Path(mod.__file__).resolve().relative_to(top.resolve()).as_posix())
            except ValueError:
                pass
    return out


def modules(src: Path, stage: Path) -> int:
    missing = sorted(m for m in repo_modules(src) if not (stage / m).is_file())
    for m in missing:
        print(f"missing module: {m}")
    return 1 if missing else 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "paths":
        return paths(Path(argv[1]), argv[2:])
    if len(argv) == 3 and argv[0] == "machine":
        return machine(Path(argv[1]), int(argv[2], 0))
    if len(argv) == 3 and argv[0] == "modules":
        return modules(Path(argv[1]), Path(argv[2]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
