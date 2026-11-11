"""tools/_paths.py - where the tools find llama.cpp's gguf-py (for the MTP packer and the fixtures).

Order: $STRATA_GGUF_PY, then <engine>/third_party/llama.cpp/gguf-py (what setup.py clones), then the development
tree's ../../.ref/llama.cpp/gguf-py, then (#3269) the install folders: in the Control package setup.py writes third_party/
to the per-user install folder ($STRATA_INSTALL_DIR, else the app's folder, the Terminal's remembered installs), not
beside the tools, so a tool run by hand from the package finds it there.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def _install_dirs() -> list[Path]:
    """The per-user install folders a Control-mode setup.py writes third_party/ into. Best effort: never raises."""
    out: list[Path] = []
    raw = os.environ.get("STRATA_INSTALL_DIR")
    if raw:
        out.append(Path(raw).expanduser())
    try:
        if str(ENGINE) not in sys.path:
            sys.path.append(str(ENGINE))
        from strata_control import discovery
        out.append(discovery.app_install_dir())
        out += [pl.path for pl in discovery.places(here=ENGINE)]
    except Exception:
        pass
    return out


def gguf_py() -> str:
    cands = [os.environ.get("STRATA_GGUF_PY"), ENGINE / "third_party" / "llama.cpp" / "gguf-py",
             ENGINE.parents[min(1, len(ENGINE.parents) - 1)] / ".ref" / "llama.cpp" / "gguf-py"]   # the dev tree; a shallow install (G:\Strata) has no grandparent
    cands += [d / "third_party" / "llama.cpp" / "gguf-py" for d in _install_dirs()]
    for c in cands:
        if c and (Path(c) / "gguf").is_dir():
            return str(c)
    sys.exit("llama.cpp's gguf-py was not found. Set STRATA_GGUF_PY to the gguf-py folder of any llama.cpp checkout "
             "(<llama.cpp>/gguf-py), or run setup.py once (it downloads llama.cpp into third_party/ of the install "
             "folder; in the desktop app or the Terminal package that folder is STRATA_INSTALL_DIR, which this tool also "
             "looks in)")


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)
