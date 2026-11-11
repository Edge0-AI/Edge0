"""Shared fixtures for the `strata launch` tests: an isolated home, a fake computer, and a text dump of a Textual screen."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from unittest import mock

HW_12 = {"ram_gb": 64.0, "os": "Test OS", "cpu": {"avx2": True, "name": "test cpu"},
         "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 12.0, "usable": True, "vendor": "nvidia"}]}
HW_24 = {**HW_12, "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}]}
HW_2X16 = {**HW_12, "gpus": [{"index": 0, "name": "NVIDIA RTX A4000", "vram_gb": 16.0, "usable": True, "vendor": "nvidia"},
                             {"index": 1, "name": "NVIDIA RTX A4000", "vram_gb": 16.0, "usable": True, "vendor": "nvidia"}]}
HW_32GB_RAM = {**HW_12, "ram_gb": 32.0}


class Home:
    """A throw-away per-user home: profiles, presets, launch presets, state, components and install folder."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-launch-test-")
        base = Path(self.tmp.name)
        self.base = base
        self.env = {
            "STRATA_PROFILE_DIR": str(base / "config" / "profiles"),
            "STRATA_PRESET_DIR": str(base / "config" / "presets"),
            "STRATA_LAUNCH_PRESET_DIR": str(base / "config" / "launch-presets"),
            "STRATA_COMPONENTS_DIR": str(base / "components"),
            "STRATA_INSTALL_DIR": str(base / "install"),
            "STRATA_RUNTIME_DIR": str(base / "runtime"),
            "XDG_STATE_HOME": str(base / "state"),
            "LOCALAPPDATA": str(base / "local"),
            "HOME": str(base / "home"),
            "USERPROFILE": str(base / "home"),
        }
        (base / "home").mkdir(parents=True, exist_ok=True)
        self.patch = mock.patch.dict(os.environ, self.env)

    def __enter__(self) -> "Home":
        self.patch.start()
        return self

    def __exit__(self, *exc) -> None:
        self.patch.stop()
        self.tmp.cleanup()


def catalog(hw: dict | None = None, root: Path | None = None):
    from strata_cli.launch.catalog import Catalog
    return Catalog(root=root or Path(os.environ["STRATA_INSTALL_DIR"]), hardware=hw or HW_12)


def text_of(app) -> str:
    """What the current screen shows, one line per terminal row (for assertions and the text previews)."""
    strips = app.screen._compositor.render_strips()
    return "\n".join(s.text.rstrip() for s in strips)


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    from strata_cli.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as e:
            code = int(e.code or 0)
    return code, out.getvalue(), err.getvalue()
