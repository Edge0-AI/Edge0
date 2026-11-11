"""The Linux AMD (HIP) engines hang under WSL (/dev/dxg present, /dev/kfd absent: librocdxg never completes a barrier-AND packet).
Setup and launch say so in plain words with one next step, as a warning, and never refuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import engine_component as ec


def dev(root: str, dxg: bool = True, kfd: bool = False) -> str:
    for name, on in (("dxg", dxg), ("kfd", kfd)):
        if on:
            Path(root, name).write_text("", encoding="utf-8")
    return root


class WslAmdNote(unittest.TestCase):
    def test_wsl_with_dxg_and_no_kfd_on_hip_gets_one_plain_sentence_with_a_next_step(self):
        with tempfile.TemporaryDirectory() as d:
            note = ec.wsl_amd_note("hip", dev(d), wsl=True)
        self.assertIn("do not run under WSL yet", note)
        self.assertIn("Windows", note)                     # the one next step: the Windows app or Terminal

    def test_nothing_to_say_elsewhere(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ec.wsl_amd_note("hip", dev(d), wsl=False))                    # a real Linux box
            self.assertIsNone(ec.wsl_amd_note("cuda", dev(d), wsl=True))                    # NVIDIA works under WSL
            self.assertIsNone(ec.wsl_amd_note(None, dev(d), wsl=True))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ec.wsl_amd_note("hip", dev(d, dxg=True, kfd=True), wsl=True))  # a /dev/kfd means the native driver
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ec.wsl_amd_note("hip", dev(d, dxg=False), wsl=True))           # no /dev/dxg: not the WSL GPU bridge

    def test_launch_warns_and_goes_on(self):
        events = []
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=mock.Mock(), emit=events.append)
        with mock.patch.object(ec, "wsl_amd_note", side_effect=lambda backend, *a, **k: "WSL note" if backend == "hip" else None), \
                mock.patch.object(ec, "detect_backend", return_value=("hip", "test")):
            r.warn_wsl_amd()
        self.assertEqual([(e.level, e.text) for e in events], [("warn", "WSL note")])
        self.assertIn("WSL note", r.notices)

    def test_launch_is_silent_when_the_backend_is_not_hip(self):
        events = []
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=mock.Mock(), emit=events.append)
        with mock.patch.object(ec, "wsl_amd_note", side_effect=lambda backend, *a, **k: "WSL note" if backend == "hip" else None), \
                mock.patch.object(ec, "detect_backend", return_value=("cuda", "test")):
            r.warn_wsl_amd()
        self.assertEqual(events, [])

    def test_setup_warns_in_the_hip_branch_and_does_not_fail_on_its_own(self):
        text = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
        i = text.index("wsl_amd_note(\"hip\", wsl=is_wsl())")
        self.assertIn("warn(wsl_note)", text[i:i + 200])
        self.assertLess(i, text.index("no AMD GPU Strata can use"))


if __name__ == "__main__":
    unittest.main()
