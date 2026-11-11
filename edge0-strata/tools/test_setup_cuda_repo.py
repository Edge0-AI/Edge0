"""The CUDA Toolkit apt install: which distros it covers and which 13.x package it picks
(Debian 13's repo has no cuda-toolkit-13-0). No GPU, downloads or installs are used.

    python -m unittest tools.test_setup_cuda_repo
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class AptRepo(unittest.TestCase):
    def test_known_and_unknown_distros(self):
        for osr, want in ({"ID": "ubuntu", "VERSION_ID": '"22.04"'}, "ubuntu2204"), \
                          ({"ID": "ubuntu", "VERSION_ID": '"24.04"'}, "ubuntu2404"), \
                          ({"ID": "debian", "VERSION_ID": '"12"'}, "debian12"), \
                          ({"ID": "debian", "VERSION_ID": '"13"'}, "debian13"), \
                          ({"ID": "debian", "VERSION_ID": '"11"'}, None), \
                          ({"ID": "fedora", "VERSION_ID": '"42"'}, None), \
                          ({}, None):
            with self.subTest(osr=osr):
                self.assertEqual(setup.cuda_apt_repo(osr), want)

    def test_debian_13_falls_back_to_the_oldest_13_x_it_has(self):
        # the repo has no 13-0: apt fails on it, setup must try 13-1 and stop there (not install 13-2+)
        installed = []

        def fake_run(cmd, **kw):
            installed.append(cmd[-1])
            return types.SimpleNamespace(returncode=100 if cmd[-1] == "cuda-toolkit-13-0" else 0)

        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_nvcc", side_effect=[(None, (0, 0)), ("/usr/local/cuda-13.1/bin/nvcc", (13, 1))]), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                mock.patch("builtins.open", mock.mock_open(read_data='ID=debian\nVERSION_ID="13"\n')), \
                mock.patch.object(setup, "download") as download, \
                mock.patch.object(setup, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            nvcc, _ = setup.install_build_tools({"arch": "86", "archs": [86]}, True)
        self.assertEqual(nvcc, "/usr/local/cuda-13.1/bin/nvcc")
        self.assertIn("repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb", download.call_args[0][0])
        self.assertEqual(installed[-2:], ["cuda-toolkit-13-0", "cuda-toolkit-13-1"])

    def test_ubuntu_still_installs_exactly_13_0(self):
        installed = []

        def fake_run(cmd, **kw):
            if cmd[0:2] == ["sudo", "apt-get"] and cmd[2] == "install":
                installed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0)

        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_nvcc", side_effect=[(None, (0, 0)), ("/usr/local/cuda-13.0/bin/nvcc", (13, 0))]), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                mock.patch("builtins.open", mock.mock_open(read_data='ID=ubuntu\nVERSION_ID="24.04"\n')), \
                mock.patch.object(setup, "download"), \
                mock.patch.object(setup, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            setup.install_build_tools({"arch": "86", "archs": [86]}, True)
        self.assertEqual(installed, ["cuda-toolkit-13-0"])

    def test_debian_without_any_13_x_stops_with_guidance(self):
        out = io.StringIO()
        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, (0, 0))), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                mock.patch("builtins.open", mock.mock_open(read_data='ID=debian\nVERSION_ID="12"\n')), \
                mock.patch.object(setup, "download"), \
                mock.patch.object(setup, "run", return_value=types.SimpleNamespace(returncode=100)), \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                setup.install_build_tools({"arch": "86", "archs": [86]}, True)
        self.assertIn("no cuda-toolkit-13-x package", out.getvalue())

    def test_ubuntu_install_failure_still_stops_setup(self):
        def fake_run(cmd, check=True, **kw):
            failed = cmd[-1] == "cuda-toolkit-13-0"
            if failed and check:
                raise SystemExit(1)                  # setup.run's own fail() on a checked command
            return types.SimpleNamespace(returncode=100 if failed else 0)

        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, (0, 0))), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                mock.patch("builtins.open", mock.mock_open(read_data='ID=ubuntu\nVERSION_ID="22.04"\n')), \
                mock.patch.object(setup, "download"), \
                mock.patch.object(setup, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                setup.install_build_tools({"arch": "86", "archs": [86]}, True)

    def test_unknown_distro_stops_with_guidance(self):
        out = io.StringIO()
        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, (0, 0))), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n), \
                mock.patch("builtins.open", mock.mock_open(read_data='ID=fedora\nVERSION_ID="42"\n')), \
                mock.patch.object(setup, "download", side_effect=AssertionError("download started")), \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                setup.install_build_tools({"arch": "86", "archs": [86]}, True)
        self.assertIn("Debian 12 / 13", out.getvalue())


if __name__ == "__main__":
    unittest.main()
