"""The Terminal finds a cloudflared that the system installed (PATH, /usr/bin, ~/.local/bin, Homebrew, Program Files), checks its
version, and does not start a second connector next to one that is already running."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import access


def fake_cloudflared(directory: Path, version: str = "2026.5.0", name: str = "cloudflared") -> Path:
    path = directory / name
    path.write_text(f'#!/bin/sh\necho "cloudflared version {version} (built 2026-05-12)"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@unittest.skipIf(os.name == "nt", "the fixture is a shell script")
class FindOnPath(unittest.TestCase):
    def setUp(self):
        access._cf_seen.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def env(self, **extra):
        return mock.patch.dict(os.environ, {"PATH": str(self.dir / "bin"), "HOME": str(self.dir), **extra}, clear=False)

    def test_a_user_local_copy_on_path_is_found_with_its_version(self):
        (self.dir / "bin").mkdir()
        exe = fake_cloudflared(self.dir / "bin")
        with self.env():
            info = access.cloudflared_info()
            self.assertEqual(info, {"path": str(exe), "version": "2026.5.0"})
            self.assertEqual(access.find_cloudflared(), str(exe))

    def test_the_local_bin_folder_is_searched_when_path_misses_it(self):
        (self.dir / ".local" / "bin").mkdir(parents=True)
        exe = fake_cloudflared(self.dir / ".local" / "bin")
        with self.env(PATH=str(self.dir / "empty")), mock.patch.object(Path, "home", return_value=self.dir):
            self.assertEqual(access.find_cloudflared(), str(exe))

    def test_a_too_old_copy_is_passed_over(self):
        (self.dir / "bin").mkdir()
        fake_cloudflared(self.dir / "bin", version="2021.3.0")
        with self.env(), mock.patch.object(Path, "home", return_value=self.dir / "nohome"):
            places = [p for p in access._cloudflared_places() if str(self.dir) in p]
            with mock.patch.object(access, "_cloudflared_places", return_value=places):
                self.assertIsNone(access.find_cloudflared())

    def test_a_file_that_does_not_run_is_passed_over(self):
        (self.dir / "bin").mkdir()
        bad = self.dir / "bin" / "cloudflared"
        bad.write_text("#!/bin/sh\nexit 3\n")
        bad.chmod(0o755)
        with self.env(), mock.patch.object(access, "_cloudflared_places", return_value=[str(bad)]):
            self.assertIsNone(access.find_cloudflared())

    def test_strata_cloudflared_picks_exactly_that_file(self):
        exe = fake_cloudflared(self.dir, version="2025.1.2", name="mine")
        with self.env(STRATA_CLOUDFLARED=str(exe)):
            self.assertEqual(access._cloudflared_places(), [str(exe)])
            self.assertEqual(access.cloudflared_info()["version"], "2025.1.2")


class Places(unittest.TestCase):
    def test_the_usual_places_are_listed(self):
        with mock.patch.dict(os.environ, {"PATH": "/opt/x/bin"}, clear=False), mock.patch.object(access, "WINDOWS", False), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_CLOUDFLARED", None)
            places = [p.replace("\\", "/") for p in access._cloudflared_places()]
        for want in ("/opt/x/bin/cloudflared", "/usr/bin/cloudflared", "/usr/local/bin/cloudflared", "/opt/homebrew/bin/cloudflared"):
            self.assertIn(want, places)
        self.assertTrue(any(p.endswith(".local/bin/cloudflared") for p in places))
        self.assertEqual(len(places), len(set(places)))

    def test_windows_places_include_program_files(self):
        env = {"PATH": "", "ProgramFiles": r"C:\Program Files", "ProgramFiles(x86)": r"C:\Program Files (x86)"}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(access, "WINDOWS", True):
            os.environ.pop("STRATA_CLOUDFLARED", None)
            places = access._cloudflared_places()
        self.assertTrue(any("Program Files" in p and p.endswith("cloudflared.exe") for p in places))

    def test_version_rule(self):
        self.assertTrue(access._cf_new_enough("2026.5.0"))
        self.assertTrue(access._cf_new_enough("2022.1.0"))
        self.assertFalse(access._cf_new_enough("2021.12.2"))
        self.assertTrue(access._cf_new_enough(""))


class OtherTunnels(unittest.TestCase):
    PS = ("  4242 cloudflared /usr/bin/cloudflared --no-autoupdate tunnel run --token SECRETSECRETSECRETSECRET\n"
          "    77 cloudflared cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080\n"
          "    90 bash bash -c cloudflared\n   100 sshd sshd\n")

    def test_every_cloudflared_is_listed_with_its_pid_and_our_own_pid_is_left_out(self):
        with mock.patch.object(access, "WINDOWS", False), mock.patch.object(access.subprocess, "run", return_value=mock.Mock(stdout=self.PS)):
            self.assertEqual([p for p, _ in access.cloudflared_procs()], [4242, 77])
            self.assertEqual([p for p, _ in access.cloudflared_procs(exclude=(4242,))], [77])

    def test_only_a_tunnel_pointing_at_this_servers_port_is_its_own(self):
        procs = [(4242, "cloudflared tunnel run --token X"), (77, "cloudflared tunnel --url http://127.0.0.1:8080"), (5, "cloudflared tunnel --url http://127.0.0.1:18080")]
        self.assertEqual(access.own_tunnel_pids(procs, 8080), [77])
        self.assertEqual(access.own_tunnel_pids(procs, 9999), [])
        self.assertEqual(access.other_tunnel_note([4242]), "cloudflared is already running for another tunnel (PID 4242).")
        self.assertEqual(access.other_tunnel_note([1, 2]), "cloudflared is already running for other tunnels (PIDs 1, 2).")

    def test_the_servers_own_tunnel_already_running_is_not_doubled(self):
        with mock.patch.object(access, "find_cloudflared", return_value="/usr/bin/cloudflared"), mock.patch.object(access, "tunnel_info", return_value=None), \
                mock.patch.object(access, "cloudflared_procs", return_value=[(31, "cloudflared tunnel --url http://127.0.0.1:8080")]), \
                mock.patch.object(access.subprocess, "Popen") as popen:
            url, message = access.start_tunnel("demo", 8080)
        self.assertIsNone(url)
        popen.assert_not_called()
        self.assertIn("already running (PID 31)", message)
        self.assertIn("did not start a second one", message)

    def test_another_tunnel_running_does_not_stop_a_quick_tunnel(self):
        proc = mock.Mock(pid=55)
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(access, "find_cloudflared", return_value="/usr/bin/cloudflared"), \
                mock.patch.object(access, "tunnel_info", return_value=None), mock.patch.object(access, "stop_tunnel"), \
                mock.patch.object(access, "_state_dir", return_value=Path(tmp)), \
                mock.patch.object(access, "cloudflared_procs", return_value=[(4242, "cloudflared tunnel run --token X")]), \
                mock.patch.object(access.subprocess, "Popen", side_effect=lambda *a, **k: (k["stdout"].write(b"https://abc-def.trycloudflare.com"), k["stdout"].flush(), proc)[2]) as popen, \
                mock.patch("strata_control.procs.start_token", return_value="t"), mock.patch("strata_control.procs.command_line", return_value=["x"]):
            url, message = access.start_tunnel("demo", 8080, wait=2)
        popen.assert_called_once()
        self.assertEqual(url, "https://abc-def.trycloudflare.com")
        self.assertIn("cloudflared is already running for another tunnel (PID 4242).", message)

    def test_a_missing_ps_is_never_a_block(self):
        with mock.patch.object(access, "WINDOWS", False), mock.patch.object(access.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(access.cloudflared_procs(), [])


if __name__ == "__main__":
    unittest.main()
