"""Owner-only files: 0600/0700 on POSIX, a one-entry protected ACL for the current user on NTFS."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_control import private_files as pf  # noqa: E402


def resolved(sddl: str) -> str:
    """``sddl`` with every trustee alias (``LA`` is the built-in Administrator) written as its string SID."""
    return re.sub(r";;;([^)]*)\)", lambda match: f";;;{pf._string_sid(match.group(1))})", sddl)


class PrivateFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_file_becomes_private_and_stays_readable_by_us(self):
        path = self.root / "key.json"
        path.write_text('{"apiKey": "x"}')
        pf.restrict_to_current_user(path)
        self.assertTrue(pf.is_private(path))
        self.assertEqual(path.read_text(), '{"apiKey": "x"}')
        path.write_text("{}")  # the owner can still replace it

    def test_a_directory_becomes_private_and_its_children_inherit(self):
        folder = self.root / "app-data"
        folder.mkdir()
        pf.restrict_to_current_user(folder)
        self.assertTrue(pf.is_private(folder))
        child = folder / "child.json"
        child.write_text("{}")
        if pf.WINDOWS:
            # the child inherits the directory's single entry rather than the profile's
            sddl = pf.dacl_sddl(child)
            self.assertIn(pf.current_user_sid(), resolved(sddl))
            self.assertNotIn("S-1-1-0", sddl)
            self.assertNotIn(";;;SY)", sddl)
            self.assertNotIn(";;;BA)", sddl)
        else:
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)

    @unittest.skipIf(pf.WINDOWS, "POSIX mode bits")
    def test_posix_modes_are_exact(self):
        path = self.root / "key.json"
        path.write_text("{}")
        os.chmod(path, 0o666)
        pf.restrict_to_current_user(path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        os.chmod(path, 0o644)
        self.assertFalse(pf.is_private(path))

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_the_effective_acl_is_one_protected_entry_for_the_current_user(self):
        path = self.root / "key.json"
        path.write_text("{}")
        self.assertFalse(pf.is_private(path), "a fresh file inherits SYSTEM/Administrators entries")
        pf.restrict_to_current_user(path)
        sddl = pf.dacl_sddl(path)
        self.assertEqual(resolved(sddl), f"D:PAI(A;;FA;;;{pf.current_user_sid()})")
        # nobody else: no Everyone, Users, Authenticated Users, SYSTEM or Administrators
        for sid in ("WD", "BU", "AU", "SY", "BA", "S-1-1-0", "S-1-5-32-545"):
            self.assertNotIn(f";;;{sid})", sddl)

    def test_a_file_made_in_the_profile_is_unshared_but_not_strictly_private(self):
        path = self.root / "inherited.json"
        path.write_text("{}")
        if not pf.WINDOWS:
            # POSIX files have mode bits rather than an inherited profile ACL.
            # Specify the private mode instead of depending on the process umask.
            path.chmod(0o600)
        self.assertTrue(pf.is_unshared(path))
        if pf.WINDOWS:
            self.assertFalse(pf.is_private(path))   # SYSTEM and Administrators still inherit
        else:
            self.assertTrue(pf.is_private(path))
            path.chmod(0o644)
            self.assertFalse(pf.is_unshared(path))
            self.assertFalse(pf.is_private(path))

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_everyone_is_noticed_by_the_relaxed_check_too(self):
        import subprocess
        path = self.root / "shared.json"
        path.write_text("{}")
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"], check=True, capture_output=True)
        self.assertFalse(pf.is_unshared(path))

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_widening_the_acl_is_noticed(self):
        path = self.root / "key.json"
        path.write_text("{}")
        pf.restrict_to_current_user(path)
        import subprocess
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"], check=True, capture_output=True)
        self.assertFalse(pf.is_private(path))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
class ElevatedRunLockoutTests(unittest.TestCase):
    """A file or folder an elevated run made names only SYSTEM and Administrators: the user's normal session cannot open it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def lock_out(path):
        """Replace the rule with SYSTEM only: this session may be elevated itself, so Administrators is left out to make the lock-out real."""
        import ctypes
        from ctypes import wintypes
        descriptor = ctypes.c_void_p()
        assert pf._advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None)
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        pf._advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted))
        code = pf._advapi.SetNamedSecurityInfoW(str(path), 1, 0x4 | 0x80000000, None, None, dacl, None)
        assert code == 0, code

    def test_a_locked_file_is_read_again_after_the_repair(self):
        path = self.root / "profile.json"
        path.write_text('{"name": "x"}')
        self.lock_out(path)
        with self.assertRaises(PermissionError):
            path.read_bytes()
        self.assertEqual(pf.read_bytes_repairing(path), b'{"name": "x"}')
        self.assertIn(pf.current_user_sid(), resolved(pf.dacl_sddl(path)))
        self.assertEqual(path.read_bytes(), b'{"name": "x"}')

    def test_a_locked_folder_and_what_is_inside_are_repaired_as_a_tree(self):
        folder = self.root / "profiles"
        folder.mkdir()
        (folder / "a.json").write_text("{}")
        (folder / "sub").mkdir()
        (folder / "sub" / "b.json").write_text("{}")
        for item in (folder / "sub" / "b.json", folder / "sub", folder / "a.json", folder):
            self.lock_out(item)
        with self.assertRaises(PermissionError):
            (folder / "a.json").read_bytes()
        os.environ["STRATA_HOME"] = str(self.root)
        self.addCleanup(os.environ.pop, "STRATA_HOME", None)
        report = pf.repair_known_roots()
        self.assertGreaterEqual(report["repaired"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual((folder / "a.json").read_bytes(), b"{}")
        self.assertEqual((folder / "sub" / "b.json").read_bytes(), b"{}")
        self.assertIn(pf.current_user_sid(), resolved(pf.dacl_sddl(folder / "sub" / "b.json")))

    def test_a_folder_made_with_private_mkdir_names_the_user_explicitly(self):
        made = pf.private_mkdir(self.root / "x" / "y", parents=True)
        self.assertIn(pf.current_user_sid(), resolved(pf.dacl_sddl(made)))
        self.assertIn(pf.current_user_sid(), resolved(pf.dacl_sddl(made.parent)))
        (made / "file.json").write_text("{}")
        self.assertIn(pf.current_user_sid(), resolved(pf.dacl_sddl(made / "file.json")))
        self.assertEqual((made / "file.json").read_text(), "{}")

    def test_the_process_defaults_can_be_set_for_the_user(self):
        self.assertTrue(pf.adopt_user_defaults())
        self.assertTrue(pf.adopt_user_defaults())       # again: nothing breaks
        path = self.root / "new.json"
        path.write_text("{}")
        self.assertEqual(path.read_text(), "{}")

    def test_default_dacl_allows_reopening_child_process_for_qos(self):
        import subprocess
        import sys
        # Isolate token changes and keep this test independent of the test runner's defaults.
        script = '''
import ctypes
from ctypes import wintypes
import subprocess
import sys
from strata_control import private_files as pf
from serve import winjob
assert pf.adopt_user_defaults()
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
k32.OpenProcess.restype = wintypes.HANDLE
k32.CloseHandle.argtypes = (wintypes.HANDLE,)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
try:
    handle = k32.OpenProcess(0x1000 | 0x0200, False, child.pid)
    assert handle, "OpenProcess failed: %s" % ctypes.get_last_error()
    try:
        assert winjob.no_throttle(handle), "Setting child QoS failed"
    finally:
        k32.CloseHandle(handle)
finally:
    child.terminate()
    child.wait()
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_only_the_users_own_folders_are_ever_repaired(self):
        self.assertFalse(pf.repair_access(os.environ.get("SystemRoot", "C:\Windows") + "\System32\drivers"))

    def test_a_profile_that_was_locked_loads(self):
        from strata_control import profiles
        os.environ["STRATA_PROFILE_DIR"] = str(self.root / "profiles")
        self.addCleanup(os.environ.pop, "STRATA_PROFILE_DIR", None)
        directory = self.root / "profiles"
        directory.mkdir()
        run_config = self.root / "strata-x.json"
        run_config.write_text("{}")
        (directory / "app-1.json").write_text(json.dumps({"schema": 1, "name": "app-1", "source_config": str(run_config)}))
        self.lock_out(directory / "app-1.json")
        try:
            profiles.load("app-1")
        except profiles.ProfileError as exc:
            self.assertNotIn("PermissionError", str(exc))     # any other complaint is the sample's shape, not access
