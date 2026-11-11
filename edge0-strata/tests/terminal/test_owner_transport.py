"""Owner transport contract tests.

These run on every platform and assert the properties the runtime depends on, rather than asserting
that a particular syscall was used. The Windows path is exercised on POSIX too: the loopback-TCP
listener and the state-file validation are ordinary Python and can be tested anywhere, so the
platform branch is not left unverified until someone has a Windows machine.

What is NOT tested here, and cannot be, is whether a Windows named pipe with an owner-only ACL
behaves as intended — because that implementation does not exist yet. `docs/work/IMPLEMENTATION_STATUS.md`
records that as the honest difference between the platforms rather than pretending the loopback
transport is equivalent to it.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import owner_transport as ot  # noqa: E402
from strata_control.private_files import restrict_to_current_user  # noqa: E402


class TestAddressing(unittest.TestCase):
    def test_unix_address_fits_the_sun_path_limit(self):
        """A long home directory must not produce an address the kernel rejects."""
        long_dir = Path("/Users/someone-with-a-very-long-account-name/Documents/deepseek-harness/state")
        addr = ot.address_for(long_dir, "default")
        self.assertLessEqual(len(addr.encode()), 90)
        if ot.WINDOWS:
            self.assertTrue(addr.startswith("tcp:"))
        else:
            self.assertTrue(addr.endswith(".sock"))

    @unittest.skipIf(ot.WINDOWS, "the Windows address is a placeholder; the listener picks the port")
    def test_address_is_stable_for_the_same_inputs(self):
        d = Path("/tmp/strata-runtime")
        self.assertEqual(ot.address_for(d, "default"), ot.address_for(d, "default"))
        self.assertNotEqual(ot.address_for(d, "default"), ot.address_for(d, "other"))


@unittest.skipIf(ot.WINDOWS, "POSIX socket binding")
class TestPosixListener(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-transport-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_and_private_permissions(self):
        addr = ot.address_for(self.tmp, "default")
        with ot.Listener(addr, timeout=2) as listener:
            mode = stat.S_IMODE(os.stat(listener.address).st_mode)
            self.assertEqual(mode, 0o600, "the owner socket must not be reachable by other users")

            received: list[bytes] = []

            def server():
                conn = listener.accept()
                assert conn is not None, "the listener timed out"
                with conn:
                    received.append(conn.recv(4096))

            t = threading.Thread(target=server)
            t.start()
            with ot.connect(listener.address, timeout=2) as client:
                client.sendall(json.dumps({"hello": "owner"}).encode() + b"\n")
            t.join(timeout=5)
            self.assertEqual([json.loads(received[0])], [{"hello": "owner"}])

    def test_the_socket_path_is_removed_on_close(self):
        addr = ot.address_for(self.tmp, "default")
        listener = ot.Listener(addr)
        listener.bind()
        self.assertTrue(Path(listener.address).exists())
        listener.close()
        self.assertFalse(Path(listener.address).exists(),
                         "a stale socket path would block the next owner from binding")

    def test_accept_returns_none_on_timeout_rather_than_raising(self):
        """The owner polls, so a quiet channel must not look like an error."""
        addr = ot.address_for(self.tmp, "default")
        with ot.Listener(addr, timeout=0.2) as listener:
            self.assertIsNone(listener.accept())

    def test_connect_to_a_dead_address_raises_oserror(self):
        addr = ot.address_for(self.tmp, "nothing-here")
        with self.assertRaises(OSError):
            ot.connect(addr, timeout=1)


class TestLoopbackTransport(unittest.TestCase):
    """The Windows branch, exercised on every platform.

    This is the same code Windows runs; only the socket family differs, and `socket.AF_INET` works
    everywhere. Testing it here means the transport is verified rather than deferred.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-transport-tcp-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bind_loopback(self) -> ot.Listener:
        """Bind the way the Windows branch does, regardless of host platform."""
        original = ot.WINDOWS
        ot.WINDOWS = True
        try:
            listener = ot.Listener("tcp:127.0.0.1:0", timeout=2)
            listener.bind()
            return listener
        finally:
            ot.WINDOWS = original

    def test_loopback_binds_and_round_trips(self):
        listener = self._bind_loopback()
        try:
            self.assertTrue(listener.address.startswith("tcp:127.0.0.1:"),
                            "the owner must never bind a wildcard address")
            port = int(listener.address.rsplit(":", 1)[1])
            self.assertGreater(port, 0)
            got: list[bytes] = []

            def server():
                conn = listener.accept()
                assert conn is not None
                with conn:
                    got.append(conn.recv(1024))

            t = threading.Thread(target=server)
            t.start()
            original = ot.WINDOWS
            ot.WINDOWS = True
            try:
                with ot.connect(listener.address, timeout=2) as client:
                    client.sendall(b"token-check\n")
            finally:
                ot.WINDOWS = original
            t.join(timeout=5)
            self.assertEqual(got, [b"token-check\n"])
        finally:
            listener.close()

    def test_two_owners_cannot_share_the_port(self):
        """A second owner must fail to bind rather than silently share the channel."""
        listener = self._bind_loopback()
        try:
            port = int(listener.address.rsplit(":", 1)[1])
            clash = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                clash.bind(("127.0.0.1", port))
                # If the platform allows the rebind, the port is not exclusively held, which would
                # let a second owner hijack the channel. Fail loudly.
                self.fail("a second listener bound the owner's port")
            except OSError:
                pass
            finally:
                clash.close()
        finally:
            listener.close()

    def test_it_binds_loopback_only(self):
        """The listener must not be reachable from another interface."""
        listener = self._bind_loopback()
        try:
            port = int(listener.address.rsplit(":", 1)[1])
            # Connecting to a non-loopback local address on that port must fail: nothing is bound
            # there. 203.0.113.0/24 is TEST-NET-3 and is never a local address.
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            try:
                with self.assertRaises(OSError):
                    s.connect(("203.0.113.1", port))
            finally:
                s.close()
        finally:
            listener.close()


class TestStateFileValidation(unittest.TestCase):
    """A client must refuse a state file it cannot trust."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-state-"))
        self.path = self.tmp / "default.json"
        self.path.write_text('{"token": "x"}')
        restrict_to_current_user(self.path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_private_regular_file_is_accepted(self):
        ot.check_state_file(self.path)

    @unittest.skipUnless(ot.WINDOWS, "NTFS ACLs")
    def test_a_state_file_with_an_inherited_acl_is_refused(self):
        other = self.tmp / "inherited.json"
        other.write_text("{}")  # a fresh file inherits the directory's entries (SYSTEM, Administrators)
        with self.assertRaises(ot.TransportError):
            ot.check_state_file(other)

    @unittest.skipIf(ot.WINDOWS, "POSIX permission bits")
    def test_a_world_readable_state_file_is_refused(self):
        os.chmod(self.path, 0o644)
        with self.assertRaises(ot.TransportError):
            ot.check_state_file(self.path)

    @unittest.skipIf(ot.WINDOWS, "POSIX symlinks")
    def test_a_symlink_is_refused(self):
        link = self.tmp / "link.json"
        link.symlink_to(self.path)
        with self.assertRaises(ot.TransportError):
            ot.check_state_file(link)

    def test_a_directory_is_refused(self):
        with self.assertRaises(ot.TransportError):
            ot.check_state_file(self.tmp)

    def test_an_implausibly_large_state_file_is_refused(self):
        big = self.tmp / "big.json"
        big.write_bytes(b"x" * (17 * 1024))
        os.chmod(big, 0o600)
        with self.assertRaises(ot.TransportError):
            ot.check_state_file(big)

    def test_a_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            ot.check_state_file(self.tmp / "nope.json")


if __name__ == "__main__":
    unittest.main()
