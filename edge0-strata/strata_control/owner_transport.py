"""Owner transport: the private channel between a Strata Control owner and its clients.

There is exactly one owner per installation. Clients attach to it over a channel that must be
reachable only by the same user, and the design differs per platform for a concrete reason:

**POSIX (macOS, Linux)** uses an ``AF_UNIX`` stream socket inside the owner's private runtime
directory (mode ``0700``), with the socket itself at ``0600``. Access control is the filesystem's:
another user cannot open the path. This is the implementation the terminal layer already used and it
is preserved unchanged here.

**Windows** cannot create a named pipe from the Python standard library — there is no
``socket.AF_UNIX`` that carries a security descriptor, and ``CreateNamedPipe`` is not exposed. The
transport is therefore a **loopback TCP listener owned by the owner process**, protected by three
things rather than by a pipe ACL:

1. it binds ``127.0.0.1`` only, never ``0.0.0.0``, and it never enables a wildcard or IPv6-any
   listener that would also accept remote traffic;
2. every request carries the owner's 256-bit control token, compared in constant time, which the
   client reads from a state file in the owner's private directory (``0700``, owner-only);
3. a connection that fails the token check is dropped immediately and counts toward a small
   per-owner failure budget, so the port cannot be used as a token oracle.

A local TCP port is reachable by other users on the same machine *if* they can guess it and *if*
they have the token. That is a weaker guarantee than a pipe ACL, and it is recorded as such: on
Windows the security of the owner channel rests on the token, exactly as it does for a Jupyter
server or any other loopback-authenticated service. A native ``CreateNamedPipe`` implementation with
``SDDL`` restricting the pipe to the owner is a scoped improvement, not a prerequisite, and
``docs/work/IMPLEMENTATION_STATUS.md`` tracks it as the honest difference between the platforms.

This module is deliberately small: it provides the two operations the runtime needs (listen,
connect), plus the addressing helpers. It owns no policy, no identity and no lifecycle.
"""

from __future__ import annotations

import errno
import os
import socket
import stat
import sys
from pathlib import Path

#: The largest single message either side will read. Both the state file and the socket protocol
#: assume it, so it lives here with the transport rather than with either caller.
MAX_MESSAGE = 64 * 1024

WINDOWS = sys.platform == "win32"


class TransportError(OSError):
    """The channel could not be established, with the reason the caller should report."""


def address_for(runtime_dir: Path, name: str) -> str:
    """The address a client connects to, stored in the owner's state file.

    On POSIX this is the socket path. The digest keeps it inside the ``AF_UNIX`` length limit
    (``sun_path`` is about 104 bytes on macOS and 108 on Linux), which a long home directory would
    otherwise exceed; when even the private-directory form is too long the address falls back to a
    short ``/tmp`` path that only this user can traverse.
    """
    if WINDOWS:
        # The port is chosen by the listener and written into the state file; the address here is
        # only a placeholder that the caller always overwrites with the real one.
        return f"tcp:127.0.0.1:0"
    import hashlib
    digest = hashlib.sha256(f"{runtime_dir}\0{name}".encode()).hexdigest()[:16]
    candidate = runtime_dir / f"{digest}.sock"
    if len(str(candidate).encode()) <= 90:
        return str(candidate)
    return f"/tmp/strata-{os.getuid()}-{digest}.sock"


class Listener:
    """An accepting endpoint for the owner's private channel.

    Use as a context manager. ``accept()`` returns a connected socket, or ``None`` on timeout so the
    owner can service its other work between polls.
    """

    def __init__(self, address: str, *, timeout: float = 1.0) -> None:
        self.requested = address
        self.address = address
        self._sock: socket.socket | None = None
        self._timeout = timeout
        self._bound = False

    def bind(self) -> str:
        if WINDOWS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Loopback only, and no SO_REUSEADDR: a second owner must fail to bind rather than
            # silently share the port with the first.
            sock.bind(("127.0.0.1", 0))
            self.address = f"tcp:127.0.0.1:{sock.getsockname()[1]}"
        else:
            path = Path(self.requested)
            if path.exists():
                # A leftover socket from a crashed owner. Only unlink when nothing is listening,
                # which is exactly what the caller's stale check established before calling bind.
                path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
            os.chmod(path, 0o600)
        sock.listen(5)
        sock.settimeout(self._timeout)
        self._sock = sock
        self._bound = True
        return self.address

    def accept(self) -> socket.socket | None:
        if self._sock is None:
            raise TransportError("the listener is not bound")
        try:
            conn, _ = self._sock.accept()
        except socket.timeout:
            return None
        except OSError as e:
            if e.errno in (errno.EINTR, errno.EAGAIN):
                return None
            raise
        return conn

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if not WINDOWS and self._bound:
            # Remove the path so a later owner can bind it; ignore a racing removal.
            try:
                Path(self.requested).unlink()
            except OSError:
                pass
        self._bound = False

    def __enter__(self) -> "Listener":
        self.bind()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def connect(address: str, *, timeout: float = 5.0) -> socket.socket:
    """Connect to an owner's private channel."""
    sock = socket.socket(socket.AF_INET if WINDOWS else socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        if WINDOWS:
            _, host, port = address.split(":", 2)
            sock.connect((host, int(port)))
        else:
            sock.connect(address)
    except OSError:
        sock.close()
        raise
    return sock


def check_state_file(path: Path) -> None:
    """Validate that a state file is one this user may trust.

    Refuses anything that is not a regular file, is not owned by this user, or is group/world
    accessible. This is what stops another account from pointing a client at its own listener.
    """
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise TransportError("the runtime state file is a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise TransportError("the runtime state file is not a regular file")
    if st.st_size > 16 * 1024:
        raise TransportError("the runtime state file is implausibly large")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise TransportError("the runtime state file is owned by another user")
    if WINDOWS:
        # NTFS has no mode bits: the promise is a protected ACL naming this user only.
        from strata_control.private_files import is_private
        if not is_private(path):
            raise TransportError("the runtime state file is readable by other users")
    elif st.st_mode & 0o077:
        raise TransportError("the runtime state file is readable by other users")
