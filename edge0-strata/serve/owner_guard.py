"""Parent-loss guard and graceful-stop channel for a server launched by the local runtime host.

The inherited pipe has exactly one writer, held by the host.  EOF therefore
means the host exited, including after SIGKILL.

POSIX: the server has its own process group so its engine subprocesses receive
the emergency stop as well.  Windows: the host passes the pipe as the server's
standard input (``--owner-fd 0``) and ends the server's whole process tree
through a kill-on-close job object if it dies; the pipe is also how the host
asks the server to drain and exit (it closes its end), because Windows has no
SIGTERM to send.  EOF is turned into a Ctrl+C in the main thread, which is the
path the server already drains and stops cleanly on.
"""

from __future__ import annotations

import os
import signal
import stat
import threading
import time


def _stop_gracefully() -> None:
    import _thread
    _thread.interrupt_main()


def start(fd: int, on_loss=None, grace: float = 60.0) -> None:
    if os.name == "nt":
        if fd < 0 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise ValueError("--owner-fd must be an inherited pipe")

        def watch_windows() -> None:
            try:
                while os.read(fd, 1):
                    pass
            except OSError:
                pass
            (on_loss or _stop_gracefully)()

        threading.Thread(target=watch_windows, name="strata-owner-guard", daemon=True).start()
        return
    if os.name != "posix" or fd < 0 or os.getpgrp() != os.getpid():
        raise ValueError("--owner-fd requires a dedicated POSIX server process group")
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise ValueError("--owner-fd must be an inherited pipe")

    def watch() -> None:
        try:
            while os.read(fd, 1):
                pass
        except OSError:
            pass                       # a broken guard must never authorize another process to manage us
        finally:
            os.close(fd)
        # This is an emergency path.  Signal the whole dedicated group so an engine already spawned during model
        # loading cannot be orphaned.  SIGTERM is the clean stop (the server drains and closes its engine); if the
        # group is still there after `grace` seconds, SIGKILL ends whatever is left.
        pgid = os.getpgrp()
        try:
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                time.sleep(0.5)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    threading.Thread(target=watch, name="strata-owner-guard", daemon=True).start()
