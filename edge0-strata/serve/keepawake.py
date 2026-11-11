"""Opt-in "prevent_sleep" (#1727): while at least one request is running or queued, the PC is kept awake.

Windows only.  SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) keeps the system from idling into sleep
(not the display: no ES_DISPLAY_REQUIRED); ES_CONTINUOUS alone puts the normal behaviour back.  The state belongs
to the thread that set it, so one daemon thread sets and clears it; it is cleared when idle and on exit.
Elsewhere this does nothing.
"""
from __future__ import annotations

import atexit
import sys
import threading

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_call():
    import ctypes
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return fn


class KeepAwake:
    def __init__(self, call=None):
        self._call = call
        self._count = 0
        self._cond = threading.Condition()
        self._closed = False
        self._thread = None

    @classmethod
    def create(cls, enabled: bool, say=print):
        """None when off.  On a system without SetThreadExecutionState: one line, and None."""
        if not enabled:
            return None
        if not sys.platform.startswith("win"):
            say('[strata] "prevent_sleep" is for Windows only: ignored here', flush=True)
            return None
        try:
            ka = cls(_windows_call())
        except (ImportError, AttributeError, OSError) as e:
            say(f'[strata] "prevent_sleep" could not be set up: {e}', flush=True)
            return None
        atexit.register(ka.close)
        say('[strata] prevent_sleep: Windows stays awake while a request is running or queued', flush=True)
        return ka

    def _loop(self):
        held = False
        while True:
            with self._cond:
                while not self._closed and (self._count > 0) == held:
                    self._cond.wait()
                if self._closed:
                    break
                want = self._count > 0
            try:
                self._call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if want else ES_CONTINUOUS)
                held = want
            except Exception:         # noqa: BLE001  never fail a request over this
                held = want
        try:
            self._call(ES_CONTINUOUS)
        except Exception:             # noqa: BLE001
            pass

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="keep-awake")
            self._thread.start()

    def acquire(self):
        with self._cond:
            if self._closed:
                return
            self._count += 1
            self._start()
            self._cond.notify_all()

    def release(self):
        with self._cond:
            self._count = max(0, self._count - 1)
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)
