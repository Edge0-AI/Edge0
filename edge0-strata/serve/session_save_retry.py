"""Bounded SAVE admission retries after opt-in engine cache reclamation.

The caller holds the service FIFO. Only an unpublished, typed RAM preflight
refusal is retried, on the same live engine process and with its floor intact.
No model unload, working-set trim or unrelated process is involved.
"""
from __future__ import annotations

import re
import time

SETTLE_SECONDS = 30.0
MAX_RETRIES = 3
POLL_SECONDS = 0.5


def required_mib(error) -> int | None:
    match = re.search(r'^not enough RAM to save the session '
                      r'\((\d+) MiB plus a floor of (\d+) MiB needed,', str(error))
    # Each diagnostic term rounds bytes down; two MiB preserve admission headroom.
    return sum(map(int, match.groups())) + 2 if match else None


def available_mib() -> int | None:
    try:
        import psutil
        available = psutil.virtual_memory().available
        if isinstance(available, bool) or not isinstance(available, int) or available < 0:
            return None
        return available // 2**20
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def retryable(action, error) -> bool:
    return (action == 'save' and error.kind == 'memory' and not error.published
            and required_mib(error) is not None)


def session_file(service, action, path, refused_type):
    """Called only when --session-save-reclaim is enabled; never changes RESTORE."""
    engine = service.engine
    process = getattr(engine, 'proc', None)
    try:
        return engine.session_file(action, path)
    except refused_type as error:
        if not retryable(action, error) or process is None:
            raise
        deadline = time.monotonic() + SETTLE_SECONDS
        last_error = error

        def same_process():
            try:
                return (service.engine is engine and engine.proc is process
                        and not getattr(engine, 'ended', False) and process.poll() is None)
            except OSError:
                return False

        print('[strata] SAVE waiting for RAM after cache reclaim: '
              f'need={required_mib(error)} MiB, limit={SETTLE_SECONDS:g}s; floor unchanged', flush=True)
        for attempt in range(MAX_RETRIES):
            while True:
                if not same_process() or time.monotonic() >= deadline:
                    raise last_error
                available = available_mib()
                needed = required_mib(last_error)
                if available is None or needed is None:
                    raise last_error
                if available >= needed:
                    break
                time.sleep(max(0.0, min(POLL_SECONDS, deadline - time.monotonic())))
            # No replacement, restart, or blind retry when telemetry is unknown.
            if not same_process():
                raise last_error
            try:
                return engine.session_file(action, path)
            except refused_type as failure:
                if not retryable(action, failure):
                    raise
                last_error = failure
                if attempt + 1 == MAX_RETRIES or time.monotonic() >= deadline:
                    raise
                time.sleep(max(0.0, min(POLL_SECONDS, deadline - time.monotonic())))
        raise last_error
