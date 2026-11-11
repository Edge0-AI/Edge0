"""Repeat the mixed 30-minute soak after startup, retaining guard stops."""
import argparse
import json
import socket
import subprocess
import sys
import time

from strata_bench_continue import Continuation, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("iq2_xs", "iq3_s"), required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", b.PORT)) == 0:
            raise RuntimeError("Benchmark service is occupied; it was preserved")
    runner = Continuation(["iq3_xxs", "iq3_s"])
    (b.OUT / f"soak-retry-{args.model}.pid").write_text(str(b.os.getpid()))
    supervisor = subprocess.Popen([sys.executable, str(b.SCRIPT), "--watchdog", str(b.os.getpid()),
        str(runner.clock["measurement_stop_epoch"]+300)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=b.FLAGS)
    began, count, outcome, reason = None, 0, "not_started", None
    try:
        runner.budget("soak_retry_prepare", min(420, max(0, runner.hard-time.monotonic())))
        runner.init_tokenizer()
        runner.inventory()
        if runner.hard-time.monotonic() < 1800+60:
            raise b.CaseError("insufficient reserve for a complete 30-minute soak")
        runner.start(args.model)
        runner.budget("soak_retry", 1800)
        began = time.monotonic()
        outcome = "running"
        while runner.left() > 5:
            if count % 4 == 0:
                runner.perform(f"soak-retry-check-{count}",
                    "Calculate 17 multiplied by 19. Reply only with the integer result.",
                    group="soak_retry", max_tokens=16, expected="323")
            else:
                target = (512, 32768, 128000)[(count-1) % 3]
                runner.perform(f"soak-retry-{count}", runner.make(target, rep=1000+count), target,
                    group="soak_retry", max_tokens=512)
            count += 1
        time.sleep(min(5, runner.left()))
        outcome = "duration_completed"
    except b.CaseError as exc:
        reason = str(exc)
        outcome = "duration_completed" if began and reason == "deadline" and time.monotonic()-began >= 1795 else "stopped"
    except Exception as exc:
        outcome, reason = "unexpected_error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {"model": args.model, "outcome": outcome, "reason": reason,
            "observed_request_window_seconds": time.monotonic()-began if began else 0,
            "completed_requests": count, "weights_mode": runner.weights_mode,
            "engine_environment": getattr(runner, "engine_environment", {}), "target_seconds": 1800,
            "startup_included_in_duration": False}
        b.write(b.OUT / f"soak-retry-{args.model}.json", result)
        runner.event("soak_retry_finish", **result)
        if runner.active_worker and runner.active_worker.poll() is None:
            b.kill_tree(runner.active_worker)
        for job in runner.child_jobs:
            b.kill_tree(job)
        runner.stop()
        runner.monitor.done.set()
        if supervisor.poll() is None:
            supervisor.terminate()


if __name__ == "__main__":
    main()
