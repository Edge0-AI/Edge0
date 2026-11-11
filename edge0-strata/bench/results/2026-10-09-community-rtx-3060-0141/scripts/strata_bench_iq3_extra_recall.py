"""Use remaining authorized reserve for IQ3 260K recall at two extra positions."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from strata_bench_continue import Continuation, b


def main():
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", b.PORT)) == 0:
            raise RuntimeError("Benchmark service is occupied; it was preserved")
    runner = Continuation(["iq3_xxs", "iq3_s"])
    (b.OUT / "supplement.pid").write_text(str(b.os.getpid()))
    supervisor = subprocess.Popen([sys.executable, str(b.SCRIPT), "--watchdog", str(b.os.getpid()),
        str(runner.clock["measurement_stop_epoch"]+300)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=b.FLAGS)
    try:
        runner.budget("reserve_iq3_recall", max(0, runner.hard-time.monotonic()))
        runner.init_tokenizer()
        runner.inventory()
        for model in ("iq3_xxs", "iq3_s"):
            eligible = [r for r in runner.rows if r["model"] == model and r["group"] == "speed"
                and r.get("target") == 260000 and r.get("ok") and r.get("token_count_matches")
                and r.get("engine_environment", {}).get("STRATA_UNBUFFERED_LOAD") == "1"]
            estimate = max((r["client_elapsed_s"] for r in eligible), default=900)*2 + 120
            if not eligible or runner.left() < estimate:
                runner.event("supplement_skipped", requested_model=model,
                    reason="no successful main 260K trial or insufficient reserve for both positions",
                    estimated_seconds=estimate, remaining_seconds=runner.left())
                continue
            def run():
                runner.start(model, 262144)
                for depth in (10, 90):
                    label = f"recall-260000-d{depth}"
                    if any(r["model"] == model and r["label"] == label and r.get("ok")
                        and r.get("engine_environment", {}).get("STRATA_UNBUFFERED_LOAD") == "1" for r in runner.rows):
                        continue
                    runner.perform(label, runner.make(260000, "code", 9, depth), 260000,
                        group="recall", max_tokens=40, expected=f"amber-quartz-260000-9-{depth}")
            runner.safe_block(run)
            runner.stop()
        runner.event("supplement_finished")
    finally:
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
