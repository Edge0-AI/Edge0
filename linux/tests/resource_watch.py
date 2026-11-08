"""Watch a test process's RSS and stop its process group on host RAM pressure."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from server_test_utils import require, snapshot


def memory_fields(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if len(parts) == 2 and parts[1] == "kB":
            fields[key] = int(parts[0]) * 1024
    return fields


def host_snapshot():
    return dict(snapshot(), memory_bytes=memory_fields("/proc/meminfo"))


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class ResourceWatch:
    def __init__(self, process, path, minimum=4 << 30, interval=0.05):
        self.process, self.path = process, Path(path)
        self.minimum, self.interval = minimum, interval
        self.finished = threading.Event()
        self.error = None
        self.summary = dict(peak_rss_bytes=0, peak_hwm_bytes=0, min_available_bytes=None, samples=0)
        self.worker = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        memory = memory_fields("/proc/meminfo")
        try:
            status = memory_fields(f"/proc/{self.process.pid}/status")
        except FileNotFoundError:
            status = {}
        return dict(time=time.monotonic(), rss_bytes=status.get("VmRSS", 0),
                    hwm_bytes=status.get("VmHWM", 0), available_bytes=memory["MemAvailable"])

    def record(self, row):
        summary = self.summary
        summary["samples"] += 1
        summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], row["rss_bytes"])
        summary["peak_hwm_bytes"] = max(summary["peak_hwm_bytes"], row["hwm_bytes"])
        previous = summary["min_available_bytes"]
        summary["min_available_bytes"] = min(row["available_bytes"], previous or row["available_bytes"])
        if row["available_bytes"] < self.minimum:
            raise RuntimeError("host available RAM fell below the safety floor")

    def run(self):
        try:
            with self.path.open("w") as output:
                while not self.finished.is_set():
                    row = self.sample()
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    self.record(row)
                    self.finished.wait(self.interval)
        except Exception as error:
            self.error = str(error)
            print(f"ERROR: resource monitor: {error}", flush=True)
            stop_group(self.process)

    def check(self):
        require(self.error is None, f"resource monitor stopped the run: {self.error}")

    def stop(self):
        self.finished.set()
        self.worker.join(timeout=35)
        require(not self.worker.is_alive(), "resource monitor did not stop")


@contextmanager
def watched_process(command, env, log, samples):
    with Path(log).open("w") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        watch = ResourceWatch(process, samples)
        watch.worker.start()
        try:
            yield process, watch
        finally:
            stop_group(process)
            watch.stop()
        watch.check()


def within_budget(summary, budget_mib):
    require(summary["samples"] > 0, "no RSS samples")
    peak = max(summary["peak_rss_bytes"], summary["peak_hwm_bytes"])
    require(peak > 0, "no valid process RSS measurements")
    return peak <= budget_mib * (1 << 20)
