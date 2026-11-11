"""Read-only, one-second sysfs/proc snapshots for attributing benchmark drift."""
from pathlib import Path
import threading
import time

from single_gpu import BDF


class ResourceSampler:
    def __init__(self, pid):
        device = Path("/sys/bus/pci/devices") / BDF
        paths = [device / name for name in ("gpu_busy_percent", "mem_busy_percent", "current_link_speed",
                                           "current_link_width", "pp_dpm_sclk", "pp_dpm_mclk")]
        for hwmon in (device / "hwmon").glob("hwmon*"):
            paths.extend(hwmon / name for name in ("temp1_input", "temp2_input", "temp3_input", "freq1_input",
                                                   "freq2_input", "power1_average", "fan1_input"))
        paths += [Path(f"/proc/{pid}/io"), Path(f"/proc/{pid}/schedstat"), Path(f"/proc/{pid}/stat"),
                  Path("/proc/loadavg")]
        paths += [Path("/proc/pressure") / kind for kind in ("cpu", "io", "memory")]
        paths += list(Path("/sys/block").glob("nvme*/stat"))
        self.paths = [p for p in paths if p.exists()]
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            begin = time.perf_counter()
            values = {}
            for path in self.paths:
                try:
                    values[str(path)] = path.read_text().strip()
                except OSError as exc:
                    values[str(path)] = {"error": str(exc)}
            self.samples.append({"monotonic_s": begin, "unix_s": time.time(), "values": values,
                                 "read_ms": (time.perf_counter() - begin) * 1000})
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join()
        return {"scope": "read-only system snapshots, raw sysfs units; shared host; no clock/power changes",
                "interval_s": 1, "samples": self.samples}
