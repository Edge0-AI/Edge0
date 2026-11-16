"""Split a launcher's allowed physical cores between independent Linux instances."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def partition_cpu_sets(count: int, allowed: set[int], reserved: set[int] | None = None,
                       sys_root: Path = Path("/sys")) -> list[list[int]]:
    if count < 1:
        raise ValueError("the instance count must be positive")
    if not allowed:
        raise ValueError("no CPUs are allowed for this launcher")
    if count == 1 and not reserved:
        return [sorted(allowed)]
    cores: dict[tuple[int, int], list[int]] = {}
    occupied: set[tuple[int, int]] = set()
    for cpu in sorted(allowed | (reserved or set())):
        topology = sys_root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read physical-core topology for CPU {cpu}: {error}") from error
        if cpu in allowed:
            cores.setdefault(key, []).append(cpu)
        if reserved and cpu in reserved:
            occupied.add(key)
    available = sorted((cpus for key, cpus in cores.items() if key not in occupied), key=min)
    if len(available) < count:
        raise ValueError(
            f"{count} instances need at least {count} free physical cores, but only {len(available)} "
            "are available; restart the managed instances together or expand the launcher's CPU set"
        )
    size, extra = divmod(len(available), count)
    result, start = [], 0
    for index in range(count):
        end = start + size + (index < extra)
        result.append(sorted(cpu for siblings in available[start:end] for cpu in siblings))
        start = end
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", type=int)
    parser.add_argument("--exclude-pid", type=int, action="append", default=[])
    options = parser.parse_args()
    try:
        reserved: set[int] = set()
        for pid in options.exclude_pid:
            try:
                reserved.update(os.sched_getaffinity(pid))
            except ProcessLookupError:
                continue
        groups = partition_cpu_sets(options.instances, os.sched_getaffinity(0), reserved)
    except (OSError, ValueError) as error:
        print(f"CPU affinity: {error}", file=sys.stderr)
        return 1
    for group in groups:
        print(",".join(str(cpu) for cpu in group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
