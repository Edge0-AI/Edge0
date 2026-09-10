#!/usr/bin/env python3
"""Benchmark page warming against the previous byte-copy/byte-sum paths.

Requires NumPy only; no MLX, model download, GPU, root, or global cache flush.
The optional Linux cold-page-cache test evicts ONLY its own temporary file.
Results measure host warming work, not token throughput or physical I/O bytes.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import platform
import struct
import tempfile
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/edge0/streaming/mmap.py"
spec = importlib.util.spec_from_file_location("edge0_mmap_benchmark", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def legacy_seq_read(shard, chunk=1 << 24):
    for offset in range(0, len(shard._mm), chunk):
        _ = shard._mm[offset:offset + chunk]


def legacy_expert_warm(shard, ranges):
    for name, expert, count in ranges:
        raw = shard.raw(name)
        per = raw.size // count
        _ = raw[expert * per:(expert + 1) * per].sum()


def page_expert_warm(shard, ranges):
    for name, expert, count in ranges:
        entry = shard.entries[name]
        per = entry["size"] // count
        shard.touch(entry["offset"] + expert * per, per)


def write_fixture(path, mib):
    # Four gate/up/down expert bundles with checkpoint-sized packed weights
    # and BF16 scale/bias byte spans. A filler tensor sets the shard size.
    shapes = {}
    offset = 0
    for proj in ("gate", "up", "down"):
        for part, per in (("weight", 524288), ("scales", 32768), ("biases", 32768)):
            size = 4 * per
            shapes[f"{proj}.{part}"] = {"dtype": "U8", "shape": [4, per],
                                        "data_offsets": [offset, offset + size]}
            offset += size
    total = mib * (1 << 20)
    if total < offset:
        raise ValueError("fixture must be at least 7 MiB")
    shapes["filler"] = {"dtype": "U8", "shape": [total - offset],
                        "data_offsets": [offset, total]}
    header = json.dumps(shapes).encode()
    rng = np.random.default_rng(812)
    block = rng.integers(0, 256, 1 << 20, dtype=np.uint8).tobytes()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for _ in range(mib):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def residents(shard):
    if platform.system() != "Linux":
        return None
    address = np.frombuffer(shard._mm, dtype=np.uint8).ctypes.data
    count = (len(shard._mm) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vector = (ctypes.c_ubyte * count)()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(len(shard._mm)), vector):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return sum(bool(x & 1) for x in vector)


def cold_own_file(shard):
    shard._mm.madvise(mmap.MADV_DONTNEED)
    os.posix_fadvise(shard._file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def timed(fn, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn()
    return (time.perf_counter_ns() - start) / loops


def paired(name, baseline, candidate, repeats, loops, rng, prepare=None):
    # Randomized AB/BA pairs; no benchmark threshold is used in unit tests.
    timings = []
    for _ in range(repeats):
        pair = {}
        for side in rng.permutation(["baseline", "candidate"]):
            if prepare is not None:
                prepare()
            pair[str(side)] = timed(baseline if side == "baseline" else candidate, loops)
        timings.append(pair)
    old = np.array([x["baseline"] for x in timings])
    new = np.array([x["candidate"] for x in timings])
    paired_reduction = 1 - new / old
    bootstrap = np.median(paired_reduction[rng.integers(0, repeats, (5000, repeats))], axis=1)
    return {"name": name, "pairs": repeats, "loops_per_sample": loops,
            "baseline_median_ms": float(np.median(old) / 1e6),
            "candidate_median_ms": float(np.median(new) / 1e6),
            "median_paired_time_reduction": float(np.median(paired_reduction)),
            "paired_reduction_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "samples_ns": timings}


def peak_allocation(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def run(args):
    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory(prefix="edge0-pagewarm-", dir=args.directory) as td:
        path = Path(td) / "fixture.safetensors"
        write_fixture(path, args.mib)
        before = digest(path)
        shard = module.SafetensorsMmap(str(path))
        try:
            ranges = [(name, expert, 4) for expert in range(4)
                      for name in shard.entries if name != "filler"]
            legacy_seq_read(shard)
            results = [
                paired("resident whole-shard warmup", lambda: legacy_seq_read(shard),
                       shard.seq_read, args.repeats, args.loops, rng),
                paired("resident four-expert warmup", lambda: legacy_expert_warm(shard, ranges),
                       lambda: page_expert_warm(shard, ranges), args.repeats, args.loops, rng),
            ]
            allocation = {"baseline_peak_traced_bytes": peak_allocation(lambda: legacy_seq_read(shard)),
                          "candidate_peak_traced_bytes": peak_allocation(shard.seq_read)}
            residency = {}
            for name, fn in (("baseline", lambda: legacy_seq_read(shard)), ("candidate", shard.seq_read)):
                if hasattr(shard._mm, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    shard._mm.madvise(mmap.MADV_DONTNEED)
                fn()
                residency[name] = residents(shard)
            if args.cold:
                if platform.system() != "Linux" or not hasattr(os, "posix_fadvise"):
                    raise RuntimeError("--cold requires Linux posix_fadvise")
                cold_details = []
                def prepare():
                    cold_own_file(shard)
                    cold_details.append({"resident_pages_before": residents(shard)})
                results.append(paired("cold file-page-cache whole-shard warmup",
                                      lambda: legacy_seq_read(shard), shard.seq_read,
                                      args.repeats, 1, rng, prepare))
                # prepare() is outside timing; preserve each precondition.
                results[-1]["preconditions_in_execution_order"] = cold_details
            total = len(shard._mm)
            result = {"schema": 1, "python": platform.python_version(), "numpy": np.__version__,
                      "system": platform.system(), "machine": platform.machine(),
                      "page_bytes": mmap.PAGESIZE, "fixture_bytes": total,
                      "fixture_sha256": before, "fixture_unchanged": digest(path) == before,
                      "fixture_kind": "synthetic safetensors with checkpoint-sized expert byte spans",
                      "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "legacy_reference_revision": "fbab5f8c08e843e204c0fc6ae18b89a154c652cf",
                      "seed": args.seed, "resident_pages_after_warmup": residency,
                      "total_mapped_pages": (total + mmap.PAGESIZE - 1) // mmap.PAGESIZE,
                      "allocation": allocation, "benchmarks": results,
                      "scope": "Host warming latency and traced temporary allocation only. No token-rate, model-quality, disk-bandwidth, GPU, or iPhone claim. Cold means this temporary file's OS page cache; storage-controller caches are not flushed."}
        finally:
            shard.close()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mib", type=int, default=128)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--directory", type=Path, help="temporary fixture directory on the drive under test")
    p.add_argument("--cold", action="store_true")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    if not 7 <= a.mib <= 4096 or a.repeats < 3 or a.loops < 1:
        p.error("require 7..4096 MiB, at least 3 pairs, and at least 1 loop")
    result = run(a)
    text = json.dumps(result, indent=2) + "\n"
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(text)
    print(text)
