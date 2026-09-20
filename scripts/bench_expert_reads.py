#!/usr/bin/env python3
"""Paired benchmark of exact expert materialization from a warm mmap.

Run from an installed checkout, or set PYTHONPATH=src. A deterministic
synthetic shard is generated unless --shard is supplied. Each timed call
includes all MLX array creation and mx.eval. This is not a tokens/s test.
The baseline below is the complete _build method at fbab5f8c08e843e204c0fc6ae18b89a154c652cf.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from types import MethodType

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from edge0.backends import core
from edge0.moe.spec import MoESpec, QuantSpec, WeightLayout
from edge0.streaming.cache import SharedExpertCache
from edge0.streaming.layer import StreamingSwitchGLU
from edge0.streaming.mmap import SafetensorsMmap, u32_view
from edge0.streaming.options import LayerOptions

BASELINE_COMMIT = "fbab5f8c08e843e204c0fc6ae18b89a154c652cf"


def legacy_build(self, expert: int):
    """Load one expert's 9 tensors (gate/up/down x weight/scales/biases).

    Hot-stack fast path: if the expert is a member of the resident
    hot stack, slice its rows from the stack instead of reading the
    shard — zero IO, zero page-cache pressure."""
    if self._hot_backing is not None and expert in self._hot_set:
        idx = self._hot_key.index(expert)
        shape = self._bundle_shape
        b = {}
        for (proj, part), buf in self._hot_backing.items():
            sh = shape[(proj, part)]
            per = buf.size // len(self._hot_key)
            sl = buf[idx * per:(idx + 1) * per]
            if part == "weight":
                b[(proj, part)] = core.array(
                    sl.view("<u4").reshape(sh[1:]))
            else:
                b[(proj, part)] = core.array(sl.view("<u2")).view(
                    core.bfloat16).reshape(sh[1:])
        return b
    per_w = {}

    def _read_rows(proj, part):
        name = f"{self._prefix}.{proj}.{part}"
        raw = self._shard_for(name).raw(name)
        shape = self._shape[(proj, part)]
        per = raw.size // self.num_experts
        return raw[expert * per:(expert + 1) * per]

    def _to_mx(sl, key):
        part = key[1]
        shape = self._bundle_shape[key]
        if part == "weight":
            return core.array(u32_view(sl, shape[1:]))
        return core.array(sl.view("<u2")).view(
            core.bfloat16).reshape(shape[1:])

    if self._fuse_gu:
        # gate rows on top of up rows (matches split(x_gu, 2) order);
        # numpy-level concat of the raw byte slices before the single
        # core.array per part (pool thread, off the critical path).
        for part in ("weight", "scales", "biases"):
            sl = np.concatenate([
                _read_rows("gate_proj", part),
                _read_rows("up_proj", part)])
            per_w[("gate_up_proj", part)] = _to_mx(
                sl, ("gate_up_proj", part))
        for part in ("weight", "scales", "biases"):
            per_w[("down_proj", part)] = _to_mx(
                _read_rows("down_proj", part), ("down_proj", part))
        return per_w
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            per_w[(proj, part)] = _to_mx(
                _read_rows(proj, part), (proj, part))
    return per_w

def write_fixture(path, experts, dim, intermediate, seed):
    """INT4/BF16-shaped payloads; no trained model or download required."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for proj, rows, columns in (
        ("gate_proj", intermediate, dim),
        ("up_proj", intermediate, dim),
        ("down_proj", dim, intermediate),
    ):
        prefix = f"model.layers.0.experts.{proj}"
        tensors[prefix + ".weight"] = rng.integers(
            0, 2**32, (experts, rows, columns // 8), dtype=np.uint32)
        shape = (experts, rows, columns // 64)
        # Finite BF16 bit patterns. No quantization is changed in this test.
        tensors[prefix + ".scales"] = np.full(shape, 0x3C00, np.uint16)
        tensors[prefix + ".biases"] = np.full(shape, 0xBD00, np.uint16)
    save_file(tensors, str(path))


def bits(array):
    dtype = mx.uint16 if array.dtype == mx.bfloat16 else mx.uint32
    return np.asarray(array.view(dtype))


def measure(fn, ids, pool=None, batch_size=1):
    start = time.perf_counter_ns()
    if pool is None:
        for expert in ids:
            bundle = fn(expert)
            mx.eval(*bundle.values())
            del bundle
    else:
        for start_index in range(0, len(ids), batch_size):
            bundles = list(pool.map(fn, ids[start_index:start_index + batch_size]))
            mx.eval(*(tensor for bundle in bundles for tensor in bundle.values()))
            del bundles
    return (time.perf_counter_ns() - start) / len(ids)


def summarize(pairs, seed):
    base = np.array([p["baseline_ns"] for p in pairs])
    changed = np.array([p["candidate_ns"] for p in pairs])
    reductions = 1 - changed / base
    rng = np.random.default_rng(seed)
    samples = rng.choice(reductions, (10000, len(pairs)), replace=True)
    interval = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    return {
        "median_baseline_us": float(np.median(base) / 1000),
        "median_candidate_us": float(np.median(changed) / 1000),
        "median_paired_time_reduction": float(np.median(reductions)),
        "paired_median_bootstrap_95_ci": interval.tolist(),
        "candidate_faster_pairs": int(np.sum(changed < base)),
        "pairs": len(pairs),
    }


def run(args):
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("--cpu requires sched_setaffinity on this host")
        os.sched_setaffinity(0, {args.cpu})
    with ExitStack() as stack:
        if args.shard:
            path = args.shard
        else:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            path = Path(directory) / "fixture.safetensors"
            write_fixture(path, args.experts, args.dim, args.intermediate, args.seed)
        shard = SafetensorsMmap(str(path))
        stack.callback(shard.close)
        pool = None
        if args.workers > 1:
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
        prefix = args.prefix
        entry = shard.entries[f"{prefix}.gate_proj.weight"]
        experts = entry["shape"][0]
        # Resolve the literal prefix at layer 0; no source data is modified.
        dim = entry["shape"][2] * 8
        intermediate = entry["shape"][1]
        results = []
        for fused in (False, True):
            spec = MoESpec(
                num_experts=experts, top_k=min(4, experts),
                intermediate_size=intermediate, quant=QuantSpec(),
                layout=WeightLayout.FUSED_GATE_UP if fused else WeightLayout.SEPARATE,
                key_template=prefix,
            )
            layer = StreamingSwitchGLU(
                [shard], 0, spec,
                LayerOptions(load_threads=1, prefetch_threads=1, use_compile=False),
                shared_cache=SharedExpertCache(1),
            )
            try:
                baseline = MethodType(legacy_build, layer)
                candidate = layer._build
                for expert in range(experts):
                    old, new = baseline(expert), candidate(expert)
                    assert old.keys() == new.keys()
                    for key in old:
                        assert old[key].dtype == new[key].dtype
                        assert old[key].shape == new[key].shape
                        assert np.array_equal(bits(old[key]), bits(new[key]))
                    del old, new
                rng = np.random.default_rng(args.seed)
                warmup = rng.integers(0, experts, args.warmup).tolist()
                measure(baseline, warmup, pool, args.batch_size)
                measure(candidate, warmup, pool, args.batch_size)
                pairs = []
                was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for pair in range(args.pairs):
                        ids = rng.integers(0, experts, args.repeats).tolist()
                        order = ("baseline", "candidate")
                        if rng.integers(0, 2):
                            order = tuple(reversed(order))
                        record = {"order": list(order)}
                        for which in order:
                            fn = baseline if which == "baseline" else candidate
                            record[which + "_ns"] = measure(fn, ids, pool, args.batch_size)
                        pairs.append(record)
                finally:
                    if was_enabled:
                        gc.enable()
                result = {
                    "layout": "fused_gate_up" if fused else "separate",
                    "bitwise_equal_experts": experts,
                    "summary": summarize(pairs, args.seed),
                    "raw_pairs": pairs,
                }
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "raw_pairs"}), flush=True)
            finally:
                layer.close()
        payload_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                payload_hash.update(block)
        report = {
            "baseline_commit": BASELINE_COMMIT,
            "scope": "Warm-file cache-miss expert build, including MLX array creation and eval; no token throughput or physical I/O claim",
            "fixture": {
                "kind": "provided shard" if args.shard else "seeded synthetic packed weights",
                "experts": experts, "hidden_size": dim, "intermediate_size": intermediate,
                "file_bytes": path.stat().st_size, "sha256": payload_hash.hexdigest(),
            },
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"), "numpy": np.__version__,
                "device": str(mx.default_device()),
                "MLX_DISABLE_COMPILE": os.environ.get("MLX_DISABLE_COMPILE", "unset"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            },
            "config": {"seed": args.seed, "repeats_per_pair": args.repeats, "warmup_calls": args.warmup, "workers": args.workers, "batch_size": args.batch_size},
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--prefix", default="model.layers.0.experts")
    parser.add_argument("--experts", type=int, default=9)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--pairs", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("expert_read_benchmark.json"))
    args = parser.parse_args()
    if min(args.experts, args.pairs, args.repeats, args.warmup, args.workers, args.batch_size) < 1:
        parser.error("counts must be positive")
    if args.dim <= 0 or args.intermediate <= 0 or args.dim % 64 or args.intermediate % 64:
        parser.error("matrix dimensions must be positive multiples of 64")
    run(args)


if __name__ == "__main__":
    main()
