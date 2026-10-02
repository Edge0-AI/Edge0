#!/usr/bin/env python3
"""Convert trained pregate tensors to a memory-mappable safetensors file.

The converter stacks the per-layer heads, stores linear weights transposed for batched
matrix multiplication, preserves their fp16 values, and records the owning layer IDs.
It accepts an NPZ or safetensors source.

Usage:
    convert_pregate.py <input.npz|input.safetensors> <output.safetensors>
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_safetensors(path):
    """Small NumPy-only reader so deployment does not require another package."""
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    base = 8 + header_size
    dtype = {"F16": np.dtype("<f2"), "F32": np.dtype("<f4"), "I32": np.dtype("<i4")}
    result = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        result[name] = np.memmap(
            path, mode="r", dtype=dtype[entry["dtype"]], offset=base + start,
            shape=tuple(entry["shape"]), order="C")
    return result


def save_safetensors(tensors, path):
    """Write the subset of safetensors used by these converted fp16/int32 heads."""
    dtype = {np.dtype("<f2"): "F16", np.dtype("<f4"): "F32", np.dtype("<i4"): "I32"}
    header = {}
    offset = 0
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        tensors[name] = value
        size = value.nbytes
        header[name] = {
            "dtype": dtype[value.dtype], "shape": list(value.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb", buffering=4 << 20) as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for value in tensors.values():
            handle.write(memoryview(value).cast("B"))
    temporary.replace(path)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])

    # Accept both supported checkpoint containers.
    if source.suffix == ".safetensors":
        npz = load_safetensors(source)
    else:
        npz = dict(np.load(source))
    owners = sorted({int(name.split(".")[1]) for name in npz})
    if not owners:
        raise SystemExit(f"no `layers.N.*` tensors in {source}")

    # Normalize the checkpoint's `prerouter` names to the runtime's `pregate` names.
    prefix = "pregate"
    parts = {}
    for part in ("fc1", "fc2", "linear_init"):
        stack = []
        for owner in owners:
            key = f"layers.{owner}.{part}.weight"
            if key not in npz:
                raise SystemExit(f"{source.name} is missing {key}")
            # `[out, in]` -> `[in, out]`, contiguous. `ascontiguousarray` is not
            # decoration: safetensors stores a flat row-major buffer and a transposed
            # view is not one, so without it the file would either fail to write or, far
            # worse, write the untransposed bytes under the transposed shape.
            stack.append(np.ascontiguousarray(npz[key].T))
        merged = np.stack(stack, axis=0)
        if merged.dtype != np.float16:
            raise SystemExit(f"{part} is {merged.dtype}, expected float16")
        parts[f"{prefix}.{part}"] = merged

    parts[f"{prefix}.owners"] = np.array(owners, dtype=np.int32)

    save_safetensors(parts, destination)
    total = sum(v.nbytes for v in parts.values())
    print(f"  {destination}")
    for name, value in parts.items():
        print(f"    {name:24s} {str(list(value.shape)):20s} {value.dtype}")
    print(f"  owners {owners[0]}..{owners[-1]}  n={len(owners)}  "
          f"{total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
