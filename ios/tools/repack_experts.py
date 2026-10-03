#!/usr/bin/env python3
"""Repack routed-expert weights so one expert is one contiguous read.

The shipped safetensors store experts as nine separate stacked tensors per layer —
`gate_proj`/`up_proj`/`down_proj` × `weight`/`scales`/`biases` — each indexed by expert
along the leading axis. Reading one expert therefore means nine reads scattered across
a 19 GB file, and reading the K experts a layer routes to means eighteen. Measured on
an iPhone 16 Pro that costs 5.73 ms per layer at roughly 430 MB/s, against a device
ceiling near 2 GB/s.

This rewrites the same bytes grouped by (layer, expert):

    block(layer, expert) = gate.weight ‖ gate.scales ‖ gate.biases
                         ‖ up.weight   ‖ up.scales   ‖ up.biases
                         ‖ down.weight ‖ down.scales ‖ down.biases

Every block is 1,769,472 bytes — exactly 108 pages of 16 KB — so with the data starting
at offset 0 of its own file, **every block is page-aligned**. One expert becomes one
sequential 1.69 MB read.

Output is **one file per layer**, not one 17 GiB file. Per-layer files make transfers
retryable one layer at a time while keeping each expert's data contiguous.

Only the experts are rewritten. The resident weights (embeddings, attention, routers,
shared experts, norms — 2.13 GB) stay in the original shards, which this tool never
modifies.

The `resident` command extracts the other 2.13 GB — embeddings, attention, routers,
shared experts, norms — into a single standard safetensors file with its own index, so
that the original 17.7 GiB of shards need not be kept on the device at all. Nearly all
of what those shards hold is expert weights that the repacked files already carry.

Usage:
    repack_experts.py pack     <model-dir> <out-dir>
    repack_experts.py verify   <model-dir> <out-dir> [samples]
    repack_experts.py resident <model-dir> <out-dir>
"""

import json
import mmap
import os
import struct
import sys
import time

FORMAT = "edge0-expert-repack-v1"
PAGE = 16384

# Order within a block. Grouped by projection so that a consumer reading only, say,
# gate_proj still reads one contiguous run rather than three strided ones.
PARTS = [
    ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj", "weight"), ("up_proj", "scales"), ("up_proj", "biases"),
    ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
]


def tensor_name(layer, projection, part):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}.{part}"


def layer_file(layer):
    return f"experts-L{layer:02d}.bin"


class Shards:
    """The original shards, mapped read-only, with their headers parsed."""

    def __init__(self, model_dir):
        self.dir = model_dir
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.weight_map = index["weight_map"]

        self.maps = {}
        self.headers = {}
        self._files = []
        for filename in sorted(set(self.weight_map.values())):
            handle = open(os.path.join(model_dir, filename), "rb")
            self._files.append(handle)
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
            self.headers[filename] = (header, 8 + header_length)
            self.maps[filename] = mmap.mmap(
                handle.fileno(), 0, prot=mmap.PROT_READ)

    def close(self):
        for mapping in self.maps.values():
            mapping.close()
        for handle in self._files:
            handle.close()

    def entry(self, name):
        filename = self.weight_map[name]
        header, base = self.headers[filename]
        field = header[name]
        start, end = field["data_offsets"]
        return filename, base + start, end - start, field["dtype"], field["shape"]

    def slice_of(self, name, expert, experts):
        """One expert's bytes out of a stacked tensor."""
        filename, offset, total, _, _ = self.entry(name)
        per = total // experts
        start = offset + expert * per
        return self.maps[filename][start:start + per]


def describe(shards, layers, experts):
    """Block layout, derived from the real headers rather than assumed."""
    parts = []
    cursor = 0
    for projection, part in PARTS:
        _, _, total, dtype, shape = shards.entry(tensor_name(0, projection, part))
        per = total // experts
        parts.append({
            "name": f"{projection}.{part}",
            "projection": projection,
            "part": part,
            "dtype": dtype,
            "shape": shape[1:],          # per-expert shape, leading axis dropped
            "offset": cursor,
            "bytes": per,
        })
        cursor += per
    return parts, cursor


def pack(model_dir, out_dir):
    shards = Shards(model_dir)
    try:
        layers = 1 + max(
            int(name.split(".layers.")[1].split(".")[0])
            for name in shards.weight_map
            if ".layers." in name and ".switch_mlp." in name)
        _, _, total, _, shape = shards.entry(tensor_name(0, "gate_proj", "weight"))
        experts = shape[0]

        parts, block_bytes = describe(shards, layers, experts)
        if block_bytes % PAGE:
            print(f"  ! block is {block_bytes} B, not a multiple of {PAGE} — "
                  f"blocks will not be page-aligned")

        os.makedirs(out_dir, exist_ok=True)
        expected = layers * experts * block_bytes
        per_layer = experts * block_bytes

        free = os.statvfs(out_dir)
        if free.f_bavail * free.f_frsize < expected:
            raise SystemExit(
                f"need {expected / 2**30:.2f} GiB, "
                f"{free.f_bavail * free.f_frsize / 2**30:.2f} GiB free — refusing to "
                f"start rather than fail half-written")

        print(f"  {layers} layers × {experts} experts × {block_bytes:,} B "
              f"= {expected / 2**30:.2f} GiB")
        print(f"  block = {block_bytes / PAGE:.2f} pages of {PAGE} B")
        print(f"  one file per layer, {per_layer / 2**20:.0f} MiB each")

        started = time.time()
        written = 0
        for layer in range(layers):
            sources = {
                (projection, part): shards.entry(tensor_name(layer, projection, part))
                for projection, part in PARTS
            }
            path = os.path.join(out_dir, layer_file(layer))
            # Written to a temporary name and renamed only on success, so an interrupted
            # run cannot leave behind a file that looks complete.
            temporary = path + ".partial"
            layer_written = 0
            with open(temporary, "wb", buffering=1 << 22) as out:
                for expert in range(experts):
                    for projection, part in PARTS:
                        filename, offset, total, _, _ = sources[(projection, part)]
                        per = total // experts
                        start = offset + expert * per
                        out.write(shards.maps[filename][start:start + per])
                        layer_written += per
            if layer_written != per_layer:
                os.unlink(temporary)
                raise SystemExit(
                    f"layer {layer}: wrote {layer_written} B, expected {per_layer}")
            os.rename(temporary, path)

            written += layer_written
            elapsed = time.time() - started
            rate = written / elapsed / 2**20 if elapsed else 0
            print(f"\r  layer {layer + 1}/{layers}  {written / 2**30:.2f} GiB  "
                  f"{rate:.0f} MB/s", end="", flush=True)
        print()

        json.dump({
            "format": FORMAT,
            "layers": layers,
            "experts_per_layer": experts,
            "block_bytes": block_bytes,
            "layer_bytes": per_layer,
            "page_size": PAGE,
            "data_file_pattern": "experts-L%02d.bin",
            "parts": parts,
            "note": "expert e of layer L is at offset e * block_bytes in that layer's "
                    "file; resident weights remain in the original shards",
        }, open(os.path.join(out_dir, "experts.json"), "w"), indent=2)

        print(f"  ✓ {written / 2**30:.2f} GiB in {time.time() - started:.0f}s")
    finally:
        shards.close()


def verify(model_dir, out_dir, samples=64):
    """Compare sampled blocks against the originals, byte for byte."""
    shards = Shards(model_dir)
    try:
        meta = json.load(open(os.path.join(out_dir, "experts.json")))
        if meta["format"] != FORMAT:
            raise SystemExit(f"unexpected format {meta['format']}")

        layers = meta["layers"]
        experts = meta["experts_per_layer"]
        block_bytes = meta["block_bytes"]
        per_layer = meta["layer_bytes"]

        # Every layer's file must be present and exactly the right size before any
        # block is compared. A missing or truncated layer is the failure mode the
        # per-layer split exists to make recoverable, so it is checked first and
        # reported in full rather than one file at a time.
        broken = []
        for layer in range(layers):
            path = os.path.join(out_dir, layer_file(layer))
            if not os.path.exists(path):
                broken.append(f"layer {layer}: missing")
            elif os.path.getsize(path) != per_layer:
                broken.append(
                    f"layer {layer}: {os.path.getsize(path)} B, expected {per_layer}")
        if broken:
            raise SystemExit("✗ " + "\n  ✗ ".join(broken))

        # Corners first, then a spread. A bug in the offset arithmetic shows up at
        # the last layer's last expert long before it shows up in the middle.
        picks = [(0, 0), (0, experts - 1), (layers - 1, 0), (layers - 1, experts - 1)]
        step = max(1, (layers * experts) // max(1, samples - len(picks)))
        for flat in range(0, layers * experts, step):
            picks.append((flat // experts, flat % experts))

        checked = 0
        for layer, expert in sorted(picks):
            with open(os.path.join(out_dir, layer_file(layer)), "rb") as handle:
                packed = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
                base = expert * block_bytes
                for entry in meta["parts"]:
                    projection, part = entry["projection"], entry["part"]
                    original = shards.slice_of(
                        tensor_name(layer, projection, part), expert, experts)
                    start = base + entry["offset"]
                    if packed[start:start + entry["bytes"]] != original:
                        packed.close()
                        raise SystemExit(
                            f"✗ mismatch at layer {layer} expert {expert} "
                            f"{projection}.{part}")
                packed.close()
            checked += 1
            print(f"\r  verified {checked}/{len(picks)} blocks", end="", flush=True)
        print(f"\n  ✓ {checked} blocks across {layers} layer files match the originals "
              f"byte for byte")
    finally:
        shards.close()


def resident(model_dir, out_dir):
    """Everything that is not a routed expert, in one safetensors file.

    Written as a standard safetensors file plus an index naming it, so the device-side
    reader opens it exactly as it opens the originals — no format to teach it.

    Verification happens in the same pass rather than as a separate command: every
    tensor is read back from the finished file and compared against its source, because
    this file replaces the shards entirely and a silent mismatch here would not surface
    until inference produced quiet nonsense.
    """
    shards = Shards(model_dir)
    try:
        names = sorted(n for n in shards.weight_map if ".switch_mlp." not in n)
        header = {}
        cursor = 0
        for name in names:
            _, _, byte_count, dtype, shape = shards.entry(name)
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [cursor, cursor + byte_count],
            }
            cursor += byte_count

        os.makedirs(out_dir, exist_ok=True)
        blob = json.dumps(header, separators=(",", ":")).encode()
        path = os.path.join(out_dir, "resident.safetensors")
        temporary = path + ".partial"

        print(f"  {len(names)} tensors, {cursor / 2**30:.2f} GiB")
        started = time.time()
        written = 0
        with open(temporary, "wb", buffering=1 << 22) as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for index, name in enumerate(names):
                filename, offset, byte_count, _, _ = shards.entry(name)
                out.write(shards.maps[filename][offset:offset + byte_count])
                written += byte_count
                if index % 100 == 0:
                    print(f"\r  {index}/{len(names)}  {written / 2**30:.2f} GiB",
                          end="", flush=True)
        print(f"\r  {len(names)}/{len(names)}  {written / 2**30:.2f} GiB")

        if written != cursor:
            os.unlink(temporary)
            raise SystemExit(f"wrote {written} B, expected {cursor} — output removed")
        os.rename(temporary, path)

        json.dump({
            "metadata": {"note": "resident weights only; experts live in experts-L*.bin"},
            "weight_map": {name: "resident.safetensors" for name in names},
        }, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

        # Read back and compare, all of it.
        base = 8 + len(blob)
        with open(path, "rb") as handle:
            packed = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
            for index, name in enumerate(names):
                filename, offset, byte_count, _, _ = shards.entry(name)
                start = base + header[name]["data_offsets"][0]
                if packed[start:start + byte_count] != \
                        shards.maps[filename][offset:offset + byte_count]:
                    packed.close()
                    raise SystemExit(f"✗ mismatch in {name}")
                if index % 100 == 0:
                    print(f"\r  verifying {index}/{len(names)}", end="", flush=True)
            packed.close()
        print(f"\r  ✓ all {len(names)} tensors match, "
              f"{cursor / 2**30:.2f} GiB in {time.time() - started:.0f}s")
    finally:
        shards.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    command, model_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    if command == "pack":
        pack(model_dir, out_dir)
    elif command == "verify":
        verify(model_dir, out_dir, int(sys.argv[4]) if len(sys.argv) > 4 else 64)
    elif command == "resident":
        resident(model_dir, out_dir)
    else:
        raise SystemExit(__doc__)
