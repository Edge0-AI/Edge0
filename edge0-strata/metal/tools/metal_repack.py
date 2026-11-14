#!/usr/bin/env python3
"""metal_repack.py: a multi-shard GGUF (Unsloth's UD-*_K_XL: 4-6 shards, shards cutting through layers) as ONE GGUF
for the Metal engine, which maps a single file into one GPU buffer (or streams its experts from it).

    python3 metal/tools/metal_repack.py --shard1 <...-00001-of-0000N.gguf> --out <model>-metal.gguf [--verify]

--verify: after writing, every tensor of the output is hashed (SHA-256) against its bytes in the source shard; any
difference stops with an error and removes the output.

The output holds shard 1's metadata (without the split.* keys) and every tensor of every shard except the n-gram
table (per_layer_token_embd: it stays in its shard, read from the SSD by the host through --ple-gguf), byte for
byte. The dense tensors come first; then each layer's routed experts (gate, up, down) together. Every tensor starts
on a 16 KiB boundary (general.alignment = 16384: Apple Silicon's page). Prints the shard holding the n-gram table.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import os
import re
import struct
import sys
from pathlib import Path

ALIGN = 16384
SYNC_EVERY = 1 << 30      # the output is flushed to the disk every GiB: at most this much (plus a chunk) is ever dirty
CHUNK = 64 << 20
LOCK_SUFFIX = ".repack.lock"   # <out>.repack.lock holds the repack's pid while it runs (setup_mac.py and the engine look)
# GGML type -> (bytes per block, values per block)
TYPE_SIZE = {0: (4, 1), 1: (2, 1), 30: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32),
             10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 16: (66, 256),
             17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256), 22: (82, 256), 23: (136, 256),
             42: (18, 64)}
PLE = "per_layer_token_embd.weight"


def nocache(f) -> None:
    """Writes through this file skip the page cache (macOS F_NOCACHE; measured with mincore on a 4 GB write: 100% of the file cached
    without it, 0% with it).  A repack writes 100+ GB: through the cache that is 100+ GB of dirty pages competing with everything
    else for memory (the 128 GB M5 Max keeps a 37 GB file-cache floor and compresses other programs' memory beyond it).  READS are
    not helped on this macOS (a cold 4 GB read was 98% cached afterwards with F_NOCACHE set); those pages are clean and evictable,
    the harm of the old tool was the unbounded dirty output."""
    try:
        import fcntl
        fcntl.fcntl(f.fileno(), getattr(fcntl, "F_NOCACHE", 48), 1)
    except (ImportError, OSError, ValueError):
        pass


def drop_cache(f) -> None:
    """Where F_NOCACHE does not exist (Linux, the BSDs): ask the kernel to drop this file's cached pages."""
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def write_all(o, buf) -> None:
    mv = memoryview(buf)
    while len(mv):
        mv = mv[o.write(mv):]


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")
        magic, self.version, nt, nkv = struct.unpack("<4sIQQ", self.f.read(24))
        if magic != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        self.kv = []                                    # (key, raw bytes of the whole entry)
        self.align = 32
        for _ in range(nkv):
            start = self.f.tell()
            key = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            val = self._val(t)
            raw_end = self.f.tell()
            self.f.seek(start)
            raw = self.f.read(raw_end - start)
            self.kv.append((key, raw))
            if key == "general.alignment":
                self.align = int(val)
        self.tensors = []
        for _ in range(nt):
            name = self._str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", self.f.read(8 * nd))
            ty, off = struct.unpack("<IQ", self.f.read(12))
            self.tensors.append({"name": name, "dims": dims, "type": ty, "off": off})
        pos = self.f.tell()
        self.data = (pos + self.align - 1) // self.align * self.align
        for t in self.tensors:
            ne = 1
            for d in t["dims"]:
                ne *= d
            b, q = TYPE_SIZE[t["type"]]
            t["bytes"] = ne // q * b
            t["src"] = self
            t["abs"] = self.data + t["off"]
        nocache(self.f)                                 # the header is read; the tensors' bytes are read once, uncached

    def _str(self) -> str:
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        sz = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t == 8:
            return self._str()
        if t == 9:
            at, n = struct.unpack("<IQ", self.f.read(12))
            return [self._val(at) for _ in range(n)]
        return struct.unpack("<" + sz[t], self.f.read(struct.calcsize(sz[t])))[0]


def shard_paths(shard1: Path) -> list[Path]:
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", shard1.name)
    if not m:
        return [shard1]
    n = int(m.group(2))
    return [shard1.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def tensor_hash(f, pos: int, n: int) -> str:
    h = hashlib.sha256()
    f.seek(pos)
    while n:
        buf = f.read(min(n, CHUNK))
        if not buf:
            break
        h.update(buf)
        n -= len(buf)
    return h.hexdigest()


def verify(out: Path, data0: int, order: list) -> list[str]:
    """The names of the tensors whose bytes in `out` differ from their source shard's (SHA-256 each)."""
    bad = []
    with open(out, "rb") as o:
        nocache(o)
        r = Reader(out)                                # the output's own table, as a reader sees it
        table = {t["name"]: t for t in r.tensors}
        for i, t in enumerate(order):
            mine = table.get(t["name"])
            if mine is None or mine["abs"] != data0 + t["new"] or mine["bytes"] != t["bytes"] or \
                    tensor_hash(o, mine["abs"], t["bytes"]) != tensor_hash(t["src"].f, t["abs"], t["bytes"]):
                bad.append(t["name"])
            if i % 50 == 0:
                print(f"\r  verify {i + 1} / {len(order)}", end="", flush=True)
        r.f.close()
    print()
    return bad


def gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard1", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--verify", action="store_true", help="hash every tensor of the output against the source")
    a = ap.parse_args()
    shards = [Reader(p) for p in shard_paths(a.shard1)]
    every = [t for s in shards for t in s.tensors]
    ple = [t for t in every if t["name"] == PLE]
    keep = [t for t in every if t["name"] != PLE]

    def layer_of(t):
        m = re.match(r"blk\.(\d+)\.", t["name"])
        return int(m.group(1)) if m else -1
    dense = [t for t in keep if "_exps." not in t["name"]]
    role = {"ffn_gate_exps": 0, "ffn_up_exps": 1, "ffn_down_exps": 2}
    exps = sorted((t for t in keep if "_exps." in t["name"]),
                  key=lambda t: (layer_of(t), role.get(t["name"].split(".")[2], 9)))
    order = dense + exps
    kv = [raw for key, raw in shards[0].kv if not key.startswith("split.") and key != "general.alignment"]
    kv.append(gguf_str("general.alignment") + struct.pack("<II", 4, ALIGN))
    off = 0
    infos = []
    for t in order:
        t["new"] = off
        infos.append(gguf_str(t["name"]) + struct.pack("<I", len(t["dims"])) +
                     struct.pack(f"<{len(t['dims'])}Q", *t["dims"]) + struct.pack("<IQ", t["type"], off))
        off = (off + t["bytes"] + ALIGN - 1) // ALIGN * ALIGN
    head = struct.pack("<4sIQQ", b"GGUF", 3, len(order), len(kv)) + b"".join(kv) + b"".join(infos)
    data0 = (len(head) + ALIGN - 1) // ALIGN * ALIGN
    total = data0 + off
    tmp = a.out.with_name(a.out.name + ".part")
    lock = a.out.with_name(a.out.name + LOCK_SUFFIX)
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink() if lock.exists() else None)
    # unbuffered writes (a write() is a write), uncached, flushed every SYNC_EVERY bytes: the dirty pages stay a chunk or two,
    # wherever the other programs' memory stands (the old buffered, cached write left 2+ GB dirty and filled the RAM)
    with open(tmp, "wb", buffering=0) as o:
        nocache(o)
        write_all(o, head)
        write_all(o, b"\0" * (data0 - len(head)))
        done = 0
        since = 0
        for t in order:
            o.seek(data0 + t["new"])
            src = t["src"].f
            src.seek(t["abs"])
            left = t["bytes"]
            while left:
                buf = src.read(min(left, CHUNK))
                if not buf:
                    sys.exit(f"{t['src'].path}: short read in {t['name']}")
                write_all(o, buf)
                left -= len(buf)
                since += len(buf)
                if since >= SYNC_EVERY:
                    os.fsync(o.fileno())
                    drop_cache(o)
                    drop_cache(src)
                    since = 0
            done += t["bytes"]
            if t is order[-1] or t["name"].endswith("ffn_down_exps.weight"):
                print(f"\r  {done / 2**30:6.1f} / {sum(x['bytes'] for x in order) / 2**30:.1f} GiB", end="", flush=True)
        o.truncate(total)
        os.fsync(o.fileno())
    print()
    if a.verify:
        bad = verify(tmp, data0, order)
        if bad:
            tmp.unlink()
            sys.exit(f"verify FAILED: {len(bad)} tensors differ from the source, first {bad[0]} (output removed)")
        print(f"verify: {len(order)} tensors, SHA-256 identical to the source shards")
    os.replace(tmp, a.out)
    print(f"{a.out}: {len(order)} tensors ({len(dense)} dense, {len(exps)} expert), {total / 2**30:.2f} GiB")
    if ple:
        print(f"n-gram table: {ple[0]['src'].path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
