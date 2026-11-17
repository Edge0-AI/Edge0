# Copyright (c) 2026 the Strata contributors (MIT).
"""tools/dsv41/engram_iq4nl_bench.py - the Phase 1.3 IQ4_NL Engram reader: O_DIRECT row reads + host dequant.

The Engram is a big IQ4_NL table (engram-IQ4_NL.gguf, 110.6 GB in helcig/tmp-oct26).  The engine, for one token,
gathers 48 rows of it by hash (tools/dsv41/engram_hash.py gives the row ids), so the reader's job is:

    random 2.3 KB row reads straight off the device (O_DIRECT: no page cache, the table does not fit in RAM),
    dequantised on the host to f32.

This is the Phase 1.3 skeleton: it does the two things the real run needs to be trusted before it is written into the
engine - it checks the dequant against ggml's own block_IQ4_NL decode, bit for bit, and it measures the two access
patterns (48 rows = one token; a prompt-chunk batch) on whatever file it is pointed at.

The block (ggml's block_iq4_nl, third_party/ggml/ggml-common.h): 32 values, one fp16 scale `d`, 16 packed nibbles:

    typedef struct { ggml_half d; uint8_t qs[16]; } block_iq4_nl;      // 18 bytes, QK4_NL = 32

    y[base + j     ] = d * kvalues_iq4nl[qs[j] & 0x0F]     (j = 0..15)
    y[base + j + 16] = d * kvalues_iq4nl[qs[j] >> 4]
    kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113}

so a row of n_embd values is (n_embd/32)*18 bytes.

No model is needed to exercise it: `--make-synthetic` writes a valid GGUF with one IQ4_NL tensor (header, one tensor
info, 32-byte alignment, random packed nibbles and small fp16 scales) of the size you ask for, and every mode runs
against that.  What the real run needs is listed in REAL RUN below and printed by --help.

    # on p1004 (no numpy there: this file is stdlib only)
    python3 tools/dsv41/engram_iq4nl_bench.py --make-synthetic /tmp/engram-synth.gguf --size-gb 1.0
    python3 tools/dsv41/engram_iq4nl_bench.py --gguf /tmp/engram-synth.gguf --mode all \\
            --rows-per-token 48 --batch-tokens 256 --iters 20

REAL RUN (what this needs that p1004 does not have today):
  * the file: engram-IQ4_NL.gguf, 110.6 GB.  p1004 has 14 GB free on /data and 65 GB on /, so it needs ~50 GB freed
    first (P1004.md's /data/launch-test + /data/rel023-amd is 71 GB).  It must sit on a real block device:
    O_DIRECT works on ext4/xfs/nvme, not on tmpfs or overlayfs.
  * the tensor geometry is read from the GGUF header (n_embd, n_rows); nothing is hardcoded here.
  * `--ggml-dump`: a 3-line C++ program that calls ggml's dequantize_row_iq4_nl and prints the float bits, linked
    against a built libggml-base.a (-I <llama.cpp>/ggml/include -I <llama.cpp>/ggml/src).  --build-ggml-dump writes
    and compiles it from the source embedded below.  Without it the check is the script's scalar reference against
    its own fast path, which is still bit-for-bit but not independent of this file.
  * numpy: optional.  It is not installed on p1004, so dequant here is pure Python (~8k rows/s); the engine's dequant
    is the C kernel in ds4_iq.cu / ds4_gg_cpu.cpp.  Use --dequant off for the I/O ceiling, --dequant sample (default)
    for a measured dequant rate, --dequant all to dequantise the whole batch.
  * the real 48-row gather per token is random over 110 GB: IOPS matter as much as MB/s.  --mode token is that
    pattern; --mode batch is the prompt chunk; --mode seq is the device's sequential ceiling for reference.
"""
from __future__ import annotations

import argparse
import array
import mmap
import os
import random
import struct
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------- ggml IQ4_NL

QK4_NL = 32                    # values per block
IQ4NL_ROW_DIV = 32             # n_embd must be a multiple of this
IQ4NL_BLOCK_BYTES = 18         # fp16 d + 16 packed nibbles
GGML_TYPE_IQ4_NL = 20          # ggml.h: IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ1_S 19, IQ4_NL 20, IQ3_S 21, IQ2_S 22
KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def iq4nl_row_bytes(n_embd: int) -> int:
    if n_embd % IQ4NL_ROW_DIV:
        raise ValueError(f"n_embd {n_embd} is not a multiple of {IQ4NL_ROW_DIV}")
    return (n_embd // QK4_NL) * IQ4NL_BLOCK_BYTES


def dequant_row_iq4nl_ref(row: bytes, n_embd: int) -> array.array:
    """ggml's dequantize_row_iq4_nl, literal: one fp16 scale per block, 16 low nibbles then 16 high ones.

    Python floats are doubles; a float32 * a small integer is exact in a double (24 + 7 bits) and array('f') rounds
    it once on store, so this reproduces ggml's float32 product bit for bit.
    """
    out = array.array("f")
    nb = n_embd // QK4_NL
    for b in range(nb):
        base = b * IQ4NL_BLOCK_BYTES
        d = struct.unpack_from("<e", row, base)[0]
        qs = row[base + 2:base + IQ4NL_BLOCK_BYTES]
        out.extend(d * KVALUES_IQ4NL[q & 0x0F] for q in qs)
        out.extend(d * KVALUES_IQ4NL[q >> 4] for q in qs)
    return out


def dequant_row_iq4nl_fast(row: bytes, n_embd: int) -> array.array:
    """The same result, with the block's 16 possible values built once: the reader's hot path."""
    out = array.array("f")
    nb = n_embd // QK4_NL
    unpack = struct.unpack_from
    kv = KVALUES_IQ4NL
    extend = out.extend
    for b in range(nb):
        base = b * IQ4NL_BLOCK_BYTES
        d = unpack("<e", row, base)[0]
        tab = (d * kv[0], d * kv[1], d * kv[2], d * kv[3], d * kv[4], d * kv[5], d * kv[6], d * kv[7],
               d * kv[8], d * kv[9], d * kv[10], d * kv[11], d * kv[12], d * kv[13], d * kv[14], d * kv[15])
        qs = row[base + 2:base + IQ4NL_BLOCK_BYTES]
        extend([tab[q & 0x0F] for q in qs])
        extend([tab[q >> 4] for q in qs])
    return out


def f32_bits(vals) -> bytes:
    return array.array("f", vals).tobytes()


# ---------------------------------------------------------------- GGUF reader

class Gguf:
    """Just enough GGUF for one tensor: magic, version, the kv block, the tensor infos, the aligned data start."""

    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        self.tensors: list[dict] = []
        self.meta: dict = {}
        self.data_start = 0
        self.alignment = 32
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"{path}: not a GGUF (magic {magic!r})")
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"{path}: GGUF v{version}, this reads v2 and v3")
            self.version = version
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                k, v = self._kv(f)
                self.meta[k] = v
            self.alignment = int(self.meta.get("general.alignment", 32)) or 32
            for _ in range(n_tensors):
                name = self._str(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                ttype = struct.unpack("<I", f.read(4))[0]
                off = struct.unpack("<Q", f.read(8))[0]
                self.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
            pos = f.tell()
            self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    @staticmethod
    def _str(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    _SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    def _kv(self, f):
        key = self._str(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        if vtype == 8:                                  # string
            return key, self._str(f)
        if vtype == 9:                                  # array
            atype = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if atype == 8:
                return key, [self._str(f) for _ in range(n)]
            fmt = self._SCALAR[atype]
            return key, list(struct.unpack(f"<{n}{fmt[1]}", f.read(struct.calcsize(fmt) * n)))
        fmt = self._SCALAR[vtype]
        return key, struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def tensor(self, name: str | None = None) -> dict:
        if name is not None:
            for t in self.tensors:
                if t["name"] == name:
                    return t
            raise KeyError(f"{self.path}: no tensor {name!r}; have {[t['name'] for t in self.tensors]}")
        for t in self.tensors:
            if t["type"] == GGML_TYPE_IQ4_NL:
                return t
        raise KeyError(f"{self.path}: no IQ4_NL tensor")

    def geometry(self, name: str | None = None) -> tuple[int, int, int]:
        """(n_embd, n_rows, row_bytes) for an IQ4_NL table; dims[0] is the row length."""
        t = self.tensor(name)
        if t["type"] != GGML_TYPE_IQ4_NL:
            raise ValueError(f"tensor {t['name']!r} is ggml type {t['type']}, not {GGML_TYPE_IQ4_NL} (IQ4_NL)")
        n_embd = int(t["dims"][0])
        n_rows = 1
        for d in t["dims"][1:]:
            n_rows *= int(d)
        return n_embd, n_rows, iq4nl_row_bytes(n_embd)

    def row_offset(self, t: dict, row: int) -> int:
        """Absolute file offset of a row: the tensor's offset is relative to the data section."""
        _, _, rb = self.geometry(t["name"])
        return self.data_start + t["offset"] + row * rb


# ---------------------------------------------------------------- O_DIRECT rows

class RowReader:
    """Aligned, unbuffered reads.  A row is 18*(n_embd/32) bytes (2304 for n_embd 4096), which is not a multiple of
    the device's 512-byte block, so a row read is a 512-aligned window that is then sliced: the device moves up to
    511 extra bytes per row and MB/s is reported both ways (payload and what the device was asked for).
    """

    def __init__(self, path: str, align: int = 512, odirect: bool = True):
        self.path = path
        self.align = align
        self.size = os.path.getsize(path)
        flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if odirect else 0)
        self.odirect = bool(odirect and getattr(os, "O_DIRECT", 0))
        try:
            self.fd = os.open(path, flags)
        except OSError as e:
            if not self.odirect:
                raise
            print(f"  (O_DIRECT unavailable on {path}: {e}; falling back to buffered reads)", file=sys.stderr)
            self.odirect = False
            self.fd = os.open(path, os.O_RDONLY)
        self._buf = None
        self._buf_len = 0
        self.device_bytes = 0

    def _buffer(self, n: int):
        if self._buf is None or self._buf_len < n:
            self._buf = mmap.mmap(-1, n)          # page aligned, which is a multiple of any 512/4096 sector
            self._buf_len = n
        return self._buf

    def read(self, off: int, length: int) -> bytes:
        """The `length` bytes at `off`, in one aligned pread."""
        if self.odirect:
            a = self.align
            start = off // a * a
            end = min((off + length + a - 1) // a * a, self.size)
            n = end - start
            if n <= 0:
                raise ValueError(f"read past end: off {off} len {length} size {self.size}")
            buf = self._buffer(n)
            got = os.preadv(self.fd, [memoryview(buf)[:n]], start)
            self.device_bytes += n
            if got != n:
                raise OSError(f"short O_DIRECT read at {start}: {got} of {n}")
            return bytes(memoryview(buf)[off - start:off - start + length])
        data = os.pread(self.fd, length, off)
        self.device_bytes += length
        if len(data) != length:
            raise OSError(f"short read at {off}: {len(data)} of {length}")
        return data

    def close(self):
        if self._buf is not None:
            self._buf.close()
            self._buf = None
        os.close(self.fd)


# ---------------------------------------------------------------- synthetic file

def make_synthetic(path: str, size_gb: float, n_embd: int = 4096, seed: int = 1, verbose: bool = True) -> tuple[int, int]:
    """A valid GGUF v3 with one IQ4_NL tensor: header, one tensor info, 32-byte alignment, then the data.

    The blocks are real blocks: fp16 scales in 0.002 .. 0.017 and random packed nibbles, so the dequant is exercised
    on the file the benchmarks read.  Returns (n_rows, file bytes).
    """
    rb = iq4nl_row_bytes(n_embd)
    n_rows = max(1, int(size_gb * 1e9) // rb)
    rng = random.Random(seed)
    scales = [0.002 + 0.001 * i for i in range(16)]
    # 8 block templates per scale cycle, each 18 bytes: the fp16 d and 16 random nibble bytes
    blobs = []
    for s in scales:
        parts = []
        for _ in range(8):
            parts.append(struct.pack("<e", s) + bytes(rng.getrandbits(8) for _ in range(16)))
        blobs.append(b"".join(parts))              # 144 bytes = 8 whole blocks, so a repeat keeps block alignment

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<QQ", 1, 3)
    for k, v in (("general.architecture", "engram"), ("general.name", "p1004-synthetic-iq4nl")):
        header += struct.pack("<Q", len(k)) + k.encode()
        header += struct.pack("<I", 8)
        header += struct.pack("<Q", len(v)) + v.encode()
    header += struct.pack("<Q", len("general.alignment")) + b"general.alignment"
    header += struct.pack("<I", 4) + struct.pack("<I", 32)
    name = b"engram.iq4nl"
    header += struct.pack("<Q", len(name)) + name
    header += struct.pack("<I", 2) + struct.pack("<QQ", n_embd, n_rows)
    header += struct.pack("<I", GGML_TYPE_IQ4_NL) + struct.pack("<Q", 0)
    header += b"\x00" * ((-len(header)) % 32)

    if verbose:
        print(f"synthetic: n_embd {n_embd} row_bytes {rb} n_rows {n_rows} data {(n_rows * rb) / 1e9:.3f} GB -> {path}")
    chunk_rows = 64
    written = 0
    with open(path, "wb", buffering=1024 * 1024) as f:
        f.write(header)
        k = 0
        t0 = time.time()
        while written < n_rows:
            rows = min(chunk_rows, n_rows - written)
            blob = blobs[k % len(blobs)]
            k += 1
            buf = (blob * (rows * rb // len(blob) + 1))[:rows * rb]
            f.write(buf)
            written += rows
            if verbose and written % (chunk_rows * 512) == 0:
                print(f"\r  {written * rb / 1e9:.2f} GB", end="", file=sys.stderr)
    total = os.path.getsize(path)
    if verbose:
        print(f"\r  wrote {total / 1e9:.3f} GB in {time.time() - t0:.1f} s" + " " * 20, file=sys.stderr)
    return n_rows, total


# ---------------------------------------------------------------- the ggml cross-check

_GGML_DUMP_SRC = r"""
// Read one IQ4_NL row from stdin, print ggml's own dequantise: one float per line, 8 hex digits of its bits.
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void dequantize_row_iq4_nl(const void * x, float * y, int64_t k);
int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: iq4nl_dump N_EMBD < row.bin\n"); return 2; }
    const int n = std::atoi(argv[1]);
    const size_t rb = (size_t) (n / 32) * 18;
    std::vector<unsigned char> row(rb);
    if (std::fread(row.data(), 1, rb, stdin) != rb) { std::fprintf(stderr, "short read\n"); return 2; }
    std::vector<float> y((size_t) n);
    dequantize_row_iq4_nl(row.data(), y.data(), n);
    for (int i = 0; i < n; ++i) {
        unsigned int b; __builtin_memcpy(&b, &y[i], 4);
        std::printf("%08x\n", b);
    }
    return 0;
}
"""


def build_ggml_dump(build_dir: str, out: str | None = None) -> str:
    """Compile the dumper against a built llama.cpp tree (ggml headers + libggml-base.a)."""
    src_root = os.path.join(build_dir, "_deps", "strata_llamacpp-src", "ggml")
    lib = os.path.join(build_dir, "ggml", "src", "libggml-base.a")
    for p in (src_root, lib):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found; pass --ggml-build-dir with a tree that built ggml")
    out = out or os.path.join(tempfile.gettempdir(), "iq4nl_dump")
    with open(out + ".cpp", "w") as f:
        f.write(_GGML_DUMP_SRC)
    cmd = ["g++", "-O2", "-std=c++17", "-I", os.path.join(src_root, "include"), "-I", os.path.join(src_root, "src"),
           out + ".cpp", lib, "-o", out, "-lpthread", "-lm"]
    subprocess.run(cmd, check=True)
    return out


def ggml_dequant_bits(dumper: str, row: bytes, n_embd: int) -> bytes:
    """ggml's dequant of one row, as raw float32 bytes (the dumper prints the bits as %08x, so repack little-endian)."""
    p = subprocess.run([dumper, str(n_embd)], input=row, stdout=subprocess.PIPE, check=True)
    hx = b"".join(p.stdout.split()).decode()
    if len(hx) != 8 * n_embd:
        raise RuntimeError(f"the dumper printed {len(hx)} hex digits for {n_embd} values; check its n_embd argument")
    return b"".join(struct.pack("<I", int(hx[i:i + 8], 16)) for i in range(0, len(hx), 8))


# ---------------------------------------------------------------- benchmarks

def _fmt(x, unit=""):
    return f"{x:,.1f}{unit}" if isinstance(x, float) else f"{x:,}{unit}"


def _dequant_rate(rows, n_embd: int, limit: int) -> tuple[float, int]:
    """(seconds, rows) for the host dequant of rows already in memory: the CPU side, apart from the I/O."""
    sel = rows[:limit]
    if not sel:
        return 0.0, 0
    t0 = time.perf_counter()
    for row in sel:
        dequant_row_iq4nl_fast(row, n_embd)
    return time.perf_counter() - t0, len(sel)


def bench_token(gg: Gguf, t: dict, n_embd: int, n_rows: int, rb: int, args) -> dict:
    """48 random rows per token, one aligned pread each: the decode-time pattern.  The reads are timed alone and
    the dequant is timed afterwards on the rows already in RAM, so neither number hides the other."""
    rnd = random.Random(args.seed)
    tok_rows = [rnd.sample(range(n_rows), args.rows_per_token) for _ in range(args.iters)]
    rd = RowReader(gg.path, align=args.align, odirect=args.odirect)
    blobs = []
    try:
        rd.read(gg.row_offset(t, tok_rows[0][0]), rb)          # warm the path, not the cache (O_DIRECT)
        t0 = time.perf_counter()
        for rows in tok_rows:
            for r in rows:
                blobs.append(rd.read(gg.row_offset(t, r), rb))
        dt = time.perf_counter() - t0
    finally:
        dev = rd.device_bytes
        rd.close()
    n = len(blobs)
    out = {
        "mode": f"token x{args.iters} ({args.rows_per_token} rows each)",
        "seconds": dt,
        "tokens_s": args.iters / dt,
        "rows_s": n / dt,
        "payload_mb_s": n * rb / dt / 1e6,
        "device_mb_s": dev / dt / 1e6,
        "reads": n,
        "read_us": dt / n * 1e6,
        "dequant": args.dequant,
    }
    if args.dequant != "off":
        lim = n if args.dequant == "all" else args.rows_per_token * min(args.iters, 2)
        ddt, dn = _dequant_rate(blobs, n_embd, lim)
        out["dequant_rows"], out["dequant_rows_s"] = dn, dn / ddt if ddt else 0.0
    return out


def bench_batch(gg: Gguf, t: dict, n_embd: int, n_rows: int, rb: int, args) -> dict:
    """A prompt chunk: batch_tokens x 48 rows, gathered at random like the engine will, plus the same rows read as
    contiguous runs of the table (the best case: a chunk's hashes are unrelated, so the real pattern is the random
    one).  Reads and dequant are timed apart, as in bench_token."""
    rnd = random.Random(args.seed + 1)
    n = args.batch_tokens * args.rows_per_token
    rows = [rnd.randrange(n_rows) for _ in range(n)]
    rd = RowReader(gg.path, align=args.align, odirect=args.odirect)
    blobs = []
    try:
        t0 = time.perf_counter()
        for r in rows:
            blobs.append(rd.read(gg.row_offset(t, r), rb))
        dt = time.perf_counter() - t0
    finally:
        dev = rd.device_bytes
        rd.close()
    out = {
        "mode": f"batch gathered {args.batch_tokens} tokens = {n} rows",
        "seconds": dt,
        "tokens_s": args.batch_tokens / dt,
        "rows_s": n / dt,
        "payload_mb_s": n * rb / dt / 1e6,
        "device_mb_s": dev / dt / 1e6,
        "reads": n,
        "read_us": dt / n * 1e6,
        "dequant": args.dequant,
    }
    if args.dequant != "off":
        lim = n if args.dequant == "all" else args.rows_per_token * 3
        ddt, dn = _dequant_rate(blobs, n_embd, lim)
        out["dequant_rows"], out["dequant_rows_s"] = dn, dn / ddt if ddt else 0.0
    del blobs
    # the same rows, contiguous: merge the sorted ids into runs and read each run with one pread
    srows = sorted(rows)
    runs = []
    lo = prev = srows[0]
    for r in srows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((lo, prev))
        lo = prev = r
    runs.append((lo, prev))
    rd = RowReader(gg.path, align=args.align, odirect=args.odirect)
    try:
        t0 = time.perf_counter()
        got = 0
        for lo, hi in runs:
            got += len(rd.read(gg.row_offset(t, lo), (hi - lo + 1) * rb))
        dt2 = time.perf_counter() - t0
    finally:
        dev2 = rd.device_bytes
        rd.close()
    out["contig"] = {
        "mode": f"batch contiguous {args.batch_tokens} tokens = {len(runs)} runs",
        "seconds": dt2,
        "tokens_s": args.batch_tokens / dt2,
        "rows_s": got / rb / dt2,
        "payload_mb_s": got / dt2 / 1e6,
        "device_mb_s": dev2 / dt2 / 1e6,
        "reads": len(runs),
        "read_us": dt2 / max(1, len(runs)) * 1e6,
        "dequant": "off",
    }
    return out


def bench_seq(gg: Gguf, rb: int, args) -> dict:
    """The device's sequential ceiling through the same O_DIRECT path (rows/s is that ceiling in 2.3 KB rows)."""
    mb = min(args.seq_mb, gg.size // (1024 * 1024))
    rd = RowReader(gg.path, align=args.align, odirect=args.odirect)
    chunk = args.align * 1024                              # 512 KiB
    try:
        rd.read(0, chunk)
        t0 = time.perf_counter()
        off = 0
        total = 0
        while total < mb * 1024 * 1024:
            n = min(chunk, gg.size - off)
            if n <= 0:
                break
            rd.read(off, n)
            off += n
            total += n
        dt = time.perf_counter() - t0
    finally:
        dev = rd.device_bytes
        rd.close()
    return {"mode": f"sequential {total / 1e6:.0f} MB, {chunk // 1024} KiB reads", "seconds": dt,
            "rows_s": total / rb / dt, "payload_mb_s": total / dt / 1e6, "device_mb_s": dev / dt / 1e6,
            "reads": total // chunk, "read_us": dt / max(1, total // chunk) * 1e6, "dequant": "off"}


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="the Engram GGUF (or a synthetic one)")
    ap.add_argument("--tensor", default=None, help="tensor name (default: the first IQ4_NL tensor)")
    ap.add_argument("--mode", default="all", choices=["header", "token", "batch", "seq", "all"])
    ap.add_argument("--rows-per-token", type=int, default=48)
    ap.add_argument("--batch-tokens", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seq-mb", type=int, default=512)
    ap.add_argument("--align", type=int, default=512, help="O_DIRECT alignment (the device's logical block size)")
    ap.add_argument("--odirect", action="store_true", default=True)
    ap.add_argument("--no-odirect", dest="odirect", action="store_false")
    ap.add_argument("--dequant", default="sample", choices=["off", "sample", "all"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify-rows", type=int, default=8, help="rows checked bit-exactly against ggml")
    ap.add_argument("--ggml-dump", help="a compiled ggml IQ4_NL dequant dumper (see --build-ggml-dump)")
    ap.add_argument("--build-ggml-dump", action="store_true", help="compile the dumper from the source embedded here")
    ap.add_argument("--ggml-build-dir", default="/data/dsv41-build", help="a Strata build tree with ggml built")
    ap.add_argument("--make-synthetic", metavar="PATH", help="write a synthetic IQ4_NL GGUF and exit (unless --gguf)")
    ap.add_argument("--size-gb", type=float, default=1.0)
    ap.add_argument("--n-embd", type=int, default=4096)
    args = ap.parse_args(argv)

    if args.make_synthetic:
        make_synthetic(args.make_synthetic, args.size_gb, args.n_embd, args.seed)
        if not args.gguf:
            args.gguf = args.make_synthetic
            if args.mode == "all":
                args.mode = "header"
    if not args.gguf:
        ap.error("--gguf or --make-synthetic is required")

    gg = Gguf(args.gguf)
    t = gg.tensor(args.tensor)
    n_embd, n_rows, rb = gg.geometry(args.tensor)
    print(f"{gg.path}: {gg.size / 1e9:.3f} GB, GGUF v{gg.meta.get('general.version', 3)}, {len(gg.tensors)} tensor(s), "
          f"alignment {gg.alignment}, data at {gg.data_start}")
    print(f"  tensor {t['name']!r}: type {t['type']} (IQ4_NL), dims {t['dims']}, n_embd {n_embd}, n_rows {n_rows}, "
          f"row_bytes {rb}, table {(n_rows * rb) / 1e9:.3f} GB")
    if t["type"] != GGML_TYPE_IQ4_NL:
        print(f"  not an IQ4_NL tensor: nothing to read", file=sys.stderr)
        return 2

    # the bit-exact check, always: the scalar reference against the fast path, then against ggml itself
    rnd = random.Random(args.seed + 7)
    dumper = args.ggml_dump
    if args.build_ggml_dump:
        dumper = build_ggml_dump(args.ggml_build_dir, dumper)
        print(f"  built the ggml dumper: {dumper}")
    rd = RowReader(gg.path, align=args.align, odirect=args.odirect)
    bad = 0
    checked = 0
    try:
        for _ in range(args.verify_rows):
            r = rnd.randrange(n_rows)
            row = rd.read(gg.row_offset(t, r), rb)
            ref = dequant_row_iq4nl_ref(row, n_embd)
            fast = dequant_row_iq4nl_fast(row, n_embd)
            if f32_bits(ref) != f32_bits(fast):
                bad += 1
                print(f"  row {r}: the scalar reference and the fast path differ", file=sys.stderr)
            if dumper:
                gw = ggml_dequant_bits(dumper, row, n_embd)
                if gw != f32_bits(ref):
                    bad += 1
                    print(f"  row {r}: ggml's dequant differs from the reference", file=sys.stderr)
            checked += 1
    finally:
        rd.close()
    print(f"  bit-exact: {checked - bad}/{checked} rows" + (" against ggml's dequantize_row_iq4_nl" if dumper else
                                                             " (reference vs fast path; no --ggml-dump given)")
          + ("  FAIL" if bad else "  OK"))
    if bad:
        return 1

    if args.mode == "header":
        return 0

    results = []
    if args.mode in ("seq", "all"):
        results.append(bench_seq(gg, rb, args))
    if args.mode in ("token", "all"):
        results.append(bench_token(gg, t, n_embd, n_rows, rb, args))
    if args.mode in ("batch", "all"):
        b = bench_batch(gg, t, n_embd, n_rows, rb, args)
        results.append(b["contig"])
        results.append(b)
        results.sort(key=lambda x: -x["payload_mb_s"])
    print()
    print(f"  {'pattern':44s} {'s':>8s} {'rows/s':>11s} {'tokens/s':>9s} {'MB/s payload':>12s} {'MB/s device':>11s} "
          f"{'us/read':>8s} {'dequant rows/s':>14s}")
    for r in results:
        tok = f"{r['tokens_s']:.1f}" if "tokens_s" in r else "-"
        dq = f"{r['dequant_rows_s']:,.0f} ({r['dequant_rows']} rows)" if r.get("dequant_rows_s") else r["dequant"]
        print(f"  {r['mode']:44s} {r['seconds']:8.3f} {r['rows_s']:11,.0f} {tok:>9s} {r['payload_mb_s']:12,.1f} "
              f"{r['device_mb_s']:11,.1f} {r['read_us']:8,.1f} {dq:>14s}")
    print(f"\n  note: O_DIRECT {'on' if args.odirect else 'off'}, alignment {args.align} B; "
          f"payload MB/s counts row bytes only, device MB/s what the reads actually asked for")
    print("  the dequant rate is measured apart from the reads (rows/s of the host dequant); the engine's dequant is "
          "the C kernel, not this Python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
