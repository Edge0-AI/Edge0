"""tqenc.py - Python side of the trellis encoder (tools/trellis/tqenc.cpp, built to libtqenc.so by
`sh tools/trellis/build.sh lib`): encode / decode blocks of TQ2_T, TQK6, TQK7 and the experimental TQK5X / TQK4X, the
input-only block Hadamard of the DeepSeek plan, and error metrics.

Rows of a tensor are blocks of 128 consecutive weights along the INPUT dimension (the GGUF fastest axis), exactly as the
decoders read them.  A tensor of shape [rows, n] (n a multiple of 128) is encoded row by row; the output has
rows * n/128 * block_bytes bytes.
"""
from __future__ import annotations

import ctypes
import os
import pathlib

import numpy as np

TQ2_T, TQK6, TQK7, TQK5X, TQK4X, TQK5T, TQK4T = 144, 145, 146, 224, 225, 226, 227
# TQK5X / TQK4X: experimental version 2, the computed (LUT-free) codebook (src/kernels/cpu/tq_computed.hpp);
# TQK5T / TQK4T: version 1, the fork's point table (what the first pilots and the Gyro-S checks used for K5 / K4)
TYPE_NAMES = {TQ2_T: "TQ2_T", TQK6: "TQK6", TQK7: "TQK7", TQK5X: "TQK5X", TQK4X: "TQK4X", TQK5T: "TQK5T", TQK4T: "TQK4T"}
TYPE_BY_NAME = {v: k for k, v in TYPE_NAMES.items()}
KBITS = {TQ2_T: 8, TQK6: 6, TQK7: 7, TQK5X: 5, TQK4X: 4, TQK5T: 5, TQK4T: 4}
COMPUTED = (TQK5X, TQK4X)
BLOCK = 128


def block_bytes(t: int) -> int:
    return 2 + 4 * KBITS[t]


def bpw(t: int) -> float:
    return block_bytes(t) * 8 / BLOCK


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        cands = [os.environ.get("TQENC_LIB", ""),
                 str(pathlib.Path(__file__).resolve().parent.parent / "build-tqenc" / "libtqenc.so"),
                 "libtqenc.so"]
        last = None
        for c in cands:
            if not c:
                continue
            try:
                _lib = ctypes.CDLL(c)
                break
            except OSError as e:
                last = e
        if _lib is None:
            raise RuntimeError("libtqenc.so not found (sh tools/trellis/build.sh lib): %s" % last)
        _lib.tqe_encode.restype = ctypes.c_double
        _lib.tqe_encode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p]
        _lib.tqe_decode.restype = None
        _lib.tqe_decode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    return _lib


_hip = None


class _gpu_lock:
    """TQENC_GPU_LOCK=<card number>: take the box locks (shared /data/gpu.lock, exclusive ~/locks/gpu<N>.lock) around every GPU
    call and release them in between, so a long job that is mostly host work does not keep the card from other users.
    TQENC_GPU_WAIT=<seconds> (default: wait for ever): give up after that long; `ok` is then False and encode() uses the CPU."""

    def __init__(self):
        self.fds = []
        self.ok = True

    def __enter__(self):
        card = os.environ.get("TQENC_GPU_LOCK")
        if card is None:
            return self
        import fcntl
        import time
        wait = float(os.environ.get("TQENC_GPU_WAIT", "1e9"))
        t0 = time.time()
        while os.path.exists(os.path.expanduser("~/locks/EXCL_WAIT")):   # a writer is waiting for the box
            if time.time() - t0 > wait:
                self.ok = False
                return self
            time.sleep(2)
        for path, mode in (("/data/gpu.lock", fcntl.LOCK_SH), (os.path.expanduser("~/locks/gpu%s.lock" % card), fcntl.LOCK_EX)):
            fd = os.open(path, os.O_RDWR | os.O_CREAT)
            while True:
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - t0 > wait:
                        os.close(fd)
                        self.ok = False
                        for f in reversed(self.fds):
                            os.close(f)
                        self.fds = []
                        return self
                    time.sleep(0.2)
            self.fds.append(fd)
        return self

    def __exit__(self, *a):
        for fd in reversed(self.fds):
            os.close(fd)
        return False


def hip_lib() -> ctypes.CDLL:
    """libtqenc_hip.so (sh tools/trellis/build.sh hip; run under the box GPU locks, HIP_VISIBLE_DEVICES picks the card)."""
    global _hip
    if _hip is None:
        cands = [os.environ.get("TQENC_HIP_LIB", ""),
                 str(pathlib.Path(__file__).resolve().parent.parent / "build-tqenc" / "libtqenc_hip.so")]
        last = None
        for c in cands:
            if not c:
                continue
            try:
                _hip = ctypes.CDLL(c)
                break
            except OSError as e:
                last = e
        if _hip is None:
            raise RuntimeError("libtqenc_hip.so not found: %s" % last)
        _hip.tqh_encode.restype = ctypes.c_double
        _hip.tqh_encode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return _hip


def encode(w: np.ndarray, t: int, h: np.ndarray | None = None, iters: int = 1, ncand: int = 1, exact: int = 0,
           scale_tries: int = 1, threads: int = 0, gpu: bool = False):
    """w: float32 array whose last axis is a multiple of 128.  Returns (packed uint8 [nblocks, block_bytes],
    per-block error float64 [nblocks], total error, stats dict).  Error = sum h (w - decoded)^2 (h = 1 if None)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.shape[-1] % BLOCK:
        raise ValueError("last axis %d is not a multiple of 128" % w.shape[-1])
    nb = w.size // BLOCK
    out = np.zeros((nb, block_bytes(t)), dtype=np.uint8)
    err = np.zeros(nb, dtype=np.float64)
    stats = np.zeros(2, dtype=np.int64)
    hp = None
    if h is not None:
        h = np.ascontiguousarray(h, dtype=np.float32)
        if h.shape != w.shape:
            raise ValueError("importance weights must have the shape of w")
        hp = h.ctypes.data
    if gpu:
        if exact or t == TQ2_T:
            raise ValueError("the GPU encoder has no exact search and no TQ2_T")
        with _gpu_lock() as gl:
            if gl.ok:
                tot = hip_lib().tqh_encode(t, w.ctypes.data, hp, nb, out.ctypes.data, iters, ncand, scale_tries, 0,
                                           err.ctypes.data, stats.ctypes.data, None, 0)
                return out, err, tot, {"nonfinite_blocks": int(stats[0]), "zero_blocks": int(stats[1])}
        threads = int(os.environ.get("TQENC_CPU_THREADS", threads or 24))   # the card stayed busy: the CPU encoder, same blocks
    tot = lib().tqe_encode(t, w.ctypes.data, hp, nb, out.ctypes.data, iters, ncand, exact, scale_tries, threads,
                           err.ctypes.data, stats.ctypes.data)
    return out, err, tot, {"nonfinite_blocks": int(stats[0]), "zero_blocks": int(stats[1])}


def decode(packed: np.ndarray, t: int) -> np.ndarray:
    """packed: uint8 array of whole blocks.  Returns float32 [nblocks * 128] (d * codebook)."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    bb = block_bytes(t)
    if packed.size % bb:
        raise ValueError("%d bytes is not a whole number of %s blocks" % (packed.size, TYPE_NAMES[t]))
    nb = packed.size // bb
    out = np.zeros(nb * BLOCK, dtype=np.float32)
    lib().tqe_decode(t, packed.ctypes.data, nb, out.ctypes.data)
    return out


# ---------------------------------------------------------------------------------------------------- Hadamard

def hadamard_matrix(n: int = BLOCK) -> np.ndarray:
    """The normalized Sylvester-Walsh-Hadamard matrix (entry (-1)^popcount(r & c) / sqrt(n)); symmetric, orthonormal."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def make_signs(width: int, seed: int) -> np.ndarray:
    """A reproducible +1/-1 vector (the plan's D).  The vector itself, not the seed, is what a file must store."""
    r = np.random.default_rng(seed)
    return (r.integers(0, 2, size=width) * 2 - 1).astype(np.float32)


def pack_signs(signs: np.ndarray) -> bytes:
    return np.packbits((signs > 0).astype(np.uint8), bitorder="little").tobytes()


def unpack_signs(b: bytes, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")[:width]
    return bits.astype(np.float32) * 2 - 1


def rotate(w: np.ndarray, signs: np.ndarray | None) -> np.ndarray:
    """The input transform U = H D on the last axis, per 128-block: Wrot = W U^T, i.e. each row w -> H (signs * w).
    The same function applied to an activation x gives U x."""
    w = np.asarray(w, dtype=np.float32)
    if signs is not None:
        w = w * signs
    shp = w.shape
    return (w.reshape(-1, BLOCK) @ hadamard_matrix()).reshape(shp)   # H is symmetric


def unrotate(w: np.ndarray, signs: np.ndarray | None) -> np.ndarray:
    """Inverse of rotate (H is its own inverse: orthonormal and symmetric; then the signs)."""
    shp = w.shape
    x = (np.asarray(w, dtype=np.float32).reshape(-1, BLOCK) @ hadamard_matrix()).reshape(shp)
    return x * signs if signs is not None else x


# ---------------------------------------------------------------------------------------------------- metrics

def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    e = np.sum((ref - np.asarray(approx, dtype=np.float64).ravel()) ** 2)
    return float("inf") if e == 0 else float(10 * np.log10(np.sum(ref ** 2) / e))


def rel_mse(ref: np.ndarray, approx: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    return float(np.sum((ref - np.asarray(approx, dtype=np.float64).ravel()) ** 2) / max(np.sum(ref ** 2), 1e-300))


# ---------------------------------------------------------------------------------------------------- .tqx container
# A small versioned file for ONE encoded matrix (research artifacts; not a GGUF).  Little endian:
#   "TQXP" | u32 version (1) | u32 type id | u32 rows | u32 cols | u32 flags (bit 0: input Hadamard, 128-blocks)
#   | u32 n_sign_bytes | sign bits (1 = +1, LSB first, one per input column) | u32 hadamard block (128) | packed blocks
# An unknown version or type is refused before any data is read.
TQX_MAGIC = b"TQXP"
TQX_VERSION = 2   # 2: types 224 / 225 are the computed codebook; a version 1 file means the point table (226 / 227)


def write_tqx(path, t: int, rows: int, cols: int, packed: np.ndarray, signs: np.ndarray | None) -> None:
    import struct
    sb = pack_signs(signs) if signs is not None else b""
    with open(path, "wb") as f:
        f.write(TQX_MAGIC + struct.pack("<IIIIII", TQX_VERSION, t, rows, cols, 1 if signs is not None else 0, len(sb)))
        f.write(sb)
        f.write(struct.pack("<I", BLOCK))
        f.write(np.ascontiguousarray(packed, dtype=np.uint8).tobytes())


def read_tqx(path):
    import struct
    data = open(path, "rb").read()
    if data[:4] != TQX_MAGIC:
        raise ValueError("not a TQXP file")
    ver, t, rows, cols, flags, nsb = struct.unpack_from("<IIIIII", data, 4)
    if ver == 1 and t in (TQK5X, TQK4X):
        t += 2   # a version 1 file: the point table
    elif ver != TQX_VERSION:
        raise ValueError("unsupported TQXP version %d (this reader handles %d, and 1 for the point-table K5 / K4)" % (ver, TQX_VERSION))
    if t not in KBITS:
        raise ValueError("unknown trellis type %d" % t)
    at = 4 + 24
    signs = unpack_signs(data[at:at + nsb], cols) if flags & 1 else None
    at += nsb
    (blk,) = struct.unpack_from("<I", data, at)
    at += 4
    if blk != BLOCK:
        raise ValueError("unsupported Hadamard block %d" % blk)
    nb = rows * (cols // BLOCK)
    packed = np.frombuffer(data, dtype=np.uint8, count=nb * block_bytes(t), offset=at).reshape(nb, block_bytes(t))
    return t, rows, cols, packed, signs
