# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/requant_gguf.py - a DeepSeek-V4.1 GGUF requantized by a recipe (the routed experts in ggml's low-bit types,
the dense matrices in Q8_0), through ggml's own quantizers (libggml-base, ctypes) - a quick test model for maya-ds4's
low-bit expert kernels until the real Maya quants exist.

    python tools/ds4/requant_gguf.py --src ~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL \\
        --out ~/ds4test/Maya-S-test --recipe recipe.json --lib ~/llama.cpp/build/bin/libggml-base.so --jobs 10

The recipe (JSON):
    {"experts": {"gate": "IQ2_XXS", "up": "IQ2_XXS", "down": "IQ3_XXS"},
     "layers": [{"layers": "0-1,38-39", "gate": "IQ2_S", "up": "IQ2_S", "down": "IQ3_S"}],
     "dense": "Q8_0",
     "dense_keep": ["hc_", "engram", "norm", "attn_sinks", "exp_probs_b", "ffn_gate_inp"],
     "engram": "skip"}
- experts: each role's type (gate = ffn_gate_exps, up = ffn_up_exps, down = ffn_down_exps); "layers" entries override it
  for the listed blocks (later entries win).  A role's type may be "keep".
- dense: the type of the other float matrices (2-D or more, F32 / F16 / BF16, rows a multiple of 32), except names
  containing a dense_keep string, which stay as they are; "keep" keeps them all.
- engram: "keep" copies the Engram tables (engram_*) into the output, "skip" leaves them out (a shared Engram shard set
  beside the quant: pack_gguf.py reads <model>/Engram/*.gguf or <model>/../Engram/*.gguf); --engram-link then links the
  source's shards there as that set.

The output has one shard a source shard (the same tensors in the same order, the metadata copied; split.tensors.count
updated), tensors at the source's alignment.  Workers (processes) read the source with O_DIRECT, so the page cache that
running engines rely on is neither evicted nor filled, and drop what they wrote from it; they run at SCHED_IDLE.

IQ2_XXS / IQ2_XS / IQ1_* need an importance matrix: --imatrix (llama.cpp's legacy .dat) or all ones.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import struct
import sys
import time

import numpy as np

# ggml type id -> (name, values a block, bytes a block)
GT = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
      12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
      17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
      21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
      30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 43: ("MXFP8", 32, 33), 67: ("MXFP8", 32, 33)}
BYNAME = {v[0]: k for k, v in GT.items()}
NEED_IMATRIX = {16, 17, 19, 29}
ROLE = {"ffn_gate_exps.weight": "gate", "ffn_up_exps.weight": "up", "ffn_down_exps.weight": "down"}
ALIGN_IO = 4096


def nbytes_of(typ: int, dims: list[int]) -> int:
    _, per, bpb = GT[typ]
    n = 1
    for d in dims:
        n *= d
    return n // per * bpb


class Shard:
    """A GGUF shard: its KV records (raw bytes each, in order) and tensor infos (absolute data offsets)."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        with path.open("rb") as f:
            hdr = f.read(24)
            if hdr[:4] != b"GGUF":
                raise SystemExit("%s: not a GGUF file" % path)
            self.version, nt, nkv = struct.unpack("<IQQ", hdr[4:])
            buf = f.read(64 << 20)   # (the metadata: the vocabulary is a few MB)
            pos = 0

            def take(n):
                nonlocal pos, buf
                while pos + n > len(buf):
                    more = f.read(64 << 20)
                    if not more:
                        raise SystemExit("%s: truncated metadata" % path)
                    buf += more
                v = buf[pos:pos + n]
                pos += n
                return v

            def s():
                return take(struct.unpack("<Q", take(8))[0]).decode("utf-8", "replace")

            fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

            def val(t):
                if t in fmt:
                    return struct.unpack("<" + fmt[t], take(struct.calcsize(fmt[t])))[0]
                if t == 8:
                    return s()
                if t == 9:
                    at, n = struct.unpack("<IQ", take(12))
                    if at in fmt:
                        return list(struct.unpack("<%d%s" % (n, fmt[at]), take(n * struct.calcsize(fmt[at]))))
                    return [val(at) for _ in range(n)]
                raise SystemExit("%s: metadata type %d" % (path, t))

            self.kv, self.kv_raw = {}, []
            for _ in range(nkv):
                p0 = pos
                k = s()
                self.kv[k] = val(struct.unpack("<I", take(4))[0])
                self.kv_raw.append((k, bytes(buf[p0:pos])))
            self.tensors = []
            for _ in range(nt):
                name = s()
                nd = struct.unpack("<I", take(4))[0]
                dims = list(struct.unpack("<%dQ" % nd, take(8 * nd)))
                typ, off = struct.unpack("<IQ", take(12))
                self.tensors.append([name, typ, dims, off])
            self.align = int(self.kv.get("general.alignment", 32))
            start = 24 + pos
            self.data_start = (start + self.align - 1) // self.align * self.align
        for t in self.tensors:
            if t[1] not in GT:
                raise SystemExit("%s: %s has ggml type %d, which this tool does not read" % (path.name, t[0], t[1]))
            t[3] += self.data_start


def kv_int(raw: bytes, v: int) -> bytes:
    """An integer KV record with a new value, in its own type."""
    kl = struct.unpack("<Q", raw[:8])[0]
    t = struct.unpack("<I", raw[8 + kl:12 + kl])[0]
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}[t]
    return raw[:12 + kl] + struct.pack("<" + fmt, v)


def layer_set(spec: str) -> set[int]:
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def plan_type(recipe: dict, name: str, typ: int, dims: list[int]) -> int | None:
    """The output type of a tensor (None: copied as it is, -1: left out)."""
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    short = m.group(2) if m else name
    if short.startswith("engram_") or short.startswith("per_layer_token_embd"):
        return None if recipe.get("engram", "skip") == "keep" else -1
    if m and short in ROLE:
        role = ROLE[short]
        t = recipe.get("experts", {}).get(role, "keep")
        for ov in recipe.get("layers", []):
            if int(m.group(1)) in layer_set(ov["layers"]) and role in ov:
                t = ov[role]
        if t == "keep":
            return None
        if t not in BYNAME:
            raise SystemExit("recipe: unknown type %r" % t)
        q = BYNAME[t]
        if dims[0] % GT[q][1]:
            raise SystemExit("%s: rows of %d are not whole %s blocks" % (name, dims[0], t))
        return q
    dense = recipe.get("dense", "keep")
    if dense == "keep" or typ not in (1, 30) or len(dims) < 2 or dims[0] % 32:   # (F32: the router, mHC - kept)
        return None
    if any(k in short for k in recipe.get("dense_keep", [])):
        return None
    q = BYNAME[dense]
    if dims[0] % GT[q][1]:
        return None
    return q


# ---------------------------------------------------------------- workers
_lib = None
_fds = {}
_imat = None


def _init(lib_path: str, imatrix: str | None):
    global _lib, _imat
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        try:
            os.nice(19)
        except OSError:
            pass
    _lib = ctypes.CDLL(lib_path)
    _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_void_p]
    _lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    _imat = load_imatrix(imatrix) if imatrix else None


def load_imatrix(path: str) -> dict:
    """llama.cpp's legacy imatrix .dat: {tensor name: float32 values (ne0, or ne0 * n_expert for experts)}."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            ln = struct.unpack("<i", f.read(4))[0]
            name = f.read(ln).decode()
            ncall, nval = struct.unpack("<ii", f.read(8))
            v = np.frombuffer(f.read(4 * nval), dtype=np.float32).copy()
            out[name] = v / max(1, ncall)
    return out


def _fd(path: str, direct: bool):
    key = (path, direct)
    if key not in _fds:
        flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if direct else 0)
        try:
            _fds[key] = os.open(path, flags)
        except OSError:
            _fds[key] = os.open(path, os.O_RDONLY)
    return _fds[key]


def read_src(path: str, off: int, n: int) -> memoryview:
    a0 = off // ALIGN_IO * ALIGN_IO
    a1 = (off + n + ALIGN_IO - 1) // ALIGN_IO * ALIGN_IO
    buf = mmap.mmap(-1, a1 - a0)
    try:
        got = os.preadv(_fd(path, True), [buf], a0)
    except OSError:   # (no O_DIRECT here: a plain read)
        got = os.preadv(_fd(path, False), [buf], a0)
    if got < off - a0 + n:
        raise IOError("%s: short read at %d" % (path, off))
    return memoryview(buf)[off - a0:off - a0 + n]


E2M1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def to_f32(typ: int, raw: memoryview, n: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    if typ == 0:
        return np.frombuffer(raw, dtype=np.float32).copy()
    if typ == 1:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if typ == 30:
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if typ == 8:
        blk = b.reshape(-1, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(-1)
    if typ == 39:   # ggml's MXFP4: [e8m0][16 code bytes], value j = kvalues[code] * 2^(e - 128)
        blk = b.reshape(-1, 17)
        e = blk[:, 0].astype(np.int32)
        d = np.ldexp(np.float32(1.0), e - 128).astype(np.float32)
        q = blk[:, 1:]
        v = np.empty((blk.shape[0], 32), dtype=np.float32)
        v[:, :16] = E2M1[q & 15]
        v[:, 16:] = E2M1[q >> 4]
        return (v * d[:, None]).reshape(-1)
    raise ValueError("source type %d" % typ)


def _job(j: dict):
    """One piece of a tensor: rows [r0, r1) (of ne0 values each) quantized, or bytes copied; written in place."""
    t0 = time.time()
    out_fd = os.open(j["out"], os.O_WRONLY)
    try:
        if j["q"] is None:   # a copy, in pieces of 256 MB
            n, done = j["n"], 0
            while done < n:
                k = min(256 << 20, n - done)
                os.pwrite(out_fd, read_src(j["src"], j["src_off"] + done, k), j["out_off"] + done)
                done += k
        else:
            ne0, r0, r1, st, q = j["ne0"], j["r0"], j["r1"], j["typ"], j["q"]
            srb = GT[st][2] * ne0 // GT[st][1]
            raw = read_src(j["src"], j["src_off"] + r0 * srb, (r1 - r0) * srb)
            x = np.ascontiguousarray(to_f32(st, raw, (r1 - r0) * ne0), dtype=np.float32)
            del raw
            qrb = GT[q][2] * ne0 // GT[q][1]
            dst = np.empty((r1 - r0) * qrb, dtype=np.uint8)
            imat = None
            if q in NEED_IMATRIX or (_imat is not None and j["iname"] in _imat):
                w = _imat.get(j["iname"]) if _imat is not None else None
                if w is None or w.size % ne0:
                    w = np.ones(ne0, dtype=np.float32)
                imat = w
            rows_per = j["rows_per_expert"]
            r = r0
            while r < r1:   # (one expert's rows at a time: an expert's own importance row)
                n = min(r1 - r, rows_per - r % rows_per if rows_per else r1 - r)
                im = None
                if imat is not None:
                    e = r // rows_per if rows_per else 0
                    im = imat[e * ne0:(e + 1) * ne0] if imat.size >= (e + 1) * ne0 else imat[:ne0]
                    im = np.ascontiguousarray(im, dtype=np.float32)
                got = _lib.ggml_quantize_chunk(q, x[(r - r0) * ne0:].ctypes.data, dst[(r - r0) * qrb:].ctypes.data,
                                               0, n, ne0, im.ctypes.data if im is not None else None)
                if got != n * qrb:
                    raise RuntimeError("%s: ggml_quantize_chunk wrote %d bytes, expected %d" % (j["name"], got, n * qrb))
                r += n
            os.pwrite(out_fd, memoryview(dst), j["out_off"] + r0 * qrb)
            j["n"] = (r1 - r0) * qrb
            j["out_off"] += r0 * qrb
        os.fdatasync(out_fd)
        try:
            os.posix_fadvise(out_fd, j["out_off"], j["n"], os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    finally:
        os.close(out_fd)
    return j["name"], j["n"], time.time() - t0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the source GGUF's shards (a directory)")
    ap.add_argument("--out", required=True, help="the output directory")
    ap.add_argument("--recipe", required=True, help="the recipe (JSON file)")
    ap.add_argument("--lib", default=os.path.expanduser("~/llama.cpp/build/bin/libggml-base.so"),
                    help="libggml-base (ggml_quantize_chunk)")
    ap.add_argument("--imatrix", help="llama.cpp's legacy imatrix .dat (default: all ones where a type needs one)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--experts-per-job", type=int, default=4)
    ap.add_argument("--engram-link", action="store_true",
                    help="(engram skip) link the source's shards as <out>/Engram/, the shared Engram set")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the output size")
    ap.add_argument("--only", help="regex: quantize only the matching tensors (the rest zero-filled: tests)")
    ap.add_argument("--first", help="regex: the matching tensors' pieces first (e.g. a few layers for a quick test)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the output files of an earlier run (same plan) and skip the pieces already written")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser()
    recipe = json.loads(pathlib.Path(a.recipe).read_text())
    shards = sorted(src.glob("*.gguf"))
    if not shards:
        raise SystemExit("%s: no .gguf files" % src)
    ss = [Shard(p) for p in shards]

    jobs, plan_sizes, outs = [], {}, []
    for si, sh in enumerate(ss):
        kept = []
        for name, typ, dims, off in sh.tensors:
            q = plan_type(recipe, name, typ, dims)
            if q == -1:
                continue
            kept.append((name, typ, dims, off, typ if q is None else q, q))
        # the output's layout: the KV records (split.tensors.count updated), the tensor infos, the data
        kv = []
        for k, raw in sh.kv_raw:
            if k == "split.tensors.count":
                total = sum(1 for s2 in ss for t in s2.tensors if plan_type(recipe, t[0], t[1], t[2]) != -1)
                raw = kv_int(raw, total)
            kv.append(raw)
        tinfo, data_off = [], 0
        for name, typ, dims, off, otyp, q in kept:
            nb = nbytes_of(otyp, dims)
            tinfo.append((name, dims, otyp, data_off, nb))
            data_off += (nb + sh.align - 1) // sh.align * sh.align
            plan_sizes[GT[otyp][0]] = plan_sizes.get(GT[otyp][0], 0) + nb
        head = struct.pack("<4sIQQ", b"GGUF", sh.version, len(tinfo), len(kv)) + b"".join(kv)
        for name, dims, otyp, doff, nb in tinfo:
            nm = name.encode()
            head += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dims))
            head += struct.pack("<%dQ" % len(dims), *dims) + struct.pack("<IQ", otyp, doff)
        dstart = (len(head) + sh.align - 1) // sh.align * sh.align
        opath = out / sh.path.name.replace("UD-Q8_K_XL", "requant")
        outs.append((opath, head, dstart + data_off))
        for (name, typ, dims, off, otyp, q), (_, _, _, doff, nb) in zip(kept, tinfo):
            if a.only and not re.search(a.only, name):
                continue
            base = {"name": name, "src": str(sh.path), "src_off": off, "out": str(opath), "out_off": dstart + doff}
            if q is None:
                jobs.append(dict(base, q=None, n=nb))
                continue
            ne0 = dims[0]
            rows = 1
            for d in dims[1:]:
                rows *= d
            rpe = dims[1] if len(dims) == 3 else 0   # rows an expert (3-D: [n_exp][rows][ne0])
            step = rpe * a.experts_per_job if rpe else max(1, (64 << 20) // (4 * ne0))
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            iname = name
            for r0 in range(0, rows, step):
                jobs.append(dict(base, q=q, typ=typ, ne0=ne0, r0=r0, r1=min(rows, r0 + step), rows_per_expert=rpe,
                                 iname=iname, n=0))
    total = sum(o[2] for o in outs)
    print("plan: %d shards, %.1f GB (%s), %d jobs" % (len(outs), total / 1e9,
          ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(plan_sizes.items())), len(jobs)), flush=True)
    # the free-space check the release runbook promises (DEEPSEEK-RELEASE.MD 9.6 item 5): the plan's bytes against what
    # the output's filesystem has, with 2% of headroom.  It runs in --dry-run too, so the operator sees the verdict
    # before committing hours of CPU.
    probe = out if out.exists() else next((p for p in [out.parent, pathlib.Path("/")] if p.exists()), pathlib.Path("/"))
    du = shutil.disk_usage(probe)
    print("space: %.1f GB planned, %.1f GB free on %s%s" % (total / 1e9, du.free / 1e9, probe,
          "" if du.free * 0.98 >= total else "  <- NOT ENOUGH"), flush=True)
    if a.dry_run:   # nothing is written: report the verdict, and 3 when the space is short (so `--dry-run && requant` works)
        return 0 if du.free * 0.98 >= total else 3
    if a.resume:
        have = sum(p.stat().st_size for p in out.glob("*.gguf") if p.is_file()) if out.exists() else 0
        need = max(0, total - have)
        if du.free * 0.98 < need:
            raise SystemExit("%s: %d GB free, %d GB more needed (%d GB already written) - free space first"
                             % (out, du.free / 1e9, need / 1e9, have / 1e9))
    elif du.free * 0.98 < total:
        raise SystemExit("%s: %d GB free, %d GB needed - free space first (or --out elsewhere)"
                         % (out, du.free / 1e9, total / 1e9))
    out.mkdir(parents=True, exist_ok=True)
    for opath, head, size in outs:
        if a.resume and opath.is_file() and opath.stat().st_size == size:
            with open(opath, "r+b") as f:
                if f.read(len(head)) != head:
                    raise SystemExit("%s: not this plan's file (--resume)" % opath)
            continue
        with open(opath, "wb") as f:
            f.write(head)
            f.truncate(size)
    if a.resume:   # a piece is written when the first and last 256 bytes of its output are not all zero
        def written(j):
            n = j["n"] if j["q"] is None else (j["r1"] - j["r0"]) * GT[j["q"]][2] * j["ne0"] // GT[j["q"]][1]
            o = j["out_off"] + (0 if j["q"] is None else j["r0"] * GT[j["q"]][2] * j["ne0"] // GT[j["q"]][1])
            with open(j["out"], "rb") as f:
                f.seek(o)
                h = f.read(min(256, n))
                f.seek(o + max(0, n - 256))
                t = f.read(min(256, n))
            return any(h) and any(t)
        before = len(jobs)
        jobs = [j for j in jobs if not written(j)]
        print("resume: %d of %d pieces already written" % (before - len(jobs), before), flush=True)
    if a.engram_link and recipe.get("engram", "skip") == "skip":
        eg = out / "Engram"
        eg.mkdir(exist_ok=True)
        for p in shards:
            link = eg / p.name
            if not link.exists():
                link.symlink_to(p)
    # the biggest pieces first (the experts), so the last minutes are not one long tensor
    jobs.sort(key=lambda j: -(j.get("r1", 0) - j.get("r0", 0)) * j.get("ne0", 0) if j["q"] is not None else 0)
    if a.first:
        jobs.sort(key=lambda j: 0 if re.search(a.first, j["name"]) else 1)
    t0, done_b, done_j = time.time(), 0, 0
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(os.path.expanduser(a.lib), a.imatrix)) as pool:
        for name, nb, dt in pool.imap_unordered(_job, jobs):
            done_b += nb
            done_j += 1
            if done_j % 50 == 0 or done_j == len(jobs):
                el = time.time() - t0
                print("%d / %d jobs, %.1f GB written, %.0f s (%.0f min left)" %
                      (done_j, len(jobs), done_b / 1e9, el, el / done_j * (len(jobs) - done_j) / 60), flush=True)
    print("done: %s (%.1f GB) in %.1f min" % (out, total / 1e9, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
