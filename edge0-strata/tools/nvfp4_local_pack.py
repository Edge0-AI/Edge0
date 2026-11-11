"""tools/nvfp4_local_pack.py - NVIDIA's Qwen3.8-Flash-Next-FP4-Local (ModelOpt, QAD) -> files Strata can pack.

    python tools/nvfp4_local_pack.py convert --model <nvidia/Qwen3.8-Flash-Next-FP4-Local dir> --outfile <model.gguf>
    python tools/nvfp4_local_pack.py ple     --model <same dir> --out <ple-nvfp4.gguf>
    python tools/nvfp4_local_pack.py mtp     --model <same dir> --out <mtp src dir>   (then tools/mtp_pack.py, tools/mtp_rt.py)
    python tools/nvfp4_local_pack.py pack    --model <same dir> --dense-gguf <dense.gguf> --out <pack dir>   (instead of iq_pack on a full GGUF)
    python tools/nvfp4_local_pack.py verify  --model <same dir> --gguf <model.gguf> [--experts 6]
    python tools/iq_pack.py --gguf <model.gguf> --out <pack dir>

Nothing of NVIDIA's weights is shipped: the user downloads the checkpoint from NVIDIA's repository and this runs
locally. What the checkpoint stores, and what is done with it:

  * routed experts: NVFP4 (E2M1 codes, E4M3 scale per 16 values, F32 weight_scale_2). The block scales are stored
    in NVIDIA's "CSF" form (config `weight_scale_encoding: "csf"`, codec byte-window4-fixed-stream-u24-exceptions/1):
    per tensor a `.nvfp4_csf_fixed` U8 stream and a `.nvfp4_csf_exceptions` U32 stream. They are decoded here to the
    plain E4M3 matrix (the format is documented by vLLM's loader, PR 60646, and flashinfer's b12x decoder, both
    Apache-2.0; this is a fresh implementation of that description), and from there the ordinary ModelOpt NVFP4
    repack of tools/nvfp4_convert.py applies, bit for bit.
  * attention, GDN projections, shared experts, indexer: MXFP8 (E4M3 values, E8M0 scale per 32 values). They are
    dequantized exactly to BF16 and then take the type policy of tools/nvfp4_convert.py (Q8_0 for the big ones,
    BF16 for the small projections the BF16 kernels read). Q8_0 has the same 32-value blocks; the added error is
    ~0.6% RMS against MXFP8's own ~3%.
  * the 26.8 GiB n-gram (PLE) table is NVFP4 too (160 values a row: 80 B of codes, 10 E4M3 scales, one F32 for the
    whole table). `ple` copies it row by row into a one-tensor GGUF of 90-byte rows.
  * the MTP head (NVFP4 experts, BF16 rest) is dequantized by `mtp` into the folder form tools/mtp_pack.py reads.
    The vision tower is not written (text first).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# #3227: llama.cpp's converter (conversion/base.py: `from transformers import AutoConfig`) and the MXFP8 / NVFP4 decoding below need
# these; setup installs them from requirements-nvfp4-local.txt before a prepare, and a run by hand says which one is missing.
NEEDED_PACKAGES = ("torch", "transformers", "safetensors")
MARKS = ("ple-nvfp4.gguf.done", "dense.gguf.done", "pack.done", "draft.done")


def require_packages() -> None:
    import importlib.util
    missing = [m for m in NEEDED_PACKAGES if importlib.util.find_spec(m) is None]
    if missing:
        req = HERE.parent / "requirements-nvfp4-local.txt"
        sys.exit("tools/nvfp4_local_pack.py needs the Python package%s %s, which %s not installed for %s.\n"
                 "Install %s with:  %s -m pip install -r %s" % ("s" if len(missing) > 1 else "", ", ".join(missing),
                                                                "are" if len(missing) > 1 else "is", sys.executable,
                                                                "them" if len(missing) > 1 else "it", sys.executable, req))


CSF_FIXED = ".nvfp4_csf_fixed"
CSF_EXC = ".nvfp4_csf_exceptions"
PLE_ROW_BYTES = 90          # 80 B of E2M1 codes + 10 E4M3 block scales, per 160-value row


# --------------------------------------------------------------------------------------------- safetensors access
class Checkpoint:
    """Header-indexed, buffered access to a checkpoint's safetensors shards (raw bytes, any dtype).

    Reads go through a cached file handle into a fresh buffer instead of a process-lifetime np.memmap:
    a memmap sweep of the ~64 GB of expert bytes charges every faulted page to the PROCESS WORKING
    SET, which Windows (and every free-RAM observer reading GlobalMemoryStatusEx) counts as used
    until the OS trims it - the pack step then looks like a ~64 GB RAM demand while its real
    committed need is a few GB, and under memory pressure it is the pack run itself that pays
    (issue: the silent exit-1 mid-pack failure).
    A cached sequential read puts the same pages in the system file cache (standby: counted as
    available).  The returned array is a fresh WRITABLE buffer - no consumer writes (a mutator would
    previously have hit the read-only memmap) - and reads share one handle per shard, so raw() is
    not thread-safe, unlike the memmap it replaces.  Each shard is read as three ascending streams
    (exceptions, weights, CSF-fixed) walked expert-ascending."""

    def __init__(self, folder, partial_ok=False):
        self.dir = pathlib.Path(folder)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            wmap = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            files = sorted(set(wmap.values()))
        else:
            files = sorted(p.name for p in self.dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError("no .safetensors files in " + str(self.dir))
        self.meta: dict[str, tuple[str, str, list[int], int, int]] = {}
        self._fh: dict[str, "object"] = {}
        for fn in files:
            path = self.dir / fn
            if not path.exists() and partial_ok:
                continue                                  # a subset of the shards (the MTP head is in two)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.meta[k] = (fn, v["dtype"], v["shape"], 8 + n + a, 8 + n + b)

    def __contains__(self, name):
        return name in self.meta

    def dtype(self, name):
        return self.meta[name][1]

    def shape(self, name):
        return self.meta[name][2]

    def raw(self, name) -> np.ndarray:
        fn, _, _, a, b = self.meta[name]
        f = self._fh.get(fn)
        if f is None:
            f = self._fh[fn] = open(self.dir / fn, "rb")
        buf = np.empty(b - a, dtype=np.uint8)
        f.seek(a)
        if f.readinto(buf) != b - a:
            raise EOFError("%s: shard %s ended early at %d" % (name, fn, a))
        return buf

    def array(self, name, np_dtype) -> np.ndarray:
        return self.raw(name).view(np_dtype).reshape(self.shape(name))


# ----------------------------------------------------------------------------------------------------- CSF decoding
def csf_decode(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """The CSF stream pair -> the plain E4M3 block-scale matrix as uint8 [rows, cols] (cols = K / 16).

    fixed:      uint8 [rows / 16, 16 * (1 + cols / 2)]. Per slab of 16 rows: the 16 rows' base bytes, then each
                row's cols 4-bit codes (two to a byte, low nibble first). A scale byte is base + code.
    exceptions: uint32 words, low 24 bits = flat row * cols + col, high 8 bits = the scale byte there.
    """
    if fixed.dtype != np.uint8 or fixed.ndim != 2 or fixed.shape[1] % 16 or fixed.shape[1] < 32:
        raise ValueError("invalid CSF fixed stream %s %s" % (fixed.dtype, fixed.shape))
    if exceptions.dtype != np.uint32:
        raise ValueError("invalid CSF exceptions dtype %s" % exceptions.dtype)
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    bases = fixed[:, :16].reshape(rows, 1).astype(np.int16)
    if (bases > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).reshape(-1)
    words = exceptions.reshape(-1).astype(np.int64)
    pos = words & 0xFFFFFF
    if pos.size and int(pos.max()) >= rows * cols:
        raise ValueError("CSF exception position out of range")
    out[pos] = words >> 24
    return out.astype(np.uint8).reshape(rows, cols)


def csf_encode(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse, for the tests: uint8 [rows, cols] -> (fixed, exceptions). Base = the row's most common
    window start; whatever falls outside base..base+15 is an exception."""
    rows, cols = scales.shape
    assert rows % 16 == 0 and cols % 2 == 0
    base = np.zeros(rows, np.int16)
    for r in range(rows):
        v = scales[r].astype(np.int16)
        best = (-1, 0)
        for b in range(max(0, int(v.min()) - 0), int(v.max()) + 1):
            c = int(((v >= b) & (v <= b + 15)).sum())
            if c > best[0]:
                best = (c, b)
        base[r] = min(best[1], 240)
    code = scales.astype(np.int16) - base[:, None]
    exc = (code < 0) | (code > 15)
    code = np.where(exc, 0, code).astype(np.uint8)
    pk = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    fixed = np.empty((rows // 16, 16 * (1 + cols // 2)), np.uint8)
    for s in range(rows // 16):
        fixed[s, :16] = base[s * 16:(s + 1) * 16].astype(np.uint8)
        fixed[s, 16:] = pk[s * 16:(s + 1) * 16].reshape(-1)
    pos = np.flatnonzero(exc.reshape(-1))
    words = (pos.astype(np.uint64) | (scales.reshape(-1)[pos].astype(np.uint64) << 24)).astype(np.uint32)
    return fixed, words


# --------------------------------------------------------------------------------------------- MXFP8 dequantizing
def mxfp8_dequant(values_u8: np.ndarray, scales_u8: np.ndarray):
    """MXFP8 (E4M3 values [R, K], E8M0 scale bytes [R, K/32]) -> float32 [R, K]. Exact (4-bit significands)."""
    import torch
    R, K = values_u8.shape
    if scales_u8.shape != (R, K // 32) or K % 32:
        raise ValueError("MXFP8 shapes %s / %s" % (values_u8.shape, scales_u8.shape))
    if (scales_u8 == 255).any():
        raise ValueError("MXFP8 scale byte 255 (NaN) in a weight")
    v = torch.from_numpy(np.ascontiguousarray(values_u8)).view(torch.float8_e4m3fn).to(torch.float32)
    s = torch.from_numpy(np.exp2(scales_u8.astype(np.float32) - 127.0))
    return (v.reshape(R, K // 32, 32) * s[:, :, None]).reshape(R, K)


# --------------------------------------------------------------------------- the overlay on llama.cpp's tensor index
def _orig_name(ck: Checkpoint, name: str):
    """The converter strips `language_model.`; find the checkpoint's own name."""
    if name in ck:
        return name
    for cand in (name.replace("model.", "model.language_model.", 1), "model.language_model." + name):
        if cand in ck:
            return cand
    return None


def overlay(tensors: dict, ck: Checkpoint, stats: dict, dense_only: bool = False) -> dict:
    """Replace the CSF stream pairs by plain `weight_scale` tensors, the MXFP8 weights by BF16, and drop what
    the PLE writer and the MTP packer own."""
    import torch
    out = dict(tensors)
    if dense_only:                                  # the GGUF the engine reads the dense weights from: no experts in it
        out = {k: v for k, v in out.items() if ".mlp.experts." not in k}
    for name in list(out):
        if name not in out:
            continue
        if name.endswith(CSF_FIXED):
            base = name[: -len(CSF_FIXED)]                       # ...weight_scale
            o_fix = _orig_name(ck, name)
            o_exc = _orig_name(ck, base + CSF_EXC)
            if o_fix is None or o_exc is None:
                raise ValueError("CSF fixed stream without its exceptions: " + name)

            def gen(o_fix=o_fix, o_exc=o_exc):
                dec = csf_decode(ck.array(o_fix, np.uint8), ck.array(o_exc, np.uint32))
                return torch.from_numpy(dec).view(torch.float8_e4m3fn)
            del out[name]
            out.pop(base + CSF_EXC, None)
            out[base] = gen
            stats["csf"] = stats.get("csf", 0) + 1
        elif ".ngram_embedding." in name and (name.endswith(".weight_scale") or name.endswith(".weight_scale_2")):
            del out[name]                                         # the PLE table has its own writer (`ple`)
    for name in list(out):
        if not name.endswith(".weight") or ".ngram_embedding." in name:
            continue
        sname = name[: -len(".weight")] + ".weight_scale"
        o_w, o_s = _orig_name(ck, name), _orig_name(ck, sname)
        if o_w is None or o_s is None or sname not in out:
            continue
        if ck.dtype(o_w) == "F8_E4M3" and ck.dtype(o_s) == "U8":     # MXFP8
            def gen(o_w=o_w, o_s=o_s):
                f = mxfp8_dequant(np.asarray(ck.raw(o_w)).reshape(ck.shape(o_w)), np.asarray(ck.raw(o_s)).reshape(ck.shape(o_s)))
                return f.to(torch.bfloat16)
            out[name] = gen
            del out[sname]
            stats["mxfp8"] = stats.get("mxfp8", 0) + 1
    return out


def install(ck: Checkpoint, stats: dict, dense_only: bool = False):
    """Import the converter, give it nvfp4_convert's type policy, and hook the tensor index."""
    import nvfp4_convert as base
    base.install_policy()
    from conversion.base import ModelBase
    from conversion import qwen4exp as Q4
    orig_index = ModelBase.index_tensors

    def index_tensors(self, remote_hf_model_id=None):
        return overlay(orig_index(self, remote_hf_model_id), ck, stats, dense_only)
    ModelBase.index_tensors = index_tensors

    # NVIDIA's repo keeps the shards in a `tensors/` folder that its index names; the converter only lists the top level
    orig_parts = ModelBase.get_model_part_names

    def get_model_part_names(dir_model, prefix, suffix):
        names = orig_parts(dir_model, prefix, suffix)
        sub = pathlib.Path(dir_model) / "tensors"
        if not names and suffix == ".safetensors" and prefix == "model" and sub.is_dir():
            names = sorted("tensors/" + p.name for p in sub.glob("model*.safetensors"))
        return names
    ModelBase.get_model_part_names = staticmethod(get_model_part_names)

    cls = Q4.Qwen4ExpTextModel

    def place_ple_shard(self, data_torch, name):
        # the shards are NVFP4 here: two values a byte, so a row of 160 values is 80 bytes
        idx = int(name.rsplit("shard_", 1)[1].split(".", 1)[0])
        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1]) * 2
        base.dropped_ple.append(name)
        return []
    cls._place_ple_shard = place_ple_shard
    return base


def cmd_convert(a) -> int:
    require_packages()
    model = pathlib.Path(a.model)
    ck = Checkpoint(model)
    stats: dict = {}
    base = install(ck, stats, a.dense_only)
    import convert_hf_to_gguf as C
    out = pathlib.Path(a.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(model), "--outtype", "q8_0", "--outfile", str(out)] + (["--dry-run"] if a.dry_run else [])
    sys.argv = ["convert_hf_to_gguf.py"] + argv
    t0 = time.time()
    rc = C.main()
    took = time.time() - t0
    print("nvfp4_local_pack: done in %.0f s; CSF scale tensors decoded %d, MXFP8 weights dequantized %d, PLE shards dropped %d"
          % (took, stats.get("csf", 0), stats.get("mxfp8", 0), len(base.dropped_ple)))
    if not a.dry_run:
        out.with_suffix(".manifest.json").write_text(json.dumps({
            "source": str(model), "csf_decoded": stats.get("csf", 0), "mxfp8_dequantized": stats.get("mxfp8", 0),
            "ple_table": "written separately: nvfp4_local_pack.py ple", "mtp": "not written", "seconds": round(took, 1)},
            indent=2) + "\n", encoding="utf-8")
    return rc or 0


# ------------------------------------------------------------------------------------------------- the PLE table
GGUF_TYPE_STRING, GGUF_TYPE_F32 = 8, 6
GGML_TYPE_I8 = 24
ALIGN = 32


def _gs(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def interleave_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """[n, 80] codes + [n, 10] E4M3 scale bytes -> [n, 90]: the codes, then the scales."""
    return np.concatenate([codes, scales], axis=1)


def cmd_ple(a) -> int:
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    ck = Checkpoint(model, partial_ok=True)           # the table is in six of the shards; a folder with only those works
    shards = {}
    for name in ck.meta:
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            shards[int(name.rsplit(".shard_", 1)[1].split(".")[0])] = name
    if not shards or sorted(shards) != list(range(len(shards))):
        sys.exit("the n-gram shards are missing or not numbered 0..N-1")
    g = [k for k in ck.meta if k.endswith(".ngram_embedding.weight_scale_2")]
    if len(g) != 1:
        sys.exit("expected one ngram_embedding.weight_scale_2")
    scale2 = float(ck.array(g[0], np.float32).reshape(-1)[0])
    rows, dim = 0, None
    for k in range(len(shards)):
        n = shards[k]
        r, c = ck.shape(n)
        if ck.dtype(n) != "U8" or c != 80 or ck.shape(n[:-6] + "weight_scale") != [r, 10]:
            sys.exit("%s: not an NVFP4 160-value row table" % n)
        rows += r
    print("%d shards, %d rows x 160, global scale %.9g, %.2f GB" % (len(shards), rows, scale2, rows * PLE_ROW_BYTES / 1e9), flush=True)
    kvs = (_gs("general.architecture") + struct.pack("<I", GGUF_TYPE_STRING) + _gs("strata-ple")
           + _gs("general.name") + struct.pack("<I", GGUF_TYPE_STRING) + _gs("PLE n-gram table, NVFP4 as shipped")
           + _gs("strata.ple.format") + struct.pack("<I", GGUF_TYPE_STRING) + _gs("nvfp4x16")
           + _gs("strata.ple.scale") + struct.pack("<I", GGUF_TYPE_F32) + struct.pack("<f", scale2)
           + _gs("strata.ple.source") + struct.pack("<I", GGUF_TYPE_STRING) + _gs(model.name))
    # the tensor is [90, rows] I8 so one row is 90 contiguous bytes; the engine reads it by the format key
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, 5) + kvs
    head += _gs("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", PLE_ROW_BYTES, rows)
    head += struct.pack("<I", GGML_TYPE_I8) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    CH = 1 << 20
    with open(tmp, "wb") as w:
        w.write(head)
        for k in range(len(shards)):
            n = shards[k]
            codes = ck.array(n, np.uint8)
            sc = ck.array(n[:-6] + "weight_scale", np.uint8)
            for i in range(0, codes.shape[0], CH):
                w.write(np.ascontiguousarray(interleave_rows(codes[i:i + CH], sc[i:i + CH])).tobytes())
            if k % 16 == 15 or k == len(shards) - 1:
                print("  %3d/%d shards" % (k + 1, len(shards)), flush=True)
    if tmp.stat().st_size != len(head) + rows * PLE_ROW_BYTES:
        sys.exit("size check failed")
    tmp.replace(out)
    print("wrote %s (%d rows x %d B)" % (out, rows, PLE_ROW_BYTES))
    return 0


# ------------------------------------------------------------------------------------------------------- the MTP head
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table() -> np.ndarray:
    x = np.arange(256)
    s, e, m = x >> 7, (x >> 3) & 15, x & 7
    f = np.where(e == 0, np.ldexp(m / 8.0, -6), np.ldexp(1.0 + m / 8.0, e.astype(np.int32) - 7))
    f = np.where((e == 15) & (m == 7), np.nan, f)
    return np.where(s == 1, -f, f).astype(np.float32)


def nvfp4_dequant(codes: np.ndarray, scales_u8: np.ndarray, scale2: float) -> np.ndarray:
    """ModelOpt NVFP4 -> float32: codes [R, K/2] (low nibble first), E4M3 scale bytes [R, K/16], one F32."""
    R = codes.shape[0]
    lo, hi = codes & 15, codes >> 4
    v = E2M1[np.stack([lo, hi], axis=-1).reshape(R, -1)]
    sc = e4m3_table()[scales_u8]
    return (v.reshape(R, -1, 16) * sc[:, :, None] * np.float32(scale2)).reshape(R, -1)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = u + 0x7FFF + ((u >> 16) & 1)               # round to nearest even
    return (r >> 16).astype(np.uint16)


def cmd_mtp(a) -> int:
    """The MTP head in tools/mtp_fetch.py's folder form (raw BF16 files + mtp-manifest.json), so tools/mtp_pack.py and
    tools/mtp_rt.py take it as they take Qwen's own: the 31 tensors, experts fused as gate_up_proj / down_proj."""
    import hashlib
    ck = Checkpoint(a.model, partial_ok=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def put(name, bits, shape):
        fn = name.replace("/", "_") + ".bin"
        data = np.ascontiguousarray(bits, np.uint16)
        (out / fn).write_bytes(data.tobytes())
        manifest.append({"name": name, "dtype": "BF16", "shape": list(shape), "file": fn,
                         "sha256": hashlib.sha256(data.tobytes()).hexdigest()})

    for name in sorted(k for k in ck.meta if k.startswith("mtp.") and ".experts." not in k):
        if ck.dtype(name) != "BF16":
            sys.exit("%s is %s, expected BF16" % (name, ck.dtype(name)))
        put(name, ck.array(name, np.uint16), ck.shape(name))
    n_exp = 512
    for fused, parts in (("mtp.layers.0.mlp.experts.gate_up_proj", ("gate_proj", "up_proj")),
                         ("mtp.layers.0.mlp.experts.down_proj", ("down_proj",))):
        rows = [ck.shape("mtp.layers.0.mlp.experts.0.%s.weight" % p)[0] for p in parts]
        cols = ck.shape("mtp.layers.0.mlp.experts.0.%s.weight" % parts[0])[1] * 2
        shape = [n_exp, sum(rows), cols]
        fn = fused + ".bin"
        with open(out / fn, "wb") as f:
            h = hashlib.sha256()
            for e in range(n_exp):
                mats = []
                for p in parts:
                    b = "mtp.layers.0.mlp.experts.%d.%s." % (e, p)
                    mats.append(nvfp4_dequant(np.asarray(ck.array(b + "weight", np.uint8)),
                                              np.asarray(ck.array(b + "weight_scale", np.uint8)),
                                              float(ck.array(b + "weight_scale_2", np.float32).reshape(-1)[0])))
                blob = f32_to_bf16_bits(np.concatenate(mats, axis=0)).tobytes()
                f.write(blob)
                h.update(blob)
        manifest.append({"name": fused, "dtype": "BF16", "shape": shape, "file": fn, "sha256": h.hexdigest()})
        print("  %s %s" % (fused, shape), flush=True)
    (out / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote %d MTP tensors to %s" % (len(manifest), out))
    return 0


# ------------------------------------------------------------------------------------------------------ the slim GGUF
def cmd_slim(a) -> int:
    """Copy a converted GGUF without its routed experts (and their .scale / .input_scale tensors): the file the engine
    reads the dense weights from (--native / --native-dense-gguf), ~9 GB instead of ~74 GB, once experts.bin is cut."""
    from _paths import add_gguf_py
    add_gguf_py()
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType as VT
    rd = GGUFReader(a.gguf)
    arch = rd.fields["general.architecture"].contents()
    w = GGUFWriter(a.out, arch)
    for key, f in rd.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vt = f.types[0]
        w.add_key_value(key, f.contents(), vt, sub_type=f.types[-1] if vt == VT.ARRAY else None)
    kept = skipped = 0
    for t in rd.tensors:
        if "_exps." in t.name:
            skipped += 1
            continue
        data = np.asarray(t.data)
        qt = t.tensor_type
        if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            w.add_tensor(t.name, data)
        else:                                   # raw bytes with the type: the quantized blocks (BF16 rides as uint16)
            raw = np.ascontiguousarray(data).view(np.uint8)
            w.add_tensor(t.name, raw, raw_dtype=qt, raw_shape=[int(x) for x in raw.shape])
        kept += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("slim: %d tensors kept, %d expert tensors left out -> %s" % (kept, skipped, a.out))
    return 0


# ------------------------------------------------------------------------------------------- the pack, without a big GGUF
def nvfp4_blocks(weight_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """ModelOpt NVFP4 rows -> ggml's block_nvfp4 rows: weight [R, K/2] (two E2M1 codes a byte, element 2i low), scale [R, K/16]
    E4M3 bytes -> [R, K/64 * 36] bytes of super-blocks `[4 UE4M3 scales | 32 code bytes]`. The same bits as llama.cpp's
    ModelBase._nvfp4_pack (the scale's sign bit is dropped; it is 0 in every scale of this checkpoint)."""
    R, nb = scale_u8.shape
    w = weight_u8.reshape(R, nb, 8)
    vals = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(R, nb, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8)
    d = (scale_u8 & 0x7F).reshape(R, nb // 4, 4)
    qs = qs.reshape(R, nb // 4, 32)
    return np.concatenate([d, qs], axis=-1).reshape(R, (nb // 4) * 36)


class _VTensor:
    """A tensor info for iq_pack: an expert tensor (or its scale) whose bytes come from the checkpoint."""

    def __init__(self, name, shape, type_id, type_name, nbytes):
        self.name, self.shape, self.type_id, self.type_name, self.offset = name, shape, type_id, type_name, 0
        self._n = nbytes

    def expected_bytes(self):
        return self._n


class _VGguf:
    data_start = 0


def experts_from_checkpoint(model, ck: Checkpoint, n_layers: int = 48, n_exp: int = 512):
    """Give an iq_pack.Model of the dense-only GGUF the routed experts of the checkpoint as virtual tensors."""
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q
    src = next(iter(model.where.values()))[3]
    cache: dict = {}
    base = "model.language_model.layers.%d.mlp.experts.%d.%s_proj."

    def layer_role(l, role):
        key = (l, role)
        if key not in cache:
            cache.clear()
            rows = []
            for e in range(n_exp):
                b = base % (l, e, role)
                dec = csf_decode(np.asarray(ck.array(b + "weight_scale" + CSF_FIXED, np.uint8)),
                                 np.asarray(ck.array(b + "weight_scale" + CSF_EXC, np.uint32)))
                rows.append(nvfp4_blocks(np.asarray(ck.array(b + "weight", np.uint8)), dec).reshape(-1))
            cache[key] = (np.stack(rows),)
        return cache[key]

    # shapes in ggml order (ne0 = K, ne1 = rows, ne2 = experts)
    geo = {"gate": (2560, 640), "up": (2560, 640), "down": (640, 2560)}

    class _Mem(_VGguf):                       # a tiny tensor held in memory: iq_pack reads it like any dense one
        pass

    for l in range(n_layers):
        for role, (k, rows) in geo.items():
            n = "blk.%d.ffn_%s_exps.weight" % (l, role)
            vt = _VTensor(n, [k, rows, n_exp], Q.NVFP4.value, "NVFP4", n_exp * rows * (k // 64) * 36)
            model.where[n] = (_VGguf(), vt, None, src)
            # the per-expert F32 scales (weight_scale_2 -> .scale, input_scale -> .input_scale), as the converter writes them
            for suffix, key in ((".scale", "weight_scale_2"), (".input_scale", "input_scale")):
                vals = np.array([ck.array(base % (l, e, role) + key, np.float32).reshape(-1)[0] for e in range(n_exp)], np.float32)
                sn = "blk.%d.ffn_%s_exps%s" % (l, role, suffix)
                raw = vals.view(np.uint8)
                model.where[sn] = (_Mem(), _VTensor(sn, [n_exp], Q.F32.value, "F32", n_exp * 4), raw, src)
    orig_bytes = type(model).bytes

    def bytes_(name):
        if name.startswith("blk.") and "_exps." in name:
            if not name.endswith(".weight"):                   # the small F32 scale tensors are in memory
                return model.where[name][2]
            l = int(name.split(".")[1]); role = name.split("ffn_")[1].split("_exps")[0]
            return layer_role(l, role)[0].reshape(-1)
        return orig_bytes(model, name)
    model.bytes = bytes_
    return model


def cmd_pack(a) -> int:
    """tools/iq_pack.py on the dense-only GGUF, with the experts read from the checkpoint: experts.bin, native_experts.txt,
    dense.bin, index.txt and the tokenizer, with no 74 GB GGUF in between."""
    import iq_pack
    ck = Checkpoint(a.model)
    orig_model = iq_pack.Model

    class M(orig_model):
        def __init__(self, first):
            super().__init__(first)
            experts_from_checkpoint(self, ck)
    iq_pack.Model = M
    sys.argv = ["iq_pack.py", "--gguf", a.dense_gguf, "--out", a.out, "--experts-bin"]
    return iq_pack.main()


# ------------------------------------------------------------------------------------------------------- prepare (setup)
def cmd_prepare(a) -> int:
    """Everything setup needs from the downloaded checkpoint, each step skipped when its mark is there:
        <out>/ple-nvfp4.gguf   the n-gram table      <out>/dense.gguf   the dense weights (no experts)
        <out>/pack/            index.txt, dense.bin, experts.bin, native_experts.txt, tokenizer/
        <out>/mtp/rt/          the draft layer (a Q2_0 copy of NVIDIA's MTP experts)
    The checkpoint is read, never changed; it can be deleted afterwards (the pack and the two GGUFs hold everything)."""
    import subprocess
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    if not all((out / m).is_file() for m in MARKS):
        require_packages()                          # before the 20 minutes of the n-gram table, not after them (#3227)
    out.mkdir(parents=True, exist_ok=True)
    me = [sys.executable, str(pathlib.Path(__file__).resolve())]
    tools = pathlib.Path(__file__).resolve().parent

    def run(cmd):
        r = subprocess.run(cmd)
        if r.returncode:
            sys.exit("prepare: %s failed (exit %d)" % (cmd[1] if len(cmd) > 1 else cmd[0], r.returncode))

    def step(name, mark, fn):
        if mark.exists():
            print("prepare: %s already done" % name, flush=True)
            return
        print("prepare: %s ..." % name, flush=True)
        t0 = time.time()
        fn()
        mark.write_text("done %.0f s\n" % (time.time() - t0))

    step("the n-gram table", out / "ple-nvfp4.gguf.done",
         lambda: run(me + ["ple", "--model", str(model), "--out", str(out / "ple-nvfp4.gguf")]))
    step("the dense weights", out / "dense.gguf.done",
         lambda: run(me + ["convert", "--dense-only", "--model", str(model), "--outfile", str(out / "dense.gguf")]))
    step("the experts and the pack", out / "pack.done",           # (native_experts.txt is the pack's own marker, not ours)
         lambda: run(me + ["pack", "--model", str(model), "--dense-gguf", str(out / "dense.gguf"), "--out", str(out / "pack")]))

    mtp = out / "mtp"

    def draft_layer():
        src = mtp / "src"
        run(me + ["mtp", "--model", str(model), "--out", str(src)])
        run([sys.executable, str(tools / "mtp_pack.py"), "--src", str(src), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")])
        run([sys.executable, str(tools / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(mtp / "rt")])
        import shutil
        shutil.rmtree(src, ignore_errors=True)
    step("the draft layer", out / "draft.done", draft_layer)
    print("prepare: done -> %s" % out)
    return 0

# ------------------------------------------------------------------------------------------------------ verifying
def cmd_verify(a) -> int:
    """Experts: decode the checkpoint (CSF + ModelOpt definition) and compare with gguf-py's dequantization of the
    GGUF's NVFP4 blocks times the scale tensor, bit for bit (zeros normalised: ggml has one zero)."""
    import torch
    import nvfp4_verify as V
    from gguf import GGUFReader, GGMLQuantizationType as Q, quants
    ck = Checkpoint(a.model)
    rd = GGUFReader(a.gguf)
    T = {t.name: t for t in rd.tensors}
    bad = 0
    rng = np.random.default_rng(1)
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in ck.meta if ".mlp.experts." in k and k.startswith("model.language_model")})
    for _ in range(a.experts):
        L = int(rng.choice(layers)); e = int(rng.integers(0, 512))
        for proj, gname in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
            b = "model.language_model.layers.%d.mlp.experts.%d.%s." % (L, e, proj)
            w = torch.from_numpy(np.array(ck.array(b + "weight", np.uint8)))
            sc = csf_decode(ck.array(b + "weight_scale" + CSF_FIXED, np.uint8), ck.array(b + "weight_scale" + CSF_EXC, np.uint32))
            s = torch.from_numpy(sc).view(torch.float8_e4m3fn)
            s2 = torch.from_numpy(np.array(ck.array(b + "weight_scale_2", np.float32))).reshape(())
            ref = V.modelopt_dequant(w, s, s2)
            t = T["blk.%d.%s.weight" % (L, gname)]
            blob = np.asarray(t.data)                              # [512, rows, bytes] of NVFP4 blocks
            got = quants.dequantize(np.ascontiguousarray(blob[e]), Q.NVFP4).astype(np.float32)
            sct = T["blk.%d.%s.scale" % (L, gname)]
            got = got * np.float32(np.asarray(sct.data)[e])
            same = np.array_equal((ref + np.float32(0)).view(np.uint32), (got.reshape(ref.shape) + np.float32(0)).view(np.uint32))
            print("  layer %2d expert %3d %-9s bitwise %s" % (L, e, proj, "OK" if same else "NO"), flush=True)
            bad += not same
    print("RESULT:", "lossless" if not bad else "MISMATCH")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--model", required=True)
    c.add_argument("--outfile", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--dense-only", action="store_true",
                   help="leave the routed experts out: the GGUF that --native-dense-gguf reads (~9 GB) beside an experts.bin")
    p = sub.add_parser("ple")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    m = sub.add_parser("mtp")
    m.add_argument("--model", required=True)
    m.add_argument("--out", required=True)
    pk = sub.add_parser("pack")
    pk.add_argument("--model", required=True)
    pk.add_argument("--dense-gguf", required=True, help="the output of `convert --dense-only` (or `slim`)")
    pk.add_argument("--out", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--model", required=True)
    pr.add_argument("--out", required=True)
    sl = sub.add_parser("slim")
    sl.add_argument("--gguf", required=True)
    sl.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--model", required=True)
    v.add_argument("--gguf", required=True)
    v.add_argument("--experts", type=int, default=4)
    a = ap.parse_args()
    return {"convert": cmd_convert, "ple": cmd_ple, "mtp": cmd_mtp, "slim": cmd_slim, "pack": cmd_pack, "prepare": cmd_prepare, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
