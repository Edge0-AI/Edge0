"""Strata P1.S6/S7 - the pack builder and verifier.

This is `strata-pack build` / `strata-pack verify` / `strata-pack info`.  It is the Python implementation of
`docs/pack-format.md`; the C++ `tools/strata-pack` is a port of it, and if the two disagree the C++ is wrong.

WHY PYTHON FIRST.  The canonicalisation in `tools/canonical_xcheck.py` is measured bit-exact over every block
of every tensor (757,139,840 blocks across 1021 tensors, plus Q2_0), against gguf-py and, for Q2_0, backed by
`dequant_xcheck` against ggml itself.  That makes the Python mapping table the specification of record.  The
pack is Phase 2's prerequisite and can be produced from it today; the port then has a golden artifact to be
tested against instead of being the thing everything else waits on.

THE ONE RULE.  A pack is only valid if, for every tensor, `decode(pack) == dequant_ggml(source)` BIT FOR BIT.
`verify` is that test and it is not optional - it re-reads the written pack from disk and compares it against
the source shard.  A pack that has not passed `verify` does not exist.

SCALE PRECISION.  A canonical scale is an FP32 product (`fl(d*sc)`), which is generally NOT representable in
FP16.  The packer therefore stores scales as FP16 only when the round-trip is exact and as FP32 otherwise, and
records which per tensor.  It never has to decide correctly: `verify` re-checks bit-exactness, so a wrong
choice is a failed build rather than a silently lossy pack.  The expert arena is the exception and is FP16 by
construction - architecture §3.2 fixes the blob at exactly 1,382,400 bytes and pack_layer.py asserts it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

import gguf_reader as G                                                    # noqa: E402
import pack_layer as PL                                                    # noqa: E402
from canonical_xcheck import (MAPPINGS, KV_IQ4NL, open_shard,             # noqa: E402
                              data_section_offset, reference_values)

FORMAT_VERSION = 1
ALIGN = 64                       # every plane starts on a 64-byte boundary (P1.S7)
_HERE = pathlib.Path(__file__).resolve()
STRATA = _HERE.parents[min(3, len(_HERE.parents) - 1)]   # the development layout; build takes the shard from --gguf
SHARD2 = STRATA / "Q2_0" / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"

EXPERT_RE = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
EMBD_NAME = "token_embd.weight"


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(EXPERT_RE)


def code_bits(form: str) -> int:
    return {"S2": 2, "S4": 4, "S8": 8}[form]


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Pack one code per element into bytes, LSB-first, `8//bits` codes per byte.

    This is Q2_0's own convention (4 codes per byte, byte j/4, bits (j%4)*2), kept for every form so the
    kernel has one unpacking loop rather than one per source type.
    """
    flat = codes.reshape(-1).astype(np.uint8)
    if bits == 8:
        return flat.tobytes()
    per = 8 // bits
    if flat.size % per:
        raise ValueError("code count %d is not a multiple of %d per byte" % (flat.size, per))
    g = flat.reshape(-1, per).astype(np.uint16)
    out = np.zeros(g.shape[0], dtype=np.uint16)
    for k in range(per):
        out |= (g[:, k] & ((1 << bits) - 1)) << (k * bits)
    return out.astype(np.uint8).tobytes()


def unpack_codes(buf: bytes, bits: int, n: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(buf, dtype=np.uint8)[:n].copy()
    per = 8 // bits
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty((b.size, per), dtype=np.uint8)
    for k in range(per):
        out[:, k] = (b >> (k * bits)) & ((1 << bits) - 1)
    return out.reshape(-1)[:n].copy()


def choose_scale_dtype(scales: np.ndarray) -> tuple[np.ndarray, bool]:
    """FP16 when the round-trip is exact, else FP32.  Returns (bytes, is_fp16)."""
    as16 = scales.astype(np.float16)
    if np.array_equal(as16.astype(np.float32), scales):
        return as16.tobytes(), True
    return scales.astype("<f4").tobytes(), False


def canonicalise(name: str, tname: str, raw: bytes):
    """Source bytes -> the canonical planes, as (codes, scales, offsets_or_None, mapping)."""
    m = MAPPINGS[tname]
    parts = m.enc(raw)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], m
    return parts[0], parts[1], None, m


def tensor_entry(name, meta, tname, raw, seg_offset):
    """Encode one tensor and return (blob_bytes, manifest_entry)."""
    m = MAPPINGS[tname]

    body = bytearray()
    base = seg_offset

    def put(b: bytes) -> dict:
        nonlocal body
        pad = (-len(body)) % ALIGN
        body += b"\0" * pad
        off = base + len(body)
        body += b
        return {"offset": off, "bytes": len(b)}

    common = {
        "name": name, "source_type": tname, "form": m.form, "codebook": m.codebook,
        "code_bias": m.code_bias, "has_offset": m.has_offset,
        "block_elems": m.block_elems, "block_bytes": m.block_bytes, "group_elems": m.group_elems,
        "shape": list(meta.shape), "source_offset": meta.offset,
    }

    # ---- P16/P32 are pass-through: one value plane, no codes, no scale, no offset
    if m.form in ("P16", "P32"):
        vals = np.ascontiguousarray(m.dec(m.enc(raw)), dtype=np.float32).reshape(-1)
        if m.form == "P32":
            vb, v16 = vals.astype("<f4").tobytes(), False
        else:
            a16 = vals.astype(np.float16)
            if not np.array_equal(a16.astype(np.float32), vals):
                raise ValueError("%s: P16 would lose precision, so this tensor needs P32" % name)
            vb, v16 = a16.tobytes(), True
        ent = dict(common, elements=int(vals.size), values_fp16=v16)
        ent["values"] = put(vb)
        return bytes(body), ent

    codes, scales, offsets, m = canonicalise(name, tname, raw)
    codes = np.ascontiguousarray(codes)
    scales = np.ascontiguousarray(scales.reshape(-1), dtype=np.float32)
    n_el = int(codes.size)
    bits = code_bits(m.form)
    codes_b = pack_codes(codes, bits)
    scales_b, sc16 = choose_scale_dtype(scales)

    ent = dict(common, elements=n_el, code_bits=bits)
    ent["codes"] = put(codes_b)
    ent["scales"] = put(scales_b)
    ent["scales_fp16"] = sc16
    if offsets is not None:
        ob, o16 = choose_scale_dtype(np.ascontiguousarray(offsets.reshape(-1), dtype=np.float32))
        ent["offsets"] = put(ob)
        ent["offsets_fp16"] = o16
    return bytes(body), ent


def decode_entry(head: bytes, ent: dict) -> np.ndarray:
    """Decode a tensor from the pack, using ONLY what the manifest says.

    Deliberately manifest-driven: if the manifest does not carry enough to decode the tensor, this cannot
    silently fall back on knowledge of the source type, and the omission becomes a failure here.
    """
    if ent["form"] in ("P16", "P32"):
        vb = head[ent["values"]["offset"]: ent["values"]["offset"] + ent["values"]["bytes"]]
        return np.frombuffer(vb, dtype="<f2" if ent["values_fp16"] else "<f4").astype(np.float32)

    n = ent["elements"]
    codes = unpack_codes(head[ent["codes"]["offset"]: ent["codes"]["offset"] + ent["codes"]["bytes"]],
                         ent["code_bits"], n)
    sc = head[ent["scales"]["offset"]: ent["scales"]["offset"] + ent["scales"]["bytes"]]
    sc = np.frombuffer(sc, dtype="<f2" if ent["scales_fp16"] else "<f4").astype(np.float32)
    if "offsets" in ent:
        ob = head[ent["offsets"]["offset"]: ent["offsets"]["offset"] + ent["offsets"]["bytes"]]
        off = np.frombuffer(ob, dtype="<f2" if ent.get("offsets_fp16") else "<f4").astype(np.float32)
    else:
        off = None
    if ent["codebook"] == "IQ4NL":
        v = KV_IQ4NL[codes]
    else:
        v = (codes.astype(np.int32) + ent["code_bias"]).astype(np.float32)
    g = ent["group_elems"]
    if g == 1:                       # one scale for the whole tensor
        return (v * sc[0]).reshape(-1) if off is None else (v * sc[0] + off[0]).reshape(-1)
    out = v.reshape(-1, g) * sc[:, None]
    if off is not None:
        out = out + off[:, None]
    return out.reshape(-1)


def sha256(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# The files the engine reads a pack's identity from (src/core/expert_source.cpp, the fingerprint): one of them in
# --out means the directory already holds a pack, this tool's or iq_pack.py's.  experts.bin on its own does not:
# build writes manifest.json last, so experts.bin without it is a build that did not finish, and is rebuilt.
PACK_MARKERS = ("manifest.json", "index.txt", "native_experts.txt")


def occupied(out_dir: pathlib.Path) -> pathlib.Path | None:
    """The pack marker already in `out_dir`, or None when the directory is empty, absent or holds no pack."""
    for name in PACK_MARKERS:
        if (out_dir / name).exists():
            return out_dir / name
    return None


def build(gguf: pathlib.Path, out_dir: pathlib.Path, n_layers: int | None, skip_hash: bool, force: bool = False) -> int:
    # refused before anything is opened or written: build used to mkdir(exist_ok=True) and overwrite experts.bin,
    # dense.bin, embd.bin and manifest.json of whatever pack was there, with nothing said
    marker = None if force else occupied(out_dir)   # --force: setup rebuilding a pack that lost index.txt or experts.bin
    if marker is not None:
        print("%s already holds a pack (%s is there) - refusing to build; delete the directory, pass another "
              "--out, or --force to rebuild it" % (out_dir, marker.name))
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        # the old pack's markers go before a byte is written, so a forced build that stops part-way leaves an
        # unfinished build (experts.bin without a manifest.json), which the next build or setup makes again - not
        # the old manifest.json and index.txt over partly new experts, which setup's "index.txt and experts.bin are
        # there" check would hand to the engine.  The data files need no removing: the build rewrites each one.
        for name in PACK_MARKERS:
            (out_dir / name).unlink(missing_ok=True)
    g = G.GGUFFile(gguf)
    head, flen = open_shard(gguf)
    data_off = data_section_offset(g, flen)
    print("source %s: %d tensors, data at %d" % (gguf.name, len(g.tensors), data_off))

    n_layers = 48 if n_layers is None else n_layers
    man = {
        "format": "strata-pack", "format_version": FORMAT_VERSION,
        "source": {"shard1": str(gguf), "shard2": str(SHARD2)},
        "n_layers": n_layers, "align": ALIGN,
        "codebooks": {"IQ4NL": [float(x) for x in KV_IQ4NL]},
        "tensors": {}, "experts": {}, "n_experts_per_layer": PL.NE,
    }

    # ---- experts.bin: delegated to pack_layer so the blob layout has exactly ONE implementation
    # pack_layer reads its own hardcoded SHARD1, so a different --gguf would silently take the expert arena
    # from a different file than the dense tensors.  Refuse rather than produce a mixed-source pack.
    # the expert arena and the dense tensors come from the same file: point pack_layer at it
    PL.SHARD1 = pathlib.Path(gguf).resolve()
    shard2 = pathlib.Path(str(PL.SHARD1).replace("00001-of-00002", "00002-of-00002"))
    man["source"]["shard2"] = str(shard2)
    exp_path = out_dir / "experts.bin"
    print("experts.bin: %d layers, %.2f GiB" % (n_layers, n_layers * PL.NE * PL.BLOB_BYTES / 2**30))
    em = PL.build(n_layers, exp_path)
    man["experts"] = {"blob_bytes": em["blob_bytes"], "offsets": em["offsets"],
                      "layers": em["layers"], "total_bytes": em["total_bytes"],
                      "scales_fp16": True, "source_type": "Q2_0", "form": "S2"}

    # ---- dense.bin and embd.bin
    dense = []
    embd = []
    for t in g.tensors:
        if is_expert(t.name):
            continue
        (embd if t.name == EMBD_NAME else dense).append(t)

    for fname, group in (("dense.bin", dense), ("embd.bin", embd)):
        path = out_dir / fname
        seg_base = 0
        with open(path, "wb") as fh:
            for t in group:
                if t.type_name not in MAPPINGS:
                    print("  %s: NO MAPPING for %s - refusing to build" % (t.name, t.type_name))
                    return 1
                raw = head[data_off + t.offset: data_off + t.offset + (t.expected_bytes() or 0)]
                if len(raw) != (t.expected_bytes() or 0):
                    print("  %s: SHORT READ - refusing to build" % t.name)
                    return 1
                blob, ent = tensor_entry(t.name, t, t.type_name, raw, seg_base)
                ent["file"] = fname
                fh.write(blob)
                seg_base += len(blob)
                man["tensors"][t.name] = ent
        print("%s: %d tensors, %.2f GiB" % (fname, len(group), path.stat().st_size / 2**30))

    man["files"] = {"experts.bin": exp_path.stat().st_size,
                    "dense.bin": (out_dir / "dense.bin").stat().st_size,
                    "embd.bin": (out_dir / "embd.bin").stat().st_size}
    if not skip_hash:
        print("hashing shards (this is the slow part)...")
        man["source"]["shard1_sha256"] = sha256(gguf)
        man["source"]["shard2_sha256"] = sha256(shard2) if shard2.exists() else None
    # shard 2 is NOT copied: the PLE table stays where it is and the manifest records how to find it
    if shard2.exists():
        g2 = G.GGUFFile(shard2)
        t2 = [t for t in g2.tensors if t.name == "per_layer_token_embd.weight"][0]
        man["shard2_tensor"] = {"name": t2.name, "type": t2.type_name, "shape": list(t2.shape),
                                "offset": t2.offset, "elements": t2.elements,
                                "path": str(shard2)}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("manifest.json: %d tensor entries" % len(man["tensors"]))
    return 0


def rebuild_role(blob: bytes, role: str) -> bytes:
    """Rebuild one role's raw Q2_0 bytes from an expert blob - the exact inverse of pack_layer's layout.

    The arena stores codes and scales in SEPARATE PLANES and interleaves gate/up ROWS (g0,u0,g1,u1,...), so
    a verifier that cannot invert that cannot check the arena.  Written as the inverse on purpose: this is
    the only independent statement about the blob layout in the tree, and if it and `pack_layer` were the
    same code the arena would be unverified by construction.
    """
    O, out = PL.OFF, bytearray()
    n_blk = PL.H * PL.FF // PL.QK
    for blk in range(n_blk):
        if role == "down":
            row, b_in = blk // PL.SCALES_D, blk % PL.SCALES_D
            co = O["down_codes"] + row * PL.ROW_BYTES_D + b_in * 16
            so = O["down_scales"] + (row * PL.SCALES_D + b_in) * 2
        else:
            row, b_in = blk // PL.SCALES_GU, blk % PL.SCALES_GU
            slot = 2 * row + (0 if role == "gate" else 1)
            co = O["gate_up_codes"] + slot * PL.ROW_BYTES_GU + b_in * 16
            so = O["gate_up_scales"] + (slot * PL.SCALES_GU + b_in) * 2
        out += blob[so:so + 2]          # fp16 d first, as block_q2_0 has it
        out += blob[co:co + 16]         # then the 16 packed code bytes
    return bytes(out)


def verify_experts(gguf: pathlib.Path, out_dir: pathlib.Path, man: dict, every: int) -> int:
    """Verify the expert arena: 85% of the pack by bytes and the part Phase 2 reads 480x per token."""
    g = G.GGUFFile(gguf)
    head, flen = open_shard(gguf)
    data_off = data_section_offset(g, flen)
    by_name = {t.name: t for t in g.tensors}
    exp_mm, _ = open_shard(out_dir / "experts.bin")
    blob = man["experts"]["blob_bytes"]
    n_layers = man["n_layers"]
    checked = bad = 0
    for layer in range(n_layers):
        for e in range(0, PL.NE, every):
            base = (layer * PL.NE + e) * blob
            b = exp_mm[base:base + blob]
            for role in ("gate", "up", "down"):
                t = by_name["blk.%d.ffn_%s_exps.weight" % (layer, role)]
                per_expert = t.elements // PL.NE
                nbytes = per_expert * PL.BLOCK_BYTES // PL.QK
                src = head[data_off + t.offset + e * nbytes: data_off + t.offset + (e + 1) * nbytes]
                ref = reference_values("Q2_0", src)
                got = MAPPINGS["Q2_0"].dec(canonicalise(t.name, "Q2_0", rebuild_role(b, role))[:2])
                checked += 1
                if not np.array_equal(ref, got):
                    print("  layer %d expert %d %-4s *** MISMATCH *** max|d| %.3e"
                          % (layer, e, role, np.abs(ref.astype(np.float64) - got.astype(np.float64)).max()))
                    bad += 1
    print("experts: %d layer/expert/role checks (%d layers, every %dth expert), %d bad"
          % (checked, n_layers, every, bad))
    return bad


def verify(gguf: pathlib.Path, out_dir: pathlib.Path, limit: int | None, expert_every: int) -> int:
    """P1.T3 over the pack: decode it from disk and compare with the source, bit for bit."""
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    # A full verify also reads back the source hash build recorded (absent after --skip-hash, which is how setup
    # builds): a pack compared against a different shard than it was built from is a mismatch, not a decode bug.
    want = (man.get("source") or {}).get("shard1_sha256")
    if want and limit is None:
        got = sha256(gguf)
        if got != want:
            print("%s: sha256 %s, but the pack was built from %s" % (gguf.name, got, want))
            print("tools/strata_pack.py verify FAIL")
            return 1
    g = G.GGUFFile(gguf)
    head, flen = open_shard(gguf)
    data_off = data_section_offset(g, flen)
    by_name = {t.name: t for t in g.tensors}
    files = {}
    bad = 0
    n_t = n_el = 0
    for i, (name, ent) in enumerate(sorted(man["tensors"].items())):
        if limit is not None and i >= limit:
            break
        if ent["file"] not in files:
            files[ent["file"]] = open_shard(out_dir / ent["file"])[0]
        t = by_name[name]
        raw = head[data_off + t.offset: data_off + t.offset + (t.expected_bytes() or 0)]
        ref = reference_values(t.type_name, raw)
        got = decode_entry(files[ent["file"]], ent)
        n_t += 1
        n_el += ref.size
        if ref.shape != got.shape or not np.array_equal(ref, got):
            d = np.abs(ref.astype(np.float64) - got.astype(np.float64)) if ref.shape == got.shape else None
            print("  %-40s *** MISMATCH *** %s" % (name, ("max|d| %.3e" % d.max()) if d is not None else
                                                   "shape %s vs %s" % (ref.shape, got.shape)))
            bad += 1
        elif i < 3 or i % 200 == 0:
            print("  %-40s bit-exact  (%d elements, %s)" % (name, ref.size, ent["form"]))
    print()
    print("verify: %d tensors, %d elements, %d bad" % (n_t, n_el, bad))
    if expert_every > 0:
        bad += verify_experts(gguf, out_dir, man, expert_every)
    print("tools/strata_pack.py verify " + ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def info(out_dir: pathlib.Path) -> int:
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    print("strata-pack v%d  layers=%d" % (man["format_version"], man["n_layers"]))
    for f, n in man.get("files", {}).items():
        print("  %-12s %12.3f GiB" % (f, n / 2**30))
    forms = {}
    for e in man["tensors"].values():
        forms[e["form"]] = forms.get(e["form"], 0) + 1
    print("  dense/embd by form: %s" % ", ".join("%s=%d" % kv for kv in sorted(forms.items())))
    print("  experts: %d blobs x %d B, %s" % (man["experts"]["layers"].__len__() * man["n_experts_per_layer"],
                                              man["experts"]["blob_bytes"], man["experts"]["source_type"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gguf", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--layers", type=int, default=None, help="expert layers to emit (default 48)")
    b.add_argument("--skip-hash", action="store_true")
    b.add_argument("--force", action="store_true", help="build into a directory that already holds a pack, unmaking that pack first (#634)")
    v = sub.add_parser("verify")
    v.add_argument("--gguf", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--limit", type=int, default=None)
    v.add_argument("--expert-every", type=int, default=32,
                   help="verify every Nth expert per layer in the arena (0 disables; the arena is 85%% of "
                        "the pack and verifying NONE of it is how a pack passes while being wrong)")
    n = sub.add_parser("info")
    n.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        return build(pathlib.Path(args.gguf), pathlib.Path(args.out), args.layers, args.skip_hash, args.force)
    if args.cmd == "verify":
        return verify(pathlib.Path(args.gguf), pathlib.Path(args.out), args.limit, args.expert_every)
    return info(pathlib.Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
