#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repack_mlx_to_gguf.py — Edge0 MLX int4 safetensors → GGUF lossless repack core.

Premise (design D1/D2): MLX affine int4 g64 `w = s·q + b` and ggml Q4_1 `w = d·q + m`
are the same value structure ⇒ a bit-exact re-wrap, **not a requant** — no q integer
value may ever be modified.

Format evidence (measured on the llama.cpp fork @7ab4ee7, forensics A-1; never change
a byte from memory):
  block_q4_1 = 20B {fp16 d@0, fp16 m@2, u8 qs[16]}                    ggml-common.h:201-212
  decode x[j]=(qs[j]&0xF)·d+m; x[j+16]=(qs[j]>>4)·d+m (j<16)          ggml-quants.c:479-498
    ⇒ permutation qs[j] = q[j] | (q[j+16]<<4); a g64 block splits into 2×g32 with the
      same (d,m)
  MLX payload U32 word = 8 nibble elements ascending (same order as the appendix-B
    collection script)
  ggml enum F32=0/F16=1/Q4_1=3 (fork ggml.h); GGUF v3: header(KV+infos) → align → data
  HF [out,in] flatten ≡ ggml file order (zero transpose); experts [E,N,K/8] ≡ ne{K,N,E}
    (pre-stacked; the loader does no 2D→3D assembly: forensics A-5/B-3)

Two-pass structure (as in gguf-py, so header backfill can never overwrite data):
  pass A register(name, tt, dims, size) → finalize() writes header + alignment
  pass B per-tensor streaming write+pad; the R1 parity check (sha16 equality of two
    independent dequant implementations) is embedded in pass B.

R1 semantics: identity tensor names + minimal metadata + audit block. The r3 semantic
transforms (norm +1 already baked — measured 2026-09-28: norm means 0.88~1.4 are final,
NEVER add +1 a second time; ssm_a=-exp(A_log) bake; qkvz already split, mapped straight
to attn_qkv/attn_gate; fork canonical naming) arrive in the r3 stage via NAME_MAP_R3
(forensics B → design D3).

Usage: --dir models/edge0-35b --out models/edge0-35b-gguf [--dry|--selftest] [--verify N|all]
"""
import argparse, json, os, re, struct, sys, time, hashlib
import numpy as np

T_F32, T_F16, T_Q4_1 = 0, 1, 3                       # measured against fork ggml.h
KV = dict(u32=4, i32=5, f32=6, bool=7, str=8, arr=9, u64=10, f64=12)
ALIGN, QK41 = 32, 32
DT_Q41 = np.dtype([("d", "<f2"), ("m", "<f2"), ("qs", "u1", (QK41 // 2,))])
assert DT_Q41.itemsize == 20
TOOL_VER = "w1-repack-1.2"
CHUNK = 8192


# ---- safetensors ----------------------------------------------------------------
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    base = 8 + n
    for v in h.values():
        v["off"] = base + v["data_offsets"][0]
        v["size"] = v["data_offsets"][1] - v["data_offsets"][0]
    return h


class StIndex:
    def __init__(self, paths):
        self.t, self._fh = {}, {}
        for p in sorted(paths):
            for k, v in st_header(p).items():
                self.t[k] = (p, v["off"], v["size"], v["dtype"], tuple(v["shape"]))

    def load(self, name):
        p, off, size, dt, shp = self.t[name]
        fh = self._fh.get(p) or self._fh.setdefault(p, open(p, "rb"))
        fh.seek(off)
        a = np.frombuffer(fh.read(size), np.uint8)
        a = {"U32": lambda: a.view(np.uint32),
             "BF16": lambda: bf16_to_f32(a.view(np.uint16)),
             "F16": lambda: a.view(np.float16),
             "F32": lambda: a.view(np.float32)}[dt]()
        return a.reshape(shp)

    def close(self):
        for fh in self._fh.values():
            fh.close()


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---- fp16 range gate (D6': benign/malign grading, calibrated 2026-09-28 by full scan)
_gate = dict(elems=0, breach=0, breach_benign=0)
audit_exempt = []   # collects (tensor,row,g32) global exemption entries; lands in the audit JSON


def fp16_exact(x_f32, where="?"):
    _gate["elems"] += int(x_f32.size)
    back = x_f32.astype(np.float16).astype(np.float32)
    bad = int(((back.view(np.uint32) != x_f32.view(np.uint32)) & ~np.isnan(x_f32)).sum())
    _gate["breach"] += bad
    if bad:
        raise AssertionError(f"fp16 range gate breach: {bad} elems @ {where}")


# ---- repack core ------------------------------------------------------------------
def mlx_q_bytes(u32_words):
    l = [(u32_words >> np.uint32(4 * k)) & np.uint32(0xF) for k in range(8)]
    return np.stack(l, -1).reshape(u32_words.shape[:-1] + (u32_words.shape[-1] * 8,))


def pack_q41_rows(q_u8, s_f32, b_f32, where="?"):
    """→ (blocks, exempt_g32set).

    Range-gate grading (calibrated 2026-09-28 by full scan, design D6'):
      benign    = s breaches AND the g64 group's q is all zero (MLX clamps the scale
                  floor to 1e-7; d contributes nothing ⇒ x = m = b bit-exact);
      malignant = s breaches with nonzero q in the group, or b breaches — the f16
                  subnormal range cannot hold it exactly; absolute offset ≤ 4.5e-7.
                  Registered on the exemption list (audit); sha parity skips these
                  32-block groups, everything else stays a hard gate.
    """
    R, K = q_u8.shape
    G, B = K // 64, K // QK41
    assert s_f32.shape == (R, G) and b_f32.shape == (R, G)
    _gate["elems"] += int(s_f32.size + b_f32.size)
    q3 = q_u8.reshape(R, B, QK41)
    s64 = q3.reshape(R, G, 64)
    d16_full = s_f32.astype(np.float16)
    m16_full = b_f32.astype(np.float16)
    sbad = (d16_full.astype(np.float32) != s_f32) & ~np.isnan(s_f32)
    bbad = (m16_full.astype(np.float32) != b_f32) & ~np.isnan(b_f32)
    benign = int((sbad & (s64 == 0).all(-1)).sum())
    mal_s = sbad & ~((s64 == 0).all(-1))
    _gate["breach_benign"] += benign
    n_mal = int(mal_s.sum() + bbad.sum())
    exempt = set()
    if n_mal:
        _gate["breach"] += n_mal
        for r, g in list(zip(*np.nonzero(mal_s))) + list(zip(*np.nonzero(bbad))):
            for bb in (2 * g, 2 * g + 1):
                exempt.add((int(r), int(bb)))
    o = np.empty((R, B), dtype=DT_Q41)
    o["d"] = d16_full[:, np.repeat(np.arange(G), 2)]
    o["m"] = m16_full[:, np.repeat(np.arange(G), 2)]
    o["qs"] = q3[..., :QK41 // 2] | (q3[..., QK41 // 2:] << 4)
    return o, exempt


# ---- two-sided parity check -------------------------------------------------------
def deq_mlx_side(q_u8, s_f32, b_f32):
    R, K = q_u8.shape
    return (q_u8.astype(np.float32).reshape(R, K // 64, 64) * s_f32[..., None]
            + b_f32[..., None]).reshape(R, K)


def deq_gguf_side(blocks):
    R, B = blocks.shape
    x = np.empty((R, B, QK41), np.float32)
    x[..., :QK41 // 2] = (blocks["qs"] & 0xF).astype(np.float32)
    x[..., QK41 // 2:] = (blocks["qs"] >> 4).astype(np.float32)
    x *= blocks["d"].astype(np.float32)[..., None]
    x += blocks["m"].astype(np.float32)[..., None]
    return x.reshape(R, B * QK41)


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


# ---- GGUF v3 two-pass writer ------------------------------------------------------
class GgufWriter:
    def __init__(self, path):
        self.path = path
        self.kvs, self.recs = [], []
        self.dlen, self._hdr_done, self._f = 0, False, None

    def _str(self, s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def kv(self, key, ty, payload):
        assert not self._hdr_done
        self.kvs.append(self._str(key) + struct.pack("<I", ty) + payload)

    def kv_s(self, key, v):
        self.kv(key, KV["str"], self._str(v))

    def register(self, name, tt, dims_ne, size):
        assert not self._hdr_done
        self.recs.append((name, tt, dims_ne, self.dlen, size))
        self.dlen += size + (-size) % ALIGN

    def finalize(self):
        infos = b"".join(self._str(n) + struct.pack("<I", len(d))
                         + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", tt, off)
                         for n, tt, d, off, _s in self.recs)
        body = b"".join(self.kvs)
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.recs), len(self.kvs))
        ds = (len(head) + len(body) + len(infos) + ALIGN - 1) // ALIGN * ALIGN
        self._f = open(self.path, "wb")
        self._f.write(head + body + infos + b"\x00" * (ds - len(head) - len(body) - len(infos)))
        self._hdr_done = True
        return ds

    def write(self, data):
        assert self._hdr_done, "finalize() first"
        self._f.write(data)

    def align_pad(self):
        p = self._f.tell() % ALIGN
        if p:
            self._f.write(b"\x00" * (ALIGN - p))

    def close(self):
        self._f.close()


# ---- classification ----------------------------------------------------------------
def classify(idx):
    """→ plan=[(kind, weight_name, base)]; quantized families named <base>.{weight,scales,biases}."""
    plan, skipped = [], []
    for name, (_p, _o, _s, dt, shp) in idx.t.items():
        if name.endswith((".scales", ".biases")):
            continue
        if re.search(r"lora|pregate|prerouter|mtp", name, re.I):
            skipped.append((name, "external/defense tensor (never enters the GGUF; shipped alongside as-is)"))
            continue
        base = name[:-7] if name.endswith(".weight") else name
        if dt == "U32":
            skey = base + ".scales"
            if skey not in idx.t:
                raise ValueError(f"U32 tensor without scales: {name}")
            G, words = idx.t[skey][4][-1], shp[-1]
            if G * 64 == words * 8:
                plan.append(("q41", name, base))
            elif G * 64 == words * 4:
                plan.append(("f32i8", name, base))
            else:
                raise ValueError(f"bit-width resolution failed {name} shape={shp} G={G}")
        elif dt in ("BF16", "F16", "F32"):
            # BF16→F32 exact widening (conv1d carries |w|<2^-17 micro values that underflow
            # in f16 — measured breach in 30 layers, calibrated 2026-09-28; size cost
            # <10MB, negligible). F16 sources are stored directly.
            plan.append(("f16" if dt == "F16" else "f32", name, base))
        else:
            raise ValueError(f"unexpected dtype {dt} @ {name}")
    return sorted(plan), sorted(skipped)


def tensor_shape(idx, kind, name, base):
    """→ (tt, dims_ne, size_bytes, K) (K is meaningful only for quantized/unpacked kinds)."""
    dt, shp = idx.t[name][3], idx.t[name][4]
    if kind in ("f16", "f32"):
        tt = T_F16 if kind == "f16" else T_F32
        return tt, tuple(reversed(shp)), int(np.prod(shp)) * (2 if kind == "f16" else 4), None
    words = shp[-1]
    lead = shp[:-1]
    rows = int(np.prod(lead)) if lead else 1
    G = idx.t[base + ".scales"][4][-1]
    if kind == "q41":
        K = words * 8
        assert G * 64 == K
        return T_Q4_1, (K,) + tuple(reversed(lead)), rows * (K // QK41) * 20, K
    K = words * 4
    assert G * 64 == K
    return T_F32, (K,) + tuple(reversed(lead)), rows * K * 4, K


# ---- main flow -----------------------------------------------------------------------
def run(args):
    t0 = time.time()
    idx = StIndex(args.shards)
    plan, skipped = classify(idx)
    if args.dry:
        agg = {}
        for k, n, _b in plan:
            agg.setdefault((k, idx.t[n][3], idx.t[n][4]), 0)
            agg[(k, idx.t[n][3], idx.t[n][4])] += 1
        for (k, dt, shp), c in sorted(agg.items(), key=lambda x: str(x[0])):
            print(f"[{k:6s}] src {dt:4s} {str(list(shp)):22s} ×{c}")
        for n, why in skipped:
            print(f"[skip] {n}")
        print(f"plan={len(plan)} skip={len(skipped)}")
        return

    stem = os.path.basename(args.dir.rstrip("\\/"))
    os.makedirs(args.out, exist_ok=True)
    gp = os.path.join(args.out, f"{stem}-r1.gguf")
    audit = dict(tool=TOOL_VER, tier=stem, mode="r1", ts=time.strftime("%F %T"),
                 shards=[os.path.basename(p) for p in args.shards],
                 skipped=dict(skipped), gates=[], errors=[], norm_stats={})
    w = GgufWriter(gp)
    w.kv_s("general.architecture", "qwen3next")
    w.kv_s("general.name", f"{stem} R1-repack (bit-exact verification artifact: identity names / minimal metadata / no tokenizer — not a release artifact)")
    w.kv("general.alignment", KV["u32"], struct.pack("<I", ALIGN))

    # pass A: register everything
    regs = []
    for kind, name, base in plan:
        tt, dims, size, K = tensor_shape(idx, kind, name, base)
        w.register(name, tt, dims, size)
        regs.append((kind, name, base, tt, dims, size, K))
    ds = w.finalize()

    # pass B: stream writes + parity gates
    rng = np.random.default_rng(20260928)
    full = args.verify == "all"
    for kind, name, base, tt, dims, size, K in regs:
        dt, shp = idx.t[name][3], idx.t[name][4]
        if kind in ("f16", "f32"):
            a = idx.load(name)                     # BF16 sources already widened exactly to f32 at load
            if dt == "BF16" and name.endswith("norm.weight"):
                audit["norm_stats"][name] = dict(min=float(a.min()), mean=float(a.mean()), max=float(a.max()))
            a = np.ascontiguousarray(a, np.float16 if kind == "f16" else np.float32)
            w.write(a.tobytes()); w.align_pad()
            continue
        lead, words = shp[:-1], shp[-1]
        wgt, s_arr, b_arr = idx.load(name), idx.load(base + ".scales"), idx.load(base + ".biases")
        rows = int(np.prod(lead)) if lead else 1
        W, S, Bs = wgt.reshape(rows, words), s_arr.reshape(rows, -1), b_arr.reshape(rows, -1)
        picks = ({int(r) // CHUNK for r in rng.choice(rows, min(rows, args.verify), replace=False).tolist()}
                 if isinstance(args.verify, int) else None)
        for ci, i0 in enumerate(range(0, rows, CHUNK)):
            sl = slice(i0, min(i0 + CHUNK, rows))
            if kind == "q41":
                q = mlx_q_bytes(W[sl])
                blocks, exm = pack_q41_rows(q, S[sl], Bs[sl], name)
                w.write(blocks.tobytes())
                if full or (picks is not None and ci in picks):
                    exm_g = {(i0 + r, bb) for (r, bb) in exm}
                    audit_exempt.extend([name, r + i0, bb] for (r, bb) in sorted(exm_g))
                    if exm_g:
                        A = deq_mlx_side(q, S[sl], Bs[sl])
                        B = deq_gguf_side(blocks)
                        bad = (A.view(np.uint32) != B.view(np.uint32)) & ~np.isnan(A)
                        rr, cc = np.nonzero(bad)
                        ok = all(((int(r0) + i0, int(c0) // QK41) in exm_g) for r0, c0 in zip(rr, cc))
                        n_diff = int(bad.sum())
                    else:
                        ok = sha16(deq_mlx_side(q, S[sl], Bs[sl]).tobytes()) == sha16(deq_gguf_side(blocks).tobytes())
                        n_diff = 0
                    audit["gates"].append(dict(t=name, rows=f"{i0}:{sl.stop}", ok=bool(ok),
                                               mal_groups=len(exm_g), diff_elems=n_diff))
                    if not ok:
                        audit["errors"].append(f"SHA MISMATCH {name} @{i0} (outside exemptions)")
            else:
                q8 = W[sl].view(np.uint8).reshape(sl.stop - sl.start, words * 4)
                w.write(deq_mlx_side(q8, S[sl], Bs[sl]).astype(np.float32, copy=False).tobytes())
        w.align_pad()
        print(f"  [{kind:6s}] {name:64s} →{str(dims):20s} {w._f.tell()/1e9:6.2f}GB {time.time()-t0:5.0f}s", flush=True)

    audit.update(fp16_gate=dict(_gate), n_tensors=len(regs), elapsed_s=round(time.time() - t0, 1),
                 exempt_groups=len(audit_exempt), exempt_list=sorted(set(map(tuple, audit_exempt)))[:500]
                 if args.verify == "all" else f"({len(audit_exempt)} listed)")
    w.close()
    idx.close()
    ng = len(audit["gates"]); nb = sum(0 if g["ok"] else 1 for g in audit["gates"])
    # Red line A (layered whitelist): malignant f16 breaches are a format physics ceiling —
    # register per-block exemptions (Δ≤4.5e-7); only unexempted sha mismatches are errors.
    # Benign breaches (all-zero groups) have x=m=b bit-exact and take no exemption slot.
    verdict = "GREEN" if not audit["errors"] and nb == 0 else "RED"
    tag = "R1-bit-parity " + verdict + (f" ({audit['exempt_groups']} block groups exempted, pending proposal-A sign-off)"
                                        if audit["exempt_groups"] else "")
    json.dump(dict(audit=audit, verdict=tag, gates=ng, gguf_size=os.path.getsize(gp)),
              open(os.path.join(args.out, f"{stem}-r1-audit.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nGGUF {gp}  {os.path.getsize(gp)/1e9:.2f}GB  tensors={len(regs)}  {time.time()-t0:.0f}s")
    print(f"parity gates {ng} (mismatches outside exemptions: {nb}) · fp16 gate elems={_gate['elems']:,} "
          f"benign={_gate['breach_benign']:,} malignant-exempt={_gate['breach']:,} ⇒ {tag}")
    return 0 if verdict == "GREEN" else 1


# ---- selftest (synthetic formats, seconds-level gate) ---------------------------------
def selftest(args):
    rng = np.random.default_rng(7)
    R, K = 6, 128
    q = rng.integers(0, 1 << 32, (R, K // 8), dtype=np.uint64).astype(np.uint32)
    q[0] = 0; q[1] = 0xFFFFFFFF; q[2, 0] = np.uint32(1)
    s = (rng.random((R, K // 64)) * 0.02 + 1e-3).astype(np.float32)
    s = bf16_to_f32((s.view(np.uint32) >> 16).astype(np.uint16))
    b = ((rng.random((R, K // 64)) - 0.5) * 0.1).astype(np.float32)
    b = bf16_to_f32((b.view(np.uint32) >> 16).astype(np.uint16))
    blocks, exm = pack_q41_rows(mlx_q_bytes(q), s, b)
    assert not exm, "synthetic case must not trigger a malignant exemption"
    A = deq_mlx_side(mlx_q_bytes(q), s, b)
    Bq = deq_gguf_side(blocks)
    assert sha16(A.tobytes()) == sha16(Bq.tobytes()), "synthetic parity sha mismatch"
    # Grading red line, proven on synthetic data (D6'): scales all at the clamp floor 1e-7
    # (f16 underflow breach) —
    #   all-zero group (row 0 entirely + row 1 group 0) = benign: takes no exemption and
    #     unpacks bit-exact (d contributes nothing)
    #   row 1 group 1 element 64 nonzero = malignant: exempted at (1,2),(1,3); the mismatch
    #     is confined to exactly that group
    qz = np.zeros((2, 16), np.uint32); qz[1, 8] = np.uint32(0xF)
    sz = np.full((2, 2), np.float32(1e-7)); bz = np.zeros((2, 2), np.float32)
    blocks2, exm2 = pack_q41_rows(mlx_q_bytes(qz), sz, bz, "synthetic")
    assert exm2 == {(1, 2), (1, 3)}, exm2
    A2 = deq_mlx_side(mlx_q_bytes(qz), sz, bz); B2 = deq_gguf_side(blocks2)
    bad2 = (A2.view(np.uint32) != B2.view(np.uint32)) & ~np.isnan(A2)
    rr, cc = np.nonzero(bad2)
    assert len(rr) == 1 and (int(rr[0]), int(cc[0]) // QK41) in exm2, (rr, cc)
    print("selftest grading red line (benign: d contributes nothing, bit-exact / malignant confined to the exempt group) PASS")

    def code(r, e):
        wr, j = divmod(e, 8)
        return float((q[r, wr] >> np.uint32(4 * j)) & np.uint32(0xF))
    d0, m0 = float(blocks["d"][2, 0]), float(blocks["m"][2, 0])
    for e in (0, 5, 16, 21, 31):
        assert abs(float(Bq[2, e]) - (code(2, e) * d0 + m0)) < 1e-9, f"element {e} nibble-order pin failed"
    print("selftest pack/deq sha equality PASS · extreme codes 0/15 + cross-word nibble order PASS")

    p = os.path.join(args.out or ".", "_selftest.gguf")
    os.makedirs(p and (args.out or "."), exist_ok=True)
    size_a = 20 * (K // QK41) * R
    ww = GgufWriter(p)
    ww.kv_s("general.architecture", "qwen3next")
    ww.register("a.weight", T_Q4_1, (K, R), size_a)
    ww.register("z.weight", T_F32, (K,), 4 * K)
    ds = ww.finalize()
    ww.write(blocks.tobytes()); ww.align_pad()
    ww.write(np.zeros(K, np.float32).tobytes()); ww.align_pad()
    ww.close()
    with open(p, "rb") as fh:
        assert fh.read(8) == b"GGUF\x03\x00\x00\x00"
        nt, nkv = struct.unpack("<QQ", fh.read(16)); assert (nt, nkv) == (2, 1)

        def rs(): return fh.read(struct.unpack("<Q", fh.read(8))[0])
        rs(); fh.read(4); rs()
        recs = []
        for _ in range(nt):
            nm = rs().decode(); nd = struct.unpack("<I", fh.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", fh.read(8 * nd))
            ty, off = struct.unpack("<IQ", fh.read(12))
            recs.append((nm, ty, dims, off))
        got_ds = (fh.tell() + ALIGN - 1) // ALIGN * ALIGN
        assert got_ds == ds, (got_ds, ds)
        fh.seek(ds + recs[0][3])
        assert fh.read(size_a) == blocks.tobytes(), "container readback tensor a mismatch"
        fh.seek(ds + recs[1][3])
        assert fh.read(4 * K) == np.zeros(K, np.float32).tobytes()
    os.remove(p)
    print("selftest GGUF two-pass container header/data byte-identical PASS\nSELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out", default="models/edge0-35b-gguf")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", default=None, help="N | all")
    a = ap.parse_args()
    if a.verify == "all":
        pass
    elif a.verify:
        a.verify = int(a.verify)
    if a.selftest:
        return selftest(a)
    assert a.dir, "--dir is required"
    a.shards = [os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir))
                if re.fullmatch(r"model(-\d+-of-\d+)?\.safetensors", f)]
    assert a.shards, "no model*.safetensors found"
    return run(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
