"""tools/canonical_xcheck.py - P1.T3 in prototype: does the canonical form round-trip BIT-EXACTLY?

P1.S6's release-blocking test is `dequant(canonical) == dequant_ggml(original)` bit for bit, for every tensor
of the real GGUF. This is that test in Python, before the C++ exists - the same order that worked for `ref/`
(the oracle before the engine), and faster to iterate.

It starts with `Q2_0 -> S2`, which is both the largest mapping (31.64 GiB, the 202 routed-expert tensors) and
the one whose codebook has a non-obvious feature:

    dequantize_row_q2_0:   value[j] = (code[j] - 1) * d        code in {0,1,2,3}

**The `- 1` is an offset in the codebook** - the representable values are {-d, 0, d, 2d}, not {0, d, 2d, 3d}.
A canonical form that stored the codes as unsigned 0..3 and decoded `code * d` would look right on most values
and be wrong on all of them by one step. That is why the source expression is transcribed rather than
re-derived, and why this test exists.

`docs/pack-format.md` §2.1 says Q2_0 -> S2 splits codes and scales into separate planes with NO value change.
This checks exactly that claim.

Run: python tools/canonical_xcheck.py [--tensors 4] [--gguf PATH]
"""
from __future__ import annotations

import argparse
import collections
import mmap
import os
import pathlib
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "ref"))

import gguf_reader as G                                    # noqa: E402
from gguf_writer import dequantize_q2_0                    # noqa: E402
from load import _dequant_flat                             # noqa: E402

# default: $STRATA_SHARD1, else the development layout (<Strata>/Q2_0 beside <Strata>/Public/Engine)
SHARD1 = os.environ.get("STRATA_SHARD1") or os.path.join(os.path.dirname(os.path.dirname(REPO)), "Q2_0",
                                                          "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")

QK = 64          # Q2_0 block: 64 elements
BLOCK_BYTES = 18  # 2-byte fp16 scale + 16 bytes of 2-bit codes


# ---------------------------------------------------------------- mappings
#
# Each mapping is (encode, decode, block_bytes, block_elems).  The DECODE is transcribed from
# `ggml-quants.c` expression by expression - the plan's rule, and the thing that decides P1.T3.

def to_s8_q8_0(raw: bytes):
    """`Q8_0` -> S8: codes + an fp16 scale per 32 elements, `codebook` the affine `code - 128`.

    `dequantize_row_q8_0`: `y[j] = x[i].qs[j] * d` with `d = GGML_FP16_TO_FP32(x[i].d)`.

    The code plane is UNSIGNED (0..255) like every other form here, so the source's `int8_t` code is stored
    `+128` and the decode subtracts 128 - the same integer-bias mechanism as Q2_0's `-1` and Q4_0's `-8`, and
    bit-exact for the same reason.  It used to be stored signed instead, which is the natural reading of
    `int8_t qs[32]` and was self-consistent with its own decoder; the pack is what showed the ambiguity,
    because the two Q8_0 tensors then decoded 128 codes away from where they belonged while all 12 other
    types were fine.  ONE code-plane contract, not two.
    """
    n = len(raw) // 34
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 34)
    # `.reshape(-1)` matters: the fp16 view of an (n,2) byte column is (n,1), and a decode doing
    # `scales[:, None]` then broadcasts it to (n,1,1) - silently producing n TIMES too many elements instead
    # of raising.  All the encoders return 1-D scales for that reason.
    scales = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    codes = np.ascontiguousarray(a[:, 2:34]).view(np.int8).astype(np.int16) + 128
    return codes.astype(np.uint8), scales


def from_s8_q8_0(parts):
    codes, scales = parts
    v = (codes.astype(np.int32) - 128).astype(np.float32)      # the +128 is undone inside the integer
    return (v * scales[:, None]).reshape(-1)


def to_s4_q4_0(raw: bytes):
    """`Q4_0` -> S4: 4-bit codes + one fp16 scale per 32 elements. No separate offset.

    `dequantize_row_q4_0`:
        x0 = (qs[j] & 0x0F) - 8 ;  y[j +  0] = x0*d
        x1 = (qs[j] >>   4) - 8 ;  y[j + 16] = x1*d

    TWO things here are easy to get wrong and both would still produce plausible numbers: the low nibbles
    fill the FIRST HALF of the block and the high nibbles the SECOND (they are NOT interleaved), and the
    `- 8` is applied to the integer BEFORE the multiply, i.e. the codebook is -8..7 rather than 0..15.
    """
    n = len(raw) // 18
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 18)
    scales = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    qs = np.ascontiguousarray(a[:, 2:18])                      # (n, 16)
    codes = np.empty((n, 32), dtype=np.uint8)
    codes[:, 0:16] = qs & 0x0F                                 # low nibbles -> first half
    codes[:, 16:32] = qs >> 4                                  # high nibbles -> second half
    return codes, scales


def from_s4_q4_0(parts):
    codes, scales = parts
    v = (codes.astype(np.int32) - 8).astype(np.float32)        # the -8 is inside the integer
    return (v * scales[:, None]).reshape(-1)


def get_scale_min_k4(scales, j):
    """`get_scale_min_k4`, ggml-quants.c L880, transcribed.

    The 6-bit scale and min of group `j` (0..7) live packed in the block's 12 scale bytes: the low four
    groups keep their 6 bits in a byte, and the high four borrow the top two bits of the LOW groups' bytes.
    Reading only `q[j]` gives the right answer for the first half of the groups and silently wrong ones for
    the second - which is why this is transcribed rather than inferred from the layout.
    """
    if j < 4:
        return int(scales[j]) & 63, int(scales[j + 4]) & 63
    return ((int(scales[j + 4]) & 0xF) | ((int(scales[j - 4]) >> 6) << 4),
            (int(scales[j + 4]) >> 4) | ((int(scales[j]) >> 6) << 4))


def to_s4_q4_k(raw: bytes):
    """`Q4_K` -> S4: 8 groups of 32 per 256-element block, each with `scale = d*sc` and `offset = -(min*m)`.

    `dequantize_row_q4_K` (L1529):
        d = fp16(x.d) ; min = fp16(x.dmin)
        for j in 0..256 step 64:
            (sc,m) = get_scale_min_k4(is+0) ; d1 = d*sc ; m1 = min*m
            (sc,m) = get_scale_min_k4(is+1) ; d2 = d*sc ; m2 = min*m
            y[l]      = d1 * (q[l] & 0xF) - m1
            y[l + 32] = d2 * (q[l] >>   4) - m2

    TWO groups per 64-element step, taking the low and high nibbles of the SAME 32 scale bytes - so the
    canonical group size is 32, not 64, and there are 8 groups per block.
    """
    n = len(raw) // 144
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 144)
    d = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    dmin = np.ascontiguousarray(a[:, 2:4]).view(np.float16).astype(np.float32).reshape(-1)
    scb = a[:, 4:16]                                   # (n, 12) the packed 6-bit scales and mins
    qs = a[:, 16:144]                                  # (n, 128)

    scales = np.empty((n, 8), dtype=np.float32)
    offsets = np.empty((n, 8), dtype=np.float32)
    for g in range(8):
        sc = np.array([get_scale_min_k4(scb[i], g)[0] for i in range(n)], dtype=np.float32)
        m = np.array([get_scale_min_k4(scb[i], g)[1] for i in range(n)], dtype=np.float32)
        scales[:, g] = d * sc                          # f32 product of the same two factors ggml uses
        offsets[:, g] = -(dmin * m)                    # negation is exact, so adding it is bit-identical

    codes = np.empty((n, 256), dtype=np.uint8)
    for j in range(4):
        blk = qs[:, j * 32:(j + 1) * 32]
        codes[:, j * 64:j * 64 + 32] = blk & 0x0F      # group is+0: LOW nibbles
        codes[:, j * 64 + 32:j * 64 + 64] = blk >> 4   # group is+1: HIGH nibbles
    return codes, scales, offsets


def from_s4_q4_k(parts):
    codes, scales, offsets = parts
    v = codes.astype(np.float32).reshape(codes.shape[0], 8, 32)
    return (v * scales[:, :, None] + offsets[:, :, None]).reshape(-1)


def unpack_q3_k_scales(sb):
    """The 12 scale bytes of a `Q3_K` block -> 16 six-bit scales, in group order.

    `dequantize_row_q3_K` L1323-1328, transcribed.  As bytes, with `s` the 12 scale bytes:
        group  j (0..3)  = (s[j]   & 0x0F) | ((s[8+j] & 0x03) << 4)
        group  4+j       = (s[4+j] & 0x0F) | (((s[8+j] >> 2) & 0x03) << 4)
        group  8+j       = (s[j]   >> 4)   | (((s[8+j] >> 4) & 0x03) << 4)
        group 12+j       = (s[4+j] >> 4)   | (((s[8+j] >> 6) & 0x03) << 4)

    The trap is the same shape as `get_scale_min_k4`'s and just as quiet: the first EIGHT groups take the
    low nibbles of `s[0..7]`, the last eight the HIGH nibbles of those SAME bytes, and `s[8..11]` supply two
    high bits each in that same order.  A reader who assumes "12 bytes = 12 groups, one nibble each" gets
    half the groups right.
    """
    n = sb.shape[0]
    out = np.empty((n, 16), dtype=np.int32)
    for j in range(4):
        t = sb[:, 8 + j].astype(np.int32)
        lo, hi = sb[:, j].astype(np.int32), sb[:, 4 + j].astype(np.int32)
        out[:, j] = (lo & 0x0F) | ((t & 0x03) << 4)
        out[:, 4 + j] = (hi & 0x0F) | (((t >> 2) & 0x03) << 4)
        out[:, 8 + j] = ((lo >> 4) & 0x0F) | (((t >> 4) & 0x03) << 4)
        out[:, 12 + j] = ((hi >> 4) & 0x0F) | (((t >> 6) & 0x03) << 4)
    return out


def to_s4_q3_k(raw: bytes):
    """`Q3_K` -> S4: 16 groups of 16 per 256-element block, codebook 0..7 of the 0..15 a 4-bit plane holds.

    `dequantize_row_q3_K` (L1305):
        d_all = fp16(x.d)
        for half in {0, 128}:                     # `q` advances by 32 per half; `hm` does NOT advance
            for j in 0..3:                        # shift = 2j, mask m = 1 << j
                dl = d_all * (scales[is++] - 32)
                for l in 0..15: y = dl * ((qs[l +  0] >> shift) & 3) - ((hm[l +  0] & m) ? 0 : 4)
                dl = d_all * (scales[is++] - 32)
                for l in 0..15: y = dl * ((qs[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)

    Three things here are each independently easy to get wrong, and each would still yield plausible values:

      * **`hm` is indexed 0..31 in BOTH halves, but the BIT moves.**  `q += 32` sits at the end of the `n`
        loop, while the hmask is only ever read as `hm[l+0]`/`hm[l+16]` - so the second half re-reads the
        SAME 32 mask bytes.  What distinguishes the halves is `m`: it is declared outside the `n` loop and
        only ever shifted, so it never resets, and the second half uses bits 4..7 where the first used
        0..3.  `shift` DOES reset per half.  Both facts are needed and they point opposite ways; getting
        `m` wrong flips the sign of roughly half the elements of the second half, which is what the first
        run of this mapping did.
      * **the group size is 16, not 32** - 16 scales per half, hence 16 groups per 256-element block.
      * **the `- 4` is an INTEGER bias on the code, not an offset in the value.**  So the stored code is
        `(2 bits) + (hm bit ? 4 : 0)`, which stays in 0..7, and the decode subtracts 4 before the multiply.
        Folding it into a value-space offset instead, as `(c+4)*dl - 4*dl`, is NOT bit-exact: `fl(5a) - 4a`
        is exact by Sterbenz but equals `a + 5ae`, not `a`.  Q2_0's `- 1` and Q4_0's `- 8` are the same
        integer-bias mechanism, which is why this is expressed the same way.

    The element order is `128*half + 32*j + 16*sub + l`, matching the sequential `*y++`.
    """
    n = len(raw) // 110
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 110)
    hmask = a[:, 0:32]                                     # hmask[QK_K/8]
    qs = a[:, 32:96]                                       # qs[QK_K/4]
    sb = a[:, 96:108]                                      # scales[12]
    d = np.ascontiguousarray(a[:, 108:110]).view(np.float16).astype(np.float32).reshape(-1)

    sc6 = unpack_q3_k_scales(sb)
    scales = d[:, None] * (sc6 - 32).astype(np.float32)    # integer bias, then ONE f32 multiply
    codes = np.empty((n, 256), dtype=np.uint8)
    for half in range(2):
        for j in range(4):
            for sub in range(2):
                g = 8 * half + 2 * j + sub
                base = half * 32 + sub * 16
                c2 = (qs[:, base:base + 16] >> (2 * j)) & 0x03
                # `m` does NOT reset between halves: it is `1 << (4*half + j)`, not `1 << j`.
                m = (hmask[:, sub * 16:sub * 16 + 16] >> (4 * half + j)) & 0x01
                codes[:, g * 16:(g + 1) * 16] = c2 + 4 * m
    return codes, scales


def from_s4_q3_k(parts):
    codes, scales = parts
    v = (codes.astype(np.int32) - 4).astype(np.float32)
    return (v.reshape(-1, 16, 16) * scales[:, :, None]).reshape(-1)


# ---------------------------------------------------------------- the IQ4 codebook
# `kvalues_iq4nl`.  `ggml-common.h` heads the IQ4 block definitions "Non-linear quants", and the name is
# load-bearing: this codebook is NOT affine, so unlike Q2_0's `-1`, Q3_K's `-4` and Q4_0's `-8` it cannot be
# expressed as a bias on the code.  It is a 16-entry table lookup, and the canonical decode has to say so.
# Checked against two independent files before use, because a codebook is exactly the kind of constant that
# has silent variants: ggml-cpu/llamafile/sgemm.cpp L1361 and ggml-opencl/kernels/mul_mv_iq4_nl_f32.cl L30.
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                        1,   13,  25,  38,  53,  69,  89, 113], dtype=np.float32)


def to_s4_iq4_nl(raw: bytes):
    """`IQ4_NL` -> S4 with the IQ4 codebook: 32 codes, one fp16 scale, no offset.

    `dequantize_row_iq4_nl` (L2725):
        d = fp16(x.d)
        for j in 0..15:
            y[j +  0] = d * kvalues_iq4nl[qs[j] & 0xf]
            y[j + 16] = d * kvalues_iq4nl[qs[j] >>  4]

    Same nibble split as `Q4_0` - low nibbles fill the first half, high the second - but the code is an index
    into the table rather than a value, so the whole `- 8` business has no analogue here.
    """
    n = len(raw) // 18
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 18)
    scales = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    qs = np.ascontiguousarray(a[:, 2:18])                  # (n, 16) = QK4_NL/2
    codes = np.empty((n, 32), dtype=np.uint8)
    codes[:, 0:16] = qs & 0x0F
    codes[:, 16:32] = qs >> 4
    return codes, scales


def from_s4_iq4_nl(parts):
    codes, scales = parts
    return (KV_IQ4NL[codes] * scales[:, None]).reshape(-1)


def to_s4_iq4_xs(raw: bytes):
    """`IQ4_XS` -> S4 with the IQ4 codebook: 8 groups of 32, each with `scale = d*(ls - 32)`.

    `dequantize_row_iq4_xs` (L2743):
        d = fp16(x.d)
        for ib in 0..7:                                   # QK_K/32 groups
            ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)
            dl = d * (ls - 32)
            for j in 0..15:
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf]
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >>  4]
            qs += 16

    The 6-bit scale `ls` is split across TWO parallel arrays by element index: `scales_l` holds one nibble
    per 32-element group (two groups per byte, low nibble first) and the `uint16_t scales_h` holds the top
    two bits of each group's scale at bit offset `2*ib`.  Reading a group's scale out of `scales_l` alone is
    right for values under 16 and silently wrong above it.
    """
    n = len(raw) // 136
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 136)
    d = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    sh = np.ascontiguousarray(a[:, 2:4]).view(np.uint16).astype(np.int32).reshape(-1)   # uint16 scales_h
    sl = a[:, 4:8].astype(np.int32)                          # scales_l[QK_K/64] = 4 bytes
    qs = a[:, 8:136]                                         # qs[QK_K/2] = 128 bytes

    ls = np.empty((n, 8), dtype=np.int32)
    for ib in range(8):
        ls[:, ib] = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
    scales = d[:, None] * (ls - 32).astype(np.float32)       # integer bias, then ONE f32 multiply

    codes = np.empty((n, 256), dtype=np.uint8)
    for ib in range(8):
        blk = qs[:, ib * 16:(ib + 1) * 16]
        codes[:, ib * 32:ib * 32 + 16] = blk & 0x0F          # low nibbles -> first 16 of the group
        codes[:, ib * 32 + 16:ib * 32 + 32] = blk >> 4        # high nibbles -> second 16
    return codes, scales


def from_s4_iq4_xs(parts):
    codes, scales = parts
    return (KV_IQ4NL[codes].reshape(-1, 8, 32) * scales[:, :, None]).reshape(-1)


def to_s8_q5_0(raw: bytes):
    """`Q5_0` -> S8: 5-bit codes, one fp16 scale per 32 elements, codebook bias `-16`.

    `dequantize_row_q5_0` (L500):
        qh = the 4 bytes of x.qh read as a native uint32
        for j in 0..15:
            xh_0 = ((qh >> (j +  0)) << 4) & 0x10
            xh_1 = ((qh >> (j + 12))     ) & 0x10
            x0 = ((qs[j] & 0x0F) | xh_0) - 16 ;  y[j +  0] = x0*d
            x1 = ((qs[j] >>   4) | xh_1) - 16 ;  y[j + 16] = x1*d

    The 5th bit of element `j` is bit `j` of the 32-bit `qh`, and of element `j+16` is bit `j+16`: one bit per
    element, taken from a field that is read as a single integer rather than as 4 bytes.  `x1`'s shift of
    `j + 12` plus the `& 0x10` is how the source reaches bit `j+16`; writing `j + 16` directly gives the same
    answer, which is worth knowing before "fixing" it.
    """
    n = len(raw) // 22
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 22)
    scales = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    qh = np.ascontiguousarray(a[:, 2:6]).view("<u4").reshape(-1)       # little-endian native read
    qs = a[:, 6:22]                                                    # (n, 16)
    j = np.arange(16)
    xh0 = (((qh[:, None] >> j[None, :]) & 1) << 4).astype(np.uint8)
    xh1 = (((qh[:, None] >> (j[None, :] + 16)) & 1) << 4).astype(np.uint8)
    codes = np.empty((n, 32), dtype=np.uint8)
    codes[:, 0:16] = (qs & 0x0F) | xh0                                 # low nibbles -> first half
    codes[:, 16:32] = (qs >> 4) | xh1                                  # high nibbles -> second half
    return codes, scales


def from_s8_q5_0(parts):
    codes, scales = parts
    v = (codes.astype(np.int32) - 16).astype(np.float32)               # the -16 is inside the integer
    return (v * scales[:, None]).reshape(-1)


def to_s8_q5_k(raw: bytes):
    """`Q5_K` -> S8: 8 groups of 32, `scale = d*sc`, `offset = -(dmin*m)`; codes are 5-bit.

    `dequantize_row_q5_K` (L1731):
        d = fp16(x.d) ; min = fp16(x.dmin)
        is = 0 ; u1 = 1 ; u2 = 2
        for j in 0..256 step 64:
            (sc,m) = get_scale_min_k4(is+0) ; d1 = d*sc ; m1 = min*m
            (sc,m) = get_scale_min_k4(is+1) ; d2 = d*sc ; m2 = min*m
            for l in 0..31: y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1
            for l in 0..31: y++ = d2 * ((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2
            ql += 32 ; is += 2 ; u1 <<= 2 ; u2 <<= 2

    Identical in shape to `Q4_K` - same `get_scale_min_k4`, same 8 groups of 32, same low-nibbles-then-high
    order - with a 5th bit added.  `qh` does NOT advance inside the loop: the same 32 bytes are re-read by
    all four 64-element steps, and what selects a distinct bit each time is `u1`/`u2` walking 1,4,16,64 and
    2,8,32,128.  This is the Q3_K hmask trap again, and it is why the masks are written out rather than the
    high bit being folded into `ql` once.
    """
    n = len(raw) // 176
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 176)
    d = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    dmin = np.ascontiguousarray(a[:, 2:4]).view(np.float16).astype(np.float32).reshape(-1)
    scb = a[:, 4:16]                                   # scales[K_SCALE_SIZE] - the same packed 6-bit pairs
    qh = a[:, 16:48]                                   # qh[QK_K/8]
    qs = a[:, 48:176]                                  # qs[QK_K/2]

    scales = np.empty((n, 8), dtype=np.float32)
    offsets = np.empty((n, 8), dtype=np.float32)
    for g in range(8):
        sc = np.array([get_scale_min_k4(scb[i], g)[0] for i in range(n)], dtype=np.float32)
        m = np.array([get_scale_min_k4(scb[i], g)[1] for i in range(n)], dtype=np.float32)
        scales[:, g] = d * sc
        offsets[:, g] = -(dmin * m)

    codes = np.empty((n, 256), dtype=np.uint8)
    for j in range(4):
        blk = qs[:, j * 32:(j + 1) * 32]
        u1, u2 = 1 << (2 * j), 2 << (2 * j)
        lo = (blk & 0x0F) + np.where(qh & u1, 16, 0).astype(np.uint8)
        hi = (blk >> 4) + np.where(qh & u2, 16, 0).astype(np.uint8)
        codes[:, j * 64:j * 64 + 32] = lo              # group is+0: LOW nibbles
        codes[:, j * 64 + 32:j * 64 + 64] = hi         # group is+1: HIGH nibbles
    return codes, scales, offsets


def from_s8_q5_k(parts):
    codes, scales, offsets = parts
    v = codes.astype(np.float32).reshape(codes.shape[0], 8, 32)
    return (v * scales[:, :, None] + offsets[:, :, None]).reshape(-1)


def to_s8_q6_K(raw: bytes):
    """`Q6_K` -> S8: 16 groups of 16, `scale = d*sc` with an int8 `sc`, codebook bias `-32`.

    `dequantize_row_q6_K` (L1939):
        d = fp16(x.d)
        for n in {0, 128}:                                 # ql += 64, qh += 32, sc += 8 per half
            for l in 0..31:
                is = l/16
                q1 = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                y[l +  0] = d * sc[is + 0] * q1
                y[l + 32] = d * sc[is + 2] * q2
                y[l + 64] = d * sc[is + 4] * q3
                y[l + 96] = d * sc[is + 6] * q4

    Four quads are built from ONE `qh[l]`, two bits each, while the low nibbles come from `ql[l]` and
    `ql[l+32]` - and then the quads land at STRIDES of 32, not contiguously: q1 fills `l+0`, q2 fills `l+32`,
    q3 fills `l+64`, q4 fills `l+96`.  So the scale index is not `l/16` in any simple reading of the quads;
    what it works out to is `(element / 16) % 8` within the half, i.e. plain consecutive 16-element groups.

    Two arithmetic details decide bit-exactness.  `d * sc[is] * q` associates LEFT, so the group scale is
    `fl(d * float(sc))` and the value is one further multiply - not `fl(d * fl(sc*q))`.  And `sc` is `int8_t`,
    so a negative scale is a sign flip and must be preserved, which is why the scale plane is read signed.
    """
    n = len(raw) // 210
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 210)
    ql = a[:, 0:128]                                   # ql[QK_K/2]
    qh = a[:, 128:192]                                 # qh[QK_K/4]
    sc = a[:, 192:208].view(np.int8)                   # scales[QK_K/16], SIGNED
    d = np.ascontiguousarray(a[:, 208:210]).view(np.float16).astype(np.float32).reshape(-1)

    scales = d[:, None] * sc.astype(np.float32)        # `d * sc` first, the leftmost multiply

    codes = np.empty((n, 256), dtype=np.uint8)
    for half in range(2):
        qlb = ql[:, half * 64:(half + 1) * 64]
        qhb = qh[:, half * 32:(half + 1) * 32]
        # (nibble source, qh bit offset) for quads 1..4; nibble 0..31 then 32..63, low then high
        quads = [(qlb[:, 0:32] & 0x0F, 0), (qlb[:, 32:64] & 0x0F, 2),
                 (qlb[:, 0:32] >> 4, 4), (qlb[:, 32:64] >> 4, 6)]
        for qidx, (nib, sh) in enumerate(quads):
            seg = nib | (((qhb >> sh) & 0x03) << 4)
            codes[:, half * 128 + qidx * 32: half * 128 + qidx * 32 + 32] = seg
    return codes, scales


def from_s8_q6_K(parts):
    codes, scales = parts
    v = (codes.astype(np.int32) - 32).astype(np.float32)
    return (v.reshape(-1, 16, 16) * scales[:, :, None]).reshape(-1)


MAPPINGS_HEADER = None   # the registry is built after the functions, at the end of the mappings section


# ---------------------------------------------------------------- the float types (P-forms)
def to_p32_f32(raw: bytes):
    """`F32` -> P32: the identity.  Every other mapping is measured against this one's triviality."""
    v = np.frombuffer(raw, dtype="<f4")
    return v, np.ones(len(v), dtype=np.float32)


def from_p32_f32(parts):
    return parts[0]


def to_p32_bf16(raw: bytes):
    """`BF16` -> P32: a bf16 is the top 16 bits of an f32, so widening shifts left and is exact.

    P32 and not P16: bf16 has the f32 exponent range and fp16 does not, so an fp16 container would send
    large-magnitude weights to infinity.  This is the one case where the wider container is the SAFE one.
    """
    v = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return v.view(np.float32), np.ones(len(v), dtype=np.float32)


def from_p32_bf16(parts):
    return parts[0]


def to_p16_f16(raw: bytes):
    """`F16` -> P16: keep the fp16, widen only to decode.  fp16 -> fp32 is always exact."""
    return np.frombuffer(raw, dtype="<f2"), np.ones(len(raw) // 2, dtype=np.float32)


def from_p16_f16(parts):
    return parts[0].astype(np.float32)


# The registry lives AFTER the functions it names.  It has to: a module-level dict referencing `to_s2`
# before `to_s2` exists raises NameError at import, which is a mistake worth not repeating.
#
# Field meanings, because the C++ packer mirrors this table and a wrong field is a wrong artifact:
#   form        the canonical form the pack stores this type in
#   group_elems elements sharing one scale/offset entry (== block_elems where there is one per block)
#   codebook    how a stored code becomes a NUMBER, before `* scale + offset`; `codebook` values are
#               non-affine and need a table lookup, `None` means the affine `code + code_bias`
#   code_bias   the integer subtracted from the code BEFORE the multiply
#   has_offset  whether the decode adds a per-group offset after the multiply
Mapping = collections.namedtuple(
    "Mapping", "enc dec block_bytes block_elems group_elems form codebook code_bias has_offset")

def to_s2(raw: bytes):
    """Canonicalise `Q2_0` to S2: codes and scales in SEPARATE PLANES, values unchanged.

    Returns (codes, scales) with codes `(nblocks, 64)` uint8 holding ONE 2-bit code per element - the same
    contract every other mapping in this file honours - and scales `(nblocks,)` float32, the fp16 `d` widened
    exactly.  Source byte j//4 holds elements 4b..4b+3 at bits 0, 2, 4, 6, so unpacking is
    `codes[:, t::4] = (byte >> 2t) & 3`.

    This used to return the 16 PACKED bytes per block instead.  It passed every check it was given, because
    `from_s2` unpacked them again, and it was still wrong: `codes.size` is what the pack reads as a tensor's
    element count, and for Q2_0 that came out at 1/4 of the truth.  A representation that is only correct
    when read back by its own decoder is not a canonical form - a pack has to be able to state the element
    count on its own.
    """
    n = len(raw) // BLOCK_BYTES
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, BLOCK_BYTES)
    scales = np.ascontiguousarray(a[:, 0:2]).view(np.float16).astype(np.float32).reshape(-1)
    qb = a[:, 2:BLOCK_BYTES]                                  # (n, 16) packed 2-bit codes
    codes = np.empty((n, QK), dtype=np.uint8)
    for t in range(4):
        codes[:, t::4] = (qb >> (2 * t)) & 0x03
    return np.ascontiguousarray(codes), scales


def from_s2(parts):
    """The decode a kernel would do.  Transcribed from `dequantize_row_q2_0`, including the `- 1`.

    One integer subtract then ONE f32 multiply, in that order, for the reason docs/pack-format.md §3.1 gives:
    folding the bias into the value domain is not bit-exact.
    """
    codes, scales = parts
    v = (codes.astype(np.int32) - 1).astype(np.float32)
    return (v.reshape(-1, QK) * scales[:, None]).reshape(-1)


# The registry lives AFTER the functions it names.  It has to: a module-level dict referencing `to_s2`
# before `to_s2` exists raises NameError at import, which is a mistake worth not repeating.
MAPPINGS = {
    "Q2_0":   Mapping(to_s2,         from_s2,         18,  64,  64, "S2",  None,     -1, False),
    "Q3_K":   Mapping(to_s4_q3_k,    from_s4_q3_k,    110, 256, 16, "S4",  None,     -4, False),
    "Q4_0":   Mapping(to_s4_q4_0,    from_s4_q4_0,    18,  32,  32, "S4",  None,     -8, False),
    "Q4_K":   Mapping(to_s4_q4_k,    from_s4_q4_k,    144, 256, 32, "S4",  None,      0, True),
    "Q5_0":   Mapping(to_s8_q5_0,    from_s8_q5_0,    22,  32,  32, "S8",  None,    -16, False),
    "Q5_K":   Mapping(to_s8_q5_k,    from_s8_q5_k,    176, 256, 32, "S8",  None,      0, True),
    "Q6_K":   Mapping(to_s8_q6_K,    from_s8_q6_K,    210, 256, 16, "S8",  None,    -32, False),
    "Q8_0":   Mapping(to_s8_q8_0,    from_s8_q8_0,    34,  32,  32, "S8",  None,   -128, False),
    "IQ4_NL": Mapping(to_s4_iq4_nl,  from_s4_iq4_nl,  18,  32,  32, "S4",  "IQ4NL",   0, False),
    "IQ4_XS": Mapping(to_s4_iq4_xs,  from_s4_iq4_xs,  136, 256, 32, "S4",  "IQ4NL",   0, False),
    "F32":    Mapping(to_p32_f32,    from_p32_f32,     4,   1,   1, "P32", None,      0, False),
    "BF16":   Mapping(to_p32_bf16,   from_p32_bf16,    2,   1,   1, "P32", None,      0, False),
    "F16":    Mapping(to_p16_f16,    from_p16_f16,     2,   1,   1, "P16", None,      0, False),
}


def reference_values(tname: str, raw: bytes):
    """The value this block MUST decode to - from an implementation other than the one under test.

    Q2_0 goes through `gguf_writer.dequantize_q2_0` and everything else through gguf-py, both reached via
    `_dequant_flat`.  The three float types are the exception and have to bypass it: `_dequant_flat` handles
    them with this project's OWN `_bf16_to_f32` and numpy views, so for those the project's reference would
    be checking itself.  gguf-py implements all three independently, so it is used directly here.
    """
    buf = np.frombuffer(raw, dtype=np.uint8)
    if tname in ("F32", "F16", "BF16"):
        from gguf import quants, GGMLQuantizationType as Q
        return np.asarray(quants.dequantize(buf, Q[tname]), dtype=np.float32)
    return np.asarray(_dequant_flat(tname, buf), dtype=np.float32)


def reference_name(tname: str) -> str:
    """Which implementation the comparison is against - printed, because these are NOT equally strong.

    `gguf-py` is an independent implementation of the same C source.  `project` is this project's own
    decoder, so a pass there is a self-consistency result: it cannot detect a misreading of the format that
    the decoder and the canonicaliser share.  Q2_0 is the only `project` row and it is 31.64 GiB of the
    artifact, so the distinction matters and is therefore never left implicit.
    """
    return "project (Q2_0 decoder) - SELF-CONSISTENCY, oracle-backed by dequant_xcheck" \
        if tname == "Q2_0" else "gguf-py"


def open_shard(path):
    """mmap the shard read-only.  Slices of an mmap are ordinary bytes, which is all the reference needs.

    This replaced `fh.read()`, which loaded all 37.6 GB of shard 1 into RAM.  It worked, which is exactly why
    it was worth removing: a tool that only survives because the machine has 64 GB is not a tool that reports
    its own limits, and it would have gone unnoticed until a machine with less memory ran it.
    """
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    return mm, mm.size()


def data_section_offset(f, filelen: int) -> int:
    """Byte offset of the tensor data, from the tensor table alone.

    `filelen - max(offset + expected_bytes)` needs no header parsing and no dependency on ref/load.py's
    private accessors.
    """
    return filelen - max(x.offset + (x.expected_bytes() or 0) for x in f.tensors)


def run_full(args) -> int:
    """P1.T3 over the WHOLE artifact: every block of every tensor, streamed.

    P1.T3 is stated as "for every tensor of the real GGUF, dequant(canonical) == dequant_ggml(original) bit
    for bit".  A per-block identity is a pure function of that block's bytes, so a sample of blocks can only
    ever be evidence ABOUT a subset; this mode removes the qualifier.  It also reports the DATA-DEPENDENT
    branches each type actually exercised, because those are the paths a sample can miss entirely: a scale
    that never reaches 16 never tests the high bits of `IQ4_XS`'s `ls`, and an hmask whose top nibble is never
    set never tests `Q3_K`'s second-half mask.
    """
    f = G.GGUFFile(pathlib.Path(args.gguf))
    head, filelen = open_shard(args.gguf)
    data_off = data_section_offset(f, filelen)
    print("shard %s: %d tensors, data at byte %d" % (args.gguf, len(f.tensors), data_off))

    by_type = collections.defaultdict(list)
    want = {s.strip() for s in args.types.split(",")}
    si, sn = (int(x) for x in args.split.split("/"))
    if not 0 <= si < sn:
        print("--split %s: need 0 <= I < N" % args.split)
        return 2
    # One index over ALL tensors, not per type: a per-type split would put the whole of a 202-tensor type
    # in one process and leave the others idle.
    for idx, t in enumerate(f.tensors):
        if idx % sn == si:
            by_type[t.type_name].append(t)
    if sn > 1:
        print("split %d/%d: %d of %d tensors in this process"
              % (si, sn, sum(len(v) for v in by_type.values()), len(f.tensors)))
    for tn in sorted(by_type):
        if tn not in MAPPINGS:
            print("  %-8s %5d tensors  NO MAPPING - not covered" % (tn, len(by_type[tn])))
        elif tn not in want:
            print("  %-8s %5d tensors  EXCLUDED by --types" % (tn, len(by_type[tn])))

    total_t = total_b = total_bad = 0
    ok = True
    for tname in sorted(by_type):
        if tname not in MAPPINGS or tname not in want:
            continue
        m = MAPPINGS[tname]
        nblocks_sec = max(1, args.chunk // m.block_bytes)          # WHOLE blocks per chunk
        n_t = n_b = n_bad = 0
        t0 = time.time()
        seen = collections.Counter()
        for t in by_type[tname]:
            if t.elements % m.block_elems:
                print("  %-34s %d elements not a multiple of %d - SKIPPED" % (t.name, t.elements, m.block_elems))
                ok = False
                continue
            nb = t.elements // m.block_elems
            base = data_off + t.offset
            first_bad = None
            for s in range(0, nb, nblocks_sec):
                cnt = min(nblocks_sec, nb - s)
                raw = head[base + s * m.block_bytes: base + (s + cnt) * m.block_bytes]
                if len(raw) != cnt * m.block_bytes:
                    print("  %-34s SHORT READ at block %d - instrument error" % (t.name, s))
                    ok = False
                    break
                ref = reference_values(tname, raw)
                got = m.dec(m.enc(raw))
                if ref.shape != got.shape:
                    print("  %-34s SHAPE %s vs %s - instrument error" % (t.name, ref.shape, got.shape))
                    ok = False
                    break
                ne = np.nonzero(ref != got)[0]
                n_b += cnt
                if ne.size:
                    n_bad += ne.size
                    if first_bad is None:
                        first_bad = (s + int(ne[0]) // m.block_elems, int(ne[0]))
                if args.branches:
                    codes = m.enc(raw)[0]
                    seen["code_hi=%d" % int(codes.max())] += 1
                    seen["code_lo=%d" % int(codes.min())] += 1
            n_t += 1
            if first_bad is not None:
                print("  %-34s FIRST MISMATCH at block %d element %d" % (t.name, first_bad[0], first_bad[1]))
                ok = False
        dt = time.time() - t0
        total_t += n_t
        total_b += n_b
        total_bad += n_bad
        print("  %-8s %5d tensors %9d blocks  %s  %6.1fs  ref=%s"
              % (tname, n_t, n_b, "BIT-EXACT" if not n_bad else "*** %d BAD ***" % n_bad, dt,
                 reference_name(tname)))
        if args.branches and seen:
            print("           branch reach: %s" % "  ".join("%s x%d" % kv for kv in sorted(seen.items())))
    print()
    print("P1.T3 full: %d tensors, %d blocks, %d mismatched elements" % (total_t, total_b, total_bad))
    print("tools/canonical_xcheck.py --full " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=SHARD1)
    ap.add_argument("--types", default=",".join(MAPPINGS),
                    help="source types to test, one mapping each; default is every mapped type")
    ap.add_argument("--tensors", type=int, default=2, help="tensors per type")
    ap.add_argument("--blocks", type=int, default=2048, help="blocks per tensor")
    ap.add_argument("--full", action="store_true",
                    help="P1.T3 over every block of every tensor (overrides --tensors/--blocks)")
    ap.add_argument("--chunk", type=int, default=64 << 20, help="bytes of source per streaming step")
    ap.add_argument("--branches", action="store_true", help="report code range reached per type")
    ap.add_argument("--split", default="0/1",
                    help="I/N: process only tensors whose index %% N == I, so N processes can share the "
                         "work.  Q2_0's reference is a pure-Python loop at ~1.7 MB/s and Q2_0 is 34 GB of "
                         "the shard, so the full run is single-core-bound and worth splitting.")
    args = ap.parse_args()
    if args.full:
        return run_full(args)

    f = G.GGUFFile(pathlib.Path(args.gguf))
    print("shard: %d tensors" % len(f.tensors))
    head, filelen = open_shard(args.gguf)
    data_off = data_section_offset(f, filelen)
    print("data section starts at byte %d (file %d bytes)" % (data_off, filelen))

    ok = True
    covered = 0
    for tname in args.types.split(","):
        tname = tname.strip()
        if tname not in MAPPINGS:
            print("  %s: NO MAPPING in this tool yet" % tname)
            continue
        m = MAPPINGS[tname]
        tensors = [t for t in f.tensors if t.type_name == tname]
        covered += len(tensors)
        print()
        print("--- %s -> %s  codebook=%s bias=%d offset=%s  group=%d  (%d tensors in the shard) ---"
              % (tname, m.form, m.codebook or "affine", m.code_bias, m.has_offset,
                 m.group_elems, len(tensors)))
        for t in tensors[: args.tensors]:
            if t.elements % m.block_elems:
                print("  %-34s %d elements is not a whole number of %d-element blocks - SKIPPED"
                      % (t.name, t.elements, m.block_elems))
                continue
            want = min(args.blocks, t.elements // m.block_elems) * m.block_bytes
            byte_off = data_off + t.offset
            raw = head[byte_off: byte_off + want]
            if len(raw) < want:
                print("  %-34s SHORT READ (%d of %d) - instrument error" % (t.name, len(raw), want))
                ok = False
                continue
            ref = reference_values(tname, raw)
            parts = m.enc(raw)
            got = m.dec(parts)
            if ref.shape != got.shape:
                print("  %-34s SHAPE %s ref vs %s got - instrument error" % (t.name, ref.shape, got.shape))
                ok = False
                continue
            same = np.array_equal(ref, got)
            d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
            print("  %-34s %6d blocks  %s   max|d| %.3e"
                  % (t.name, want // m.block_bytes, "BIT-EXACT" if same else "*** MISMATCH ***", d.max()))
            if not same:
                bad = np.nonzero(ref != got)[0]
                print("      first differing element %d: ref %r got %r" % (bad[0], ref[bad[0]], got[bad[0]]))
                ok = False
            if m.block_elems <= 256:      # a value set is informative for a quantized block, noise for f32
                vals = np.unique(got[: m.block_elems])
                print("      first block's value set: %s" % np.array2string(vals, precision=6))
    print()
    print("mapped types cover %d of %d tensors in this shard" % (covered, len(f.tensors)))
    print("tools/canonical_xcheck.py " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
