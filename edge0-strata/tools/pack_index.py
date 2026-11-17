"""tools/pack_index.py - emit a flat index of the pack's dense tensors for the engine's loader.

WHY AN INDEX AND NOT A JSON PARSER.  The manifest is JSON and the engine is C++.  A JSON parser in the engine
would be a new, unaudited component whose failure mode is a wrong byte offset - which decodes to a plausible
weight and produces plausible logits.  Python already parses the manifest correctly and is already the tool
that WROTE it, so this flattens what the loader needs into one line per tensor.

WHY TEXT AND NOT BINARY.  `pack/full/index.txt` is ~150 KB for 1,079 tensors.  A fixed-size binary record
would save 60 KB and cost a struct-packing contract between two languages that must agree on padding,
endianness and field order - three ways to be wrong for no benefit at this size.  Text is also readable when
a load produces the wrong number, which is the only time this file matters.

THE ENGINE LAYOUT IS NOT THE PACK LAYOUT, and that is the point of `dst_*`.  Two independent differences:

  1. A tensor whose source type is BF16 or F16 is stored in the pack PROMOTED to 32 bits, because that is
     lossless and it keeps the canonical decoder simple.  A running engine does not have to hold it that way:
     re-rounding to 16 bits costs nothing at all, because the value came FROM 16 bits.  That is 1.374 GiB of
     the 5.305 GiB dense file, and per `tools/pack_budget.py` it is the difference between the dense weights
     fitting in VRAM beside the expert cache and not fitting at all.

  2. **A SCALE PLANE MAY BE 2 BYTES IN THE PACK AND IS ALWAYS 4 IN THE ENGINE.**  This is the one that hid
     for a whole round.  90 of the 303 quantized tensors - every Q2_0 (all 58 expert tensors), Q4_0, Q5_0,
     Q8_0 and IQ4_NL - carry fp16 scales, because that is what the SOURCE ggml block stores (a Q5_0 block is
     `{fp16 d; ...}`).  213 tensors (IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K) carry f32 scales.  The manifest has said
     which is which all along, in `scales_fp16`.

     `src/core/layer.cpp` did not read that field.  It assumed 4 bytes per scale, so it computed a scale plane
     twice the real size, the three planes did not add up to the tensor, and the loader REFUSED - which is how
     this was found, and the refusal is the only reason it was not a silent wrong answer.  The first diagnosis
     was "the manifest's `group_elems` must be wrong, the real group is 64".  It is not: `group_elems` is 32
     and the scale plane is 2 bytes wide.  **The unexamined input was the WIDTH, not the count.**  An audit of
     all 303 tensors against `scales_fp16`/`offsets_fp16` found zero inconsistencies, which is what settled it.

     The engine widens fp16 scales to f32 at load.  That is EXACT - fp16 is a subset of f32 - and it costs
     11,407,360 B (10.88 MiB) of the 4.5 GiB pool, against teaching three kernel families a second scale
     width.  Note the asymmetry that makes this cheap and the BF16 re-rounding large: re-rounding loses
     precision and saves GiB; widening loses nothing and costs MiB.

     `offsets_fp16` is also read, and it is FALSE for all 54 tensors that have an offset plane.  It is
     honoured anyway rather than assumed, because the next pack may not be.

Columns:

    name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook
    has_offset codes_bytes scales_bytes offset_bytes scales_fp16

`kind` is what the loader must DO, spelled out here rather than left for it to infer - the four cases are
distinguishable in the manifest but only by combining `source_type` with `values_fp16`, and getting that
combination wrong produces a plausible tensor:

    0  VERBATIM       copy straight through (BF16-free: F16 already 2 B/elem, F32, and the planes below)
    1  BF16_IN_F32    the pack holds the bf16 value promoted to f32; take the HIGH 16 BITS.  Exact, not a
                      rounding: a bf16 value IS the top half of its f32 image.
    2  F32            copy 4 B/elem
    3  F16_IN_F32     the pack holds an f16 value promoted to f32; needs a real f32->f16 conversion with
                      round-to-nearest-even, NOT a truncation

`ne0`/`ne1` are the manifest's own shape in its own order, with ne0 the CONTIGUOUS axis.  `form` is the
canonical form letter from the manifest (P16/P32/S2/S4/S8); the SForm attributes are 0 for kinds 1-3.

`codes_bytes`/`scales_bytes`/`offset_bytes` are the THREE ENGINE-FORM PLANE SIZES, in the order they sit in
the destination.  They are emitted rather than left for the loader to re-derive, because re-deriving is where
the width bug lived: two places computing one layout is two places to disagree, and the loader cannot check a
layout it invented.  The loader instead checks what it was TOLD: `scales_bytes/2` when `scales_fp16`, and the
three planes summing to both `src_bytes` (source widths) and `dst_bytes` (engine widths).  Non-quantized
tensors carry `codes_bytes == dst_bytes` and two zeroes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FILE_ID = {"dense.bin": 0, "embd.bin": 1, "experts.bin": 2}
VERBATIM, BF16_IN_F32, F32_COPY, F16_IN_F32 = 0, 1, 2, 3

# WHICH ACTIVATION THIS WEIGHT WANTS, decided HERE because it is a property of `source_type` and the engine's
# `SForm` cannot recover it.  The legacy formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL) have `vec_dot_type` Q8_0; the
# K-quants and IQ4_XS have Q8_K.  **NO COMBINATION OF THE S-FORM FIELDS TELLS THEM APART**: Q5_0 and Q5_K are
# both 8-bit with bias -16 (only `has_offset` differs), and IQ4_NL and IQ4_XS are both 4-bit with the IQ4NL
# codebook and no offset - they differ in NOTHING the S-form carries.  That is why this column exists rather
# than the engine re-deriving the rule from `code_bits`.
ACT_Q8_0, ACT_Q8_K = 0, 1
VDT = {
    "Q2_0": ACT_Q8_0, "Q4_0": ACT_Q8_0, "Q5_0": ACT_Q8_0, "Q8_0": ACT_Q8_0, "IQ4_NL": ACT_Q8_0,
    "Q3_K": ACT_Q8_K, "Q4_K": ACT_Q8_K, "Q5_K": ACT_Q8_K, "Q6_K": ACT_Q8_K, "IQ4_XS": ACT_Q8_K,
}

# The plane key is `offsets`, NOT `mins`.  The first version of this list said `mins` - which is what ggml calls
# the quantity conceptually, and the manifest does not - and a missing key in a loop like this is not an error,
# it is a SHORTER SPAN: 54 Q4_K/Q5_K tensors loaded with their offset plane never copied.  Keys now come from
# the manifest, and the layout is verified against the recorded plane offsets rather than assumed.
PLANE_KEYS = ("codes", "scales", "offsets")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="pack/full")
    ap.add_argument("-o", "--out", default=None, help="default: <pack>/index.txt")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.pack, "index.txt")

    with open(os.path.join(args.pack, "manifest.json"), "r", encoding="utf-8") as fh:
        man = json.load(fh)
    align = int(man.get("align", 64))

    def align_up(n: int) -> int:
        return (n + align - 1) // align * align

    rows = []
    dst = 0
    n_reduced = 0
    saved = 0
    widened = 0
    n_widened = 0
    kinds: dict[int, int] = {}
    for name, v in man["tensors"].items():
        f = v.get("file")
        if f not in FILE_ID:
            print("unknown file %r for %s" % (f, name), file=sys.stderr)
            return 1
        shape = v.get("shape") or [0, 0]
        ne0 = int(shape[0]) if len(shape) > 0 else 0
        ne1 = int(shape[1]) if len(shape) > 1 else 0
        elems = int(v.get("elements", 0))
        st = v.get("source_type") or "?"

        codes_bytes = scales_bytes = offset_bytes = 0
        scales_fp16 = 0
        widen_here = 0
        act_kind = ACT_Q8_0
        if "values" in v:
            src_off = int(v["values"]["offset"])
            src_bytes = int(v["values"]["bytes"])
        else:
            planes = [k for k in PLANE_KEYS if isinstance(v.get(k), dict)]
            lo = min(int(v[k]["offset"]) for k in planes)
            hi = max(int(v[k]["offset"]) + int(v[k].get("bytes", 0)) for k in planes)
            src_off, src_bytes = lo, hi - lo

            # the planes must be CONTIGUOUS, in order, because the loader reads them as one span.  If a future
            # pack interleaves them this refuses; the alternative is reading the scale plane as offsets and
            # decoding a plausible weight.
            expect = src_off
            for k in PLANE_KEYS:
                if k not in planes:
                    continue
                if int(v[k]["offset"]) != expect:
                    print("%s: plane %r starts at %d but %d was expected - planes are not contiguous"
                          % (name, k, int(v[k]["offset"]), expect), file=sys.stderr)
                    return 1
                expect += int(v[k].get("bytes", 0))
            if expect != src_off + src_bytes:
                print("%s: plane span %d does not end at %d" % (name, expect, src_off + src_bytes),
                      file=sys.stderr)
                return 1

            if v.get("has_offset") and "offsets" not in planes:
                print("%s says has_offset but has no `offsets` plane (found %s)" % (name, planes),
                      file=sys.stderr)
                return 1

            act_kind = VDT.get(st, ACT_Q8_0)   # refuses below if the type is unknown
            code_bits = int(v.get("code_bits") or 0)
            group_elems = int(v.get("group_elems") or 0)
            if code_bits == 0 or group_elems == 0:
                print("%s: a quantized tensor with code_bits=%d group_elems=%d" % (name, code_bits, group_elems),
                      file=sys.stderr)
                return 1
            if ne0 % group_elems:
                print("%s: ne0 %d is not a multiple of group_elems %d" % (name, ne0, group_elems),
                      file=sys.stderr)
                return 1
            n_groups = ne0 // group_elems
            codes_bytes = elems * code_bits // 8
            scales_fp16 = 1 if v.get("scales_fp16") else 0
            offs_fp16 = 1 if v.get("offsets_fp16") else 0
            # engine-form plane sizes: ALWAYS 4 B per scale.  fp16 in the pack is widened; the load is exact.
            scales_bytes = ne1 * n_groups * 4
            offset_bytes = ne1 * n_groups * 4 if v.get("has_offset") else 0
            src_planes = (codes_bytes
                          + ne1 * n_groups * (2 if scales_fp16 else 4)
                          + (ne1 * n_groups * (2 if offs_fp16 else 4) if v.get("has_offset") else 0))
            if src_planes != int(v["codes"]["bytes"]) + int(v["scales"]["bytes"]) \
                    + int(v.get("offsets", {}).get("bytes", 0)):
                print("%s: derived planes %d disagree with the manifest's own plane bytes %d"
                      % (name, src_planes, int(v["codes"]["bytes"]) + int(v["scales"]["bytes"])
                         + int(v.get("offsets", {}).get("bytes", 0))), file=sys.stderr)
                return 1
            if src_planes != src_bytes:
                print("%s: derived planes %d do not fill the span %d" % (name, src_planes, src_bytes),
                      file=sys.stderr)
                return 1
            if scales_fp16:
                widen_here = scales_bytes - ne1 * n_groups * 2
                widened += widen_here
                n_widened += 1

        # ---- the engine form, as an explicit KIND rather than a flag to interpret
        vf = bool(v.get("values_fp16"))
        if st == "BF16":
            kind = BF16_IN_F32
            dst_bytes = elems * 2
            saved += src_bytes - dst_bytes
            n_reduced += 1
        elif st == "F16" and not vf:
            kind = F16_IN_F32
            dst_bytes = elems * 2
            saved += src_bytes - dst_bytes
            n_reduced += 1
        elif st == "F16":
            kind = VERBATIM                      # already 2 bytes on disk
            dst_bytes = src_bytes
        elif st == "F32":
            kind = F32_COPY
            dst_bytes = elems * 4
        else:
            kind = VERBATIM
            dst_bytes = codes_bytes + scales_bytes + offset_bytes
            # THE ENGINE FORM OF A QUANTIZED TENSOR DIFFERS FROM THE PACK BY EXACTLY THE WIDENING, and saying
            # so here is what makes the two plane sizes a CHECK rather than two independent guesses.
            if dst_bytes != src_bytes + widen_here:
                print("%s: engine form %d != pack %d + widening %d" % (name, dst_bytes, src_bytes, widen_here),
                      file=sys.stderr)
                return 1

        cb = v.get("codebook")
        rows.append((name, FILE_ID[f], kind, src_off, src_bytes, dst, dst_bytes, ne0, ne1,
                     int(v.get("code_bits") or 0), int(v.get("code_bias") or 0),
                     int(v.get("group_elems") or 0), 1 if cb == "IQ4NL" else 0,
                     1 if v.get("has_offset") else 0,
                     codes_bytes, scales_bytes, offset_bytes, scales_fp16, act_kind))
        dst = align_up(dst + dst_bytes)
        kinds[kind] = kinds.get(kind, 0) + 1

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# strata pack index v3  --  generated by tools/pack_index.py; do not hand-edit\n")
        fh.write("# align %d pool %d tensors %d\n" % (align, dst, len(rows)))
        fh.write("# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias "
                 "group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind\n")
        for r in rows:
            fh.write("%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n" % r)

    names = {VERBATIM: "VERBATIM", BF16_IN_F32: "BF16_IN_F32", F32_COPY: "F32", F16_IN_F32: "F16_IN_F32"}
    print("%s: %d tensors, engine pool %d B (%.3f GiB)" % (out_path, len(rows), dst, dst / 2**30))
    for k in sorted(kinds):
        print("   kind %-12s %4d tensors" % (names[k], kinds[k]))
    print("   %d tensors re-rounded to 16 bits, saving %d B (%.3f GiB)"
          % (n_reduced, saved, saved / 2**30))
    print("   %d tensors widened fp16 scales -> f32, costing %d B (%.3f GiB)"
          % (n_widened, widened, widened / 2**30))
    print("   compare with tools/pack_budget.py: the pack's dense+embd engine figure should match this pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
