"""tools/iq_pack.py - plan v0.3 P6: a native pack for any of the model files (Q2_0, IQ2_XS, IQ3_XXS).

    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The i-quant experts cannot be re-expressed in the Q2_0 pack form, so this pack keeps every quantized tensor in
its GGUF form:

  experts.bin          optional (--experts-bin): per layer, 512 blobs of [gate rows | up rows | down rows], the
                       raw GGUF slices.  Blob size is per layer (the files mix IQ1_M ... IQ3_S gate/up and Q2_0 /
                       IQ4_NL down).  experts.bin.src.json says which source it was cut from (the shards' names and
                       sizes, the hash of native_experts.txt); an experts.bin is reused only when that matches.
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard];
                       written last, so a pack without it is not finished
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: the BF16/F16/F32 tensors in the form the engine reads them (index kinds 4/5/2;
                       3 = F32 narrowed to F16 by the loader) - as the GGUF stores them, or converted (FORM below).
  conversions.json     every tensor that was converted: shard, source/destination type, method, whether it was
                       exact, the largest absolute error, the source bytes' sha256 (schema 1).
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.

Split files: every shard of the model is read (<name>-0000N-of-0000M.gguf beside --gguf; a missing shard is an
error), so the layers may be split anyhow (Swift 1.5's GGUFs put layers 13-47 in shard 2 and the PLE table in
shard 1).  A layer whose experts are not in shard 1 names its shard in native_experts.txt (v3).  A shard boundary
may even fall inside a layer (Unsloth's UD-Q4_K_XL: layer 11's down in shard 2, its gate and up in shard 3): that
layer's shard column is per role, `gate,up,down` (an empty field = the --gguf shard), and only then is the file
v4, so an older engine refuses it instead of misreading it.  Every other pack stays v3, byte for byte.  Router
tensors stored as F32 whose values are exactly BF16 (Swift 1.5) are written as BF16, the form the engine's router
takes (so are the other F32 tensors the engine reads as BF16, when exact); anything else is refused.

For ordinary quants, --compat-bf16 dequantizes the small projections that the engine reads as BF16, using
round-to-nearest-even. This introduces BF16 rounding; it does not reconstruct the original full-precision
weights. Experts, native attention projections, token embeddings, a Q2_0/Q8_0 PLE key and the disk-backed PLE
table stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
NOT_IN_PACK = {"per_layer_token_embd.weight"}      # the 28.8 GB PLE table: read from its GGUF by the engine

# The float form the engine reads each pack tensor in (name without `blk.N.`; from eddoursul/Strata's FORM table).
# The residual, router, GDN, QSA and PLE kernels read these small projections as BF16, the norms as F32 and the PLE
# conv as F16.  GSQ-RCO files already store them that way; other files do not (Unsloth's UD-Q4_K_XL: the routers,
# SSM gates and injections F32, the hyper-connection projections and the PLE value Q8_0, the PLE conv F32), so a
# tensor stored otherwise is converted, and every conversion is recorded in conversions.json:
#   F32 -> BF16 whose values are exactly BF16 (low 16 bits zero)    exact, by default
#   F32 with other values, F16 or a quantized type -> BF16          rounds (nearest-even): only with --compat-bf16
#   F32 -> F16 (ple_conv1d)                                         index kind 3, narrowed by the loader (#255),
#                                                                   by default - the kernel reads nothing else
#   F16 / BF16 -> F32                                               exact widening, by default
# Tensors not named here are written as stored (floats) or served from the GGUF (quantized), as before.
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}
# PLE key encodings left in the GGUF for the engine's native PLE key: the formats NativeDense serves (Q2_0; Q8_0 in
# UD-Q4_K_XL; IQ3_XXS and IQ4_XS, #381).  Other quantized keys take the BF16 path (--compat-bf16), as before.
NATIVE_PLE_KEY = {"Q2_0", "Q8_0", "IQ3_XXS", "IQ4_XS"}
KIND = {"BF16": "4", "F16": "5", "F32": "2"}


def form_of(name: str):
    return FORM.get(re.sub(r"^blk\.\d+\.", "", name))


def needs_bf16(name: str, type_name: str) -> bool:
    """Whether the engine reads `name` as BF16 from the pack (not natively from the GGUF)."""
    if name == "blk.1.ple_key.weight" and type_name in NATIVE_PLE_KEY:
        return False
    return form_of(name) == "BF16"


def dequantize(raw: np.ndarray, type_name: str) -> np.ndarray:
    """The tensor's values as float32 (float types exactly, quantized ones through gguf-py's dequantizer)."""
    if type_name == "F32":
        return raw.view(np.float32)
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    if type_name == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    return quants.dequantize(raw, Q[type_name]).astype(np.float32).reshape(-1)


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    values = quants.dequantize(raw, Q[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    # ggml's round-to-nearest-even conversion, including correct halfway rounding.
    return quants.quantize(values, Q.BF16).tobytes()


def f16_values(values: np.ndarray, name: str) -> np.ndarray:
    """values -> F16, nearest-even (numpy's conversion, which is the engine loader's f16_from_f32); refuses a value
    that is not finite or overflows F16."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: cannot convert non-finite weights to F16")
    with np.errstate(over="ignore"):
        h = values.astype(np.float16)
    if not np.isfinite(h).all():
        raise ValueError(f"{name}: a value is beyond F16's range, which the engine reads it as")
    return h


class Model:
    """All shards of one model: name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path):
        import re
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        paths = [first]
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing model shards (wait for the download): " + ", ".join(missing))
        self.paths = paths
        self.files = [G.GGUFFile(p) for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        check_split(self.files)
        self.where = {}
        for p, g in zip(paths, self.files):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise ValueError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ValueError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)


def check_split(files) -> None:
    """The split keys, as the engine checks them (strata::GgufModel): shard 1 carries the metadata, and every shard
    declares split.count / split.no (and split.tensors.count) consistently - a shard of another model, or a shard
    renamed into the family, is refused rather than mixed in."""
    n = len(files)
    meta0 = files[0].metadata
    if n == 1:
        if int(meta0.get("split.count", 1)) > 1:
            raise ValueError(f"{files[0].path.name} is shard 1 of {meta0['split.count']}, but its name has no "
                             "-00001-of-0000N.gguf to find the others by")
        return
    if "general.architecture" not in meta0:
        raise ValueError(f"{files[0].path.name} has no general.architecture; the first shard of a split model "
                         "carries the metadata")
    total = meta0.get("split.tensors.count")
    for i, g in enumerate(files):
        md = g.metadata
        if md.get("split.count") != n or md.get("split.no") != i or \
                (total is not None and md.get("split.tensors.count") != total):
            raise ValueError(f"{g.path.name} does not declare itself shard {i + 1} of {n} of this model "
                             "(split.count / split.no / split.tensors.count)")
    if total is not None and sum(len(g.tensors) for g in files) != total:
        raise ValueError(f"the {n} shards hold {sum(len(g.tensors) for g in files)} tensors, but "
                         f"split.tensors.count is {total}")


ROLES = ("gate", "up", "down")
N_EXPERT = 512
ALIGN = 64
NVFP4_TAIL = 16          # {s_gate, s_up, s_down, 0} after an NVFP4 expert blob


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def tensor_bytes(mm, g, t) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def convert(name: str, type_name: str, raw: np.ndarray, compat_bf16: bool):
    """The pack form of one tensor the engine reads from dense.bin: (kind, stored bytes, destination bytes,
    record or None), or an error string.  `record` describes a conversion for conversions.json."""
    form = form_of(name)
    if form is None or form == type_name:
        return KIND[type_name], raw.tobytes(), raw.nbytes, None
    values = dequantize(raw, type_name)
    rec = {"src_type": type_name, "dst_type": form}
    if form == "BF16":
        u = values.view(np.uint32) if type_name == "F32" else None
        if u is not None and not np.count_nonzero(u & 0xFFFF):
            if not np.isfinite(values).all():
                return f"{name} is F32 with non-finite values; refused rather than converted"
            data = (u >> 16).astype("<u2").tobytes()       # the exact BF16 values
            rec.update(method="f32->bf16: the values are BF16's (low 16 bits zero)", exact=True, max_abs_err=0.0)
            return "4", data, len(data), rec
        if not compat_bf16:
            what = "F32 with values that are not BF16" if type_name == "F32" else type_name
            return f"{name} is {what}, but the engine requires BF16; use --compat-bf16 (rounds to nearest-even)"
        data = bf16_bytes(raw, type_name)
        got = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        rec.update(method="dequantize->bf16 round-to-nearest-even (--compat-bf16)" if type_name not in FLOAT else
                   "%s->bf16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        rec.update(exact=bool(np.array_equal(got, values)),
                   max_abs_err=float(np.max(np.abs(got.astype(np.float64) - values))) if values.size else 0.0)
        return "4", data, len(data), rec
    if form == "F16":
        if type_name != "F32" and not compat_bf16:
            return f"{name} is {type_name}, but the engine reads F16; use --compat-bf16 (rounds to nearest-even)"
        h = f16_values(values, name)
        err = float(np.max(np.abs(h.astype(np.float64) - values))) if values.size else 0.0
        rec.update(exact=bool(np.array_equal(h.astype(np.float32), values)), max_abs_err=err)
        if type_name == "F32":
            # The F32 bytes go into dense.bin as index kind 3 and the loader narrows them to F16 (nearest-even), the
            # form the PLE conv kernel reads (#255, gopinath87607): 4 B/elem in, 2 B/elem out.
            rec.update(method="f32->f16 by the loader (index kind 3, round-to-nearest-even)")
            return "3", raw.tobytes(), raw.nbytes // 2, rec
        rec.update(method="%s->f16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        data = h.astype("<f2").tobytes()
        return "5", data, len(data), rec
    # form == "F32"
    if type_name not in ("F16", "BF16") and not compat_bf16:
        return f"{name} is {type_name}, but the engine reads F32; use --compat-bf16"
    data = values.astype("<f4").tobytes()
    rec.update(method="%s->f32 widening" % type_name.lower(), exact=True, max_abs_err=0.0)
    return "2", data, len(data), rec


def index_standalone(src, out, model: Model, compat_bf16: bool = False) -> int:
    """Every non-expert tensor of the model: the floats the engine reads from the pack into dense.bin in the form it
    reads them (FORM; converted when stored otherwise, see above), quantized ones served natively from the GGUF."""
    todo, problems = [], []
    for name, (g, t, mm, p) in model.where.items():
        if is_expert(t.name) or t.name in NOT_IN_PACK:
            continue
        if len(t.shape) > 2:
            print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
            return 1
        # quantized: served from the GGUF unless the engine reads it from the pack (FORM), which takes
        # --compat-bf16 to dequantize - except the native PLE key encodings
        form = form_of(name)
        native = t.type_name not in FLOAT and (form is None or (
            name == "blk.1.ple_key.weight" and t.type_name in NATIVE_PLE_KEY))
        if t.type_name not in FLOAT and not native and not compat_bf16:
            problems.append(f"{name} is {t.type_name}, but the engine requires {form}; use --compat-bf16")
        todo.append((name, g, t, mm, p, native))
    # refused before anything is written: the previous pack stays as it was
    if problems:
        for m in problems[:8]:
            print(m)
        if len(problems) > 8:
            print("... and %d more" % (len(problems) - 8))
        return 1
    rows, at = [], 0
    served = 0
    converted, records = [], []
    with open(out / "dense.bin.tmp", "wb") as fo:
        for name, g, t, mm, p, native in todo:
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            if native:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
                continue
            raw = tensor_bytes(mm, g, t)
            try:
                got = convert(name, t.type_name, raw, compat_bf16)
            except ValueError as e:
                got = str(e)
            if isinstance(got, str):
                print(got)
                fo.close()
                (out / "dense.bin.tmp").unlink(missing_ok=True)
                return 1
            kind, data, dst_bytes, rec = got
            if rec is not None:
                rec = {"name": name, "shard": p.name, **rec,
                       "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
                records.append(rec)
                if "--compat-bf16" in rec["method"]:          # the rounding the flag allowed
                    converted.append({"name": name, "source_type": t.type_name, "bytes": len(data)})
            rows.append([t.name, "0", kind, str(at), str(len(data)), "0", str(dst_bytes), str(ne0), str(ne1),
                         "0", "0", "1"] + ["0"] * 7)
            fo.write(data)
            pad = (-len(data)) % ALIGN
            fo.write(b"\0" * pad)
            at += len(data) + pad
    write_index(out, rows, src, served, 0, publish=False)
    conv = out / "conversions.json.tmp"
    conv.write_text(json.dumps({
        "schema": 1, "tool": "tools/iq_pack.py", "compat_bf16": compat_bf16,
        "source_shards": [{"name": q.name, "size": s} for q, s in zip(model.paths, model.sizes)],
        "tensors": records,
    }, indent=1) + "\n", encoding="utf-8")
    # published together, and the completion marker (native_experts.txt, written last by main) goes first: a stop
    # from here on leaves a pack that setup and the engine see as unfinished, never a new dense.bin under an old
    # index or the reverse
    (out / "native_experts.txt").unlink(missing_ok=True)
    (out / "dense.bin.tmp").replace(out / "dense.bin")
    (out / "index.txt.tmp").replace(out / "index.txt")
    conv.replace(out / "conversions.json")
    if records:
        exact = sum(1 for r in records if r["exact"])
        print("conversions.json: %d tensors converted to the engine's form (%d exact, %d rounded; max |err| %.3g)"
              % (len(records), exact, len(records) - exact, max(r["max_abs_err"] for r in records)))
        for r in records:
            if r["dst_type"] == "F16":
                print("narrowed to F16 by the loader: %s (%s, index kind 3)" % (r["name"], r["method"]))
    if compat_bf16:
        (out / "compat-bf16.json").write_text(json.dumps({
            "source": str(src), "rounding": "nearest-even", "tensors": converted,
        }, indent=2) + "\n", encoding="utf-8")
        print("compat-bf16: %d tensors, %.2f GiB; expert and PLE table bytes unchanged"
              % (len(converted), sum(t["bytes"] for t in converted) / 2**30))
    return 0


def write_index(out, rows, src, served, n_extra, publish=True):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt.tmp", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    if publish:
        (out / "index.txt.tmp").replace(out / "index.txt")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(a, src, base, out, g, T, mm) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bg = G.GGUFFile(base_src)
    BT = {t.name: t for t in bg.tensors}
    bmm = np.memmap(base_src, dtype=np.uint8, mode="r")
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = T.get(name), BT.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(tensor_bytes(mm, g, t), tensor_bytes(bmm, bg, bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(tensor_bytes(mm, g, t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def expert_layout(model: Model, src: pathlib.Path):
    """The pack's expert table: (layout rows, native_experts.txt text, n_expert, total bytes), or an error string.
    Each role is resolved by name in whichever shard holds it, and its offset is absolute in THAT shard: two
    shards do not start their data section at the same byte, so one role's data_start must not be used for
    another's (per-role data_start as in #255, gopinath87607)."""
    T = {n: w[1] for n, w in model.where.items()}
    exps = [n for n in T if n.startswith("blk.") and n.endswith("_exps.weight")]
    if not exps:
        return "the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)"
    n_layers = 1 + max(int(n.split(".")[1]) for n in exps)
    n_expert = int(T["blk.0.ffn_gate_inp.weight"].shape[1])   # router rows = experts kept (pruned models ship < 512)
    if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != n_expert for l in range(n_layers)):
        return "the routers disagree on the expert count; a per-layer pruned model cannot be packed"
    layout, lines, offset, n_split = [], [], 0, 0
    for l in range(n_layers):
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
        if any(n not in T for n in names):
            return "layer %d: missing %s" % (l, ", ".join(n for n in names if n not in T))
        ts = [T[n] for n in names]
        if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            return "layer %d: an expert tensor is not [*, *, %d] of whole blocks" % (l, n_expert)
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            return "layer %d: gate and up differ in type" % l
        blob = per[0] + per[1] + per[2]
        nvfp4 = ts[0].type_name == "NVFP4" or ts[2].type_name == "NVFP4"
        if nvfp4:
            # NVFP4 (ModelOpt) carries one F32 weight_scale_2 per expert and projection, which the block format has
            # no room for. Each blob gets a 16-byte tail {s_gate, s_up, s_down, 0} so every consumer that already
            # moves whole blobs (the arena, the device cache, the prefill staging, a second GPU) moves the scales
            # with the weights; the gate/up kernels read it. See tools/nvfp4_convert.py.
            # a mixed expert (e.g. NVFP4 gate/up + Q8_0 down) has the tail too, with 1.0 for the other projection
            if any(t.type_name == "NVFP4" and "blk.%d.ffn_%s_exps.scale" % (l, r) not in T for t, r in zip(ts, ROLES)):
                return "layer %d: NVFP4 experts without their .scale tensors" % l
            blob += NVFP4_TAIL
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        ws = [model.where[n] for n in names]
        files = ["" if w[3] == src else w[3].name for w in ws]
        column = files[0] if len(set(files)) == 1 else ",".join(files)
        n_split += len(set(files)) != 1
        if nvfp4:   # no GGUF offsets: experts.bin is the only source that carries the scale tails
            lines.append("%d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob))
            offset += blob * n_expert
            continue
        line = "%d %d %d %d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob,
                                            *[w[0].data_start + w[1].offset for w in ws])
        lines.append(line + ("" if not column else " " + column))
        offset += blob * n_expert
    if n_split:
        head = ("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                "[shard | gate,up,down] (n_expert %d, total %d; absolute offsets in %s, or in the named shard "
                "beside it - per role where the column is gate,up,down)\n" % (n_expert, offset, src.name))
        print("%d layer(s) have their gate/up/down in different shards: native_experts.txt v4, per-role shard "
              "column for %s" % (n_split, ", ".join(str(l) for l, *_ in layout if "," in lines[l])))
    else:
        head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                "(n_expert %d, total %d; absolute offsets in %s, or in the named shard beside it)\n"
                % (n_expert, offset, src.name))
    return layout, head + "".join(line + "\n" for line in lines), n_expert, offset


NATIVE_HEAD = re.compile(r"^# strata native experts v(\d+):")


def native_experts_v5(text: str) -> str:
    """native_experts.txt as expert_layout writes it (v3, or v4 with per-role shards) -> the same table marked v5
    (Hadamard-folded experts, hadamard.txt beside it), or an error string naming the version it cannot mark."""
    head, rest = text.split("\n", 1) if "\n" in text else (text, "")
    m = NATIVE_HEAD.match(head)
    if not m or m.group(1) not in ("3", "4"):
        found = "v" + m.group(1) if m else "without a version header (%r)" % head[:40]
        return ("cannot mark native_experts.txt %s as v5 for Hadamard-folded experts: only v3 and v4 tables are "
                "supported" % found)
    return "# strata native experts v5:" + head[m.end():] + " - v5: Hadamard-folded experts, see hadamard.txt\n" + rest


HADAMARD_ROLE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def hadamard_spec(meta: dict, layout, n_embd: int, n_ff: int):
    """prism.hadamard.* (Agention's APR files: routed experts stored in a block-Hadamard-rotated basis; the runtime
    must rotate their input activations, see include/strata/kernels/cpu/hadamard.hpp) -> the text of hadamard.txt,
    None for a model without the metadata, or an error string.  Anything this engine cannot apply is refused: a
    folded weight it would run unrotated computes plausible garbage."""
    if "prism.hadamard.version" not in meta:
        return None
    if int(meta["prism.hadamard.version"]) != 1:
        return "unsupported prism.hadamard.version %s" % meta["prism.hadamard.version"]
    block = int(meta.get("prism.hadamard.block_size", 0))
    if block <= 1 or block & (block - 1):
        return "invalid prism.hadamard.block_size %r" % block
    if meta.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard":
        return "unsupported prism.hadamard.transform %r" % meta.get("prism.hadamard.transform")
    if meta.get("prism.hadamard.axis") != "input-last-dimension":
        return "unsupported prism.hadamard.axis %r" % meta.get("prism.hadamard.axis")
    mode = meta.get("prism.hadamard.sign_mode")
    if mode not in ("identity", "explicit"):
        return "unsupported prism.hadamard.sign_mode %r" % mode
    if meta.get("prism.hadamard.inverse_weight_names"):
        return ("prism.hadamard.inverse_weight_names (a rotated token embedding) is not supported by this engine: "
                "%s" % ", ".join(meta["prism.hadamard.inverse_weight_names"][:4]))
    names = meta.get("prism.hadamard.weight_names") or []
    if not names:
        return "prism.hadamard.weight_names is empty"
    n_layers = len(layout)
    roles = [0] * n_layers
    for name in names:
        m = HADAMARD_ROLE.match(name)
        if not m or int(m.group(1)) >= n_layers:
            return ("prism.hadamard folds %s; this engine applies the activation rotation to the routed experts "
                    "(blk.N.ffn_{gate,up,down}_exps) only" % name)
        bit = {"gate": 1, "up": 2, "down": 4}[m.group(2)]
        if roles[int(m.group(1))] & bit:
            return "duplicate prism.hadamard weight %s" % name
        roles[int(m.group(1))] |= bit
    for l, r in enumerate(roles):
        if bool(r & 1) != bool(r & 2):
            return "layer %d: prism.hadamard folds only one of gate/up (they share one input)" % l
        if r and 42 in (layout[l][1] if r & 3 else -1, layout[l][2] if r & 4 else -1):
            return "layer %d: a Hadamard-folded Q2_0 expert (the engine's Q2_0 rows cannot rotate)" % l
    lines = ["# strata hadamard v1: prism.hadamard (normalized-sylvester-walsh-hadamard, input-last-dimension, "
             "sign_mode %s), written by tools/iq_pack.py" % mode, "block %d" % block]
    widths = {n_embd for r in roles if r & 3} | {n_ff for r in roles if r & 4}
    if any(w % block for w in widths):
        return "prism.hadamard block %d does not divide the expert widths %s" % (block, sorted(widths))
    if mode == "explicit":
        sw = [int(w) for w in meta.get("prism.hadamard.sign_widths") or []]
        sv = [int(v) for v in meta.get("prism.hadamard.sign_values") or []]
        if not sw or sum(sw) != len(sv) or any(w <= 0 or w % block for w in sw) or len(set(sw)) != len(sw):
            return "prism.hadamard sign_widths / sign_values are inconsistent"
        if any(v not in (1, -1) for v in sv):
            return "prism.hadamard sign values must be +1/-1"
        at, table = 0, {}
        for w in sw:
            table[w] = sv[at:at + w]
            at += w
        missing = sorted(widths - set(table))
        if missing:
            return "prism.hadamard has no sign vector for width(s) %s" % missing
        for w in sorted(table):
            lines.append("signs %d %s" % (w, " ".join(str(v) for v in table[w])))
    for l, r in enumerate(roles):
        if r:
            lines.append("layer %d %d %d %d" % (l, r & 1, (r >> 1) & 1, (r >> 2) & 1))
    return "\n".join(lines) + "\n"


def experts_source(model: Model, text: str, total: int) -> dict:
    """What experts.bin is cut from: the shards (names and sizes) and the hash of native_experts.txt (every
    per-role file and offset, the formats and the blob sizes).  A same-size experts.bin of another model or
    another packing is not this one."""
    return {"schema": 1,
            "shards": [{"name": p.name, "size": s} for p, s in zip(model.paths, model.sizes)],
            "native_experts_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": total}


def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="dequantize small non-native projections to BF16 for ordinary Qwen4Exp GGUFs "
                         "(rounds weights; leaves experts and the PLE table unchanged)")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    if a.compat_bf16 and a.base:
        ap.error("--compat-bf16 cannot reuse --base dense weights")
    # HF snapshot files are symlinks to hash-named blobs. Keep the shard filename for discovery: .absolute(), not
    # .resolve(), which would follow the link to the blob and lose the -0000N-of-0000M name.
    src = pathlib.Path(a.gguf).absolute()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    model = Model(src)
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    # ---- the expert table first: a model that cannot be packed is refused before any file of the pack changes
    got = expert_layout(model, src)
    if isinstance(got, str):
        print(got)
        return 1
    layout, text, n_expert, offset = got
    if any(t.type_name == "NVFP4" for *_, ts in layout for t in ts):
        a.experts_bin = True   # the GGUF holds the weights without the scale tails: the engine must read experts.bin
    # ---- prism.hadamard: rotated experts need hadamard.txt beside native_experts.txt, which becomes v5 so an engine
    # that cannot rotate refuses the pack (instead of running the folded experts on unrotated activations)
    t0 = layout[0][5]
    had = hadamard_spec(g.metadata, layout, int(t0[0].shape[0]), int(t0[2].shape[0]))
    if isinstance(had, str) and not had.startswith("# strata hadamard"):
        print("prism.hadamard: " + had)
        return 1
    if had is not None:
        text = native_experts_v5(text)
        if not text.startswith("# strata native experts v5:"):
            print("prism.hadamard: " + text)
            return 1
        print("prism.hadamard: %d layer(s) of Hadamard-folded experts (block %s, %s signs): hadamard.txt, "
              "native_experts.txt v5" % (sum(1 for ln in had.splitlines() if ln.startswith("layer ")),
                                         g.metadata["prism.hadamard.block_size"],
                                         g.metadata["prism.hadamard.sign_mode"]))
    path = out / "experts.bin"
    sidecar = out / "experts.bin.src.json"
    want = experts_source(model, text, offset)
    reuse = path.exists() and path.stat().st_size == offset and read_json(sidecar) == want

    def nvfp4_tails(l, ts):
        """(n_expert, NVFP4_TAIL) bytes {s_gate, s_up, s_down, 0} of a layer with an NVFP4 projection; a projection in
        another format gets exactly 1.0 (the engine multiplies by the tail unconditionally). None for a bad scale."""
        s = [np.frombuffer(model.bytes("blk.%d.ffn_%s_exps.scale" % (l, r)).tobytes(), dtype=np.float32)
             if t.type_name == "NVFP4" else np.ones(n_expert, dtype=np.float32) for t, r in zip(ts, ROLES)]
        tail = np.zeros((n_expert, NVFP4_TAIL // 4), dtype=np.float32)
        tail[:, 0], tail[:, 1], tail[:, 2] = s[0], s[1], s[2]
        if not np.isfinite(tail).all() or (tail[:, :3] <= 0).any():
            print("layer %d: a non-finite or non-positive NVFP4 scale" % l)
            return None
        return tail.view(np.uint8)

    def layer_blobs(l, blob, ts):
        """(n_expert, blob): gate | up | down [| the NVFP4 scale tail] per expert; None for a bad NVFP4 scale."""
        parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
        if any(t.type_name == "NVFP4" for t in ts):
            tail = nvfp4_tails(l, ts)
            if tail is None:
                return None
            parts.append(tail)
        chunk = np.concatenate(parts, axis=1)
        assert chunk.shape == (n_expert, blob)
        return chunk

    if reuse:
        # the sidecar names the shards by name and size, which a re-quantized or fine-tuned checkpoint of the same
        # geometry packed into the same folder can share: compare the first and last blobs of the first, middle and
        # last layers with the GGUF too (six blobs, so an unchanged pack is still reused at once)
        with open(path, "rb") as f:
            for l, gt, dt, off, blob, ts in (layout[0], layout[len(layout) // 2], layout[-1]):
                for e in (0, n_expert - 1):
                    f.seek(off + e * blob)
                    want_blob = b"".join(model.bytes(t.name).reshape(n_expert, -1)[e].tobytes() for t in ts)
                    if any(t.type_name == "NVFP4" for t in ts):
                        tail = nvfp4_tails(l, ts)
                        want_blob += tail[e].tobytes() if tail is not None else b""
                    if f.read(blob) != want_blob:
                        reuse = False
        if not reuse:
            print("%s matches %s but not this GGUF's expert blobs" % (path, sidecar.name))
    if path.exists() and not reuse and not a.experts_bin:
        if sidecar.exists():
            print("%s was cut from another source than this model (%s): the engine would read it instead of "
                  "the GGUF.  Delete it, or rerun with --experts-bin to rewrite it." % (path, sidecar.name))
            return 1
        print("warning: %s has no %s, so nothing says it belongs to this model; the engine reads it instead of "
              "the GGUF (rerun with --experts-bin to rewrite it)" % (path, sidecar.name))
    if a.base:
        if any(w[3] != src for w in model.where.values() if not w[1].name in NOT_IN_PACK):
            print("--base needs a model whose tensors are all in shard 1")
            return 1
        (out / "native_experts.txt").unlink(missing_ok=True)
        rc = index_from_base(a, src, base, out, g, {t.name: t for t in g.tensors}, mm)
    else:
        rc = index_standalone(src, out, model, a.compat_bf16)
    if rc:
        return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts.  native_experts.txt is written to a temporary name and renamed only when every layer is
    # in: a stop part-way (a layer split across shards, #171) left a partial native_experts.txt that the next setup
    # run took as a finished pack (#172).  It is the pack's completion marker, so it is published last.
    htmp = out / "hadamard.txt.tmp"
    if had is not None:
        htmp.write_text(had, encoding="utf-8", newline="\n")
        htmp.replace(out / "hadamard.txt")
    else:
        (out / "hadamard.txt").unlink(missing_ok=True)      # a previous Hadamard model's, packed into this folder
    tmp = out / "native_experts.txt.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(text)
    tmp.replace(out / "native_experts.txt")
    if a.skip_experts or not a.experts_bin:
        if path.exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    if reuse:
        print("experts.bin was cut from this model's shards (%s); not rewritten" % sidecar.name)
        return 0
    # written under a temporary name and renamed when complete, then the sidecar: an interrupted write leaves no
    # experts.bin, and an experts.bin without its sidecar is never reused
    sidecar.unlink(missing_ok=True)
    part = out / "experts.bin.tmp"
    progress = out / "experts.bin.tmp.json"
    # A stop part-way (a kill, a power cut) used to cost the whole write again (40 GB, minutes).  The finished layers are kept:
    # after every layer the file is flushed to disk and experts.bin.tmp.json says how many layers are in it and which source they were
    # cut from.  The next run continues at that layer when the source is the same and the last kept layer still matches the GGUF.
    start = 0
    if part.exists():
        done = read_json(progress)
        k = done.get("layers") if isinstance(done, dict) and done.get("want") == want else None
        if isinstance(k, int) and 0 < k <= len(layout) and part.stat().st_size >= (layout[k][3] if k < len(layout) else offset):
            l, gt, dt, off, blob, ts = layout[k - 1]
            last = layer_blobs(l, blob, ts)
            if last is not None:
                with open(part, "rb") as f:
                    f.seek(off)
                    kept = f.read(n_expert * blob)
                if kept == last.tobytes():
                    start = k
        if start:
            print("the earlier packing was interrupted: keeping its first %d of %d layers and continuing with layer %d"
                  % (start, len(layout), start), flush=True)
        else:
            print("the earlier packing was interrupted and cannot be continued: starting it again", flush=True)
    resume_at = layout[start][3] if start < len(layout) else offset
    with open(part, "r+b" if start else "wb") as fo:
        if start:
            fo.truncate(resume_at)       # whatever followed the last recorded layer is a partial write
            fo.seek(resume_at)
        for l, gt, dt, off, blob, ts in layout[start:]:
            chunk = layer_blobs(l, blob, ts)
            if chunk is None:
                return 1
            fo.write(chunk.tobytes())
            fo.flush()
            os.fsync(fo.fileno())
            note = out / "experts.bin.tmp.json.new"
            note.write_text(json.dumps({"want": want, "layers": l + 1}), encoding="utf-8")
            note.replace(progress)
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    part.replace(path)
    progress.unlink(missing_ok=True)
    side_tmp = out / "experts.bin.src.json.tmp"
    side_tmp.write_text(json.dumps(want, indent=2) + "\n", encoding="utf-8")
    side_tmp.replace(sidecar)
    print("experts.bin: %d layers, %.2f GiB" % (len(layout), offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
