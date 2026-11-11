"""tools/embd_bf16_pack.py - the token embedding exactly as the checkpoint ships it (BF16), as a GGUF the engine reads.

    python tools/embd_bf16_pack.py --model <checkpoint dir> --out <token-embd-bf16.gguf>

nvfp4_convert.py stores token_embd as Q8_0 (0.56% off per value). The engine keeps the table in mapped host memory
and reads one row per token, so BF16 costs 0.6 GB of RAM more and no VRAM; `--embd-gguf` takes it from this file.

The output is a one-tensor GGUF: `token_embd.weight`, ne = [n_embd, n_vocab], type BF16, the bytes copied straight
out of the safetensors file - nothing decoded or rounded.

    python tools/embd_bf16_pack.py --gguf <model.gguf> --out <token-embd-bf16.gguf>

instead dequantizes a GGUF's own token_embd.weight to BF16 (round-to-nearest-even, through gguf-py), for a file whose
embedding type has no GPU dequantizer in the engine (Agention's Gyro-S stores it as Q6_K).  That is the GGUF's values
rounded to BF16, not the checkpoint's.
"""
import argparse
import json
import pathlib
import struct
import sys

GGUF_TYPE_STRING = 8
GGML_TYPE_BF16 = 30
ALIGN = 32
CHUNK = 64 << 20


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def write_header(dim, vocab, source, what):
    kvs = [kv_string("general.architecture", "strata-embd"),
           kv_string("general.name", what),
           kv_string("strata.embd.source", source)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, vocab)
    head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)
    return head


def from_gguf(src, out):
    """token_embd.weight of a GGUF, dequantized and rounded to BF16 (nearest-even), in row chunks."""
    import numpy as np
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import gguf_reader as G
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    g = G.GGUFFile(src)
    t = next((t for t in g.tensors if t.name == "token_embd.weight"), None)
    if t is None or len(t.shape) != 2 or t.expected_bytes() is None:
        sys.exit("%s: no 2-D token_embd.weight of a known type" % src)
    dim, vocab = int(t.shape[0]), int(t.shape[1])
    row = t.expected_bytes() // vocab
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    base = g.data_start + t.offset
    print("token_embd.weight: %d x %d %s -> BF16, %.2f GB" % (vocab, dim, t.type_name, vocab * dim * 2 / 1e9), flush=True)
    head = write_header(dim, vocab, src.name, "token embedding, %s dequantized to BF16" % t.type_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    rows = 8192
    with open(tmp, "wb") as w:
        w.write(head)
        for r0 in range(0, vocab, rows):
            r1 = min(vocab, r0 + rows)
            raw = np.asarray(mm[base + r0 * row: base + r1 * row]).reshape(r1 - r0, row)
            vals = quants.dequantize(raw, Q[t.type_name]).astype(np.float32)
            if not np.isfinite(vals).all():
                sys.exit("token_embd.weight has non-finite values")
            w.write(quants.quantize(vals, Q.BF16).tobytes())
    if tmp.stat().st_size != len(head) + vocab * dim * 2:
        sys.exit("size check failed")
    tmp.replace(out)
    print("wrote %s (header %d B)" % (out, len(head)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="the checkpoint directory (model.safetensors.index.json)")
    src.add_argument("--gguf", help="a model GGUF whose token_embd.weight is dequantized to BF16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.gguf:
        from_gguf(pathlib.Path(a.gguf).absolute(), pathlib.Path(a.out))
        return
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    wmap = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]

    names = [n for n in wmap if n.endswith("embed_tokens.weight") and "visual" not in n and "mtp" not in n]
    if len(names) != 1:
        sys.exit("expected one text embed_tokens.weight in %s, found %s" % (model, names))
    name = names[0]
    hdr, base = safetensors_header(model / wmap[name])
    t = hdr[name]
    if t["dtype"] != "BF16" or len(t["shape"]) != 2:
        sys.exit("%s is %s %s, not a 2-D BF16 tensor" % (name, t["dtype"], t["shape"]))
    vocab, dim = t["shape"]
    n = vocab * dim * 2
    if t["data_offsets"][1] - t["data_offsets"][0] != n:
        sys.exit("%s: data size does not match its shape" % name)
    print("%s: %d x %d BF16, %.2f GB" % (name, vocab, dim, n / 1e9), flush=True)

    head = write_header(dim, vocab, model.name, "token embedding, BF16 as shipped")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as w, open(model / wmap[name], "rb") as f:
        w.write(head)
        f.seek(base + t["data_offsets"][0])
        left = n
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                sys.exit("short read in " + str(model / wmap[name]))
            w.write(b)
            left -= len(b)
    if tmp.stat().st_size != len(head) + n:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + n))
    tmp.replace(out)
    print("wrote %s (header %d B)" % (out, len(head)))


if __name__ == "__main__":
    main()
