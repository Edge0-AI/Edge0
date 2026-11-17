"""Compare a rebuilt MTP GGUF with the original mtp-q2_0.gguf.

Same-type tensors must match byte for byte. A Q8_0 tensor rebuilt from the runtime files is compared with the
original BF16 tensor quantized the way the Metal engine does at load (ref::quantize_q8_0: fp32 d = amax / 127,
fp16 d by round-to-nearest-even, roundf of x * (1 / d)).
"""
import struct
import sys

import numpy as np


def read(path):
    f = open(path, "rb")
    magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF" and ver == 3

    def rs():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode()

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rv(t):
        if t == 8:
            return rs()
        if t == 9:
            at, n = struct.unpack("<IQ", f.read(12))
            return [rv(at) for _ in range(n)]
        b = f.read(sizes[t])
        return int.from_bytes(b, "little") if t in (4, 5, 10, 11) else b

    kv = {}
    for _ in range(nkv):
        k = rs()
        t, = struct.unpack("<I", f.read(4))
        kv[k] = rv(t)
    align = kv.get("general.alignment", 32)
    info = []
    for _ in range(nt):
        name = rs()
        nd, = struct.unpack("<I", f.read(4))
        dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
        ty, off = struct.unpack("<IQ", f.read(12))
        info.append((name, dims, ty, off))
    pos = f.tell()
    data0 = pos + (-pos) % align
    blob = open(path, "rb").read()
    out = {}
    for i, (name, dims, ty, off) in enumerate(info):
        ne = int(np.prod(dims))
        nbytes = {0: ne * 4, 30: ne * 2, 8: ne // 32 * 34, 42: ne // 64 * 18}[ty]
        out[name] = (dims, ty, blob[data0 + off:data0 + off + nbytes])
    return out


def q8_engine(bf16_bytes):
    x = (np.frombuffer(bf16_bytes, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    v = (x * inv[:, None]).astype(np.float32).astype(np.float64)       # the fp32 product, then exact rounding
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)      # roundf: half away from zero
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out


def main():
    orig, new = read(sys.argv[1]), read(sys.argv[2])
    ok = True
    if sorted(orig) != sorted(new):
        print("tensor names differ:", sorted(set(orig) ^ set(new)))
        return 1
    for name in sorted(orig):
        od, ot, ob = orig[name]
        nd, nt, nb = new[name]
        if od != nd:
            print(f"{name}: dims {od} vs {nd}")
            ok = False
            continue
        if ot == nt:
            same = ob == nb
            if not same and ot == 0:
                # the engine only uses 1 + w for these RMSNorm weights
                a = np.frombuffer(ob, dtype="<f4")
                b = np.frombuffer(nb, dtype="<f4")
                folded = np.array_equal(a + np.float32(1), b + np.float32(1))
                print(f"{name}: type 0 bytes differ in {int((a != b).sum())}/{a.size} values; "
                      f"1 + w {'identical' if folded else 'DIFFERENT'}")
                ok &= folded
                continue
            print(f"{name}: type {ot} bytes {'identical' if same else 'DIFFERENT'}")
            ok &= same
        elif ot == 30 and nt == 8:
            ref = q8_engine(ob)
            got = np.frombuffer(nb, dtype=np.uint8).reshape(-1, 34)
            bad_blocks = int((ref != got).any(axis=1).sum())
            bad_scales = int((ref[:, :2] != got[:, :2]).any(axis=1).sum())
            qd = np.abs(ref[:, 2:].view(np.int8).astype(int) - got[:, 2:].view(np.int8).astype(int))
            print(f"{name}: BF16 -> Q8_0 vs engine quantization: {bad_blocks}/{ref.shape[0]} blocks differ "
                  f"(scales {bad_scales}, values {int((qd > 0).sum())}, max step {int(qd.max())})")
            if bad_blocks > 0:
                ok = False
        else:
            print(f"{name}: type {ot} vs {nt}")
            ok = False
    print("RESULT:", "same-type tensors identical" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
