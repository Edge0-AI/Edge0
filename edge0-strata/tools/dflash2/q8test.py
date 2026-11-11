"""Does a Q8_0 drafter (the matrices block-quantized like ggml Q8_0) keep the acceptance? Walk over two prompts, block 4 and 8."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"C:\dflash2-work")
import dflash2_ref as R

W = Path(r"C:\dflash2-work")
embed = R.load_dequant(str(W / "embd_iq4xs.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.IQ4_XS)
head = R.load_dequant(str(W / "head_q6k.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.Q6_K)
d = R.Drafter(str(W / "drafter.safetensors"), str(W / "drafter_config.json"), embed, head, threads=6)
d.enable_fast_head()


def q8_0(w):
    """ggml Q8_0 along the last dimension: blocks of 32, scale = amax/127 stored as fp16, int8 codes"""
    shp = w.shape
    x = w.reshape(-1, 32)
    amax = x.abs().amax(dim=1, keepdim=True)
    sc = (amax / 127.0).to(torch.float16).float()
    sc_safe = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.round(x / sc_safe).clamp(-127, 127)
    return (q * sc).reshape(shp)


def run(name, b, tag):
    seqj = json.load(open(W / "p" / ("%s.seq.json" % name)))
    seq = seqj["prompt"] + seqj["out"]
    layers, pos, tok, taps = R.read_taps(str(W / "taps" / ("%s.bin" % name)))
    keep = pos < len(seq)
    pos, taps = pos[keep], taps[keep]
    feats = np.zeros((int(pos.max()) + 1, taps.shape[1]), np.float32)
    for i in range(len(pos)):
        feats[pos[i]] = taps[i]
    feats[0] = feats[1]
    d.set_context(torch.from_numpy(feats))
    s = len(seqj["prompt"])
    cyc = []
    with torch.no_grad():
        while s + b <= len(seq):
            path, _ = d.draft(s, seq[s], b)
            truth = seq[s + 1:s + b]
            a = 0
            while a < b - 1 and path[a] == truth[a]:
                a += 1
            cyc.append(a + 1)
            s += a + 1
    print("%s %s block %d: %.3f tokens per round (%d rounds)" % (tag, name, b, np.mean(cyc), len(cyc)), flush=True)


names = sys.argv[1].split(",")
orig = {k: v.clone() for k, v in d.w.items() if k.endswith("proj.weight") or k == "fc.weight" or k.endswith("kernel_projection.weight")}
for tag in ("fp32", "q8_0"):
    if tag == "q8_0":
        for k in orig:
            d.w[k] = q8_0(orig[k])
    for n in names:
        run(n, 4, tag)
        run(n, 8, tag)
