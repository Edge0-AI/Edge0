"""Per-anchor drafter results at EVERY generated position (stride 1), resumable: the exact path and the selector's confidences."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"C:\dflash2-work")
import dflash2_ref as R

W = Path(r"C:\dflash2-work")
names = sys.argv[1].split(",")
b = int(sys.argv[2])
threads = int(sys.argv[3])
budget = float(sys.argv[4])          # seconds this call may spend before it saves and stops
t_start = time.time()
embed = R.load_dequant(str(W / "embd_iq4xs.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.IQ4_XS)
head = R.load_dequant(str(W / "head_q6k.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.Q6_K)
d = R.Drafter(str(W / "drafter.safetensors"), str(W / "drafter_config.json"), embed, head, threads=threads)
d.enable_fast_head()
out_path = W / ("anchors1_b%d.json" % b)
res = json.load(open(out_path)) if out_path.exists() else {}
for name in names:
    if res.get(name + "#done"):
        continue
    seqj = json.load(open(W / "p" / ("%s.seq.json" % name)))
    seq = seqj["prompt"] + seqj["out"]
    layers, pos, tok, taps = R.read_taps(str(W / "taps" / ("%s.bin" % name)))
    keep = pos < len(seq)
    pos, tok, taps = pos[keep], tok[keep], taps[keep]
    n = int(pos.max()) + 1
    feats = np.zeros((n, taps.shape[1]), np.float32)
    for i in range(len(pos)):
        feats[pos[i]] = taps[i]
    feats[0] = feats[1]
    d.set_context(torch.from_numpy(feats))
    rows = res.get(name, [])
    s0 = rows[-1]["s"] + 1 if rows else len(seqj["prompt"])
    t1 = time.time()
    finished = True
    with torch.no_grad():
        for s in range(s0, len(seq) - b):
            path, top1 = d.draft(s, seq[s], b)
            truth = seq[s + 1:s + b]
            rows.append({"s": s, "ok": [int(path[i] == truth[i]) for i in range(b - 1)], "p": [round(x, 4) for x in d.last_probs]})
            if time.time() - t_start > budget:
                finished = False
                break
    res[name] = rows
    if finished:
        res[name + "#done"] = True
    json.dump(res, open(out_path, "w"))
    print(name, b, len(rows), "anchors so far, %.0f s" % (time.time() - t1), "done" if finished else "PAUSED", flush=True)
    if not finished:
        break
