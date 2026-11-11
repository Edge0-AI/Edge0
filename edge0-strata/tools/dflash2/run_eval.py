"""Offline DFlash2 acceptance on OUR quantized target: walks each prompt's dumped sequence the way the speculative loop would
(anchor, draft block, accept the leading matches of the drafted path against the engine's own greedy tokens, move on)."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"C:\dflash2-work")
import dflash2_ref as R

W = Path(r"C:\dflash2-work")
names = sys.argv[1].split(",")
blocks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6

t0 = time.time()
embed = R.load_dequant(str(W / "embd_iq4xs.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.IQ4_XS)
head = R.load_dequant(str(W / "head_q6k.bin"), 248320, 2560, R.gguf.GGMLQuantizationType.Q6_K)
print("dequantized embed/head in %.0f s" % (time.time() - t0), flush=True)
d = R.Drafter(str(W / "drafter.safetensors"), str(W / "drafter_config.json"), embed, head, threads=threads)
res_path = W / "eval_results.json"
results = json.load(open(res_path)) if res_path.exists() else {}

for name in names:
    seqj = json.load(open(W / "p" / ("%s.seq.json" % name)))
    prompt, out = seqj["prompt"], seqj["out"]
    seq = prompt + out
    layers, pos, tok, taps = R.read_taps(str(W / "taps" / ("%s.bin" % name)))
    keep = pos < len(seq)             # the tail of the last window is padding (zeros) past the sequence
    pos, tok, taps = pos[keep], tok[keep], taps[keep]
    n = int(pos.max()) + 1
    feats = np.zeros((n, taps.shape[1]), np.float32)
    seen = np.zeros(n, bool)
    for i in range(len(pos)):          # the last record of a position wins
        feats[pos[i]] = taps[i]
        seen[pos[i]] = True
        assert tok[i] == seq[pos[i]], ("token mismatch at", pos[i], tok[i], seq[pos[i]])
    if not seen[0] and seen[1]:        # position 0 goes through the batched prefill (no tap): the next row stands in
        feats[0] = feats[1]
        seen[0] = True
    assert seen.all(), "positions missing: %s" % np.where(~seen)[0][:5]
    d.set_context(torch.from_numpy(feats))
    print("%s: prompt %d, generated %d, tap records %d, layers %s" % (name, len(prompt), len(out), len(pos), layers), flush=True)
    results.setdefault(name, {})
    for b in blocks:
        s = len(prompt)                # the first generated token is the first anchor, as after a prefill
        cyc, cond_ok, cond_n = [], np.zeros(b - 1), np.zeros(b - 1)
        top1_ok = np.zeros(b - 1)
        t1 = time.time()
        with torch.no_grad():
            while s + b <= len(seq) - 0 and s < n:
                path, top1 = d.draft(s, seq[s], b)
                truth = seq[s + 1:s + b]
                a = 0
                while a < b - 1 and path[a] == truth[a]:
                    a += 1
                for i in range(b - 1):                  # conditional on all earlier correct
                    if i <= a:
                        cond_n[i] += 1
                        cond_ok[i] += (i < a)
                    if i <= a and top1[i] == truth[i]:
                        top1_ok[i] += 1
                cyc.append(a + 1)
                s += a + 1
        tau = float(np.mean(cyc))
        results[name]["b%d" % b] = {"anchors": len(cyc), "tokens_per_round": tau,
                                    "cond_accept": (cond_ok / np.maximum(cond_n, 1)).round(3).tolist(),
                                    "hist": np.bincount(cyc, minlength=b + 1).tolist()}
        print("  block %d: %d rounds, %.3f tokens per round, cond %s  (%.0f s)" %
              (b, len(cyc), tau, (cond_ok / np.maximum(cond_n, 1)).round(2).tolist(), time.time() - t1), flush=True)
    json.dump(results, open(res_path, "w"), indent=1)
