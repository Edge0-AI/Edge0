"""Teacher-forced logits, stock vs the experimental-speed-projection vector, on the same texts and fixed experts.

    python esp_kl.py run      # the dumps (per-token path)
    python esp_kl.py compare  # same top-1, mean/median KL(stock || projection), perplexities
"""
import json
import os
import subprocess
import sys

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
E = r"C:\Users\AI-Server\Desktop\Strata\Public\Engine"
EXE = E + r"\build-dev\strata.exe"
VEC = r"C:\Users\AI-Server\Desktop\Strata\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
TEXTS = {"code": S + r"\p512.ids", "doc": S + r"\4k-02-long-doc-1k.ids", "chat": S + r"\4k-04-chat-1k.ids"}
OUT = S + r"\esp-kl"
ARMS = {"stock": [],
        "proj": ["--control-vector-scaled", VEC + ":1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]}


def base_args():
    c = json.load(open(S + r"\cfg-rel.json"))
    a = list(c["args"])
    i = a.index("--expert-cache")
    a[i + 1] = "4000"
    j = a.index("--prefill")          # the prompt one token at a time: a logits row per position
    del a[j:j + 2]
    return a + ["--pcie-frac", "0", "--adapt-swaps", "0"], c


def run():
    os.makedirs(OUT, exist_ok=True)
    args, c = base_args()
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(c["lib_dirs"]) + os.pathsep + env["PATH"]
    for name, path in TEXTS.items():
        for arm, extra in ARMS.items():
            out = f"{OUT}\\{name}-{arm}.bin"
            if os.path.exists(out):
                continue
            cmd = [EXE, *args, *extra, "--tokens-file", path, "--max-new", "1", "--dump-logits", out]
            with open(f"{OUT}\\{name}-{arm}.log", "w") as log:
                r = subprocess.run(cmd, cwd=c["cwd"], stdout=log, stderr=subprocess.STDOUT, env=env)
            print(name, arm, "exit", r.returncode, flush=True)


def load(p):
    h = np.fromfile(p, dtype=np.int32, count=2)
    return np.memmap(p, dtype=np.float32, mode="r", offset=8, shape=(int(h[1]), int(h[0])))


def lsm(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def compare():
    print(f"{'text':<6}{'tokens':>7}{'same top-1':>11}{'mean KL':>9}{'median KL':>11}{'p99 KL':>8}{'ppl stock':>10}{'ppl proj':>9}")
    allkl = []
    for name, path in TEXTS.items():
        ids = [int(x) for x in open(path).read().replace(",", " ").split()]
        a, b = load(f"{OUT}\\{name}-stock.bin"), load(f"{OUT}\\{name}-proj.bin")
        n = min(a.shape[0], b.shape[0], len(ids) - 1)
        kl, same, la, lb = [], 0, [], []
        for r in range(0, n, 64):
            pa, pb = lsm(a[r:min(n, r + 64)]), lsm(b[r:min(n, r + 64)])
            kl += list((np.exp(pa) * (pa - pb)).sum(axis=1))
            same += int((pa.argmax(1) == pb.argmax(1)).sum())
            nxt = np.array(ids[r + 1:min(n, r + 64) + 1])
            la += list(pa[np.arange(len(nxt)), nxt])
            lb += list(pb[np.arange(len(nxt)), nxt])
        kl = np.array(kl)
        allkl += list(kl)
        print(f"{name:<6}{n:>7}{100 * same / n:>10.1f}%{kl.mean():>9.3f}{np.median(kl):>11.4f}{np.percentile(kl, 99):>8.2f}"
              f"{np.exp(-np.mean(la)):>10.3f}{np.exp(-np.mean(lb)):>9.3f}")
    allkl = np.array(allkl)
    print(f"all {len(allkl)} tokens: mean KL {allkl.mean():.3f}, max {allkl.max():.2f}, p99 {np.percentile(allkl, 99):.2f}")


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
