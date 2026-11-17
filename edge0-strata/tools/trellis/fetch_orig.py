"""fetch_orig.py - copy the routed-expert tensors of chosen layers of the ORIGINAL Qwen3.8-Flash-Next (BF16 safetensors) to
local raw files, by HTTP range requests (resumable, per chunk of experts).  Files: <out>/L<l>_gate_up.bf16 and
L<l>_down.bf16 (raw bf16, shape [512, 1280, 2560] and [512, 2560, 640]).

  python3 tools/trellis/fetch_orig.py --layers 4,16,28,40 --out /data/dsv41-enc/orig [--workers 48]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hf_slice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=16, help="experts per range")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = hf_slice.Repo(a.repo)
    for layer in [int(x) for x in a.layers.split(",")]:
        for tname, fname in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
            name = "model.language_model.layers.%d.mlp.experts.%s" % (layer, tname)
            shard, t, base = repo.info(name)
            n_exp = t["shape"][0]
            inner = t["data_offsets"][1] - t["data_offsets"][0]
            inner //= n_exp
            path = out / ("L%d_%s.bf16" % (layer, fname))
            done = out / ("L%d_%s.done" % (layer, fname))
            if done.exists():
                continue
            marks = out / ("L%d_%s.chunks" % (layer, fname))
            have = set(int(x) for x in marks.read_text().split()) if marks.exists() else set()
            mode = "r+b" if path.exists() else "w+b"
            with open(path, mode) as f:
                f.truncate(inner * n_exp)
                t0 = time.time()
                for e0 in range(0, n_exp, a.chunk):
                    if e0 in have:
                        continue
                    e1 = min(n_exp, e0 + a.chunk)
                    s = base + t["data_offsets"][0] + e0 * inner
                    raw = hf_slice.get_parallel(repo.base + shard, s, s + (e1 - e0) * inner - 1, a.workers)
                    f.seek(e0 * inner)
                    f.write(raw)
                    f.flush()
                    with open(marks, "a") as m:
                        m.write("%d\n" % e0)
                    print("L%d %s experts %d-%d  %.1f MB/s" % (layer, fname, e0, e1, len(raw) / 1e6 / (time.time() - t0 + 1e-9)),
                          flush=True)
                    t0 = time.time()
            done.write_text("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
