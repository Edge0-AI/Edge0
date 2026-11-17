"""tools/qwen35/compare_dumps.py - two runs of the Qwen3.6 parity dumps against each other (e.g. llama.cpp GPU vs
llama.cpp CPU: the noise floor the strata-q36ref comparison is judged against).

    python tools/qwen35/compare_dumps.py DIR_A KIND_A DIR_B KIND_B      (KIND: llama or strata)
Per prompt: greedy tokens equal up to the answer's end (llama.cpp's first <|im_end|> in run A), the first divergence's
top-1/top-2 margins, and the layer outputs' median and max ||a-b||/||b|| over the positions before the divergence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N_EMBD, N_LAYER, EOS = 2048, 40, 248046


def gen_of(d: Path, tag: str, kind: str) -> list[int]:
    f = d / (f"{tag}.strata.gen" if kind == "strata" else f"{tag}.llama.log")
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("gen:"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no gen line in {f}")


def prompt_len(d: Path, tag: str) -> int:
    for line in (d / f"{tag}.llama.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("prompt:"):
            return len(line.split(":", 1)[1].split())
    raise SystemExit("no prompt line")


def main() -> int:
    da, ka, db, kb = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    tags = sorted({p.name.split(".")[0] for p in da.glob(f"p*.{ka}.layers")})
    rows = []
    for tag in tags:
        ga, gb = gen_of(da, tag, ka), gen_of(db, tag, kb)
        upto = min(len(ga), len(gb))
        if EOS in ga[:upto]:
            upto = ga.index(EOS) + 1
        same = 0
        while same < upto and ga[same] == gb[same]:
            same += 1
        la = np.fromfile(da / f"{tag}.{ka}.logits", dtype=np.float32).reshape(-1, 4)
        lb = np.fromfile(db / f"{tag}.{kb}.logits", dtype=np.float32).reshape(-1, 4)
        A = np.fromfile(da / f"{tag}.{ka}.layers", dtype=np.float32).reshape(-1, N_LAYER, N_EMBD)
        B = np.fromfile(db / f"{tag}.{kb}.layers", dtype=np.float32).reshape(-1, N_LAYER, N_EMBD)
        npos = min(len(A), len(B), prompt_len(da, tag) + same)
        nb = np.linalg.norm(B[:npos], axis=2)
        na = np.linalg.norm(A[:npos], axis=2)
        err = np.where((nb > 0) & (na > 0), np.linalg.norm(A[:npos] - B[:npos], axis=2) / np.maximum(nb, 1e-30), np.nan)
        div = ""
        if same < upto:
            div = (f"; step {same}: A {int(la[same, 0])} (margin {la[same, 1] - la[same, 3]:.3f}) vs B {int(lb[same, 0])} "
                   f"(margin {lb[same, 1] - lb[same, 3]:.3f})")
        per_layer = np.nanmedian(err, axis=0)
        rows.append((same == upto, np.nanmedian(err), np.nanmax(err)))
        print(f"{tag}: {same}/{upto} equal; layer err median {np.nanmedian(err):.4f} max {np.nanmax(err):.4f}; "
              f"median by layer 0/10/20/30/39: " + " ".join(f"{per_layer[i]:.4f}" for i in (0, 10, 20, 30, 39)) + div)
    print(f"answers identical: {sum(r[0] for r in rows)} of {len(rows)}; layer err median of medians "
          f"{np.median([r[1] for r in rows]):.4f}, worst max {max(r[2] for r in rows):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
