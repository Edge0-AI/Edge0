"""Run one shared-code case under whichever backend EDGE0_BACKEND selects.

Invoked by tests/test_backend_parity.py as a subprocess, once per backend,
so the same framework code (routing, sampling, streaming) is exercised on
MLX and on the torch backend with identical inputs:

    EDGE0_BACKEND=cuda python tests/backend_parity_worker.py CASE IN.npz OUT.npz

Inputs and outputs are numpy arrays in .npz files; bf16 never crosses the
boundary (cases upcast to float32 before returning).
"""

from __future__ import annotations

import sys

import numpy as np

from edge0.backends import core


# .tolist() is an array method on both backends (mlx.core has no tolist
# function, despite the contract listing one).
def _np(a):
    return np.array(core.astype(a, core.float32).tolist(), dtype=np.float32)


def _np_int(a):
    return np.array(a.tolist(), dtype=np.int64)


def _sorted_pair(inds, scores):
    """Router outputs as (indices, scores) sorted by expert id per row --
    argpartition leaves the order within the top-k unspecified."""
    i, s = _np_int(inds), _np(scores)
    order = np.argsort(i, axis=-1)
    return np.take_along_axis(i, order, -1), np.take_along_axis(s, order, -1)


def case_select_from_logits(inp):
    from edge0.moe.routing import select_from_logits
    inds, scores = select_from_logits(core.array(inp["logits"]), top_k=4)
    i, s = _sorted_pair(inds, scores)
    return {"inds": i, "scores": s}


def case_group_select_from_logits(inp):
    from edge0.moe.routing import group_select_from_logits
    inds, scores = group_select_from_logits(
        core.array(inp["logits"]), top_k=8, n_group=8, topk_group=4,
        routed_scaling=2.5, expert_bias=core.array(inp["expert_bias"]))
    i, s = _sorted_pair(inds, scores)
    return {"inds": i, "scores": s}


def case_mask_logits(inp):
    from edge0.sampling import _mask_logits
    out = {}
    for name, kw in {
        "topk": dict(temperature=0.7, top_k=20, top_p=None),
        "topp": dict(temperature=0.9, top_k=None, top_p=0.8),
        "both": dict(temperature=1.0, top_k=50, top_p=0.9),
        "greedy": dict(temperature=0.0, top_k=None, top_p=None),
    }.items():
        out[name] = _np(_mask_logits(core.array(inp["logits"]), **kw))
    return out


CASES = {name[len("case_"):]: fn for name, fn in globals().items()
         if name.startswith("case_")}


def main(argv):
    case, inp_path, out_path = argv[1:4]
    inp = dict(np.load(inp_path))
    np.savez(out_path, **CASES[case](inp))


if __name__ == "__main__":
    main(sys.argv)
