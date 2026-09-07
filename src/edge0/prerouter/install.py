"""Install prerouter heads into a loaded base model.

Two conventions exist (both supported here):

* qwen (``patch_call=True``): heads are NEW ``PrerouterHead`` modules
  attached to the MoE blocks, and the MoE block's ``__call__`` is patched
  at class level so decode routes through the prerouter's
  ``pred_inds``/``pred_scores`` (100% replacement, no matching/drops).
  Prefill is untouched (router path).  The patch mirrors the deployment's
  proven ``_trained_call``, with guards per instance so non-prerouter
  engines keep the normal router.
* ling (``patch_call=False``): the vendored model already contains one
  ``BailingPrerouter`` per MoE layer and consumes ``prerouter_cache``
  logits internally; this installer only fills the head weights (and
  zeroes ``linear_init`` when the checkpoint has none).
"""

from __future__ import annotations

from edge0.backends import core
from edge0.backends import nn

from edge0.backends.mlx._impl.qwen3_next import Qwen3NextSparseMoeBlock
from edge0.backends.mlx.io import load_safetensors
from edge0.moe.spec import MoESpec
from edge0.prerouter.heads import PrerouterHead, topk_onehot
from edge0.prerouter.spec import PrerouterSpec
from edge0.prerouter.state import PrerouterState

_PATCHED_MARK = "_edge0_prerouter_patched"


def _parse_weights(weights: dict[str, core.array], dtype):
    """``layers.<N>.<part>.weight`` keys -> ``{N: {"<part>.weight": mx}}``.

    Accepts both ``layers.N.fc1.weight`` and the nested
    ``layers.N.mlp.prerouter.fc1.weight`` form."""
    heads: dict[int, dict[str, core.array]] = {}
    for key, arr in weights.items():
        parts = key.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            continue
        try:
            owner = int(parts[1])
        except ValueError:
            continue
        if parts[-2] not in ("fc1", "fc2", "linear_init"):
            continue
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        heads.setdefault(owner, {})[f"{parts[-2]}.weight"] = arr
    return heads


def install_prerouter(
    *,
    model,
    spec: MoESpec,
    pspec: PrerouterSpec,
    weights: dict[str, core.array] | None = None,
    state: PrerouterState | None = None,
    n_layers: int,
) -> tuple[PrerouterState, dict[int, nn.Module]]:
    """Attach prerouter heads to a loaded model.

    Returns ``(state, heads)`` where ``state`` is the cross-token
    double-buffer the engine drives at the step boundary and ``heads``
    maps owner layer -> head module.  ``weights`` is a flat
    ``name -> core.array`` dict (from ``load_safetensors``); when None the
    file from ``pspec.weights_file`` is loaded instead.
    """
    if weights is None:
        if not pspec.weights_file:
            raise ValueError(
                "install_prerouter needs weights or PrerouterSpec.weights_file")
        weights = load_safetensors(pspec.weights_file)
    dtype = core.float16 if pspec.dtype == "fp16" else core.float32
    head_weights = _parse_weights(weights, dtype)
    missing = [n for n in pspec.owner_layers(n_layers) if n not in head_weights]
    if missing:
        raise ValueError(
            f"prerouter weights missing owner layers {missing[:5]}... "
            f"(have {sorted(head_weights)})")

    if state is None:
        state = PrerouterState(n_layers, spec.num_experts,
                               pspec.start_layer,
                               owners=pspec.owner_layers(n_layers))

    heads: dict[int, nn.Module] = {}
    for owner in pspec.owner_layers(n_layers):
        block = spec.block_of(model, owner)
        w = head_weights[owner]
        if pspec.patch_call:
            # hidden = the MoE block's input width, read off the router's
            # gate projection (weight is [num_experts, dim]).
            dim = block.gate.weight.shape[1]
            head = PrerouterHead(dim, spec.num_experts, pspec.hidden,
                                 dtype=dtype)
            # hidden size: MoESpec doesn't carry it; read from the gate
            # projection weight shape.
            head.fc1.weight = w["fc1.weight"]
            head.fc2.weight = w["fc2.weight"]
            head.linear_init.weight = w["linear_init.weight"]
            block.prerouter_head = head
            block.prerouter_li = owner
            block.prerouter_start = pspec.start_layer
            block.prerouter_enabled = True
            block.prerouter_state = state
            heads[owner] = head
        else:
            pg = getattr(block, "prerouter", None)
            if pg is None:
                raise AttributeError(
                    f"layer {owner}: patch_call=False but block has no "
                    "'prerouter' head (wrong model family?)")
            pg.fc1 = nn.Linear(w["fc1.weight"].shape[1],
                               w["fc1.weight"].shape[0], bias=False)
            pg.fc1.weight = w["fc1.weight"]
            pg.fc2 = nn.Linear(w["fc2.weight"].shape[1],
                               w["fc2.weight"].shape[0], bias=False)
            pg.fc2.weight = w["fc2.weight"]
            wk = w.get("linear_init.weight")
            if wk is not None:
                pg.linear_init = nn.Linear(wk.shape[1], wk.shape[0],
                                           bias=False)
                pg.linear_init.weight = wk
            else:
                pg.linear_init.weight = core.zeros(
                    pg.linear_init.weight.shape, dtype=dtype)
            heads[owner] = pg

    if pspec.patch_call and not getattr(
            Qwen3NextSparseMoeBlock, _PATCHED_MARK, False):
        _patch_qwen_consume()
    return state, heads


def _patch_qwen_consume():
    """Patch ``Qwen3NextSparseMoeBlock.__call__`` (deploy-proven
    ``_trained_call`` semantics).

    A class-level patch is required because implicit ``moe(x)`` calls look
    up the type, not the instance.  Guarded per instance: only blocks with
    ``prerouter_enabled`` and single-token inputs (decode) route through
    the prerouter; everything else takes the original router path."""
    orig_call = Qwen3NextSparseMoeBlock.__call__

    def prerouter_call(self, x):
        sm = getattr(self, "switch_mlp", None)
        st = getattr(self, "prerouter_state", None)
        if not (getattr(self, "prerouter_enabled", False) and st is not None
                and x.ndim == 3 and x.shape[1] == 1
                and getattr(sm, "_staged_replace", None) is False
                and sm is not None):
            return orig_call(self, x)
        li = self.prerouter_li
        # This layer CONSUMES the prediction produced by layer li-1's head
        # at the previous token (prev-layer + prev-token double shift).
        pred = st.pred_inds[li] if li >= st.start else None
        if pred is not None:
            inds = pred
            scores = st.pred_scores[li]
        else:
            # First decode tokens / router layers below start: fall back to
            # the real router (matches training's pos-0 fallback).
            gates = core.softmax(self.gate(x), axis=-1, precise=True)
            k = self.top_k
            inds = core.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = core.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)

        oh = topk_onehot(inds, self.num_experts)
        st.oh[li] = oh

        # Capture for the head this layer OWNS (runs at the step boundary).
        self.prerouter_m_in = x
        self.prerouter_oh = oh

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = core.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    Qwen3NextSparseMoeBlock.__call__ = prerouter_call
    setattr(Qwen3NextSparseMoeBlock, _PATCHED_MARK, True)
