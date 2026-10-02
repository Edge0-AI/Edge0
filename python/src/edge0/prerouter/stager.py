"""Prerouter step-boundary stager.

After each forward's logits eval, run every owner head as ONE stacked
batch (the base graph is already materialized), turn the logits into
expert selections with ONE eval + ONE tolist, and submit the next
decode step's staged slots per consumer layer — so the pool's SSD
builds overlap the next forward entirely (no per-layer sync points
inside the graph).

Two feature/consume conventions exist across the supported models, both
implemented here:

* qwen (``CrossTokenStager``): heads installed on the MoE blocks, the
  block patch captures ``(m_in, executed one-hot)``; the stager makes
  the selection itself (precise softmax -> top-k -> renormalize) and
  stores ``pred_inds`` / ``pred_scores`` keyed by the consuming layer.
* ling (``LingPrerouterStager``): heads baked into the vendored model,
  features read from the model's own caches (``m_in_cache`` /
  ``last_topk`` teacher feature / ``prev_topk_oh``); the stager stores
  the raw LOGITS in ``pg_cache`` and the consuming MoE block performs
  the sigmoid group-limited selection internally.
"""

from __future__ import annotations

import os

from edge0.backends import core

from edge0.moe.routing import group_select_from_logits, select_from_logits
from edge0.moe.spec import MoESpec, RouterKind
from edge0.prerouter.heads import gelu_erf, topk_onehot
from edge0.prerouter.spec import PrerouterSpec
from edge0.prerouter.state import PrerouterState
from edge0.streaming.layer import StreamingSwitchGLU


def _topk_onehot_pos(idx, num_experts: int, pos: int = -1):
    """One-hot set of the top-k experts at position ``pos`` of ``idx``
    ([B, T, K] -> [B, 1, E], OR over the K slots) — ling feature
    semantics.  Returns None when ``idx`` has no such position (e.g. a
    T=1 prefill asking for pos=-2) so the caller can fall back to the
    zeros feature (the deployment's ``_topk_onehot`` crashes on this
    case; the T=1 prompt is one BPE token like 你好 = [34355])."""
    if idx is None or idx.ndim < 3:
        return None
    t = idx.shape[-2]
    lo = pos if pos >= 0 else t + pos
    if lo < 0 or lo >= t:
        return None
    idx = idx[..., pos, :][..., None, :]  # [B, 1, K]
    ar = core.arange(num_experts, dtype=idx.dtype)
    oh = (idx[..., None] == ar)  # [B, 1, K, E]
    return core.astype(oh.sum(axis=-2), core.float32)


class PrerouterStager:
    """Base: one-batch collect -> stack -> select -> eval -> tolist ->
    per-consumer stage_experts.

    Two cost fixes on top of the naive loop (measured on the 8b tier,
    docs/experiments/prerouter-decode-speed-2026-09-14.md §6):

    * ``head_batch`` (default on): the owner heads share one shape, so their
      weights are stacked once and each step runs 3 ``einsum`` ops instead of
      3 small ops per head.  Mathematically identical up to fp16 rounding;
      ``head_batch=False`` (or ``EDGE0_HEAD_BATCH=0``) keeps the per-head loop
      for A/B.
    * when nothing consumes host-side expert IDs (``stream_layers`` empty,
      i.e. ``staged=False``) the ``core.eval`` + ``tolist`` are skipped: the
      routing cache holds lazy arrays, so the sync produces nothing usable.
    """

    def __init__(
        self,
        *,
        model,
        spec: MoESpec,
        pspec: PrerouterSpec,
        state: PrerouterState,
        stream_layers: dict[int, StreamingSwitchGLU],
        top_k: int,
    ):
        self.model = model
        self.spec = spec
        self.pspec = pspec
        self.state = state
        self.stream_layers = stream_layers
        self.top_k = top_k
        self.num_experts = spec.num_experts
        self.records: list | None = None
        self.cur_step = 0
        self.head_batch = os.environ.get("EDGE0_HEAD_BATCH", "1") != "0"
        self._stacked_key: tuple | None = None
        self._stacked: tuple | None = None

    # ---- feature hooks (overridden per family) ----------------------------

    def _features(self, owner: int):
        """Return (m_in, cur_oh, prev_oh) for the owner layer's head."""
        raise NotImplementedError

    def _select(self, logits: core.array):
        """Turn stacked head logits into (inds, scores)."""
        raise NotImplementedError

    def _store(self, consumer: int, logits_i, inds_i, scores_i,
               experts: list[int] | None) -> None:
        """Record this consumer's prediction + submit its staged fill
        (``experts=None`` when no consumer needs host-side IDs)."""
        raise NotImplementedError

    def _note(self, owner: int, cur_oh) -> None:
        """Per-owner side effects (e.g. roll the prev-token feature)."""

    # ---- batched head execution (fix 1) -----------------------------------

    def _stacked_weights(self, metas: list[int]):
        """Stack the owner heads' weights once (all heads share one shape).

        Returns None (and stays on the per-head loop) if anything about the
        shapes/dtypes is not uniform."""
        key = tuple(metas)
        if not self.head_batch:
            return None
        if self._stacked_key == key:
            return self._stacked
        self._stacked_key = key
        self._stacked = None
        try:
            heads = [self._head_of(o) for o in metas]
            t = lambda w: core.transpose(  # noqa: E731
                core.stack([h.weight for h in w]), (0, 2, 1))
            w1 = t([h.fc1 for h in heads])
            w2 = t([h.fc2 for h in heads])
            wl = t([h.linear_init for h in heads])
            core.eval(w1, w2, wl)
            self._stacked = (w1, w2, wl)
        except Exception:  # noqa: BLE001 — any shape mismatch: per-head loop
            self._stacked = None
        return self._stacked

    def _logits(self, metas: list[int], inputs: list[core.array],
                cur_ohs: list[core.array], prev_ohs: list[core.array]):
        """Stacked head logits [n,1,1,E] (batched, else per-head loop)."""
        w = self._stacked_weights(metas)
        if w is None:
            return core.stack(
                [self._head_of(o)(m, co, po)
                 for o, m, co, po in zip(metas, inputs, cur_ohs, prev_ohs)],
                axis=0)
        w1, w2, wl = w
        # The fc1 precision is the cast the per-head implementations use
        # (``PrerouterHead._dtype`` / ``BailingPrerouter`` reading
        # ``fc1.weight.dtype``); take it from the stacked weight so this
        # stays family-agnostic.
        dtype = w1.dtype
        feats = core.concatenate([core.stack(inputs, 0), core.stack(cur_ohs, 0),
                                  core.stack(prev_ohs, 0)], axis=-1)
        if feats.dtype != dtype:
            feats = core.astype(feats, dtype)
        f2 = feats.reshape(len(metas), -1)
        h1 = core.einsum("ni,nij->nj", f2, w1)
        act = gelu_erf(h1)
        out = (core.einsum("ni,nij->nj", f2, wl)
               + core.einsum("ni,nij->nj", act, w2))
        return out.reshape(len(metas), 1, 1, self.num_experts)

    # ---- the step-boundary batch ------------------------------------------

    def stage_all(self) -> dict[int, list[int]]:
        metas: list[int] = []
        inputs: list[core.array] = []
        cur_ohs: list[core.array] = []
        prev_ohs: list[core.array] = []
        for owner in self.state.owners:
            feats = self._features(owner)
            if feats is None:
                continue
            m_in, cur_oh, prev_oh = feats
            metas.append(owner)
            inputs.append(m_in)
            cur_ohs.append(cur_oh)
            prev_ohs.append(prev_oh)
        if not metas:
            return {}
        logits = self._logits(metas, inputs, cur_ohs, prev_ohs)  # [n,1,1,E]
        inds_all, scores_all = self._select(logits)
        flat = None
        if self.stream_layers:
            # Only the consumers' staged fills need host-side expert IDs.
            # Without them this sync + tolist is pure waste (fix 2).
            core.eval(inds_all, scores_all)
            flat = inds_all.reshape(len(metas), -1).tolist()   # ONE tolist
        expert_sets: dict[int, list[int]] = {}
        for i, owner in enumerate(metas):
            consumer = owner + 1
            experts = (sorted({int(v) for v in flat[i]})
                       if flat is not None else None)
            expert_sets[consumer] = experts or []
            self._store(consumer, logits[i], inds_all[i],
                        scores_all[i], experts)
            self._note(owner, cur_ohs[i])
        if self.records is not None and flat is not None:
            for owner in metas:
                self.records.append(
                    (self.cur_step, owner + 1, expert_sets[owner + 1]))
        self.cur_step += 1
        return expert_sets

    def _head_of(self, owner: int):
        raise NotImplementedError


class CrossTokenStager(PrerouterStager):
    """qwen semantics: heads on the MoE blocks, selection made here,
    ``pred_inds`` / ``pred_scores`` keyed by the consuming layer."""

    def __init__(self, *, model, spec: MoESpec, pspec: PrerouterSpec,
                 state: PrerouterState,
                 stream_layers: dict[int, StreamingSwitchGLU], top_k: int):
        super().__init__(model=model, spec=spec, pspec=pspec,
                         state=state, stream_layers=stream_layers,
                         top_k=top_k)

    def _head_of(self, owner: int):
        block = self.spec.block_of(self.model, owner)
        return block.prerouter_head

    def _features(self, owner: int):
        block = self.spec.block_of(self.model, owner)
        m_in = getattr(block, "prerouter_m_in", None)
        oh = getattr(block, "prerouter_oh", None)
        if m_in is None or oh is None:
            return None
        prev_oh = self.state.oh_prev[owner]
        if prev_oh is None:
            prev_oh = core.zeros((1, 1, self.num_experts), dtype=oh.dtype)
        return m_in, oh, prev_oh

    def _select(self, logits: core.array):
        return select_from_logits(logits, self.top_k)

    def _store(self, consumer: int, logits_i, inds_i, scores_i,
               experts: list[int] | None):
        st = self.state
        st.pred_inds[consumer] = inds_i
        st.pred_scores[consumer] = scores_i
        st.logits[consumer - 1] = logits_i
        exp = self.stream_layers.get(consumer)
        if exp is not None and experts:
            exp.stage_experts(experts)

    def _note(self, owner: int, cur_oh) -> None:
        pass


class LingPrerouterStager(PrerouterStager):
    """ling semantics: heads baked into the vendored model, raw logits
    cached per consumer layer (the MoE block re-selects internally with
    its own ``_select_from_logits``); teacher top-8 feature; the
    prev-token feature is rolled on the block."""

    def __init__(self, *, model, spec: MoESpec, pspec: PrerouterSpec,
                 state: PrerouterState,
                 stream_layers: dict[int, StreamingSwitchGLU], top_k: int):
        super().__init__(model=model, spec=spec, pspec=pspec,
                         state=state, stream_layers=stream_layers,
                         top_k=top_k)
        self.pg_cache: dict[int, core.array] = {}

    def _head_of(self, owner: int):
        block = self.spec.block_of(self.model, owner)
        return block.prerouter

    def _features(self, owner: int):
        layer = self.spec.layer_of(self.model, owner)
        block = self.spec.block_of(self.model, owner)
        m = getattr(layer, "m_in_cache", None)
        if m is None:
            return None
        m_in = m[..., -1:, :]
        last_topk = getattr(block, "last_topk", None)
        cur_oh = _topk_onehot_pos(last_topk, self.num_experts, pos=-1)
        prev_oh = getattr(block, "prev_topk_oh", None)
        if prev_oh is None:
            # Initial staging right after prefill: the "prev token" feature
            # is the second-to-last prefill position.
            prev_oh = _topk_onehot_pos(last_topk, self.num_experts, pos=-2)
        if cur_oh is None:
            cur_oh = core.zeros((1, 1, self.num_experts), dtype=core.float32)
        if prev_oh is None:
            prev_oh = core.zeros((1, 1, self.num_experts), dtype=core.float32)
        return m_in, cur_oh, prev_oh

    def _select(self, logits: core.array):
        return group_select_from_logits(
            logits, self.top_k,
            n_group=self.spec.n_group, topk_group=self.spec.topk_group,
            routed_scaling=self.spec.routed_scaling or 1.0)

    def _store(self, consumer: int, logits_i, inds_i, scores_i,
               experts: list[int] | None):
        self.pg_cache[consumer] = logits_i
        exp = self.stream_layers.get(consumer)
        if exp is not None and experts:
            exp.stage_experts(experts)

    def _note(self, owner: int, cur_oh) -> None:
        block = self.spec.block_of(self.model, owner)
        # Remember this token's top-k so the next decode step can feed the
        # "prev-token" one-hot feature.
        block.prev_topk_oh = cur_oh

    def reset(self):
        self.pg_cache = {}
        self.cur_step = 0
