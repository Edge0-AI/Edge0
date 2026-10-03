"""Token sampling — port of the deployment's HF-style sampler
(vectorized repetition penalty; one host-sync categorical draw)."""

from __future__ import annotations

from edge0.backends import core


def _mask_logits(logits, temperature, top_k, top_p):
    """Temperature / top-k / top-p truncation, shared by ``sample`` and
    any speculative verifier so both draw from the exact same
    distribution."""
    logits = logits.astype(core.float32)
    if temperature > 0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        top = core.topk(logits, k, axis=-1)
        # core.topk returns values in ASCENDING order, so the smallest
        # surviving value is the FIRST entry (top[..., 0]), not the last.
        threshold = top[..., :1]
        logits = core.where(logits < threshold, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_vals = core.sort(logits, axis=-1)[..., ::-1]
        cum = core.cumsum(core.softmax(sorted_vals, axis=-1), axis=-1)
        cutoff = cum <= top_p
        counts = core.sum(cutoff.astype(core.int32), axis=-1)
        k = core.maximum(counts, 1)
        threshold = core.take_along_axis(
            sorted_vals, (k - 1)[..., None], axis=-1)
        logits = core.where(logits < threshold, float("-inf"), logits)
    if temperature <= 0:
        # Deterministic decode (HF / llama.cpp semantics): temperature
        # <= 0 means greedy — keep only the argmax (after any top-k
        # truncation above), so categorical always draws it.
        logits = core.where(
            logits < core.max(logits), float("-inf"), logits)
    return logits


def sample(logits, temperature=0.7, top_k=None, top_p=None,
           repetition_penalty=1.0, history=(), seed=None) -> int:
    """Sample one token id from raw logits (HF-style penalty).

    ``history`` is an iterable of already-generated token ids; the
    penalty applies to them (vectorized, no per-token host syncs).
    """
    if repetition_penalty != 1.0 and history:
        logits = logits.astype(core.float32)
        ids = core.array(sorted(set(int(x) for x in history)),
                       dtype=core.int32)
        if ids.size:
            vals = core.take(logits, ids)
            logits[ids] = core.where(
                vals > 0,
                vals / repetition_penalty,
                vals * repetition_penalty)
    logits = _mask_logits(logits, temperature, top_k, top_p)
    if seed is not None:
        core.random.seed(seed)
    return int(core.random.categorical(logits[None]).item())
