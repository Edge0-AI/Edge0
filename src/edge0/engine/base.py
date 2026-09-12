"""Edge0 engine base: lifecycle + shared prefill / decode loops.

Family engines (``edge0.engine.qwen`` / ``edge0.engine.ling``) implement
the model-specific hooks; everything here is model-agnostic:

* ``prefill`` — chunked prompt processing; the family hook decides
  whole-layer (E3b) vs on-demand paths per chunk, and ``_prefill_end``
  drops the prefill working set, refreshes hot pins and primes the
  staged slots for the first decode step.
* ``step`` — one-token forward + the family's step-boundary staging
  (early submit for router layers, ``sync_actuals``, prerouter
  ``stage_all`` + state swap).
* ``generate`` — sampling loop with the family ``GenerationConfig``.

Loop mechanics ported from the deployment engines (``engine_qwen.py`` /
ling ``engine.py``) so the measured production behavior — one cheap sync
point per step, fills overlapping the next forward — is preserved.
"""

from __future__ import annotations

import time

from edge0.backends import core

from edge0.config import GenerationConfig
from edge0.sampling import sample


class Edge0Engine:
    """Base streaming engine.  Subclasses fill in the family hooks."""

    name = "edge0"

    def __init__(self, model_dir: str, cfg, tokenizer=None):
        self.dir = model_dir
        self.cfg = cfg
        self._tok = tokenizer
        self.prefill_chunk = getattr(cfg, "prefill_chunk", 2048)
        self.pos = 0
        self._prefill_active = False
        self._last_logits = None
        self._cap_mlx_cache()
        t0 = time.time()
        self.conversation_cache = None
        self._build()
        self._processed_tokens = []
        self._cache_phase = "ready"
        cache_config = getattr(cfg, "conversation_cache", None)
        if cache_config and cache_config.directory:
            from pathlib import Path
            if Path(cache_config.directory).expanduser().resolve() == Path(model_dir).resolve():
                raise ValueError("use a separate directory for conversation checkpoints")
            from edge0.conversation import CheckpointStore
            from edge0.conversation.store import engine_namespace
            self.conversation_cache = CheckpointStore(cache_config, "initializing")
            self.conversation_cache.namespace = engine_namespace(self, self.conversation_cache)
        print(f"[{self.name}] built in {time.time() - t0:.1f}s", flush=True)

    # ---- family hooks -----------------------------------------------------

    def _build(self):
        raise NotImplementedError

    def _forward(self, ids, intra_stage: bool = True) -> core.array:
        raise NotImplementedError

    def _step_pre(self, token_id: int) -> None:
        """Called before a decode forward (early-submit fills)."""

    def _step_post(self, token_id: int) -> None:
        """Called after a decode forward's logits eval (staging/swap)."""

    def _prefill_end(self) -> None:
        """Called after the last prefill chunk."""

    def _reset_state(self) -> None:
        pass

    def _lm_logits(self, h: core.array) -> core.array:
        raise NotImplementedError

    # ---- shared loops -----------------------------------------------------

    def prefill(self, token_ids, chunk_size=None, on_progress=None):
        if chunk_size is None:
            chunk_size = self.prefill_chunk
        if chunk_size < 1:
            raise ValueError("chunk size must be positive")
        total = len(token_ids)
        done = 0
        self._prefill_active = True
        try:
            start = 0
            while start < total:
                size = chunk_size
                if self.conversation_cache:
                    interval = self.conversation_cache.config.interval
                    size = min(size, interval - self.pos % interval)
                chunk = token_ids[start:start + size]
                self._last_logits = self._forward(chunk)
                self.pos += len(chunk)
                done += len(chunk)
                start += len(chunk)
                if self.conversation_cache:
                    self._processed_tokens.extend(chunk)
                    if self.pos % self.conversation_cache.config.interval == 0 and done < total:
                        self._save_checkpoint('prefill')
                if on_progress is not None:
                    on_progress(done, total)
        finally:
            self._prefill_active = False
        self._prefill_end()
        self._cache_phase = "ready"
        self._save_checkpoint()
        return total

    def next_logits(self) -> core.array:
        return self._last_logits

    def step(self, token_id: int) -> core.array:
        self._step_pre(token_id)
        logits = self._forward([token_id])
        self.pos += 1
        self._step_post(token_id)
        if self.conversation_cache:
            self._last_logits = logits
            self._processed_tokens.append(token_id)
            if self.pos % self.conversation_cache.config.interval == 0:
                self._save_checkpoint()
        return logits

    def generate(self, token_ids, gen_config: GenerationConfig | None = None,
                 max_new_tokens: int | None = None, on_token=None) -> list[int]:
        """Prefill ``token_ids`` then sample until EOS or the cap."""
        if gen_config is None:
            gen_config = getattr(self.cfg, "gen", GenerationConfig())
        if max_new_tokens is None:
            max_new_tokens = gen_config.max_new_tokens
        suffix = token_ids
        if self.conversation_cache and self.pos == 0:
            self.conversation_cache.metrics = {}
            from edge0.backends.mlx.checkpoint import read
            hit = self.conversation_cache.restore(token_ids, read)
            if hit:
                saved, (caches, logits, state, pos, phase) = hit
                try:
                    if len(caches) != len(self.cache) or pos != len(saved):
                        raise ValueError("incompatible checkpoint")
                    self.cache, self._last_logits, self.pos = caches, logits, pos
                    family_started = time.perf_counter()
                    self._restore_checkpoint_family(state)
                    self.conversation_cache.metrics["restore_s"] += time.perf_counter() - family_started
                    self._processed_tokens = list(saved)
                    self._cache_phase = phase
                    suffix = token_ids[len(saved):]
                except (ValueError, KeyError, TypeError, RuntimeError):
                    self.reset()
                    self.conversation_cache.metrics['reused_tokens'] = 0
        started = time.perf_counter()
        if self.conversation_cache:
            if len(suffix) == 1 and self.pos == 0:
                # Preserve the existing fresh single-token prompt path.
                self._last_logits = self._forward(suffix)
                self.pos = 1
                self._processed_tokens = list(suffix)
                self._save_checkpoint()
            elif suffix:
                # Even a one-token restored suffix needs the prefill lifecycle.
                self.prefill(suffix)
            elif self._cache_phase == 'prefill':
                self._prefill_end()
                self._cache_phase = 'ready'
            self.conversation_cache.metrics['prefill_s'] = max(0, time.perf_counter() - started - self.conversation_cache.metrics.get('write_s', 0))
            self.conversation_cache.metrics['remaining_prefill_tokens'] = len(suffix)
        elif len(token_ids) > 1:
            self.prefill(token_ids)
        elif len(token_ids) == 1:
            self._last_logits = self._forward(token_ids)
            self.pos += 1
        self.last_generation_metrics = {
            'prefill_s': (self.conversation_cache.metrics['prefill_s']
                          if self.conversation_cache else time.perf_counter() - started)}
        decode_started = time.perf_counter()
        writes_before_decode = self.conversation_cache.metrics.get('write_s', 0) if self.conversation_cache else 0
        out: list[int] = []
        history = list(token_ids)
        logits = self.next_logits()
        first_token = True
        for _ in range(max_new_tokens):
            if logits is None:
                raise RuntimeError("no logits available before first step")
            if first_token and gen_config.first_token_greedy:
                # The FIRST token is always greedy
                # (production-verified pattern): after the
                # think opener a randomly-sampled first token can derail
                # the whole block into '!' loops.
                tid = int(core.argmax(logits, axis=-1).item())
            else:
                tid = sample(
                    logits,
                    temperature=gen_config.temperature,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                    repetition_penalty=gen_config.repetition_penalty,
                    history=history,
                    seed=gen_config.seed,
                )
            first_token = False
            if gen_config.is_eos(tid):
                break
            out.append(tid)
            history.append(tid)
            # The deployment's step() returns the next logits and does NOT
            # update _last_logits (engine_qwen.py:471-474), so the loop
            # must carry the step return value — re-reading next_logits()
            # here would replay the prefill-final logits forever (the
            # greedy repetition bug: [353]*8 instead of [353, 2688, ...]).
            logits = self.step(tid)
            if on_token is not None:
                on_token(tid)
        decode_s = time.perf_counter() - decode_started
        if self.conversation_cache:
            decode_s -= self.conversation_cache.metrics.get('write_s', 0) - writes_before_decode
        self.last_generation_metrics['decode_s'] = max(0, decode_s)
        self.last_generation_metrics['decode_tok_s'] = len(out) / decode_s if decode_s > 0 else 0
        self._save_checkpoint()
        return out

    def _save_checkpoint(self, phase="ready"):
        if self.conversation_cache and self._processed_tokens:
            from edge0.backends.mlx.checkpoint import write
            try:
                self.conversation_cache.publish(self._processed_tokens,
                    lambda put: write(self, put, phase))
            except (OSError, ValueError, RuntimeError) as exc:
                self.conversation_cache.metrics["last_write_error"] = str(exc)
                self.conversation_cache.metrics["write_errors"] = self.conversation_cache.metrics.get("write_errors", 0) + 1

    def reset(self):
        self._reset_state()
        self._processed_tokens = []
        self._cache_phase = "ready"
        self.pos = 0
        self._last_logits = None

    def stats(self) -> dict:
        layers = getattr(self, "_all_stream_layers", {})
        return {li: exp.stats() for li, exp in layers.items()}

    def close(self):
        for exp in getattr(self, "_all_stream_layers", {}).values():
            try:
                exp.close()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass

    # ---- misc -------------------------------------------------------------

    @staticmethod
    def _cap_mlx_cache():
        import os
        try:
            mb = int(os.environ.get("MLX_CACHE_LIMIT_MB", "256"))
            core.set_cache_limit(mb * 1024 * 1024)
        except Exception:  # noqa: BLE001 — best-effort
            pass
