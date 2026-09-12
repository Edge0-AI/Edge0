"""Bounded packed GGUF buffers. Workers read bytes; only the caller uses MLX."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
from edge0.checkpoints.encodings import QUANT_SIZES, ENCODINGS


class PackedCache:
    def __init__(self, source, budget_bytes, chunk_bytes):
        if chunk_bytes < 4096 or budget_bytes < chunk_bytes:
            raise ValueError('weight cache budget must be >= weight chunk size >= 4096')
        self.source, self.budget = source, budget_bytes
        # Reserve space for a demand buffer and its host staging copy.
        self.chunk_bytes = min(chunk_bytes, budget_bytes // 3)
        self.target_chunk_bytes = chunk_bytes
        self._cache = OrderedDict()
        self._pending = {}
        self._held = set()
        self._staging = []
        self._outputs = []
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gguf-read')
        self._closed = False
        self.resident_bytes = self.pending_bytes = self.staging_bytes = 0
        self.peak_bytes = self.staging_peak_bytes = self.inflight_peak_bytes = 0
        self.hits = self.misses = self.prefetch_hits = self.evictions = 0
        self.dispatches = self.retirements = 0

    def _range(self, key):
        name, start, count, *columns = key
        e = self.source.tensors[name]
        block, size = QUANT_SIZES[e.encoding]
        col, width = columns or (0, e.shape[-1])
        if start < 0 or count < 1 or start+count > math.prod(e.shape[:-1]):
            raise ValueError(f'row range outside tensor {name!r}')
        if col < 0 or width < 1 or (col+width)>e.shape[-1] or col%block or width%block:
            raise ValueError(f'invalid block range for {name!r}')
        if columns and count != 1:
            raise ValueError('column slices require one row')
        return (start*e.shape[-1]+col)//block*size, count*width//block*size

    def _size(self, key):
        return self._range(key)[1]

    @property
    def inflight_bytes(self):
        # In-flight buffers are a subset of resident buffers, counted once.
        return sum(self._cache[k].nbytes for k in self._held)

    def _peak(self):
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes+self.pending_bytes+self.staging_bytes)
        self.staging_peak_bytes = max(self.staging_peak_bytes, self.staging_bytes)
        self.inflight_peak_bytes = max(self.inflight_peak_bytes, self.inflight_bytes)

    def _room(self, size):
        while self.resident_bytes+self.pending_bytes+self.staging_bytes+size > self.budget:
            key = next((k for k in self._cache if k not in self._held), None)
            if key is None:
                return False
            self.resident_bytes -= self._cache.pop(key).nbytes
            self.evictions += 1
        return True

    def _read(self, key):
        offset, size = self._range(key)
        return self.source.read_bytes(key[0], offset, size)

    def prefetch(self, keys):
        if self._closed:
            return
        for key in keys:
            if key in self._cache or key in self._pending:
                continue
            size = self._size(key)
            if size > self.chunk_bytes or len(self._pending)>=2 or not self._room(size+2*self.chunk_bytes):
                break
            self.pending_bytes += size
            self._pending[key] = self._pool.submit(self._read, key)
            self._peak()

    def get(self, key):
        from edge0.backends import core as mx
        if self._closed:
            raise RuntimeError('GGUF cache is closed')
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            self._held.add(key)
            self._peak()
            return self._cache[key]
        size = self._size(key)
        if size > self.chunk_bytes:
            raise ValueError('packed range exceeds bounded chunk size')
        self.misses += 1
        future = self._pending.pop(key, None)
        if future is not None:
            self.prefetch_hits += 1
            try:
                raw = future.result()
            finally:
                self.pending_bytes -= size
        else:
            raw = None
        if not self._room(2*size):
            self.retire()
        if not self._room(2*size):
            self.cancel_prefetch()
        if not self._room(2*size):
            raise RuntimeError('packed cache could not reserve staging and device buffer')
        raw = self._read(key) if raw is None else raw
        self._staging.append(raw)
        self.staging_bytes += size
        buffer = mx.array(np.frombuffer(raw, dtype=np.uint8))
        self._cache[key] = buffer
        self.resident_bytes += size
        self._held.add(key)
        self._peak()
        return buffer

    def track(self, output):
        self._outputs.append(output)
        self.dispatches += 1

    def retire(self):
        from edge0.backends import core as mx
        try:
            if self._outputs or self._held:
                mx.eval(*self._outputs, *(self._cache[k] for k in self._held))
                self.retirements += 1
        except BaseException:
            mx.synchronize()
            raise
        finally:
            self._outputs.clear()
            self._held.clear()
            self._staging.clear()
            self.staging_bytes = 0

    def keys(self, name, start, count):
        e = self.source.tensors[name]
        block, size = QUANT_SIZES[e.encoding]
        stride = e.shape[-1]//block*size
        if stride > self.chunk_bytes:
            width = (self.chunk_bytes//size)*block
            for row in range(start, start+count):
                for col in range(0, e.shape[-1], width):
                    yield (name, row, 1, col, min(width, e.shape[-1]-col))
        else:
            rows = max(1, self.chunk_bytes//stride)
            for row in range(start, start+count, rows):
                yield (name, row, min(rows, start+count-row))

    def cancel_prefetch(self):
        jobs, self._pending = self._pending, {}
        error = None
        for job in jobs.values():
            try:
                if not job.cancel():
                    job.result()
            except BaseException as exc:
                error = error or exc
        self.pending_bytes = 0
        if error is not None:
            raise error

    def clear(self):
        try:
            self.retire()
        finally:
            try:
                self.cancel_prefetch()
            finally:
                self._cache.clear()
                self.resident_bytes = 0

    def close(self):
        if self._closed:
            return
        try:
            self.clear()
        finally:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)

    def stats(self):
        return dict(packed_cache_bytes=self.resident_bytes, pending_packed_bytes=self.pending_bytes,
            inflight_packed_bytes=self.inflight_bytes, staging_bytes=self.staging_bytes,
            peak_cached_pending_staging_bytes=self.peak_bytes,
            staging_peak_bytes=self.staging_peak_bytes, inflight_peak_bytes=self.inflight_peak_bytes,
            cache_hits=self.hits, cache_misses=self.misses, prefetch_hits=self.prefetch_hits,
            cache_evictions=self.evictions, weight_cache_budget_bytes=self.budget,
            weight_chunk_bytes=self.target_chunk_bytes, effective_weight_chunk_bytes=self.chunk_bytes,
            kernel_dispatches=self.dispatches, batch_retirements=self.retirements,
            checkpoint_bytes_read=self.source.bytes_read,
            largest_checkpoint_read_bytes=self.source.largest_read_bytes,
            ngram_embedding_rows_read=self.source.embedding_rows_read)


class GGUFWeights:
    def __init__(self, source, budget_bytes, chunk_bytes):
        import os
        from edge0.backends import core as mx
        # Attention/state operations still use MLX matmul; preserve float32 there.
        os.environ['MLX_ENABLE_TF32'] = '0'
        probe = mx.full((32, 32), 1.0001, dtype=mx.float32)
        product = probe @ probe
        mx.eval(product)
        if abs(product[0, 0].item()-32*np.float32(1.0001)**2)>0.001:
            raise ValueError('GGUF requires full float32 matmul; restart with MLX_ENABLE_TF32=0 before importing MLX')
        self.source = source
        self.cache = PackedCache(source, budget_bytes, chunk_bytes)

    def vector(self, name):
        if len(self.source.tensors[name].shape) != 1:
            raise ValueError(f'expected vector: {name}')
        return self.rows(name, [0])[0]

    def rows(self, name, indices):
        from edge0.backends import core as mx
        e = self.source.tensors[name]
        indices = list(indices)
        runs = []
        for row in map(int, indices):
            if runs and row == runs[-1][0]+runs[-1][1]:
                runs[-1] = (runs[-1][0], runs[-1][1]+1)
            else:
                runs.append((row, 1))
        outputs, parts = [], []
        for start, count in runs:
            for key in self.cache.keys(name, start, count):
                width = key[4] if len(key)==5 else e.shape[-1]
                out = ENCODINGS[e.encoding].rows(self.cache.get(key), key[2], width)
                self.cache.track(out)
                self.cache.retire()
                if len(key)==5:
                    parts.append(out)
                    if key[3]+width != e.shape[-1]:
                        continue
                    out = mx.concatenate(parts, axis=-1)
                    parts = []
                outputs.append(out)
        if name == 'per_layer_token_embd.weight':
            self.source.embedding_rows_read += len(indices)
        return mx.concatenate(outputs, axis=0) if outputs else mx.zeros((0, e.shape[-1]))

    def linear(self, name, x, expert=None):
        from edge0.backends import core as mx
        e = self.source.tensors[name]
        if len(e.shape)==1:
            return (x @ self.vector(name))[..., None]
        if expert is None and len(e.shape)!=2:
            raise ValueError(f'expert index required for {name}')
        if expert is not None and (len(e.shape)!=3 or not 0<=expert<e.shape[0]):
            raise ValueError(f'invalid expert slice {expert} for {name}')
        return self._project(name, x, [expert or 0], mx.zeros((math.prod(x.shape[:-1]),), dtype=mx.uint32))

    def _project(self, name, x, experts, ids):
        from edge0.backends import core as mx
        e = self.source.tensors[name]
        rows = e.shape[-2]
        if x.shape[-1] != e.shape[-1]:
            raise ValueError(f'input width mismatch for {name}')
        outputs = []
        partial = None
        # Multiple experts share a dispatch. Bound device plus staging for the group.
        block, size = QUANT_SIZES[e.encoding]
        stride = e.shape[-1]//block*size
        group_chunk = min(self.cache.chunk_bytes, self.cache.budget//(3*len(experts)))
        if stride <= group_chunk:
            step = max(1, group_chunk//stride)
            chunks = [(r, min(step, rows-r), None) for r in range(0, rows, step)]
        else:
            width = max(block, group_chunk//size*block)
            chunks = [(r, 1, (c, min(width, e.shape[-1]-c))) for r in range(rows)
                      for c in range(0, e.shape[-1], width)]
        for chunk_index, (row, count, columns) in enumerate(chunks):
            keys = [(name, expert*rows+row, count, *(() if columns is None else columns)) for expert in experts]
            self.cache.retire()
            self.cache._held.update(k for k in keys if k in self.cache._cache)
            needed = 2*sum(self.cache._size(k) for k in keys if k not in self.cache._cache)
            if not self.cache._room(needed):
                self.cache.cancel_prefetch()
            if not self.cache._room(needed):
                raise RuntimeError('expert batch exceeds packed buffer budget')
            buffers = [self.cache.get(key) for key in keys]
            xe = x if columns is None else x[..., columns[0]:columns[0]+columns[1]]
            out = ENCODINGS[e.encoding].matmul(buffers, xe, ids, count)
            self.cache.track(out)
            if chunk_index+1 < len(chunks):
                next_row, next_count, next_columns = chunks[chunk_index+1]
                mx.async_eval(out)
                self.cache.prefetch((name, expert*rows+next_row, next_count,
                    *(() if next_columns is None else next_columns)) for expert in experts)
            self.cache.retire()
            del buffers
            if columns is not None:
                partial = out if partial is None else partial+out
                mx.eval(partial)
                if sum(columns) != e.shape[-1]:
                    continue
                out, partial = partial, None
            outputs.append(out)
        return mx.concatenate(outputs, axis=-1) if len(outputs)>1 else outputs[0]

    def close(self):
        try:
            self.cache.close()
        finally:
            self.source.close()


class GGUFExperts:
    """Demand-load every selected expert; reduce in routing slot order."""
    def __init__(self, weights, layer, n_experts):
        self.weights, self.layer, self.n_experts = weights, layer, n_experts
        self.last_used = []

    def __call__(self, x, indices, scores=None):
        from edge0.backends import core as mx
        prefix = f'blk.{self.layer}.ffn_'
        selection = np.array(indices)  # The only routing transfer to the host.
        if selection.ndim != 2 or np.any(selection<0) or np.any(selection>=self.n_experts):
            raise ValueError('invalid routed expert selection')
        self.last_used = sorted(set(selection.reshape(-1).tolist()))
        # Limit kernel pointer count and guarantee one block plus staging per expert.
        max_block = max(QUANT_SIZES[self.weights.source.tensors[prefix+p+'_exps.weight'].encoding][1]
                        for p in ('gate', 'up', 'down'))
        batch_size = max(1, min(16, self.weights.cache.budget//(3*max_block)))
        output = mx.zeros((*selection.shape, x.shape[-1]), dtype=mx.float32)
        for start in range(0, len(self.last_used), batch_size):
            experts = self.last_used[start:start+batch_size]
            rows, slots = np.where(np.isin(selection, experts))
            ids = mx.array(np.searchsorted(experts, selection[rows, slots]).astype(np.uint32))
            ri, si = mx.array(rows), mx.array(slots)
            xe = x[ri]
            gate = self.weights._project(prefix+'gate_exps.weight', xe, experts, ids)
            up = self.weights._project(prefix+'up_exps.weight', xe, experts, ids)
            z = self.weights._project(prefix+'down_exps.weight', mx.sigmoid(gate)*gate*up, experts, ids)
            output = output.at[ri, si].add(z)
            mx.eval(output)  # Bounded expert batch retirement, never per-expert.
        if scores is None:
            return output
        # Explicit slot order avoids nondeterministic scatter accumulation.
        result = output[:, 0]*scores[:, 0, None]
        for slot in range(1, selection.shape[1]):
            result = result+output[:, slot]*scores[:, slot, None]
        mx.eval(result)
        return result

    def prefetch(self):
        if self.last_used:
            name = f'blk.{self.layer}.ffn_gate_exps.weight'
            rows = self.weights.source.tensors[name].shape[1]
            self.weights.cache.prefetch(self.weights.cache.keys(name, self.last_used[0]*rows, rows))

    def reset(self):
        self.last_used = []
