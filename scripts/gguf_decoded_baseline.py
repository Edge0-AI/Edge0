"""Benchmark-only reproduction of the previous decoded GGUF execution path.

Explicitly selected by validate_gguf.py; never imported by production execution.
Keeps the previous float32 NumPy LRU, dense retention policy, worker prefetch,
per-chunk matmul evaluation and per-expert reduction/evaluation.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np


class Cache:
    def __init__(self, source, budget, chunk):
        self.source, self.budget, self.chunk_bytes = source, budget, chunk
        self._cache, self._pending = OrderedDict(), {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._protected = set()
        self._dense = {n for n in source.tensors if '_exps.' not in n and n not in
                       ('token_embd.weight', 'per_layer_token_embd.weight')}
        self._retain = sum(math.prod(source.tensors[n].shape)*4 for n in self._dense)+4*chunk < budget*.95
        self.resident_bytes = self.pending_bytes = self.peak_bytes = 0
        self.hits = self.misses = self.prefetch_hits = self.evictions = self.dispatches = 0

    def _size(self, key):
        return key[2]*self.source.tensors[key[0]].shape[-1]*4

    def _room(self, size):
        while self._cache and self.resident_bytes+self.pending_bytes+size>self.budget:
            key = next((k for k in self._cache if k not in self._protected), None)
            if key is None:
                break
            self.resident_bytes -= self._cache.pop(key).nbytes
            self.evictions += 1
        return self.resident_bytes+self.pending_bytes+size<=self.budget

    def prefetch(self, keys):
        for key in keys:
            if key in self._cache or key in self._pending:
                continue
            size = self._size(key)
            if len(self._pending)>=2 or not self._room(size+self.chunk_bytes):
                break
            self.pending_bytes += size
            self._pending[key] = self._pool.submit(self.source.read_rows, *key)
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes+self.pending_bytes)

    def get(self, key):
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        future = self._pending.pop(key, None)
        if future is None:
            data = self.source.read_rows(*key)
        else:
            self.prefetch_hits += 1
            try:
                data = future.result()
            finally:
                self.pending_bytes -= self._size(key)
        if self._room(data.nbytes):
            self._cache[key] = data
            self.resident_bytes += data.nbytes
            if self._retain and key[0] in self._dense:
                self._protected.add(key)
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes+self.pending_bytes)
        return data

    def keys(self, name, start, count):
        rows = max(1, self.chunk_bytes//(self.source.tensors[name].shape[-1]*4))
        for row in range(start, start+count, rows):
            yield name, row, min(rows, start+count-row)

    def stats(self):
        return dict(decoded_cache_bytes=self.resident_bytes, pending_decode_bytes=self.pending_bytes,
                    decoded_cache_budget_bytes=self.budget, cache_hits=self.hits, cache_misses=self.misses,
                    cache_evictions=self.evictions, prefetch_hits=self.prefetch_hits,
                    projection_dispatches=self.dispatches, checkpoint_bytes_read=self.source.bytes_read)

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._cache.clear()


class Weights:
    def __init__(self, source, budget_bytes, chunk_bytes):
        self.source = source
        self.cache = Cache(source, budget_bytes, chunk_bytes)

    def vector(self, name):
        import mlx.core as mx
        return mx.array(self.cache.get((name, 0, 1))[0])

    def rows(self, name, indices):
        import mlx.core as mx
        indices = list(indices)
        # The original adapters read entire small convolution tensors directly.
        if 'conv1d' in name:
            return mx.array(self.cache.get((name, indices[0], len(indices))))
        return mx.array(np.concatenate([self.cache.get((name, int(i), 1)) for i in indices]))

    def linear(self, name, x, expert=None):
        import mlx.core as mx
        shape = self.source.tensors[name].shape
        if len(shape)==1:
            return (x@self.vector(name))[..., None]
        count = shape[-2]
        keys = list(self.cache.keys(name, (expert or 0)*count, count))
        outputs = []
        for i, key in enumerate(keys):
            self.cache.prefetch(keys[i+1:i+2])
            w = mx.array(self.cache.get(key))
            y = x@w.T
            mx.eval(y)
            self.cache.dispatches += 1
            outputs.append(y)
        out = mx.concatenate(outputs, axis=-1) if len(outputs)>1 else outputs[0]
        mx.eval(out)
        return out

    def close(self):
        self.cache.close()
        self.source.close()


class Experts:
    def __init__(self, weights, layer, n_experts):
        self.weights, self.layer, self.n_experts = weights, layer, n_experts
        self.last_used = []

    def __call__(self, x, indices, scores=None):
        import mlx.core as mx
        prefix = f'blk.{self.layer}.ffn_'
        selection = np.array(indices)
        self.last_used = sorted(set(selection.reshape(-1).tolist()))
        out = mx.zeros_like(x) if scores is not None else mx.zeros((*indices.shape, x.shape[-1]), dtype=x.dtype)
        for expert in self.last_used:
            rows, slots = np.where(selection==expert)
            xe = x[mx.array(rows)]
            g = self.weights.linear(prefix+'gate_exps.weight', xe, expert)
            u = self.weights.linear(prefix+'up_exps.weight', xe, expert)
            z = self.weights.linear(prefix+'down_exps.weight', mx.sigmoid(g)*g*u, expert)
            if scores is None:
                out = out.at[mx.array(rows), mx.array(slots)].add(z)
            else:
                out = out.at[mx.array(rows)].add(z*scores[mx.array(rows), mx.array(slots), None])
            mx.eval(out)
        return out

    def prefetch(self):
        if self.last_used:
            name = f'blk.{self.layer}.ffn_gate_exps.weight'
            rows = self.weights.source.tensors[name].shape[1]
            self.weights.cache.prefetch(self.weights.cache.keys(name, self.last_used[0]*rows, rows))

    def reset(self):
        self.last_used = []
