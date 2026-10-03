"""Expert-bundle caches shared by all streaming MoE layers.

* ``SharedExpertCache`` — global LRU across layers (one shared memory
  budget: a layer that thrashes evicts entries from other layers, which is
  exactly the behavior that keeps the whole-model budget flat).
* ``PrefetchBuffer`` — cap-bounded buffer of eagerly prefetched bundles
  that have NOT yet been promoted into the LRU.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class SharedExpertCache:
    """Global LRU across all MoE layers (shared memory budget)."""

    def __init__(self, slots: int):
        self.slots = max(0, slots)
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._cache.get(key)
            if v is not None:
                self._cache.move_to_end(key)
            return v

    def peek(self, key):
        with self._lock:
            return self._cache.get(key)

    def put(self, key, bundle):
        with self._lock:
            self._cache[key] = bundle
            if self.slots > 0 and len(self._cache) > self.slots:
                self._cache.popitem(last=False)

    def pop(self, key):
        with self._lock:
            return self._cache.pop(key, None)

    def size(self):
        with self._lock:
            return len(self._cache)


class PrefetchBuffer:
    """Cap-bounded staging area for prefetched bundles.

    ``put`` evicts the oldest entry when over capacity (counted as
    ``prefetch_wasted`` by the caller — the build was spent, the bundle was
    never used).  ``pop`` removes a bundle when the layer actually consumes
    it (it is then promoted into the shared LRU by the caller).
    """

    def __init__(self, cap: int = 48):
        self.cap = max(0, cap)
        self._buf = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key, bundle):
        with self._lock:
            self._buf[key] = bundle
            evicted = None
            if self.cap > 0 and len(self._buf) > self.cap:
                evicted = self._buf.popitem(last=False)
            return evicted[0] if evicted is not None else None

    def pop(self, key):
        with self._lock:
            return self._buf.pop(key, None)

    def contains(self, key) -> bool:
        with self._lock:
            return key in self._buf

    def set_cap(self, cap: int):
        self.cap = max(0, cap)
