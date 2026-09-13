"""Opt-in CLI diagnostics; counters are per request, peaks are process lifetime."""
import ctypes
import json
import os
import sys
import time


def memory_stats():
    import psutil
    process = psutil.Process()
    result = dict(rss_bytes=process.memory_info().rss, process_threads=process.num_threads())
    if sys.platform == "darwin":
        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]
        usage = Usage()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) == 0:
            result.update(physical_footprint_bytes=usage.values[7],
                          lifetime_peak_physical_footprint_bytes=usage.values[28])
    return result


class ChatStats:
    def __init__(self, engine, verbose_tokens=False):
        self.engine = engine
        self.verbose_tokens = verbose_tokens
        self.before = engine.stats()
        self.started = time.perf_counter()
        self.first = self.last = None
        self.count = 0

    def token_record(self, phase, index, token, **extra):
        tokenizer = self.engine._tok
        record = dict(phase=phase, index=index, token_id=int(token),
                      piece=tokenizer.convert_ids_to_tokens(int(token)),
                      text=tokenizer.decode([int(token)], skip_special_tokens=False), **extra)
        # JSON escaping preserves whitespace and prevents token text emitting
        # terminal control sequences. Pieces preserve partial UTF-8 byte tokens.
        print("[edge0 token] " + json.dumps(record, ensure_ascii=True),
              file=sys.stderr, flush=True)

    def on_prompt(self, ids):
        if self.verbose_tokens:
            print(f"[edge0 tokenize] {len(ids)} prompt tokens in "
                  f"{self.engine.last_tokenization_s:.6f}s", file=sys.stderr, flush=True)
            for index, token in enumerate(ids):
                self.token_record("prompt", index, token)

    def on_token(self, token):
        now = time.perf_counter()
        previous = self.last
        if self.first is None:
            self.first = now
        self.last = now
        self.count += 1
        if self.verbose_tokens:
            self.token_record("generated", self.count - 1, token,
                              elapsed_s=now - self.started,
                              since_previous_token_s=None if previous is None else now - previous)
        if self.count == 1 or self.count % 8 == 0:
            self.emit("progress")

    def emit(self, phase, meta=None):
        stats = dict(self.engine.stats())
        for key in ("cache_hits", "cache_misses", "prefetch_hits", "cache_evictions",
                    "checkpoint_bytes_read", "ngram_embedding_rows_read",
                    "kernel_dispatches", "batch_retirements"):
            if key in stats:
                stats[key] -= self.before.get(key, 0)
        hits, misses = stats.get("cache_hits", 0), stats.get("cache_misses", 0)
        if hits + misses:
            stats["resident_cache_hit_fraction"] = hits / (hits + misses)
        stats.update(phase=phase, elapsed_s=time.perf_counter() - self.started,
                     generated_tokens=self.count, **memory_stats())
        if self.first is not None:
            stats["time_to_first_token_s"] = self.first - self.started
        if self.count > 1 and self.last > self.first:
            stats["decode_tokens_per_s_after_first"] = (self.count - 1) / (self.last - self.first)
        if meta:
            stats["usage"] = meta["usage"]
        print("[edge0 stats] " + json.dumps(stats, sort_keys=True), file=sys.stderr, flush=True)
