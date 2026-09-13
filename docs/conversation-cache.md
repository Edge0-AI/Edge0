# Persistent conversation checkpoints

Caching is opt-in. It saves repeated prompt computation and releases completed
HTTP requests' attention/recurrent state from RAM. It does not offload the active
attention working set, quantize KV, or implement Pi's tool protocol.

```sh
edge0 serve /path/to/model --cache-dir /path/to/conversation-cache
edge0 chat /path/to/model --cache-dir /path/to/conversation-cache --prompt 'Hello'
edge0 cache inspect --cache-dir /path/to/conversation-cache
edge0 cache clear --cache-dir /path/to/conversation-cache
```

Use a separate directory from the model. Defaults are a 20 GiB checkpoint payload
budget and a 2,048-token interval. `--cache-budget-gib` and `--cache-interval`
override these. The separate exact-tokenization budget defaults to 16 MiB;
Python can configure it. SQLite's allocated pages are included in an additional
directory bound: checkpoint budget + tokenization budget + 64 KiB schema reserve.
Compaction and additional LRU eviction enforce that bound after writes. Atomic
publication and SQLite journaling require temporary disk headroom during a write.

```python
from edge0 import AutoEngine
from edge0.conversation import CacheConfig

engine = AutoEngine.from_pretrained(
    '/path/to/model', name='edge0-8b',
    conversation_cache=CacheConfig(
        directory='/path/to/conversation-cache',
        budget_bytes=20 * 1024**3,
        interval=2048,
        token_budget_bytes=16 * 1024**2,
    ),
)
engine.reset()
tokens = engine.generate(prompt_ids, max_new_tokens=128)
print(engine.conversation_cache.metrics)
engine.reset()  # release the active Python continuation when done
engine.close()
```

`generate` looks up a prefix when the engine position is zero. Direct Python
`prefill`/`step` callers retain their active continuation until `reset`. Server
requests still use the existing single generation lock; `ChatSession.run` resets
before lookup and releases request state on completion or cancellation. Expert
weight caches remain shared across requests.

The store saves interval boundaries, prompt completion, and completed generation.
Only forwarded tokens are recorded: a sampled EOS that was never forwarded is
excluded. A cancellation leaves earlier published boundaries usable. An interval
checkpoint taken during prefill carries its lifecycle phase, so an exact hit
finishes prefill staging before decode. A one-token unmatched suffix uses the
prefill lifecycle. Sampling penalties receive the full prompt history.

## Identity and persistence

Namespace identity includes model and adapter contents, tokenizer and chat-template
artifacts, the effective tokenizer vocabulary/serialization, model configuration,
checkpoint interval, family/backend environment settings, backend versions, and
Edge0 source contents. Artifact SHA-256 hashes are computed at initialization and
reused on later starts only when device, inode, size, nanosecond mtime, and ctime
match. First initialization can therefore read the whole model. Do not mutate a
loaded engine's model, adapters, tokenizer, or inference configuration in place;
construct a new engine for those changes.

A compact radix index searches token edges in memory, verifies full token equality,
and loads only the chosen boundary. Local writes update the index incrementally.
After another process changes the directory, the index is rebuilt from SQLite
metadata once; this synchronization cost is separate from steady-state radix
lookup. It does not load tensors or retain inactive conversation tensors.

Attention keys and values use immutable content-addressed safetensors blocks.
Recurrent tensors, next-token logits, position, and family prerouter fields are
saved in a boundary-specific snapshot. Expert staging is rebuilt through family
hooks. No tensor dtype conversion, pickle, or executable-object deserialization
is used. Blocks deduplicate only when their serialized contents match exactly.
Currently shared blocks are reserialized and checksummed on publication; this
cost is included in the measured write overhead.

Payload and metadata checksums turn corrupt entries into misses. A directory lock
coordinates readers, publication, recovery, and eviction across processes. Files
are fsynced and atomically renamed before transactional SQLite publication.
Unpublished managed files are collected on recovery. LRU checkpoint eviction drops
unreferenced blocks; oversized snapshots are skipped. Cleanup only removes managed
payload names. `clear` covers every model namespace in the selected cache directory.

The exact-tokenization cache keys the entire message list, template namespace,
and thinking settings. It never joins independently tokenized messages.

## Metrics and validation

HTTP usage retains the full `prompt_tokens` count and adds
`prompt_tokens_details.cached_tokens`. The optional `edge0_cache` response field
(and terminal SSE chunk) reports lookup, tokenization, restore, remaining prefill,
publication, reused-token, eviction, and stored-byte metrics. The Python store
exposes the same dictionary. `write_s` includes synchronous serialization and
checksumming; `restore_s` includes checksum validation, tensor restore, and family
staging. Remaining `prefill_s` excludes recorded checkpoint writes.

The benchmark uses a fixed 48-function Python review conversation, then an
appended turn and a branch. Run it from the repository with the installed environment:

```sh
python scripts/bench_conversation_cache.py --model-dir /path/to/model \
  --cache-dir /path/to/benchmark-cache
python scripts/bench_conversation_cache.py --model-dir /path/to/model \
  --cache-dir /path/to/benchmark-cache --restart
python scripts/bench_conversation_cache.py --lookup-only
```

The first command clears the specified benchmark cache. Model initialization and
artifact hashing occur before request timers. The restart command launches a new
model process, but does **not** flush the OS filesystem cache. TTFT is measured at
the engine's first token callback, which follows its first decode forward.
`decode_tok_s` measures sampling and decode, excluding checkpoint writes;
`post_first_token_effective_tok_s` also includes final publication and request
cleanup. RSS is sampled every 20 ms; MLX reports its own peak allocation.
System disk counters include other processes, while per-process block counters
may remain zero on macOS. These counters are not a claim of isolated SSD traffic.

See [measured results](conversation-cache-results.md) for local timings and limits.
