# Custom local checkpoints (experimental)

`--model-path` selects a local GGUF file (any shard of a split checkpoint)
or a supported MLX safetensors checkpoint directory. It performs no automatic
model downloads. Do not combine it with a named tier or `--model-dir`.

```bash
MLX_ENABLE_TF32=0 edge0 chat \
  --model-path ~/Documents/Qwen-3.8/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --prompt "Explain why the sky is blue." --max-new 128
```

The same options work with `edge0 demo` and `edge0 serve`. Add
`--tokenizer-path /local/tokenizer-directory` when GGUF tokenizer metadata is
incomplete. The override must include a compatible vocabulary, EOS token, and
chat template. Missing metadata produces an error instead of a guessed template.

Add `--stats` to `edge0 chat` for JSON diagnostics on stderr at request start,
the first generated token, every eight generated tokens, and completion or
interruption. These include packed-weight cache budget/occupancy, resident hit fraction,
misses, prefetch hits, evictions, checkpoint bytes read, time to first token,
and generation tokens/s after the first token. Counters are per request; cache
and physical-memory peaks are lifetime measurements. A prefetch hit means a
pending packed read was reused, not necessarily that it finished before demand.
Checkpoint bytes are requested mapped bytes, not measured physical SSD traffic.
RSS, process thread count, and macOS physical footprint are also reported.
`edge0 serve` accepts the same `--stats` and `--verbose-tokens` flags for
each streaming or non-streaming request. Diagnostics go to server stderr;
request timing starts after acquiring the generation lock and excludes queue wait.

```bash
PYTHONPATH=src MLX_ENABLE_TF32=0 ../Edge0/.venv/bin/python -m edge0.cli serve \
  --model-path ~/Documents/Qwen-3.8/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --host 127.0.0.1 --port 8000 --stats --cache-mib 3072
```

Time to first token includes tokenization and prefill. Total footprint exceeds
the packed-weight cache budget. High eviction/read counts help identify cache churn;
compare speed and physical footprint across equivalent prompts before enlarging
the cache. Diagnostics are disabled unless the flag is supplied.

Use `--verbose-tokens` for individual token logs (it also enables `--stats`).
It logs the actual templated prompt's token IDs, vocabulary pieces, decoded text,
and tokenization duration, then every generated token with elapsed time and the
interval since the preceding token. Whitespace and control characters are JSON
escaped. A byte token's decoded text may be a replacement character; its vocabulary
piece and ID remain available. Token logs contain prompt and response content.
Logging overhead is included in timing measurements.

Set the chat or server packed-weight cache budget with `--cache-mib 3072` for 3 GiB, or
`--cache-mib 1024` for 1 GiB. This option requires a GGUF `--model-path` and
an integer budget of at least 8 MiB. Omitting it preserves the model default.
It controls packed weights and their staging buffers.

```python
from edge0 import AutoConfig, AutoEngine, AutoModel

cfg = AutoConfig.from_pretrained(model_path="/local/checkpoint.gguf")
engine = AutoEngine.from_pretrained(
    model_path="/local/checkpoint.gguf",
    weight_cache_bytes=3 * 1024**3,  # Qwen3.8 Flash Next default; optional override
    weight_chunk_bytes=8 * 1024**2,
)
try:
    ids = engine.encode_chat([{"role": "user", "content": "Hello"}])
    engine.prefill(ids)
finally:
    engine.close()
```

`AutoModel.from_pretrained` accepts the same checkpoint selector. Close its GGUF
model after use. Existing named-tier and `model_dir` workflows retain their
trained adapters and routing defaults. Custom checkpoints derive configuration
from their own metadata and do not use Edge0 LoRA, prerouter, reduced routing
width, or hidden-state clipping defaults.

## Implemented formats and validation status

| Format / architecture | Scope | Validation status |
|---|---|---|
| GGUF `qwen4exp` | Text, Gated DeltaNet, sparse attention/indexer, hyperconnections, n-gram embeddings, native MoE routing | Tiny-model reference and real UD-IQ1_S short-prompt parity passed; long-run acceptance incomplete |
| GGUF `qwen35moe` | Text through existing Qwen3.5 backbone | Tiny-model reference passed; no real compatible GGUF checkpoint available for acceptance |
| Custom MLX safetensors Qwen3.5 MoE | Existing quantized text backbone, native configuration | Configuration and existing regression tests; real custom checkpoint acceptance pending |
| Named Edge0 tiers | Existing safetensors pipeline | Existing regression suite |

GGUF encoding is per tensor. Implemented IDs are F32 (0), Q8_0 (8), Q4_K
(12), Q5_K (13), Q6_K (14), IQ2_XXS (16), IQ1_S (19), IQ4_NL (20), and BF16
(30). Every decoder is tested against pinned upstream C reference values.
Other encodings and architectures are rejected. An IQ1_S filename alone does
not determine compatibility.

Vision and speculative MTP decoding are excluded. Unknown tensors and unsupported
text configurations fail validation; checkpoints containing auxiliary tensors
outside the implemented schema are not accepted. The adapters are not a claim
of compatibility with every checkpoint in either family.

## Storage, memory, and precision

GGUF files are mapped without whole-file warming. Split numbering, tensor counts,
duplicate names, bounds, and architecture metadata are checked before payload
loading. The local three-shard checkpoint's metadata-only first shard is supported.
Keep all checkpoint files immutable while the engine is open.

All projections consume their original GGUF blocks directly in custom Metal
kernels, with float32 activations and accumulation. F32, BF16, Q8_0, Q4_K,
Q5_K, Q6_K, IQ2_XXS, IQ1_S, and IQ4_NL are supported per tensor, including
mixed encodings. Unsupported types fail during header validation with the tensor
name and encoding. There is no production CPU-decoder fallback or requantization.
Only requested embedding rows, normalization vectors, and small convolution
weights become decoded arrays.

The packed cache defaults to 3 GiB for `qwen4exp` and 512 MiB otherwise.
`weight_cache_bytes` and `weight_chunk_bytes` configure its budget and 8 MiB target
chunk size. The old `decoded_cache_bytes` and `decode_chunk_bytes` names are
accepted with deprecation warnings; conflicting old and new values are rejected.
For minimum budgets, effective chunks shrink to leave space for staging. Oversized
rows are split on complete GGML blocks and their partial products accumulated.

Routing selections move to the host once per layer. Up to 16 selected experts
share each gate, up, and down dispatch, with bounded row/column chunks. All routing
slots and scores are retained, including repeated experts. Reduction follows slot
order. Synchronization happens when bounded batches retire; there is no loop that
synchronizes separately for every expert. A worker reads packed bytes, while the
calling thread creates and evaluates MLX arrays.

Cache diagnostics include resident, pending, in-flight, and staging bytes, peaks,
hits, misses, evictions, dispatches, and batch retirements. In-flight bytes are a
subset of resident bytes and are not added twice to total occupancy. Held buffers
cannot be evicted until their queued work retires. Checkpoint-read counters measure
requested bytes, not physical SSD traffic. Model state, decoded small tensors,
activations, allocator overhead, and mapped pages are outside the packed buffer
budget; it is not a process memory limit.

Persistent conversation caching is outside this branch. Integration with the
separate conversation-cache proposal is deferred to a follow-up. Ordinary
in-memory attention and recurrent state remain part of inference. Named
safetensors tiers are unchanged.

Set `MLX_ENABLE_TF32=0` before the first MLX matrix multiplication for reference
precision on M5. The loader sets this automatically and checks the active mode;
if an earlier operation cached TF32 mode, restart the process with the environment
variable set. MLX and mlx-lm dependency pins remain unchanged.

## Packed-path validation status

The standalone regression suite passed 138 tests, with one skipped and two slow
tests deselected. On the earlier combined development branch, two local M5 runs
each completed 512-token prefill and 128-token
generation using a 3 GiB packed cache. Prefill took 215.42 s and 270.44 s;
generation took 147.42 s and 121.37 s. The process lifetime peak physical footprint
was 4.65 GiB. These are local observations, not controlled speedup measurements.
[Recorded statistics](benchmarks/gguf-packed-m5.json) also include a separate
21-prompt-token, 32-generated-token CLI run with diagnostics enabled.

Matched performance comparisons against the former decoded path, compilation
measurements, and broader real-checkpoint numerical validation remain unfinished.

## Historical decoded-path measurements

The following measurements predate packed execution. Measured on a 24 GB Apple M5 with MLX 0.30.6, using the three local Unsloth
Qwen3.8 Flash Next UD-IQ1_S shards, on 2026-09-12:

| Check | Result |
|---|---|
| Model construction | 2.17 s; zero tensor payload bytes read; approximately 0.36 GiB sampled physical footprint |
| Short smoke test, 512 MiB decoded cache | 13 chat tokens; 63.66 s prefill; one generated token `Hello` in 16.35 s; approximately 1.37 GiB physical footprint |
| Real short-prompt logits vs float32 reference | Maximum absolute error 1.91e-5; RMS error 2.58e-6; same greedy token |
| Same logits vs standard CPU reference | Maximum absolute error 3.07; RMS error 0.454; same greedy token; CPU Q8 activation quantization changes arithmetic |
| Unicode / special-token sample | Exact token IDs matched upstream vocabulary runtime |
| 512-token prompt + 128 generated tokens, repeated | Interrupted before a completed run; last observed kernel peak approximately 19.3 GiB, with heavy swapping |

The float32 reference uses the validation-only patch described below. It must not
be described as numerical parity with the unmodified CPU runtime. Short-prompt
parity does not establish long-context, generated-sequence, or memory acceptance.

## Reproducing validation

The pinned reference is llama.cpp commit
`6c84c7d5d8833c6e0df69628f75a0f599797934e`. Its MIT license and attribution are
included in this repository. Check out that exact revision separately; reference
tools do not download models.

```bash
PYTHONPATH=src MLX_ENABLE_TF32=0 python -m pytest -q
PYTHONPATH=src python scripts/validate_gguf.py /local/model.gguf --inspect
PYTHONPATH=src python scripts/validate_gguf.py /local/model.gguf \
  --prompt Hello --max-new 1 --cache-mib 512 --output /tmp/gguf-smoke.json
```

The suite covers malformed and split containers, reference quantization values,
selective reads, tokenizer and argument errors, tiny-model reference logits,
chunked prefill, state restoration, reset, and tiny-engine CLI and HTTP responses.
Real-weight tests are excluded by the repository's default pytest configuration.

Build the pinned reference with CMake using `GGML_METAL=OFF`. Generate quant
fixtures with:

```bash
PYTHONPATH=src python scripts/gguf_quant_reference.py /reference/build/bin/libggml-base.dylib
```

Compile `scripts/gguf_reference_logits.cpp` against the pinned checkout's `include`
and `ggml/include` directories, linking `llama`, `ggml`, and `ggml-base`, and setting
the runtime library path to its build `bin` directory. The harness takes a model
path, an output file for float32 logits, and comma-separated input token IDs:
`reference MODEL OUTPUT.f32 1,3,7,4,2,256,12,10,16`.
For vocabulary parity use `reference MODEL OUTPUT.json --tokenize TEXT`.
Tiny checkpoints are generated by `tests/gguf_fixture.py`; corresponding committed
logits are in `tests/fixtures/qwen4exp_logits.npy` and `qwen35_logits.npy`.

For float32 mathematical parity, apply `scripts/gguf_reference_fp32.patch` to that
checkout and rebuild `ggml-cpu`. Set `EDGE0_GGML_FP32_REFERENCE=1` when running the
harness. The patch replaces CPU matrix multiplication's Q8 activation conversion
with bounded per-row weight dequantization and float32 dot products. Without that
environment variable, the original reference arithmetic is retained.

The full acceptance benchmark remains pending. Resume only with sufficient system
headroom, monitoring physical footprint rather than RSS. `--memory-pid PID` on
the validation script reports Darwin current and lifetime peak physical footprint
without loading MLX or the model. A real Qwen3.5 checkpoint, full generated-output
comparison, repeated-request memory validation, and real safetensors regression
are still required before contribution acceptance.
