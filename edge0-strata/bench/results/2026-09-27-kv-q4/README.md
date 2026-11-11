# 2026-09-27: Q4_0 KV with Hadamard rotation (PR #21, `--kv q4_0`)

PR #21 (code-martin) adds a 4-bit K/V format: an orthonormal 256-point Walsh-Hadamard rotation of K, V and Q, then
ggml `q4_0` blocks (576 B per cell and layer, vs 1,056 B in int8).

## Integration fixes made on merge

- **Q4 + KV streaming (`--kv-resident`).** The PR had no host copy for Q4 and skipped the staging, so with streaming
  on, the prompt path wrote K/V through the residency map's -1 entries (out-of-bounds writes).
  - Q4 is now a third format in `kv_stream`: host copy, block residency, prompt-path staging, MTP ring and STATE_HASH.
  - `kv_stream_parity` checks it bitwise, streamed vs resident, with ring restore.
- **Duplicate attention.** The per-token decode path ran attention twice, the first time before the selection's
  blocks were made resident; it now runs once.
- **Tie-break.** The group maximum's tie-break is now deterministic across lanes: equal |x| with opposite signs
  could give one block two scales.
- **Unrelated changes dropped:** duplicate path fixes (already in 0.1.7), personal `.gitignore` entries, and a
  re-encoded CMakeLists.

`kv_q4_parity` (the PR's test) and all 70 ctest tests pass.

## Quality (teacher-forced, `tools/kv_precision_compare.py`)

Same text, same fixed expert set (Q2_0, 3,000 slots, `--adapt-swaps 0`); only `--kv` differs. fp16 is the reference.

| Text | KV | Same top-1 as fp16 | Mean KL | Perplexity | vs fp16 (nats/token) |
|---|---|---:|---:|---:|---:|
| code agent (512) | int8 | 98.4% | 0.014 | 2.649 | -0.011 +- 0.006 |
| | q4_0 | 95.7% | 0.030 | 2.598 | -0.031 +- 0.014 |
| document (1,024) | int8 | 95.6% | 0.009 | 30.18 | +0.012 +- 0.005 |
| | q4_0 | 91.3% | 0.039 | 32.26 | **+0.078 +- 0.011** |
| chat (1,024) | int8 | 95.5% | 0.009 | 18.24 | -0.003 +- 0.005 |
| | q4_0 | 90.7% | 0.028 | 18.52 | +0.012 +- 0.008 |
| document, 8K (every 8th position) | int8 | 92.3% | 0.114 | 7.63 | -0.013 +- 0.020 |
| | q4_0 | 88.0% | 0.297 | 8.65 | **+0.112 +- 0.037** |

- **int8 is indistinguishable from fp16** (within about 2 standard errors everywhere).
- **q4_0 costs measurable precision, and more as the context grows:**
  - document perplexity +8% at 1K and +12% at 8K;
  - the KL from fp16 is 2.6-4.2x int8's;
  - code and chat are near-neutral.
- **Needles with `--kv q4_0 --kv-resident 32768`: 5/5** (1K, 32K, 128K at 10% and 90% depth, 262K;
  `needles-q4.*`). Retrieval holds.

## Speed and memory (Q2_0, 128K, matrix bench, `--kv-resident 32768`)

| KV | Expert slots | Decode tok/s | Tokens/round | Rounds/s | K/V in RAM |
|---|---:|---:|---:|---:|---:|
| int8 (`kv-stream/matrix-stream`) | 4,044 | 62.4 | 2.52 | 24.8 | 1.52 GiB |
| q4_0 (`matrix-q4`) | 4,190 | 67.4 | 2.62 | 25.7 (+3.7%) | 0.83 GiB |

Without streaming (contexts up to 64K, or too little RAM for it) q4_0 halves the whole KV cache in VRAM instead.

## Decision

`--kv q4_0` is a user option:
- `START-HERE.bat --kv q4_0`, or the question setup asks above 8K context;
- or `--kv q4_0` in a `strata-*.json`.

int8 stays the default: every published quality number was measured with it, and q4_0's precision cost is real,
though it passes the needles.
