# Community benchmark on 2x Intel Arc Pro B60

Measured on 2026-10-04 by LocalXPU. This is the SYCL port (`sycl/`) on two Arc Pro B60 cards, one request at a time, at an 8,192-token context. It is a short test: two prompt sizes, 2 to 3 requests each. It is not a 4K/32K/128K sweep.

## Hardware and software

- 2x Intel Arc Pro B60 24 GB (`8086:e211`, 456 GB/s each), PCIe 3.0 x8 per card; Ryzen 5 5600; 64 GB DDR4; storage type not recorded
- Ubuntu 24.04, kernel 6.17 (`xe` driver), Level Zero V2, compute runtime NEO 26.09.37435.12, oneAPI DPC++ 2026.1.1, oneMKL 2026.1.0
- Strata `6f32ec0` plus two `sycl/` fixes (the build fix for #559/#626 and the `cudaStreamQuery` fix in the ring wait); engine reports `0.1.39-sycl`; AOT `-DSTRATA_SYCL_AOT=bmg-g21`; source build, no Docker
- Nothing else ran on the GPUs. The kernel log was watched for `xe` resets during every run: none.

## Model and configuration

- Coder IQ1_M (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M`, shard hashes checked against the Hugging Face LFS ids), `--layer-split 24`
- Flash-Next IQ2_XS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`), `--layer-split 25`
- Packs from `tools/iq_pack.py`, MTP layer from `tools/mtp_fetch.py` / `mtp_pack.py` / `mtp_rt.py`, profiles from `data/`
- Context 8,192; KV int8; `--spec 4 --spec-min-p 0.5`; greedy (`temperature 0`), `enable_thinking: false`; no vision

`data/coder2.json` and `data/iq2b.json` are the exact server configs (engine arguments and environment); the engine's startup and stage lines are in `data/coder2.engine.txt`, `data/iq2b.engine.txt` and, for the one-card run, `data/B2.run.txt`. Served with
`sycl/serve/server_intel.py --engine strata --config <json> --port 8085`.

## Method

Requests went through `/v1/chat/completions`, one at a time. Prompt and decode speed are the server's own `timings` object
(`prompt_per_second`, `predicted_per_second`), per request; `bench.py` prints them (copy in `data/`, raw output in `data/c2-*.txt`, `data/i2-*.txt`).
Model loading is not included. Three prompts:

- short: "Write a Python function that returns the n-th Fibonacci number" (23 prompt tokens cold, 7 fresh after the cached prefix)
- 2K code: about 2,000 tokens of the repository's `tools/*.py` plus "write a binary search tree class" (1,922 prompt tokens, none reused); the second turn reuses the first
- 2K prose: about 2,100 tokens of source plus "say what this code is for" (2,129 prompt tokens, none reused); the repeat reuses 2,122 of them

## Results

All experts were resident in VRAM on both cards in both runs (engine line `layer split: 100% of the experts resident`), so `STRATA_VERIFY_NO_HOST=1` was set.

| Configuration | Prompt tokens (fresh) | Reused | Generated | Requests | Prompt tok/s | Decode tok/s | Drafts accepted |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Coder IQ1_M, short | 23 / 7 | 0 / 16 | 159 | 3 | 50.4 cold, 74-80 | 52.8 first, 56.6 and 56.6 | 112/138, 114/133 |
| Coder IQ1_M, 2K code | 1,922 | 0 | 256 | 1 | 425.3 | 57.2 | 186/208 |
| Coder IQ1_M, 2K code, next turn | 21 | 2,178 | 256 | 1 | 61.4 | 71.4 | 209/212 |
| Coder IQ1_M, 2K prose | 2,129 | 0 | 128 | 1 | 427.9 | 40.1 | 70/98 |
| Coder IQ1_M, 2K prose, repeat | 7 | 2,122 | 128 | 1 | 78.1 | 41.3 | 70/98 |
| IQ2_XS, short | 23 / 7 | 0 / 16 | 112 | 2 | 50.4 cold, 81.7 | 58.6 and 61.3 | 84/88, 84/89 |
| IQ2_XS, 2K code | 1,922 | 0 | 256 | 1 | 445.1 | 55.5 | 183/213 |
| IQ2_XS, 2K code, next turn | 284 | 1,915 | 256 | 1 | 177.0 | 72.5 | 211/217 |
| IQ2_XS, 2K prose | 2,129 | 0 | 104 | 1 | 436.4 | 38.8 | 56/87 |
| IQ2_XS, 2K prose, repeat | 7 | 2,122 | 104 | 1 | 79.3 | 40.0 | 56/87 |

The two prose rows are slower because fewer drafts were accepted (71% and 64%), not because of the context: the 256-token code reply at the same context ran 57.2 and 55.5.
Time to first token was not measured separately.

One card, Coder IQ1_M, 4,042 of 12,288 experts mirrored in pinned RAM (`STRATA_MIRROR_MIB=8192`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`),
by-hand `--tokens` run, 11-token prompt, 128 greedy tokens: **11.9 tok/s** (11.2 to 11.9 over five builds that differ only in experiments), coherent Python.
Without the ring-wait fix this configuration stopped at layer 15 with `never rang (graph finished)` in 5 of 5 tries.
The engine's stage table shows `wait for rings` at 186.8 ms per round, about 3.9 ms per layer. It drops to 72.9 ms and 25.1 tok/s with `kSpinMax` 2,000 in
`sycl/include/strata/sycl_doorbell.hpp`, which suggests the GPU does not see the host's flag stores during the spin kernel and waits out the bound. That change
risks proceeding before a slow host answers, so it is not part of this report.

Memory after load (`xpu-smi`, MiB, card 0 / card 1): Coder 14,592 / 17,456; IQ2_XS 20,246 / 20,573. The tightest card kept 3.7 GB free of the 24 GB with `--vram-reserve-mib 1536`.

## Correctness and limitations

- Both models, `/v1/chat/completions`: a Fibonacci function executed against the first ten values and `fn(50) == 12586269025`; a Euclid `gcd` executed on four pairs; a palindrome check (IQ2_XS); `sum(i*i for i in range(1,11))` answered `385`. All passed.
- A three-turn conversation (a name and a number, then arithmetic on it, then a lambda using both) was answered correctly by both models, and the conversation cache reused 41 and 74 tokens.
- No NaN, garbage or repetition loop in any output.
- Not done: the 19-token chat-templated prompt from the docs (no token-for-token comparison with the reference text), streaming, the Anthropic route, tool calls, contexts above 2.2K, `--layer-split auto`, a run with `STRATA_VERIFY_NO_HOST=1` and experts outside VRAM.
- Kernel tests (`ctest`, one B60): 21 of 25 pass; `iq_parity` passes for all 10 formats once its fixtures exist; `ple_parity` needs the Q2_0 shard; `s2_expert_grouped_parity` fails 4 bitwise comparisons while its host-reference check is within tolerance; `conversation_snapshot_test` aborts with `munmap_chunk(): invalid pointer`.
- One `xe` GT reset on card 0, during the first `iq_parity` run (it did not finish in 300 s and was killed by a timeout; the same binary then passed in under 90 s). The card recovered without a reboot.

Paths in `data/` were shortened: `<data root>` is the directory with the models, packs and MTP layer, `<strata checkout>` the Strata source tree.
