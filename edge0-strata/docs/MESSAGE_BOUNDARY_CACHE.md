# Reuse long chat history when the last user message is edited

## Why this helps

An agent can keep the same long history but replace the final user message.
The final cached state no longer matches, and the nearest periodic checkpoint
may be far behind. The engine then rereads history it already processed.

This opt-in change saves one checkpoint at the previous turn boundary, before
the final short message. It uses the existing bounded cache and leaves normal
token and image-prefix validation in charge of reuse.

## Use and limits

Set `STRATA_CACHE_MESSAGE_BOUNDARY=1` before starting the engine.
It is off by default, requires prompt caching and a valid turn token, and
only selects prefixes of at least 8,192 tokens with a tail of at most 1,024.
Layer-split execution is excluded (`multi_gpu`); helper-expert GPUs still work.
It can split prefill chunks and add snapshot cost to a fresh request.

## Validation environment

- Two modified RTX 3080 20 GB cards; CUDA SM86, CUDA 13.1, Linux.
- Intel Xeon E5-2686 v4 (18 cores / 36 threads), about 94 GiB usable RAM.
- PCIe 3.0, no GPU peer-to-peer access; 17 expert-pool workers.
- Qwen3.8-Flash-Next IQ3_S, 131,072-token context, INT8 KV,
  32,768 resident KV cells.
- Medium thinking: 2,048-token reasoning cap, 8,192-token output cap.
- Physical GPU power limits stayed at 250 W and 280 W. No power increase.
- Historical performance tests used an isolated engine based on official
  v0.1.38 plus reviewed architectds/Strata changes (best, 05c0f36).
  This PR contains only our change, ported to official main (99f3dbd).
  The historical timings are NOT a measured speedup of this pure-upstream port.

## Historical benefit

Same-binary ABCCBA comparison, two repetitions per variant:

| Metric | On-demand boundary + 16K chunks | Anticipatory boundary + 32K chunks |
| --- | ---: | ---: |
| First complete 100-record edit | 17.236 s | 11.527 s |
| Three complete edits | 41.266 s | 35.802 s |
| Cold setup plus three edits | 58.688 s | 53.146 s |
| Eleven-case wall-time sum | 133.844 s | 127.983 s |

First-edit time fell by 33.1%; reused prefix grew from 22,016 to 32,831 tokens.
This is a COMBINED boundary-policy and chunk-size comparison, not the isolated
benefit of adding this checkpoint to pristine upstream.
Hot generation was unchanged (97.55 vs 97.60 tokens/s).
The mixed-suite reduction was 4.4%; ordinary cold 8K requests cost about 0.2 s more.

The historical comparison passed 77 task/cache checks and cancellation/recovery.
A separate safety arm for the adopted 32K choice passed 22 task/cache checks,
including 120K, image-history isolation, and cancellation/recovery.

## Validation and related work

- Standalone selector test passes all 21 cases, without assertions that disappear
  in Release builds.
- Linux compile/link against pure official main (99f3dbd) passed. No engine
  was started and no production settings were changed.
- PR #614 checkpoints an already-reached chunk near the tail without splitting it.
  This proposal chooses a message boundary, which can split chunks. They solve
  related cases with different cost and numerical tradeoffs.
- Chunk geometry can change floating-point results. No pure-upstream bitwise
  parity or isolated upstream speedup is claimed. Keep draft until validated.
