# Overlap streamed KV uploads with prefill computation

## Why this helps

When long-context KV lives partly in system RAM, prefill uploads each attention
layer's cached prefix. Waiting for every upload on the compute stream leaves
the GPU idle.

This opt-in change starts the next attention layer's upload on a separate stream
while the current layer continues computing. It reuses the existing staging
pool; it does not allocate another KV buffer.

## Use and ordering

Set `STRATA_KV_PREFETCH=1` before starting the engine. It is off by default.
A release event protects the staging pool's last reader; a ready event orders
the next attention consumer. Early returns drain pending copies before the caller
can refill borrowed expert slots. Relayout and destruction also drain the stream.

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

ABBA comparison, two observations per arm, all 20 complete JSON checks passed:

| Cold input | Serial prefill | Overlapped prefill | Serial request | Overlapped request |
| --- | ---: | ---: | ---: | ---: |
| 8K | 5.013 s | 5.025 s | 6.727 s | 6.726 s |
| 32K | 16.989 s | 16.082 s | 18.468 s | 17.559 s |
| 64K | 32.604 s | 31.259 s | 34.354 s | 32.775 s |

Prefill time fell by 5.3% at 32K and 4.1% at 64K.
There was no useful 8K improvement and no decode speedup.
Profiling showed roughly 939 ms / 1,553 ms of serial KV-stage wait moving off
the critical path. The copies still execute; they are not eliminated.

## Validation and remaining work

- Linux compile/link against pure official main (99f3dbd) passed. No engine
  was started and no production settings were changed.
- No claim of a pure-upstream GPU speedup or bitwise numerical parity yet.
- The standalone historical prototype did not run the full vision/120K/cancel
  safety suite. Later combined builds did, but they do not validate this port.
- Keep draft until pure-upstream long-context, cancellation, and image-history
  tests pass. HIP and other GPUs are unvalidated.
