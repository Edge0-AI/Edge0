# Avoid duplicate expert slots across helper and primary GPUs

## Why this helps

On a PC with two small GPUs, helper experts start out separate from the primary
cache. Later, the adaptive primary cache can promote an expert that the helper
already holds. Both cards then spend memory on the same expert.

We observed about 1,900-2,000 duplicate slots during sustained requests.
This opt-in change keeps helper-owned experts out of primary promotion candidates,
in both the server and command-line generation paths.

## Use

Set `STRATA_DISJOINT_ADAPT=1` before starting the engine. It is off by default.
It leaves the existing peer-tier check and expert arithmetic unchanged.
The check is live: each promotion candidate is asked of the helper's own cache (`RemoteExperts::holds`), so an expert
the helper's tier swaps in or out later is followed (a mask copied at load would go stale). Needs the helper-share
change (#854) for `holds`.

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

Same binary and helper capacity, six complete JSON tasks per arm:

| 32K cached request | Original | Ownership guard |
| --- | ---: | ---: |
| Generation | 81.5 tokens/s | 94.1 tokens/s |
| Complete request | 12.997 s | 11.417 s |

Generation improved by 15.5%; request time fell by 12.2%.
The tasks returned complete 100-record arrays and correct source facts.
All 18 checks passed across the original, guarded, and larger-cache arms.
The larger-helper arm is not included in the comparison above.
These repetitive tasks do not represent all agent workloads.

## Validation and remaining work

- Standalone ownership test passes, including disabled, empty, unowned,
  owned, and out-of-range entries, plus candidate filtering.
- Linux compile/link against pure official main (99f3dbd) passed. No engine
  was started and no production settings were changed.
- Pure-upstream GPU performance and session/vision/cancellation regression
  checks still need to be run. Keep this PR as a draft until then.
- Historical combined builds passed those safety checks, but that is not a
  substitute for validating this independent port.
