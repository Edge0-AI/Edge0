# A770 decode optimization specification

## Objective and completion criteria

Achieve **at least 20 generated tokens/second in decode on the local Intel Arc
A770 running Strata**, with reproducible benchmark evidence and correct output.
This specification describes the next optimization campaign; it does not claim
that the target has already been met.

Completion requires:

1. Run the same Qwen3.8-Flash-Next Coder IQ1_M pack, tokenizer and MTP artifacts
   used for the existing measurements, on the A770 alone.
2. Preserve the current 131,072-token context capacity and INT8 KV configuration
   for the primary comparison. Label shorter-context experiments separately.
3. Demonstrate >=20 tok/s median engine decode throughput across at least three
   warmed repetitions of the representative coding workload, generating at least
   256 tokens per repetition unless the model reaches a natural stop first.
   Report shorter natural-stop samples separately rather than substituting them
   for the sustained-throughput gate.
4. Count reasoning tokens consistently with the previous engine measurements.
   Report final-answer content and finish reason as well as speed. A repeated
   garbage stream, an artificial repetitive-token workload, or a cached response
   is not a valid success measurement.
5. Pass the correctness gates below, including a final `41` for the original
   arithmetic regression prompt and executed tests for a generated coding answer.
6. Make the selected implementation and run configuration reproducible from the
   recorded source revision, build options and config. Kernel microbenchmarks,
   expected gains, or a single favorable request do not close the decode goal.

## Verified baseline and hardware

- Worktree: a local Strata v0.1.40.3 checkout.
- Current source baseline: `501f675727ae8a54abed3ec467de1cbfbecd03ea`, branch
  `a770-speed-upstream`, based on upstream `main` at `fb58e0d` when last checked.
- Existing upstream work: correctness PR #1602 and draft performance PR #1710.
- Engine binary: `build-sycl/strata`; reports engine 0.1.41.
- GPU: Intel Arc A770, PCI `0000:06:00.0`, device id `56a0`, `xe` driver,
  `ONEAPI_DEVICE_SELECTOR=level_zero:1`. B580 at `level_zero:0` drives the display
  and is outside this single-A770 benchmark.
- CPU: Ryzen 5 5600, six cores/twelve threads, AVX2; approximately 31 GiB RAM.
- A770 is behind the chipset's PCIe 3.0 x4 connection. Engine transfer probes
  measured 1.7-1.8 GB/s. Do not infer the loaded link from the GPU endpoint's
  idle sysfs speed alone; record the upstream bridge and the measured transfer.
- Build/runtime: oneAPI 2026.1 inside `strata-sycl-dev`.

Measured progress, not estimates:

| Configuration | Warm decode | Notes |
| --- | --- | --- |
| Pinned mirror, 4.46 GiB VRAM experts | about 2.5 tok/s | Most missing experts read by GPU over PCIe |
| CPU misses, larger cache and tuned prompt reservation | about 9-11 tok/s | Same context capacity |
| Fast router plus borrowing and paired mapped transfers | 13.5, 11.9, 12.3 tok/s | 546 input tokens, 128 generated, no prefix reuse |
| Same final configuration, short warm prompt | 11.4 tok/s | 73 input tokens, 128 generated |

The final configuration had 4,717 expert slots, approximately 8.98 GiB in VRAM
and 15.08 GiB of resident RAM experts. Existing runs did not test a full 128K
prompt. The fast router's isolated 256-expert/top-10 timing improved from
395.47 to 118.62 microseconds per call, with bitwise matching results.

Artifacts from the earlier campaign live in a temporary directory:
`a770-speed-results.json`, `a770-tuning-round2-results.json`,
`a770-tuning-round3-results.json`, `a770-tuning-round4-results.json`,
`a770-tuning-paired-results.json`, their config files, and engine logs.
They are useful historical evidence, but temporary paths are not durable
deliverables for a completed optimization.

## Scope and constraints

- Optimize the current SYCL engine rather than changing model weights or swapping
  to an unrelated inference engine. Keep the >4 GiB kernel-address build fix.
- Preserve BF16 weight interpretation, FP32 accumulation contracts, routing
  semantics, MTP verification and the payload checksum. Faster FP16 arithmetic
  is a separate numerical experiment, not a silent BF16 replacement.
- Keep an old/reference path selectable for every new dispatch. Default changes
  require parity and served-output validation.
- Tune per kernel and per architecture. Do not globally change subgroup 32 to 8
  or 16 while CUDA-shaped shuffle and reduction code still assumes 32 lanes.
- Avoid executing another repository's build scripts while studying it; port
  selected techniques with their license notices and tests.
- No automatic commits, pushes, PR updates or changes to the user's normal run
  config are implied by this specification. Keep experimental configs separate.
- Record CPU/RAM contention, available RAM, swap activity and GPU clocks before
  each full-model test. Ask before closing newly opened user applications.

## Work package 1: reliable event-based measurements

### Problem

The original base stage profiler wrote zero timestamps. The experimental port
uses a host thread continuously updating a host USM clock word while the GPU
reads it uncached. Review flagged the atomic-host/non-atomic-device clock access
as lacking a correctness-guaranteed synchronization contract. Enabled runs also
reported `layer ... rang but its payload never arrived whole`; the causal link
between these facts is unproven.

### Initial implementation

Add an isolated SYCL event benchmark for the BF16 projection family. Its queue
must share the chosen device/context and have both `in_order` and
`enable_profiling`. Device memory and launches must use that same queue.

Measure a warmed batch using device event timestamps, with native queue marker
kernels delimiting the batch if the public launcher does not return its kernel
event. Wait only after all measured submissions are enqueued. Report this as a
**device queue span**, including any inter-command scheduling gaps; do not label
it a pure per-kernel execution duration. Also report host wall time and the
empty-marker span so that marker overhead and submission starvation are visible.

The initial benchmark must not enable `STRATA_VERIFY_PROFILE`, start a host
clock writer, record a command graph, or query an event while it is still pending.
Require nonzero and monotonically ordered start/end timestamps; otherwise fail
with a useful diagnostic rather than print a zero or wrapped duration.

### Subsequent integration

For eager diagnostic launches, collect each operation's returned `sycl::event`
in a bounded host-side collection and query profiling start/end only after
completion. Label eager timings separately from captured-graph production rates.
Do not assume recording-time node events contain timestamps for each replay of
a command graph. A graph-node profiler needs a supported replay profiling API;
otherwise benchmark constituent kernels separately and validate whole-window
time with normal uninstrumented graph replay.

The unsafe host-clock profiler must remain disabled in every speed acceptance
run. Replacing it must remove the busy writer and retain clear behavior when
profiling is unavailable. Profiling instrumentation must be opt-in and must not
alter the synchronization of the ordinary expert-pool path.

### Output schema

Every benchmark row must include: source revision, device name and PCI identity
when available, queue mode, compiler/backend options, family, implementation,
N/K/T dimensions, input/output leading dimensions, repeat count, warm-up count,
device-span median/minimum, host wall time, empty-marker cost, effective weight
bandwidth, correctness result and output hash. Do not turn absence of data into
a passing or faster result.

## Work package 2: existing few-token BF16 projection paths

### First comparison

Compare the default `bf16_gemv_fp32_mmvf_multi` path with
`STRATA_MMVF_ROWS=1`. The latter is already implemented in
`sycl/src/kernels/cuda/native_bf16.dp.cpp`; its selector is cached on first use.
Run each mode in a separate process, setting the environment before any launch.
Do not change `setenv` between measurements in one process and assume the cached
selector will change.

For each mode, compare against one independent single-token MMVF call per output
column. Check exact output bits where that is the implementation's contract.
Use an FP64 CPU oracle for a bounded subset as a second check; the oracle does
not replace the bitwise old/new comparison.

### Shape matrix

Interpret shapes as `[N output rows, K input columns]`, applied to T activation
vectors. Use T in `{1, 2, 3, 4, 6, 8}` and at least:

| N | K | Purpose |
| ---: | ---: | --- |
| 4 | 10240 | Hyper-connection inject; few output rows |
| 320 | 10240 | Hyper-connection down |
| 10240 | 320 | Hyper-connection up |
| 256 | 2560 | Coder router |
| 2560 | 2560 | Dense projection |
| 10240 | 2560 | Wide dense projection |

Include padded even `ldx` and `ldy`, a non-full output tile (for example N=65),
zero-size/no-op or rejected shapes according to the public API, guard regions,
finite random data, all-zero inputs and scale-extreme finite data. Record which
shapes actually select the row kernel; N<64 currently falls back.

Only move a winner into the engine when a matched-config A/B run shows a useful
improvement. If the row kernel helps a narrow class and regresses another, use a
shape-specific dispatch rather than a global new default.

## Work package 3: targeted register and subgroup tuning

Investigate 128- versus 256-GRF compilation on register-heavy A770 kernels,
preserving the existing backend options including
`-ze-opt-greater-than-4GB-buffer-required` and correctly rounded divide/sqrt.
Confirm the actual emitted compile option and reject unsupported options.

Candidate subgroup sizes are 8/16/32 only where the reduction and work-group
geometry can be adapted. Initial targets: BF16 projection rows, GDN recurrence,
hyper-connection down/up and quantized dense MMVQ. Record register spills or ISA
evidence when toolchain dumps are available. More registers can reduce occupancy;
an accepted flag alone is not evidence of a speed improvement.

Evaluate one variable at a time first. Then evaluate combined winners. Each
matrix must include the two-to-four-token regime used by the actual verifier,
not only large prompt GEMMs. Arcint's A770 large-GRF and subgroup-8 results are
motivation, not transferable performance claims.

## Work package 6: items folded in from Infernix (round 2026-10)

Infernix (`github.com/Wallawalla47/Infernix`, reviewed `main` at the 2026-10
README/RESEARCH_NOTES) is a CUDA/RTX-5090 engine; its kernels (NVFP4/FP8 MMA,
`mma.sync`, `cp.async`, TMA, PDL, L2 prefetch) do not port to Intel Arc. Three
ideas are transferable and go into this round. Its own CPU/PCIe miss-split is
tuned for PCIe Gen5 x8 (~32 GB/s); the A770 is PCIe 3.0 x4 (~1.8 GB/s, ~18x
less), so any split must be re-measured here, not copied.

1. **Hybrid CPU + PCIe expert-miss split (`--pcie-frac`).** Infernix "computes
   part of each layer's cache misses on the CPU while the rest cross PCIe." We
   run `--pcie-frac 0`. Test a small explicit `--pcie-frac` (for example 0.02,
   0.05) in the resident-experts config to overlap the CPU pool with GPU-side
   PCIe fetches. CAVEAT: on `xe` without `--stream-experts` the GPU page-faults
   on pageable host memory and an explicit nonzero value warns it "hung an Arc
   Pro B70 (device lost)". Run it guarded: watch for device loss and revert
   immediately. If the resident complement is not actually pinnable, record this
   as unsafe on `xe` and stop.
2. **Controlled ms/round measurement.** Infernix ranks changes by **ms per
   round** on a **controlled token stream** (identical round/acceptance counts),
   arms **interleaved in one window**, std dev <0.02 ms, and warns that one-side
   probes and single-kernel rankings mislead (our own router microbench 3.3x gave
   no end-to-end gain). Adopt this for every A/B here: fixed seed, greedy,
   compare ms/window and rounds, not just tok/s.
3. **Adaptive draft length.** Infernix drafts "down to none on text it predicts
   poorly", so a bad draft never slows a round. We have `--spec-min-p`; sweep it
   (0.0, 0.3, 0.5, 0.7) with the controlled protocol to confirm the current 0.5
   is right and whether a higher threshold helps prose-like output.

Non-transferable from Infernix (recorded so they are not retried): all quantized
MMA kernels and KV formats (fp8/nvfp4/vq2/k4v2), PDL/TMA/`cp.async` overlap, its
VRAM ledger and Windows budget handling, and its 6-CPU-worker finding (we are at
7; both agree "more is slower because CPU and PCIe share host memory bandwidth").

### WP6 results so far

- **Item 1, `--pcie-frac` — inert on `xe` resident-experts mode (resolved).**
  `--pcie-frac 0.05` was accepted (with the pageable-memory warning) and did not
  fault or lose the device, but the decode timing reported **`PCIe 0.00`**
  experts: the counter never moved, so no GPU-side PCIe expert fetch happened.
  Decode was 12.0 / 11.9 / 12.2 tok/s, level with the 12.4 / 12.6 baseline (and
  that run was RAM-starved: the resident complement fell to 11.12 GiB, fewer
  experts resident). `--pcie-frac` does not wire a PCIe read path in this mode;
  the only PCIe expert path is the pinned mirror (`--stream-experts`), which is
  ~5x slower here because the A770 link is PCIe 3.0 x4 (~1.8 GB/s). The
  Infernix-style hybrid CPU+PCIe split is therefore **not reachable / not useful
  on this card**. Keep `--pcie-frac 0`.
- **Item 3, adaptive draft length** — resolved: keep the default 0.5. Steady
  decode in one window cluster (546-token prompt, 256 generated):
  `--spec-min-p 0.0` = 10.4 / 10.5 / 10.4 tok/s (draft acceptance ~52%);
  `0.5` = 12.2 / 12.0 / 11.7 (acceptance ~68%); `0.7` = 11.8 / 11.4 / 11.6
  (acceptance ~93%). Higher thresholds raise acceptance but offer fewer drafts;
  0.5 is best or tied, 0.0 clearly worse. No change.

Additional rejected lever this round: **`--adapt-swaps 0`** (no adaptive expert
swapping) measured 11.2 / 10.9 / 10.5 tok/s against the 12.0 default (8) in the
same window — the default's swap schedule is better. Keep 8.

Same-window A/B of the overlap lever: **`--spec-split` HURTS** — 9.9 / 10.2 /
10.3 tok/s vs `--no-spec-split` 11.7 / 12.1 / 12.6. So overlapping the drafter
with the verify window is negative on this A770 (opposite the fork's B70 result).
The engine default matches `--no-spec-split`; keep it. This closes the last
cheap "overlap" knob.

Exhausted-lever summary (all measured, same-window where it matters): spec window
width (wider worse), spec-min-p (0.5 best), spec-split (worse), pool-workers
6/7 (wash), cache size (smaller worse), adapt-swaps 0 (worse), pcie-frac (inert),
GRF 128/256 (no-op/worse), expert-lanes 8/16 (wash), mmvf-rows (worse). No config
or on/off switch reaches 20 tok/s or even materially past ~12.6.

Also neutral: **`--kv q4_0`** (halves KV VRAM, should allow more resident experts) —
12.0 / 12.3 / 12.6 tok/s, level with `--kv int8`. So KV VRAM was not the binding
constraint on the expert cache, or freeing it did not reduce the CPU pool enough
to matter.

This exhausts the configuration and switch surface. Every lever tried is null or
worse; the best reliable steady state remains **~12.0–12.6 tok/s**. The only
remaining paths are (a) the architectural change to the CPU-pool handoff, and
(b) the bounded aligned-IQ GPU-kernel port from the llama.cpp SYCL PR (minor by
our own stage table).

### Dual-GPU on the A770 + B580 (peer tier / layer split): BROKEN in the SYCL port

The one lever that could eliminate the CPU handoff is fitting all experts in
VRAM. A770 (16 GB) + B580 (12 GB) = 28 GB vs ~19 GB of experts. Tried all three
SYCL multi-GPU paths:

- **`--peer-device` (second expert tier): stubbed.** `sycl/src/core/peer_experts.cpp`
  refuses: "the second-GPU expert tier is not ported to SYCL yet" (also INTEL.md:639).
- **`--expert-cache-remote`:** disallowed across a layer split (generate.cpp:1970).
- **`--layer-split auto` with `ONEAPI_DEVICE_SELECTOR=level_zero:1,0`: loads, then
  crashes.** It configures *ideally* — "layer split across 2 GPUs", K=2, 5946 of
  12288 profiled pairs in the caches (**~97.5% of the routed mass**), CUDA1 = A770
  runs layers 2-47 with a 10.06 GiB expert cache, and the window runs as the
  **"100% VRAM resident: zero-doorbell graph"** (no CPU handshake — exactly the
  target state). But every request then fails:
  - `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` in
    the prefill path (caught at prefill.cpp:4509), then
  - `Cannot submit to a queue with a dependency from a graph that is associated
    with a different context` (verify.cpp:1706).

The second error is the root: the port's command-graph capture is **single-context**,
and a two-device layer split has **two SYCL contexts**, so the captured window
graph cannot be submitted. The port has no multi-context graph support. So the
dual-GPU path is **not usable**, and making it work is a substantial port (multi-
context graph capture/replay, plus the prefill device-lost).

**Net:** the highest-upside lever is confirmed blocked by an unported feature, not
by a tuning choice. Reaching 20 tok/s now requires either porting SYCL multi-
context graphs (large), or a machine where all ~19 GB of experts fit in one card's
VRAM.

### Aligned/planar IQ kernels (llama.cpp SYCL PR idea): no safe minor win

Inspected the port's IQ decode (`sycl/src/kernels/cuda/iq_kernels.dp.cpp`). The
`Split<>` loaders for the Coder's formats already read **words**, not bytes:

- `Split<22>` (IQ2_S): `get_int_b2` for `qs` and the sign word, `qh`/`scales` as
  one-byte reads only.
- `Split<42>` (Q2_0): `int16_t` reads.
- `Split<17>` (IQ2_XS): `get_int_b2` words.

So the PR's "32-bit instead of byte" change has nothing to bite on — the port
already does it (the base carries the fork's kernel optimizations). The remaining
misalignment needs the PR's **planes repack** (fields laid out back to back per
block), which is a **loader + cache + verification change**, not safe-minor. And
even done, the payoff is small: the expert kernel is ALU-bound (grid lookups +
dp4a) and is only ~20% of the window; the PR's ~13% on one sub-kernel would move
the whole decode by ~2-3%. Not worth the bitwise-regression risk.

**Conclusion after exhausting every lever:** ~12.0-12.6 tok/s is the ceiling for
this model on one A770 with the port as it stands. The only paths past it are the
large SYCL multi-context-graph port (to make the two-card split work) or a single
card with >= 24 GB VRAM.

## Work package 4: shape-specific projection kernels

If existing-path tuning leaves a significant bottleneck, adapt arcint patch
0022's **few-column reuse and split-K geometry** to the SYCL BF16 implementation.

- Read a weight chunk once and update separate accumulators for 2-8 tokens.
- For N small and K long, split K over a work-group, then reduce in a deliberate,
  documented order. For larger N, compare one row and several rows per subgroup.
- Preserve BF16-to-FP32 conversion. Arcint's storage of BF16 as F16 is not the
  target engine's contract and must not be copied blindly.
- Handle strided activation/output rows and non-full tiles without out-of-bounds
  loads or stores. Unsupported geometry must use the verified fallback.
- Keep a negative-control switch and direct old/new comparisons.

Prefer the old operation/reduction order for bitwise parity. If a changed tree
is necessary, state that explicitly, compare with a high-precision oracle, define
tolerances before interpreting measurements, compare logits/routing and run the
served correctness suite. A successful arithmetic answer alone is insufficient
to approve a numerically different path.

## Work package 5: low-bit layouts and XMX crossover

These follow measured BF16/HC work rather than starting a broad backend rewrite.

### Aligned low-bit fields

Study arcint patch 0022's plane layouts for IQ2_XXS, IQ2_S, IQ3_S and Q2_0 and
its four-lane treatment of small Q2_0 rows. Inventory equivalent optimizations
already present in `iq_kernels.dp.cpp` and `native_mmvq.dp.cpp` before adding a
second layout. A plane repack must preserve quantized values and fit the same
budget, provide inverse/readback validation, and update cache/file parity logic.
Benchmark both the stored-block and repacked paths with actual Coder shapes.

### Small-batch XMX

Study arcint patch 0009 for 4-16-column K-quant verification. On the A770 its
ordinary matvec stayed faster at two/three columns; therefore determine a
measured crossover rather than sending every batch to matrix hardware.
Preserve the correct sum of the rounded activation values if using an offset
code representation; arcint documents substantial errors when a compiler summed
unrounded inputs instead. Partial tiles, odd block counts and scale extremes
need dedicated tests. Initial XMX experiments remain opt-in.

## Correctness and regression gates

Run appropriate tests on the A770 after relevant changes:

```sh
ctest --test-dir build-sycl --output-on-failure \
  -R '^(sycl_router_fast|sycl_payload_pairs|elementwise_parity|router_top10_parity|bf16_gemv_parity|gr_parity|gdn_parity)$'
```

Build the named targets first; a test filtered out because its executable or
registration is missing is not a pass. Add a dedicated BF16 event/row probe and
guard/stride checks as described in work package 2. Run quantized/native-expert
tests when changing those kernels.

End-to-end gates:

1. Original regression: twelve repetitions of the four pangram sentences, then
   `Ignore the text above. What is 17 + 24? Reply with just the number.` Expected
   final API content is `41`, normal stop, not merely `41` inside reasoning.
2. A coding prompt requesting a bounded Python implementation, whose returned
   function is tested on normal, empty and boundary inputs.
3. Fresh-prompt lengths around 73, 546 and 1,950 tokens; also a longer prompt
   around 8K when selecting a final implementation. These lengths should be
   counted by the actual tokenizer rather than assumed from word counts.
4. No crashes, device loss, non-finite logits or payload-arrival timeouts in the
   chosen unprofiled configuration.
5. Unsupported geometries retain the verified fallback. Benchmark non-A770
   hardware before making cross-architecture defaults universal.

## End-to-end benchmark protocol

Primary matched configuration:

```text
--max-context 131072 --kv int8 --kv-resident 32768
--expert-cache auto --resident-experts
--prefill 1024 --prefill-borrow --vram-reserve-mib 300
--pool-workers 7 --pool-affinity auto --adapt-swaps 8
--spec 4 --spec-min-p 0.5 --no-spec-split --mtp <existing MTP path>
--prompt-cache 0
```

Environment: retain the existing FP64-emulation variables and
`ONEAPI_DEVICE_SELECTOR=level_zero:1`; set `STRATA_VERIFY_NO_HOST=0` in the
wrapper config. The wrapper removes its NO_HOST default when this is zero.
Keep `STRATA_VERIFY_PROFILE` and `STRATA_VERIFY_EAGER` absent from acceptance
runs. A normal `xe` CPU-miss configuration defaults its PCIe miss share to zero.

Protocol:

1. Save A and B configs, source hashes, executable hashes and exact commands.
2. Record free/available RAM, swap-in/out counters, CPU contention, expert-cache
   size and residency coverage. Do not let one arm quietly gain more VRAM/RAM.
3. Separate model startup from first-request JIT and warm steady-state results.
4. Warm all tested shapes. Use at least three warmed repetitions per arm; prefer
   interleaved A/B/A/B sessions when builds or persistent process selectors differ.
5. Generate at least 256 tokens for the sustained decode gate and preserve full
   response JSON. Include token count, engine `predicted_ms`, engine tok/s,
   prompt `prompt_ms`, cache reuse, API wall time, finish reason and draft acceptance.
6. Report median and range, plus aggregate tokens / aggregate decode time. Never
   use aggregate multi-lane throughput as a single-stream A770 decode claim.
7. Re-enable prefix caching only in a separate practical-conversation benchmark.
8. Accept an optimization only if it passes correctness and wins outside ordinary
   run-to-run noise, or addresses a clearly measured pathological shape without
   degrading the primary workload. Record rejected experiments too.

Portable build commands inside the existing image (run from the worktree):

```sh
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/..:/work" strata-sycl-dev \
  "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; \
   cmake --build /work/strata-v0.1.40.3/build-sycl --target strata -j 8"
```

Use the actual `cmake --build` exit status. `sycl/tools/build.sh` from 0.2.7 on exits with it; older copies
printed `BUILD EXIT 0` but could return nonzero because their last failure-search pipeline found no failures.
Do not chain tests behind the status of an older script.

## Artifacts and implementation sequence

Initial artifacts:

- This specification: `docs/A770_SPEED_SPEC.md`.
- Kernel probe: `sycl/probe/bf16_event_bench.cpp` (planned), with a CMake target
  under `STRATA_SYCL_PARITY`.
- A machine-readable microbenchmark output with the fields from work package 1.
- A persisted benchmark config and response/timing records for selected winners.
- A short result ledger under this document's execution record.

Sequence:

1. Confirm source/resource state and finalize this specification.
2. Implement the isolated event-timed BF16 benchmark and run baseline/row arms.
3. Validate bits, guards and strides; record per-shape crossover results.
4. Benchmark a matched full-model `STRATA_MMVF_ROWS=1` arm if microbenchmarks
   justify it, without the experimental host-clock profiler.
5. Tune registers/subgroups or add a split-K specialization for the measured
   bottleneck. Repeat correctness and end-to-end gates.
6. Consider low-bit layouts or XMX only after the earlier evidence ranks them.
7. Run the mandated independent quality pass on source changes before final
   delivery. Preserve known limitations and do not claim goal completion from
   partial progress.

## Reference implementations

Arcint reference checkout: a local temporary checkout, reviewed revision
`fcb071c6c781c3fb5125e311cbea06209bb9c642`.

- [0022: low-bit formats and few-column F16/BF16 geometry](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0022-opencl-intel-low-bit-iq-and-f16-small-n.patch)
- [0009: few-column K-quant XMX](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0009-opencl-intel-few-column-kquant-xmx.patch)
- [0003: register/subgroup-tuned GatedDeltaNet](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0003-opencl-intel-gated-delta-net.patch)
- [Event-profiling harness](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/tools/native_kernel_harness.py)
- [Measured kernel notes](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/README.md)

Arcint's headline A770 coder numbers use a different model/configuration.
Its Flash-Next results are mainly on a B60. Treat those numbers as evidence for
candidate techniques, not predictions for this A770 workload. Several arcint
ideas already originated in Strata; audit existing equivalents before porting.

## Execution record

- Specification created before starting the new implementation.
- The existing 20 tok/s goal was explicitly resumed for this requested work.
- At start: source `501f675`, `a770-speed-upstream`; no tracked source changes.
- At start: approximately 17 GiB RAM available; no A770 test container running.
- Next active task: isolated profiling-enabled BF16 kernel benchmark and
  default-versus-row comparison. No new performance result is claimed yet.

### Work package 1-2 results (event-timed BF16/F32 MMVF)

Probe `sycl/probe/bf16_event_bench.cpp`, A770 `level_zero:1`, in-order
profiling queue, 32 repeats x 7 samples, device queue span per call. All arms
passed bitwise old/new parity, guard-region, stride, zero-input, scale-extreme,
non-finite and invalid-geometry-rejection checks.

- Default path versus `STRATA_MMVF_ROWS=1`: the rows path is 1.4-6x SLOWER.
  320x10240 at T=3: 16.06 -> 79.00 us. 2560x2560 at T=4: 38.68 -> 245.54 us.
  With the caveat that the small (T=1, n=4) rows are launch/clock noise,
  `STRATA_MMVF_ROWS=1` is rejected as an A770 default.
- Raw baselines (default path, us): 320x10240 T=3 16.06; 2560x2560 T=4 38.68;
  10240x2560 T=1 182.71; 10240x320 T=3 47.56. These are small relative to the
  ~180 ms decode window, so the BF16 projection family is NOT the decode
  bottleneck on this card.
- Artifacts: `a770-bf16-default.jsonl` and
  `a770-bf16-rows.jsonl`.

### Work package 3 results (register allocation)

Added `-DSTRATA_SYCL_BF16_GRF=0|128|256` (default 0), applied only to
`native_bf16.dp.cpp` via a kernel-local property. No subgroup-size change.

- `GRF=0` reproduces the saved baseline (ratio 1.000 on the steady shapes),
  confirming the default path is unchanged.
- `GRF=128` is a no-op (ratio ~1.000 on all steady shapes).
- `GRF=256` is 10-58% SLOWER across the board (for example 320x10240 T=8
  44.86 -> 70.82 us; 2560x2560 T=8 79.14 -> 98.09 us; 10240x2560 T=8
  320.11 -> 425.62 us). Rejected.
- Conclusion: register allocation is not a lever for this family on this card.
  Leave the default (0).

### Where the decode time actually is

Prior per-window timing (`strata decode timing`) at ~13 tok/s: about 180 ms per
window (T~2.2), of which verify ~171 ms = GPU-reach wait ~130 ms + per-layer
host ~35 ms (CPU experts ~33 ms) + staging, plus ~8 ms draft. The **GPU-reach
wait dominates**, i.e. the GPU expert/attention compute for VRAM-resident
experts, not the BF16 dense projections and not the host handshake. The
remaining lever toward 20 tok/s is the quantized expert (IQ) GPU kernels and
the effective token/window count, not the BF16 projection geometry tested here.

### End-to-end confirmation (this revision, unprofiled)

Config: `--resident-experts --prefill-borrow --prefill 1024 --vram-reserve-mib
300 --pool-workers 7 --adapt-swaps 8 --spec 4 --no-spec-split`, 131072 context,
INT8 KV, 32K resident, `--prompt-cache 0`, `STRATA_VERIFY_NO_HOST=0`. 8.98 GiB
VRAM expert cache, 15.08 GiB resident RAM.

- Sustained 546-token prompt, 256 generated, two warm runs: **9.6 and 10.9
  tok/s** (engine `predicted_per_second`); zero prompt tokens reused.
- Decode breakdown (the 135-window run, 174.19 ms/window, T=2.22):
  **verify 164.87 ms = GPU-reach wait 123.27 + per-layer host 36.08
  (plan 0.16 + actq 0.12 + jobs 8.79 + CPU 25.45) + stage 0.01**, commit 0.30,
  draft 8.01. Per layer-window: CPU experts 4.42 (5.32 entries), VRAM hits 16.90,
  PCIe 0.00.
- So the GPU-reach wait is ~71% of the window. The GPU is busy on the
  VRAM-resident expert kernels plus attention/GDN, not waiting on the CPU: the
  CPU pool contributes only ~25 ms of the 174 ms.
- Arithmetic regression prompt returned `41` with normal stop in 3 of 3 runs.
  One earlier cold run returned `41!!!!!!!!!!!` (content) once and did not
  reproduce; keep the sustained arithmetic gate in every acceptance run and
  treat a reproducible degenerate as a release blocker.

**Bottleneck verdict (evidence-based):** the next real lever is the quantized
expert GPU kernel (`native_expert_grouped`, IQ dequant + grouped matmul), and
the tokens-per-window count. BF16 projection geometry (WP2) and register
allocation (WP3) are exhausted as levers. WP4/WP5 should target the IQ expert
kernels directly with the same event-timed, bitwise-gated method.

### Expert-rebalance experiment (rejected)

`ms_wait` is the host waiting for the GPU to ring each layer = GPU compute on
the critical path, not the GPU waiting on the CPU. Hypothesis: move experts off
the GPU onto the underused CPU (CPU pool ~36 ms vs GPU ~123 ms/window) by
shrinking the VRAM cache. Result, warm runs, 546-token prompt, 192 generated:

| VRAM cache | decode | GPU-reach wait | CPU/layer | note |
| --- | --- | --- | --- | --- |
| auto 4717 slots (8.98 GiB) | 10.9 tok/s | 123 ms | 25 ms | reference |
| 3200 slots (7.93 GiB) | 10.8 tok/s | 121 ms | 30 ms | neutral |
| 2400 slots (5.95 GiB) | 9.6 tok/s | 152 ms | 51 ms | worse |

Shrinking the cache did not reduce GPU-reach wait: it grew (the resident RAM
complement spills partly to the model folder and the CPU pool lengthens), so
the GPU is not waiting on expert count in a way this balances. Cache size in the
3200-4717 range is neutral; below that it regresses. Rejected as a lever.

**Updated verdict:** the ~121 ms/window GPU-reach wait is largely independent of
the knobs tested so far (projection geometry, registers, cache size, window
width). Moving toward 20 tok/s requires making the GPU hot path itself faster
(quantized expert kernels, attention/GDN) or raising tokens/window — a kernel
porting effort, not parameter tuning.

### Grouped-expert lane count (`STRATA_EXPERT_LANES`)

Added `-DSTRATA_SYCL_EXPERT_LANES=1|2|4|8|16|32` (default 8), applied to
`iq_kernels.dp.cpp`. The IQ grouped-expert decode kernel uses 8 lanes per row.

- Interleaved A/B on the sustained gate, same session and config:
  **8 lanes = 12.40 / 12.60 tok/s; 16 lanes = 12.1 / 12.6 tok/s.** No
  difference. The earlier 9.6/10.9 "8-lane" numbers were a worse session, not
  the lane count. `STRATA_EXPERT_LANES` is a **null result**; default stays 8.
- This also resets the reliable steady-state reference: the current best config
  is **~12.4-12.6 tok/s** (first run after load is still cold/JIT).
- `iq_multi_parity` and `native_grouped_parity` pass; the Coder's IQ hit path
  (`moe_hit_grouped_s2`) is byte-identical at 8 and 16.

### Pre-existing test failure (not caused by this work)

`s2_expert_grouped_parity` fails its `moe_grouped_s2` and
`moe_group_resident + moe_grouped_s2` cases ("DIFFERS") **at the default
8-lane build with unmodified `iq_kernels.dp.cpp`** (the option value equals the
source's `#ifndef` default, so that binary is identical to a stock build). The
`moe_hit_grouped_s2` cases all pass. This is a pre-existing SYCL port gap in the
S2 grouped/residency path, not a regression from this campaign. Flagged so it is
not mistaken for one; the fork's Intel notes also report 43/44 on the base.

### Profiling blocker (blocks WP4/WP5 targeting)

`STRATA_VERIFY_PROFILE=1` still fails on the current build: a request returns
HTTP 400 with the engine erroring inside the verify window (the same host-clock
stamp path documented in WP1). The base engine's only per-stage profiler is this
one, and it is unreliable, so there is **no trustworthy per-stage breakdown** of
the ~121-123 ms/window GPU time. No external GPU tracer (`unitrace`/`onetrace`/
`ze_tracer`) is installed in the container.

### Expert kernel measured standalone: NOT the bottleneck (WP4/WP5 retarget)

`native_expert_parity` with `NATIVE_BENCH=1` times the GPU
`native_expert_grouped` kernel on real GGUF rows (10 experts x 4 entries = 40
expert computations, host-wall per call). Coder IQ1_M, A770:

| layer | expert types | blob | GPU grouped |
|---|---:|---:|---:|
| 0 | iq3_xxs / iq4_nl | 2.18 MB | 0.328 ms |
| 1 | iq2_s / q2_0 | 1.51 MB | 0.375 ms |
| 2, 3, 24 | iq2_s / iq4_nl | 1.97 MB | 0.351 / 0.361 / 0.383 ms |
| 47 | iq4_xs / iq4_nl | 2.66 MB | 1.205 ms |

So the grouped-expert kernel is ~0.33-0.38 ms/layer for the common types and
~1.2 ms on layer 47. Even the pessimistic 48 x 0.5 ms is **~24 ms/window**, a
small fraction of the ~123 ms GPU-reach wait. **The expert kernel is not the
bottleneck.** WP4/WP5 as written (target the IQ grouped-expert kernel) are
**retargeted**: the GPU time is spread over the other per-layer kernels (GDN
hyper-connection read, conv/recurrence, QSA attention, router, combine), each
small, so there is no single dominant kernel to win 1.6x from. Reaching 20 tok/s
therefore needs an architectural lever (more tokens/window, deeper kernel
fusion, or more layer/round overlap), not one kernel rewrite. Confirm the
per-kernel split with a working profiler before committing to that effort.

Consequence: before an architectural change is chosen, the campaign needs a
**working profiler** — fix the host-clock stamp, add per-kernel timing behind
`STRATA_VERIFY_EAGER` (no graph) on a profiling-enabled queue, or install an
external Intel GPU trace (unitrace / Level Zero timers). The parameter/config
surface (WP1-WP3, WP6) is exhausted with evidence.

### Other standalone kernels measured

- **Hyper-connection read** (`gr_bench`, `fused_gr_read_multi`, Coder shapes):
  180.4 / 179.0 / 203.7 / 218.0 / 241.7 / 259.8 us for 1..6 tokens (73 GB/s at
  1 token down to 51 GB/s at 6). At ~36 GDN layers x ~180 us that is
  **~7 ms/window** — also a minor slice of the 123 ms. Not the bottleneck.
- So the two kernels biggest in the old (inflated) stage table — experts and
  hyper-connection read — together account for only ~30 ms of ~123 ms.

### The window is nearly GPU + CPU serial (architectural hypothesis)

Steady-state numbers (546-token prompt, ~2.2 tokens/window): window ~166 ms =
**GPU-reach wait 123 ms + per-layer host/CPU 36 ms + draft 8 ms**, and
123 + 36 + 8 = 167 is the whole window. That is the signature of the GPU and
the CPU expert pool running **largely serially within a window**, not overlapped.
If the CPU pool (36 ms) hid under the GPU (123 ms), the window would fall to
~123-131 ms, i.e. ~2.2/0.125 = **~17-18 tok/s**, near the goal. So the most
promising lever may be **overlapping the CPU expert computation with the GPU**
(a scheduling/architecture change), not any single kernel. This is a hypothesis,
not yet proven — the profiler is needed to confirm the GPU is busy (not idle
waiting on the flag) during the CPU pool. WP7 candidate pending that
confirmation; do not start it blind.

Follow-up test (this round): `--pool-workers` 6 (physical cores) vs 7 default
measured 12.1 / 12.8 / 13.0 vs 13.3 / 12.8 / 12.4 tok/s — a wash. So if the CPU
pool is serial with the GPU, it is **not** worker-count-limited; adding/removing
a worker does nothing. Rejected as a lever.

### GPU utilization: the GPU is mostly IDLE during decode (major finding)

`gputop` (the Xe counterpart to intel_gpu_top) CAN be read for the engine if the
engine process is user-owned. `sycl/serve/strata-sycl.sh` runs the container as
root (no `--user`), so `gputop` cannot attribute it; a temporary copy adding
`--user <uid>:<gid>` made the engine user-owned and readable. During a 700-token
decode (240-token prompt) the A770's compute engine (ccs) sampled:

    0%  0%  4.1%  14.8%  41.4%  2.8%  0%  8.5%  65.0%  84.3% ...

— i.e. **mostly 0-20%, spiking occasionally**, mean over the run ~20%. The GPU is
**not saturated during decode**. (1 Hz sampling and ccs attribution make this
coarse, but the many 0% samples during an active generation are unambiguous.)

This **overturns the "ms_wait = GPU compute" reading** in the hypothesis above.
`ms_wait` is the host waiting for the GPU's layer ring, but the ring is late
because the GPU is **stalled on the CPU/handoff**, not because it is busy. So the
decode window is dominated by **per-layer host<->GPU synchronization and the CPU
expert pool, not GPU kernels** — consistent with every kernel measured standalone
being minor. The path to 20 tok/s is therefore to **cut the per-layer handoff
latency / overlap the CPU pool with the GPU**, not to optimize kernels.

Next empirical step (cheap, needs no profiler): quantify the handoff cost by
measuring round-trip latency of the doorbell (ring -> host -> flag -> GPU) on this
`xe`/PCIe-3.0-x4 card and how much of the window is idle GPU time, then decide
the architectural change. This is now the primary direction; WP4/WP5 (kernel
optimization) are closed as mis-targeted.

### Both in-engine profilers are broken (blocks all targeting)

`STRATA_VERIFY_PROFILE=1` and `STRATA_VERIFY_EAGER=1` both fail on the current
build with the engine erroring in the verify window: **`verify: layer 0 rang but
its payload never arrived whole`** (HTTP 400), reproducibly (3/3 eager requests
failed). The PROFILE mode succeeded exactly once, much earlier — so it is a race,
not a hard logic error, but it is reliable enough in EAGER mode to make that path
unusable. There is no env switch to bypass the payload wait
(`doorbell_wait_payload` has none).

Consequence: the engine has **no usable profiler** on this build. The likely
defect (a real bug, relevant to PR #1602's payload checksum) is that the EAGER
path does not publish the doorbell payload that `doorbell_payload_ready` waits
for, so the host spins to the 2 s timeout; in the graph path the publish runs and
the checksum matches. **Fixing this eager-path bug is the concrete next task**,
because it unblocks the per-stage breakdown everything else needs. (It is a
debug-only path; normal graph-mode inference is unaffected.)

### Per-stage breakdown obtained (decisive): the CPU pool handoff dominates

Added an opt-in `STRATA_DOORBELL_NOCHECK` that makes `doorbell_wait_payload`
trust the ring (debug only). With `STRATA_VERIFY_EAGER=1 STRATA_VERIFY_PROFILE=1
STRATA_DOORBELL_NOCHECK=1` the eager stage table finally printed. T=2 window,
ms/window (host-clocked, so absolute values are inflated by the stamp overhead;
the **relative** split is the point):

```
GDN layers: waitA 393.2  waitB 407.8  copy+combine 399.2  hc-read1+router 51.4
            VRAM hits 26.3  hc-read0 25.3  out-proj 17.2  shared+quant 13.7
            z 9.6  q8+qkv 7.3  rec 6.4  conv/ab/head ~5
QSA layers: waitA 134.3  waitB 134.2  copy+combine 132.4  kv+idx append 15.4
            hc-read1+router 13.0  hc-read0 8.2  out-proj 6.5  attention 3.3
```

`waitA`/`waitB` are the GPU **spinning for the CPU expert-pool flags**;
`copy+combine` is the CPU results copied to the GPU and combined. Together they
are **~87% of the window**; every actual GPU kernel is small (tens of ms). This
independently confirms the gputop ~20%-busy finding.

**Definitive conclusion:** A770 decode here is limited by the **CPU expert pool
and its per-layer host<->GPU handoff**, not by GPU compute. WP4/WP5 (kernel
optimization) are closed. The path to 20 tok/s is architectural: cut the
per-layer handoff latency and/or overlap the CPU pool with GPU work. (Note the
absolute stage numbers are inflated and `NOCHECK` distorts the pipeline, so the
next step is an *honest* profiler — fix the eager payload path — to size the real
handoff cost before designing the change.)
