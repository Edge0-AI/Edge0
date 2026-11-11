# Community benchmark: RTX 4080 SUPER (32 GB), Core i7-13790F

Measured on 2026-10-04 by [1314521gjy](https://github.com/1314521gjy), on one
Windows 11 machine. This exercises Strata **0.1.39** with the Flash-Next
**IQ3_S** pack, one GPU, INT8 KV, `--max-context 524288`, and vision enabled.

The report covers three knobs and one cost measurement:

1. **Expert-pool worker count.** The engine's own default on this CPU topology is
   **15** workers (#642's `max(1, p-1+e//2)` applied to 8P+16E). Here, **4 workers
   are 28.7% faster than 15** in a direct four-arm ABBA on 0.1.39, the two
   pool-4 arms agree to 0.21%, and 8 of 8 workload-by-pair cells favour 4. In an
   earlier 0.1.36 sweep, the pool's own per-round cost rose monotonically from
   6.38 to 17.87 ms/round between 4 and 15 workers.
2. **`STRATA_PF_FUSED=1`** (fused int8 prompt kernels, #136) with **IQ3_S**
   experts: cold prefill **+8.9%**, decode unchanged within noise.
3. **The context tier itself, at a fixed prompt.** 524,288 tokens costs +0.9…+6.3% on a
   43,969-token fresh read over 262,144. 1,048,576 costs almost nothing **if the expert
   cache fits**: with an over-sized explicit `--expert-cache`, the same tier ran 7x slower,
   because on Windows an over-committed card pages instead of failing (see the section on
   that below — the mechanism was suggested by the maintainer in #781).

## Hardware and software

- NVIDIA GeForce RTX 4080 SUPER reporting **32,760 MiB** of VRAM (a 32 GB card;
  the retail 4080 SUPER is 16 GB). Driver 616.56. PCIe: Gen 4, x16. The engine's
  startup transfer probe reported **24.0 GB/s host-to-device**. GPU clocks were
  not fixed; the power limit is 320 W.
- Intel Core i7-13790F: 8 performance cores, 16 efficiency cores, 24 logical
  processors. The engine selected **AVX2** (no AVX-512 on this part). Unpinned,
  the engine's own startup line reports **15** expert-pool workers on this machine
  (observed on an arm configured with `--pool-workers 0`, i.e. all cores, on
  0.1.36), which is also what #642's formula gives for 8P+16E.
- 95.78 GiB installed RAM. Weights and packs are on an ORICO 1.9 TB NVMe SSD.
- Windows 11 Pro. Engine 0.1.39, `source: release`, CUDA 13.0, archs
  `[75, 86, 89, 120]`, portable build; see [BUILD.json](BUILD.json).
- The GPU was dedicated to Strata and its vision helper during these runs. Other
  desktop and background services stayed running; this was not an isolated
  operating system. The user's own interactive session was not active during the
  measured runs.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ3_S/`:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` — 54,817,524,224 bytes
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` — 28,800,138,432 bytes
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf` — 907,543,008 bytes

These were downloaded from a mirror of `main`, not from the Hub directly, and
**no per-file SHA-256 or revision pin was recorded for the three GGUF files**;
only their byte sizes are known. MTP was installed by the setup from
`Qwen/Qwen3.8-Flash-Next`; its manifest is [mtp-manifest.json](mtp-manifest.json).
This is a weaker provenance statement than the RTX 5090 report's, and is stated
as such.

The pack built by the installer is the native IQ3_S pack; its identity is in
[artifact-hashes.json](artifact-hashes.json) (`dense.bin`, `index.txt`,
`native_experts.txt`, tokenizer, the two executables).

Cross-check against the RTX 5090 report already in this repository: the pack's
`dense.bin`, `vocab.json`, `token_type.json` and `chat_template.jinja`, plus the
MTP `dense.bin`, `draft_vocab.bin` and `experts.bin`, are **byte-identical** to
the corresponding files that report lists (same SHA-256), so the shared
artifacts are the same installs. The per-quantization files (`index.txt`,
`native_experts.txt`) and `merges.txt`, `tokenizer.json`, the MTP `dense.txt`
differ.

Launch configuration, [strata-iq3_s.json](strata-iq3_s.json):

- `--max-context 524288` with `--rope-scaling yarn --rope-scale 2
  --yarn-orig-ctx 262144` (the model's trained context is 262,144).
- `--kv int8`, `--kv-resident 32768`, `--pcie-frac 0.35`, `--vram-reserve-mib 700`.
- `--expert-cache 8900` — the engine reports **11,631 slots / 22.07 GiB**, i.e.
  the profile admits slightly more experts than requested. The bundled expert
  profile was used; no calibration was run.
- `--prefill auto` selected chunks of **8,192** tokens with a 512-slot ring and
  borrowed 2,232 cache slots (4.24 GiB) for the ruler prompts; on the
  43,969-token probe it chose **6,912**-token chunks and 2,056 slots (3.90 GiB).
- `--spec 4 --spec-min-p 0.5`; built-in suffix drafting stayed enabled.
- `--pool-workers 4`; **the engine default would be 15 on this machine.**
- Environment: `STRATA_DEC_BATCH=1`, `STRATA_PF_FUSED=1`.
- `--vision` on, 1,024 image tokens; no benchmark request contained an image.
- Thinking was left at the server default (enabled) for every measurement below.
  No control vectors, no calibration, no experimental speed projection.

Everything above is the configuration attached here. **At the end of the session it moved
to `--max-context 1048576` (yarn factor 4) with `--expert-cache auto
--vram-reserve-mib 1500`** — the configuration in the cache-budget table below — after the
1M tier was found to be limited by the explicit cache budget rather than by the context
window. Both are in the attached config's git history.

The engine's startup banner for this configuration is
[engine-banner.log](engine-banner.log); it prints
`rope scaling yarn, factor 2 (freq_scale 0.5, base 1e+07, mscale 1.069315),
--max-context 524288 against a trained context of 262144`,
`KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB
of pinned RAM`, and `expert cache 11631 slots, 22.07 GiB`.

Two startup observations worth reporting. The engine warns that this
configuration leaves only **226 MiB of VRAM free** and suggests
`--vram-reserve-mib 986` rather than the requested 700 — at IQ3_S the expert
cache fills the card even with a 32 GB GPU. And the expert arena, 50.3 GB, is
allocated on **4 KB pages** (`large pages refused ... VirtualAlloc error 1314`);
the arena is read unbuffered and the model loads in 46.84 GiB at 2.90 GiB/s.
The `pcie_frac 0.55` in the transfer-probe line is the probe's own default; the
engine's self-report shows the configured `pcie_frac: 0.35`.

## Method and reproduction

Everything was driven over the OpenAI-compatible endpoint at
`http://127.0.0.1:8080/v1/chat/completions` on loopback, on the same loaded
engine, in increasing-length order, with the engine restarted between
*configurations* but not between requests.

Two instruments were used.

**Four-workload ruler.** Four real corpora (a coding task, Chinese prose, English
prose, and engine documentation), rendered to chat prompts of 6,845 / 6,867 /
5,121 / 4,170 tokens with a different five-token nonce near the start of each
request, generating 1,000 tokens greedily. Each workload is requested four times;
run 1 is dropped as a warm-up and the median of runs 2–4 is reported. Cold
prefill throughput is taken from the dropped run 1, which is the only request in
the group that reads its prompt from scratch — the later runs reuse all but ~5
tokens. Because the nonce differs, the generated text differs between runs and
greedy output hashes are not expected to match across runs; the acceptance rate
and cache hit rate move with it, which is the main source of the few-percent
run-to-run spread seen below. Configurations are compared in an **ABBA** order
(`A1 B1 B2 A2`) so that any drift over the batch shows up as a disagreement
between the two same-configuration arms.

**Single-read probes.** A fresh server process per arm, one 43,969-token cold
read (`--cold`, no warm-up), 200 generated tokens. These are n=1 per arm, so they
are used only for effects that are too large to be noise; every number from this
instrument below says so explicitly.

Throughput is quoted exactly as the engine reports it in its own protocol lines
(`prompt ... read in N ms (X tok/s)`, `Y generated in M ms (Z tok/s)`). Wall-clock
totals over HTTP are given separately where useful. Reused prompt tokens are
recorded; no speed number in this report is taken from a request that reused
tokens.

## Results

### Expert-pool workers: 4 against the engine's 15, on 0.1.39

Four-workload ruler, ABBA order (`A1 pool 4 · B1 pool 15 · B2 pool 15 · A2 pool 4`),
a fresh engine process per arm, 3 kept runs per workload, median per workload.
This is the current configuration (524,288 context, `--pcie-frac 0.35`), on the
current engine, and it is the direct 4-against-15 comparison that #642's
recommendation needs.

| Arm (run order) | Pool workers | work | prose-zh | prose-en | code-engine | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | **4** | 97.1 | 95.4 | 93.8 | 93.5 | **94.95** |
| B1 | 15 | 81.1 | 81.6 | 57.0 | 58.1 | 69.45 |
| B2 | 15 | 67.1 | 83.4 | 86.1 | 76.3 | 78.23 |
| A2 | **4** | 94.5 | 95.5 | 92.2 | 98.4 | **95.15** |

Each cell is a median of three runs of 1,000 generated tokens; the prompts were
6,845 / 6,867 / 5,121 / 4,170 tokens, and the cold prefill in each group came from
the discarded warm-up run.

- Adjacent pairs, both the same sign: A1→B1 **−26.9%**, B2→A2 **+21.6%** (pool 15
  is slower in both).
- Four-arm means: **95.05 (pool 4) vs 73.84 (pool 15) ⇒ +28.7%** for 4 workers.
- **8 of 8 workload-by-pair cells favour 4** (work +19.7/+40.8, Chinese prose
  +16.9/+14.5, English prose +64.6/+7.1, code +60.9/+28.9).
- Same-configuration spread: **pool 4 = 0.21%** (94.95 vs 95.15) but **pool 15 =
  12.6%** (69.45 vs 78.23). The default is not only slower here, it is much less
  reproducible, and two of its four workload medians collapsed to 57–58 tok/s.
- Cold prefill: 2,535 (pool 4) vs 2,486 (pool 15) tok/s — inside the per-arm
  spread (5.0% and 11.4%), so prefill is **unchanged**; this is a decode-path
  effect.

### Expert-pool workers: the worker sweep behind it (engine 0.1.36)

Engine 0.1.36, `corpus-long44k` (43,918-token prompt), 256 generated, single-arm
screens, one process per arm. That batch ran the configuration of the time
(`--pcie-frac 0.55`, `--spec 4`, expert cache 11,631 slots); only the pool-worker
count differs between rows. The same batch also ran `--pool-workers 4` and `8`
twice on a 6,845-token prompt (200 generated): 91.60 vs 90.76 tok/s, i.e. neutral
within that instrument's ~8% noise, while the pool counters moved the same way as
in the long table.

| Pool workers | Decode tok/s | Pool ms/round | Wait-for-rings ms/round | Cold prefill tok/s |
| ---: | ---: | ---: | ---: | ---: |
| **4** | **98.33** | **6.382** | **14.228** | 3,098.3 |
| 6 | 90.56 | 7.787 | 16.111 | 3,057.6 |
| 8 | 89.18 | 8.845 | 15.705 | 3,078.2 |
| 10 | 84.88 | 9.962 | 16.459 | 3,099.1 |
| 12 | 87.89 | 9.437 | 16.800 | 2,938.6 |
| **15** (engine default here) | **56.11** | **17.868** | **27.513** | 3,091.1 |

The pool's own cost is monotone in worker count: 6.38 → 17.87 ms/round from 4 to
15 workers (2.8x), and the ring wait moves with it (14.23 → 27.51). Cold prefill
is flat across all six arms, so this is a decode-path effect only.

Two ABBA confirmations on engine 0.1.36, both comparing 4 against 8 (the
configuration that was adopted at the time):

| Instrument | Prompt | A (pool 4) | B (pool 8) | Pair 1 | Pair 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-workload ruler, median of 3 kept runs | 6,845 | 102.40 / 101.80 | 89.70 / 90.65 | **+14.16%** | **+12.30%** |
| Serve ABBA, median of 3 kept runs | 43,969 | 101.05 / 100.35 | 87.95 / 95.40 | **+14.89%** | **+5.19%** |

On the ruler, **8 of 8 workload-by-pair cells were positive** (work +7.7/+10.8,
Chinese prose +16.3/+11.9, English prose +17.8/+16.8, code +14.9/+9.8) and the
same-configuration spread was −0.59% / +1.06%, so a +13% effect is resolved well
above the floor. On the long prompt the two pairs disagreed in size (+14.89% vs
+5.19%) and the pool-8 arms themselves spread by 8.47%; read that one as a
consistent weak positive, not as a precise 10%.

**Outputs are bit-identical.** On a deterministic recipe (0.1.36), changing only
`--pool-workers 8 → 4` produced byte-identical output hashes on all four
workloads across three runs each, so this is a scheduling change with no quality
cost. On the ruler above, run-to-run output hashes differ because each run carries
a different nonce; that is by construction, not a determinism failure.

### `STRATA_PF_FUSED=1` with IQ3_S experts

Engine 0.1.38, four-workload ruler, ABBA, cold prefill from the dropped warm-up
run of each workload, 5 arms (one discarded warm-up arm first).

| Arm | PF | work | prose-zh | prose-en | code-engine | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm-up (dropped) | 1 | 2,161.8 | 2,226.3 | 2,333.8 | 2,206.4 | 2,232.1 |
| A1 | 1 | 2,157.4 | 2,198.3 | 2,294.1 | 2,257.3 | 2,226.8 |
| B1 | 0 | 1,982.7 | 2,005.7 | 2,097.2 | 1,934.5 | 2,005.0 |
| B2 | 0 | 1,958.4 | 2,050.8 | 2,053.9 | 2,035.6 | 2,024.7 |
| A2 | 1 | 2,159.5 | 2,317.9 | 2,084.9 | 2,092.0 | 2,163.6 |

`STRATA_PF_FUSED=1` = **+8.9%** cold prefill (2,195.2 vs 2,014.9), pairs
**+11.1%** and **+6.9%**, and **8 of 8** workload-by-pair cells positive.
Same-configuration spread was 2.9% (PF=1) and 1.0% (PF=0). Decode was not
distinguishable (PF=1 93.0 vs PF=0 95.8, with 6.6% spread on the PF=1 side).
An earlier measurement of the same switch on an older engine gave +12.8%; this
re-measurement revises the size to +8.9%, which is the same order as the
`+8% / 0%` the source comments record for the native IQ2_XS pack.

The engine's banner names the path per run, which is the product-side check that
the switch took effect: `strata: prompt experts on the fused int8 kernels
(STRATA_PF_FUSED=1, #136)` appears in the fused arms and not in the others.

### Context tier at a fixed prompt

Single-read probes: one fresh process per arm, one 43,969-token read, 200
generated. **n=1 per arm**, so the tiers are separated only where the gaps are
much larger than the run-to-run spread this instrument shows at a fixed config
(which is 0–8% on the same prompt).

Engine **0.1.39** (current configuration as the base):

| Arm | `--max-context` | rope | Prompt read (ms) | Prompt tok/s | Decode tok/s | Total wall (s) | KV in pinned RAM |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | 262,144 | none | 13,074 | 3,363.2 | 91.3 | 15.44 | 3.09 GiB |
| B | 524,288 | yarn 2, ring off | 13,186 | 3,334.5 | 85.3 | 15.71 | 6.19 GiB |
| C | **524,288** | yarn 2, ring on (default) | 13,896 | 3,164.1 | 90.8 | 16.27 | 6.19 GiB |

So 524,288 costs **+0.9…+6.3%** on the fresh read and **−0.5…−6.6%** on decode
against 262,144 at the same prompt, and doubles pinned KV (3.09 → 6.19 GiB of
95.78 GiB). `STRATA_RING_BYTES=0` (#583) was not faster than the default here.

Engine **0.1.38**, same instrument and prompt, for the tier above:

| Arm | `--max-context` | rope | Prompt read (ms) | Decode tok/s | KV in pinned RAM |
| --- | ---: | --- | ---: | ---: | ---: |
| live | 262,144 | none | 2,581 | 103.9 | 3.09 GiB |
| 512K | 524,288 | yarn 2 | 2,655 | 94.4 | 6.19 GiB |
| 1M | 1,048,576 | yarn 4 | 4,065 | 46.4 | 12.38 GiB |
| 1M | 1,048,576 | none | 5,303 | 39.3 | 12.38 GiB |

The expert cache reported **11,631 slots / 22.07 GiB at all four settings** — the
context tier costs host RAM and time, not VRAM capacity. (The cache is identical, but the
1M session needs about 1 GiB more VRAM, and the 524,288 configuration had only 299 MiB of
headroom to give it — which is what the next section is about.) YaRN was not the cause
of the 1M cost: at 1M, YaRN was faster than no scaling at both sample sizes
(46.4 vs 39.3 tok/s; 44.2 vs 41.4 on a shorter 64-token sample). One 0.1.39 run
on the 524,288 configuration read its prompt in 22,989 ms instead of ~13,000 ms
with every counter in its normal range; it is the first run after a fresh model
load and we treat it as a cold-start outlier, not as a result.

### Where the 1M cost goes: the expert-cache budget, not the context tier

This is the part that changed our conclusion. The mechanism was suggested by
[enkynakamura in #781](https://github.com/Niko1221/Strata/issues/781) after reading the
first version of this report, and the numbers below are ours.

The engine sizes the expert cache in two different ways. With `--expert-cache auto` it
sizes from the free VRAM it reads, prefills the slots, and then **checks again after they
are written**, shrinking if they do not fit. With an explicit `--expert-cache N` the
budget is `N x the largest blob`, compared only against the free figure read **before**
the slots are written, and that second check is auto-only. Under WDDM an allocation is not
resident until it is touched and the free figure read before can be high, so an
over-sized explicit cache is committed and the driver pages to system memory instead of
failing — with nothing in the log.

Measured on 0.1.39, IQ3_S, one card, fresh 43,969 / 45,670 / 47,956-token prompts with 200
generated each, one engine start per row, at `--max-context 1048576`:

| cache | slots | free VRAM at READY (engine's own line) | decode tok/s |
| --- | ---: | --- | --- |
| explicit 8900 | 11,631 | **0 MiB** (`LOW`, suggests `--vram-reserve-mib 1212`) | **13.7 / 14.7** |
| `auto` | 11,178 | 217 MiB (`LOW`) | 102.1 / 122.3 |
| `auto` + `--vram-reserve-mib 1500` | 10,766 | 1,075 MiB | 88.9 / 96.5 / 101.4 |
| explicit 7000 | 9,148 | 4,130 MiB | 94.1 / 97.3 |
| *the same 11,631-slot cache at 524,288, for reference* | 11,631 | 299 MiB | 113.2 / 107.9 |

**453 slots — 0.86 GiB — is the difference between 13.7 and 102 tok/s**, and the slower
arm had the **higher** cache hit rate (91.4 / 94.7% against 90.4 / 93.9%), so this is not
misses: being at 0 MiB free is what costs 7x. With a cache that fits, **1,048,576 costs
almost nothing** — read ~13.4 s against 12.6 s at 524,288, decode ~95 against ~110.
`--vram-reserve-mib` is the direct lever, because the engine deducts it *before* sizing:
700 → 1000 → 1500 took the free figure from 217 to 575 to 1,075 MiB and cost only about
400 slots.

The per-window profiler table was taken at a fixed expert cache on 0.1.38, so it describes
a 1M run **with the over-sized cache**: it is the shape of the 7x, not a cost of
`--max-context` itself.

| Per window | 262,144 | 524,288 | 1,048,576 |
| --- | ---: | ---: | ---: |
| ms/window | 38.47 | 35.30 | **69.15** |
| GPU-reach wait | 12.94 | 12.33 | **41.52** |
| GDN `VRAM hits` | 3.61 | 3.58 | **25.35** |
| QSA `VRAM hits` | 1.23 | 1.23 | **10.27** |
| GDN `PCIe grp` | 0.28 | 0.28 | **1.81** |
| q8 gemv / attention / out-proj | 1.26 / 0.74 / 0.93 | 1.27 / 0.73 / 0.89 | 1.37 / 0.78 / 1.00 |
| Work: experts per layer, VRAM hits per window | 2.51, 33.24 | 2.58, 34.22 | 2.54, 33.74 |

The work counters are the same at all three tiers and the pure-compute stages move about
10%, while the stages that reach VRAM cost 7–8x more and the host's wait for the GPU grows
3.4x. That is what an over-committed card looks like from the inside.

**One negative result, reported as such.** The Windows counters suggested in #781 —
`\GPU Process Memory(*)\Shared Usage` and `Dedicated Usage`, sampled once a second for
`strata.exe` through all of the above — **do not separate the slow arm from the fast ones
on this machine**. `Shared` reads ~55.5 GB at 262/524K and ~62.4 GB at 1M in *every* arm;
the +6.9 GB is our pinned K/V going from 6.19 to 12.38 GiB. The arm that ran at 13.7 tok/s
read 62,931 MB, against 62,409 MB for the arm at 102–122 tok/s. On this box the counter
tracks the deliberately pinned memory (the 50.3 GB expert arena plus the pinned K/V), so
it does not expose paging, and `Dedicated` does not separate them either (30,425 against
30,136 MB). The Control Panel "Prefer No Sysmem Fallback" check has not been run.

**Corrections to the earlier version of this report.** It guessed TLB/locality as the
mechanism, and described 1M as costing "about 1.6x the read and 2.2x the decode". Both
were measured with the over-sized explicit cache; the cost was this budget effect, not the
context tier. The GDN state cannot be what grows, either: in `session_bytes` the
`max_cells` terms are `qsa_state_bytes` and `qsa_buffers_bytes`, and `gdn_state_floats(g)`
has none — so the GDN layer's `VRAM hits` above is the expert-cache read that every layer
does.

## Recall and limitations

**No needle or long-context recall test was run**, at any tier, so this report makes no
claim about retrieval quality at 524,288 or 1,048,576 tokens; it only shows that the tiers
load, read, and generate, and what they cost. The engine's
`KV streaming: 32768 of 524288 cells per QSA layer in VRAM` banner and fresh
43,969-token reads are all that is shown here.

The only quality check attached is a fixed 15-question GSM8K subset, which scored
**14/15** with the same wrong answer before and after the 0.1.39 upgrade. A
15-question subset can falsify a large regression and nothing finer; it is not a
licence.

Other limits: one machine, one quantization, one GPU; the ruler numbers come from
a 1,000-token reasoning-on workload and the probes from a 200-token one, so they
are not interchangeable; the context tiers of the first ladder are n=1 per arm, and the
cache-budget table below them is n=2–3 per arm on one set of three prompts; the pool-worker
sweep is single-arm screens on an older engine version, with the ABBA results
separately versioned; `--pcie-frac 0.35` was carried over from earlier tuning on this
machine and was not re-measured here. Sampled
decoding, vision input, tool use, multi-request concurrency, and sustained
thermal behaviour were not evaluated. Decoding here is greedy with thinking
enabled, so these throughputs are not comparable to reports measured with
reasoning off.
