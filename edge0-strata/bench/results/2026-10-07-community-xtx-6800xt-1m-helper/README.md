# RX 7900 XTX + RX 6800 XT helper cache, Xeon E5-2696 v4, 1M context

Measurements from 2026-10-06/07 on a custom Linux HIP build. This is a
results-only report of combined tuning, with exact prompts and per-request data.
It does not establish a universal speedup, an isolated helper-allowance gain or
an isolated Q8-kernel gain. It does not claim 100+ sustained tokens/s.

## Hardware and software

- RX 7900 XTX, 24 GiB, gfx1100: primary GPU, PCIe 3 x16 upstream link.
- RX 6800 XT, 16 GiB, gfx1030: expert-cache helper, PCIe 3 x8 upstream link.
- Xeon E5-2696 v4, 22 physical cores / 44 threads, AVX2, one socket/NUMA node.
- 121 GiB OS-visible RAM; installed DIMM capacity,storage model/type and GPU
  power limits are not recorded here. No host-wide clock/power/governor changes
  were made by this campaign. Peak host/VRAM usage is not summarized in this report.
- Ubuntu 26.04.1, amdgpu kernel 7.0.0-38-generic, GCC 15.2.0, CMake 4.2.
- TheRock/ROCm 10.2.0a20261005, gfx1030 + gfx1100 source build; exact Python/ROCm
  package versions are in [packages.json](packages.json),flags in [build-recipes](build-recipes/).
- A separate NVIDIA 27B service ran on two RTX 3060s on the same host. It was
  managed independently; its CPU/IO load was not isolated or measured here.
  Compilation was kept out of the timed inference windows.

## Model and runtime

Unsloth UD-Q4_K_XL of Qwen3.8-Flash-Next, all four GGUF shards at revision
`c8b5954a88c2775c546b92593eda40ea041d3176`. Exact immutable URLs and SHA256s
are in [provenance.json](provenance.json). Native routed weights are retained;
the auxiliary dense pack uses the existing `--compat-bf16` conversion. The MTP
proposal layer is Q2_0,derived from Qwen's revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`; target verification remains Q4_K_XL.
No target requantization or experimental speed projection was used.

Both baseline and final use one client slot, 1,048,576 context, int8 KV with 32768
resident cells, static YaRN 4 / original 262144, and the helper layout. All target
layers remain on the XTX. Both already use primary/helper allowances 448/128 MiB,
`--remote-expert-opt` and lag 2. Both keep 4096 MiB / 4 slots RAM conversation parking;
those parked conversations do not add client concurrency. CPU vision uses the
original F16 projector and encoder, 12 threads and at most 1024 image tokens.

Baseline native commit`477b5f7bf946d4d8a4f0d55877f84163d8ba9a82` is 0.1.40.1
plus the helper-allowance patch; API`2c1407dfde3fd9831853ec4984c3cef59bc659bd`
adds local metadata/thinking compatibility. The final native is
`96c35a0e54cb505579f83c1adde4b71d27c33436`,based on 0.1.40.2
`e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`; API is
`5597378da8b5c2fe180e30f6b1903342e12e60ee`. These are historical measured
sources,not measurements of a new isolated PR or stock 0.1.40.3.

The final configuration combines spec 8, 15 workers, adapt-every 4, lookup-chain 3,
streamed MTP normalization,exact kernel switches, coupled/Gumbel sampling and
the local opt-in HIP interleaved/Q8 path. [variants.json](variants.json) records
each measured arm's full args/env. The Q8 extension alone was not a consistent
whole-model win. Kernel bitwise parity and sampled model quality are different
checks; neither proves identical generated text for a changed sampler/cache.

The source incorporates or applies work from [#1181](https://github.com/Niko1221/Strata/pull/1181),
[#1125](https://github.com/Niko1221/Strata/pull/1125),
[#1123](https://github.com/Niko1221/Strata/pull/1123),
[#1305](https://github.com/Niko1221/Strata/pull/1305),
[#1240](https://github.com/Niko1221/Strata/pull/1240),
[#1087](https://github.com/Niko1221/Strata/pull/1087),
[#1281](https://github.com/Niko1221/Strata/pull/1281) and
[#786](https://github.com/Niko1221/Strata/pull/786). The interleaved layout builds
on the upstream implementation credited to Eddoursul. These contributions belong
to their original authors; the source history preserves that attribution.

## Results

All rates below are single-stream engine timings in tokens/s; reasoning tokens
are included. Coding/prose use medium thinking, temperature 1, top_p .95, top_k 20,
min_p 0, presence_penalty 0, seed 314159. Each long request has an 8192 output cap.

| Long coding arm | Individual rates | Actual output tokens | Repetitions |
|---|---|---|---|
| 0.1.40.2 helper control,interleaving off |82.5 /83.9|8192 each|2|
| Generic HIP interleaved port,row1 |84.8 /84.1|8192 each|2|
| Q8 extension,row1,otherwise same helper settings |82.2 /81.3|8192 each|2|
| Final combined15worker/spec8 arm |97.2 /96.7 /97.0|8192 each|3|

Final coding median 97.0, range 96.7–97.2. Final long prose 65.0/63.9, with 4820/3911
actual output tokens: the second ended before the campaign's 4096 token minimum.
The campaign's original 100 TG gate failed. Short 1536 token fixtures should be
kept separate: baseline coding median 76.2, n3; final 78.3, n1. Baseline prose
median 62.3, n3; final 65.5, n1. No paired confidence interval is claimed.

| Cold prompt fixture | Baseline PP | Final PP | Fresh input | Outputs |
|---|---:|---:|---:|---:|
| Nominal50K |1348.8|1350.1|47641|768|
| Nominal130K |1286.6|1294.2|123747|768|
| Full usable1M |587.4|591.7|1048064|504|

Cold/depth tests are greedy,thinking off; cached-prefix tokens are0. Full-depth
decode 39.3 baseline / 39.2 final. The full request exercises 1048064 input + 504 output
+ 8 API-reserved = 1048576 configured tokens. All three recall markers were found.
Markers were inserted at 10/50/90% of the initial character haystack,then the tail
was adjusted to capacity; these are not claims of exact token-percent depths.
Details are in [full-context.json](full-context.json).

First 200 HellaSwag and 200 WinoGrande items in the local files' order, greedy with
thinking off: baseline 181/200 and 173/200; final 181/200 and 175/200; zero unparseable
answers/errors in both. [quality.json](quality.json) includes item IDs/gold/answers,
scores and source-file hashes. This is a small matched sample, not overall model
intelligence or validated correctness of the generated coding task.

The final arm was selected after CPU worker, spec window and draft-confidence
screens. Its first two coding runs alternated with prose. The third followed
other trials,restoration of the winning configuration and functional checks;
it was an additional validation run,not three consecutive uninterrupted coding
requests. Expert-cache adaptation stayed enabled and was warmed by preceding
requests. Requests carried different leading tags; no prefix reset is hidden.
Full chronology and actual per-request cached-token counts are in
[provenance.json](provenance.json) and [measurements.json](measurements.json).

Earlier layer-split experiments produced hangs and an illegal-memory failure
with a 6800 XT kernel page fault; host reboots followed. Those failed configurations
are excluded from successful speed summaries. Their cause was not fixed or
established by this work. The final helper layout completed the checks without
another recorded AMD kernel fault. Text, tools, thinking variants, metadata and
image checks passed locally; concurrent-client coverage does not apply.

## Reproduction

The public archive contains the exact historical sources. It is research source,
not a proposed default change or a CUDA-qualified release:

```bash
git clone https://github.com/saikiran-rs/Strata.git strata-research
cd strata-research
git checkout community/amd-helper-1m-report-20261007
cd bench/results/2026-10-07-community-xtx-6800xt-1m-helper
```

Use the commit/path mapping in [source-layout.json](source-layout.json) for the
separate baseline/API/packing trees. `python prepare_sources.py --root <reproduction-directory>`
creates those source worktrees and the pinned GGML tree without building or serving.
Install the pinned package versions from `packages.json`; the recipes expect the
ROCm wheels under the virtualenv path named in `source-layout.json`.
GGML is pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. [build-recipes](build-recipes/)
contains the measured HIP flags; set`STRATA_RESEARCH_ROOT` to the reproduction
directory. Expected measured binary hash identifies the historical build;
compiler/loader/path differences can change a rebuilt binary's hash.

Download the pinned GGUFs,then use the packing source's`tools/iq_pack.py
--compat-bf16 --gguf <first-shard> --out <pack>`. Recreate the draft with
`tools/mtp_fetch.py fetch/verify`,`mtp_pack.py --experts q2_0`,and`mtp_rt.py`;
the original checkpoint revision is in provenance. Binary/tensor hashes identify
the measured artifacts in [artifact-hashes.json](artifact-hashes.json); relocating
path-bearing JSON metadata can change its hash without changing tensor bytes.
The measured expert profile
and hipBLASLt table are included in this folder; they are not automatic defaults.

[baseline.json](baseline.json) and [tuned.json](tuned.json) preserve the measured
parameters with portable placeholders. Replace`${RESEARCH_ROOT}`,`${MODELS}`,
`${PACK}`,`${MTP}` and`${OUTPUT}` with your paths; copy the provided profile/table
to the paths the configs name. Place the matching`.shared-settings.json` sidecar
beside each rendered config. Serve with the pinned API using
`python -m serve.server --engine strata --config <rendered-config> --port 8080`. Check that the HIP ordinals select only the
intended AMD cards. No included script starts/stops a service or changes tuning.

Run the client against that already-running server:

```bash
python bench.py --check-recorded
python bench.py --url http://127.0.0.1:8080 --case coding --reps 3 --out my-code
python bench.py --url http://127.0.0.1:8080 --case prose --reps 2 --out my-prose
python bench.py --url http://127.0.0.1:8080 --case depth50000 --out my-depth50k
python bench.py --url http://127.0.0.1:8080 --case depth130000 --out my-depth130k
python bench.py --url http://127.0.0.1:8080 --case full --out my-full1m
```

A full1M request took about 30 min
on this setup. The short commands reproduce the request fixtures/protocol;
matching the historical order matters because the adaptive cache stays enabled.
The client records server timings,usage,client first/last generated-delta times,
elapsed time,output text and failures. Client TG uses(completion_tokens−1)/
(last−first delta time),as in the original harness; SSE/network batching can
affect it. Engine TG is reported separately; total request time is not decode time.

For the quality sample,obtain the exact datasets matching the hashes,then:

```bash
python quality.py final quality-out http://127.0.0.1:8080 --hs 200 --wg 200 \
  --hellaswag hellaswag_val_full.txt --winogrande winogrande-debiased-eval.csv
```

Dataset content is not redistributed;
sample ordering and the answer-letter parser are preserved in`quality.py`.

Historical results are retained; no new inference measurements were run for publication.
