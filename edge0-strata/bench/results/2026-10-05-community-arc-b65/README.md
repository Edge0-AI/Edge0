# Community benchmark: Intel Arc Pro B65, PCIe Gen4 x16

Measured 2026-10-06 by timnevits. Original full 512-expert Flash-Next IQ2_XS,
Strata v0.1.40 release source with six local patches (see BUILD.md), 8K context, INT8 KV and
confidence-limited MTP window 4. Three fresh native engine launches; five fixed
synthetic tasks at each of 512 and 7000 exact input tokens, 640 output tokens per
request. This is a native greedy speed/recovery report, not broad answer-quality
qualification or a comparison against another card/runtime.

The range of the three independently warmed five-task decode medians is
40.37–41.28 tok/s at 512 inputs and
39.81–40.90 tok/s at 7000 inputs.
Task-level results vary materially; the table below preserves those differences.

## Hardware and software

- One Intel Arc Pro B65, PCI ID 8086:e222, xe driver. Current-boot kernel reports
  physical VRAM 32 GiB and usable VRAM 31.89 GiB (`0x7f9000000` bytes). Physical
  root/card-upstream links both 16 GT/s x16, PCIe Gen4. The internal endpoint's
  2.5 GT/s x1 report is not the host-facing link.
- Intel Core i5-12600K, 10 cores / 16 logical CPUs, AVX2; four bounded engine workers.
- Four 32 GB DDR4 DIMMs, 128 GB installed, 3200 MT/s configured; Linux usable
  MemTotal 123.28 GiB. No swap during the campaign.
- Samsung SSD 980 1 TB NVMe; lookup-table reads remain file-backed. Page cache
  was not flushed, so fresh engine processes do not imply cold SSD/cache state.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; Intel compute runtime/OpenCL/LevelZero
  GPU driver 26.22.38646.7, LevelZero loader 1.28.6; oneAPI compiler 2026.1.0.
  Python 3.14.4. Exact versions/compiler flags are in [build.json](build.json).
- GPU power cap 200 W; no power/clock changes for this benchmark. No other GPU
  compute owner was allowed. Normal model router and its UI services were paused;
  core OS/network/SSH services and passive display probes remained. The display
  is driven by the integrated GPU. Monitoring was active during inference.

## Model and configuration

[BUILD.md](BUILD.md) gives pins, six local patches, build/pack steps and run
commands. The measured binary is SHA256
`520d1a72a7866956efc0feb4250cafeaf485ef86904e11950a392468e1bb91cb`.
This is a source-built SPIR-V/JIT build from the v0.1.40 tag. The Intel engine
still self-reports 0.1.39-sycl; source commit and patches identify what ran.
The listed patches are required to reproduce this build of the included Intel port.

Original `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, full 512 experts per layer,
IQ2_XS; two GGUF shards named in [profile.json](profile.json). Draft source is
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`,
packed Q2_0 using the upstream tools. [artifacts.json](artifacts.json) lists
pack, draft, tokenizer and weight identities. No custom draft vocabulary,
vision encoder, reasoning, calibration or experimental speed projection.

One request at a time; combined context 8192; INT8 KV, no KV streaming;
prefill 512; VRAM reserve 3072 MiB; `--expert-cache auto`, full fixed upstream
ranking, `--stream-experts`, `--pcie-frac 1`; protected 16 GiB pinned host mirror
and `STRATA_VERIFY_DEVICE_PLAN=1` / `STRATA_VERIFY_NO_HOST=1`. Missing experts
execute on the GPU through direct pinned-host reads; no CPU expert computation.
All missing experts must be mirrored or startup is refused.

`STRATA_DBG_NAN` is explicitly unset. The expensive qualification-only
intermediate-tensor downloads and CPU scans are disabled. The retained model and
launch settings are used with the disclosed patched v0.1.40 source build. All
three repetitions of each task/length reproduce the same output-ID hash.

Native verify/commit graphs enabled; eager off. `--spec 4 --mtp-max-t 4
--spec-min-p 0.5` permits up to three drafts per verification window, shortened
by confidence. `STRATA_WARM_GRAPHS=0` does not disable these serve graphs.
Suffix drafting, adaptation, prompt/conversation caches and short-read mode
are off. Sampling is temperature 0, seed 17; thinking disabled in the model's
template. [profile.json](profile.json) contains every native launch argument.

Startup geometry and lifecycle, per process:

|Run|Resident expert slots|Expert cache MiB|Startup free VRAM MiB|Startup→READY s|QUIT→exit s|Native exit|
|---|---:|---:|---:|---:|---:|---:|
|1|17885|24594|3211|5.048|0.114|0|
|2|17885|24594|3197|5.098|0.114|0|
|3|17885|24594|3211|4.990|0.114|0|

See [runs.json](runs.json) for native INFO and startup mirror/transfer lines.
Auto residency is sized once at startup and remains fixed within each process;
it is not an adaptive cache. NO_HOST hit/lookup/offloaded counters are unpopulated
zeros in this path, not measurements of zero PCIe traffic or a 100% hit rate.

## Method

[benchmark.py](benchmark.py) contains every synthetic task and deterministic
padding rule. It renders the exported model template with thinking disabled,
then inserts filler token IDs between the system/header and final task to make
exact 512/7000-token inputs. These are deliberately padded workload fixtures,
not natural documents containing 512/7000 tokens of useful task instructions.
[fixtures.json](fixtures.json) contains the exact input-ID hashes.

Each fresh engine first performs isolated canary/JSON checks, then an excluded
640-output community-garden warmup for the 512-token shape and its five measured
tasks. It then performs the analogous excluded 7000-token warmup and five long
tasks, checks post-long isolation, sends QUIT, holds stdin open through cleanup,
requires native exit 0 and waits five seconds for delayed faults. The same task
order is used in all three runs. JIT/startup and warmups are excluded from speed
results and recorded separately. OS page cache persists across runs.

EOS is respected; generation is never continued past an end-of-turn. All 30
measured requests reached 640 tokens and finished at the length cap. Every
task/length has exactly three repetitions, one in each engine process. All
per-task greedy 640-token output-ID hashes match across the three runs.

Prompt rate is freshly read tokens / native prompt duration; decode rate is
generated tokens / native decode duration. Both come from native `DONE` fields,
cross-checked against streamed IDs, and are separate from total request time.
Exact DONE and stderr request timing lines: [native-timings.log](native-timings.log).
Native token counts include control/EOS tokens where emitted. TTFT is monotonic
request-send to first actual native `T` token, ignoring heartbeat/progress lines.
Total is request-send through DONE, excluding loading/queueing/HTTP. Additional
post-first rate `(N-1)/(last-first)` is retained as a different timing convention.

Raw numeric records include all excluded checks/warmups: [results.json](results.json),
[results.csv](results.csv). Draft acceptance is accepted/offered; the table pools
those counts across the three repeats per task. No output text is published.
[summarize.py](summarize.py) validates counts/hash equality/timing calculations
and regenerates [summary.json](summary.json). For the published data, run
`python summarize.py . /absolute/path/to/new-summary` from this report folder.
[result.json](result.json) records the complete campaign counts/status.

## Results

Each rate/time cell below is the median [minimum–maximum] of **three repetitions
of that exact task and length**. Prompt/decode are native timings; TTFT/total
are native-client monotonic measurements. Rates tok/s; times seconds.

|Input tokens|Task|Runs|Generated|Reused|Prompt tok/s|Decode tok/s|TTFT s|Total s|Draft acceptance|
|---:|---|---:|---:|---:|---|---|---|---|---:|
|512|LRU cache code|3|640|0|352.96 [337.86–352.98]|48.82 [45.76–48.82]|1.48 [1.48–1.55]|14.56 [14.56–15.50]|81.9%|
|512|CSV import code|3|640|0|353.54 [338.00–353.57]|47.29 [44.72–47.32]|1.48 [1.48–1.55]|14.98 [14.97–15.83]|75.0%|
|512|Lighthouse story|3|640|0|349.61 [334.16–349.82]|37.46 [33.80–37.48]|1.50 [1.50–1.57]|18.55 [18.54–20.47]|66.9%|
|512|Inventory explanation|3|640|0|358.77 [342.70–358.82]|41.21 [39.78–41.22]|1.46 [1.46–1.53]|16.96 [16.96–17.58]|64.4%|
|512|Incident plan|3|640|0|350.93 [335.87–351.05]|41.28 [40.37–41.28]|1.49 [1.49–1.56]|16.96 [16.96–17.38]|66.7%|
|7000|LRU cache code|3|640|0|307.75 [292.99–307.78]|47.24 [44.62–47.30]|22.78 [22.78–23.93]|36.29 [36.28–38.24]|76.7%|
|7000|CSV import code|3|640|0|307.71 [293.06–307.75]|47.73 [45.20–47.74]|22.78 [22.78–23.92]|36.16 [36.15–38.05]|79.9%|
|7000|Lighthouse story|3|640|0|307.35 [292.70–307.43]|35.82 [31.77–35.83]|22.81 [22.80–23.95]|40.64 [40.63–44.06]|65.0%|
|7000|Inventory explanation|3|640|0|307.16 [292.84–307.41]|40.89 [39.81–40.90]|22.82 [22.81–23.94]|38.44 [38.42–39.98]|66.7%|
|7000|Incident plan|3|640|0|307.61 [293.07–307.62]|40.16 [39.48–40.20]|22.79 [22.79–23.92]|38.69 [38.68–40.10]|66.0%|

Five-task medians within each fresh process, kept separate from the per-task
repeat ranges above:

|Run|Input tokens|Prompt tok/s|Decode tok/s|TTFT s|Total s|
|---:|---:|---:|---:|---:|---:|
|1|512|352.98|41.28|1.48|16.96|
|1|7000|307.61|40.90|22.79|38.42|
|2|512|337.86|40.37|1.55|17.38|
|2|7000|292.99|39.81|23.93|39.98|
|3|512|352.96|41.28|1.48|16.96|
|3|7000|307.62|40.89|22.79|38.44|

## Memory, hardware health and recovery

Independent approximately 2-second monitoring covered startup, requests and
shutdown: 521 samples over 1051.3 s.
Observed maximum across GPU/VRAM sensors 72 °C;
peak DRM-resident VRAM 29.071 GiB;
minimum whole-host available memory 109.486 GiB;
peak cgroup memory 1.162 GiB. Pinned USM is not fully
charged to the cgroup, so that figure is not total process/host RAM use. Host
available memory includes other services and is an estimate; sampled peaks
can miss brief transients. [telemetry.csv](telemetry.csv) and
[health-summary.json](health-summary.json) preserve the observations. CSV memory
columns use GiB, frequency uses MHz and elapsed time uses seconds. Empty power
values mean the live power interface was unavailable, not zero consumption.

No swap or new swap-out, cgroup OOM, new kernel GPU fault, foreign compute owner,
or PCIe correctable/nonfatal/fatal counter event was observed. Both physical
links remained Gen4 x16. All three native exits were 0. Normal services and private
API/retained-model recovery passed afterward; temporary RAM content was empty
and normal crash capture restored. Guards were not relaxed for this report.

## Correctness and limitations

Every native stream count matched DONE, all measured outputs reached the cap,
prefix reuse was zero, exact canary isolation and JSON-shape checks passed,
and all per-task greedy streams reproduced across processes. Math answer
accuracy is recorded separately in the excluded JSON check; it is not evidence
of general arithmetic or coding quality. No emitted text, private conversation,
credential or model payload is included.

This campaign measures only greedy, one-request operation at two input lengths
inside the qualified 8K profile. It does not establish sampled/RP quality,
concurrency, larger contexts, cold storage performance, bandwidth saturation or
a causal B65/B70 comparison. Earlier qualification has separate broader
operational evidence; neither missing full PLE test fixtures nor old S2 parity
failures are claimed resolved by this speed report. No GPU profiler or special
decode/PLE timing flags were active during these measured runs.

## Additional public prompt fixtures

[Public benchy-v1 fixtures](benchy-v1/README.md) provide three fresh-process
repetitions at20/2185 input tokens and256 outputs using the same qualified
configuration. Their prompt content and output lengths differ from the five-task
640-output suite, so their results are kept separate. The native serving protocol
uses those public fixtures; the unmodified community benchy.sh harness was not
used. See the linked method, raw records and per-shape median/ranges.

## Arc qualification and upstream observations

[qualification.json](qualification.json) records 22 selected kernel tests plus
separate IQ/native-expert checks, 12 additional GEMM stride checks and 24 candidate
API checks, including streaming/tools, near-8K context, cancellation during
long prefill and clean restart/two native exit0 shutdowns. The candidate was
tested in isolation; normal serving retains the existing qualified 0.1.39 build.
No boot/service/driver change was made, and no candidate reboot test is claimed.
Retained serving/private API and model recovery passed after the benchmark.

The included Intel port still needs shared-interface synchronization to build
against this tag; see [build-findings.json](build-findings.json), [BUILD.md](BUILD.md)
and the two additional reproduction patches0007/0008. This measures the repaired
default paths and does not establish support for the new CUDA/HIP speed features.
The tagged Intel engine still labels itself 0.1.39-sycl, so the source commit and
patches identify the measured build.

Two further source observations for maintainers: the tagged setup Intel card
table lacks this B65's verified PCI ID 8086:e222 and 32 GiB VRAM; and
STRATA_VERIFY_COHERENT is guarded by STRATA_USE_HIP in the tagged SYCL verifier.
The Intel CMake target does not define that macro. No speed effect from that
switch is claimed. These are setup/source observations, not GPU faults.
