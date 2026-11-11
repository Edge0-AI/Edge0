# Preliminary native RTX 3090 / EPYC 7453 results

Issue: [#165](https://github.com/Niko1221/Strata/issues/165)
Engine: Strata v0.1.26, source build at `ac8b251`
Status: preliminary; prompt set and harness guidance requested upstream.

## Hardware

| Component | Guest-visible details |
|---|---|
| CPU | AMD EPYC 7453, 20 vCPU, AVX2; AVX-512 not exposed |
| RAM | 165 GiB visible to Ubuntu 24.04.5 KVM guest; physical DIMM layout and bandwidth not verified |
| GPUs | 2x RTX 3090 24 GB; PCIe Gen4 x16 maximum reported per card; driver 595.91.07 |
| Power limit | 280 W configured on each GPU; defaults are 350 W / 420 W, max-adjustable limits 366 W / 450 W |
| Storage | 541 GiB ext4 guest data disk; host storage layout not verified |

The power values above are configured caps, not measured workload draw. Idle draw during the hardware capture was approximately 35 W / 29 W.

## Method

Native Linux setup, local build from v0.1.26, `--prefill auto`, shipped expert profile, automatic expert cache, int8 KV, MTP on, images off, experimental speed projection off. The GPU power limit was 280 W per card. Native runs used a systemd service with `LimitMEMLOCK=infinity`; other llama-swap model containers were stopped.

The local API harness used the public Strata source tree as its prompt corpus, with generated tasks, `temperature: 0`, `reasoning_effort: none`, and `max_tokens: 256`. It is not the upstream code-agent prompt set or harness. Runs use unique request IDs and no conversation reuse in speed mode. Three runs per speed cell are summarized by medians; actual `usage.prompt_tokens` are recorded because nominal context labels can drift. The first decode run is also shown. Several requests ended early on EOS, so actual generated counts are in the raw JSON.

No calibration comparison, upstream prompt comparison, `STRATA_SPLIT_TIMING` stage timing, or under-load power-draw capture has been done. Treat all prompt/decode comparisons with upstream as indicative only until the requested upstream method is available.

## Results

See [`matrix.md`](matrix.md) for summary tables and [`matrix.json`](matrix.json) for all successful run records. Per-run source JSON, logs, the local harness, and scrubbed hardware facts are in [`data/`](data/).

Highlights:

- IQ3_XXS on one 3090: 1K–262K cells, three runs each; 262K prompts were 232,183 / 259,589 / 261,068 tokens and fit within the 262,144 context.
- Coder IQ1_M at 32K: single-GPU median decode 66.8 tok/s; dual-GPU auto (K=19) median 98.6 tok/s.
- Fixed Coder K sweep: K=22 was the highest observed three-run median (103.3 tok/s); independently generated prompts mean this is not a paired ranking.
- IQ3_XXS and IQ3_S dual-GPU auto both selected K=24 at 32K.
- IQ3_XXS and IQ3_S needles passed 3/3 at 8K / 16K / 28K on the tested single/dual configurations; two-turn checkpoint reuse passed for both.

## Limits and follow-up

- Upstream prompts/harness are not yet available; all current speed data uses a local source-tree corpus.
- The guest exposes one NUMA node and 165 GiB RAM, but physical DIMM topology and memory bandwidth are unknown.
- PCIe x16 is the reported maximum link width; the engine probes were about 23–26 GB/s. Verify PCIe generation/width under sustained load before claiming a direct x16-vs-x4 comparison.
- Coder dual-GPU needle/two-turn checks and dual-GPU timing breakdowns are not run yet.
- K=18 has one successful run at 83.1 tok/s; a repeated startup failed while uploading a dense weight with an out-of-memory error, so it is not a three-run median.
- Calibration/default comparisons and IQ3_S single-GPU results remain outstanding.
