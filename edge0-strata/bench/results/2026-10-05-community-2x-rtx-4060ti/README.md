# Community benchmark: 2x RTX 4060 Ti 16 GB, Threadripper PRO 3975WX

Measured on 2026-10-05. Strata 0.1.39 (`6f32ec0`), Unsloth UD-IQ4_XS, 131,072-token context,
layer split across two RTX 4060 Ti 16 GB cards, with the experts only partly in VRAM. Two configurations
differ only in the MTP draft vocabulary: an Italian-built one (`prod`, what this machine serves day to day) and
setup's default one (`stock`). The workload is the RTX 5090 report's synthetic code-explanation prompts
(English), with greedy decoding and a 256-token output cap. It does not establish answer quality on other workloads.

**Medians, `prod`:** prompt **880 / 1,917 / 2,221 tok/s** and decode **54.4 / 56.6 / 54.7 tok/s** at
4,096 / 32,768 / 128,000 prompt tokens; first token at 128K after **58 s**. 6 of 6 needles found (32K and 128K).

## Hardware and software

- 2x NVIDIA GeForce RTX 4060 Ti 16 GB (AD106, 34 SMs, 16,380 MiB each), PCIe Gen 4 x8 each (the card's
  maximum width), 165 W power limit, default clocks. No NVLink or P2P is used: plain layer split.
- AMD Ryzen Threadripper PRO 3975WX (32 cores / 64 threads, AVX2, no AVX-512); the engine used the AVX2
  i-quant kernels. 128 GB DDR4 installed (125.6 GiB usable); the DIMM population and channel count were not checked.
- Model files on a Crucial P310 500 GB NVMe (all experts were loaded into RAM at start, so the disk is not
  on the decode path).
- Pop!_OS 24.04 LTS, kernel 7.1.5, NVIDIA driver 610.57.04.
- Strata `6f32ec070f23ced9f50e704d854d775da52591ab`, engine 0.1.39, **source build** for sm_89 with CUDA 12.6
  (setup's own build; [BUILD.json](BUILD.json)). Image encoder built for the CPU (`vision: cpu`).
- Background: the desktop session (about 100 MiB of VRAM on GPU 0) and idle services. The machine's usual
  web front end was stopped for the whole run, so no other request reached the server. An Ollama runner that
  normally sits on GPU 1 was unloaded at the start of the run (Ollama log) and was not present during it.

## Model and configuration

- `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `38bb39ee97821de2c9009abb7e93950eec396e66` (setup's pinned
  revision), `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`. All three SHA-256 hashes
  were checked by setup against its pinned values (`5ce89370…`, `577a38a2…`, `d4634e6d…`).
- Pack prepared by setup (`iq_pack.py`, native experts); the bundled `data/expert-profile.bin`; no calibration;
  experimental speed projection off.
- Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` on the CPU encoder, `--vision --vram-reserve-mib 700`
  (no images were sent during this benchmark).
- Context 131,072; INT8 KV; `--kv-resident 32768`; expert cache `auto` (`prod`: 4,083 slots / 9.20 GiB on
  CUDA0 and 3,678 slots / 8.30 GiB on CUDA1); prefill `auto`; layer split `auto` (CUDA0 layers 0-22, CUDA1
  layers 23-47); `--remote-expert-opt`; `--pcie-frac 0.00`; one request at a time (no `"parallel"`).
- MTP `--spec 4 --spec-min-p 0.50`:
  - `prod`: `--mtp Strata-data/mtp/rt-it`, the same Q2_0 draft layer with its draft vocabulary rebuilt from an
    Italian Wikipedia text (`draft_vocab_it.bin`, about 7 MB of itwiki text), for an Italian-speaking deployment;
  - `stock`: `--mtp Strata-data/mtp/rt`, setup's default draft vocabulary.
- Requests: `temperature 0`, `reasoning_effort "none"`, `max_tokens 256` (the benchmark script's defaults).

The exact run configurations are [prod/strata-config.json](prod/strata-config.json) and
[stock/strata-config.json](stock/strata-config.json) (local paths left in; no API key is configured).

## Method

[benchmark.py](benchmark.py) and [monitor.py](monitor.py) are the RTX 5090 report's scripts, **unchanged**.
For each configuration: a fresh server start, one warm-up request, then three runs each at 4,096, 32,768 and
128,000 prompt tokens. Every prompt starts with its own nonce, so no prefix is reused: the engine reported
0 reused tokens for every measured run. Prompt and decode rates are the engine's own timings (`/metrics`),
and TTFT is client-side over streaming (the first non-empty delta; reasoning is off, so it is answer text).
Model loading is not included. `monitor.py` sampled RAM and GPU memory every second.
[`tools/needle_bench.py`](../../../tools/needle_bench.py) ran on `prod` only: `--lengths 32k,128k --depths 10,50,90`.

Per-run results with the engine metrics and the generated text are in `*/results.json`, summaries in
`*/summary.json`, engine logs in `*/engine.log`, telemetry in `*/telemetry.jsonl`, the machine snapshot in
[machine.txt](machine.txt) and the run log in [run.log](run.log).

## Results

| Configuration | Prompt tokens | Reused | Generated | Runs | Prompt tok/s median [range] | Decode tok/s median [range] | TTFT s median [range] |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| prod (Italian draft vocab) | 4,096 | 0 | 256 | 3 | 880 [821-881] | 54.4 [52.4-55.0] | 4.7 [4.7-5.0] |
| prod | 32,768 | 0 | 256 | 3 | 1,917 [1,904-1,928] | 56.6 [55.4-57.4] | 17.2 [17.1-17.3] |
| prod | 128,000 | 0 | 256 | 3 | 2,221 [2,220-2,245] | 54.7 [54.0-56.2] | 58.0 [57.3-58.0] |
| stock (default draft vocab) | 4,096 | 0 | 256 | 3 | 906 [870-911] | 51.9 [51.2-52.4] | 4.6 [4.5-4.7] |
| stock | 32,768 | 0 | 256 | 3 | 1,934 [1,932-1,944] | 55.3 [53.6-56.1] | 17.0 [17.0-17.1] |
| stock | 128,000 | 0 | 256 | 3 | 2,234 [2,198-2,247] | 55.7 [55.4-55.9] | 57.6 [57.3-58.6] |

- Every measured run ended at the 256-token cap (`finish: length`). No request failed or stalled.
- Decode expert-cache hit rate 0.80-0.90 (the first 4,096-token run of each configuration is the lowest).
  MTP drafts accepted: `prod` 151-164 of 214-243 offered, `stock` 141-162 of 206-244.
- Memory, peaks over the whole run: VRAM 15,691 / 15,822 MiB (GPU 0 / GPU 1) for `prod`, 15,705 / 15,794 MiB
  for `stock`; system RAM in use (all processes) 80.8 GiB of 125.6 GiB; no swap used. GPU power peaked at
  146-149 W, temperature at 74-76 °C.

## Correctness and limitations

- Needles (`prod`): **6 of 6 found**, at 32,171-32,172 and 125,449-125,451 prompt tokens, depths 10 / 50 / 90 %
  ([prod/needles.json](prod/needles.json)).
- The Italian draft vocabulary did not slow down these English/code prompts: decode is within the run-to-run
  range at 32K and 128K, and slightly faster at 4K (54.4 vs 51.9 tok/s). One machine and three runs each,
  so this is an observation, not a general result.
- Concurrency was not part of this benchmark. A separate measurement of `"parallel": 2` on this machine is in
  the comment on #857.
- Not measured: images (the encoder was loaded, but no image was sent), thinking-on decoding, sampled decoding,
  other quantizations, and the open prompt-path PRs (#835, #849, #854).
