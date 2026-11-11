# Community benchmark on NVIDIA GeForce RTX 5090 D v2

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 D v2, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-14900K (24 cores, 32 threads); installed RAM: 125.6 GiB; storage: nvme0n1 ZHITAI TiPlus7100s 2TB nvme
- OS: Linux 6.17.0-14-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.55, commit 84c6f585; engine 0.1.55
- GPU detail (after the run, near idle): power_limit_w=450.00, power_draw_w=216.48, pcie=gen5 x16, pstate=P1, temp_c=55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-latent-moe
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack latent_fp4_pack --native model.gguf --native-dense-gguf model.gguf --ple-gguf ple-fp8.gguf --ple-io ram --expert-profile expert-profile-bench2.bin --expert-cache auto --prefill auto --pcie-frac 0.25 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vram-reserve-mib 2000 --pool-workers 24 --pcie-mode dma
```

Environment of the config: `STRATA_LATENT_DENSE=latent_dense_q8.bin`, `STRATA_LATENT_A=experts.bin`, `STRATA_LATENT_LABELS=labels.bin`, `STRATA_EMBED_VRAM=1`, `STRATA_LATENT_FUSED=1`, `STRATA_LATENT_FUSED_DU=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 52.8 (49.1-57.8) | 0.66 (0.58-0.69) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1757.9 (1756.1-1759.7) | not measured | 2.31 (2.31-2.31) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2554.1 (2546.8-2607.0) | not measured | 6.31 (6.18-6.33) |
| agent session, follow-up turns | 12919 (12916-12931) at the last turn | see below | up to 64 | 3 | see below | 68.3 (52.6-83.6) | 2.12 (1.95-2.34) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.1-69.5); earlier-turn tokens read again per session: 73 (3-144) (earlier turns that the cache did not hold; the 12063 (12063-12063) new tokens of the files are read once); first turn (nothing cached) first token: 1.64 s (1.53-1.68).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 404.0 | 1.532 | 42.4 |
| 1 | 3362 | 634 | 2728 | 1297.9 | 2.126 | 54.5 |
| 2 | 4895 | 3426 | 1469 | 757.3 | 1.963 | 52.6 |
| 3 | 8139 | 4958 | 3181 | 1459.7 | 2.203 | 69.7 |
| 4 | 10190 | 8185 | 2005 | 992.5 | 2.043 | 78.1 |
| 5 | 12916 | 10233 | 2683 | 1272.5 | 2.129 | 65.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 106 s.
