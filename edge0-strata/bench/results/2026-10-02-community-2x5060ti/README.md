# Community benchmark on 2× RTX 5060 Ti 16 GB (PCIe gen3)

Measured on 2026-10-02 (speed) and 2026-10-01 (agent task) by @Efs-O.
Strata 0.1.34 running Flash-Next IQ3_S across two RTX 5060 Ti 16 GB on an older
PCIe gen3 desktop. The report has three parts: a speed test that follows
`docs/COMMUNITY_BENCHMARKS.md`, one real agentic coding task with hidden tests,
and, for context, the same task run by a dense local model and by paid hosted
models. The main limitation: the agent task ran once per setup (n=1), on Strata
0.1.32 rather than 0.1.34.

## Hardware and software

- GPUs: 2× RTX 5060 Ti 16 GB (Strata uses these two only). A third card, an RTX 3060
  12 GB, runs only the vision encoder (`cuda_device: 2`).
- PCIe: gen3, x8 / x8 for the two 5060 Tis (x4 for the 3060). nvidia-smi reports a
  lower generation at idle because of link power saving.
- Power limits: 180 W on each 5060 Ti (stock) and no overclock for the main results. A
  second set with a memory overclock is reported separately below.
- CPU: Intel i9-9900KF (8 cores / 16 threads, AVX2, no AVX-512). Strata started
  7 expert-pool workers plus the host thread.
- RAM: 128 GB DDR4-3200.
- Storage: models and packs on NVMe (Samsung 9100 PRO).
- OS: Windows 10 Pro 22H2 (19045). Driver 616.56, CUDA 13.4 (runtime from the venv's `nvidia/cu13`).
- Strata: tag `v0.1.34` (`1678de3`), engine 0.1.34, release binaries set up with
  `guarded_setup.py --model IQ3_S --gpus 0,1`.
- Background load: an idle desktop and VS Code. No other GPU work during the runs.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`). Pack built by setup.
  Uses the expert profile shipped in the repo (`data/expert-profile.bin`).
- Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` on the 3060. It was not used in any run below.
- Context 154,624, KV `int8`, KV resident 32,768. Expert cache `auto`, prefill `auto`, layer split `auto`.
- MTP drafts on (`--spec 4 --spec-min-p 0.5`). Reasoning: the server default
  (`xhigh`) for the speed test. Temperature 0.6. No calibration, no
  experimental speed projection, not in low-RAM mode.

```text
strata.exe --pack <data>/packs/iq3_s
  --native <models>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf <models>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt
  --max-context 154624 --kv int8 --kv-resident 32768 --vision
(run config: "gpu": [0, 1], "layer_split": "auto")
```

## Part 1: Speed (Strata 0.1.34)

### Method

The script is [`speed_bench.py`](speed_bench.py) (`python speed_bench.py <code folder> <strata log>`), using streaming `/v1/chat/completions` on the local server.

- Output capped at 256 tokens. Every run hit the cap.
- One unmeasured warm-up, then 3 measured runs per condition. The server had been
  running for several minutes, so model loading is not included. The expert cache
  was warm from earlier use.
- **short**: a 104-token coding request.
- **long-cold**: about 20.9K tokens of source code plus a question. Each run starts
  with a unique UUID line, so no prefix can be reused.
- **long-reuse**: the same 20.9K-token text sent again after its cold run, with only
  the last part changed. The engine reused 16,384 tokens and read about 4.5K fresh.
- Prompt and decode rates come from the engine's own `strata serve: prompt …` line.
  TTFT and total time are measured at the client and include HTTP. TTFT is the first
  non-empty delta, which is always reasoning text here. Keep-alive lines are ignored.
- VRAM is from `nvidia-smi` after each request: the whole card, desktop included.
- The prompt text was a private codebase, so it is not included. Any ~80K-character
  block of source code reproduces the long conditions.

### Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) | Total s median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| short | 104 | 0 | 256 | 3 | 55.6 (53.0–56.3) | 62.9 (60.2–68.9) | 1.89 (1.87–1.99) | 5.94 (5.56–6.21) |
| long-cold | 20,905–20,913 | 0 | 256 | 3 | 1,108 (1,097–1,109) | 56.4 (56.0–58.1) | 19.10 (19.09–19.27) | 23.61 (23.49–23.82) |
| long-reuse | 20,891–20,899 | 16,384 | 256 | 3 | 574 (566–576) | 56.8 (52.0–59.8) | 8.05 (8.02–8.17) | 12.51 (12.31–13.07) |

- MTP drafts accepted: 61–79% per run.
- Decode expert-cache hit rate: 89–96%.
- KV reads served from VRAM: 97.3–99.6%.
- VRAM after each request: 14,949 MiB on GPU 0 and 15,077–15,079 MiB on GPU 1. No paging or out-of-memory errors.
- Per-run data, including the engine's timing lines: [`runs.json`](runs.json).

Short prompts read at about 55 tok/s, and a 104-token request takes about 1.9 s
before its first token. That is the fixed per-request cost discussed in #340, and
it is unchanged on 0.1.34. Setting `STRATA_SPLIT_OWN=1` did not reduce it, and it
doubled cold prefill time (details in #340).

### Failures

None in these 9 measured runs.

### With a memory overclock (+1500 MHz on both 5060 Ti)

The same script on the same server, run again after setting a +1500 MHz memory
offset in MSI Afterburner on both RTX 5060 Ti. Memory ran at 15,302 MHz, against
14,001 MHz stock. Power limit 90% (162 W), core clock and voltage unchanged. Same
method: 3 runs per condition after a warm-up, 256-token cap.

| Configuration | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT s median (range) | Total s median (range) | Drafts accepted |
| --- | --- | --- | --- | --- | ---: |
| short | 55.0 (54.2–55.7) | 68.2 (65.7–68.2) | 1.94 (1.82–1.95) | 5.68 (5.67–5.70) | 71% (stock 66%) |
| long-cold | 1,125 (1,116–1,125) | 61.8 (59.5–62.1) | 18.76 (18.72–18.91) | 23.01 (22.84–23.04) | 74% (stock 70%) |
| long-reuse | 581 (578–582) | 61.1 (59.7–64.1) | 7.98 (7.95–7.99) | 12.16 (11.96–12.21) | 80% (stock 71%) |

- **Decode:** 8–10% faster than stock.
- **Prompt reading:** within about 2%, so not limited by VRAM speed.
- **Caveat on draft acceptance:** it was also higher in this set and raises decode
  speed on its own, so part of the gain may be run-to-run variation.
- **A cleaner check:** an intermediate run at +1000 MHz had the same 74%
  acceptance on long-cold as this one, and decoded at 57.0 tok/s against 61.8
  here, which points to a real memory effect.
- **Power:** each 5060 Ti stayed under 100 W.
- **Errors:** none.
- **Data:** [`runs-mem1500.json`](runs-mem1500.json).

## Part 2: One agentic coding task (n=1)

The task: implement a small semver library (`parse`, `compare`, `satisfies`) from a
32-line spec, inside an agent loop with file and shell tools. The agent sees 3
tests; 55 hidden tests score the result. Four rules are traps: the prerelease gate,
caret on `0.x`, comparators containing X, and partial hyphen bounds. A reference
implementation scores 54/55.

- The prompt was identical for every setup.
- "Time to done" runs from sending the prompt to the agent's final reply, model load included.
- These are **agent wall-clock numbers, not engine speed**. Every round re-sends the
  conversation, so prompt processing is inside the time.
- The rate is output tokens ÷ (response complete − request sent), averaged over
  each run's requests. It is end-to-end and lower than the decode speed in Part 1.

### Local models (same machine)

| Setup | Engine | Hidden tests | Time to done | Output tokens | End-to-end tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Next IQ3_S, effort `medium` | Strata 0.1.32, 2× 5060 Ti | 55/55 | 797 s | 38.8K | 49 |
| Flash-Next IQ3_S, effort `xhigh` | Strata 0.1.32, 2× 5060 Ti | 55/55 | 872 s | 42.4K | 49 |
| Flash-Next IQ3_S, effort `low` | Strata 0.1.32, 2× 5060 Ti | 54/55, unfinished (see below) | ~1,770 s | 41.5K delivered | 52 |
| Qwen3.8-27B UD-Q6_K + MTP, effort `low` | llama.cpp, tensor split on 2× 5060 Ti | 55/55 | 1,785 s | 69.6K | 44 |
| Qwen3.8-27B UD-Q6_K + MTP, effort `xhigh` | llama.cpp, tensor split on 2× 5060 Ti | 55/55 | 1,824 s | 64.3K | 42 |

On this task, Flash-Next on Strata finished in about half the time of the dense 27B
at Q6, on about 40% fewer output tokens, with the same score. At `low`, Flash did not
think less than at `medium`; it produced the longest requests of any Flash run.

**The `low` run did not finish.** Twice, a long tool call (about 18K generated tokens)
ran for over two minutes with no visible output, because tool-call arguments arrive
at the end. The client's 120 s idle timeout then disconnected, and after that the
server stopped answering new requests although `/health` still returned OK. It needed
a restart. That was on 0.1.32. On 0.1.34 we tried to reproduce it: we cut the
connection during a long tool-call generation, and the next request was answered
within 0.5 s every time. So it appears fixed by the cancel-on-disconnect changes
(#430/#431), and we are not filing an issue.

### Paid hosted models on the same task, for reference only

These ran in different harnesses (each vendor's own CLI, or a hosted API), on
different hardware, with their own tool sets. They are here only so readers can
place the local results. They are not Strata measurements and not a fair speed
comparison.

| Model | Harness | Hidden tests | Time to done | Output tokens |
| --- | --- | ---: | ---: | ---: |
| gpt-6-luna, CLI default effort | GitHub Copilot CLI | 55/55 | 108 s | ~4.3K (visible only) |
| qwen-3.8-27b on Cerebras (paid tier) | same agent loop as the local runs | 55/55 | ~211 s (one manual resume after hitting the output cap) | 167K |
| gpt-6-luna, effort `xhigh` | Codex CLI | 53/55 | 447 s | 23.5K |
| claude-haiku-4-5 | Claude Code (headless) | 44/55 | 452 s | 44.7K |

The paid models finished about 2–7× sooner. The local setups matched or beat them on
the score: every finished local run scored 55/55.

## Correctness and limitations

- Speed: 3 runs per condition, one machine, one model and config. The expert cache
  was warm from earlier use.
- Agent task: **n=1 per setup**. Run-to-run variance on agent tasks is easily ±30% in time.
- The task is small and fully specified, and most setups hit the 55/55 ceiling. It
  shows time and token spend, not which model codes better on harder work.
- The agent-task Flash runs used Strata 0.1.32, not 0.1.34.
- The agent harness used for the local runs and Cerebras is a personal VS Code
  extension. Its tool schemas and prompts differ from the vendor CLIs.
- Not measured: needle recall, vision, a context longer than ~21K for speed, and
  power draw. A spot check showed ~145 W for all GPUs during decode and a 212 W peak
  during a cold 20K prefill.
