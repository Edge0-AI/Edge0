# Community report: Flash-Next IQ2_XS on Strata vs a dense 27B, as coding agents on one RTX 4090 + 32 GB RAM

Measured 2026-10-04 by [T-Crypt](https://github.com/T-Crypt). Same card, same prompts, same thinking budget:

- **Strata**: Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, engine 0.1.39 + #700 #783 #787 #790.
- **Dense baseline**: Qwen3.8-27B on the NInfer engine (ninfer-4090 fork, deploy build), a dense model that fits
  in VRAM.

Two stages: a synthetic bench with long-context recall, then three real tickets done by Claude Code subagents, three
at once per engine, graded against a rubric fixed before the runs. The prefix-reuse section is the main finding for
Strata users.

## Hardware and configs

- RTX 4090 24 GB (450 W limit), i9-14900K (AVX-2, no AVX-512), 32 GB RAM (31 GiB usable), Linux, desktop on the iGPU.
- Strata: `--max-context 262144 --kv q4_0 --resident-experts --expert-cache auto --prefill auto --spec 4
  --spec-min-p 0.5 --pcie-frac 0.20 --mtp`, `reasoning_budget_tokens` 4096, `STRATA_RESIDENT_HEADROOM_GIB=6`,
  `conversation_cache_mib` 0. Expert cache ~11.8K of 24,576 slots in VRAM, the rest page-locked in RAM.
- NInfer: `--max-context 262144 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --default-thinking-budget 4096
  --max-concurrency 1`.

| | Strata IQ2_XS | NInfer 27B |
|---|---|---|
| peak VRAM | 23,936 MiB | 23,134 MiB |
| peak host RAM | 27,804 to 27,968 MiB | not sampled |

## Stage 1: synthetic bench

Build = one HTML dashboard from a spec (CSS rules counted). Review = a script with 12 planted bugs. Raw rows:
`r14-results.jsonl`.

| run | engine | decode tok/s | CSS rules | bugs found | wall time |
|---|---|---|---|---|---|
| a | Strata | 173 | 113 | 12/12 | 2m11s |
| b | Strata | 182 | 108 | 10/12 | 2m06s |
| c | Strata | 185 | 106 | 9/12 | 2m21s |
| a | NInfer | 140 | 181 | 12/12 | 5m21s |
| b | NInfer | 127 | 190 | 10/12 | 4m56s |

Recall, three facts at 10/50/90% depth (`needle-strata.txt` for Strata):

| tokens | Strata found / prompt tok/s / s | NInfer found / prompt tok/s / s |
|---|---|---|
| 55,586 | 3/3 / 3,494 / 17.3 | 3/3 / 3,037 / 23.3 |
| 117,153 | 3/3 / 3,904 / 31.3 | 3/3 / 2,391 / 51.2 |
| ~199,500 | 3/3 / 3,833 / 53.7 | 3/3 / 1,857 / 110.2 |

## Stage 2: three real tickets, three agents at once

Claude Code subagents, one git worktree per model and task, a 150-minute wall-clock cap per engine, engines run one
after the other. Prompts: `prompt-T1..3.md`. Rubric: `rubric.md` (correctness, verification, scope, quality,
honesty, 0 to 10 each). The grader re-ran every test, lint and dry run the agents claimed.

| task | Strata IQ2_XS | NInfer 27B |
|---|---|---|
| T1 Python config renderer | 42 | 35 |
| T2 Quickshell QML feature | 0 (no edits in 150 min) | 34 (complete; the harness hit its prompt limit before the commit) |
| T3 evidence write-up, every number cited | 44 (done in 120 min) | 43 (done in 62 min) |
| total /150 | 86 | 112 |
| finished in the cap | 1 of 3 | 2 of 3 |

On the two tasks both engines finished (T1, T3), Strata scored 86 of 100 and NInfer 78.

## Prefix reuse with three concurrent agents

| | Strata | NInfer |
|---|---|---|
| prompt tokens served from cache | 17.5M of 54.0M (32.4%), 938 requests | 11.8M of 13.3M (88.5%), last 101 requests |
| a logged turn | `prompt 109780 tokens = 2559 reused + 107221 read in 27966 ms` | 172,578 prompt tokens, 172,343 from its state cache, first token 524 ms |

With `conversation_cache_mib` 0, three interleaved agent conversations left Strata reusing only the shared ~2.5K
system prompt. Each turn re-read 70K to 110K tokens at ~3,800 tok/s, about 90% of the wall time. Strata's decode
and prompt speed were higher than NInfer's in every measurement here; prefix reuse decided the elapsed time.

We ran with the conversation cache off because host RAM is the binding budget on 32 GB (27.8 to 28.0 GiB peak).
Untested here: a conversation cache sized for three agents, or one agent at a time. Both are the obvious next runs.

## Limits

- One run per engine per ticket; two to three runs per engine on the synthetic bench.
- One grader model (Claude), rubric fixed before the runs.
- Strata ran first, NInfer second.
- NInfer's reuse figure covers its last 101 requests (log buffer).
- Single machine class: 24 GB card, 32 GB RAM.
