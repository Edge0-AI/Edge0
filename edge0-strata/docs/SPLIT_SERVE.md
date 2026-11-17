# Split serving: read the prompt on one card, make the answer on another

> **Status: opt-in, first version, same make of cards on one PC. Not recommended on two alike cards: measured slower than the layer
> split on every workload we tried ([Measured](#measured)).** Nothing changes unless you turn it on. It needs two cards that can
> each run the whole model, and it can only pay when one card is clearly better at reading prompts and another at making answers.

The idea, the hand-made setup and the first measurements are by @MarcoTool in issue #4000: a router in front of two
engines, an RTX 3070 reading the prompt and two MI50 cards making the answer, with the session handed over through
`/dev/shm`. This page describes the built-in version of the same idea.

> **On this page:** [What it does](#what-it-does) · [Turning it on](#turning-it-on) · [When it helps](#when-it-helps) ·
> [Measured](#measured) · [How it works](#how-it-works) · [What must match](#what-must-match) · [Limits](#limits) ·
> [NVIDIA with AMD, and other PCs](#nvidia-with-amd-and-other-pcs)

## What it does

A normal two-card setup ([MULTI_GPU.md](MULTI_GPU.md)) cuts the model into layers and every card works on both stages. Split
serving does not cut the model. It starts two engines, each on one card, each with the whole model:

- the **prefill engine** reads the prompt and stops just before the first answer token;
- the engine saves that session into RAM, the **decode engine** loads it, and makes the answer.

The two engines share one copy of the experts in RAM (`--shared-expert-arena`, Linux). The client still talks to one
OpenAI-compatible endpoint; the server hands the session over between the two.

A prompt whose new part is short (a chat follow-up, an agent step) never leaves the decode engine: it is read in place, as
without the option. Only a long new part goes to the prefill engine.

## Turning it on

In the run config (`strata-<model>.json`), next to the other keys:

```json
{
  "split_serve": {"prefill_gpu": 0, "decode_gpu": 1}
}
```

or `python serve/server.py --split-serve 0,1` (prefill card first). The cards are numbered as `nvidia-smi` numbers them. The option
replaces `gpu`; the config must not also name `replicas`, `--layer-split`, `--batch`/`--slots` or `--prompt-cache 0` (the
server stops at start and says which).

| Key | Default | Meaning |
| --- | --- | --- |
| `prefill_gpu`, `decode_gpu` | required | The two cards. Different cards. |
| `min_prompt_tokens` | 2048 | A request is handed over when the decode engine would otherwise read at least this many new tokens. Below it the decode engine reads them in place. |
| `dir` | `/dev/shm` if it has room, else the temp folder | Where the session file lives for the seconds of a hand-over. |
| `shared_arena` | `true` | One copy of the experts in RAM for both engines (Linux, a file in `/dev/shm` when the experts fit there). `false`: each engine keeps its own copy (two copies of the experts in RAM). A file name: that file backs the arena (a file on disk works; it is written once). |
| `prefill_args`, `decode_args` | none | Extra engine arguments for one role, for example a bigger expert cache for the decode engine. They must not change the [session settings](#what-must-match). |

The startup line says what was chosen, including when the arena cannot be shared:

```
[strata] split serving: the prompt is read on card 0, the answer is made on card 1 (prompts with 2048+ new tokens are handed over); one shared expert arena in /dev/shm (32.0 GiB)
```

`/metrics` has a `split_serve` block: requests, hand-overs, fallbacks, bytes moved and the time spent saving and restoring.

## When it helps

Prefill needs matrix units and compute. Decode is limited by memory bandwidth and by how much of the model sits in VRAM. If one
card is much better at the first and another at the second, each can do what it is best at. That was the case in #4000: an
RTX 3070 reads a 32K prompt at about 1,600 tokens a second, the two MI50 cards read it at about 390 but decode at 40 tokens a
second with every expert in VRAM. The hand-over costs 2 seconds at 32K.

It does **not** pay when:

- the two cards are alike. A layer split over both cards reads a long prompt faster than one card and decodes faster than one card.
  Split serving gives both up, and each engine has only one card's VRAM for experts. On two RTX A6000 it was 4% to 48% slower;
- the hand-over (the session file's size over RAM bandwidth, written and read) costs more than reading the new tokens on the
  decode card would have;
- the new part of each request is short, which is most chat and agent use: nothing is handed over.

[Measured](#measured) shows the numbers on two identical cards.

## Measured

Two identical cards, so this is the case where split serving has the least to offer: 2x RTX A6000 48 GB (a lent box, Linux,
CUDA 13.1), Qwen3.8-Flash-Next IQ3_S, `--kv int8`, MTP drafting, default engine flags, greedy decoding. Three arms, each started
fresh for every round (3 rounds, 2 repetitions each, so 6 values per cell; every request is a cold prompt, no cache hit):

- **layer**: one engine over both cards (`--layer-split auto`), the normal two-card setup;
- **single**: one engine on one card;
- **split**: the prefill engine on one card, the decode engine on the other, through this option.

Wall time of the whole request, median of 6 (a prompt of the given length and 256 answer tokens; the chats are 3 turns, 128 answer
tokens each, follow-ups of 200 or 6,000 tokens, no turn markers in the token stream):

| Workload | layer | single | split | split / layer | split faster than layer |
| --- | --- | --- | --- | --- | --- |
| 8K prompt + 256 | 4.91 s | 5.27 s | 5.69 s | 1.16 | 0 of 6 |
| 32K prompt + 256 | 9.82 s | 13.64 s | 14.48 s | 1.48 | 0 of 6 |
| chat, 4K + two 200-token follow-ups | 4.75 s | 5.27 s | 5.99 s | 1.30 | 0 of 6 |
| chat, 8K + two 6,000-token follow-ups | 12.09 s | 13.43 s | 12.58 s | 1.04 | 0 of 6 |

Where the time goes:

| | layer | single | split |
| --- | --- | --- | --- |
| Prompt reading, 8K (to first token) | 3.10 s | 3.25 s | 3.68 s (3.23 s on the prefill card + 0.41 s hand-over) |
| Prompt reading, 32K | 7.89 s | 11.53 s | 12.34 s (11.54 s + 0.74 s) |
| Decode, tokens per second at 8K / 32K | 141 / 131 | 126 / 120 | 128 / 117 |

- **The hand-over is small but not free:** 0.20 s to save and 0.19 s to restore a 345 MiB session at 8K, 0.35 s and 0.39 s for 702 MiB
  at 32K. The file is about 225 MiB plus 14.7 MiB for every 1,000 tokens (285 MiB at 4K, 464 MiB at 16K). That is 7% of an 8K request and
  5% of a 32K one.
- **Alike cards give it nothing to win with.** The prefill card reads at the speed of one card (about 2,500 to 2,900 tokens a
  second) and the decode card decodes at the speed of one card. A layer split reads a 32K prompt in 32% less time than one card,
  because the pipeline keeps both cards busy, and decodes 9% to 12% faster. It wins every request here.
- **Short follow-ups are not handed over** (they are under `min_prompt_tokens`): the decode card reads them in place, as one engine would.
- **Long follow-ups are handed over every time** (the second chat). The prefill engine does not hold the answer the decode engine just
  made, so it reads the answer and the follow-up again (2.5 s for 6,128 tokens) and the session moves again. That chat is the closest
  to the layer split (12.58 s against 12.09 s). Nothing runs at the same time in this version, so the two cards still take turns.

**Exactness.** A fresh engine and the split path give the same answer, token for token, and the same session state:

| Check | Model, cards | Prompt | New tokens | Result |
| --- | --- | --- | --- | --- |
| two fresh single engines, then prefill engine + session file + fresh decode engine | Q2_0, 2x RTX A4000 | 4,096 | 64 | tokens equal, state fingerprints equal |
| the same | IQ3_S, 2x RTX A6000 | 4,096 | 64 | tokens equal, state fingerprints equal |
| the same | IQ3_S, 2x RTX A6000 | 16,384 | 128 | tokens equal, state fingerprints equal |

"State fingerprint" is the engine's `STRATA_STATE_HASH`: the whole session (running state, K/V of every layer, the draft layer's K/V) hashed after
the request. The fingerprint of the prefill engine's session also equals the decode engine's right after the restore. The check ran with
`--adapt-swaps 0 --pcie-frac 0 STRATA_IQ_MT_MIN=1`, the settings under which two fresh single engines already give the same answer
(without them the engine is not always deterministic from run to run, with or without this option). The tool is `tools/split_serve_check.py exact`.

On two alike cards, split serving does not pay for one request at a time. It can pay when one card is clearly better at reading
prompts and another at making answers, as in #4000. The gain there is the prompt time the weaker card would have needed, minus the
hand-over (0.4 s at 8K and 0.7 s at 32K on this box). We have not measured such a pair.

## How it works

1. The server sees a request with at least `min_prompt_tokens` new tokens for the decode engine (it tracks what the decode
   engine's session holds).
2. It sends the prompt to the prefill engine as `GEN 0 prefill=1 <ids>`. The engine reads the prompt as a request would, in
   the same chunks, with the same checkpoints, up to the place where the first answer window starts, and stops. Its session
   then holds all but the last prompt token. That is exactly the state a following turn resumes from, so the decode engine
   needs no new code path.
3. `SAVE <file>`: the engine's existing session file (see [DETAILS.md](DETAILS.md#session-files-disk)): the running state, the
   K/V of every layer and the draft layer's K/V, the deepest checkpoint.
4. `RESTORE <file>` on the decode engine, which checks the file against its own model files and settings.
5. The request goes to the decode engine as usual. It finds the restored session as the start of the prompt, reads the last
   token, and makes the answer. The file is deleted.

If anything on the way fails (the prefill engine died, the file was refused, the card is out of memory), the decode engine reads
the prompt itself, as without the option, and the log says why. A prefill engine that died is started again in the background.

The figures the server reports for the request (prompt time, tokens read, tokens reused) count the prefill engine's work and
the two file steps, so a client sees one request.

## What must match

The session file is bound to everything that changes what its bytes mean. Both engines start from one config, so these match;
they would not if `prefill_args`/`decode_args` changed them:

- the model files (sampled by size and first and last MiB) and the engine version;
- the backend (CUDA or HIP): **no cross-backend restore**;
- `--kv`, `--max-context`, `--kv-resident`, `--mtp-window`, `STRATA_KV_ROT`, the resolved rope settings, a loaded control vector;
- the arithmetic switches that change the computed state (the engine lists them in the file's fingerprint).

These may differ between the two engines: the expert cache size, `--prefill` chunk, sampling, drafting depth, the expert tier.
That is the point of the two roles: the prefill engine can spend its VRAM on prompt buffers and the decode engine on experts.

## Limits

- One request at a time, as for a single engine. The prefill engine could read the next request's prompt while the decode engine
  answers the current one; this version does not overlap them.
- The decode engine holds one conversation. A hand-over replaces it (the engine's restore does not park the session it
  replaces). A request that continues the conversation the decode engine holds is read in place.
- No parallel slots (`--batch`), no layer split inside an engine, no pictures (a request with pictures goes to the decode
  engine alone).
- Needs RAM for the experts once (shared arena, Linux) or twice (otherwise), plus the session file while it exists:
  about 225 MiB plus 14.7 MiB per 1,000 tokens at `--kv int8` on the Qwen3.8 model.
- Token for token the same answer as one engine, in the checks listed under [Measured](#measured).

## NVIDIA with AMD, and other PCs

This version runs on any two cards of one make that each hold the model (the engine drives one make of card, and a session file is
bound to its backend). It was tested on NVIDIA only. What the cases beyond it need, from reading the code and from #4000, none of it
built or measured here:

**NVIDIA with AMD on one PC** (the setup in #4000):

- *The session file must be allowed across backends.* Today the file's fingerprint contains the backend (`cuda` or `hip`) and the
  arithmetic switches that differ per card family, so a restore is refused. A cross-backend mode has to compare only what defines the
  bytes: model files, engine version, `--kv`, `--max-context`, `--kv-resident`, `--mtp-window`, the rope settings, `STRATA_KV_ROT`. The K/V,
  running-state and draft-layer layouts come from one source tree, but nobody has checked that the bytes are interchangeable on both
  backends.
- *A byte check is no longer possible.* The two backends do not compute the same bits, so the proof is not a state fingerprint. It
  would be a teacher-forced KL of the answer after a hand-over against the same backend's own continuation, with the noise floor of the
  decode engine against itself, plus needle and task checks, as for any change of arithmetic.
- *Two engine builds.* One CUDA and one HIP executable of the same version, so the config needs an executable and a card per role
  (`prefill_exe`, `decode_exe`), the card variable per backend (`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`), and a start check that both
  report the same version.
- *The shared expert arena works across makes* (#4000 does it with `--shared-expert-arena` in `/dev/shm`): it holds the pack's bytes and
  each engine pins them with its own runtime. It needs the experts to fit `/dev/shm`, or a file on disk, as for the same-make case.
- *Roles chosen from measurement.* Setup and `--calibrate` would read each card's prompt speed and decode speed and suggest which
  card reads and which answers, and say when the pair is not worth splitting. Without that the owner of the PC has to know.
- *A test machine with both makes.* We have none in 0.2.7.

**Another PC** (prefill on one machine, decode on another):

- The session file is 225 MiB plus 14.7 MiB per 1,000 tokens (about 2 GiB at 128K). A 10 Gb/s link moves that in 2 to 3 seconds; a
  gigabit link in 20.
- Each PC loads its own copy of the experts, so there is no shared arena. Both need the same model files and engine version; the
  fingerprint samples the files' size and first and last MiB, not their path.
- The engines read a session file from a local path and trust it ("restore only files this engine wrote"). A network hand-over needs a
  transport with authentication and an upload/download call on the server, which does not exist.
- Requests would have to be routed to the PC that holds the conversation, which is the replica router's job already.

**Several clients at once** (not built, not measured): the prefill card could read the next request's prompt while the decode card
answers the current one. That would raise the throughput of a busy server. It would not make one request faster than the layer split on
alike cards. This version runs one request at a time. Overlapping needs the server to take two requests at once (a prefill queue and a
decode queue), and the decode engine to take a restore between two of its own requests, which the session file already allows.

Credit: the idea, the hand-built router and the first measurements are @MarcoTool's (issue #4000, repository
`MarcoTool/PDHyperMix-Strata-MI50`).
