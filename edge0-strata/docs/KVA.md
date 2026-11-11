# KVA: a faster prompt read for long cold prompts (opt-in, off by default)

KVA ("approximate late-layer state") reads a long prompt exactly through the early layers, then for the late layers
builds **only their recurrent state** — the GDN recurrence and convolution history, the QSA K/V pages and the
indexer keys — instead of running the whole layer. The attention output, the output projection, the MoE half
(router, experts, shared expert) and the late layers' expert transfers are skipped, because nothing reads their
results. The last positions of the prompt are then read fully, exactly as before, and decode runs the ordinary
full path against the state KVA built.

**It is off by default.** Unset, the engine is 0.2.6's, prompt path and draft layer included. Set `STRATA_KVA=1` to
turn it on (the Strata app shows it as the "Faster long prompts" switch in Expert settings, off until the user turns
it on; the Terminal passes `STRATA_KVA=1`). Switched on, it uses the configuration the 0.2.7 quality runs found
cleanest: **split 36** (only layers 36–47 of 48 are approximated) and a **2048-token exact tail**. Both can be
changed with `STRATA_KVA_SPLIT` and `STRATA_KVA_SUFFIX`.

## Why it is off, and why split 36

KVA is an approximation, and the 0.2.7 quality runs (greedy, same build, KVA off against on, with a noise run of off
against off) show where it costs answers when more layers are approximated:

- **Split 24 (the first version, layers 24–47).** On long extraction tasks (a list of scattered register entries in a
  14K–30K document) 4 of 6 tasks passed against 6 of 6 exact, 249 of 270 facts found against 270 on a Tesla P100:
  records were silently dropped (3 of 8 at 14K) and a 30K answer began with an invented row, the request's own reference
  id. Agent loops with a ~10K-token system prompt, of which the first 8192 tokens (about 84 %) were approximated, had
  a final answer without its totals and a quote with wrong arithmetic. On the RTX 5070 (`--agent-long`, 22 tasks) the
  split-24 default passed 0 of 10 agent runs and found 556 extraction facts, by applying the handbook's rules wrongly
  in the approximated region; exact passed 4 and 7 of 10 in two runs. A tail of 8192 tokens with split 24 found 574.
- **Split 36.** On the same 22 tasks on the RTX 5070: agents 6 of 10 (exact: 4 and 7), extraction 585 of 606 facts
  (exact: 585), with prompt speed +11 % at 14K and +18 % at 30K. This is the cleanest configuration measured and the
  one `STRATA_KVA=1` selects.
- **Speed of the shipped opt-in (split 36, 2048-token tail).**
  - Tesla P100, IQ3_XXS: prompt speed +12 % at 14K (323.8 to 363.5 tok/s) and +20 % at 30K (336.1 to 401.9).
  - Radeon AI PRO R9700 (gfx12), IQ3_S, final build d1a57dc9d, 27K needle prompt: prompt 2411 to 3086 tok/s (+28 %), decode
    94.7 to 84.3 tok/s (−11 %, tokens per round 4.08 to 3.33). The needle was found with and without KVA. Nothing is
    approximated below 16K there, because the chunks are 16384 tokens.
  - The trade, plainly: a long prompt is read faster, and the answers after it decode somewhat slower on fast GPUs,
    because the draft layer's guesses are accepted less often once most of the context was approximated. On a slow GPU
    the decode difference is within the noise.
- Needle retrieval, unanswerable questions, short summaries, code and JSON were unchanged in every variant.
- Split 24 is faster (about 1.3x at 14K and 1.5x at 30K on the P100); the extra layers that stay exact are what buy
  the quality.

## When it runs

KVA runs when all of these hold, and says so in the engine log:

- Qwen3.8-Flash-Next on CUDA or HIP (not SYCL, not Metal). The split layer and the tail were measured on one shape, 48
  layers and 512 experts. Any other model (Qwen3.6-35B-A3B, other layer or expert counts) reads exactly even with
  `STRATA_KVA=1` and logs `KVA configured but NOT applied: this model's shape (L layers, E experts) is not one KVA was
  measured on`. `STRATA_KVA_ANY_MODEL=1` (research) lets it run there anyway, with one warning;
- one GPU, no `--batch` slots, no `--peer-device`, no DFlash2 drafter;
- the read is cold (it starts at position 0 of the session) and the text has no pictures;
- the prompt (or the part of it read in one go, see below) has at least `STRATA_KVA_MIN_PROMPT` tokens (4096) and
  more than the fully computed tail plus 64.

Otherwise the prompt is read exactly. An engine started with `STRATA_KVA=1` logs the reason for every read it turned
away (`strata prefill: KVA configured but NOT applied: <reason>`), and once at start-up when the engine configuration
rules KVA out for the whole run. Without `STRATA_KVA=1` it logs nothing about KVA.

## The fully computed tail and the draft layer

The last `suffix` positions (2048, or `STRATA_KVA_SUFFIX`) are always read exactly.

The MTP draft layer builds its K/V from the final residuals of the prompt. An approximated chunk has no final
residual of its own, so its draft cells are built from the rows that chunk does have. By default nothing about the
draft layer changes: its window is whatever it was (`--mtp-window`, default 32768 cells), it builds the same cells and
attends to the same cells as with `STRATA_KVA=0`. The target model verifies every token, so the draft only decides
how many tokens a verify window keeps (decode speed), never the answer.

Measured on the P100 (IQ3_XXS, auto cache and prefill, 14K and 30K prompts, 256 tokens), the draft layer attends to
those cells with acceptance within noise of the exact read at 14K and a few points lower at 30K (decode 6 % lower at
30K). A forced small window (`--mtp-window 2048`) and the research option below both cost acceptance on faster cards.

`STRATA_KVA_DRAFT_FLOOR=1` (research): the draft layer gets a *cell floor* per session, the first cell built from a real
row. It builds no draft cell below it and attends to `[max(floor, pos - window), pos)`, the exact tail growing with
the conversation. The floor follows the tokens of the state a request continues from (`kva_floor.hpp`); a state no KVA
read produced has floor 0. It is off by default because it costs draft acceptance after long prompts.

## Chats, follow-up turns and checkpoints (`--serve`)

The server reads a prompt in parts: up to the end of a long system prompt (the shared checkpoint), up to the last
turn header, then the new turn's header and the last token. Each part that starts at position 0 and is long enough
can take the approximate path; its own last `suffix` positions are exact, and every later part starts at a position
greater than 0 and is read exactly.

- **Turn 2 and later.** The server continues from the live session or from a conversation checkpoint (the state
  after the earlier turns) and reads only the new tokens, at positions greater than 0, so they are read in full. They
  are read exactly *on top of* the cells the first read left: nothing is approximated twice. Questions about the
  old document in turn 5 are answered from the same cells turn 1 had, so their quality is that of the first read.
- **Checkpoints and parked conversations** hold whatever cells the session held, approximated or not, and give
  them back as they were. A checkpoint taken inside an approximated stretch is a consistent state: the read
  continues from it exactly. Restoring a checkpoint and re-reading the same prompt from 0 (the `STRATA_CKPT_REREAD`
  debug check) differ with KVA on, as they would between any two reads that take different paths.
- **A cold read again** (the cache evicted the conversation, a new chat with the same system prompt) approximates
  its own prefix again.
- **Unset (or `STRATA_KVA=0`)** is 0.2.6: every prompt is read exactly, the draft window is 32768 cells, nothing else
  changes.

## Layer splits, batch slots, SYCL, Metal

These never take the approximate path and say why: a layer split reports "a layer split (not the whole trunk)", the
batch slots and a peer device switch KVA off for the process (one line at start-up with `STRATA_KVA=1`), the SYCL
engine has its own prompt path without KVA, and the Metal build has no state-only kernels (`STRATA_KVA` is ignored
there, with a line when it is set).

## Settings

| variable | default | what it does |
|---|---:|---|
| `STRATA_KVA` | off | `1` turns KVA on (split 36, 2048-token tail). Unset or `0`: the exact 0.2.6 prompt read and draft layer |
| `STRATA_KVA_ANY_MODEL` | off | `1` (research): also run KVA on a model whose shape (layers, experts) it was not measured on, with a warning |
| `STRATA_KVA_SPLIT` | 36 | first approximated layer (zero-based); 24 approximates more, is faster and lost records and agent loops |
| `STRATA_KVA_SUFFIX` | 2048 | fully computed tail positions; see above |
| `STRATA_KVA_MIN_PROMPT` | 4096 | shorter prompts stay on the full path |
| `STRATA_KVA_DRAFT_FLOOR` | off | `1`: the draft layer skips the draft cells of approximated chunks (research) |
| `STRATA_KVA_SKIP` | state | `moe` (opt-in): the late layers of an approximated chunk run their whole attention half (GDN and QSA outputs into the residual) and skip only the MoE half and its expert transfers; the log line says `mode moe-skip`. Slower than the default state-only form, closer to the exact residual |
| `STRATA_KVA_ONLY` | all | `gdn` or `qsa`: approximate only that layer type |
| `STRATA_KVA_MODE` | identity | `identity`, `oracle`, `linear` (linear is not implemented) |
| `STRATA_KVA_CAPTURE` | — | directory: write every late layer's real input for calibration (the read is exact while capturing) |
| `STRATA_KVA_CAPTURE_FP32` | off | write fp32 instead of bf16 records |
| `STRATA_KVA_ORACLE` | — | directory: replay a capture instead of a projector |
| `STRATA_KVA_GDN_NORM` | off | keep the full GDN recurrence (the A/B control for the state-only one) |
| `STRATA_LOGITS_DUMP` | — | debug (not KVA-specific): write every verify window's sampler logits to `path`, for KL / top-1 between two runs |

Every read says what it did:

- applied: `strata prefill: KVA on: split 36, suffix 2048, approximated 5120 of 7533 positions in 5 of 8 chunks, mode identity`
- turned away: `strata prefill: KVA configured but NOT applied: <reason> (prompt … positions, min_prompt …, suffix …, split …)`

A quality comparison needs the KVA arm to say `STRATA_KVA=1`; the control arm leaves it unset or says `STRATA_KVA=0`.

## Measured with split 24 and a 2048-token tail (Tesla P100 16 GB, Qwen3.8-Flash-Next IQ3_XXS, chunk 1024, expert cache 1500, MTP window 2048)

| prompt tokens | control prompt tok/s | KVA prompt tok/s | speedup | approximated | residual error on the suffix |
|---:|---:|---:|---:|---:|---:|
| 5,642 | 203.3 | 265.7 | 1.31× | 54 % | 0.357 |
| 14,581 | 202.5 | 311.2 | 1.54× | 85 % | not measured |
| 29,997 | 199.7 | 338.3 | 1.69× | 92 % | 0.451 |

- Decode throughput was unchanged within the run-to-run spread (24–28 tok/s) in every arm: KVA changes prompt
  processing only.
- Quality: on a 14.6K-token prompt with a fact inserted at 10/50/90 %, the KVA arm retrieved the fact at all
  three positions and produced the identical answer string. On a "summarise this documentation" prompt the
  answers stayed coherent and semantically identical, with different wording after 4–12 tokens.
- **Likelihood** (`STRATA_LOGITS_DUMP`): at the first generated token the approximation moves the next-token
  distribution by KL 0.018 nats at a 5.6K prompt (two control runs differ by 0.0013) and 0.020 nats at 30K
  (control floor 0.0032). The top-1 token is the same and 9 of the top 10 are shared in both cases. The effect
  does not grow with context even though the residual error does.
- Layer type matters more than the count: approximating only the 6 late QSA layers cost a 0.153 residual error
  at 1.08×; approximating 18 GDN layers cost 0.339 at 1.18×. A gated recurrence integrates its input over the
  whole prefix, so a wrong input keeps biasing the state, while a wrong QSA K/V only misplaces attention keys.
- On IQ2_XS the same 14.6K-token three-needle test gives 208.0 → 328.7 prompt tok/s (1.58×) with all three
  codes retrieved, against 201.6 → 320.7 (1.59×) on IQ3_XXS: the feature is not tied to one quantization.
- The control is not bit-reproducible on this box (identical runs differ in the last chunks by ~0.85 %), so the
  residual figures are quoted against that measured floor, never as bit equality.

## What the block input is

The late layers' input is the 10,240-value hyper-connection residual (`hc * n_embd`) that the prompt path
carries. In `identity` mode every late layer reuses the residual entering layer 24 — no fitted weights, no extra
work. Capturing the *real* per-layer inputs (`STRATA_KVA_CAPTURE`) and replaying them (`STRATA_KVA_ORACLE`)
reproduces the full path's answer exactly; that is the mode a fitted projector would replace.

A per-position linear projector was tested and rejected: fitted on 5,636 positions, a rank-256 map from the
boundary to each late layer's input is *worse* than reusing the boundary for the first nine late layers and never
gets below 0.80 relative error (`bench/2026-10-25-predictability.md` in the research folder). The residual is
moved by routing and attention decisions a linear map cannot follow.

## Capturing inputs

```sh
STRATA_KVA_CAPTURE=/data/kva-cap STRATA_KVA_CAPTURE_FP32=1 \
  strata --serve ... --prefill 1024 --tokens-file ids.txt --max-new 1
```

writes `L24.bin … L47.bin` in that directory: one row of `hc * n_embd` floats per prompt position per late
layer, in position order. A second prompt overwrites them. `STRATA_KVA_ORACLE=<dir>` replays such a directory in
place of a projector; the replay must use the same prompt and the same chunking.

## Implementation notes

- `include/strata/prefill/kva.hpp` — the plan, the environment variables and the reasons a read is turned away (`src/prefill/kva_test.cpp` tests them).
- `src/prefill/prefill.cpp` — the chunk plan (`kva_approx`), the capture/oracle taps, the state-only layer
  exits, the transfer-plan skip, and `give_back()` on a skipped MoE half: a layer that consumes no expert must
  still release its stream entries, or the issuer thread fills the ring and the next fully computed layer waits
  forever.
- `src/prefill/kernels.cu` — `gdn_recurrence_state_only()`, the same recurrence dispatch without the output
  normalization.
- CUDA and HIP builds (`src/`); the SYCL engine (`sycl/`) does not have the feature.
