# Strata on two or more GPUs (layer split)

This page is for people with two or more graphics cards of the same make in one PC. A layer split puts one model on
several cards. NVIDIA is described here; AMD limits and measurements are in [AMD_HIP.md](AMD_HIP.md).

> **On this page:** [Using it](#using-it) · [Main, Helper, Off](#main-helper-off) · [How the split works](#how-the-split-works) ·
> [Tuning options](#tuning-options) · [Pipeline windows](#pipeline-windows) · [Two models on two cards](#two-models-on-two-cards) ·
> [DeepSeek](#deepseek-v41-flash) · [Limits](#limits) · [History](#history--older-measurements)

## Using it

**Desktop app:** choose Main, Helper or Off for each card. **Terminal:** run `strata launch` and use the Cards step, or name
the cards:

```sh
strata launch --gpu 0,1
strata launch --gpu-roles 1=main,0=helper     # cards you do not name are Off
```

Choose a model that fits the combined memory. A split can add capacity without adding speed.

**Source checkout only:** `START-HERE.bat` and `./setup.sh` have their own card picker. Their flags are not `strata launch` flags:

| Source setup flag | Meaning |
| --- | --- |
| `--gpus 0,2` | These cards together, numbered as `nvidia-smi` numbers them. Strata puts the faster card last. Remembered. |
| `--gpus all` | Every card that can share the model. |
| `--gpu 0` | One card (at a start: for that start only). |
| `--layer-split auto` | The default. Or the first layer of each later card, for example `18` or `16,32`. |
| `--gpu-roles 1=main,0=helper` | Roles, saved. |

Or edit an existing run config (`strata-*.json`) and restart. Merge these fields into the config; this is not a complete config:

```json
{
  "gpu": [0, 2],
  "layer_split": "auto"
}
```

Docker: `GPU_ROLES=1=main,0=helper` is passed as `--gpu-roles` ([DOCKER.md](DOCKER.md)).

## Main, Helper, Off

Every card has one of three roles.

| Role | What it does |
| --- | --- |
| **Main** | Runs the last stage of the split: the output head, the draft layer and the check of its guesses. A prompt chunk waits on it. Exactly one card is Main. |
| **Helper** | Holds a share of the layers and of the expert cache, so more of the model sits on graphics cards instead of being read from the drive. The effect on speed depends on the cards and the workload. |
| **Off** | Strata does not touch the card (a game, the screen, a card that is too small). |

The Main card is the last card in the engine's order. The first card runs the first layers and, by default, the picture reader.

**Recommended roles** (the same rule in the app, `strata launch` and setup; `strata_control/gpu_roles.py`):

- The faster card is Main. Speed is multiprocessors x max clock. Cards within 5% count as the same card, and the one with more memory wins, then the lower number. Of two identical cards the highest number is Main. A card with clearly more memory beats a faster one that is at least half as fast.
- Every other card starts as Helper, except a card with 8 GB or less when a bigger card is present (Off; with only 8 GB cards the second one is a Helper), built-in graphics beside a card with memory of its own (Off), and a card the engine cannot run (Off).
- A role is one click (app), one key (`strata launch`: M, H, O; R restores the recommendation) or one line (above).

Only a few things are warned about, one short line each: built-in graphics as Main, a card the engine cannot run, and too
little memory for the chosen setup. All cards Off is allowed. The CUDA, HIP and SYCL engines cannot start without a card and
say so; the Mac engine says "Performance is degraded on CPU only." and goes on.

**In the run config** (what the app writes). Merge these fields into the existing config:

```json
{
  "gpu_roles": {"1": "main", "0": "helper", "2": "off"},
  "gpu": [0, 1],
  "gpu_order": "as_given"
}
```

`gpu` is the same choice for older engines and the Monitor: the Helpers (slowest first), then the Main card, Off cards left out.
The server reads `gpu_roles`. If `gpu` was edited by hand and no longer matches, `gpu` wins and the log says so. A file with only
`"gpu": [1, 0]` shows as Main = the last card, Helper = the first, the rest Off. The recommendation is not saved, so a card you
add later is recommended again; a card you add after choosing roles starts Off.

**Which cards count** (`strata_control/gpu_fit.py` holds the memory arithmetic): not an integrated chip beside a real card; at
least 4 GB of graphics memory to be listed; 8 GB or less starts Off beside a bigger card. Each extra card of a split costs an
estimated 2 GB of its own buffers.

**Not supported** (setup says so and names the cards that can be used):

- A card older than the RTX 20 series (compute capability below 7.5). Pascal and Volta cards are admitted when you name them
  with `--gpus` (or `--gpu`, `--cuda 12`) and then run the experimental CUDA 12 engine ([OLDER_GPUS.md](OLDER_GPUS.md)).
- A card with less than 8 GB of VRAM together with others, because each card holds a copy of the dense weights and its own
  prompt buffers. If you name it with `--gpus`, setup states the risk and asks (`--yes` goes ahead).
- Intel GPUs in a split, and a mix of NVIDIA and AMD cards. An engine drives one make of card. The app, `strata launch` and setup
  offer only the cards the engine can drive, and say so when the other make has the bigger card. AMD cards share a model among
  themselves: `./setup.sh --backend hip --gpus 1,0` ([AMD_HIP.md](AMD_HIP.md)).
  A prompt read on one card and an answer made on another, with the session handed over in RAM, is a different opt-in: [SPLIT_SERVE.md](SPLIT_SERVE.md).

## How the split works

The layers are cut into contiguous ranges, one per card. The first card runs layers 0 to K-1, the next runs K onward, and the
last card also runs the output head and the draft (MTP) layer. Each card keeps an expert cache for its own layers only, so two
cards hold about twice the experts one card holds.

This is pipeline parallelism, not tensor parallelism. A token crosses from one card to the next once per verify window (a few
hundred KB through pinned RAM). No NVLink or peer-to-peer access is needed, cards on x4 or x1 slots work, and each card's PCIe
share is probed on its own link. A `--pcie-frac` you give applies to every card and skips those probes.

**What each card holds**

- Every card: a copy of the dense weights, its own session state (the KV cache of the full context), its verify window, its
  prompt-path buffers, and an expert cache for its layers filled from the expert profile.
- The last card: also the output head and the draft layer (about 0.8 GB).
- Host RAM: the expert arena once, shared by all cards. The CPU pool computes whatever no card holds.

Prompts are read in chunks that flow through the cards in turn: while a later card reads chunk c, the first card reads chunk c+1.
Conversation checkpoints save and restore every card's state.

**`"layer_split"`** is `"auto"` (placed by each card's free VRAM) or the first layer of each later card: one rising number per
card after the first, not a count per card. With four cards and a 48-layer model, `"24,36,42"` puts layers 0-23 on the first
card, 24-35 on the second, 36-41 on the third and 42-47 on the last. The server checks the value before the start and says what
is wrong.

**`auto`** tries every placement for two, three or four cards (proportional to free VRAM beyond that) and keeps the one whose
caches hold the most of the expert profile, hottest pairs weighted most. Ties go to the placement that leaves the fullest card
the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

**Card order with `auto`:** the faster card goes last, because the last stage runs the head, the draft layer and the verify, and
a prompt chunk waits on it. Cards within 5% keep your order, and so does a manual split. `"gpu_order": "as_given"` keeps your
order under `auto` as well. The log line `layer split: card order ...` says when the order changed.

**Unequal cards under `auto`.** The search asks for at least 3 layers per card and, on two cards, first for a placement in which
each card owns at least its share of the layers by its room for experts. The log line says which rule chose the split. Prefill
that stays flat from 4K to 16K tokens is the sign of an idle card. On an unequal pair, sweep `"layer_split"` once, because a
4K-only benchmark cannot see it. On x16 plus x4 rigs the link-miss scaling can move `auto` off the balanced placement;
`STRATA_SPLIT_UNIFORM=1` (with `"gpu_order": "as_given"`) prices every link the same.

## Tuning options

These are engine flags and environment variables. Put flags in the config's `"args"` and variables in its `"env"` object, then
restart. Defaults are off unless the table says otherwise.

| Setting | What it does | Notes |
| --- | --- | --- |
| `"split_skip_if_fits": true` (flag `--split-skip-if-fits`) | Runs on the first card alone when it holds every profiled expert plus the context's KV, the draft layer and the reserve. Otherwise the split stays. | Opt-in. Needs `--layer-split auto`. |
| `STRATA_SPLIT_OWN=1` or `auto` | Gives every card its own prompt buffers. `auto` does it only where the buffers are at most 12% of each card's VRAM. | Output differs from the default's (stable and coherent). |
| `STRATA_PREFILL_HELP=1` | The idle card computes a share of the streamed experts of a one-chunk prompt. `STRATA_PREFILL_HELP_FRAC=f` fixes the share. | Helps prompts up to about 3.3K tokens on slow links only; not bit-identical; native packs on the MMQ prompt path only. |
| `STRATA_SPLIT_MTP_BATCH=1` | The last card fills the draft layer's prompt K/V in batches, as on one card. | Needs a `--native` pack and a paged or ring draft K/V; says why once if it declines. |
| `STRATA_STAGE_TRIM=1` | Each card loads only its own layers' dense weights (needs explicit `--layer-split K`, not `auto`). The freed VRAM goes to that card's expert cache. | Output can differ slightly. |
| `--vram-reserve-later-mib N` | A separate VRAM reserve for the later cards (default: `--vram-reserve-mib`). Use it when the display sits on the last card. | `--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card that VRAM. |
| `--prefill-pipe R` (or `STRATA_PREFILL_PIPE=R`) | Reads each prompt in the chunk count with the least pipeline time for b/a = R tokens. `1` evens out today's chunk count. | Opt-in. `./setup.sh --calibrate` measures it per PC and writes it to the config. |
| `STRATA_SPLIT_RING=N` | Sets the prompt path's streamed ring to N slots (`0`: the pinned-share rule). A split uses 96 when at least 75% of the (layer, expert) pairs sit in some card's cache. | The startup line `lend sizing: ...` shows the result. |
| `STRATA_SPLIT_CHUNK16K=1` | Keeps the 16384-token auto chunk on AMD gfx12 splits (default 8192). | |
| `STRATA_SPLIT_OVERLAP=1` | Overlapped hand-off between stages on different cards. | Helps AMD; no gain on NVIDIA. See [below](#overlapped-hand-off). |
| `--split-device D1[,D2..]` | The later stages' devices (default: the next visible ones). `--layer-split K --split-device 0` runs both stages on one card; a bit-exact hand-off check, not a speed mode. | |

**The resident RAM mode works on a split.** `--resident-experts` (the low-RAM mode) leaves out of the RAM copy the experts any
card's cache holds. An adaptive swap copies the evicted expert back into RAM from the card that owns its layer. A RAM budget
(`--resident-budget-gib`) works on a split as well ([UNSLOTH_Q4.md](UNSLOTH_Q4.md)). With `--pcie-frac 0 --adapt-every 0` the
split's greedy output is the same in either mode.

**Under WDDM (Windows and WSL2)** only 8 GiB of the expert arena is pinned on a multi-GPU run; the rest streams through the
pinned staging ring. Linux pins the whole arena. `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.

### Overlapped hand-off

By default each verify window runs the stages one after another. `STRATA_SPLIT_OVERLAP=1` (stages on different cards only) gives
the hand-off a ready flag, so the next stage's graph is launched while the earlier stage still runs and waits on the flag on
its GPU. It takes one stream sync and one graph launch per window off the critical path. Batch windows keep the serial order, and
it is off beside `--pipeline-windows`. The wait is bounded by `STRATA_SPLIT_WAIT_MS` (default 30000, `0` = no bound); on timeout
the window fails with an error naming the setting and the server restarts the engine.

## Pipeline windows

`--pipeline-windows 2` lets the first card start the next verify window while the last card still verifies this one. The next
window is a guess: that this window is accepted whole and that its bonus token is the one the draft layer predicts. If the guess
holds, half of the next window is already done. If not, the first card restores its state and builds the window from the real
tokens. The tokens are the serial loop's, up to CPU rounding.

- **Default:** on for AMD RDNA4 (gfx1200 / gfx1201) with `--serve`, the draft layer, and a split of two or more stages when the
  drafter can sit on the first card, otherwise three or more. The log says "is on by default for a layer split of N stages".
  `--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off. NVIDIA builds keep it opt-in (`--pipeline-windows 2`).
  The Intel Arc engine turns it on for exactly two Arc B-series (Battlemage) cards, with `--serve` and the draft layer (`--spec 2`
  or more): on 2x Arc Pro B65 decode is 17% to 23% faster, and setup writes `--spec 2` for those cards ([INTEL_ARC.md](INTEL_ARC.md)).
  It runs exactly two cards, so one card, Arc A-series cards and three or more cards stay serial.
- **Draft layer placement:** with the pipeline on, the draft layer loads on the first card, whose window has finished when the
  chain starts. `STRATA_MTP_STAGE0=0` keeps the old place.
- **Off, with one log line saying why:** `--batch` slots, `--peer-device`, the helper caches (`--expert-cache-device1..3`, which
  `--remote-expert-opt` builds on), a split onto one GPU, no draft layer, and on three or more stages `--pipeline-windows 1` and
  `--adapt-async 1`. A request with repetition penalties or coupled draft sampling decodes serially.
- **Cost:** a second verify window on each card, 160 MiB less expert cache per card, plus two copies of the first card's
  recurrent state (about 3 MiB per GDN layer it runs).
- **Config:** `"args": [ ..., "--pipeline-windows", "2" ]` with `"layer_split": "20"`.

`--remote-expert-opt` does something only with a helper cache; on a plain layer split it is inert.
The `STRATA_PIPELINE_*` tuning and test variables are read only with `STRATA_PIPELINE_DEBUG=1`.

## Two models on two cards

The other way to use two cards is two models, one per card, each answering on its own. In the app, turn on **Run multiple
models** on the Models page. The card list then says which card a loaded model uses ("In use by ..."), starts that card as Off
for the next model, and explains a failed start on a card that is taken. Two models at once on two makes is not supported.

**Measured on:** a PC with two of four Radeon AI PRO R9700 cards, Linux, AMD build, Q2_0 on one card and IQ3_XXS on the other.
Both loaded (33 s and 59 s with the files cached, 114 s the first time) and answered chats at about 60 tokens/s each. Unloading
one left the other answering. The same app code drives NVIDIA cards, but two models on two NVIDIA cards were not run.

## DeepSeek V4.1-Flash

DeepSeek V4.1-Flash runs in the app and the Terminal and spreads its experts and its dense weights over several cards. It does not use
the Qwen layer split described above. On four Radeon AI PRO R9700 cards it reaches about 31 tokens/s. Hardware, sets and speeds:
[MODELS.md](MODELS.md#deepseek-v41-flash).

## Limits

- Every card needs compute capability 7.5 (RTX 20 or newer), unless you opt into the CUDA 12 engine for older cards.
- Works across cards: images (`--vision`), control vectors, KV streaming (`--kv-resident`) and mid-prompt checkpoints
  (`--prompt-cache-every`). Each card keeps its own part.
- The older helper-GPU caches (`--expert-cache-device1..3`, [SECOND_GPU.md](SECOND_GPU.md)) take the visible GPUs no stage runs on.
  A weak helper can make decoding slower, because its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`); a native (IQ) pack says so at start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024` halves that).
  An explicit `--expert-cache` on the first card is capped to leave room for them.
- More cards pay off when the model's routed experts do not fit the faster ones. An extra slow card costs its own round per
  window and can make a good pair slower.
- **The split pays most when the CPU pool is the bottleneck.** On two identical RTX 2080 Ti 22 GB over a dual-socket Xeon E5-2696 v4 (AVX2 only,
  88 logical CPUs), a plain split moved decode from 35.2 to 51.0 tok/s (+45%) and an 11.6K-token prompt from 672 to 908 tok/s (+35%) at 180 W,
  engine 0.1.49, six 300-token coding prompts. `STRATA_DECODE_TIMING=1` shows why: the window fell from 70.06 to 37.96 ms, almost entirely
  because the CPU pool fell from 9.52 to 0.79 ms per window. The 5080 + 3090 measurement below, where decode with the second card is on par
  with the 5080 alone, was taken over a fast CPU (Ryzen 9 9950X3D). When the CPU pool is what the window waits on, moving half the layers to a
  second identical card removes that work. This is one pair of cards and one CPU, so there is no threshold, and the window breakdown from the
  engine log is not the same statistic as the decode numbers beside it. Setup splits across two cards by itself; with two identical older cards
  on a slow workstation CPU, compare one card against the split and tell us in [#3930](https://github.com/Niko1221/Strata/pull/3930).
- For several conversations at once, `--batch N --batch-groups G --trim-stage-weights` pipelines them through the cards
  ([BATCHING.md](BATCHING.md)).

## History / older measurements

Dated numbers behind the settings above. Each row is one machine; none is a claim about yours.

<a id="measured"></a>

**Layer split, Coder model.** Measured on: RTX 5080 + RTX 3090, Ryzen 9 9950X3D, Coder, 32K context, engine 0.1.20-era
(`bench/results/2026-09-29-layer-split/`).

| Setup | Prompt 16K / 28K tok/s | Decode story / code tok/s |
| --- | --- | --- |
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

A third card (RTX 2080 Ti) made that pair slower: 68 / 90 tok/s decode.

**Option measurements**

| Setting | Measured on | Result |
| --- | --- | --- |
| Unequal pair, `auto` rule | 2x RTX A4000 made unequal, IQ3_XXS, 14.5K prompt, 5 interleaved pairs | prompt 945 -> 1,441 tok/s (+52%), decode 50.2 -> 65.9 (+31%) |
| Unequal pair, hand sweep of `"layer_split"` | 4060 Ti 16 GB + 3060 12 GB, IQ3_S | K=12/18/24/28/32: 582 / 693 / 762 / 802 / 803 prompt tok/s; the old rule chose K=3: 452 |
| x16 + x4 link scaling | 2x RTX 3090, x16 + x4 | K=31: 114-119 tok/s decode, K=27: 130-136 |
| `split_skip_if_fits` | R9700 32 GB + RX 9070 XT, Coder | 4K prompts 1,244 -> 1,776 tok/s, decode ~51 -> ~60; 16K prompts ~5% slower |
| `STRATA_SPLIT_OWN=1` | R9700 + RX 9070 XT | 2K prompts 1,450 and 16K 2,227 tok/s (default 1,265 and 1,950) |
| `STRATA_PREFILL_HELP=1` | 2x RTX 3090, UD-Q4_K_XL, no P2P | prompt tok/s without / with: 1.1K 496 / 723, 2K 878 / 1,128, 3K 1,260 / 1,440; 4K and 8K unchanged |
| `STRATA_PREFILL_HELP=1` | 2x R9700 (16 GB each), Q2_0, PCIe 4.0 x16 | 700 tokens 638 -> 672; 1.5K 839 -> 809; 3K 1,105 -> 929; slower from 1.5K, so it stays off |
| `STRATA_SPLIT_MTP_BATCH=1` | 2x RTX 3080 20 GB, Xeon E5-2696 v4, UD-Q4_K_XL, `--layer-split 23` | prompt read +6.7% at 25K, +4.3% at 51K, +3.3% at 104K; decode not shown unchanged (about 3% lower, within restart noise) |
| `STRATA_STAGE_TRIM=1` | 2x MI50 16 GB, Coder | experts in VRAM 8,819 -> 10,626; decode 39.2 -> 41.7 tok/s |
| Prompt chunk on a split, AMD gfx12 | 2x / 3x / 4x R9700 | 16K prompt +36% / +45% / +53%; 32K +18% / +21% / +29%; 64K +4% / +12% / +17% |
| `STRATA_SPLIT_OVERLAP=1` | 2x RX 7900 XTX, Swift 1.5 IQ3_XXS, 128K, median of 8 | code decode 53 -> 70.1 tok/s (one card ~70); prose 45 -> 55.9 (one card ~68) |
| `STRATA_SPLIT_OVERLAP=1` | 2x RTX A4000, IQ3_XXS | +0.3% story, +2.1% code, -0.3% facts, inside the noise |
| Two Tesla P40, layer split | community report, IQ2_XS (#1028), experimental | decode 19.6 tok/s on one card, 34.8 on both |

**Resident RAM mode on a split.** Measured on: RTX 4060 Ti (layers 0-19) + RTX 5080 (20-47), i9-14900KF, 32 GB RAM, Windows 11,
Swift 1.5 IQ3_XXS, 160K context, four greedy prompts at a time, decode tok/s.

| Setup | First four prompts | After three more rounds |
| --- | ---: | ---: |
| 5080 alone, `--resident-experts` | 25 | 29 |
| split, `--mmap-experts` | 32 | 64 |
| split, `--resident-experts` (22 GiB locked in RAM) | 71 | 69 |

**Prompt chunks for the pipeline (`--prefill-pipe`).** Prompt tok/s, fresh prompts, two interleaved rounds per row.

| Prompt tokens | `--prefill auto` | `--prefill 2048` | `--prefill-pipe 1024` (chunk) |
| ---: | ---: | ---: | ---: |
| 2,013 | 180.9 | 181.4 | 228.0 (1024) |
| 3,865 | 207.6 | 276.5 | 312.4 (1024) |
| 10,041 | 261.1 | 433.0 | 432.6 (2048) |
| 28,998 | 475.7 | 554.1 | 559.7 (3328) |

Measured on: four GP100 dies (PH402 SKU 200), `--layer-split 12,24,36`, Flash-Next IQ3_XXS, 64K context. On 2x RTX 3090 Ti
(Swift 1.5 IQ3_XXS, 1M context) calibration kept it off, and a fixed 1024 chunk read 19-24% slower there and 22-36% slower on
2x RTX 4090 (community report).

**Pipeline windows.** Decode tok/s.

| Measured on | Result |
| --- | --- |
| 2 cards, R9700, code, serial loop -> draft layer on the first card | 103 -> 125 (+21%); 3 cards +19%, 4 cards +22% |
| RTX 4060 Ti + RTX 5080, i9-14900KF, 32 GB, Swift 1.5 IQ3_XXS, 160K, greedy 500 tokens | mean 72.4 -> 84.0 (+16%): Python 90.4 -> 103.1, English prose 63.0 -> 71.1, copy-heavy edit 90.5 -> 117.9 |
| 4x RX 7900 XT, Flash-Next GSQ-RCO IQ3_S, 262K, `--layer-split 12,24,36` | 4K 59.4 -> 67.6; 32K 51.8 -> 65.6 |

**Several settings on one rig.** Measured on: 2x RTX 3090 (250 W, x16 + x4, no NVLink), Ryzen 7 9800X3D, 32 GB RAM, Flash-Next
GSQ-RCO IQ3_S, `--resident-experts`, int8 KV, `--spec 4 --mtp`; decode is the mean of 1,500-token coding replies, prefill one
cold 19.9K prompt; one run per arm (community report #1447, not repeated here).

| Arm | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| setup defaults | 137.0 | 1,798 |
| `--adapt-async 1` | 144.5 | 1,799 |
| `--pipeline-windows 2` | 136.7 | 2,097 |
| `STRATA_PF_FUSED=1` | 137.2 | 1,913 |
| pipeline-windows 2 + async + lag 2 + rotate + pf_fused | 142.0 | 2,302 |
