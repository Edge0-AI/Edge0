# Several requests at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).

**Qwen3.6-35B-A3B** has the same option on its own engine path (`"parallel": N`, `--batch N`, `--batch-mtp`, up to 8 slots, the same
engine protocol and the same server): its numbers, what a slot costs and where setup recommends it are in [QWEN36.md](QWEN36.md).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

```
"parallel": 2
```

With MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the server's
environment) lets each batch slot verify one MTP proposal per window. It is opt-in; without it the batch behaviour
described below is exactly the one without MTP. It needs VRAM per slot for the draft state and buffers, so check the
engine's free-memory log before using it on a smaller card. If it cannot run (one slot, no `--mtp`, a layer split with
`--batch-groups` above 1 or with helper-GPU expert caches or `--remote-expert-opt`, a layer split with two stages on one
GPU) the engine says so and batches as usual. RTX PRO 5000 owners measured +31% to +39% total throughput with 2 to 4
clients on one GPU (a RX R9700 run too).

**With a layer split** (e.g. `"layer_split": "24"` on two GPUs, `"parallel": 2`, `--spec 4 --mtp ...`) it works the
same way, with the slot drafters on the **last stage's GPU** (where the solo drafter and the head are): each slot gets a
drafter that shares the solo drafter's weights and owns only its K/V ring and buffers (the engine logs `--batch-mtp: N
slot drafters on CUDAk (X MiB ... each)` at start; the K/V ring of each also keeps a pinned-RAM copy, as the solo
drafter's does). A window carries two rows per active slot (its token and its proposal) through every stage, so at most
four slots are in one window; more slots (`--batch` up to what fits) rotate through. The windows run through the stages
one after the other (`--batch-groups 1`): on 0.1.41 a layer split with `--batch` 2 or more pipelines the slots in groups
by default, and the pipelined path does not run the slots' MTP drafts, so with `--batch-mtp` and no `--batch-groups` the
engine (0.1.43.1) pipelines and leaves `--batch-mtp` off, and says so (0.1.42/0.1.43 ran one serial group with it: 4 streams on 2x A4000 aggregated 63.9 tok/s against 91.8 pipelined, 85.8 pipelined without `--batch-mtp`; #2038). `--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps `--batch-mtp` with serial windows, which is the better choice for about two streams as measured below; `--batch-groups G` (G above 1) or `--batch-groups auto` given on the command line
keeps the pipeline and turns `--batch-mtp` off. Only a layer split of two stages on two GPUs has been run.

Measured on a layer split of 2x RTX 3080 20 GB (220 W cap) with a Xeon E5-2696 v4, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--spec 4`: two concurrent greedy decodes of 300 tokens, the two streams' tok/s added, median of
the measurements (10 rounds of two streams after each restart of the engine, 20 measurements per restart). The
engine was 0.1.41 with the rest of our production stack on it (the shared KV pool, and the adaptive expert tier
running in batch windows in the first two arms; the pipelined path of the third never adapts in windows), so this is
`--batch-mtp` on top of that, not this change alone on main.

| arm | aggregate tok/s | restarts, measurements | per window |
| --- | --- | --- | --- |
| `--batch-mtp`, one group | **89.2** | 3, 60 | 3.3 rows, 34.6 ms, 66-67% of the proposals accepted |
| no `--batch-mtp`, one serial group | 77.3 | 2, 40 | 2.0 rows, 23.7 ms |
| no `--batch-mtp`, `--batch-groups` unset (the 0.1.41 default: two pipelined groups of one slot) | 77.7 | 1, 20 | one row per group step |

That is +15.4% for `--batch-mtp` against serial windows without it (95% bootstrap interval of the ratio of medians
+12.5% to +17.9%) and +14.8% against the pipelined default; the two arms without it tie. One stream alone decodes the
same with and without it (81.7 and 82.0 tok/s), and the two slot drafters cost 35 to 110 expert-cache slots on the
last GPU (50 MiB of private state each).

The pipelined default is better in one place: a new prompt of about 48k tokens arriving while one stream decodes was
read at 2508 tok/s with pipelined groups and 1217 tok/s in serial windows, and the decoding stream finished sooner
(25.0 against 18.0 tok/s over its whole run); `--batch-mtp` does not change that. So it pays where concurrent decodes
dominate; with long reads arriving beside decodes the pipelined groups can be the better choice. Not measured: more
than two stages (the report on #1253 found no gain at four stages), more than two slots, sampled decoding, an
upstream-main binary (the pipelined arm is our build, started with the flags 0.1.41 resolves to).

Our measurement of `--batch-mtp` on a layer split with 2x R9700 and all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM.

Earlier, on engine 0.1.40.3 with this change and the same two cards, two concurrent streams: UD-Q4_K_XL 39.9 to 43.5
tok/s without and 44.5 to 51.3 with `--batch-mtp` (2 runs per arm), and a small Coder IQ1_M test model with all
experts in VRAM 84.0 without and 101.2 with it, the greedy text of both streams identical.

Text: with the same expert cache on both arms (the slot drafters take VRAM, so the cache differs unless the
reserve is adjusted) and `--pcie-frac 0`, `--adapt-every 0`, the greedy text with `--batch-mtp` was equal to the text
without it for 5 short prompts in all 6 comparisons (one restart per arm). That is text equality, not a bit-exactness
proof, and a 25k-token prompt read while the other stream decodes was not repeatable between two identical streams
of the same arm.

With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8 normally; with a layer split up to 8 per group, 16 in a HIP build, see below) | up to N conversations have batch slots; more requests wait for a free slot. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. With grouped MTP, more than 8 slots can rotate through eight-row windows if memory permits. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the default on a layer split since 0.1.41 (give no `--batch-groups`): the engine pipelines one group per GPU stage (the most that divide the slots; 8 slots on 4 GPUs = 4 groups of 2) and says so (`INFO batch_groups=G`). `--batch-groups 1` turns it off (all slots in one window, GPU after GPU). Measured, 8 clients, total tok/s against one group: 4 x R9700 166 against 86, 2 GPUs 109 against 78, 3 GPUs 110 against 78. With `--adapt-async 1` and no `--batch-groups` the default is one group (the adaptive tier between batch windows does not run in pipelined groups) and the log says so; with `--batch-mtp` and no `--batch-groups` the pipeline wins and `--batch-mtp` is left off (`--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps it); `--batch-groups G` or `--batch-groups auto` given with `--batch-mtp` pipelines the groups and turns `--batch-mtp` off, and given with `--adapt-async 1` pipelines the groups and turns `--adapt-async` off (the pipelined path does not adapt). | PROPOSED for 0.2.1 (owner decides; branch f-021-split): on NVIDIA builds with exactly two stages the default is one group instead (2x RTX A4000, IQ3_XXS, aggregate tok/s against the pipelined default: +16% at 2 slots, +12% at 4, +15% at 8, 5 of 5 pairs each; `--batch-groups auto` or a number asks for the pipeline, `STRATA_BATCH_GROUPS_NV_SERIAL=0` restores the old default).
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The engine never refuses a count it cannot run: it says so in its log and runs what it can - at most 8 slots without a
layer split (a window holds 8 rows; with a layer split and pipelined groups, 8 per group, 16 in a HIP build: see "Windows of 16 rows" below), as many as fit in VRAM, or none (one request at a time) when not two fit. The server reads
the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

## More than 8 slots on a layer split, and the opt-in admission and decode switches

(From havok2-htwo's patch series, PR #2364.)

**More than 8 slots.** With a layer split a pipelined window holds at most 8 rows, but each group of the pipeline is its
own window, so `--batch 16` on 4 GPUs runs 4 groups of 4 slots (`--batch-groups auto`), `--batch 32` runs 4 groups of 8.
Until 0.1.44.1 the engine cut such a request down to 8 slots and said so; it now runs what is asked for, as many slots
as fit in VRAM and at most 8 per group, 16 in a HIP build (`INFO batch_slots=N`; "Windows of 16 rows" below). `--batch-mtp` keeps its limit (rows per window are
doubled by the proposals). Nothing changes for 8 slots or fewer. Every slot of a 16-slot pipelined batch decodes
exactly the tokens it decodes alone (`tools/batch_test.py --n 16`, which now takes up to 32 prompts).

**Opt-in switches** (environment of the engine, i.e. `"env"` in the config; none is on by default). Each one changes
how the work is scheduled or copied, not what is computed; `tools/batch_test.py` checks that every slot decodes the tokens it
decodes alone.

| Switch | What it does |
|---|---|
| `STRATA_SLOT_COPY_DIRECT=1` | an admission (and a slot's resume) copies the running state and the QSA K/V device to device, instead of through host images |
| `STRATA_SLOT_COPY_BATCHSYNC=1` | with the above: queue the ~116 copies and synchronize each device once |
| `STRATA_SLOT_CKPT_MOVE=1` | the admission's turn checkpoint moves into the slot (no 112 MB copy); it also keeps the dropped checkpoints' storage, as `STRATA_CKPT_POOL` does; a session SAVE or parked conversation taken right after an admission lacks the moved checkpoint (the slot holds it) |
| `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1` | a checkpoint is saved into the vectors of a dropped one; a split's stage parts are saved at once |
| `STRATA_ADMIT_MERGE_FIRST=1` | an admission reads its last prompt tokens and its first token in one window (all experts resident only) |
| `STRATA_PIPE_BYIELD_NODRAIN=1` | a BYIELD that reaches a pipelined batch is dropped without draining the groups |
| `STRATA_BATCH_SELF_COMMIT=1` | a one-row-per-slot window advances the GDN state itself, no commit graph |
| `STRATA_QSA_SLOTS=1` | a window's QSA block scores, top-k and attention run as one launch over its rows |
| `STRATA_HANDOFF_PIPE=P` | a layer split hands a prompt chunk to the next card in P pieces (2..16), gated by events, on a persistent thread |
| `STRATA_ADMIT_SHORT_FIRST=N` (server) | a new prompt of at most N tokens takes the control lines before longer ones (a longer one that waited `STRATA_ADMIT_MAX_WAIT_S`, default 30, counts as short) |
| `STRATA_TRACE=1` | one line per admission with its phases (drain, prompt, first window, slot copy, checkpoint copy) |
| `STRATA_MMQ_DIRECT=1`, `STRATA_MMQ_GROUP_SORT=1` | CUDA, needs the build option `-DSTRATA_MMQ_TAB=ON`: the prompt MMQ reads resident experts in their cache slots; full MMQ groups sorted by row count (docs/DETAILS.md) |

Measured (4x R9700 32 GB, HIP, IQ3_S all experts resident, layer split 12,25,37, 3 interleaved rounds per arm, 0.1.44.1
with 8 slots against the same build with 16 slots and the switches above; aggregate output tok/s, TTFT p50 in seconds):

| users (prompt) | 0.1.44.1, 8 slots | 16 slots, no switch | 16 slots, all switches |
|---|---|---|---|
| 8 (147 tokens) | 209 / 2.70 s | 209 / 2.72 s | 251 / 1.82 s |
| 16 (147 tokens) | 208 / 20.6 s | 252 / 5.4 s | 324 / 2.0 s |
| 16 (1.2K tokens) | 145 / 23.9 s | 167 / 10.6 s | 204 / 5.3 s |
| 16 agents x 3 turns, follow-up TTFT | 50 tok/s / 21.2 s | 53 / 7.1 s | 56 / 5.7 s |

One user alone is unchanged (73-74 tok/s). Each slot's session takes VRAM from the expert cache, so on a card or split
where the experts do not all fit, 16 slots cost the single-user speed (see docs above on what a slot costs).


### What a slot costs, and what setup recommends

Every slot's session takes VRAM that the expert cache would otherwise hold: 0.56 GiB at a 32K context with 8-bit
KV, more with a longer context unless the KV cache streams (`--kv-resident`: then only the attention's 32K window
stays in VRAM, and each slot's whole KV cache takes pinned RAM - 1.6 GB at 128K). On a card whose experts mostly run
on the CPU, a batch also reads about as many distinct experts as the requests one by one (different conversations
route to different experts), so the gain is in **latency** (nobody waits for a whole answer), and a request alone
runs slower (the smaller expert cache): 11-24 % on a 12 GB card, see the measurements below.

So setup recommends `"parallel"` **only where the experts mostly fit in VRAM**: the expert cache (each card's VRAM
less ~5 GB, every card of a layer split counted) must still hold at least half of the model's experts beside the
slots, and the slots may take at most a fifth of it, up to 4 slots. With Q2_0 at 32K that is 3 slots on a 24 GB
card, 4 from 32 GB or on a split such as 2 x 16 GB; IQ3_S needs 32 GB or a split. Everywhere else (any 12 or 16 GB
card alone) it stays at one at a time and setup says: "parallel N reduces waiting for several users but costs
about 10-25% speed per request on this card". `--parallel N` is honoured as asked either way.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt
  plus what it generated so far - the engine's prompt cache holds exactly that, so nothing is read again - and the
  new request is admitted next to it. By default, a request in a slot decodes **without MTP drafts** (one token per window).
  With `--batch-mtp`, each slot verifies one MTP proposal alongside its current token.
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path: the slot is
  stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with
  `--prompt-cache 0` it stays in the slot; `STRATA_PARALLEL_SOLO=0` turns it off). The draft layer's own K/V was
  built for another conversation then, but measured it accepted as many drafts (140 of 172) as a draft layer that
  read the conversation (140 of 173).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or
  decoding, its tokens and tok/s; `live.running` the requests in flight).
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time, and **the slots decode between the prompt's chunks**: after each chunk (`--prefill`,
  2048-8192 tokens) they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so
  a long prompt slows the others down instead of stopping them. The chunks are the ones one uninterrupted read
  takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the
  checkpoint at the prompt's last turn boundary); the next turn of that conversation goes to that slot and the
  engine copies its state back (50-60 ms for a short conversation) instead of reading the history again - also for
  a client that drops the reply's thinking from the history (the checkpoint matches up to the new turn). A new
  conversation takes an empty slot, else the one used longest ago.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

### Long prompts that move from solo to a batch slot

The handoff above needs prompt-cache state. With `--prompt-cache 0`, moving a long request from the solo path into
a slot can read the whole prompt again. This minimal example enables handoff reuse without parking completed
conversations. Set these options in the model config's `args`, keeping the other arguments unchanged:

```
"args": [ ..., "--prompt-cache", "1", "--prompt-cache-every", "0",
          "--prompt-cache-root", "0", "--conversation-cache-mib", "0" ]
```

`--prompt-cache 1` keeps the latest turn-boundary checkpoint so the `GEN` to `BGEN` handoff can reuse the live
prefix. `--prompt-cache-every 0` disables periodic checkpoints and `--prompt-cache-root 0` disables an extra root
checkpoint. The limit is per cache set: a slot can keep its own live state, so it is not one checkpoint shared by
every slot. `--conversation-cache-mib 0` still disables parking independent completed conversations in RAM; it does
not remove the prompt checkpoint or the state held by an active slot.
If conversation parking is already enabled, keep its existing budget instead of setting it to zero: that budget is
independent of the handoff configuration.

This setting has RAM costs. The checkpoint and every slot's normal context state still consume memory, even with the
conversation cache off. Check the engine's free-memory report and the slot count that it actually accepts before
raising the context length or `"parallel"` value.

It does not make independent cold prompts prefill in parallel. The server admits them one at a time; a running value
of four can mean four requests are in flight, not four slots decoding together. A slot already decoding may make
progress while the next prompt is read, but long independent prompts can finish before enough admissions overlap to
fill every configured slot.

After changing the setting, send a long request alone and add a second request while it is decoding. In the engine
log, the first request's `GEN` to `BGEN` transfer should report a large reused prefix and only the new token(s) read.
Then check `/metrics` for the slots that were actually reading or decoding, and `/v1/status` for the effective
serving count. Treat this as handoff evidence only: it does not prove that every configured slot can stay resident or
that cold prompts have concurrent prefill.

## Reading a prompt beside pipelined groups (experimental, opt-in)

With a layer split and `--batch-groups` above 1 a prompt is read in one run: every decoding slot stops until it is
read, however long it is (#1906). One GPU does not have this problem: there the slots decode between the prompt's
chunks for `STRATA_BATCH_DECODE_SHARE` x the chunk's time.

`STRATA_PIPED_CONCURRENT=1` reads a prompt on a thread of its own while the slot groups keep decoding through the
pipeline, instead of stopping them: the prompt's chunks and the slots' windows run on the same GPUs at once, on their
own streams (the windows' at the device's highest priority on CUDA; `STRATA_PIPED_CONCURRENT_PRIO=0` keeps the default),
so the slots never wait for the read and the read is one run with no drain inside it.

- The prompt's K/V goes to the main sessions and the slots' to their own, so they do not touch each other's state. The
  n-gram (PLE) table both of them read is locked. The adaptive VRAM tier stands still during the read: its round lands
  before the read and none starts while it runs.
- The VRAM expert slots the prompt path borrows are lent before the read and refilled after it, with nothing in flight;
  in between the slots' windows compute those experts on the CPU.
- The slots' doorbells are served on a thread of their own during the read, which makes no CUDA call. A CUDA call can
  wait for the driver's lock while its holder waits for the GPU - a free, or the module the first kernel of a new shape
  loads - and a window on that GPU waits for its doorbell (issue #31's pattern: two such hangs were found and read from
  the engine's stall dumps while this was built). The MTP drafter's prompt buffers and the prompt path's pinned upload
  slots are sized before the read, so its thread allocates and frees nothing. Between windows the doorbell thread waits
  for the next launch, which wakes it (a timed sleep is a whole timer tick on Windows, up to 15.6 ms). A window that
  will not finish is still found (its graph ended with a layer still to ring, or no layer rang for 20 s), and the stall
  watchdog stays on for the whole read.
- A waiting shorter request takes over at the read's next chunk (`BYIELD`), also when no slot decodes: the run ends
  there, the pipeline is drained and the part read waits in a slot, as off the pipeline. A read nobody waits for is not cut.
- Off, said once, where the read's thread would allocate or free (`STRATA_PREFILL_HELP=1`, `STRATA_SPLIT_MTP_BATCH=1`),
  where serving a doorbell makes a CUDA call (`--pcie-mode dma`, `STRATA_FS_SLOTS`, the helper caches
  `--expert-cache-device1..3`), and with an expert source that hands out buffers it recycles (the GGUF in place, an
  unbuffered source, `STRATA_IO_PREFETCH`): the experts come from the RAM arena or the mapped file. A prompt with
  pictures is read the usual way.

One log line per read (`strata batch: read N tokens on a thread of its own in X ms (concurrent, to Q of T); the slots
decoded R rows meanwhile (S rows/s; stage k: W windows, H ms host each)`). Built for CUDA and HIP (HIP keeps the default
stream priority, and is not run on a GPU here); SYCL does not have it.

Measured on 2x RTX 5090 (Swift 1.5 IQ3_S, split 25, `--batch 6 --batch-groups 2`, resident experts, k8v4, 262K,
`--kv-resident 32768`, the engine-default decode share): main (`a0116b27`) against this mode, a fresh engine per arm in
the order main, this mode, this mode, main (`tools/piped_read_join_bench.py`; the fixed-prompt rows send the same four
stream prompts every run, so the streams' answers and work are the same):

| | main | `STRATA_PIPED_CONCURRENT=1` |
| --- | ---: | ---: |
| 64K + 32K cold prompts beside four streams: longest stop of a stream | 11.9 s | 0.77-0.84 s |
| ... the streams' tokens meanwhile | 289-291 | 1,411-1,679 |
| ... the prompts' TTFT | 6.8 / 5.8-5.9 s | 8.1-8.2 / 8.2-8.3 s |
| the same with fixed stream prompts: the streams' rate while the prompts are read | 2-3 tok/s | 85-92 tok/s |
| ... the streams finish at | 26.1-27.1 s | 24.6-25.3 s |
| ... GPU utilization while the prompts are read | 75-86% | 85-91% |
| a one-line question 1 s into a cold 45K read, nothing decoding: its TTFT | 5.7-5.9 s | 1.9 s |
| ... and the 45K prompt's own TTFT | 6.2-6.4 s | 7.4-7.5 s |

The joining prompts pay for the GPU time the streams now get. Over a whole run with the same work, the streams
finished 5-8% sooner (6.4% on average); with prompts that change from run to run (the bench's), the run's token rate
varies more than that between runs. Between windows the doorbell thread first slept 50 us, which is a whole timer
tick on Windows: the same build with `STRATA_TEST_BELL_SLEEP_US=50` gave the streams 64-67 tok/s during the reads
instead of 85-92.

The same randomised load (`soak`: three streams, cold joins of 4-64K, short questions, follow-ups of earlier joins,
clients leaving mid-read and mid-answer; 5 minutes, the same seed) on main and in this mode: 167 and 180 requests
answered; the longest gap inside a stream 9.8 and 1.0 s (90th percentile 8.5 and 0.7 s); the joins' TTFT median 3.8
and 4.6 s. A request that comes during a read still waits for it, unless its prompt is under half as long (`BYIELD`).

A 16 GB card on each side (`--vram-reserve-mib 16600 --kv-resident 16384`, the same flags otherwise):

| | main | this mode |
| --- | ---: | ---: |
| fixed stream prompts, 64K + 32K joining: the streams' rate while the prompts are read | 0.9 tok/s | 52.9 tok/s |
| ... the streams finish at | 70.6 s | 63.8 s |
| ... the prompts' TTFT | 9.5 / 10.1 s | 15.8 / 21.1 s |
| a one-line question 1 s into a cold 45K read: its TTFT | 7.3 s | 3.8 s |
| the 45K prompt read alone | 7.33 s | 7.25 s |

A weaker host as well (16 GB cards, the experts from the mapped file, AVX2 expert kernels, 6 pool workers), in #1906's
shape (`--batch 2 --batch-groups 2`): one stream decoding while another's cold prompt is read:

| cold prompt | main: the stream's longest stop / tok/s meanwhile / the prompt's TTFT | this mode |
| --- | ---: | ---: |
| 64K | 11.1 s / 2.4 / 10.9 s | 0.41 s / 25.7 / 11.7 s |
| 127K | 20.3 s / 1.5 / 19.9 s | 0.64 s / 32.7 / 20.7 s |

So the cost is the joining prompt's: +4-7% beside one stream, +20-45% beside four on these cards, and +67-109% beside
four on 16 GB cards, where the streams' windows take a larger share of the GPU.

**Checked:** with the variable unset this build gives main's tokens (six greedy answers' SHA-256 equal in two engine
starts each, `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`) at main's speed. In this mode, 25 minutes of the
randomised load above (also on `--batch 2`): about 760 requests, 340 concurrent reads and 88 give-ways, with no hang, no
error and no stall report.

**Text equality** (`batch_interleave_test.py --long 40000` with its exact settings, `--batch 4 --batch-groups 2`, no
conversation cache): this test is not reproducible run to run on this PC even for main (A differed from its solo tokens
at token 142 in one run of two). In this mode every request equaled its solo tokens in two runs of three - D giving way
at its first chunk (`YIELDED 2 2048`), E admitted meanwhile, D going on and A going back to the solo path included,
which main does not do on this path (D never gives way there); the third run had B off at token 74, a near tie that
flips on main as well. Greedy only; other GPU pairs, three or more stages and sampled decoding not measured.

## A prompt read beside pipelined groups (opt-in)

With a layer split and `--batch-groups` above 1 a prompt is read in one run: every decoding slot stops until it is read,
however long it is, `STRATA_BATCH_DECODE_SHARE` does not apply and no `BYIELD` is taken (`batch_read_piece` returns 0
for the pipelined groups; #1906). One GPU does not have this problem: there the slots decode between the prompt's chunks
for `STRATA_BATCH_DECODE_SHARE` x the chunk's time. These variables give the pipelined groups the same choice:

- `STRATA_PIPED_READ_PIECE=N`: a prompt read on the pipelined groups goes in pieces of N tokens, rounded down to whole
  chunks (at least one, at most 32), so a piece reads exactly the chunks a single run would, the two stages stay
  overlapped inside it and only the piece's end drains the pipeline. Between two pieces the slot groups decode through
  the pipeline for `STRATA_BATCH_DECODE_SHARE` x the piece's time, at most `STRATA_PIPED_DECODE_MAX_MS` and at least
  `STRATA_PIPED_DECODE_MIN_MS`, and every group in flight finishes before the next piece, as before any prompt read.
- At a piece's end the pipeline is drained, so a waiting request with a prompt under half as long can take over there
  (`BYIELD`, as between the chunks of a read off the pipeline): the part read waits in a slot and the read goes on from
  it after the short request is in. The pieces are also taken when no slot decodes, for this reason (off the pipeline
  the same read goes in `STRATA_BATCH_YIELD_CHUNKS` pieces then).
- A read that starts within a second (plus the decode budget) of the previous such read's end - prompts arriving back to
  back - first gives the slots one decode budget (the same rule, from the previous read's last full piece); otherwise
  the slots would stop for both reads.

With the variables unset the read is one run as before. One log line per piece (`strata batch: piped read N tokens in X
ms (to Q of T)`, with `, then the slots decoded R rows in Y ms` when slots decode), per give-way (`the prompt read gives
way at Q of T tokens; slot S holds it`) and per lead-in (`a read right after a read: the slots decoded R rows in Y ms
first`).

**What it trades.** Every piece ends by draining the pipeline, so a long read takes longer - also on an idle server,
since the pieces are what lets a short request in - and the decoding streams get tokens where they had none. It is a
latency choice, not a throughput gain: leave it off where throughput or the long prompt's own TTFT matters most, use it
where a stream that stops for several seconds, or a short question stuck behind a long read, is worse than a long prompt
that waits a little longer.

Measured on 2x RTX 5090 (Swift 1.5 IQ3_S, `--layer-split 25 --trim-stage-weights --batch 6 --batch-groups 2
--resident-experts`, k8v4 KV, 262K context, `--kv-resident 32768`, `STRATA_ARENA_PIN_GIB=24`, Windows 11; a CUDA 13.0
sm_120 build of this change on 0.1.43, "release" = 0.1.43 built the same way; a fresh engine for every run, all from one
window, `tools/piped_read_join_bench.py`). Four streams decode greedy coding answers (thinking off, up to 2048 tokens)
when cold prompts of this repository's own source text join. "Longest stop": the longest time any stream received no
token, from the first prompt's arrival to the last one's first token + 1 s (client-side, SSE); "stream tokens": what the
four streams received in that window; "token rate": the streams' tokens over the run's wall time.

Two cold prompts, 64K then 32K, arriving 6 s apart (`--scenarios two_joiners`; release and 32K pieces: median of 5 runs
each, interleaved; 16K pieces: one run):

| arm | prompts' TTFT | longest stop of a stream | stream tokens | token rate |
| --- | ---: | ---: | ---: | ---: |
| release (one run per prompt) | 6.9 s / 6.1 s | 12.2 s | 289 | 173 tok/s |
| `PIECE=32768 DECODE_MAX_MS=400` | 8.1 s / 8.5 s | 4.2 s | 527 | 166 tok/s |
| `PIECE=16384 DECODE_MAX_MS=400` | 10.5 s / 12.0 s | 2.6 s | 793 | 154 tok/s |

The longest stop was 12.16-12.23 s for the release in all five runs and 4.19-4.22 s for 32K pieces in four; in the fifth
the first 32K piece of the 64K prompt took 9.3 s instead of 3.95 s (one uninterrupted run before any slot decoded),
which made that run's stop 9.5 s. The token rate varied 169-180 tok/s (release) and 136-174 tok/s (32K) between runs.

A short request behind a long read, nothing decoding (#1504's test, `--scenarios followup,read_alone --repeat 3`, median
of 3): a one-line question sent 1 s after a cold 45K prompt, and the same 45K prompt read alone:

| arm | the question's TTFT | the 45K prompt's TTFT | the 45K prompt alone |
| --- | ---: | ---: | ---: |
| release | 5.4 s | 6.0 s | 5.7 s |
| `PIECE=32768 DECODE_MAX_MS=400` | 3.6 s | 7.3 s | 6.4 s |
| `PIECE=16384 DECODE_MAX_MS=400` | 1.8 s | 8.0 s | 7.1 s |

With the variables unset this change ran as the release in the same window (the 45K read alone 5.66 vs 5.69 s, the two
joiners' longest stop 12.2 s on both) and gave the same SHA-256 for six greedy answers under `--prompt-cache 0
--adapt-swaps 0 --pcie-frac 0`. For reference, one RTX 5090 (the same model and flags without the split, 6 slots) with
one cold 64K prompt joining four streams: `STRATA_BATCH_DECODE_SHARE=2.0` 2.4 s longest stop and 37.2 s TTFT, `=0` 14.0
s and 13.8 s. Pieces of one chunk (`PIECE=8192`) cost the long prompt the most. Not measured: other GPU pairs, three or
more stages, HIP and SYCL (built for CUDA only), sampled decoding, prompts with pictures.

## The VRAM expert tier keeps adapting in batch windows

Batch windows count which experts they route to, and every `--adapt-every` windows (default 4) the engine swaps the
most-routed experts that are not in VRAM into the cache in place of the least-routed resident ones, the same rule the
solo path uses (`--adapt-swaps` and `--adapt-decay` apply too; on a layer split every GPU's cache adapts). It runs when the
tier is not the whole model and both `--adapt-every` and `--adapt-swaps` are above 0. By default the round runs beside the
window's commit and drafts and has landed before the next window starts. It stands still while a long prompt is read
between decoding windows (the prompt may hold a loan of cache slots). The `strata batch:` log line shows the rounds and swaps, each stage's
GPU-reach wait and pool time, and how many routed entries the VRAM tier, the PCIe share and the CPU served per window.
Swaps move experts between the GPU and the CPU, which round differently, so greedy outputs can differ from run to run;
`--adapt-every 1000000` keeps the tier fixed (as in the exactness settings below). The pipelined `--batch-groups` path
does not adapt by default: its windows count the routing, but no round starts while groups are in flight, so the tier
moves only when the slots go back to the solo path.

`STRATA_PIPE_ADAPT_EVERY=N` (opt-in; unset or 0 keeps the old behaviour) lets the pipelined path adapt: after N group
windows it starts no new group, lets the groups in flight finish, runs one round as the serial batch loop does (nothing
in flight reads a cache slot, and the swapped-in experts land before the next launch), and goes on. It stands still while
a prompt is read beside the slots, and does nothing with `--adapt-async 1`. The engine says so at start, and the
`strata batch (pipelined ...)` line adds the rounds, the experts swapped in and the time spent. Measured on 2x RTX 5090
(layer split 25, Swift-Qwen3.8-Flash-Next IQ3_S, `--batch 6 --batch-groups 2`, resident experts, a profile-filled cache,
`--adapt-every 16`), six streams of 1,024 greedy tokens from `tools/serve_load.py`, the engine's rows/s of eight bursts
right after start, two fresh engines per setting:

| `STRATA_PIPE_ADAPT_EVERY` | bursts 1-8, rows/s | mean | experts swapped before burst 8 |
| --- | --- | ---: | ---: |
| unset (stock 0.2.4.1) | 246, 257, 291, 292, 335, 338, 356, 355 | 309 | 287 |
| unset (this branch) | 249, 242, 297, 296, 333, 336, 372, 362 | 311 | 288 |
| 128 | 363, 415, 414, 420, 420, 411, 415, 420 | 410 | 6,181 |
| 128 | 359, 407, 416, 422, 419, 417, 404, 411 | 407 | 6,294 |
| 32 | 375, 405, 408, 405, 409, 390, 405, 409 | 402 | 5,957 |
| 32 | 384, 414, 404, 402, 401, 412, 406, 401 | 403 | 5,678 |

Without it the tier swaps 96 experts every burst or two (only when a burst ends) and the engine was still speeding up
after eight bursts; with 128 it reached about 415 rows/s from the second burst, the speed an engine without the switch
showed only after thousands of swaps had come from other traffic. 32 empties the pipeline four times as often and ends a
little lower. One machine, one model and one configuration; not measured on one GPU (no pipelined path there) or with
more than two stages.

With `--adapt-async 1` (needs the resident RAM mode) the round does not hold a window up: each batch window moves the
round on one step (table changes and uploads on the main thread, the copies on a helper thread and the cards' refill
streams), as on the solo path. A request that arrives finishes the round in flight before it starts, and a prompt read
between decoding windows does not move it. `--adapt-min-gain F` (default 1.5) raises the bar a swap must clear (the
routing count of the expert coming in against the one going out), so fewer experts move per round; both tiers use it.
The log line's `adapt wait` is the time the main thread spent on the tier per window (the blocking tier: waiting for
its round).

Measured with the same setup as the `--batch-mtp` numbers above (2x RTX 3080 20 GB, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--batch-mtp`, engine 0.1.41 with our production stack; two concurrent greedy decodes, the streams'
tok/s added, median of 20 measurements per restart):

| tier | aggregate tok/s | restarts | per window | routed entries served from VRAM |
| --- | --- | --- | --- | --- |
| `--adapt-async 1 --adapt-every 2` | **89.2** | 3 | 34.6 ms, adapt wait 0.2 ms | 92.1% |
| blocking, `--adapt-every 2` | 74.7 | 1 | 41.4 ms, adapt wait 8.4 ms | 92.3% |
| none, the cache frozen from a profile the tier had learned on the same prompts | 69.5 | 1 | 43.6 ms | 77.2% |
| none, `--adapt-every 0` | 45.2 | 2 | 66-68 ms | 52.7% |

The switch changes the tier in the solo path as well (there is no switch for batch windows alone), so the table
compares the tier as a whole, not "batch windows only": the solo decode speed moves the same way (81.7, 66.3, 69.0 and
43.1 tok/s). The prompts are the benchmark's own, eight topics that repeat, which the tier learns quickly, so on a
more varied workload the gain may be smaller (not measured). Upstream main runs no tier in batch windows, and this
was not measured against an upstream binary.

## Two NVIDIA cards: the pipelined groups are the default (0.2.7)

With a layer split over exactly two NVIDIA cards, `--batch` and no `--batch-groups`, 0.2.1 to 0.2.6 ran one serial group
(measured then on two RTX A4000, IQ3_XXS: +12 to 16% against the pipelined groups). From 0.2.7 the default is the pipelined
groups, like on three or more cards. `STRATA_BATCH_GROUPS_NV_SERIAL=1` or `STRATA_DEFAULTS=026` gives the serial group back;
`--batch-groups 1` asks for one group, `--batch-groups auto` or a number asks for the pipeline, and an explicit
`STRATA_BATCH_GROUPS_NV_SERIAL=0` wins over `STRATA_DEFAULTS=026`.

Why: with the 0.2.7 code the pipelined groups were faster at two and four streams on every box and quant measured
(`tools/serve_load.py`, `--batch 4`, 256 tokens per request, interleaved pairs, total tokens per second, pipelined / serial):

| box, model | experts in VRAM | 1 stream | 2 streams | 4 streams |
| --- | --- | ---: | ---: | ---: |
| 2x RTX A6000 48 GB, IQ3_S | all | 101.7 / 101.7 | 125.7 / 107.4 (+17%) | 171.4 / 141.0 (+22%) |
| 2x RTX A4000 16 GB, Q2_0 | 17037 of 24576 pairs | 63.6-68.7 / 68.2-70.3 | 80.2-82.5 / 75.0-75.5 (+7 to 9%) | 110.8-111.5 / 94.8-95.3 (+17%) |
| 2x RTX A4000 16 GB, IQ3_XXS | 13056 of 24576 pairs | 58.4-60.5 / 60.7-62.6 | 73.2-73.7 / 65.8-66.5 (+12%) | 100.5-100.7 / 82.9-83.6 (+20%) |

The pipelined side was faster in every pair at two and four streams (3 pairs on the A6000, repeated in 3 more with the v0.2.6 binary; 4 clean pairs on the A4000
with Q2_0, 3 on the A4000 with IQ3_XXS). The v0.2.6 binary measured the same on the A6000 (serial default 104.9 and 137.2
tok/s at 2 and 4 streams, `--batch-groups auto` 124.6 and 169.6). A single stream runs on the solo path in both; on the A4000s
the serial setting was about 4% faster for it (3 of 3 pairs with IQ3_XXS), on the A6000s it was equal. The 0.2.1 result
(serial faster) did not repeat. Every slot's tokens equal its solo decode with the default (4 and 2 slots, greedy and seeded
sampled streams, finishing slots, a cancelled slot).

## MTP drafts in the pipelined groups (`STRATA_PIPED_MTP`, opt-in)

The pipelined groups run no drafts. `--batch-mtp` with serial windows (`--batch-groups 1`) keeps one card busy at a
time. `STRATA_PIPED_MTP=N` together with `--batch-mtp` gives both: the pipeline stays, and each slot drafts in it.
Off by default. Without the variable nothing changes.

How it works. A group's window carries, for each of its slots, the slot's token and up to N MTP proposals (at most
`--spec` - 1; the first always, each further one while the draft layer was at least `--spec-min-p` sure of the one
before). The last stage judges each slot's proposals on their own. How many rows to keep is known only then, and by
then the earlier stages already run other groups' windows, which reuse the scratch the commit replays from. So a
small save graph copies those rows right behind each window (`Verifier::batch_launch_rows`), and after the verdict every
stage queues the commit of the kept rows (`Verifier::batch_commit_rows`). The slot's drafter catches up on its own rows
and drafts the next proposals as an asynchronous chain (`MtpDrafter::chain_launch`). A group whose chain has not landed
lets another group start first. Anything that reads a slot's state (an admission, a parked conversation, the solo path
taking a slot back) first drains the pipeline, which also waits for the queued commits.

It needs a layer split with two or more groups, any number of slots per group (as long as a group's window,
slots x (1 + N), is at most 8 rows). When it cannot run, the engine says why and goes on without it: fewer than two
groups left, `STRATA_BATCH_SELF_COMMIT` on, or a window of more than 8 rows.

Measured on a box with 2x RTX A6000 48 GB (`--layer-split auto`), Qwen3.8-Flash-Next IQ3_S, every expert resident in
VRAM, `--batch 4 --batch-groups auto` (2 groups of 2 slots), `tools/serve_load.py`, 256 tokens per request, reasoning
off. Total output tokens per second: the median of the pairs, each pair the median of 3 bursts on a fresh engine start. The base arm
(pipelined groups, no drafts) and `STRATA_PIPED_MTP=2` were run in 5 interleaved pairs, the other arms in 3 pairs. "Default" is
`--batch 4` with no `--batch-groups` as it ran in 0.2.1 to 0.2.6 on exactly two NVIDIA cards: one serial group (see the next section):

| streams | pipelined, no drafts | `=1` | `=2` | `=3` | serial `--batch-mtp` (`--batch-groups 1`) | default (no flags) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (solo path) | 101.7 | 101.4 | 101.5 | 103.0 | 101.5 | 101.7 |
| 2 | 125.7 | 146.3 (+16.0%) | 145.1 (+15.4%) | 138.4 (+10.0%) | 124.0 (-1.3%) | 107.4 (-14.8%) |
| 4 | 171.4 | 188.8 (+10.2%) | 179.5 (+4.7%) | 165.6 (-3.4%) | 139.9 (-18.5%) | 141.0 (-17.8%) |

Gains are the median of the per-pair ratios. `=2` was faster than the base in 5 of 5 pairs at 2 and at 4 streams, `=1` in 3 of 3
pairs at both, `=3` in 3 of 3 at 2 streams and in none at 4. With `--batch 2` (2 groups of 1 slot) and 2 streams `=2` gave 146.8 against 125.7 tok/s (+16.4%, 5 of 5 pairs). One
stream runs on the solo path in every arm and does not change. About 55% of the checked proposals were kept at `=2`
and about 67% at `=1`. The acceptance falls as the window grows, which is why `=1` or `=2` is the choice and `=3` is not.

Best setting: `=1`. On the A6000 box `=1` gained the most at four streams and as much as `=2` at two; on the 2x RTX A4000 box
(Q2_0 pack, the experts do not all fit in VRAM, `--batch 4`, 4 clean pairs, total tok/s, `=1` / `=2` / no drafts) it gave
2 streams 96.7, 96.5, 96.3, 95.7 / 95.0-95.9 / 80.2-82.5 (+18 to 20%, +17 to 19% for `=2`) and 4 streams 117.8-120.0 / 99.7-105.8 /
110.8-111.5 (`=1` +6 to 8% in 4 of 4 pairs, `=2` slower than no drafts in 4 of 4). One stream was unchanged. More than four streams was
not measured, and why four streams gain less than two is not explained. Cost: the slot
drafters and a save buffer per stage (25.5 MiB at `=1` and 38.2 MiB at `=2` with 4 slots), taken from the expert cache.
On the A6000 box the cache stayed fully resident. On the A4000 box, where the experts do not all fit, `--batch-mtp` took
165 of 6959 cache slots on the second card; that changes which experts are in VRAM and so, in the last digits, which tokens come out.

The first version of this was written by wannagoxuf (PR #4121), who measured +39% on 4x RTX 4090 with 4 streams
(366 to 510 tok/s at `=2`). That is a different rig and not repeated here.

Exact. Checked with the engine's batch protocol (the method of `tools/batch_test.py`: each prompt alone, then the
prompts together in the slots, token ids compared). On the A6000 box, 4 slots (2 groups of 2) and 2 slots (2 groups of 1)
with the feature unset and with `=1`, `=2` and `=3` (`=2` only for 2 slots): greedy streams, greedy and seeded sampled streams
together, slots that finish at different lengths, a slot cancelled in the middle of a draft, and a 3,000-token prompt read beside the
others. Every slot got the tokens it gets alone, and the same tokens as the build without the feature (a cancelled slot stops one
or two tokens apart, so its tokens are compared up to the shorter). Through the server the same cases (and two clients on four
slots) gave the same text as the build without the feature, apart from one or two streams of 19 per run; two runs of the
build without the feature differ from each other in the same way, so this is not the feature. On a box where the experts
do not all fit in VRAM the drafters' memory changes which experts are cached, and then the text of a feature run differs from a
run without it; inside one run every slot still equals its own solo decode.

Not run: AMD (HIP), the draft layer on the first stage, helper expert caches, more than two stages, SYCL (the SYCL
tree has its own copy of the batch code and does not have this).

## Windows of 16 rows, and the four-card AMD profile

A verify or batch window held at most 8 rows. On a layer split every group of the pipeline is its own window, so 32 slots ran as
4 groups of 8 and nothing larger could run. HIP builds now take windows of 16 rows: `--batch 64` on four cards is 4 groups of
16, and `--batch 16 --batch-groups 1` is one 16-row window over all stages. The limit is a build option
(`-DSTRATA_WINDOW_MAX_T=auto|8|16`, default `auto`):

| build | window | why |
|---|---|---|
| HIP (AMD) | 16 | the code objects load when a kernel is first used: the bigger binary (44.9 to 53.7 MB) costs +7 MiB per card after the first request and about 30 to 45 ms once per server start |
| CUDA | 8 | CUDA loads every kernel when the engine starts: 16 adds 14 MB of kernels and +16 MiB of VRAM (RTX PRO 5000) and nothing is gained below 9 rows. On a 16 GB card those MiB shrink the auto-sized expert cache (2 x A4000: 5292 slots instead of 5301), and a cached expert rounds differently from a CPU one, so the texts differ from the 8-row build by that alone (main run with a 17 MiB larger `--vram-reserve-mib` gives the 16-row build's texts, 10 of 10). `-DSTRATA_WINDOW_MAX_T=16` builds it |
| SYCL, Metal | 8 | not touched (they build the shared sources with the limit left at 8) |

A window of up to 8 rows runs exactly the kernels of 0.2.7; the code for 9 to 16 rows is only reached by a wider window. Checked
instruction by instruction: HIP gfx1201, window 8 against the previous main, 3096 of 3097 kernels identical (the other is a
`STRATA_DBG_GDN` debug variant) and 5 renamed; CUDA sm_86 SASS, 3393 of 3397 identical or renamed, the other four differ in the
order of the same instructions. The CPU expert kernels keep 8 tokens per job (a group of 9 to 16 rows is two jobs over the same
expert), and the hyper-connection read is made in slices of 8 rows.

Exactness, `tools/batch_test.py` (greedy tokens of every slot against the same prompt alone): 4 x R9700, 64 slots in 4 groups of 16
(average 13.7 rows a window), 64 of 64 identical, with and without the three switches below; one R9700, `--batch 16`, 16 of 16;
4 x R9700, `--batch 16 --batch-groups 1` (one 16-row window), 16 of 16; 32 slots in 4 groups of 8 with the profile below, 32 of 32.
(Note: `--batch 16` on four stages is 4 groups of 4 slots, windows of 4 rows; it does not test 16-row windows.) Identity
(`rules-144` flags, 10 prompts and a 4K and a 20K prompt, residual dumps compared): p620 (IQ3_S and IQ3_XXS), r9700, l620 and boston (CUDA, window 8 and 16), z440 and p1004 (CUDA, window 8) identical to main
(z440 with window 16: see the CUDA row above).

### Four Radeon AI PRO R9700 cards: the serving profile

The four cards hold the whole of Qwen3.8-Flash-Next IQ3_S, so what limits them is how many requests decode together and how the
four stages are fed. Measured through `serve/server.py` with the launcher's own configuration (400-token answers, short
prompts, total tokens per second of the second round; 7.9K-token prompts for the long row):

| | 1 request | 8 at once | 32 at once | 4 long prompts, out / in tok/s | follow-up TTFT |
|---|---:|---:|---:|---:|---:|
| before: `parallel` 4, split auto | 66.6 | 156.6 | 155.4 | 24.3 / 1892 | 0.78 s |
| profile (below) | 67.9 | 261.2 | 369.6 | 35.0 / 2706 | 0.77 s |

The profile, filled in by `setup.py` and by the launcher only for four cards of that kind (R9700, at least 28 GB each, no
`--peer-device`) running Qwen3.8 IQ3_S:

- `parallel` 32 and `--batch-groups 4`: 32 slots, one group of 8 per card;
- layer split `13,26,39`: the output head sits on the last card and costs about four layers, so it takes 9 and the others 13;
- `--prefill 2048` instead of `auto`;
- `STRATA_QSA_SLOTS=1`, `STRATA_BATCH_SELF_COMMIT=1`, `STRATA_SLOT_COPY_DIRECT=1` (switches from the table above).

What each part gives at 32 at once (tok/s, against 398.7 for the profile with the prompt cache off, which is the owner's profile B): without the three switches 354.3,
split `auto` 374.6, 8 slots instead of 32 277. `--prefill auto` gives the same throughput but a longer first token for a 4.7K prompt
(2.06 against 1.52 s). 64 slots give 418.9 at 64 clients against 398.4 for 32 slots (+5%, measured at an 8K context only) and were left out; `GPU_MAX_HW_QUEUES=4` changes nothing. The prompt cache stays on: turning it off was +8% at 32 clients
with new prompts, but a follow-up turn of a conversation then waits 1.5 s for its first token instead of 0.8 s, and agent loops
are mostly follow-up turns.

A setting of the person always wins: the profile fills only what is not set (the config, `--parallel`, `--layer-split`,
`--batch`, `--prefill` other than `auto`, an environment variable). `STRATA_FOUR_CARD_PROFILE=0`, or `"four_card_profile": false`
in the run config, turns it off. 25 minutes of 32 clients (1088 requests of 600 tokens) through the launcher's configuration ran
without an error at a steady 435 tok/s and with no change in VRAM.

Not done: the hand-off of a batch window between stages (`§7f` of the parallelization report, one flag a group) is not part of
this; it can emit a wrong token from time to time and is listed for later.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch
produces exactly the tokens it produces alone** - verified token by token for 8 concurrent conversations of 150
tokens, with and without the pipeline (`tools/batch_test.py`), and on one RTX 5070 for 4 conversations, for a long
prompt read while two others decode, for a prompt that gave way and went on, for a next turn continued from its
slot (from all it holds, and from its turn checkpoint without the reply's thinking), and for a request stopped in
its slot and continued on the solo path (`tools/batch_interleave_test.py`). These
settings make the comparison exact:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests: by default an expert's rows round differently alone than in a group, so the output depends on
  how many rows of a window share an expert - which differs between a batch and a request alone),
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently, and
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding
  slots: their windows then see the expert cache without the slots the prompt borrowed (those experts run on the
  CPU).

With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose
windows differ can. Measured on the RTX 5070 (Q2_0, 4 conversations of 200 tokens, `tools/batch_test.py` without
the settings above): one equal to its solo run, the others apart from token 0, 99 and 174 - the first token already
differed for one, because the adaptive VRAM tier had moved experts between the solo runs and the batch; all four
texts read as well as their solo runs.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)).

## Limits (for now)

- By default, batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo
  path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- `--batch-mtp` uses one proposal per slot. With a layer split it needs each stage on its own GPU, does not combine
  with `--batch-groups` above 1 (pipelined slot groups run no drafts, #1413) unless `STRATA_PIPED_MTP` is set with
  one slot per group ([above](#mtp-drafts-in-the-pipelined-groups-strata_piped_mtp-opt-in)), and has not been run
  with helper expert caches.
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. A pipelined slot is a conversation cache again (0.1.41): a request left alone in its slot goes back to the solo path with its drafts, as on one GPU.
- The pipelined `--batch-groups` path adapts the VRAM tier only with `STRATA_PIPE_ADAPT_EVERY` (above).
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

## Measured

One RTX 5070 (12 GB), Ryzen 5 7600, 64 GB DDR5, Q2_0, 32K context, through the HTTP server: C different requests
sent at once (an 800-word essay each, 256 tokens per answer, greedy, thinking off), median of 3 rounds:

| Concurrent | Setting | Total tok/s | vs one at a time | Per request tok/s | First token: median / last of the round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | one at a time (default; what setup recommends on this card) | 74.3 | - | 83.6 | 0.4 s / 0.4 s |
| 1 | `"parallel": 2` | 67.3 | -9 % | 74.8 | 0.4 s / 0.4 s |
| 1 | `"parallel": 4` | 57.9 | -22 % | 63.8 | 0.4 s / 0.4 s |
| 2 | one at a time | 71.6 | - | 77.3 | 2.1 s / 4.1 s |
| 2 | `"parallel": 2` | 61.0 | -15 % | 32.6 | 0.5 s / 0.9 s |
| 2 | `"parallel": 4` | 54.6 | -24 % | 28.7 | 0.6 s / 0.7 s |
| 4 | one at a time | 70.7 | - | 79.6 | 6.0 s / 11.2 s |
| 4 | `"parallel": 2` | 61.2 | -13 % | 32.2 | 4.5 s / 9.3 s |
| 4 | `"parallel": 4` | 63.1 | -11 % | 16.9 | 1.0 s / 1.8 s |

On this card the slots buy **waiting time, not speed**: the fourth of four requests starts after 1.8 s instead of
11.2 s, but together they decode 11-24 % slower than one after the other, and a request alone loses 11 % (2 slots)
or 24 % (4 slots), because the slots' sessions (0.56 GiB each) come out of the expert cache and most experts run
on the CPU: a batch window over 4 conversations reads 24 CPU experts per layer against ~8 for one, so it costs about
what the 4 tokens cost one after the other (`strata batch:` in the engine log: 54 ms per 4-row window, ~20 ms per
1-row window). This is why setup leaves a 12 GB card at one at a time. Cards that hold most experts in VRAM, and a
layer split, are where the slots also add speed (below).

A 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, `--batch 8 --batch-groups 4 --trim-stage-weights`, through
the HTTP server, 400 tokens per answer, temperature 0.7 (PR #559):

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

`--trim-stage-weights` alone raised the share of experts held in VRAM on that machine from 76-85 % to 84-100 %
per card.

### One launch for all slots (opt-in)

In a batch window, QSA block scoring, top-k and decode attention ran as one launch per row, and the GDN recurrence
(in the window and in the 1-row commit) as one launch per slot. Two switches run each of them as one launch for all
rows, every row still reading its own slot's indexer keys, K/V pools and GDN state (passed by value as kernel
params, so the graph captured per row layout stays valid):

| Switch | What it does |
| --- | --- |
| `STRATA_BATCH_ATTN_ROWS=1` | block scores, top-k and decode attention of all the window's rows (up to 8) in one launch each. Not with KV streaming (a slot with `kv_mode` 1 keeps the per-row launches). |
| `STRATA_BATCH_GDN_GROUPS=1` | the GDN recurrence in the window, and the commit of 1-row groups, for all slots in one launch. The conv commits stay per slot. |

Both are off by default; off, the engine runs the same launches as before. On, the arithmetic per row is the same:
4 concurrent prompts at temperature 0 gave the same bytes as without them, with `--batch 4` and `--batch 8`.

One RTX PRO 6000 Blackwell (96 GB), IQ3_S, all 24,576 experts in VRAM, `--kv int8`, 262144 context, `--mtp`,
`--spec 4`, through the HTTP server: C different stories sent at once, 1024 tokens each, temperature 0.7, mean of
2 rounds, total tok/s (engine a0116b2, same day):

| Concurrent | `--batch 4` | with both switches | `--batch 8` (reserve 6144 MiB) | with both switches |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 179.2 | 179.4 | 186.6 | 179.7 |
| 2 | 204.1 | 218.7 | | |
| 3 | 243.2 | 276.6 | | |
| 4 | 268.4 | 311.7 | 267.7 | 311.6 |
| 6 | | | 297.7 | 354.8 |
| 8 | | | 308.4 | 374.8 |

The window (`strata batch:` in the engine log) costs 14.3 ms for 4 rows and 24.6 ms for 8 rows without them, and
12.2 ms and 19.9 ms with them: about 1.7-1.9 ms per added row instead of 2.4 ms.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

## Testing

Four scripts drive a built engine or a running server; each exits non-zero on a failure. `serve/test_parallel.py`
tests the server's side with a scripted engine (no GPU).

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. On a two-stage layer split with pipelined groups add `--batch 4 --batch-groups 2` to `--extra` (its 3 slots do not divide into two groups) and set `STRATA_PIPED_READ_PIECE`; run it without a conversation cache, or step 4 restores the prompt instead of reading it. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

For single-GPU `--batch-mtp`, use a model with `rt/draft_vocab.bin` to exercise admission with a
shared draft-vocabulary head. This compares both slots' output against solo decoding:

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 2 --n 2 --max-new 64 \
    --extra "--batch-mtp --pcie-frac 0 --adapt-every 1000000"
```

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | end that slot at its next window |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
