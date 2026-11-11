# Lend the output head's VRAM to the prompt path (`--prefill-head-alias`, opt-in)

Two large VRAM users never work at the same time. The native output head (417 MiB for the Q5_K head of
IQ3_XXS) turns the last hidden state into logits, so it is needed only while tokens are generated. The batched prompt
path's buffers are needed only while a prompt is read. With `--prefill-head-alias` the engine takes turns: the head's
VRAM holds prompt buffers while a prompt is read, and the head is read back in before the first generated token.

The flag is off by default. With it off, the engine does exactly what it did before.

## Why

On an 8 GB card the prompt chunk is as large as the free VRAM allows, and prefill speed follows the chunk closely. Here
is the same 29,854-token prompt with a fresh engine per trial and one request each. The card and settings are listed
under [Measurements](#measurements).

| Prompt chunk | Trials | Prefill, median (range) | How the engine got there | Free VRAM after start |
|---:|---:|---:|---|---:|
| 256 | 4 | 185.5 (177.8-195.9) tok/s | `--prefill auto`, reserve 686-745 MiB | 383-443 MiB |
| 512 | 12 | 292.5 (281.0-306.1) tok/s | `--prefill auto`, reserve 442-667 MiB (590 is our tuned value) | 137-365 MiB |
| 768 | 7 | 387.4 (375.8-397.2) tok/s | `--prefill auto`, reserve 331-405 MiB | 27-103 MiB |
| 1,024 | 4 | 442.9 (440.8-444.1) tok/s | `--prefill auto`, reserve 239-295 MiB | 0 MiB |
| 1,280 | 5 | 514.5 (512.8-517.3) tok/s | `--prefill auto`, reserve 147-221 MiB | 0 MiB |
| 1,536 | 2 | 573.6 (571.0-576.3) tok/s | `--prefill auto`, reserve 73-110 MiB | 0 MiB |
| 1,792 | 3 | 631.4 (628.7-632.9) tok/s | `--prefill auto`, reserve 0-36 MiB | 0 MiB |
| **2,048** | **6** | **667.1 (661.5-667.7) tok/s** | **`--prefill-head-alias`, reserve 590 MiB** | **289 MiB** |

Without the flag, every chunk above 768 tokens came only from lowering `--vram-reserve-mib`, and that left 0 MiB free.
Under WDDM that is the state in which Windows starts paging. With the flag, the 2,048-token chunk came with the 590 MiB
reserve unchanged.

## What it does

1. **At start**, after the MTP draft layer has copied its draft-head rows from the output head, the head is unloaded.
   The auto expert cache is then sized with the head's VRAM counted as free (on our card: 749 slots instead of 508).
   The prompt path borrows cache slots 1 to N (slot 0 stays). Whatever does not fit there is allocated by the prompt
   path itself, which makes a "hybrid" layout. The chunk is the largest of 2,048, 1,536, 1,024 and 512 tokens whose
   buffers fit in the borrowed slots plus free VRAM with 512 MiB to spare.
2. **For each request**:
   1. If the head is loaded, unload it and lay the prompt path out again (median 20 ms).
   2. Read the prompt.
   3. Refill the lent slots and free the prompt path's own buffers.
   4. Load the head again (median 255 ms) before the first decode window.
3. **Graph safety.** The verify windows' CUDA graphs are captured during the first decode and hold the head's address.
   If a later reload puts the head at a different address, the engine stops with an error, and the server starts it
   again. It does not run graphs that point at freed memory. We never saw this happen; see
   [Open questions](#open-questions).

The engine refuses the flag, with the reason, without `--serve`, with more than one GPU or a layer split, with
`--batch`, with `--pipeline-windows`, with `--short-read` above 0, with a fixed `--expert-cache` above 1, without `--mtp`,
or without the native output head.

## Costs

- **Short reads get slower.** Each request reloads the head (median 255 ms over 2,717 restores, p90 261 ms, max
  495 ms) and rebuilds the prompt path (20 ms). It also lends and then refills almost the whole cache, even for a few
  tokens. The engine's own prompt time, which includes all of this, by fresh tokens read (production, medians):

  | Fresh tokens read | Default | Head alias |
  |---:|---:|---:|
  | 1-64 | 671 ms (293 requests) | 1,311 ms (614) |
  | 65-512 | 1,703 ms (409) | 2,453 ms (955) |
  | 513-2,048 | 4,062 ms (278) | 3,331 ms (618) |
  | 2,049-8,192 | 15,206 ms (118) | 6,876 ms (277) |

  So the flag costs about 0.6-0.75 s on reads of up to 512 tokens and pays off from somewhere between 513 and 2,048
  tokens. In the agent loop, 614 of 2,717 requests read 64 fresh tokens or fewer.
- **Less free VRAM while decoding.** In the default run, the verify windows were captured with 207-287 MiB free. With
  the flag, they were captured with 0 MiB free. Over 2,717 requests (about 26 hours) we saw no paging symptoms: decode
  held about 29 tok/s from the first request to the last (table below). This is the main risk on other cards and
  drivers.
- **More slots to refill after each prompt.** The prompt path borrows almost the whole cache (748 of 749 slots instead
  of 328 of 508), so more experts are copied back after each prompt. That time is counted in the prompt time above.

## Measurements

Card and settings for every number on this page:

- NVIDIA RTX PRO 2000 Blackwell Laptop GPU, 8,151 MiB, Windows 11 Pro (WDDM, driver 597.16), CUDA 13.0, sm_120.
- Intel Core Ultra 9 285H (AVX2), 64 GiB RAM, PCIe probe 25 GB/s (`pcie_frac` 0.55 picked by auto).
- Qwen3.8-Flash-Next IQ3_XXS native pack, `--max-context 262144`, `--kv q4_0`, `--kv-resident 32768`, `--spec 4`,
  `--spec-min-p 0.5`, MTP with the English draft vocabulary, `--prompt-cache 3`, `--vram-reserve-mib 590`,
  `--expert-cache auto`, `--prefill auto`.
- The engine was a local 0.1.41 build with this change. Both sides also used a local `STRATA_ARENA_REGISTER=0` switch
  that is not part of this change: it skips `cudaHostRegister` of the 40 GiB arena on this machine.

### Controlled: one prompt, one forced output

Each trial ran a fresh engine on the 29,854-token prompt. The same 3,072-token continuation was forced one verify row per
token, with no speculative drafts, so the decode numbers compare the same work. A local benchmark switch did the forcing;
it is not part of this change.

| Setting | Trials | Expert slots | Prompt chunk | Prefill | Decode | Expert-cache hits |
|---|---:|---:|---:|---:|---:|---:|
| 1-slot cache | 4 | 1 | 1,024 (own buffers) | 441.0 tok/s | 22.26 tok/s | 0.2% |
| 1-slot cache + head alias | 2 | 1 | 2,048 | 666.3 tok/s | 24.50 tok/s | 0.2% |
| auto cache + head alias | 3 | 749 | 2,048 | 667.7 tok/s | 26.21 tok/s | 24.9% |

We did not run the default (auto cache, no head alias) in this forced mode. Its free-running result on the same prompt
is the 512-token row of the first table: one trial at the 590 MiB reserve read 306.1 tok/s and decoded 27.9 tok/s, and
six head-alias trials decoded 26.6-29.0 tok/s (median 27.9). Free-running outputs differ from trial to trial, so their
decode rates are only a rough comparison.

### Production: an agent loop

The workload was OpenCode agents (worker, reviewer and retrospective) running a coding loop against `/v1/chat/completions`.
Prompts went up to about 150K tokens, with `--prompt-cache 3` prefix reuse.

- Default: 1,200 requests over 8 loop cycles.
- `--prefill-head-alias`: 2,649 requests over 20 cycles.

The flag also needs `--short-read 0`; the default run had 64. Everything else was the same. Rates come from the engine's
own lines (`reading the prompt` progress over at least 5 s, and the decode rate of requests with at least 32 output
tokens). They are compared inside 16K-token context bands, so a heavier or lighter mix of prompts on either side does not
count as a speed change. Intervals are 95% bootstrap intervals.

| | Default | Head alias | Change within context bands |
|---|---:|---:|---:|
| Prefill, median | 256 tok/s | 583 tok/s | +125% (+123% to +129%) |
| Decode, median | 24.8 tok/s | 29.2 tok/s | +19% (+17% to +23%) |
| Expert-cache hits, median | 20.8% | 25.9% | 508 -> 749 slots |

The decode difference needs care. The default runs alternated between long stretches near 27.5 tok/s and long stretches
near 19.5 tok/s. A rolling median over 51 requests showed plateaus of hundreds of requests each, and in one run the
switch from slow to fast came with no engine restart. In the head-alias run, each twelfth of the 2,649 requests had a
median decode of 28.7-29.8 tok/s. Against the default's
fast stretches only, decode was about 6% faster. We do not know what caused the slow stretches, so we do not claim that
this change removes them.

The engine binary was also rebuilt between the two runs, to add prompt-cache support to this change. The controlled
table above is the comparison that holds the binary fixed.

## Port check on 0.1.43.1

This change was moved to upstream main (0.1.43.1) and checked on the same PC. Both runs used the same build, a fresh
engine each, and `STRATA_ARENA_PIN_GIB=auto`: without it, the full sliced arena pin on this 64 GB PC left WDDM with
0 GiB free for the cache (#243). The requests were the 29,842-token prompt cold, then a 31-token follow-up on its
prefix, two 45-token prompts, and the long prompt again. Output was capped at 768 tokens with low reasoning, and
generation was free (not forced).

| | Default | Head alias |
|---|---:|---:|
| Expert slots, prompt chunk | 507, 512 | 753, 2,048 |
| Free VRAM when the verify windows were captured | 207-245 MiB | 0 MiB |
| Cold 29,842-token prompt, two runs | 276.5, 277.1 tok/s | 643.1, 649.7 tok/s |
| Decode on those two | 35.7, 34.7 tok/s | 34.1, 31.6 tok/s |
| 31-token follow-up (29,837 reused), prompt time | 632 ms | 1,721 ms |
| 45-46-token prompts, prompt time | 871, 1,005 ms | 1,780, 1,424 ms |

All five requests completed with coherent answers. The head came back at the same address every time, so the guard
never fired. 0.1.43.1's new start-of-request warning ("below 1024 MiB free") appears with both settings, at 289 MiB
for the default and 0 MiB with the flag.

## Open questions

- **A fixed address for the head.** The head goes back with `cudaMalloc`, which does not promise the same address. The
  check above makes a moved head a clean engine restart. A cleaner design would keep the head's virtual range and swap
  its backing (CUDA VMM), or lay prompt buffers directly into the head's allocation instead of freeing it.
- **Skip the swap for short reads.** A request that reads a few dozen fresh tokens pays about 0.6 s (table above) for a
  2,048-token chunk it does not need. A threshold on fresh tokens, below which the head stays and the read goes through
  the decode windows as `--short-read` does, would remove that cost.
- **Other cards.** We measured only the 8 GB card above, on Windows. A larger card already gets a large chunk without
  the flag, so we expect the gain to be smaller there.

## Not tested

- This change on upstream main (0.1.43.1) was compiled for CUDA (13.0, sm_120, MSVC, Windows) and checked with the
  short run above. The longer measurements come from the 0.1.41 build.
- HIP (the same sources) and SYCL. On SYCL, `Prefill::init` refuses the hybrid layout; the SYCL engine has no flag
  for it.
- Linux, multi-GPU, `--batch` and `--pipeline-windows` (all refused), and models other than Qwen3.8-Flash-Next IQ3_XXS.
