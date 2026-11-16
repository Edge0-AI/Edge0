# Strix Halo (Ryzen AI Max, gfx1151) on Windows: install, what you get, what to know

This page is for a Windows PC with a Ryzen AI Max "Strix Halo" chip (Radeon 8060S / 8050S, `gfx1151`). It says what Strata does there out
of the box, what speed to expect with Qwen3.8-Flash-Next IQ3_S and with DeepSeek V4.1-Flash, and the traps on this hardware. The Linux
build, the toolchain and the opt-in switches are on [STRIX_HALO.md](STRIX_HALO.md); the DeepSeek details are in
[DEEPSEEK41.md](DEEPSEEK41.md).

The structure of this page comes from the first Windows guide for this PC, written by HaiYunChuang (pull request #3029). It measured
engines 0.1.41 to 0.1.47 with a hand-made configuration (`--mmap-experts`, a list of switches, a batch file). Since then the app and
`strata launch` have taken over most of that, so this version describes the 0.2.7 install and every number in it was measured again.

## 1. What this PC reaches

Measured on 2026-10-26 on a GMKtec NucBox EVO-X2 (Ryzen AI Max+ 395, 128 GB shared memory: a 64 GB BIOS carve-out for the graphics chip
and 64 GB that Windows sees; Windows 11 Pro, AMD Adrenalin 26.10.1 / driver 32.0.32015.7029). Strata was the 0.2.7 release candidate
(Terminal and engine 0.1.51 built from commit `3e468f5de`, ROCm 10.2.0a20260930). The server was started with `strata launch` and
nothing else was set by hand; the speeds come from `tools/strata_bench.py` against the running server (three runs per test, medians,
the engine's own timings; the model is loaded before the clock starts). The numbers can still move before the release is cut.

**Qwen3.8-Flash-Next IQ3_S** (`strata launch --family qwen --model iq3_s`; 131,072 tokens of context, int8 KV, pictures on):

| Test | Result |
| --- | --- |
| Short chat, answer speed | **44.0 tok/s** (40.6 to 59.5: code-like answers are faster than prose) |
| Short chat, time to first token | 0.60 s |
| 4K prompt, read speed | **755 tok/s** (first token after 5.4 s) |
| 16K prompt | **821 tok/s** (19.6 s) |
| 32K prompt | **792 tok/s** (40.6 s) |
| Coding-agent session, 6 turns | 69% of the prompt came from the conversation cache; answers 47 tok/s; 4.5 s to the first token per turn |

**DeepSeek V4.1-Flash 2.00 bpw** (`strata launch --family deepseek41 --model ist-das-2.00bpw`, Faster mode, the drafter on; the 4K row is
the default 8K context, the others 64K context with `--context 64k`):

| Test | Result |
| --- | --- |
| Short chat, answer speed | **12.1 to 13.6 tok/s** |
| 4K prompt | **214 to 234 tok/s** (about 18 s to the first token) |
| 16K prompt | **326 tok/s** (51 s) |
| 32K prompt | **342 tok/s** (97 s) |
| Coding-agent session, 6 turns | 70% from the conversation cache; answers 14.2 tok/s; 17 to 31 s to the first token per turn |

The prompt rows use the word-list prompt of `tools/strata_bench.py`; ordinary text is costlier for the experts and reads slower (124 to 164 tok/s on 3 to 5K-token pieces of prose). Measured again on the final build, the 4K row reads 229 to 234 tok/s warm. These are speed measurements. They say nothing about answer quality, and the short-chat range depends on the text. The DeepSeek 2.25 bpw set
and the other Qwen sizes were not measured for this page.

The DeepSeek 4K prompt here (214 to 234 tok/s) is higher than the 87 tok/s on [STRIX_HALO.md](STRIX_HALO.md), which was measured on 2026-10-25 with an
earlier build. The 0.2.7 DeepSeek defaults explain most of the gap (the same build with `DS4_DEFAULTS=026` reads 133 to 139 tok/s); the table there
changes one thing at a time.

## 2. Memory and the BIOS

- Allocate a **fixed carve-out** to the graphics chip in the BIOS (64 GB on the measured PC). The engine counts the carve-out and the
  RAM that Windows sees as one pool and does not change any BIOS or driver setting.
- The graphics chip and the CPU share that pool. For Qwen IQ3_S the engine reads the experts into RAM, keeps them there (it does not
  register the arena with the GPU driver on an integrated chip) and gives the expert cache every slot: the start log says
  `expert cache auto ... -> 24576 slots` and `expert arena: NOT registered with the GPU driver (integrated GPU, UMA mode)`. The process held
  about 49 GB and 10 GB of RAM stayed free. Close other large programs before you load a model this size.
- The default of four parallel slots takes 7.5 GiB of the shared pool for slot sessions (the log says `--batch: 4 slot sessions ... take
  7.54 GiB`). With IQ3_S the cache still held every expert, so there was nothing to gain from fewer slots here; with a bigger model there may be.

## 3. Install

1. Use the Windows app (`Strata-<version>-windows-x64-setup.exe`) or the Terminal. The engine for this chip is the `windows-x86_64-hip-rdna3`
   package; the app or `strata launch` downloads it on the first start, and `strata update` keeps it current.
2. Choose the model in the app, or `strata launch --family qwen --model iq3_s`. With model files you already have, add
   `--model-path <folder>` (for IQ3_S the two shards are in one folder: shard 1 is the main model, shard 2 holds the per-layer embedding
   table and must stay a separate file).
3. The first start took about 20 minutes on the measured PC with the model files already on disk: the engine package (158 MB), the
   check of the 84 GB of model files, the picture reader (0.9 GB), the draft layer (about 5 GB) and the one-time pack. The next start
   took under two minutes.

What `strata launch` wrote for IQ3_S, with no switches of mine (these are the settings to copy if you drive the engine by hand):

```
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <draft layer>
--max-context 131072 --kv int8 --vision --vram-reserve-mib 700   (backend hip, GPU picture reader)
```

and for DeepSeek 2.00 bpw: `--max-context 8192 --dspark auto --dspark-conf 0.65 --prefill-chunk 16384`, with `DS4_ASYNC_T=4` in the
environment and the package's `rocm/bin` folder on the library path.

## 4. What is on by default

- On `gfx1151` the engine turns on exact speed switches by itself (19 on Linux, 18 on Windows, which leaves `STRATA_SH_STREAM` out, see below; the start line lists them; `STRATA_GFX1151_DEFAULTS=0` turns them all
  off, and a switch you set is kept). They are the ones that give the same answers as without. `STRATA_HC_Q8=1` is also on here (decode
  about 9% faster, near-tie tokens can differ); `STRATA_HC_Q8=0` keeps the old read.
- The hipBLASLt table for `gfx1151` ships in the package and is chosen without a setting (`prefill gemm: hipBLASLt tuning enabled (32 rows,
  gfx1151, version 100500)` in the log). A hand-built engine has to be pointed at it: [AMD_HIP.md](AMD_HIP.md).
- **The matrix-core prompt set is on by default on `gfx1151` on Windows** (engine 0.1.52 and the build after it): `STRATA_HIP_WMMA`,
  `STRATA_PF_GEMM`, `STRATA_PA_FAST`, `STRATA_HC_UPMIX` and `STRATA_PF_FUSED`. It is the set that has been on the Radeon 780M since
  021-minipc, and the start line lists it (`integrated GPU (gfx1151): the matrix-core prompt set is on ...`). Prompts on Qwen IQ3_S:
  4K 688.8 -> 1,192.8 tok/s, 8K 787.0 -> 1,283.5, 12K 750.4 -> 1,256.5, 32K 775.4 -> 1,245.1, 64K 748.5 -> 1,188.9 (x1.59 to x1.72;
  interleaved pairs, one engine at a time). The answers differ in the last bits: the first-token KL against the 0.2.7 default is
  0.0014 mean over four prompts of 4K to 24K, top-1 the same everywhere, and the measured noise floor is 0.00000 - against the
  accepted `HC_Q8` bar of 0.0228 mean. `STRATA_IGPU_PROMPT_SET=0` turns it off, `STRATA_SELECT_WMMA=1` adds the block scorer back
  (it measured x1.003), a single switch can be turned off on its own (`STRATA_HC_UPMIX=0`).
- **The prompt keeps its own buffers instead of lending expert-cache slots** (`--no-prefill-borrow`, engine 0.1.52 and after, on
  `gfx1151` Windows only). A lent slot's expert has to be read back from the host arena during every prompt chunk, and on this PC
  that copy runs at about 0.5 GB/s (the arena is not registered with the driver in UMA mode): 823 slots took 2.75 s of a 3.3 s 4K
  prompt inside the copy thread. With its own buffers every expert stays in the cache (`experts streamed 0 ... resident 24576` in the
  log) and prompts are 8.5% (4K) to 8.9% (8K) faster. `STRATA_PREFILL_BORROW=1` brings lending back. `--prefill auto` keeps a
  16,384-token chunk on this PC instead of falling back to 2,048: the prompt reads every expert once per chunk, so a bigger chunk reads
  them fewer times (32K 1,250.6-1,260.0 -> 1,286.3-1,307.9 tok/s, 64K 1,188.4-1,191.0 -> 1,234.3-1,238.2; at 4K and 8K the chunk is
  the prompt itself and nothing changes). `STRATA_GFX1151_CHUNK=0` keeps 6,144, and `--prefill auto:32768` is another ~2% at 64K for
  about 12.8 GiB of prompt buffers.
- `--mmap-experts` is not needed. The first Windows guide found that without it the pinned expert arena used up the graphics memory
  budget and decode fell to 28 tok/s on engine 0.1.41; the engine on this page sizes the cache from the shared pool and reaches every slot.
- DeepSeek runs in "Faster" mode here: 5 of 6 experts per word. The app and `strata launch` say so; the quality change is small.

**`STRATA_SH_STREAM` (the shared expert on its own stream) is off by default on Windows `gfx1151` in 0.2.7, because it is slower there.** It stays
on by default on Linux `gfx1151`, where it was measured to help ([STRIX_HALO.md](STRIX_HALO.md)). The first Windows guide found the Windows
result on engine 0.1.41 (+13% decode with it off). Measured again on the 0.2.7 build as it was before this default changed (the switch was
on), with Qwen IQ3_S, three starts each, run in turn (on, off, on, off, on, off), the same quick suite after every start:

| Start | `STRATA_SH_STREAM` | Chat decode, median of 9 answers (tok/s) | 4K prompt (tok/s) |
| --- | --- | ---: | ---: |
| 1 | 1 (the default then) | 43.9 | 742.9 |
| 2 | 0 | 48.3 | 742.9 |
| 3 | 1 | 43.6 | 746.8 |
| 4 | 0 | 46.0 | 740.8 |
| 5 | 1 | 43.2 | 745.5 |
| 6 | 0 | 46.6 | 739.4 |

With the switch off, decode was 6.9% faster (46.6 against 43.6, the median of the three starts) and the code-like question 14% faster
(67 against 59 tok/s); all three pairs went the same way. The prompt speed did not change.

The answers stay the same: on the same PC and build I ran five fixed greedy prompts (a short question, a code request, an explanation, a
4,091-token prompt and a 16,383-token prompt; 895 generated token ids) through the engine four times, with the switch on, off, on, off. The
token ids were identical in all four runs for every prompt.

What 0.2.7 does: on Windows `gfx1151` the engine no longer turns the switch on by itself. To turn it back on, use `strata launch ...
--set STRATA_SH_STREAM=1` (or put `STRATA_SH_STREAM=1` in the environment of an engine you start by hand); a value you set is always kept.
`STRATA_DEFAULTS=026` brings back everything 0.2.6 did, including the switch on. This is one model, one PC and short answers; DeepSeek was
not tested with it, and Linux `gfx1151` has not been measured against it.

## 5. Traps

1. **Keep RAM free.** Qwen IQ3_S held 49 GB of 63.6 GB. A browser with many tabs, or a second model, is enough to push the PC into paging.
2. **Do not run two engines.** One model at a time on this PC; `strata launch --status` shows what is serving, `strata launch --stop` ends it.
3. **The graphics driver.** The 2025-03 driver failed at 32K context for DeepSeek. The measured driver is Adrenalin 26.10.1 (32.0.32015.7029).
4. **If Windows resets the driver** (screen flicker, the engine dies mid-answer), see the driver-reset entry in
   [TROUBLESHOOTING.md](TROUBLESHOOTING.md#amd-cards).
5. **Other tuning from the first guide** (a longer `--prefill`, `STRATA_PF_*` switches, `--lookup-chain`): those results are for engines
   0.1.41 to 0.1.47 and a hand-made configuration. They were not repeated here, so they are not part of this page.

## 6. Reproduce it

```
strata launch --family qwen --model iq3_s            # then, from a source checkout:
python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long
```

`--long` adds the 32K prompt. For DeepSeek add `--context 64k` to `strata launch` first; the default 8K context skips the 16K prompt and
the agent session.


