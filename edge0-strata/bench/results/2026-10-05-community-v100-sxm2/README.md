# Community benchmark: Tesla V100-SXM2-32GB on Windows (CUDA 12.6, MSVC), three quantizations

Measured on 2026-10-05 on a Windows 11 desktop. This report tests the **experimental Volta build**
(`-DSTRATA_EXPERIMENTAL_SM60=ON`, #236) on a **V100-SXM2-32GB** and compares three quantizations of
Qwen3.8-Flash-Next on the same machine, including a small correctness battery.

There is already a [V100 document](../NVIDIA_V100.md) covering the kernel paths and the Linux build.
This report adds a **Windows / MSVC / CUDA 12.6** data point and the **quantization comparison**.

**Headline:** with the official `Q2_0` (2-bit) the engine decodes at **76.4 tok/s** and reads an
8,000-token prompt at **1,268 tok/s**; the 4-bit Unsloth `UD-IQ4_XS` is **26.3 tok/s**, i.e. the
2-bit files are **~3x faster**. On a 5-question reasoning battery `Q2_0` answered **5/5** and
`IQ2_XS` answered everything it finished (4/4, one ran out of its token budget).

## Hardware and software

- **GPU:** Tesla V100-SXM2-32GB (Volta, sm_70, 32 GB HBM2), PCIe **Gen3 x16**. A GeForce GTX 1050
  is also present and is the display adapter; CUDA sees the V100 as device 1, so every run sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1`.
- **PCIe:** the engine's own probe measured **13.1 GB/s** host→device. This matters more than
  anything else here - see "The link is the first thing to check" below.
- **CPU and RAM:** AMD Threadripper 2990WX (32 cores / 64 threads, **AVX2, no AVX-512**), 96 GB RAM.
  The engine ran with its default expert-pool worker count.
- **OS and runtime:** Windows 11 (build 26200), **CUDA 12.6**, Visual Studio 2022 Build Tools
  (MSVC 19.44), Ninja. NVIDIA driver as shipped with the machine.
- **Build:** upstream `main` at `6f32ec0` (engine 0.1.27), **no source changes**:

  ```
  cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF ^
        -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 ^
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
  cmake --build build-sm70 --target strata -j 8
  ```

  Configure and build completed with **0 errors**. `-allow-unsupported-compiler` is needed because
  CUDA 12.6's nvcc does not recognise MSVC 19.44. The MSVC + static-CUDA-runtime `LNK1169` problem
  (#585) does not appear: with `STRATA_EXPERIMENTAL_SM60=ON` the build already selects the shared
  runtime. The engine logs `GPU 0: Tesla V100-SXM2-32GB, compute capability 7.0` at startup.
- **Other services:** the machine's usual desktop services kept running (browser, chat apps);
  nothing else used the V100.

## Models and configuration

Three complete model sets, all packed with the repository's own `tools/iq_pack.py`:

| Model | Source | Size | Pack |
| --- | --- | --- | --- |
| **Q2_0** | ISTA-DASLab GSQ-RCO (2 shards) | 66 GB | `--gguf shard1` |
| **IQ2_XS** | ISTA-DASLab GSQ-RCO (2 shards) | 68 GB | `--gguf shard1` |
| **UD-IQ4_XS** | unsloth (merged, single file) | 87 GB | `--compat-bf16` |

For the two GSQ-RCO models `--native` is shard 1 (the experts) and `--ple-gguf` is shard 2
(`per_layer_token_embd.weight`). The UD file is single, so both options point at it.

Common configuration: `--expert-cache auto --prefill 2048 --spec 3 --max-context 32768`
(262144 for the long-context point) `--kv fp16 --vram-reserve-mib 2048`, greedy decoding.
The MTP draft layer was built from the original checkpoint with `tools/mtp_fetch.py` +
`mtp_pack.py` + `mtp_rt.py`.

## Speed

Protocol: an **8,000-token prompt**, **256 generated tokens**, one run per configuration (the engine
itself reports about ±20% between runs; see the limits). These are engine-side numbers from the
`decode` / `prefill` lines, not client timings.

| Model | context | decode tok/s | prefill tok/s | VRAM expert cache |
| --- | ---: | ---: | ---: | --- |
| **Q2_0** | 32,768 | **76.4** | **1,268** | ~18,900 slots, 24.4 GiB |
| IQ2_XS | 32,768 | 71.2 | 1,183 | ~18,200 slots, 24.4 GiB |
| IQ2_XS | 262,144 | 75.7 | 1,165 | ~14,200 slots, 18.3 GiB |
| UD-IQ4_XS | 32,768 | 26.3 | 665 | 10,078 slots, 22.7 GiB |

The 2-bit files are ~3x faster for a structural reason: an expert is about half the bytes, so nearly
twice as many fit in the same VRAM, and the number of experts per layer left to the CPU falls from
3.9 to 0.7. The engine's slot accounting (18,900 vs 10,078) and its "CPU experts per layer" line
both show it.

MTP acceptance was 63-76% across these runs (2.2-2.7 tokens per round).

### Long context

On this machine `--max-context` from 32K to 256K does **not** change steady-state decode speed
(32K: 22.6 / 64K: 17.5 / 128K: 19.3 / 256K: 22.3 tok/s for one model in an earlier sweep - all
inside the run-to-run noise). It only costs VRAM: fp16 KV at 256K takes about 6.9 GB, which shrinks
the expert cache. `--kv-resident 32768` moves most of the KV to RAM instead and gives the VRAM back
to the experts.

## The link is the first thing to check

The same machine, the same 30,000-token prompt, the same model, two different PCIe link widths:

| Link | engine's probe | prefill tok/s | time to first token |
| --- | ---: | ---: | ---: |
| Gen3 **x4** | 3.3 GB/s (`pcie_frac 0.00`) | 179 | 168 s |
| Gen3 **x16** | 13.1 GB/s (`pcie_frac 0.28`) | **682** | **45 s** |

**3.8x** from the slot alone. The engine adapts: at 3.3 GB/s it decides to keep every expert on the
CPU, at 13.1 GB/s it offloads a third. If a V100 looks far slower than these numbers, check
`nvidia-smi --query-gpu=pcie.link.width.current` before changing anything else.

(These two runs are from an earlier build of the engine on the same machine - the rest of this
report is from `6f32ec0`. The link was the only thing that changed between them.)

## Correctness on a reasoning battery

Five prompts with checkable answers, greedy, 1024-token cap, through the resident server:

| Question | Q2_0 | IQ2_XS |
| --- | --- | --- |
| Wolf, goat, cabbage (the classical 7-step solution) | **correct, complete** | hit the token cap mid-thought |
| Reverse arithmetic (answer 10) | correct, with a check | correct, with a check |
| Measure 4 L with 3 L and 5 L jugs | correct (7 steps) | correct (6 steps) |
| Chickens and rabbits (23 / 12) | correct | correct |
| `two_sum` in O(n) | correct | correct |

Both answered every question they finished. The difference is **verbosity of the thinking block**:
`Q2_0` finished the wolf problem in 983 tokens, `IQ2_XS` spent its whole budget re-deriving and
never wrote the answer. Nothing here establishes general answer quality - a 4-bit model is still
the safer default for hard reasoning, and it is the 3x slower option.

### A 2-bit failure mode worth knowing

Twice, on different engine revisions, a 2-bit model fell into a **verbatim repetition loop** (the
same few lines of reasoning repeated dozens of times, eating the whole budget). It is not an
incorrect answer, it is a stuck one. On the one occasion it was chased down, **enabling sampling
fixed it**: the same prompt with `--seed 42 --temperature 0.7` answered both questions correctly.
Greedy decoding reproduces the loop deterministically, so a retry with sampling (or another seed)
is the practical escape.

## Limits

- One machine, one operator. PCIe width, RAM speed and CPU differ from any other V100 host.
- 1-2 runs per configuration and 5 prompts for the quality table: enough to rank 2-bit against
  4-bit (3x), **not** enough to separate `Q2_0` from `IQ2_XS` precisely.
- No needle/recall test, no client-side timings, no per-request JSON records: these are engine-side
  summary numbers, and the prompt sizes are 8,000 and 30,000 tokens rather than a 4K/32K/128K sweep.
- CPU is AVX2 only (no AVX-512), so the expert pool is at its slowest here and the gap between
  quantizations is at its widest; a machine with AVX-512 will show smaller differences.

## Windows notes (not Volta-specific)

1. **Batch files must be ASCII-only.** Under a GBK console code page cmd.exe mis-parses a batch file
   that contains non-ASCII bytes: it eats a fixed number of characters from the start of *every*
   line, so `rem Rebuild` executes as `build` and `set "ROOT` becomes `OOT`.
2. **`taskkill` does not reliably kill the engine.** `taskkill /PID <engine> /F` answers
   `Access denied` and leaves it running, still holding ~34 GB of pinned RAM. `TerminateProcess`
   (PowerShell `Stop-Process -Force`, or `psutil.Process(pid).kill()`) works, and the engine then
   needs **10-30 s** to hand the pinned memory back. Verify before starting the next run: process
   gone, `nvidia-smi` back to idle, port free. We lost a measurement round to a stale engine that
   silently answered in place of the model we thought we had started.
3. There is a **multi-second cold start** on the first tokens after a large `--max-context` (the KV
   buffers are touched for the first time). Any comparison with fewer than ~30 generated tokens is
   measuring that instead of throughput.
