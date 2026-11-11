# Community benchmark on Tesla V100-SXM2-32GB (sm_70): the opt-in Volta decode kernels

Measured on 2026-10-08/09 by **Mitsuasa513**. This report answers the question
`docs/NVIDIA_V100.md` leaves open: whether `STRATA_SM70_TABLE=1` — the interleaved `native_mmvq_il`
with a Volta rows table, expert mode 8 and the latency-hidden norm/up — should become the default on
sm_70 with the 0.1.41 code. On this machine it should: **decode +10% to +15%, with output that is
bit-for-bit identical** to the default path.

The main limitation: one machine, one model, greedy decoding, and four back-to-back engine runs
(plus the repository's own A/B tool, which is too noisy here to resolve the short-chat cases).

## Hardware and software

- **GPU:** Tesla V100-SXM2-32GB (sm_70), PCIe Gen3 **x16** (engine probe: 13.1 GB/s host→device),
  32,768 MiB, driver 560.94. A GTX 1050 is present in the machine but not used (`--gpu 1`).
- **CPU:** AMD Threadripper 2990WX (32 cores, AVX2, **no AVX-512**); **RAM:** 96 GB;
  **storage:** NVMe SSD.
- **OS:** Windows 11 (build 26200). **CUDA 12.6** (nvidia-smi reports CUDA 12.6 runtime). MSVC 19.44.
- **Strata:** `Niko1221/Strata` at **`fb58e0d`** (v0.1.41), source build:

  ```text
  cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
        -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DSTRATA_GGML_DIR=<llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d>
  ```

  `strata.exe` SHA-256 `5bfb80e584053443904b19e7723faa41b7c97390487a5bb8c9e902d8901fc501`.
  The ggml source is the commit `third_party/ggml/VERSION.txt` pins (unchanged since 0.1.39).
  `NMake Makefiles` rather than Ninja: on the Windows host used for this run, ninja could not start
  any child process (its job-object use was refused by the tooling sandbox), while nmake and the
  MSVC toolchain worked normally. That is a property of this host, not of Strata.

## Model and configuration

- Unsloth **UD-IQ4_XS** (`Qwen3.8-Flash-Next-UD-IQ4_XS`, merged single GGUF, 87.2 GiB), the model
  `docs/NVIDIA_V100.md` measures. Pack built with `tools/iq_pack.py` (native experts, pack index v3),
  MTP runtime from `tools/mtp_rt.py`. Vision off. Shipped expert profile.
- Context **262144**, **`--kv int8`**, `--expert-cache auto`, **`--prefill auto`**,
  `--vram-reserve-mib 1024`, `--spec 3`, greedy.
- The two arms differ in **one** environment variable:
  `STRATA_SM70_TABLE=0` (A: exactly what 0.1.40.3 ran) versus `STRATA_SM70_TABLE=1` (B).

```text
strata.exe --pack <E:\V100-strata-0.1.41\packs\iq4xs> \
  --native <Qwen3.8-Flash-Next-UD-IQ4_XS-merged.gguf> \
  --ple-gguf <same file> --expert-profile <data\expert-profile.bin> \
  --expert-cache auto --prefill auto --spec 3 --mtp <mtp\rt> \
  --max-context 262144 --kv int8 --vram-reserve-mib 1024 \
  --tokens-file prompt_8k.txt --max-new 256
```

## Method

Two measurements, both on the same machine and workload:

1. **Engine-level paired runs** (the primary evidence). The same pretokenized 7,999-token prompt and
   a 256-token greedy cap, arms run back to back, in **two pairs with the order swapped**
   (A→B and B→A) so a warm-up drift cannot favor one arm. Each arm is one engine process
   (load ≈ 2.5 min, not included in the timings). Decode/prefill come from the engine's own lines.
2. **`tools/ab_engine.py`** (the repository's A/B tool), 3 rounds, arms alternating per round:
   three short chats plus one 6,000-word prompt, `max_tokens 256`, `temperature 0`,
   `enable_thinking false`. Raw rows in `ab-engine.jsonl`.

The MTP draft acceptance and the number of tokens per round are reported because they are the same
in both arms: the arms generate **the same tokens**, so the comparison is of the same work done
at different speed.

## Results

### Engine-level, paired and order-swapped (UD-IQ4_XS, 262144 context, int8 KV)

| Pair | Order | Arm | Decode tok/s | Prefill tok/s | TTFT s | Draft accept | Tokens/round | pool ms/round | host ms/round |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A→B | A `STRATA_SM70_TABLE=0` | 28.16 | 1626.8 | 5.91 | 0.612 | 2.23 | 46.5 | 19.1 |
| 1 | A→B | **B `=1`** | **30.92** | 1633.1 | 5.87 | 0.612 | 2.23 | 39.9 | 12.6 |
| 2 | B→A | **B `=1`** | **32.94** | 1619.0 | 5.81 | 0.612 | 2.23 | 37.3 | 18.2 |
| 2 | B→A | A `STRATA_SM70_TABLE=0` | 28.64 | 1629.3 | 5.99 | 0.612 | 2.23 | 45.8 | 18.6 |

**Decode: +9.8% (pair 1) and +15.0% (pair 2); the two A runs agree within 1.7%** (28.16 / 28.64),
so the difference is larger than the run-to-run spread. Prompt throughput is unchanged
(1619–1633 tok/s in every run).

For context on the version upgrade itself, the same arm on the **0.1.39** engine measured
**29.82 tok/s** (single run, `bench/results/2026-10-05-community-v100-sxm2`); 0.1.41's default is
28.2–28.6, i.e. the same within this machine's spread — **the 0.1.41 gain on a V100 is in the
opt-in kernels**, which is exactly what this report is about.

Q2_0 (official GSQ-RCO Q2_0, fp16 KV, `--prefill 8192`) with `STRATA_SM70_TABLE=1` measured
**69.70 tok/s decode / 1561.5 prompt tok/s**, against **65.16 / 1571.8** for the same arm on 0.1.39
(single runs, one each).

### `tools/ab_engine.py`, 3 rounds, medians

| Request | Prompt tokens | A decode | B decode | A prompt | B prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| chat0 | 34 | 26.2 | 21.5 | 34.9 | 31.1 |
| chat1 | 27 | 31.4 | 30.3 | 33.0 | 30.2 |
| chat2 | 29 | 27.2 | 26.4 | 29.7 | 30.6 |
| long | 6345 | 30.2 | **31.7** | 291.9 | 287.9 |

The three short chats do **not** resolve the two arms on this machine: their per-round decode
medians swing between 20.9 and 37.2 tok/s within one arm, more than the effect being measured
(the runtime is dominated by things that are not the decode kernels at a 30-token prompt).
The long-prompt arm, which does the real work, points the same way as the engine-level pairs
(+5%). Reported as measured, not as support.

### Correctness: the arms are bit-identical

All four engine-level runs (both arms, both pairs) produced **the same generated token sequence**:

```text
pair1 A  len=1508  sha256[0:24]=258d9017223cd4d36c7b47bf
pair1 B  len=1508  sha256[0:24]=258d9017223cd4d36c7b47bf
pair2 B  len=1508  sha256[0:24]=258d9017223cd4d36c7b47bf
pair2 A  len=1508  sha256[0:24]=258d9017223cd4d36c7b47bf
```

and the draft acceptance (142 of 232) and tokens per round (2.23) are identical too. The kernels
therefore change the speed and not the arithmetic, as `docs/NVIDIA_V100.md` claims.

## Limitations

- One machine (Windows, 96 GB RAM, no AVX-512, Gen3 x16), one model (UD-IQ4_XS), greedy decoding.
  The decode numbers move ±10–20% between runs on this box (the adaptive expert tier); that is why
  the arms are compared **paired and order-swapped**, not across sessions.
- Loading time is not part of the timings (≈2.5 min per arm, excluded).
- The short-chat cases of `tools/ab_engine.py` are below the noise floor here; only the long-prompt
  arm is informative.
- No needle test, no image test in this report; the expert cache state after load is the shipped
  profile's, not a warmed one.
