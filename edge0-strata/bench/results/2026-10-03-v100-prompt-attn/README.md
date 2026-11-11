# V100 (sm_70): a tensor-core QSA prompt attention (2026-10-03)

On sm_70 the prompt path ran the QSA attention on the decode kernel, one query at a time with FP32 FMAs. `prompt_attn_v70_kernel`
(`src/kernels/cuda/qsa_prompt_attn.cu`) does the two matrix products on Volta's tensor cores (`mma.sync.m8n8k4`).
`STRATA_PROMPT_ATTN_OLD=1` is the old path, so every number below is the same binary with and without that switch.

## Rig

- Tesla V100-PCIE-32GB (sm_70, PCIe gen3 x16), driver 550.120, one card used; a second V100 and a Quadro RTX 4000 (sm_75) are in the
  PC and hidden with `CUDA_VISIBLE_DEVICES`. Intel Core i9-7960X (AVX-512), 125 GB RAM, Linux 6.8, CUDA 12.8, GCC 13.
- Engine: `origin/main` at `99f3dbd` (0.1.35 + 83 commits) with this change, built with
  `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`.
- Model: Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` (3 shards), packed with `tools/iq_pack.py --compat-bf16`, MTP head from
  `tools/mtp_fetch.py`. Not one of setup's models (see the limits). `--max-context 204800`, `--kv int8`, `--spec 2`.

## Prompt speed

One server start per configuration (cold), a 2K warm-up request, then three prompts of random words, each answered greedily
(temperature 0, 48 tokens). The numbers are the server's own line `strata serve: prompt N tokens = 0 reused + N read in T ms`. One run per
cell; repeated starts of the same configuration gave the same seconds.

| Prompt tokens | old path | new kernel | gain | read time, old -> new |
| ---: | ---: | ---: | ---: | ---: |
| 7,194 | 1,010 tok/s | 1,164 tok/s | +15% | 7.1 s -> 6.2 s |
| 28,650 | 1,063 tok/s | 1,251 tok/s | +18% | 26.9 s -> 22.9 s |
| 114,338 | 973 tok/s | 1,123 tok/s | +15% | 117.5 s -> 101.8 s |

Decode is unchanged within the noise (38-51 tok/s, 48 tokens per request): it does not use this kernel.

## Where the time went (nsys, one V100, 28,650-token prompt, GPU kernel time of the whole capture: prefill plus 128-193 decode tokens)

`strata-1gpu-32k-*.nsys-rep`; the "before" capture is 0.1.35 (`d9ab843`), the "after" one is the same base with this change and the BF16 change
of #540 / the closed #593 (BF16 GEMMs through FP16; that one only removes `magma_sgemmEx`, which is not in this table).

| | before | after |
| --- | ---: | ---: |
| QSA attention on the decode kernel (`attn_chunk_kernel<1>`) | 5.78 s (19.2%) | 0.18 s (decode steps only) |
| `prompt_attn_v70_kernel` | - | 2.16 s (9.0%; 12% of the prompt part) |

## Accuracy

`qsa_prompt_attn_parity` (synthetic, no model; build with `-DSTRATA_PARITY_PROMPT_ATTN=ON`), V100, 32K context, 2,048 queries per chunk, output scale 3.6:

| KV | error vs FP64, old kernel | error vs FP64, new kernel | time per chunk, old -> new |
| --- | ---: | ---: | ---: |
| int8 | 2.4e-6 | 5.1e-6 | 38.1 -> 13.4 ms (2.8x) |
| fp16 | 1.9e-6 | 5.3e-6 | 37.3 -> 28.3 ms (1.3x) |

The 1,500- and 2,100-token contexts also pass (int8: 3.2x and 2.2x). The test's Q4_0 cases are skipped below sm_80, where the dispatcher keeps the
old kernel (upstream's mode 4 is sm_80+ only).

## Output checks

- **Needle recall** (`tools/needle_bench.py --lengths 8k,32k,128k --depths 10,30,50,70,90`, greedy, thinking off, a fresh server per
  configuration; prompts of 7.9K, 32.0K and 122.7K tokens): **15 of 15 found with the old path, 15 of 15 with the new kernel.** The read times of
  the later depths are shorter because the server reuses the checkpoint of the text before the needle.
- **Greedy output** (48 tokens, the three prompts above): identical for the 28,650- and 114,338-token prompts. For the 7,194-token prompt the
  answer starts the same and **differs later**: the kernel sums in another order and a greedy decode can follow a near tie elsewhere. A recall test
  with a code word does not tell small shifts of the output distribution apart, and a teacher-forced top-1 comparison was not run.

## Distribution check (teacher-forced, against the old path and against controls)

The method of `docs/UNSLOTH_Q4.md`: `STRATA_LOGPOS=<file> STRATA_LOGPOS_TOPK=256` on a serve engine (`--short-read 640 --adapt-every 100000`, greedy,
`--max-context 204800`). Each prompt is a chat: a long first message (8,543 / 32,906 / 99,040 tokens of this repository's docs and code, read by the
batched prompt path, i.e. by the kernel under test), a short reply, and a last message of ~600 tokens of other repository text that goes through the verify
windows, where every token is scored: 617 positions per prompt. KL is old || other over the old run's top 256 plus a bucket for the rest.

| comparison | KL mean (8.5K / 33K / 99K) | argmax same | top-10 overlap |
| --- | --- | --- | --- |
| old path vs old path (a second run) | 0 / 0 / 0 | 100% | 100% |
| new kernel vs new kernel (a second run) | 0 / 0 / 0 | 100% | 100% |
| **old path vs new kernel** | 0.028 / 0.040 / 0.116 | 97.7 / 93.2 / 86.4% | 93.3 / 91.9 / 88.4% |
| control A: old path, BF16 GEMMs without the FP16 detour (`STRATA_BF16_VIA_F16=0`) | 0.026 / 0.051 / 0.090 | 97.2 / 91.4 / 91.2% | 92.9 / 91.4 / 88.5% |
| control B: old path, `--prefill 2048` instead of auto (8192) | 0.023 / 0.035 / 0.038 | 97.9 / 94.8 / 95.6% | 92.9 / 92.2 / 89.7% |

Median KL old vs new: 0.0007 / 0.0088 / 0.022; p99: 0.35 / 0.39 / 1.12 (heavy-tailed). The perplexity of the true tokens moves a lot under every
change (old 365 / 127 / 24; new 346 / 192 / 12; control A 395 / 161 / 18), so it says nothing about quality here.

Reading: the model amplifies small numeric differences, and both controls, which change nothing but the order of the rounding, move the distribution as
much as the kernel does. At 99K the kernel is somewhat above both controls; this is one document and 617 positions, so a real effect there is neither
confirmed nor excluded.

**The kernel on the model's own data.** A temporary check (not in this PR) ran the decode kernel beside the new one on the same inputs, for every 8th batch of
32 queries, in all 12 QSA layers, on the three prompts above (positions up to 98K, ~5 million (query, head) rows of 256): ||new - old|| / ||old|| per row is
4.4e-7 on average and 4.8e-6 in the worst row; the largest absolute difference is 4e-5 against values up to 31. No trend over the layers or the context
length. That is FP32 rounding level, so the differences above come from the model's sensitivity, not from the kernel's arithmetic.

## Limits

- sm_70 only (`__CUDA_ARCH__` 700-749). Q4_0 KV (`--kv q4_0`) and the QSA block scores (`block_scores_kernel`, the warp kernel below sm_80) still use their
  old kernels on Volta.
- One request, one card. The Unsloth IQ4_XS pack is not a setup model; the GSQ-RCO quants were not run on this rig.
- The profile above is one capture of one prompt. The distribution check is one document per length; the 99K row is the least certain.

## Reproduce

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PARITY_PROMPT_ATTN=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata qsa_prompt_attn_parity -j
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5       # the accuracy table
STRATA_PROMPT_ATTN_OLD=1 <start script>                                   # the old path; without the variable: the new kernel
python tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 8k,32k,128k --depths 10,30,50,70,90
```
