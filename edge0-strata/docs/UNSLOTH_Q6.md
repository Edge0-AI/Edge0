# Unsloth UD-Q6_K_XL

This page is for people who want to run Unsloth's 6-bit Qwen3.8-Flash-Next file with Strata, and for people who pack it
by hand. UD-Q5_K_XL and Q8_0 are set up the same way. Quality, speed and images across the Unsloth sizes are in
[QUANTS.md](QUANTS.md). UD-Q4_K_XL, with the full setup details, is in [UNSLOTH_Q4.md](UNSLOTH_Q4.md).

The file is [unsloth/Qwen3.8-Flash-Next-GGUF, `UD-Q6_K_XL`, revision `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL),
the same model the other packs use.

## Quick start

```sh
strata launch --family unsloth --model UD-Q6_K_XL
```

Setup keeps every expert in RAM from about 118 GB of RAM. Below that it asks and keeps a RAM budget, as for UD-Q4_K_XL
([What setup does](UNSLOTH_Q4.md#what-setup-does)). For UD-Q5_K_XL and UD-Q6_K_XL it writes `STRATA_MMQ_KQ=1`, the
K-quant MMQ prompt kernels, into the run config.

## What is in the file

Six shards, 169,165,382,688 bytes (169.2 GB) together ([sizes and SHA-256 from the HF tree API](https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL)).
Shard 1 holds only metadata. Shard 2 holds only the output head (`output.weight`, `output_hc_*`, all Q8_0 but
`output_hc_norm`, F32). Shard 3 is exactly the PLE table.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf` | 10,946,624 | `8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf` | 682,434,912 | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf` | 54,400,261,312 | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf` | 49,814,108,512 | `0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf` | 49,419,317,344 | `9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf` | 14,838,313,984 | `e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f` |

Formats, read from the shard headers with `tools/gguf_reader.py`:

- **Routed experts.** `ffn_gate_exps` and `ffn_up_exps` are Q6_K in 47 of 48 layers and Q8_0 in layer 2. `ffn_down_exps`
  is Q8_0 in every layer (891.3 MB per layer at 640x2560x512, against Q6_K's 688.1 MB for gate or up). The pack's
  `native_experts.txt` has 47 rows of ggml types 14/8, and layer 2 is 8/8. The engine multiplies Q6_K gate/up natively on
  the GPU, and the CPU has the ggml block layout (210 bytes per 256 elements). The per-role layout in
  `native_experts.txt` v4 handles shards whose boundaries cut through a layer.
- **Everything else.** Attention and shared-expert projections, SSM, hyper-connections, the PLE key/value, the token
  embedding and the output head are Q8_0, or F32/BF16 where the engine reads floats. This is the same pattern
  [UD-Q4_K_XL](UNSLOTH_Q4.md) handles with `tools/iq_pack.py --compat-bf16`.
- **The PLE table** (`per_layer_token_embd.weight`, [160, 320001536]) is **Q8_0**, in its own shard 3. A row is 5 blocks
  of 32 elements at 34 bytes, 170 bytes in all, and the whole table is 54.4 GB (UD-Q4_K_XL's IQ4_NL table is 28.8 GB).
  The engine reads Q8_0 rows natively. The FP8 table of [tools/ple_fp8_pack.py](../tools/ple_fp8_pack.py) needs the
  original checkpoint, which this file is not a substitute for.

Q6_K gate/up costs 210 bytes per 256 elements, against Q4_K's 144, so the routed experts' share of the file grows about
45% per element read. Decode is accordingly slower than UD-Q4_K_XL's on the same hardware. The table row gather grows
170/90 = 1.89x over IQ4_NL.

## The pack (by hand)

Same workflow as [UD-Q4_K_XL](UNSLOTH_Q4.md#the-pack-by-hand), from the repository root:

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  --out packs/ud-q6_k_xl --compat-bf16
```

**`--compat-bf16` is required**, for the same reason as there (the 195 small Q8_0 projections the engine reads as BF16).
Do not add `--experts-bin`; the experts stay in the GGUF files. The pack on the finished file has 1079 tensors indexed,
303 served natively, 460 converted (264 exact, 196 rounded, max abs error 0.0144), 195 compat-BF16 tensors (1.20 GiB),
an arena of 1.38 GiB, and `native_experts.txt` v4 with the per-role shard column on layers 20 and 41. The packer's tests:
`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`.

## The server

Like UD-Q4_K_XL's [config](UNSLOTH_Q4.md#the-server), with the Q6 paths and names. `--ple-gguf` is not needed: the engine
finds shard 3 by the table's name. On a machine whose RAM holds the roughly 110 GB of routed experts plus the rest, drop
`--resident-budget-gib` and `--expert-cache-device1` from the Q4 config. The 54.4 GB table is mapped (the default). Do
not switch it to `--ple-io ram` unless the RAM really is there to lock it. `mtp/rt` is the base model's draft layer.

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/ud-q6_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q6_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q6_k_xl",
  "log": "strata-ud-q6_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

On a 64 GB machine, start from UD-Q4_K_XL's RAM-budget advice. The budget matters the same way, and everything it leaves
out comes from the SSD.

## Kernels and tests

- Decode windows: the grouped expert kernels (`native_expert_grouped`) take Q6_K gate/up (a pinned copy of llama.cpp's
  `vec_dot_q6_K_q8_1` behind `Fmt<14>`). Per-token MMVQ (`native_mmvq`) has Q6_K too.
- Prompt path: llama.cpp's MMQ prompt kernels instantiate Q6_K in the K-quant library (`STRATA_MMQ_KQ=1`). Without it
  the prompt takes the FP16 path.
- Tests: `native_grouped_parity` loops Q6_K/Q8_0 pairs, `prefill_mmq_kquant_test` runs Q6_K gate/up products at
  1280x2560 against a double-precision reference, and `native_expert_parity --synthetic q6_K/q8_0` checks GPU and CPU
  against ggml's dequantized weights. `ple_reader_test --selftest` runs 90, 110 and 170-byte PLE rows, and
  `ple_q8_parity` matches ggml's reference dequantizer bit for bit through both Mmap and Direct.
- Not measured on this page: quality against llama.cpp on this file ([QUANTS.md](QUANTS.md) has the Unsloth quality
  comparison).

## History / older measurements

> **Measured on:** Linux, RTX PRO 4500 Blackwell (compute capability 12.0) plus a second GPU, Threadripper PRO 5975WX
> (AVX2, no AVX-512), 30 expert-pool workers, 499 GB RAM, 2026-10-04. One code-review session.

| Engine | Experts loaded | Prompt | Decode | Drafts accepted |
| --- | --- | --- | --- | --- |
| 0.1.39 | 101.75 GiB at 8.43 GiB/s | 14,915 tokens cold at 3,176 tok/s | 5,799 tokens in 74.1 s (78.3 tok/s) over 14 turns, ending at 44,160 tokens of context | 3,485 of 4,406 (79.1%) |
| 0.1.38 | | 249,193 tokens cold at 2,802 tok/s | 1,079 tokens at 69.0 tok/s | |
