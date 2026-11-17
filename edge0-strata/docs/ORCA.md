# OrcaRouter IQ3_XXS compatibility

This page is for people who want to run `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` (**IQ3_XXS**, two shards,
85.20 GB) with Strata. It is a manual conversion workflow from a [source checkout](SOURCE_INSTALL.md). OrcaRouter is
not an installer menu entry, and there is no `--family orca` command.

## Quick start

1. Build Strata normally, then pack the model ([Preparation](#preparation)).
2. Prepare the optional draft head (MTP) with the same section's commands.
3. Save the config from [Local server](#local-server) and start the server.

## How it works

The model has Strata's Qwen4Exp geometry. Ordinary quantization also compresses the hyperconnection, small attention
and PLE projections, which Strata's kernels read as BF16. `tools/iq_pack.py --compat-bf16` expands only those
projections, rounding to BF16 with nearest-even. This adds rounding relative to the GGUF's dequantized weights. It does
not recover the original BF16 checkpoint or its quality.

Expert weights, native attention weights, embeddings and the 28.8 GB PLE lookup table stay unchanged, and the lookup
table stays on disk. The pack records its conversions in `compat-bf16.json`. Hugging Face snapshot symlinks work. All
split shards must finish downloading before packing. Pass the snapshot filename, not its hash-named blob target.

## Preparation

Python needs numpy, regex and gguf-py from the pinned llama.cpp dependency. `STRATA_GGUF_PY` can point to its
`gguf-py` directory.

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
```

Use this model's own tokenizer, exported into the pack. Do not share another model's `dense.bin`, and do not rename
the Orca files to pass as one of setup's GSQ-RCO models.

The persistent server can use the MTP runtime. It is optional: without `--mtp` the server drafts by lookup only.

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

This uses the original model's draft head, as setup does for Swift. The target model verifies every proposal. Measure
draft acceptance and speed for this fine-tune; the original model's numbers do not carry over.

## Local server

Save the following as `strata-orca-iq3_xxs.json` at the repository root. Replace both `/path/to/` entries with the
**same first shard**, because Orca's PLE table is in shard 1. The config uses a 32K context and 512-token prefill
chunks. The expert cache sizes itself to the available VRAM. The expert arena alone needs about 49.8 GiB of free system
RAM, plus draft and runtime buffers and your other applications.

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs",
    "--native", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-iq3_xxs.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

Run it from the repository root:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-orca-iq3_xxs.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`. API clients use `http://127.0.0.1:8080/v1`. This example is text-only;
vision is not configured.

## Scope

- IQ3_XXS is the only validated Orca quantization.
- IQ3_M uses Q5_0 expert down matrices, which the native GPU expert path does not support.
- The installer model menu does not list this model.
- Packing tests: `.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`.

## History / older measurements

> **Measured on:** Linux, RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB RAM, 2026-09-27, the configuration above.

| Check | Result |
| --- | --- |
| Packing | 460 tensors converted, 1.39 GiB of converted BF16 weights, 1.43 GiB total packed dense weights; the 48 expert layers stay IQ3_XXS gate/up and IQ4_NL down |
| Tests | all eight packing tests, 17 server tests and the GPU gated-residual parity check passed |
| Server smoke tests | arithmetic, Python code, translation, a longer explanation, multi-turn reuse (25 prompt tokens reused), streaming cancellation and a correct answer afterwards; the engine PID stayed unchanged |
| Decode speed | 111-token explanation at 77.7 tokens/s (engine decode timing), 1.77 s total request time |

These are smoke tests and one short measurement, not a quality benchmark, a full-context stress test or a general
throughput claim.
