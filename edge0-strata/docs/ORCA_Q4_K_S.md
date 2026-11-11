# OrcaRouter Uncensored Q4_K_S (manual setup)

This path uses the three original `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` Q4_K_S shards without rewriting their expert, embedding or PLE table bytes. It is separate from the installer model menu. Keep all three shard filenames together.

The ordinary GGUF quantizes some small projections which Strata reads as BF16. `--compat-bf16` dequantizes and rounds those projections to BF16 in the separate pack; it does not restore their original precision. The Q5_0 PLE table remains in the source GGUF and currently uses the mapped reader (`--ple-io mmap`). The Q4_K gate/up experts and Q5_0/Q5_1 down experts remain in GGUF form, including layers whose three tensors cross shard boundaries (each role is read from its own shard). The GPU expert kernels take Q4_K with Q5_1 (or Q8_0) downs; Q5_0 downs are not among them yet, and the engine refuses to start on a pack with such a layer (it names the layer).

## Build and prepare

Build Strata with CUDA, native experts, `STRATA_ORCA_Q4KS_MMQ=ON` (and `STRATA_BUILD_TESTS=ON` for the two parity programs below). This opt-in compiles the Q5_0 GGML MMQ instance for CUDA or HIP (the Q4_K and Q5_1 instances come with the engine's K-quant MMQ build on CUDA); the usual Q8_0 draft-layer instance remains in both builds. HIP also needs `STRATA_PREFILL_MMQ=ON`. Select your card's compute capability, for example `89` on an L40S. On Windows with Ninja and CUDA 12.6, use the Visual Studio developer environment and `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` to keep the CUDA runtime linkage consistent. The default pinned ggml checkout is supported by the project build.

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_ORCA_Q4KS_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata native_expert_parity ple_q5_parity
python tools/iq_pack.py --gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --out packs/orca-q4_k_s --compat-bf16
```

`iq_pack.py` needs NumPy and the `gguf-py` module from the pinned llama.cpp dependency (`STRATA_GGUF_PY` can point to that directory). It writes a roughly 1.43 GiB dense pack and a native-expert manifest. The original 112 GB of model shards are read in place; `experts.bin` is optional and is not written by this command.

Check the changed kernels against the real files before starting the service:

```sh
build/native_expert_parity /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00002-of-00003.gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00003-of-00003.gguf 0 5 6 20 47
build/ple_q5_parity /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf
```

## Engine flags

Pass the same first shard to `--native` and `--ple-gguf`; Strata discovers the other two shards by name. For a first text-only run, use 32K context and a modest cache. Once that passes, increase the context and cache with memory measurements on the target machine.

```sh
--pack packs/orca-q4_k_s --native /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --ple-gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --ple-io mmap --expert-profile data/expert-profile.bin --expert-cache auto --prefill 512 --spec 4 --mtp /path/to/mtp/rt --max-context 32768 --kv int8
```

For two GPUs in persistent server mode, add `--layer-split auto --split-device 1` (with CUDA0 and CUDA1 visible). The usual Strata MTP runtime is built from the original model's draft layer; draft acceptance on the Uncensored fine-tune must be measured. Keep this model's own tokenizer from the pack. `--expert-cache-per-layer` on a native pack gives every layer the same number of slots, each the size of that layer's own blob. `--expert-cache N` is still the budget of N largest blobs.

This port does not claim that Q4_K_S beats llama.cpp on a given computer. Compare the same model bytes, prompt, context, output length and GPU load; report prefill and decode separately.
