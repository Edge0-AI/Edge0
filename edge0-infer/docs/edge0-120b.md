# Edge0 120B

Edge0 120B is a mixture-of-experts model for a single NVIDIA GPU. It runs on **edge0-strata**, the engine built for it (see [Two layers](architecture.md#two-layers)): the `edge0-strata/` directory of the Edge0 repository, a fork of [Strata](https://github.com/Niko1221/Strata). `edge0-infer serve` and `edge0-infer chat` start edge0-strata for this model, wait until it is ready and talk to it on a loopback port, so clients use the same commands and endpoints as for every other model. [Edge0 145B](edge0-145b.md) runs the same way.

This page is the reference for running Edge0 120B with `edge0-infer`; [Getting started](getting-started.md#edge0-145b-and-edge0-120b-edge0-strata) walks through the steps. The model itself (architecture and measured speed) is described on the model page of Edge0 145B and Edge0 120B, `docs/models/edge0-145b.md` in the Edge0 repository.

## Requirements

- Linux or Windows with an NVIDIA GPU and CUDA 13.
- At least 16 GiB of GPU memory and 20 GiB of system memory.
- `edge0-infer` built as in [Build](getting-started.md#build) (the `cpu` preset on Linux) or [Windows](getting-started.md#windows); it needs no GPU backend of its own for these models.
- edge0-strata built as described in `edge0-strata/EDGE0.md` in the Edge0 repository. `edge0-infer` uses the engine `edge0-strata/build/strata` (`strata.exe` on Windows) and the Python environment `edge0-strata/.venv` it produces.

## Model directory

Use each released model folder as provided: one model and one expert format per folder, with files laid out flat. `edge0-infer` reads only the `model_name` field of the folder's `model.json` and ignores its other fields (`architecture`, `args`, `env` and the rest); a `model.json` that contains `format_version` is an asset manifest of the native engine instead. The folder also holds a `start.sh` and a `model-bench.json`, which `edge0-infer` does not use.

| model_name | Registration (default alias) | Expert format |
| --- | --- | --- |
| `edge0-120b-q3k` | `edge0_120b` | `q3k` |
| `edge0-120b-fp4` | `edge0_120b` | `fp4` |

The expert format `q3k` is Q3_K (Q3_K gate/up, IQ4_NL down) and `fp4` is NVFP4.

The commands below use the folder that contains `model.json`, for example `Edge0-120B-q3k`, not the folder that holds several models:

```sh
export EDGE0_120B=/path/to/Edge0-120B-q3k
```

On Windows, set it with `$env:EDGE0_120B = "C:\path\to\Edge0-120B-q3k"` in PowerShell and run `build\windows\bin\edge0-infer.exe` instead of `build/cpu/bin/edge0-infer`.

| Path | Content | Required |
| --- | --- | --- |
| `model.json` | Model identifier (above) | yes |
| `experts.bin` | Compressed experts in the directory's format, 48 layers x 512 | yes |
| `labels.bin` | The projection group of each expert, read beside the expert file | yes |
| `dense.bin`, `index.txt`, `native_experts.txt` | The rest of the expert pack | yes |
| `scales/` | Per-layer column scales | for Q3_K folders |
| `latent_dense_q8.bin` | Expert projections, Q8 | for `du=q8` |
| `latent_dense_q4.bin` | Expert projections, Q4 | for `du=q4` |
| `model-skeleton.gguf` | Attention, routing, shared experts, embeddings and output | yes |
| `ple-fp8.gguf` | Per-layer embedding table (FP8) | yes |
| `profile.bin` | Expert ranking used to choose which experts to keep on the GPU | yes |
| `mtp/` | Draft layer for speculative decoding | yes |
| `tokenizer/` | Tokenizer | yes |

`edge0-infer`'s asset tools do not produce this directory. Check it with:

```sh
build/cpu/bin/edge0-infer inspect --model "$EDGE0_120B" --json
```

The report lists files with `exists` and `bytes`; `ok` is `true` (exit code 0) when the selected configuration is complete.
The expert format comes from `model_name` and is not a model option; `--model-option experts=` is rejected.
D/U defaults to `q4`; add `--model-option du=q8` to check the other choice.
Only Q3_K folders list or require `scales/` (in the report its `required` value is `experts=q3k`).

When files are missing, standard output remains valid JSON and standard error lists the missing files.
A hint suggests another D/U choice only when its files complete the directory.
Starting `serve` or `chat` uses the same defaults and reports missing files without falling back.

## Serve

```sh
build/cpu/bin/edge0-infer serve --model "$EDGE0_120B"
```

`edge0-infer` finds edge0-strata through `--strata-dir`, then `EDGE0_STRATA_DIR`, then an `edge0-strata/` directory beside an ancestor of the `edge0-infer` executable (`edge0-infer.exe` on Windows) (in the Edge0 repository layout, the sibling of `edge0-infer/`). The `Listening:` line appears when edge0-strata has loaded the model and is ready; `--strata-ready-timeout` bounds the wait (default 1800 seconds). While it loads, `serve` prints edge0-strata's server messages on its standard error; the engine process's own output is not shown or kept. The model name is its registration identifier (`--alias` changes it).

Once `Listening:` appears (default port 8000):

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "edge0_120b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Generation requests are forwarded to edge0-strata unchanged, and `/health`, `/v1/models` and `/status` are answered by `edge0-infer serve` in its own formats (`/health` also carries edge0-strata's own fields, and edge0-strata's `/v1/status`, `/metrics` and `/props` are passed through, so its tools work against `edge0-infer serve`); see [Delegated models](local-server.md#delegated-models) for the options, the forwarding rules and what each `/status` field holds.

## Chat

```sh
build/cpu/bin/edge0-infer chat --model "$EDGE0_120B"
```

`chat` starts its own edge0-strata; it does not connect to a running `edge0-infer serve`. Do not run the two on the same GPU at the same time. To chat with a model that `serve` already runs, use an OpenAI- or Anthropic-compatible client against it.

The commands are those of the native chat (`/file`, `"""`, `/think on|off`, `/new`, `/regen`, `/quit`). Thinking is on unless `/think off` is given. Reasoning is printed to standard error between `[thinking]` and `[/thinking]`, the answer to standard output, and after each turn a line with the reused and computed tokens, the first-token time, the decode speed and the stop reason. Ctrl-C stops the current answer; at the prompt it stops edge0-strata and exits (exit code 130). edge0-strata's messages, including those printed while it loads, are written to a temporary log file whose path chat prints at start (the engine process's own output is not kept); the file is removed when chat ends normally. If edge0-strata fails to start or exits during the session, chat prints the last lines of its log, keeps the log file, prints its path and exits with code 1; invalid options exit with code 2 before anything starts.

## Options

| Option | Values | Default |
| --- | --- | --- |
| `--device` | `cuda` or `cuda:<n>` | `cuda:0` |
| `--model-option du=` | `q8` or `q4` | `q4` |
| `--model-option ple-io=` | `direct` (read the embedding table from disk as needed) or `ram` (load it into memory; to keep it locked there, raise the locked-memory limit, e.g. `ulimit -l unlimited`) | `direct` |
| `--context` | tokens | 131072 |
| `--strata-arg` | one word of edge0-strata's engine command line; repeat it for each word, e.g. `--strata-arg --pool-workers --strata-arg 24` | none |

The model settings (automatic expert cache and prefill, speculative decoding, and int8 KV cache) are in `edge0-strata/edge0/models/edge0-120b.json`. CPU worker count, PCIe mode and fraction, VRAM reserve, and KV residency use the engine's defaults unless given with `--strata-arg`. This is meant for tuning to one machine, for example:

```sh
build/cpu/bin/edge0-infer serve --model "$EDGE0_120B" \
  --strata-arg --pool-workers --strata-arg 24 --strata-arg --pcie-mode --strata-arg dma \
  --strata-arg --pcie-frac --strata-arg 0.1 --strata-arg --vram-reserve-mib --strata-arg 700 \
  --strata-arg --kv-resident --strata-arg 32768
```

The words are appended in the order given, after the template's arguments; write each option and its value as two words (not `--option=value`). An option the template already sets (such as `--kv` or `--pack`), `--max-context` (use `--context`) and an empty value are rejected before anything starts. Other words are not checked by `edge0-infer`: edge0-strata decides, and since the engine process's own output is not shown, a word it does not accept appears as a failed start (run edge0-strata directly to see why). The template requires launch contract version 4; an older engine template is rejected with both versions in the error.

## CUDA libraries on Windows

The default interpreter is `edge0-strata/.venv/Scripts/python.exe`
on Windows and `edge0-strata/.venv/bin/python` on POSIX; `--strata-python` overrides it.
The template expands `${EXE_SUFFIX}` to `.exe` on Windows and to an empty string on POSIX.

The CUDA toolkit sets `CUDA_PATH`. Existing `bin/x64` and `bin` directories under that path
replace the template's `${CUDA_LIB_DIR}` item, in that order, and are prepended to the
server's `PATH`. On POSIX that item expands to no directories and the inherited environment
is unchanged. If neither Windows directory contains `cublas64_13.dll`, the CLI warns,
lists the directories checked, and still launches: the inherited `PATH` may already provide
the DLL. A missing runtime DLL can produce exit code `0xC0000135`.

## Known limitations

- The native API does not support this model: `inspect_assets`, `resolve_load_configuration` and `load_model` return an `unsupported` error for a `model.json` that has no `format_version`, before reading any other file. Of the commands, only `serve`, `chat` and `inspect` (with or without `--json`, without `--digest`) accept it; `run`, `bench`, `describe`, `tokenize`, `prompt` and `forward` refuse it.
- Concurrent requests wait in edge0-strata's queue instead of receiving 429 as with the native server.
- `/status` holds no memory breakdown or expert statistics for this model; those fields are null. Progress of a request in flight is counted by `edge0-infer serve` from the text fragments it forwards (one token per fragment), and only for streaming requests; the finished records use the engine's own numbers (see the `Delegated` column of the [status fields](local-server.md#status-and-lan-authentication)).
- Identical requests at temperature 0 can produce different answers: edge0-strata's decoding is not bit-for-bit repeatable.
