# Image generation (Qwen-Image-2.1-Turbo)

This page is for people who want Strata's server to make images through `POST /v1/images/generations`, OpenAI's
images API. It is optional, off unless you ask for it, and separate from reading pictures in chat (for that, use
`strata launch --vision on` with a supported model and reader).

It needs a [source checkout](SOURCE_INSTALL.md) and build tools. The steps below are not part of a normal app install.

The model is [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (8 steps, CFG 1) in
[Unsloth's GGUF quants](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF), run by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`sd-cli`), which setup compiles for your GPU.
It is under the **Qwen Research License**, not Apache 2.0. Read
[its license](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) before you use the images
commercially.

## Quick start

```sh
./setup.sh --setup --images qwen-image          # Linux
START-HERE.bat --setup --images qwen-image      # Windows
```

For an install that is already set up, without running the whole setup again:

```sh
.venv/bin/python tools/image/install.py --config strata-<model>.json
```

Then request an image:

```sh
curl http://127.0.0.1:8080/v1/images/generations -H "Content-Type: application/json" \
  -d '{"prompt": "a cartoon sloth mascot waving, flat vector illustration", "size": "1024x1024", "seed": 42}'
```

## What setup does

Setup compiles `sd-cli` (stable-diffusion.cpp at a pinned commit) into `engine-image/`, downloads three files into the
data folder's `models/qwen-image-2.1-turbo/`, checks each against its SHA-256 and writes an `"images"` section into the
model's run config.

| Piece | File | Size |
| --- | --- | --- |
| Denoiser, chosen by GPU memory | `qwen-image-2.1-turbo-Q4_K_M.gguf` up to 8 GB, `Q6_K` up to 12 GB, `Q8_0` above | 4.2 / 6.3 / 7.6 GB |
| Text encoder | `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` | 5.1 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB |

### Setup options

| Option | Effect |
| --- | --- |
| `--image-quant Q4_K_M\|Q6_K\|Q8_0` | Picks the denoiser. A bigger one than recommended is kept, with a warning. |
| `--image-backend cuda\|hip\|vulkan\|cpu` | Picks the backend. |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | Sets `images.gpu_policy` ([below](#sharing-the-gpu-with-the-text-model)). |
| `--images no` | Takes the section out of the config. |

### Backends

| Backend | Used for |
| --- | --- |
| CUDA | NVIDIA. Needs the CUDA Toolkit. |
| HIP | AMD cards with their own memory, on Linux, with the same ROCm as the engine. |
| Vulkan | AMD APUs and iGPUs (Radeon 780M, Strix Point and Halo), Intel Arc, and AMD under Windows. |
| CPU | Fallback. |

A Linux PC without the Vulkan development files gets the LunarG Vulkan SDK unpacked into `third_party/` (no sudo). On
Windows, install the Vulkan SDK first. On Vulkan, setup adds `--vae-tiling --vae-tile-overlap 0.25`. An untiled
1024x1024 VAE decode on a Radeon 780M is one GPU job long enough to trip the kernel driver's job timeout, and the
quarter overlap decodes in half the time of sd.cpp's default half overlap.

## Request fields

| Field | Meaning |
| --- | --- |
| `prompt` | The description (required). |
| `size` | `WIDTHxHEIGHT`, each side a multiple of 32 from 256 to 2048 (`max_side`). Default and `auto`: `1024x1024`. |
| `n` | Images, 1 to 4 (`max_n`). Image *i* uses seed + *i*. |
| `response_format` | `b64_json` (default) or `url` (`/v1/images/files/<random name>.png`; the server keeps the newest 64). |
| `seed` | Strata's: a whole number for a repeatable image. Default: random; the answer says which. |
| `steps` | Strata's: 8 (Turbo's schedule). Another count uses stable-diffusion.cpp's own schedule, which Turbo was not made for. |

The answer has OpenAI's shape (`created`, `data[].b64_json` or `data[].url`, `revised_prompt` = the prompt) plus
`data[].seed` and a `strata` object with the seconds, steps and backend. The settings are the model card's for
stable-diffusion.cpp: 8 steps, CFG 1, euler, Turbo's own sigmas, `--diffusion-fa`. An image from the API is the PNG
that plain `sd-cli` writes with the same arguments, byte for byte.

## Sharing the GPU with the text model

One job runs at a time. An image job takes the same turn a chat request takes, so a chat request waits for the image
and the other way round. The config's `images.gpu_policy` decides what the text model does meanwhile.

| Policy | The text model during an image job |
| --- | --- |
| `auto` (default) | `keep` when the image job fits next to it (an APU always), else `shrink` when the engine was started with `--vram-elastic`, else `unload`. |
| `keep` | Stays loaded and idle. |
| `shrink` | Gives VRAM from its expert cache (as `POST /v1/vram`) and takes it back afterwards. The conversation stays. |
| `unload` | Its conversation is saved to a file and the engine is stopped. The next start restores the conversation. `"reload": "eager"` starts it again right after the image; the default `lazy` starts it with the next chat request. |

Each image runs its own `sd-cli` process, so no image memory stays on the GPU between images. Loading the three files
takes a few seconds per image from the page cache.

## History / older measurements

> **Measured on:** seconds per image through the API, Q8_0, 8 steps, one `sd-cli` process per image with loading
> included. Strata version not recorded.

| GPU | Backend | 1024x1024 | 512x512 |
| --- | --- | --- | --- |
| RTX A4000 16 GB | CUDA | 27 s | 8.8 s |
| Radeon 780M (iGPU) | HIP | 131 s | 29 s |
| Radeon 780M (iGPU) | Vulkan | 179 s | 43 s |

On the 780M, HIP is faster but runs through the ROCm path whose queue evictions reset this GPU on the firmware tested
(a driver or firmware defect, not this feature; no reset was seen in these image runs). Vulkan is setup's default
there. `--image-backend hip` picks HIP.
