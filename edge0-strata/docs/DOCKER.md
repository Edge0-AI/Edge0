# Strata in Docker

Run Strata in a container with a prebuilt engine. Install Docker, the host GPU driver and the GPU-access
requirements in the table below. NVIDIA and AMD images can download and prepare a model on first start;
Intel Arc needs prepared model files. Keep the data volume to reuse downloads and settings.

> **On this page:** [Which image](#which-image-for-which-card) · [One command per card](#one-command-per-card) ·
> [Docker Compose](#docker-compose) · [Settings](#settings) · [The API key](#the-api-key) · [Connect your apps](#connect-your-coding-apps) ·
> [Watch and stop](#watch-it-and-stop-it) · [Update](#updating) · [Troubleshooting](#troubleshooting) ·
> [From the older image](#from-the-older-image) · [Build it yourself](#build-it-yourself)

Docker on a **Mac** has no access to the Mac's graphics chip (no Metal): use the Mac app or the Terminal
([MAC.md](MAC.md), [TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)). Docker Desktop on **Windows** passes NVIDIA cards through
(WSL 2) but not Radeon or Arc cards; on Windows use the native app for supported NVIDIA or AMD cards. Intel Arc has no native Windows engine; follow the Linux guide.

## Which image for which card

| Your graphics card | Image tag | Needs on the computer |
| --- | --- | --- |
| NVIDIA **RTX 20, 30, 40, 50** series (and the matching RTX A / Quadro cards) | `0.2.7-cuda13` | NVIDIA driver **580 or newer**, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| NVIDIA **GTX 10 (Pascal), Tesla P100, V100 (Volta)**, or any NVIDIA card whose driver is **575 to 579** | `0.2.7-cuda12` | NVIDIA driver **575 or newer**, the same toolkit |
| AMD **RX 9000** series and Radeon AI PRO R9700 (RDNA 4, gfx1200 / gfx1201) | `0.2.7-rocm-gfx120x` | the `amdgpu` kernel driver (Linux 6.8+) |
| AMD **RX 7000** series (RDNA 3, gfx1100 / gfx1101 / gfx1102) | `0.2.7-rocm-gfx110x` | the `amdgpu` kernel driver |
| AMD **Ryzen AI Max** ("Strix Halo", Radeon 8060S, gfx1151) | `0.2.7-rocm-gfx1151` | the `amdgpu` kernel driver |
| Intel **Arc** A-series and B-series (including Arc Pro B70) | `0.2.7-sycl` (if no image exists for a release, [build it](#build-it-yourself)) | the `i915` or `xe` kernel driver; **prepared model files** (see [Intel](#intel-arc)) |

**Measured on:** images built from the 0.1.45 engine packages. Size on disk: `cuda13` 2.2 GB, `cuda12` 2.9 GB, `rocm-gfx120x` 2.8 GB,
`rocm-gfx110x` 2.4 GB, `rocm-gfx1151` 1.5 GB. The download is about a third of that: `cuda13` 640 MB, `cuda12` 845 MB,
`rocm-gfx120x` 505 MB, `rocm-gfx110x` 330 MB, `sycl` 450 MB. The model comes on top, on the first start. Current sizes may differ.

Not sure which NVIDIA image? Take `cuda13` when `nvidia-smi` shows driver 580 or newer and the card is an RTX; otherwise `cuda12`.
A wrong pick does no harm: the container starts, says in plain words that the engine does not fit this card, and stops.

**Ready-made images are published from version 0.2.1 on**, to the GitHub Container Registry: `ghcr.io/niko1221/strata:<version>-<flavor>`
(for example `0.2.7-cuda13`), and `ghcr.io/niko1221/strata:latest-<flavor>` for the newest release of that flavor. Pin the version
tag if you want updates only when you decide. The examples below use these images. If you built an image yourself
([Build it yourself](#build-it-yourself)), use the tag you gave it (for example `strata:0.2.7-cuda13`) in their place.

## One command per card

Each command runs Strata in the background, keeps your files in the volume `strata-data`, and listens only on this computer
(`127.0.0.1:8080`). Choose `FAMILY` and `MODEL`, and generate your own key. The examples select ISTA-DASLab originals explicitly. The first start downloads the model (tens of GB): watch it with
`docker logs -f strata`. Already have the model's GGUF files? Add `-e MODEL_PATH=/models -v /your/folder:/models:ro` and nothing is
downloaded for the model itself ([Files you already have](#files-you-already-have)).

**NVIDIA**

```sh
docker run -d --name strata --gpus all -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_XXS -e API_KEY=$(openssl rand -hex 24) \
  ghcr.io/niko1221/strata:0.2.7-cuda13
```

**AMD**

```sh
docker run -d --name strata --device /dev/kfd --device /dev/dri \
  --group-add video --group-add $(getent group render | cut -d: -f3) -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_XXS -e API_KEY=$(openssl rand -hex 24) \
  ghcr.io/niko1221/strata:0.2.7-rocm-gfx120x
```

**Intel Arc**

```sh
docker run -d --name strata --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  -p 127.0.0.1:8080:8080 --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data -e FAMILY=qwen -e MODEL=IQ3_S -e MODEL_PATH=/models -v /srv/gguf/IQ3_S:/models:ro \
  -e API_KEY=$(openssl rand -hex 24) ghcr.io/niko1221/strata:0.2.7-sycl
```

What the options are for: `--ulimit memlock=-1:-1` lets the engine pin the RAM it uses for the model, `--shm-size 2g` gives
it shared memory, and `--stop-timeout 120` gives the model two minutes to unload when you stop the container (Docker's
default 10 seconds is too short for a big model). Do not put a memory limit on the container: Strata plans from the whole
computer's RAM and loads tens of GB.

To print the key you made: `echo` it before you start (it is in your shell history otherwise), or ask the running container:
`docker exec strata strata launch --show-key --preset docker`.

### Intel Arc

Strata's downloader prepares NVIDIA and AMD cards. For an Arc card the container **uses prepared model files**: the model's
GGUF files in a folder you mount (`-v /srv/gguf/IQ3_S:/models:ro` and `-e MODEL_PATH=/models`), and the prepared pack and MTP
layer in the data volume (`/data/Strata-data/packs/<model>` and `/data/Strata-data/mtp/rt`, made by the Intel installer on
this or another computer, see [INTEL.md](INTEL.md)). Without them the container tells you which file is missing and stops.
Nothing is downloaded or prepared for Arc inside the container. Arc B-series cards (xe driver) stream the experts from the
GGUF files, so the GGUF folder should be on a fast SSD.

## Docker Compose

Ready-made files are in [docker/compose/](../docker/compose): `docker-compose.nvidia.yml`, `docker-compose.amd.yml`,
`docker-compose.intel.yml` and `.env.example`. Copy the one for your card and its `.env.example`, edit, start:

```sh
cd docker/compose
cp .env.example .env              # set FAMILY=qwen, MODEL=..., API_KEY=...
docker compose -f docker-compose.nvidia.yml up -d
docker compose -f docker-compose.nvidia.yml logs -f
```

They set what the commands above set: the GPU (`deploy.resources.reservations.devices` for NVIDIA, `/dev/kfd` and
`/dev/dri` plus the `video` and `render` groups for AMD, `/dev/dri` for Intel), `ulimits.memlock`, `shm_size`, a two-minute
`stop_grace_period`, the `strata-data` volume, and the port bound to `127.0.0.1`. To let other computers reach it, set
`STRATA_BIND=0.0.0.0` in `.env` and keep a real `API_KEY`.

## Settings

Everything is an environment variable (`-e NAME=value`, or in compose). They are the same questions `strata launch` asks on its
screens; leave out what you do not care about.

| Variable | What it does | Default |
| --- | --- | --- |
| `MODEL` | model size: `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `Q2_0`, `UD-Q4_K_XL`, ... ([MODELS.md](MODELS.md)) | Strata picks what suits the card, and says so in the log |
| `FAMILY` | `qwen`, `unsloth`, `gyro`, `coder`, `swift`, ...; use `qwen` for the original ISTA-DASLab packs | installer recommendation; may differ by hardware |
| `MODEL_PATH` | GGUF files you already have: a folder or one `.gguf`, **inside the container** (mount it with `-v`, `:ro` is fine) | download |
| `ALLOW_DOWNLOAD` | `1`: go on and download the model although model files of the older layout were left unread in `/data/models` (see [From the older image](#from-the-older-image)) | stop |
| `CONTEXT` | how much text it remembers at once: `32k`, `64k`, `128k`, `256k` or a number | the largest that fits |
| `PARALLEL` | requests answered at the same time, 1 to 8 | fitted to the card |
| `VISION` | `off`, `on` (the model reads pictures), `gpu`, `cpu`; the picture-reader file (about 0.9 GB) is downloaded with the model | `off` |
| `KV` | conversation memory format: `int8`, `q4_0` (half the memory), `k8v4`, `fp16` | `int8` |
| `GPU` / `GPUS` | card numbers as the monitor lists them, `0` or `0,1`; empty or `all` = the engine's choice. Changing it later makes the model's run settings again (the files stay) | all that fit |
| `GPU_ROLES` | a role for each card, `1=main,0=helper` (`main`, `helper`, `off`; cards you do not name are off; not together with `GPU` / `GPUS`). Passed to `launch --gpu-roles`; see [MULTI_GPU.md](MULTI_GPU.md) | none |
| `TUNE` | `quick` (runtime depends on hardware and workload), `thorough`, `skip`; done once and kept in the preset (`RETUNE=1` tunes again) | `skip` |
| `API_KEY` / `API_KEY_FILE` | the key clients must send ([below](#the-api-key)); the file form is for Docker secrets | made for you |
| `TUNE_ALLOW_SMALL` | `1` passes `--tune-allow-small` to `launch` (allow tuning on a small card) | `0` |
| `PORT` | the port inside the container | `8080` |
| `PRESET` | the preset name in `/data` | `docker` |
| `STRATA_SET` | advanced engine settings, space separated, like `strata launch --set` (`"prefill=65536 STRATA_PIN_BY_NEED=1"`) | none |

Your choices are saved as a preset in the volume. Start again with no settings and the saved preset is used; start with
settings and they replace the saved ones (tuning results are kept). To change one thing, recreate the container with the new
variable; the model and tuning stay in `/data`. Anything else: `docker exec strata strata launch --help`, or
`docker run --rm IMAGE strata launch --plan --family qwen --model q2_0` to see what a choice would do without starting anything.

What is in the volume `/data`: `Strata-data/` (model files, the prepared pack), `state/` (presets, tuning, the key), `install/`
(run settings and logs), `home/`. The engine and the Terminal are in the image, so updating the image never touches your files.

To keep it all on a big disk, mount a folder there instead of the named volume: `-v /mnt/big/strata:/data` (the folder must be writable by the
container's user). Nothing large is written anywhere else in the container.

### Files you already have

Mount the folder with the model's GGUF files read-only and name it: `-v /srv/gguf/IQ3_XXS:/models:ro -e MODEL_PATH=/models
-e FAMILY=qwen -e MODEL=IQ3_XXS`. Nothing is downloaded for the model itself; the prepared pack is made in the volume.

- The folder holds the shards of one model directly (`...-00001-of-00002.gguf`, `...-00002-of-00002.gguf`), not a folder of folders.
  If you mount a parent folder, the container says which sub-folder to name in `MODEL_PATH`.
- The folder has to be mounted: a `MODEL_PATH` that is not in the container, or a folder with no `.gguf` files, stops the start at
  once with a message, and nothing is downloaded.
- Read-only (`:ro`) is fine. The pack (about 40 GB for IQ3_XXS) and the draft layer (about 6 GB, fetched once) are written to the
  volume, so it needs that room. With `VISION=on` the picture reader (0.9 GB) is fetched too; `VISION=off` skips it.
- Links to the files work when the links' targets are mounted too, and `MODEL` is set (without `MODEL`, Strata identifies the files itself
  and does not follow links).
- Without `MODEL_PATH`, files you put into the volume by hand (`/data/Strata-data/models/<model folder>/`) are found as well, if they have the
  published names.

## The API key

The server listens on every address **inside** the container (that is what lets the published port work), and Strata does not
serve beyond this computer without a key. So a key is always on:

- **You set it** with `API_KEY` (16 characters or more; `openssl rand -hex 24` makes one) or `API_KEY_FILE`.
- **You do not**: the first start makes one, keeps it in the volume, and the log tells you how to show it:
  `docker exec strata strata launch --show-key --preset docker`.

Publish the port as `-p 127.0.0.1:8080:8080` to keep it on this computer. Publish it as `-p 8080:8080` (or bind `0.0.0.0`) only
when other computers should reach it, and then use a key you chose. Strata warns in the log when the key is short. Every
request must send `Authorization: Bearer <key>`; `GET /health` answers without one.

## Connect your coding apps

From outside the container, the address is the published port and the key is yours:

- OpenAI-compatible apps: base URL `http://127.0.0.1:8080/v1`, API key = your key, model name from `GET /v1/models`.
- Anthropic-compatible apps (Claude Code): base URL `http://127.0.0.1:8080`, same key.

`strata connect` writes the settings file for OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and OpenClaw, and
checks that a prompt and a tool call work. **Inside** the container it fills in the key and the model for you:

```sh
docker exec strata strata connect opencode          # prints the file and the environment
docker exec strata strata connect opencode --check  # one prompt and one tool call, the way that app sends them
```

It prints `http://127.0.0.1:8080`, which is the address *inside* the container; on your computer use the published address
(`-p 127.0.0.1:9000:8080` makes it `http://127.0.0.1:9000`). If your coding app runs on your computer, install the `strata`
Terminal there ([TERMINAL_LAUNCH.md](TERMINAL_LAUNCH.md)) and run `strata connect opencode --url http://HOST:8080 --key <your key>`.

## Watch it and stop it

```sh
docker logs -f strata                       # what it is doing (the first start: the download)
docker exec -it strata strata monitor       # the live monitor (q leaves it, the server keeps running)
docker exec strata strata monitor --once    # one snapshot, for scripts
docker ps                                   # "healthy" once the model is loaded; "starting" while it downloads or loads
docker stop strata                          # drains, unloads the model, frees the graphics memory, exits 0
docker start strata                         # serving again in about a minute: the model and your choices are in the volume
```

The health check asks `GET /v1/models` with the key. During the first download the container shows `starting`, not `unhealthy`.
`docker stop` sends SIGTERM; the container runs `strata launch --stop`, so the engine releases its memory properly. Give it time
(`--stop-timeout 120` or compose's `stop_grace_period: 2m`): with the default 10 seconds a big model can be killed (exit 137).

## Updating

```sh
docker pull ghcr.io/niko1221/strata:0.2.7-cuda13      # the new tag (with a self-built image, build it again instead)
docker stop strata && docker rm strata
# run the same docker run command again with the new tag (compose: docker compose pull && docker compose up -d)
```

The models, the prepared pack, your preset and tuning stay in the volume `strata-data`; nothing is downloaded again. A new
engine can make an old tuning result less useful: start once with `-e TUNE=quick -e RETUNE=1` to tune again.
`strata update` inside the container is not the way to update: the engine is part of the image.

## Troubleshooting

**"The engine in this image does not match this computer's graphics card"** or the container stops right at the start.
The card is not visible to the container, or the image is for another card. Check the options for your card in
[the commands](#one-command-per-card) and the table in [Which image](#which-image-for-which-card).

**NVIDIA: `could not select device driver "" with capabilities: [[gpu]]`, or `docker: Error response from daemon: unknown or
invalid runtime name: nvidia`.** The NVIDIA Container Toolkit is missing or Docker was not told about it:

```sh
sudo apt install nvidia-container-toolkit          # or your distribution's package; see NVIDIA's install guide
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi       # must list your card
```

On Fedora / RHEL with Podman, use CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, then `--device nvidia.com/gpu=all`.
If `nvidia-smi` inside the container works but Strata says the driver is too old, you picked `cuda13` with a driver below 580:
use the `cuda12` image or update the driver.

**AMD: `Permission denied` on `/dev/kfd`, or no card found.** The container needs the device files and your user (or the
container's user) must be allowed to use them. Add `--device /dev/kfd --device /dev/dri` and the groups
`--group-add video --group-add $(getent group render | cut -d: -f3)` (the render group's number differs from computer to
computer, so use the number, not the name). `ls -l /dev/kfd /dev/dri/renderD*` shows who owns them. On a computer with
SELinux (Fedora, RHEL) that still refuses the device files, add `--security-opt label=disable` to the `docker run`.

**Intel: no device found.** Pass `--device /dev/dri` and `--group-add <render group number>` as above. An Arc card needs the
`i915` or `xe` driver on the host (`ls /dev/dri/renderD*` must show a device; `lspci -k | grep -A3 -i arc` shows the driver in use).

**SELinux (Fedora, RHEL, Rocky): `Permission denied` on a folder you mounted.** Add `:z` (shared) or `:Z` (private) to the
mount: `-v /srv/gguf/IQ3_XXS:/models:ro,z`. A named volume like `strata-data` needs nothing.

**The container is killed or the computer swaps.** Strata loads tens of GB into RAM: leave the container without a memory
limit, close other big programs, and pick a smaller `MODEL` ([MODELS.md](MODELS.md)). The log prints a warning when it sees a
memory limit.

**`docker stop` took long or ended with exit code 137.** The model was still unloading when Docker's time ran out. Use
`--stop-timeout 120` (compose: `stop_grace_period: 2m`).

**Port already in use.** Publish another host port: `-p 127.0.0.1:9000:8080` (inside the container it stays 8080).

**The first start says it is downloading but nothing moves.** `docker logs -f strata` shows the percentage every few seconds;
an interrupted download continues where it stopped when you start the container again. The volume needs the model's size free
(the log names it).

**"No MODEL was set"** warning: fine, Strata chose a model for your card; set `MODEL` to choose yourself.

## From the older image

The 0.1.x image compiled the engine inside `docker build` and was set up with `setup.py`. The new ones are built from the engine
package and start through `strata launch`. Mapping:

| Older variable | Now |
| --- | --- |
| `MODEL`, `FAMILY`, `CONTEXT`, `API_KEY`, `PORT`, `KV`, `GPU`, `GPUS`, `VISION` | the same (`VISION` takes `off\|on\|gpu\|cpu`; `no` / `yes` still work) |
| `GGUF_DIR` | `MODEL_PATH` (the old name still works) |
| `HOST` | not needed: the server listens on every address inside the container; choose where it is reachable with `-p` |
| `LOW_RAM`, `RESIDENT_BUDGET_GIB`, `KV_STREAMING`, `LAYER_SPLIT`, `CONFIG`, `REINSTALL` | ignored with a warning; use `STRATA_SET` for engine settings and `strata launch --help` for the rest |

A volume from the older image is picked up: the first start moves `/data/models`, `/data/packs` and `/data/mtp` into
`/data/Strata-data` (a rename inside the volume: nothing is copied or downloaded again, and the prepared pack is kept). The old
`/data/config` is not read; the run settings are made again from the files. A folder you mounted yourself at one of those places
is left alone: point `MODEL_PATH` at it. If model files are still in `/data/models` when the container starts (a folder you mounted there, or
a move that failed) and `MODEL_PATH` is not set, the start **stops at once and says where the files are**: it would otherwise download the model
again, tens of GB, because this image does not read that place. Name the folder in `MODEL_PATH`, or set `ALLOW_DOWNLOAD=1` to download anyway.
The container name, the volume and the port can stay as they were.

## Build it yourself

This is the way to get an image for a computer without registry access or after changing Strata. You need an engine package
folder (made by `packaging/package_engine.py`, or fetched by `strata update` into a components folder) and Docker on Linux:

```sh
docker/build.sh --flavor cuda --engine ~/components/engine/<cuda13-package-folder> --tag strata:0.2.7-cuda13
docker/build.sh --flavor hip  --engine ~/components/engine/<hip-rx9070-package-folder> --tag strata:0.2.7-rocm-gfx120x
docker/build.sh --flavor sycl --engine ~/components/engine/<sycl-package-folder> --tag strata:0.2.7-sycl
```

`build.sh` assembles the build context (the Terminal's files and the package) and runs `docker build`. The build makes the
Terminal exactly as the release archive (Python 3.12 and the locked packages, fetched at build time only), then checks the
engine package before it may enter the image: every file's SHA-256, and that every library it imports is bundled or belongs
to the system or the graphics driver (`packaging/binary_deps.py`); finally it checks that the dynamic linker resolves every
library inside the finished image. A package that fails any of these stops the build. Nothing is pushed anywhere.
`docker/smoke-test.sh` runs the checks we ran on real cards (serves, key, chat, tool call, monitor, clean stop, memory freed, restart); see its first lines for the variables.

**From source.** `docker build -t strata:src --build-arg STRATA_COMMIT=$(git rev-parse HEAD) .` in the repository compiles the CUDA 13
engine inside Docker, makes the package and puts it in the same kind of image (`--build-arg CUDA_ARCHITECTURES=89` for one card only,
much faster). `.git` is not copied into the build, so the commit comes in as `STRATA_COMMIT`; without it the Terminal stage stops.
For a GTX 10, P100 or V100 build the CUDA 12 variant: `docker build -t strata:src-cuda12 --build-arg STRATA_COMMIT=$(git rev-parse HEAD)
--build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 --build-arg CUDA_ARCHITECTURES="60;61;70;75;80;86;89" --build-arg BUILD_VISION=0 .`
(the build turns on the Pascal / Volta code by itself when an architecture below 75 is listed, and names the package `cuda12` with
driver 575.51). For AMD and Intel builds from source see [AMD_HIP.md](AMD_HIP.md) and [INTEL.md](INTEL.md); then package the result
with `packaging/package_engine.py` and use `docker/build.sh`.

## Publishing the images (for maintainers)

The images are built from the **engine packages of a release**, never from a build on the runner, so an image holds exactly the
files the signed update feed names. Nothing publishes by itself: a maintainer starts the workflow, after the owner's OK.

1. Make the release first (its `strata-feed.json`, `strata-feed.json.sig` and the engine archives must be on the GitHub release).
2. GitHub, **Actions, Docker images, Run workflow** ([.github/workflows/docker-images.yml](../.github/workflows/docker-images.yml); the file has to be on
   the default branch to show up). Inputs: the release tag (`v0.2.7`), the flavors (empty: every flavor the release has a package
   for), **push** and **latest**. Run it once with **push off**: it builds every image and publishes nothing (a dry run).
3. Run it again with **push on**. It logs in to `ghcr.io` with the job's `GITHUB_TOKEN` (`packages: write`) and pushes
   `ghcr.io/niko1221/strata:0.2.7-<flavor>` and `latest-<flavor>`. Turn **latest** off for a pre-release.
4. The first push creates each package as private. In the repository's Packages, open every package, **Package settings, Change
   visibility, Public**, and check it is linked to the repository (the image carries the `org.opencontainers.image.source` label).
5. Check one pull from a computer that is not logged in: `docker pull ghcr.io/niko1221/strata:0.2.7-cuda13`.

For each flavor the job runs [docker/fetch-engines.py](../docker/fetch-engines.py), which reads the release's signed feed (the
Ed25519 key in `strata_control/update_keys.json`), downloads the package, and checks the archive's size and SHA-256, the manifest
hash, the package signature and every file's hash, exactly as `strata update` does; a package that fails stops the job before
anything is built. [docker/build.sh](../docker/build.sh) then builds the image (and checks the package again inside the build), and
the job starts the image once to see that the Terminal inside it runs.

| Flavor (tag suffix) | Engine package in the release feed |
| --- | --- |
| `cuda13` | `linux-x86_64`, cuda, variant `cuda13` |
| `cuda12` | `linux-x86_64`, cuda, variant `cuda12` |
| `rocm-gfx120x` | `linux-x86_64`, hip, variant `rx9070` (gfx1200 / gfx1201) |
| `rocm-gfx110x` | `linux-x86_64`, hip, variant `rx7900` (gfx1100 / gfx1101 / gfx1102) |
| `rocm-gfx1151` | `linux-x86_64`, hip, variant `halo` (Strix Halo, gfx1151) |
| `sycl` | `linux-x86_64`, sycl; only when the feed lists one |

A flavor the feed has no package for is left out, with a line in the log. The same steps run from a Linux computer with Docker:
`docker/publish.sh --release-tag v0.2.7` builds everything and pushes nothing; add `--push` (after `docker login ghcr.io`) to publish.
The images run on the runner without a graphics card, so the card checks (`docker/smoke-test.sh`) stay a manual step on a computer with the card.

## Known limits

- Linux x86-64 hosts only. Docker on Windows reaches NVIDIA cards only; Docker on a Mac cannot use the GPU.
- The images run as `root` inside the container (GPU device files, page-locked memory). Run the container on a computer you
  trust the way you would run the app; do not mount folders you do not want it to read.
- An Arc card needs prepared model files (see [Intel](#intel-arc)).
- AMD **RX 7000** (gfx110x) and RDNA 3 iGPUs: the gfx110X image is built for gfx1100 / 1101 / 1102 (RX 7900, 7800, 7700, 7600). The
  Radeon 780M (gfx1103) is experimental and has its own engine package, not in these images.

## History / older measurements

**Tested on real cards.** Checks (`docker/smoke-test.sh`): the container serves, `GET /v1/models` answers (and refuses a request
without the key), a one-word chat and a tool call work, `docker exec strata monitor --once` works, `docker stop` ends it with exit
code 0 and the graphics memory is free again, and a second start reuses the saved preset.

| Date | Images built from | Card | Result |
| --- | --- | --- | --- |
| 2026-10-19 | 0.2.0-beta engine packages (engine 0.1.45.3), fetched and checked against the signed feed | `cuda13` and `cuda12` on one RTX A4000 (driver 580); `rocm-gfx120x` on one Radeon AI PRO R9700 | All passed. Serving 71-161 s after `docker run` with the prepared files; `docker stop` took 4-5 s with exit code 0. `rocm-gfx110x` and `rocm-gfx1151` passed the package and library checks but did not run on a card. On a computer without an AMD card the `rocm-gfx120x` container says the engine does not match the card and exits with code 3. |
| 2026-10-17 | 0.1.45 engine packages | RTX 3060 (`cuda13`, `cuda12`), Radeon AI PRO R9700 (`rocm-gfx120x`), Intel Arc Pro B70 (`sycl`) | All checks passed on each. Not run on a card: `cuda12` on Pascal / Volta (it ran on the RTX 3060), `rocm-gfx110x` (RX 7000). |
