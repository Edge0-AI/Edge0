# `strata launch`: set up, serve and watch

This page is for anyone who uses the installed Terminal. One command chooses a model, tunes it, starts serving and
opens a monitor. It also connects your coding apps. Text previews of the screens are in
[docs/terminal-previews/](terminal-previews/) (`launch-*.txt`).

## Quick start

1. Install the Terminal with the one-line command in [INSTALL.md](INSTALL.md#the-terminal).
2. Run `strata launch` and follow the screens, or run a command without screens:

```sh
strata launch --plan --family qwen --model q2_0            # what would happen; nothing changes
strata launch --family qwen --model q2_0 --yes --no-tui    # download, serve, print the address, return
strata launch --status                                     # what is serving now
strata launch --stop
```

`strata launch --help` lists every flag with examples. An unknown model or flag gets a plain message with the closest match.

## Install (one line)

Linux and Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

Windows PowerShell (then open a new terminal):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

More in [INSTALL.md](INSTALL.md#the-terminal). The installer needs no administrator rights and puts `strata` on your PATH.
How far to trust a piped installer is in [SECURITY.md](../SECURITY.md). The engine (the part that talks to your graphics
card) comes with the first model: `strata launch` fetches the one for your card from the same release, checks its signature
and test-starts it before using it.

## The screens

1. **Start:** saved presets (name, model, context, parallel requests, tuned or not, last used), "Set up from the beginning"
   or "Use model files I already have".
2. **Model:** the catalog the app and `setup.py` share, with download size, what it needs (graphics memory and RAM) and
   "fits / tight / does not fit this computer". This is a warning, never a block. The n-gram table stays on your SSD and is
   not counted. `o` picks your own files (a folder or a `.gguf`).
3. **Folder:** where model files are stored, with free space. For a model that is already installed, including one the
   desktop app installed, it starts where the model is and lists only what is missing (the draft layer, about 5.2 GB; the
   picture reader, 0.9 GB).
4. **Context** (8K to 512K, with the extra memory each costs) and the memory format (`m`: 8-bit or 4-bit).
5. **Pictures:** only for models that can read them. They are on by default (`--vision off` or "No" turns them off). The
   reader adds about 0.9 GB. It runs on the CPU for a card under 24 GB and on the graphics card from 24 GB (about 1 to 1.4 GB kept
   free). The rule is in `strata_control/vision_reader.py`. `--vision gpu` or `--vision cpu` overrides it per model. If the
   engine package has no picture reader, pictures default to off and the screen says so.
6. **Parallel requests:** "fit to your card" (12 GB: 1, 20: 2, 24: 3, 32 and more: 4) or a number from 1 to 8, with a warning
   if it will not fit. Check the effective slots after startup.
7. **Tuning:** quick (about 20 minutes), thorough (up to 2 hours) or skip. The screens suggest quick the first time; the
   command line skips tuning unless you add `--tune`.
8. **Advanced** (optional): every engine setting with its default, tuned value and your value. You can edit, reset, or add a
   setting by name (checked against the engine's own list, with a warning instead of a refusal). It also shows the exact command and
   the matching `strata launch` flags.
9. **Access:** this computer only (127.0.0.1), your network (LAN address and key), a Cloudflare quick tunnel (needs `cloudflared`)
   or Tailscale (needs it installed and connected). Also the port and an API key. Beyond this computer a key is recommended:
   Strata makes one, and `t` turns it off. Without a key the server still starts and warns that anyone who can reach the port
   can use the model and its tools; `--no-api-key` does the same without screens, and the warning is stronger on a Cloudflare tunnel.
10. **Save as preset** and launch. Downloads show progress, then the engine starts and the monitor opens. No web page opens
    unless you pass `--open`.

Keys: arrows choose, Enter goes on, Esc goes back, `?` explains the screen, `q` leaves.

With two or more usable graphics cards the Model step also asks for a role per card (M main, H helper, O off). On the command
line, `--gpu-roles 1=main,0=helper` does the same ([MULTI_GPU.md](MULTI_GPU.md)).

## Flags

```sh
strata launch --preset "IQ3_XXS 128K"                                  # a saved preset (opens the monitor)
strata launch --family qwen --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \
              --access lan --port 8080 --save-preset "Office" --no-tui
strata launch --model-path /data/models/IQ3_XXS --no-tui               # files you already have
strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
strata launch --family qwen --model q2_0 --context 64k --restart --no-tui
strata launch --list-presets
strata launch --show-key
```

| Flag | What it does |
| --- | --- |
| `--preset NAME` | Start a saved preset. Other flags change that start only; add `--save-preset` to keep them. |
| `--family`, `--model` | The model family and size, for example `qwen`, `q2_0`. Any case. |
| `--model-path PATH` | Use model files you already have: a folder of shards or one `.gguf`. |
| `--data-dir DIR`, `--move-data` | Where new downloads go. `--move-data` also moves the models you already have. |
| `--context SIZE` | `32k`, `64k`, `128k`, `256k` or a number of tokens. |
| `--kv FORMAT` | Conversation memory format: `int8` (default), `q4_0` (half the memory), `k8v4`, `fp16`. |
| `--vision off\|on\|gpu\|cpu` | Picture reading. Default: on when the model and the engine can. |
| `--parallel N` | Requests answered at the same time, 1 to 8. Default: fit to your card. |
| `--mtp on\|off` | The draft layer. `off` saves about 1 GB of graphics memory and is slower. Default: on. |
| `--drafting auto\|mtp\|dflash2\|off` | Which drafter guesses ahead. `dflash2` is for Qwen3.8-Flash-Next only and downloads a 1.2 GB drafter once. If it cannot run, MTP is used and the start says why. |
| `--gpu LIST`, `--gpu-roles LIST` | Which cards to use, and a role per card. |
| `--tune quick\|thorough\|skip`, `--tune-allow-small`, `--retune` | Tuning. `--retune` measures again; a model tuned before for the same choices on this computer reuses the stored result. |
| `--set KEY=VALUE` | An advanced engine setting or `STRATA_*` variable (repeatable). Unknown names get a warning. |
| `--pool-cores LIST` | Experimental: pin CPU workers to these logical processors, for example `2-14:2` ([DETAILS.md](DETAILS.md)). |
| `--access local\|lan\|cloudflare\|tailscale`, `--port N` | Who can reach the server. |
| `--api-key KEY`, `--api-key-env NAME`, `--new-key`, `--no-api-key` | The key clients must send. |
| `--save-preset NAME`, `--list-presets`, `--delete-preset NAME` | Presets. |
| `--no-tui` | No screens: serve, print the address and return. |
| `--yes` | No questions. It never restarts a running model. |
| `--restart` | Stop a model that is serving other choices (or holds the port), then start these. |
| `--open` | Open the server's web page once it serves. |
| `--plan` | Show what would happen and exit. It lists the model, context, picture setting and what would download (0 when everything is here). |
| `--feed URL\|DIR` | The signed update feed. Default: `STRATA_UPDATE_FEED` or the project's. |
| `--output human\|json` | Output format. |
| `--status`, `--stop`, `--show-key` | Show, stop, or print the key of the running launch (add `--preset NAME` to pick one). |

Presets are small files in `launch-presets/` beside the profiles folder. A run uses the shared profile `launch-<name>` and the shared
tuning presets (`strata presets`), so the app sees them too.

**Launching while a model is serving.** The same choices restart nothing and print the running addresses. Different choices
(another context, pictures, access, or another model on the same port) make the screens ask once, "Restart the model with these
settings? Answers in progress stop." With `--no-tui` the command refuses with one line naming the running model, its port and settings. Add
`--restart` or run `strata launch --stop` first. A different `--port` runs beside the first server; two big models may not fit in graphics memory together, and the launch says so.

## DeepSeek V4.1-Flash

DeepSeek V4.1-Flash runs in the Terminal and in the app, on NVIDIA RTX 30, 40 and 50 (Windows and Linux) and on the Radeon AI PRO R9700 (Linux). Hardware, sizes and speeds are in [MODELS.md](MODELS.md#deepseek-v41-flash).

```sh
strata launch --plan --family deepseek41 --model ist-das-2.00bpw
strata launch --family deepseek41 --model ist-das-2.00bpw
```

| Model name | Download |
| --- | ---: |
| `ist-das-1.75bpw` | 239 GB |
| `ist-das-2.00bpw` | 257 GB |
| `ist-das-2.25bpw` | 274 GB |
| `ud-q8_k_xl` | 508 GB |

- The context is 8192 tokens (8K), the size the engine was checked at.
- `--speed faster|full` picks how many of the model's 6 experts it uses per word. Faster uses 5 (a small quality change, usually not noticeable) and is the default on a PC with 16 GB of graphics memory or less and on PCs whose graphics share the computer's memory. Full uses all 6. `--full-quality` is the same as `--speed full`. Both need DeepSeek engine 0.1.50 or newer, which `strata launch` fetches. The engine's own flag is `--experts N`.
- It answers one request at a time. `--parallel` above 1 is ignored with a note.
- It reads pictures (on by default) with the engine's own vision tower, loaded with the model; `--vision off` turns them off. Setup fetches the tower's weights (mmproj-BF16.gguf, 1 GB) with the other shared files and packs them with the model.
- `--tune` runs DeepSeek's own short tuner (drafter on or off, CPU threads), not the Qwen tuner.
- `--model-path DIR` starts from the shards you already have.
- `--plan` shows where the model goes on your cards and how much is held in RAM.

## After the start: the monitor

The monitor has seven views. Press `1` to `7` or use the arrow keys.

| Key | View |
| --- | --- |
| 1 | Overview: model state, speed, GPU load, VRAM, temperature, power, PCIe, CPU, disk, context fill, experts in VRAM, system RAM, recent requests. It reads the server's `/metrics`, like the browser Monitor. |
| 2 to 6 | Serving, Requests, Engine, Hardware, Logs |
| 7 | Connect: the address, the key (hidden until `k`) and the setup for each app |

In Connect, `c` copies a setup, `w` writes it after a backup and `h` checks it by sending one prompt and one tool call the way that
app sends them. The list keeps two columns apart. **Vendor tested** is the app version its maker tested the setup with.
**This server** is Not checked, Passed or Failed for your server after `h`.

`q` asks whether to keep serving in the background or stop. Stopping unloads the model and frees graphics memory.

With two or more cards the Overview shows the busiest card's load, all cards' memory, the hottest card and the total power, and a
"Cards" strip lists up to 4 cards (`5` opens Hardware). The CPU tile shows sockets and cores.

**Windows console.** In the classic Windows 10 console the monitor switches that window's font to Cascadia Mono when it is installed
and draws the full look. Otherwise it draws a solid-block look and shows one line suggesting Windows Terminal. Nothing is saved and
Windows Terminal, Linux and macOS are never touched. `STRATA_GLYPHS=unicode|ascii|solid` forces a look.

## Connect your apps: `strata connect`

`strata connect` lists OpenCode, Codex CLI, Claude Code, oh-my-pi (`omp`), pi, Hermes Agent, OpenClaw and the DeepSeek Harness
(built into the app).

```sh
strata connect opencode           # print the file and environment, with address, key, model and context filled in
strata connect opencode --write   # write it after asking; anything replaced is backed up
strata connect opencode --check   # send one prompt and one tool call the way that app does
strata connect opencode --check --output json
```

| Flag | What it does |
| --- | --- |
| `--url URL`, `--key KEY`, `--key-env NAME` | The server address and key. Default: the running launch, else `http://127.0.0.1:8080`. |
| `--model NAME`, `--context SIZE` | The model name the server reports and the context it started with. Default: the running launch's, else 32k. |
| `--preset NAME` | Take address, key and model from a saved preset. |
| `--yes`, `--show-key` | Do not ask before `--write`; print the real key instead of a masked one. |

With `--output json` the object has the file and environment fields, `checks` (one entry per check with `name`, `ok`, `detail`,
`seconds`) and `checked_server`. `ok` is true only when both checks passed. The exit code is 0 when both worked and 1 otherwise;
then `error` says which check failed or that the server could not be reached.

Each configuration was tested against a Strata server at 32K context. Known limits are printed in plain words, for example:
Claude Code needs `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; OpenCode needs `limit.input`; Hermes refuses windows under 64K unless set up
as the config does; OpenClaw needs Node 24.16 or newer.

## Models you already have

`strata models list` looks in these places (one function, `strata_control/discovery.py`):

- the folder you name with `--install-dir` or `STRATA_INSTALL_DIR` (then only that one)
- the Terminal's own folder
- folders earlier installs remembered in `settings.json`
- the desktop app's install folder, which the Terminal shares: `%LOCALAPPDATA%\Strata\install` on Windows,
  `$XDG_DATA_HOME/strata/install` (normally `~/.local/share/strata/install`) on Linux and macOS

The data folder is the one you set, else the app's (`app-settings.json` beside its install folder, else a `Documents/Strata` that holds
models), else Strata's default.

`strata models import PATH --plan` recognises the Gyro-S and Gyro-M files (TQK6/TQK7 and TQ2_T experts) and every file set setup
installs, also under another name when the header matches. It says which run config, prepared pack, draft layer and picture reader
already exist. Nothing is copied or moved.

## When there is no engine for this computer

`strata launch` does not refuse. It says "Strata couldn't find an engine for this computer" with the reason, asks whether to go on
(Enter means yes; `--yes` skips the question), downloads the model, and ends with a message that names the problem and the next step.
The exit code is 0 and nothing is started.

## Intel Arc and cards of different makes

- **Intel Arc:** the Terminal downloads the released Linux SYCL engine, but its model preparation for Arc is a container build that it does not
  run. If the model's files and prepared pack are already on disk, `strata launch` writes the run settings itself. Otherwise it reports the preparation problem. Use
  [Intel setup](INTEL_ARC.md) for the preparation route.
- **Mixed makes:** one engine drives the cards of one make (CUDA: NVIDIA, HIP: AMD). Only those cards are offered, and a model is never
  split across makes. Two models at once, one per make, is not supported. If the other make has the bigger card, you get one line saying so.

## Limits

- The model download uses `setup.py` in its control mode (no pip, no engine fetch, no compiler), as the desktop app does. It needs internet and room for the files.
- The Linux SYCL package updates through `strata update`. Do not copy it into the components folder by hand.
