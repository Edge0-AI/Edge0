# Set up Strata for a user

Use this guide when a user asks an AI agent to install Strata (version 0.2.5). There are two release routes. Pick
one; do not start from a source checkout.

- **The Terminal and `strata launch`** (below). This is the route for an agent that has a shell.
- **The desktop app.** One installer per system: `Strata-0.2.9-windows-x64-setup.exe`,
  `Strata-0.2.9-linux-x86_64.AppImage`, `Strata-0.2.9-macos-arm64.dmg`, from the
  [latest release](https://github.com/Niko1221/Strata/releases/latest). The Windows and Linux installers carry no
  engine: on first start the app shows the engine for the user's graphics card and downloads it, or installs a file the
  user already has ("I already have the file"). Details: [INSTALL.md](INSTALL.md).

Model availability, provider preference and API details are in [AI_REFERENCE.md](AI_REFERENCE.md).

## 1. Check the computer and the requested model

- Check OS, CPU architecture, graphics cards, driver, available RAM/VRAM and free disk space.
- Use the [hardware guides](INSTALL.md#what-you-need). Linux and Windows packages are x86-64; Mac packages are
  Apple Silicon and need macOS 15 or newer. Older GPUs, Intel Arc and low-memory machines have specific limits.
- Model support is separate from hardware support. Qwen3.6 currently uses CUDA or HIP; it is not a Metal/SYCL model.
  DeepSeekV4.1-Flash-1017 (`--family deepseek41`) runs in the app and in the Terminal, on NVIDIA RTX 30, 40 and 50
  (Windows and Linux) and on the AMD Radeon AI PRO R9700 under Linux. It has three sets: 1.75 bpw (239 GB of disk),
  2.00 bpw (257 GB) and 2.25 bpw (274 GB), and for a 12 GB card it needs 46 to 64 GB of RAM. The context is 8K.
  `--speed faster|full` (or `--full-quality`) chooses how many experts it uses per word; Faster is the default on a
  PC with 16 GB of graphics memory or less. Print the plan first:
  `strata launch --plan --family deepseek41 --model ist-das-2.00bpw`.
- Check the **chosen pack's** memory and disk needs. Do not reuse a Flash-Next estimate for every family.
  Unified memory is one pool, not RAM plus VRAM. A completed download does not establish that a model can run.
- Keep the default local-only access. Share the server over a network only when the user asks; recommend an API key
  beyond localhost and explain any deliberate use of `--no-api-key`.
- Explain substantial downloads and follow the user's existing authorization and network/storage constraints.
  Do not install drivers or system build tools as an unannounced side effect of a normal Terminal install.

## 2. Install the Terminal

The installer downloads a private Python runtime and Control, verifies the release checksums and signed feed, and
installs into the user's home folder. The engine and model are acquired during launch.

**Linux / Mac:**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell:**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Open a new terminal on Windows. On Linux and Mac, use the full executable path printed by the installer if
`~/.local/bin` is not on PATH. Confirm with `strata --version` and `strata launch --help`.
Do not search for `START-HERE.bat` in the installed app or Terminal: it is not part of them.

## 3. Select a compatible pack and inspect the plan

Keep the user's model and provider choice. Otherwise prefer **ISTA-DASLab → Unsloth → Gyro → others**, when a pack
fits the selected model and computer. This is the maintainer's recommendation policy; current executable defaults
can differ. Gyro is a memory-saving fallback, currently slower, not a general speed recommendation.

| Choice | Explicit launch selection |
| --- | --- |
| ISTA-DASLab Flash-Next | `--family qwen --model q2_0` (or a compatible IQ2_XS / IQ3_XXS / IQ3_S pack) |
| ISTA-DASLab Coder | `--family coder --model IQ1_M` |
| Unsloth Flash-Next | `--family unsloth --model UD-Q4_K_XL` (check the pack's backend limits) |
| Gyro Flash-Next | `--family gyro --model Gyro-S` |
| Unsloth Qwen3.6 | `--family qwen36 --model UD-Q4_K_XL` |

These are choices, not an instruction to install every row. The full list and memory estimates are in
[MODELS.md](MODELS.md). For example, inspect the ISTA-DASLab Q2_0 plan:

```sh
strata launch --family qwen --model q2_0 --context 32k --plan --output json
```

Check the resolved family, model, storage folder, context, picture reader and missing downloads. If the plan names
an unavailable backend, do not continue as though compatibility were confirmed. Intel Arc has a released Linux
engine, but model preparation still has special requirements: [INTEL_ARC.md](INTEL_ARC.md).

## 4. Start it

After the plan matches the user's computer and requested setup:

```sh
strata launch --family qwen --model q2_0 --context 32k --yes --no-tui
```

`--no-tui` prints the address and returns after startup while the server continues in the background. Keep the
launch process alive during model acquisition; large downloads and preparation can take a long time. Poll progress
and logs rather than promising a universal startup time. Repeating an interrupted launch resumes completed work.

Useful options:

| Option | Meaning |
| --- | --- |
| `--model-path PATH` | Use existing supported model files; a matching filename alone does not prove compatibility |
| `--data-dir PATH` | Folder for new downloads; existing files stay put unless `--move-data` is also given |
| `--vision on\|off\|gpu\|cpu` | Picture reading, subject to model and packaged-reader support |
| `--gpu 0,1` | Selected GPUs; do not use source setup's `--gpus` spelling here |
| `--parallel N` | Requested concurrent slots, 1–8; check what actually fits and starts |
| `--save-preset NAME` | Keep these launch choices for later |
| `--open` | Open the server's browser page after it starts |
| `--restart` | Replace an already running launch with different settings; interrupts active answers |

`--yes` does not authorize a restart by itself. When a model is already running, inspect its state before using
`--restart` or `--stop`. Use `strata launch --help` for the installed release's full list.

## 5. Verify the running service

```sh
strata launch --status
strata doctor --live
```

Check the printed address and model, then query `/health` and `/v1/models` at that address. With the default port:

**Linux / Mac:**

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
```

**Windows PowerShell:**

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/models
```

Use the configured credential for protected endpoints. Confirm the model is loaded, then send a real request or
run the relevant tool check below. **Exit code 0 alone is insufficient:** launch can finish after downloading a model
without starting it when no usable engine is available. Report that as downloaded, not installed and verified.

## 6. Connect a tool

```sh
strata connect
strata connect opencode
strata connect opencode --check --output json
```

The first command lists adapters; the second prints settings; the third sends a prompt and tool-call check. Use the
adapter the user actually needs. `--write` changes that tool's configuration and keeps a backup; use it within the
user's authorization. Printed configuration and a previous vendor test are not a successful current-server check.

- OpenAI-compatible clients: use the printed server address plus `/v1`.
- Anthropic-compatible clients: use the server root; the client adds `/v1/messages`.
- Use the actual API key when enabled. Tool identifiers and limits: [connection guide](TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).
- Flash-Next's template defaults to thinking on with `xhigh` effort; `high` maps to `xhigh`. Requests, shared Chat
  settings and run-config settings can override defaults. Do not assert a sampling configuration without inspecting it.

## 7. Handoff, update or stop

Tell the user which model is running, where its files are, its address, how the tool check went, and any limitations.

```sh
strata launch --preset NAME
strata launch --status
strata launch --stop
strata update --check
```

Stop active models before `strata update`. Very old Terminals require rerunning the installer; see
[updating](INSTALL.md#updating). Show credentials only to the intended user; redact them from shared logs and reports.

## 8. Source-only and unsupported setups

A source build is a separate workflow: [SOURCE_INSTALL.md](SOURCE_INSTALL.md). `START-HERE.bat`, `SETUP.bat`,
`UPDATE.bat`, `setup.sh` and generated `run-<model>` scripts belong to that workflow. Do not pass their flags to
`strata launch`, promise a source build will support an unknown model, or describe a source-only command as an
installed-app command. If a required backend or artifact is missing, say exactly what is missing.
