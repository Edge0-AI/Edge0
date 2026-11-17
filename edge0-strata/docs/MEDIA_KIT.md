# Strata media and video guide

For people and AI tools creating an article, overview or video. Last reviewed: 25 October 2026, for Strata 0.2.5.
Use [AI_REFERENCE.md](AI_REFERENCE.md) for product facts and [README.md](../README.md) for the public introduction.

## Short description

Strata runs supported AI models on your own computer. Use its desktop app to chat and work with code and images,
use its terminal app, or connect a coding tool. One installer per system (Windows, Linux, Mac). Strata's code is free and open source under the MIT license.

Suggested opening: **“Run powerful AI on your own computer. Chat, code and work with images, locally.”**

## A short video outline

1. **Show the desktop app.** A chat and a visible result explain the product quickly.
2. **Show the choice of interfaces.** The desktop app for a visual workspace; the terminal app for keyboard-driven use.
3. **Show the three model families.** DeepSeekV4.1-Flash-1017, Qwen3.8-Flash-Next and Qwen3.6-35B-A3B.
   Explain the [availability limits](AI_REFERENCE.md#model-families): DeepSeek V4.1-Flash is in the app and the Terminal since 0.2.5, on NVIDIA RTX 30, 40 and 50 and the Radeon AI PRO R9700 (Linux).
4. **Choose a model for the machine.** Prefer ISTA-DASLab, then Unsloth, Gyro and other providers where compatible.
   Gyro is a memory-saving option, currently slower. State the model, quantization and hardware used in any live demo.
5. **Connect a familiar tool.** Examples include Claude Code, Codex CLI, OpenCode, oh-my-pi, pi.dev and Hermes Agent.
   Show a real successful connection if demonstrating one.
6. **Close with the two useful links.** [Download Strata](https://github.com/Niko1221/Strata/releases/latest)
   and [support development](https://buymeacoffee.com/strataengine).

This is an outline, not evidence that any particular model, workflow or benchmark was tested for the video.

## Existing assets

Use the existing assets with accurate captions. These screenshots demonstrate the interfaces; they do not
establish a current default model or universal performance.

| Asset | File | Suggested caption / context |
| --- | --- | --- |
| Logo, light background | [strata-lockup-light.svg](../apps/host/public/brand/strata-lockup-light.svg) | Strata |
| Logo, dark background | [strata-lockup-dark.svg](../apps/host/public/brand/strata-lockup-dark.svg) | Strata |
| Desktop app | [app-chat.png](media/app-chat.png) | Chat, files and a live preview in one window. The capture shows a Gyro model; do not relabel it as an ISTA or DeepSeek run. |
| Terminal app | [terminal-monitor.png](media/terminal-monitor.png) | Model activity and hardware use. The capture shows a Gyro-S session; its live counters are not a standardized benchmark. |
| Coding demo thumbnail | [pagoda-preview.webp](media/pagoda-preview.webp) | A voxel garden generated locally. |
| Coding demo video | [Pagoda.mp4](https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4) | Existing 49-second demo, documented with an RTX 5070, IQ3_S and 128K context. Keep it separate from the GUI and TUI screenshots. |
| Engine explanation | [how-it-works.svg](media/how-it-works.svg), [guess-and-check.svg](media/guess-and-check.svg) | Illustrations of the documented Flash-Next engine path, not proof that every model uses the same path. |

## Accuracy rules

- Use actual footage or label illustrations. Do not invent a successful install, generated answer or speed reading.
- Keep the provider preference distinct from measured performance. The maintainer's collaboration with ISTA-DASLab
  explains the recommendation order; it does not prove a benchmark result.
- DeepSeek V4.1-Flash runs in the desktop app and in the Terminal (0.2.5), on NVIDIA RTX 30, 40 and 50 and the Radeon AI PRO R9700 on Linux. Take its sets (1.75, 2.00 and 2.25 bpw), sizes and speeds from
  [MODELS.md](MODELS.md#deepseek-v41-flash) and name the hardware. Do not invent quants or memory figures.
- Do not promise “any PC,” “every model,” or identical results across GPUs. Follow the model and hardware guides.
- For speed comparisons, use a source report with hardware, model, quantization, engine version and workload.
  Keep reasoning tokens, decode speed, prompt speed and time to first answer distinct.
- Local inference, internet downloads and connected tools are separate. Refer to [SAFETY.md](SAFETY.md) for privacy claims.
- Strata's MIT license does not replace the licenses of model weights, dependencies or other creators' work.
  Use the [credits and licensing guide](HOW_IT_WORKS.md#credits).

## Links and contact

- Project: [Niko1221/Strata](https://github.com/Niko1221/Strata)
- Downloads: [Releases](https://github.com/Niko1221/Strata/releases/latest)
- Support: [Buy Me a Coffee](https://buymeacoffee.com/strataengine)
- Sponsorships, hardware partnerships and collaborations: [niko.veit@outlook.com](mailto:niko.veit@outlook.com)
