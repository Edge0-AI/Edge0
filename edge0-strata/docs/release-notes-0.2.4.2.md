# Strata 0.2.4.2: DeepSeek launch and install fixes

Fixes for `strata launch` with DeepSeek and for the one-line installer. The Terminal's Control (0.1.49.2), `install.sh` and the three DeepSeek engine packages (0.1.49.2) changed. The app stays 0.2.3 and every Qwen engine package stays as it was, so there is nothing new to download for them.

## What is fixed

- **Faster DeepSeek on AMD Ryzen 3000 / Threadripper 3000 / EPYC 7002 CPUs.** The 2.00 and IQ1_M sets answered up to 3x slower there; the answers are unchanged. The DeepSeek engine is now 0.1.49.2 (`strata-dsv41-0.1.49.2-windows-x86_64-cuda-cuda13`, `-linux-x86_64-cuda-cuda13` and `-linux-x86_64-hip-rx9070`).
- **DeepSeek setup no longer stops on a Qwen picture file in your models folder (#3927, #3917).** If the folder you share between models held a Qwen vision file, DeepSeek setup picked it up by mistake and failed. It now ignores any picture file that is not DeepSeek's.
- **Model files you already have are recognised for DeepSeek in the guided screen too, including Unsloth's UD-Q8_K_XL set (#3914).** The "use model files I already have" screen called a downloaded UD-Q8_K_XL file unreadable. It now recognises DeepSeek sets by their file names, as `strata launch --model-path` already did.
- **The right DeepSeek engine runs when you have more than one Strata install.** Launch now picks the newest DeepSeek engine package of the install that is running, and of the install your model's settings came from, instead of an old path saved in the model's settings.
- **A plain message when GitHub cannot be reached.** DeepSeek setup checks for 10 seconds before it downloads anything and says in one line that GitHub is not reachable, instead of retrying for minutes.
- **Faster returns to a conversation.** DeepSeek keeps recent conversations (and a shared system prompt) on disk, so a returning agent or a sub-agent with the same system prompt answers sooner; up to 2 GB, cleaned after a week.
- **The DeepSeek engine updates itself.** A newer DeepSeek engine package in the signed feed is installed on the next `strata launch` (see Updating).
- **DeepSeek no longer downloads the Qwen engine.** A DeepSeek launch does not fetch or test-start the Strata (Qwen) engine, the llama.cpp source or CUDA libraries it never uses. `strata launch --plan` no longer promises that download.
- **The disk check counts only what is still to download.** If most of the model files are already in your models folder (also under the first release's file names), the model list no longer says "tight: needs more disk space". The room for the prepared model is still counted.
- **Smaller wording fixes.** A setup message that ended with a full stop no longer gets a second one ("engine.."), and the second sentence of "The model could not be installed. The model does not answer sensibly after the preparation: ..." starts with a capital.
- **The install one-liners keep your terminal open on an error.** `install.sh` is now a function that returns an error instead of closing the shell it runs in, and the commands in the README and docs download it first: `curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh`.

## Updating

- **DeepSeek engine:** it updates itself on the next launch. When the signed feed has a newer DeepSeek engine package for your PC (0.1.49.2 over 0.1.49), `strata launch` downloads it, checks it, starts it once, switches to it and removes the old one. If that fails you see one line and the engine you have keeps running; with no internet the launch goes on without waiting. An engine you gave with `--dsv41-package` or `STRATA_DSV41_EXE` is never replaced.
- **From 0.2.4 or 0.2.4.1:** run `strata update` in the Terminal. It fetches Control 0.1.49.2 and the next `strata` command runs it (`strata --version` says 0.2.4.2).
- **New installs:** use the command in the README (download `install.sh`, then run it).
- **Mac:** the Mac Terminal gets Control 0.1.49.2 too. DeepSeek is not on the Mac yet.

## Downloads

Everything from 0.2.4.1 is attached again, so the release page still holds every file. New in this release: the three Control packages (`strata-control-0.1.49.2-windows-x86_64.zip`, `-linux-x86_64.tar.gz`, `-macos-arm64.tar.gz`), the three DeepSeek engine packages (`strata-dsv41-0.1.49.2-windows-x86_64-cuda-cuda13.tar.gz`, `-linux-x86_64-cuda-cuda13.tar.gz`, `-linux-x86_64-hip-rx9070.tar.gz`), `install.sh`, the feed and `SHA256SUMS.txt`. The installers, AppImages, dmg and Qwen engine packages are the 0.2.3 files.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
