# Strata 0.2.4.1: DeepSeek setup works from the installed Terminal

**Fixed:** setting up DeepSeek V4.1-Flash from the installed Terminal stopped at "Preparing your model" with "can't open file .../tools/dsv41/pack_gguf.py". The Terminal and Control packages of 0.2.4 did not contain the DeepSeek preparation scripts (`pack_gguf.py`, `smoke.py` and the short tune, `tune.py`); they ran only from a source checkout. They are in the packages now, and so are three scripts that other commands run by path (`strata_inspect.py`, `strata_bench.py`, `nvfp4_local_pack.py`).

Only the Terminal's Control changed (version 0.1.49.1). The app stays 0.2.3 and every engine package stays as it was, so there is nothing new to download for them.

## Corrected speed figure for the RTX 3060

The 0.2.4 notes said DSpark raised an RTX 3060 PC from 6.1-6.7 to 8.2-8.7 tokens/s. That was measured with DSpark forced on; on 12 GB cards DSpark stays off by default. Measured with the release defaults on an RTX 3060 PC with 46 GB of RAM: 8 to 10 tokens/s for answers (code 9.9, agent work 7.8, chat 8.9) and 149 tokens/s for an 8K prompt. Turning DSpark on (`--dspark on`) adds 2 to 13% there.

## Updating

- **From 0.2.4:** run `strata update` in the Terminal. It fetches Control 0.1.49.1; the next `strata` command already runs it (`strata --version` says 0.2.4.1). Then run your DeepSeek setup again.
- **New installs:** the install command (`install.ps1` on Windows, `install.sh` on Linux) installs this version.
- **Mac:** the Mac Terminal gets Control 0.1.49.1 too. DeepSeek is not on the Mac yet.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
