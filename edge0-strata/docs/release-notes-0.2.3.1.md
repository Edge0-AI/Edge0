**0.2.3.1: a small fix for setup stopping on the Windows NVIDIA package.**

## What is fixed

- **Setup no longer stops on the Windows NVIDIA engine package (#3810).** Setup read the list of graphics-card generations the picture reader was built for one character at a time, so it crashed on the Windows CUDA package from 0.2.3. It now reads that list the way the package writes it. The Linux NVIDIA package writes the same list, so the same fix covers it. An installed AMD engine also had a check that never recognised it; that is fixed too.

Only the Terminal's Control changed (version 0.1.48.1). The app stays 0.2.3 and the engine packages stay 0.1.48, so there is nothing new to download for them.

## Updating

- **From 0.2.3:** run `strata update` in the Terminal. It fetches Control 0.1.48.1, and the next `strata` command already runs it (`strata --version` says 0.2.3.1). The app is not touched.
- **From an older version:** follow the steps in the 0.2.3 notes. The install command (`install.ps1` on Windows, `install.sh` on Linux) now installs this fix.
- **Mac:** the Mac Terminal stays on Control 0.1.48. This fix does not touch the Mac.

## Downloads

Everything from 0.2.3 is attached again, so the release page still holds every file. The files that changed are `strata-control-0.1.48.1-windows-x86_64.zip`, `strata-control-0.1.48.1-linux-x86_64.tar.gz`, the feed (`strata-feed.json` with its `.sig`), `install-manifest.txt`, `latest.json` and `SHA256SUMS.txt`. The installers, AppImages, dmg and engine packages are the 0.2.3 files.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
