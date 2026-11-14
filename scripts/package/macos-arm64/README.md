# Packaging for macOS (Apple Silicon)

`package.sh` builds `edge0-chat-<version>-macos-arm64.zip`: `Edge0 Chat.app` with `edge0-infer`, edge0-strata for Metal and the Python runtime of edge0-strata inside `Contents/Resources`, plus the licenses and a README for users (`user-README.md`). The version is the app's version in `edge0-infer/examples/desktop/src-tauri/tauri.conf.json`.

Requirements on the build Mac: Apple Silicon, the Xcode Command Line Tools, Git, Rust 1.90 or newer (`cargo`, `rustc`) and Node.js with npm. CMake and Ninja come from the fetched wheels.

```sh
# 1. Download every input for a commit into a cache folder (needs the network; any Apple Silicon Mac).
scripts/package/macos-arm64/package.sh fetch --cache ~/edge0-package-cache --commit <commit>

# 2. Build and pack from that cache. Nothing is downloaded; the cache can be copied from another Mac.
scripts/package/macos-arm64/package.sh build --cache ~/edge0-package-cache --out ~/edge0-package-out --commit <commit>

# 3. Check the package; with --model, the edge0-infer inside it also serves one prompt with that model
#    (--device picks its device as for edge0-infer serve: without it, Edge0 35B runs on the CPU).
scripts/package/macos-arm64/package.sh verify --zip ~/edge0-package-out/edge0-chat-<version>-macos-arm64.zip \
  --model /path/to/Edge0-145B-q3k
```

`--commit` defaults to `HEAD` and must be a commit of this repository; the build checks out a clean copy of it, so uncommitted changes are not packed. The cache belongs to one commit: run `fetch` again after `Cargo.lock`, `package-lock.json` or `edge0-strata/requirements.txt` change.

`build` works in `/tmp/edge0-macos-package` and removes it at the end. `verify` refuses to run while that folder exists, so that the package cannot fall back to files of the build tree. It checks the signature, that every binary is arm64 for the minimum macOS, that no file holds the user's home directory, `edge0-infer info` and `strata --version`, the Python packages and `server.py --help`. With `--model` (and no other model service running) it starts `edge0-infer serve`, asks one question and checks the signature again afterwards.

After `verify`, open the app once and load a model to check the window.
