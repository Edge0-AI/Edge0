# Source checks

These checks catch accidental dependency and layering mistakes. They are not
intended to prevent deliberate evasion.

- CMake checks inspect literal commands only; they do not interpret variables,
  functions, or conditions.
- Macro includes such as `#include BAD_HEADER` and include path traversal are
  not checked.
- Includes inside block comments may produce false positives.

## Third-party manifest

`third_party.toml` lists approved libraries under `[libraries.<directory>]`.
Adding a library requires maintainer approval. Each entry records `upstream`, `version`,
SPDX `license`, exact `headers`, and `allowed_prefixes`. Its `files` table maps
relative file names to their retrieval `url` and `sha256`. Files are vendored
unchanged; unexpected files, missing files, symlinks and hash mismatches fail.
GGML alone delegates verification to `check_ggml_origin.py`.

Portal is the reference CLI and its HTTP server in `tools/cli/` (C++ namespace
`portal`). Native `src/` and `include/` cannot use Portal dependencies regardless of manifest
prefixes. Portal sources and CLI unit tests may include existing Portal private
headers, but cannot include native implementation headers. Other public consumers
cannot include Portal private headers. Vendored directories are excluded from
source traversal and checked against their hashes instead.

## Example App dependencies

`app_dependencies.toml` records approved direct dependencies separately from
`third_party.toml`. Adding or replacing a library, or changing its major version,
requires maintainer approval. Patch adjustments must update the manifest in the same commit.
`check_app_deps.py` compares npm and Cargo declarations and direct lockfile resolutions,
the Android version catalog and wrapper, and Swift package declarations. npm versions
must be exact; Cargo versions use `=VERSION`. Compose UI and Material 3 inherit the
registered Compose BOM (`bom` in the allowlist). Android build scripts use catalog
aliases, not literal dependency coordinates. iOS has no third-party packages.

App directories that do not exist yet are skipped during staged implementation;
once present, their dependency manifests and lockfiles are required. Generated
directories are excluded by the shared traversal. Run the check with:

```sh
uv run --no-project python -I tools/checks/check_app_deps.py .
```

## Example App design tokens

`check_app_tokens.py` keeps colors in each App's single design-token file:
`examples/desktop/src/theme.css`, Android `ui/theme/Tokens.kt` and
`examples/ios/App/Tokens.swift`. It scans desktop `src/**/*.{ts,tsx,css}`,
Android `app/src/main/kotlin/**/*.kt` and iOS `App/**/*.swift`, skipping the token
files and unit test files. It reports file, line and rule for hex colors (`#rgb`,
`#rrggbb`, `#rrggbbaa`), `rgb(`, `rgba(`, `hsl(`, `hsla(`, `Color(0x` and
`Color(red:`, `UIColor(red:`, `NSColor(red:`. `transparent`, `currentColor`, CSS
variables and platform semantic colors are allowed. Run it with:

```sh
uv run --no-project python -I tools/checks/check_app_tokens.py .
```
