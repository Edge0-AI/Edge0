#!/usr/bin/env bash
set -euo pipefail

fail() { echo "check-prereqs: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || fail "macOS is required"
[[ "$(uname -m)" == "arm64" ]] || fail "Apple Silicon (arm64) is required"

macos_major="$(sw_vers -productVersion | cut -d. -f1)"
[[ "$macos_major" -ge 14 ]] || fail "macOS 14 or later is required (found $(sw_vers -productVersion))"

command -v rustc >/dev/null || fail "Rust 1.88+ is required (https://rustup.rs)"
command -v cargo >/dev/null || fail "cargo is required"
command -v cmake >/dev/null || fail "CMake 3.24+ is required"
command -v node >/dev/null || fail "Node 22 or later is required"
node_major="$(node -p 'process.versions.node.split(".")[0]')"
[[ "$node_major" -ge 22 ]] || fail "Node 22 or later is required (found $(node --version))"
# yarn 1.22: use PATH if present, otherwise Corepack (ships with Node).
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v yarn >/dev/null 2>&1; then
  command -v corepack >/dev/null || fail "yarn 1.22 is required (install Corepack or: npm i -g yarn@1.22.22)"
  mkdir -p "${HOME}/.local/bin"
  corepack enable --install-directory "${HOME}/.local/bin"
  corepack prepare yarn@1.22.22 --activate
fi
command -v yarn >/dev/null || fail "yarn 1.22 is required"
command -v xcrun >/dev/null || fail "Xcode Command Line Tools are required (Metal compiler)"
xcrun --find metal >/dev/null 2>&1 || fail "the Metal compiler is missing (install Xcode or CLT)"

echo "check-prereqs: ok"
echo "  macOS $(sw_vers -productVersion) $(uname -m)"
echo "  rustc $(rustc --version)"
echo "  cmake $(cmake --version | head -1)"
echo "  node  $(node --version)"
echo "  yarn  $(yarn --version)"
