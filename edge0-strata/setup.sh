#!/bin/sh
# Strata for Linux: the first run installs everything and starts the model; later runs just start it.
# Needs only an NVIDIA driver (or, for an AMD Radeon card, the kernel's amdgpu driver: see docs/AMD_HIP.md).
# Python (with venv) is installed through apt/dnf if it is missing (asks for sudo).
# NixOS (or STRATA_UV=1) with uv installed: uv makes the project venv; every other Linux uses the standard venv.
cd "$(dirname "$0")" || exit 1
# A bundled Python runtime (the python/ folder of the terminal archive) is used when it is there: nothing to find,
# install or download. update.sh runs this file, so it uses it too.
if [ -x python/bin/python3 ]; then
  PYTHONNOUSERSITE=1; export PYTHONNOUSERSITE
  unset PYTHONHOME PYTHONPATH
  exec python/bin/python3 setup.py "$@"
fi
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  # uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: STRATA_UV=1.
  # Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
  UV=""
  if [ "${STRATA_UV:-}" = "1" ] || grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
    UV="$(command -v uv 2>/dev/null || true)"
  fi
  PY=""
  if [ -n "$UV" ]; then
    # Prefer uv's own interpreter: unlike Nix's Python, it works naturally with pip wheels using dlopen.
    PY=3.12
  else
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then
        PY=$c; break
      fi
    done
  fi
  # macOS: the system Python is 3.9; a standalone Python 3.12 goes into .python (no sudo, no Homebrew; pinned)
  if [ -z "$PY" ] && [ "$(uname -s)" = Darwin ]; then
    if [ ! -x .python/python/bin/python3 ]; then
      PBS="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only.tar.gz"
      echo "Python 3.10+ is needed; downloading a standalone Python 3.12 into .python (25 MB) ..."
      mkdir -p .python && curl -fL --retry 3 -o .python/python.tar.gz "$PBS" || { echo "could not download Python"; exit 1; }
      if [ "$(shasum -a 256 .python/python.tar.gz | cut -d" " -f1)" != "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8" ]; then
        rm -f .python/python.tar.gz; echo "the Python download does not match its pinned SHA-256"; exit 1
      fi
      tar xzf .python/python.tar.gz -C .python && rm -f .python/python.tar.gz
    fi
    PY=.python/python/bin/python3
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    fi
    PY=python3
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again. On NixOS, add uv to environment.systemPackages."
      exit 1
    fi
  fi
  if [ -n "$UV" ]; then
    # Force an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store.
    "$UV" venv --managed-python --seed --python "$PY" .venv || { echo "Could not create the uv environment in .venv."; exit 1; }
  else
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
  fi
fi
exec .venv/bin/python setup.py "$@"
