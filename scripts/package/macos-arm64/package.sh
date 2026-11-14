#!/bin/bash
# Builds the Edge0 Chat package for macOS on Apple Silicon: Edge0 Chat.app with edge0-infer, edge0-strata (Metal) and
# the Python runtime of edge0-strata inside Contents/Resources, as a zip. See README.md in this directory.
#
#   package.sh fetch  --cache DIR [--commit REV]                  download every input (needs the network)
#   package.sh build  --cache DIR --out DIR [--commit REV]        build and pack from the cache, offline
#   package.sh verify --zip FILE [--model DIR [--device NAME]]    check a package; with --model, answer one prompt
set -euo pipefail

# Pinned inputs.
PY_VERSION=3.12.16
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-aarch64-apple-darwin-install_only_stripped.tar.gz"
PY_SHA256=2f11b8aa3726bdbd87dd9eb76568412faddf4f47e26913fa45d6a2e424972acd
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d   # the ggml pin in edge0-strata/EDGE0.md
METAL_CPP_URL=https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
METAL_CPP_SHA256=4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
MACOS_MIN=26.0
# The fixed build directory. The Metal engine keeps the path of its kernel sources in its source tree as a last
# fallback; with a fixed directory that path names no user, and the package's own copy beside the engine comes first.
BUILD_ROOT=/tmp/edge0-macos-package

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
STRATA_FILES=(edge0-strata/LICENSE edge0-strata/edge0 edge0-strata/serve edge0-strata/strata_control
    edge0-strata/strata_cli/__init__.py edge0-strata/strata_cli/client.py
    edge0-strata/tools/mtp_gguf.py edge0-strata/tools/gguf_reader.py edge0-strata/tools/strata_tokenizer.py
    edge0-strata/tools/dflash2_pack.py)

die() { echo "error: $*" >&2; exit 1; }
step() { echo "==> $*"; }
need() { for tool in "$@"; do command -v "$tool" >/dev/null || die "$tool not found"; done; }
sha_ok() { echo "$2  $1" | shasum -a 256 -c --status; }

COMMIT=HEAD CACHE="" OUT="" ZIP="" MODEL="" DEVICE=""
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
COMMAND=$1; shift
while [ $# -gt 0 ]; do
    case $1 in
        --commit) COMMIT=$2; shift 2 ;;
        --cache) CACHE=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --zip) ZIP=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[ "$(uname -s) $(uname -m)" = "Darwin arm64" ] || die "run on macOS on Apple Silicon"

fetch() {
    [ -n "$CACHE" ] || die "--cache is required"
    need git curl shasum cargo npm
    local sha tmp
    sha=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")
    mkdir -p "$CACHE"; CACHE=$(cd "$CACHE" && pwd)
    tmp=$(mktemp -d); trap "rm -rf '$tmp'" EXIT
    mkdir "$tmp/src"; git -C "$REPO" archive "$sha" | tar -x -C "$tmp/src"

    step "Python $PY_VERSION"
    sha_ok "$CACHE/python.tar.gz" "$PY_SHA256" 2>/dev/null || curl -fsSL --retry 3 -o "$CACHE/python.tar.gz" "$PY_URL"
    sha_ok "$CACHE/python.tar.gz" "$PY_SHA256" || die "python.tar.gz: SHA-256 mismatch"
    step "ggml (llama.cpp $LLAMA_COMMIT)"
    [ -s "$CACHE/llama.cpp-$LLAMA_COMMIT.tar.gz" ] || curl -fsSL --retry 3 -o "$CACHE/llama.cpp-$LLAMA_COMMIT.tar.gz" \
        "https://github.com/ggml-org/llama.cpp/archive/$LLAMA_COMMIT.tar.gz"
    step "metal-cpp 26"
    sha_ok "$CACHE/metal-cpp_26.zip" "$METAL_CPP_SHA256" 2>/dev/null || curl -fsSL --retry 3 -o "$CACHE/metal-cpp_26.zip" "$METAL_CPP_URL"
    sha_ok "$CACHE/metal-cpp_26.zip" "$METAL_CPP_SHA256" || die "metal-cpp_26.zip: SHA-256 mismatch"

    step "Python wheels"
    tar -xzf "$CACHE/python.tar.gz" -C "$tmp"
    "$tmp/python/bin/python3" -m pip download --quiet --disable-pip-version-check --only-binary=:all: -d "$CACHE/wheels" \
        -r "$tmp/src/edge0-strata/requirements.txt" -r "$tmp/src/edge0-strata/requirements-build.txt"
    step "Rust crates"
    rm -rf "$CACHE/cargo-vendor"
    cargo vendor --quiet --locked --versioned-dirs \
        --manifest-path "$tmp/src/edge0-infer/examples/desktop/src-tauri/Cargo.toml" "$CACHE/cargo-vendor" >/dev/null
    step "npm packages"
    (cd "$tmp/src/edge0-infer/examples/desktop" && npm ci --cache "$CACHE/npm" --ignore-scripts --no-audit --no-fund --loglevel=error >/dev/null)
    echo "$sha" > "$CACHE/COMMIT"
    step "cache ready for $sha: $CACHE"
}

# The files of a tree that hold any of the given strings (ASCII or UTF-16LE).
find_paths() {
    local root=$1; shift
    "$BUILD_PY" - "$root" "$@" <<'EOF'
import os, sys
root, needles = sys.argv[1], [n for n in sys.argv[2:] if n]
pats = [p for n in needles for p in (n.encode(), n.encode("utf-16-le"))]
bad = 0
for d, _, files in os.walk(root):
    for f in files:
        p = os.path.join(d, f)
        if os.path.islink(p):
            continue
        data = open(p, "rb").read()
        hit = [n for n in pats if n in data]
        if hit:
            bad += 1
            print("  build path in", os.path.relpath(p, root))
sys.exit(1 if bad else 0)
EOF
}

build() {
    [ -n "$CACHE" ] && [ -n "$OUT" ] || die "--cache and --out are required"
    need git cargo rustc node npm xcrun ditto codesign shasum lipo
    local sha
    CACHE=$(cd "$CACHE" && pwd); mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
    sha=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")
    [ "$(cat "$CACHE/COMMIT" 2>/dev/null)" = "$sha" ] || die "the cache was fetched for another commit: run fetch for $sha"
    if [ -e "$BUILD_ROOT" ]; then
        [ -f "$BUILD_ROOT/.edge0-macos-package" ] || die "$BUILD_ROOT exists and was not made by this script"
        rm -rf "$BUILD_ROOT"
    fi
    mkdir -p "$BUILD_ROOT"; touch "$BUILD_ROOT/.edge0-macos-package"
    local SRC=$BUILD_ROOT/src
    step "source $sha"
    git clone --quiet --no-checkout "$REPO" "$SRC"
    git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$sha"

    step "build tools"
    tar -xzf "$CACHE/python.tar.gz" -C "$BUILD_ROOT"
    "$BUILD_ROOT/python/bin/python3" -m venv "$BUILD_ROOT/tools"
    "$BUILD_ROOT/tools/bin/python" -m pip install --quiet --disable-pip-version-check --no-index \
        --find-links "$CACHE/wheels" -r "$SRC/edge0-strata/requirements-build.txt"
    BUILD_PY=$BUILD_ROOT/tools/bin/python
    export PATH="$BUILD_ROOT/tools/bin:$PATH" MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN
    local map="-ffile-prefix-map=$SRC/="
    local flags=(-DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOS_MIN -DCMAKE_C_FLAGS="$map" -DCMAKE_CXX_FLAGS="$map"
        -DCMAKE_OBJC_FLAGS="$map" -DCMAKE_OBJCXX_FLAGS="$map")

    step "edge0-infer"
    (cd "$SRC/edge0-infer" && cmake --preset mac-metal -DMOE_BUILD_TESTS=OFF -DMOE_BUILD_EXAMPLES=OFF "${flags[@]}" >/dev/null \
        && cmake --build --preset mac-metal --target edge0-infer >/dev/null)

    step "edge0-strata (Metal)"
    local ST_SRC=$SRC/edge0-strata
    mkdir -p "$ST_SRC/third_party/llama.cpp"
    tar -xzf "$CACHE/llama.cpp-$LLAMA_COMMIT.tar.gz" --strip-components=1 -C "$ST_SRC/third_party/llama.cpp"
    sha_ok "$CACHE/metal-cpp_26.zip" "$METAL_CPP_SHA256" || die "metal-cpp_26.zip: SHA-256 mismatch"
    unzip -q "$CACHE/metal-cpp_26.zip" -d "$ST_SRC/third_party"
    cmake -S "$ST_SRC/metal" -B "$ST_SRC/build-metal" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMETAL_CPP_DIR="$ST_SRC/third_party/metal-cpp" -DSTRATA_GGML_DIR="$ST_SRC/third_party/llama.cpp" "${flags[@]}" >/dev/null
    cmake --build "$ST_SRC/build-metal" --target strata >/dev/null

    step "Edge0 Chat"
    local DESKTOP=$SRC/edge0-infer/examples/desktop
    (cd "$DESKTOP" && npm ci --offline --cache "$CACHE/npm" --no-audit --no-fund --loglevel=error >/dev/null)
    # Later --remap-path-prefix options win, so the most specific prefix comes last.
    export RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$CACHE/cargo-vendor=vendor --remap-path-prefix=$SRC=."
    (cd "$DESKTOP" && npm run tauri -- build --no-sign --config "{\"bundle\":{\"macOS\":{\"minimumSystemVersion\":\"$MACOS_MIN\"}}}" \
        -- --locked --offline --config 'source.crates-io.replace-with="vendored"' \
        --config "source.vendored.directory=\"$CACHE/cargo-vendor\"")
    unset RUSTFLAGS

    local version pkg stage app res st
    version=$("$BUILD_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$DESKTOP/src-tauri/tauri.conf.json")
    pkg=edge0-chat-$version-macos-arm64
    stage=$BUILD_ROOT/stage/$pkg; app="$stage/Edge0 Chat.app"; res="$app/Contents/Resources"; st=$res/edge0-strata
    step "assemble $pkg"
    mkdir -p "$stage"
    ditto "$DESKTOP/src-tauri/target/release/bundle/macos/Edge0 Chat.app" "$app"
    # The debug map of the executable names the object files it was linked from, the toolchain's rlibs included.
    xcrun strip -S "$app/Contents/MacOS/"*
    install -m 755 "$SRC/edge0-infer/build/mac-metal/bin/edge0-infer" "$res/edge0-infer"
    mkdir -p "$BUILD_ROOT/strata-files"
    git -C "$SRC" archive "$sha" -- "${STRATA_FILES[@]}" | tar -x -C "$BUILD_ROOT/strata-files"
    ditto "$BUILD_ROOT/strata-files/edge0-strata" "$st"
    mkdir -p "$st/build-metal/metal/kernels" "$st/build-metal/metal/engine-msl"
    install -m 755 "$ST_SRC/build-metal/strata" "$st/build-metal/strata"
    # What the engine reads at run time, in the folder it looks in first (metal/src/resource_path.h).
    cp "$ST_SRC"/metal/kernels/*.metal "$st/build-metal/metal/kernels/"
    cp "$ST_SRC"/metal/src/engine/msl/* "$st/build-metal/metal/engine-msl/"
    cp "$ST_SRC/third_party/ggml/ggml-common.h" "$st/build-metal/metal/ggml-common.h"
    git -C "$SRC" archive "$sha" -- edge0-strata/data | tar -x -C "$BUILD_ROOT/strata-files"
    ditto "$BUILD_ROOT/strata-files/edge0-strata/data" "$st/build-metal/metal/data"
    # The Python runtime: a relocatable CPython with the pinned server packages in its own site-packages.
    mkdir "$BUILD_ROOT/runtime"; tar -xzf "$CACHE/python.tar.gz" -C "$BUILD_ROOT/runtime"
    mv "$BUILD_ROOT/runtime/python" "$st/.venv"
    [ -e "$st/.venv/bin/python" ] || ln -s python3 "$st/.venv/bin/python"
    "$st/.venv/bin/python" -m pip install --quiet --disable-pip-version-check --no-index --no-compile \
        --find-links "$CACHE/wheels" -r "$SRC/edge0-strata/requirements.txt"
    # Compiled now, so that running the server writes nothing inside the signed bundle.
    "$st/.venv/bin/python" -m compileall -q -j 0 -s "$st/" -p "edge0-strata/" "$st" >/dev/null

    step "licenses and README"
    cp "$REPO/LICENSE" "$REPO/NOTICE" "$stage/"
    mkdir "$stage/licenses"
    cp "$SRC/edge0-infer/third_party/GGML_LICENSE" "$stage/licenses/GGML-MIT.txt"
    cp "$SRC/edge0-infer/src/app/text/LICENSE.llama.cpp" "$stage/licenses/llama.cpp-tokenizer-MIT.txt"
    cp "$SRC/edge0-infer/third_party/cpp-httplib/LICENSE" "$stage/licenses/cpp-httplib-MIT.txt"
    cp "$SRC/edge0-infer/third_party/nlohmann-json/LICENSE.MIT" "$stage/licenses/nlohmann-json-MIT.txt"
    cp "$ST_SRC/LICENSE" "$stage/licenses/Strata-MIT.txt"
    cp "$ST_SRC/third_party/llama.cpp/LICENSE" "$stage/licenses/llama.cpp-MIT.txt"
    cp "$ST_SRC/third_party/project-maya/LICENSE" "$stage/licenses/Project-Maya-MIT.txt"
    cp "$ST_SRC/third_party/project-maya/NOTICE.md" "$stage/licenses/Project-Maya-NOTICE.md"
    cp "$ST_SRC/third_party/metal-cpp/LICENSE.txt" "$stage/licenses/metal-cpp-Apache-2.0.txt"
    "$BUILD_PY" "$HERE/notices.py" --title "Edge0 Chat $version (macOS, Apple Silicon)" \
        --cargo-manifest "$DESKTOP/src-tauri/Cargo.toml" --vendor "$CACHE/cargo-vendor" --npm "$DESKTOP" \
        > "$stage/licenses/THIRD-PARTY-NOTICES.txt"
    "$BUILD_PY" - "$HERE/user-README.md" "$stage/README.md" <<EOF
import sys
text = open(sys.argv[1]).read()
for key, value in {
    "VERSION": "$version", "COMMIT": "$sha", "PYTHON": "$PY_VERSION", "MACOS_MIN": "$MACOS_MIN",
    "MACOS_BUILD": "$(sw_vers -productVersion)", "SDK": "$(xcrun --show-sdk-version)",
    "CLANG": "$(xcrun clang --version | head -1)", "RUST": "$(rustc -V | cut -d' ' -f2)",
    "NODE": "$(node -v | tr -d v)", "CMAKE": "$(cmake --version | head -1 | cut -d' ' -f3)",
    "STRATA_VERSION": "$("$st/build-metal/strata" --version | cut -d' ' -f2)", "LLAMA": "$LLAMA_COMMIT",
}.items():
    text = text.replace("{{" + key + "}}", value)
assert "{{" not in text, "unfilled placeholder"
open(sys.argv[2], "w").write(text)
EOF

    step "sign (ad hoc)"
    while IFS= read -r -d '' f; do
        file -b "$f" | grep -q Mach-O || continue
        # Universal wheels: keep the arm64 slice only.
        if [ "$(lipo -archs "$f")" != arm64 ]; then lipo -thin arm64 -output "$f.thin" "$f" && mv "$f.thin" "$f"; fi
        codesign --force --sign - --timestamp=none "$f" 2>/dev/null
    done < <(find "$res" -type f -print0)
    codesign --force --sign - --timestamp=none "$app"
    codesign --verify --strict --deep "$app"

    step "check build paths"
    find_paths "$stage" "$HOME" "$REPO" "$CACHE" "/$(id -un)/" || die "the package holds paths of this machine"

    step "zip"
    rm -f "$OUT/$pkg.zip" "$OUT/$pkg.zip.sha256"
    (cd "$BUILD_ROOT/stage" && ditto -c -k --keepParent "$pkg" "$OUT/$pkg.zip")
    (cd "$OUT" && shasum -a 256 "$pkg.zip" > "$pkg.zip.sha256")
    rm -rf "$BUILD_ROOT"
    step "done: $OUT/$pkg.zip"
    cat "$OUT/$pkg.zip.sha256"
}

verify() {
    [ -n "$ZIP" ] && [ -f "$ZIP" ] || die "--zip FILE is required"
    need ditto codesign lipo otool curl
    [ ! -e "$BUILD_ROOT" ] || die "$BUILD_ROOT exists: remove it so that nothing is found through build paths"
    local tmp pkg app res st
    tmp=$(mktemp -d); trap "rm -rf '$tmp'" EXIT
    ditto -x -k "$ZIP" "$tmp"
    pkg=$(basename "$ZIP" .zip); app="$tmp/$pkg/Edge0 Chat.app"; res="$app/Contents/Resources"; st=$res/edge0-strata
    for f in "$app" "$res/edge0-infer" "$st/build-metal/strata" "$st/.venv/bin/python" "$tmp/$pkg/README.md" \
        "$tmp/$pkg/licenses/THIRD-PARTY-NOTICES.txt"; do
        [ -e "$f" ] || die "missing: ${f#"$tmp/"}"
    done
    BUILD_PY=$st/.venv/bin/python

    step "signature"
    codesign --verify --strict --deep "$app"
    step "architecture and minimum macOS"
    while IFS= read -r -d '' f; do
        file -b "$f" | grep -q Mach-O || continue
        [ "$(lipo -archs "$f")" = arm64 ] || die "not arm64 only: ${f#"$tmp/"}"
        local minos; minos=$(otool -l "$f" | awk '/LC_BUILD_VERSION/{b=1} b&&/minos/{print $2; exit}')
        [ -z "$minos" ] || [ "$(printf '%s\n%s\n' "$minos" "$MACOS_MIN" | sort -V | tail -1)" = "$MACOS_MIN" ] \
            || die "minos $minos > $MACOS_MIN: ${f#"$tmp/"}"
    done < <(find "$tmp/$pkg" -type f -print0)
    step "build paths"
    find_paths "$tmp/$pkg" "$HOME" "/$(id -un)/" || die "the package holds paths of this machine"
    step "edge0-infer info"
    "$res/edge0-infer" info | tee "$tmp/info.txt" | grep -E 'engine\.(version|source_commit|source_modified)'
    grep -q 'engine.source_modified: no' "$tmp/info.txt" || die "edge0-infer was built from a modified tree"
    step "edge0-strata"
    "$st/build-metal/strata" --version
    "$BUILD_PY" -c 'import numpy, jinja2, regex, yaml, tqdm, requests, PIL, psutil, certifi, truststore, charset_normalizer, idna, urllib3, markupsafe; print("python packages: ok")'
    (cd "$st" && .venv/bin/python serve/server.py --help >/dev/null) && echo "server.py --help: ok"

    if [ -n "$MODEL" ]; then
        step "serve one prompt: $MODEL${DEVICE:+ on $DEVICE}"
        ! ps -axo comm= | grep -Eq '(^|/)(strata|edge0-infer)$' || die "another model service is running"
        "$res/edge0-infer" serve --model "$MODEL" ${DEVICE:+--device "$DEVICE"} --port 0 >"$tmp/serve.log" 2>&1 &
        local pid=$! url="" i
        for i in $(seq 1 1800); do
            url=$(grep -o 'Listening: http://127\.0\.0\.1:[0-9]*' "$tmp/serve.log" | head -1 | cut -d' ' -f2) && [ -n "$url" ] && break
            kill -0 $pid 2>/dev/null || { tail -20 "$tmp/serve.log"; die "serve exited"; }
            sleep 1
        done
        [ -n "$url" ] || { kill -INT $pid; die "not ready in 1800 s"; }
        local name
        name=$(curl -sf "$url/v1/models" | "$BUILD_PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') \
            || { kill -INT $pid; die "no model listed"; }
        curl -sf "$url/v1/chat/completions" -H 'Content-Type: application/json' -d \
            "{\"model\":\"$name\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2 + 3? Reply with the number only.\"}],\"temperature\":0,\"max_tokens\":64}" \
            > "$tmp/answer.json" || { kill -INT $pid; die "request failed"; }
        kill -INT $pid; wait $pid || true
        "$BUILD_PY" -c 'import json,sys; t=json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]; print("answer:", repr(t)); sys.exit(0 if "5" in t else 1)' \
            "$tmp/answer.json" || die "unexpected answer"
        codesign --verify --strict --deep "$app" && echo "signature still valid after the run"
    fi
    step "verified: $ZIP"
}

case $COMMAND in
    fetch) fetch ;;
    build) build ;;
    verify) verify ;;
    *) die "unknown command: $COMMAND" ;;
esac
