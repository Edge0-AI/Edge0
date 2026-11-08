#!/usr/bin/env bash
# Assemble pinned llama.cpp in wt/linux; never patch vendor/llama.cpp in place.
# Dependencies: Bash, git, flock, CMake and a C/C++ toolchain (GPU SDK if selected).
# wt/linux is disposable: each invocation resets it and replays the patch bands.
set -Eeuo pipefail
export LC_ALL=C
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
DEPOT=${EDGE0_DEPOT:-$(dirname -- "$ROOT")}
BACKEND=cpu BUILD_DIR='' JOBS=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}
ASSEMBLE_ONLY=0 CONFIGURE_ONLY=0 CLEAN=0

log() { printf '[edge0-build] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
on_error() { local rc=$?; log "ERROR: command failed at line $1 (exit $rc)" >&2; exit "$rc"; }
trap 'on_error "$LINENO"' ERR

usage() {
    cat <<'USAGE'
Usage: linux/scripts/vendor-build.sh [options]
  --cuda | --vulkan   Select one GPU backend (default: CPU/AVX2).
  --build-dir NAME   Build directory name under wt/linux (default: build-<backend>).
  --jobs N           Parallel build jobs (default: CMAKE_BUILD_PARALLEL_LEVEL or nproc).
  --assemble-only    Replay patches and stage linux/serve sources; no CMake needed.
  --configure-only   Assemble and configure, without compiling.
  --clean            Remove the selected build directory before configuring.
  -h, --help         Show this help.
EDGE0_DEPOT overrides the depot root; EDGE0_LLAMA_URL selects a llama.cpp mirror.
USAGE
}

select_backend() {
    [[ "$BACKEND" == cpu || "$BACKEND" == "$1" ]] || die "select only one GPU backend"
    BACKEND=$1
}

parse_args() {
    while (( $# )); do
        case "$1" in
            --cuda|--vulkan) select_backend "${1#--}"; shift ;;
            --build-dir|--jobs)
                (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"
                if [[ "$1" == --jobs ]]; then JOBS=$2; else BUILD_DIR=$2; fi
                shift 2 ;;
            --assemble-only) ASSEMBLE_ONLY=1; shift ;;
            --configure-only) CONFIGURE_ONLY=1; shift ;;
            --clean) CLEAN=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
    BUILD_DIR=${BUILD_DIR:-build-$BACKEND}
    [[ "$BUILD_DIR" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die "--build-dir must be a directory name"
}

check_inputs() {
    [[ "$(uname -sm)" == 'Linux x86_64' ]] || die "this build requires Linux x86-64"
    local tool
    for tool in git flock; do command -v "$tool" >/dev/null || die "$tool is required"; done
    if (( ! ASSEMBLE_ONLY )); then command -v cmake >/dev/null || die "cmake is required"; fi
    [[ -f "$DEPOT/vendor.llama.pin" ]] || die "vendor.llama.pin missing; set EDGE0_DEPOT"
    DEPOT=$(cd -- "$DEPOT" && pwd -P)
    read -r PIN _ < "$DEPOT/vendor.llama.pin"
    [[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || die "vendor.llama.pin must contain a full commit hash"
    VENDOR=$DEPOT/vendor/llama.cpp WT=$DEPOT/wt/linux
    PDIR=$DEPOT/patches/llama.cpp
    check_bands
    mkdir -p -- "$DEPOT/wt"
    exec 9> "$DEPOT/wt/.linux-build.lock"
    flock -n 9 || die "another Linux vendor build is running"
}

check_bands() {
    local band patches
    for band in common linux; do
        patches=("$PDIR/$band/"*.patch)
        [[ -f "${patches[0]}" ]] || die "patch band missing: $band"
    done
}

prepare_vendor() {
    if [[ ! -e "$VENDOR" ]]; then
        log "[0/4] materializing pinned vendor"
        mkdir -p -- "$DEPOT/vendor"
        git clone --filter=blob:none --no-checkout -- \
            "${EDGE0_LLAMA_URL:-https://github.com/ggml-org/llama.cpp}" "$VENDOR"
        git -C "$VENDOR" checkout --detach "$PIN"
    fi
    log "[1/4] checking pristine vendor"
    [[ -d "$VENDOR/.git" && ! -L "$VENDOR" ]] || die "vendor must be a standalone checkout"
    [[ "$(git -C "$VENDOR" rev-parse HEAD)" == "$PIN" ]] || die "vendor is not at the pin; refusing to reset it"
    [[ -z "$(git -C "$VENDOR" status --porcelain --untracked-files=all)" ]] || die "vendor is dirty"
    if git -C "$VENDOR" symbolic-ref -q HEAD >/dev/null; then die "vendor must have a detached HEAD"; fi
}

prepare_worktree() {
    log "[2/4] resetting isolated Linux worktree and replaying bands"
    if [[ ! -e "$WT" ]]; then git -C "$VENDOR" worktree add --detach "$WT" "$PIN"; fi
    [[ ! -L "$WT" && "$(git -C "$WT" rev-parse --show-toplevel)" == "$WT" ]] || die "invalid Linux worktree"
    [[ "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" == "$VENDOR/.git" ]] \
        || die "Linux worktree does not belong to the vendor"
    if [[ -d "$(git -C "$WT" rev-parse --git-path rebase-apply)" ]]; then git -C "$WT" am --abort; fi
    git -C "$WT" reset --hard "$PIN"
    git -C "$WT" clean -fdx -- src/edge0/
    replay_band common
    replay_band linux
    log "assembled tree: $(git -C "$WT" rev-parse 'HEAD^{tree}')"
}

replay_band() {
    local patch patches=("$PDIR/$1/"*.patch)
    [[ -f "${patches[0]}" ]] || die "empty patch band: $1"
    for patch in "${patches[@]}"; do
        log "am $1/$(basename -- "$patch")"
        if ! git -c user.name='Edge0 build' -c user.email=build@edge0.invalid \
            -c commit.gpgsign=false -c core.hooksPath=/dev/null -C "$WT" am --3way "$patch"; then
            git -C "$WT" am --abort || true
            die "patch replay failed: $1/$(basename -- "$patch")"
        fi
    done
}

stage_serve() {
    log "[3/4] staging Linux serving pieces"
    local sources=() file
    shopt -s nullglob
    sources=("$ROOT/serve/"*.cc "$ROOT/serve/"*.cpp "$ROOT/serve/"*.h)
    shopt -u nullglob
    if (( ${#sources[@]} == 0 )); then
        log "linux/serve is empty; optional memory/prefetch hooks are not built"
        return
    fi
    mkdir -p -- "$WT/src/edge0"
    for file in "${sources[@]}"; do cp -- "$file" "$WT/src/edge0/"; done
}

configure() {
    local bd=$WT/$BUILD_DIR cuda=OFF vulkan=OFF
    [[ "$BACKEND" != cuda ]] || cuda=ON
    [[ "$BACKEND" != vulkan ]] || vulkan=ON
    [[ ! -L "$bd" ]] || die "build directory must not be a symlink"
    if (( CLEAN )); then rm -rf -- "$bd"; fi
    log "[4/4] configure Release/$BACKEND (AVX2 CPU backend)"
    cmake -S "$WT" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DGGML_BMI2=ON -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF \
        -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
        -DGGML_CUDA="$cuda" -DGGML_VULKAN="$vulkan" -DGGML_METAL=OFF -DGGML_BLAS=OFF \
        -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
}

main() {
    parse_args "$@"
    check_inputs
    prepare_vendor
    prepare_worktree
    stage_serve
    if (( ASSEMBLE_ONLY )); then log "OK (assemble-only): $WT"; return; fi
    configure
    if (( CONFIGURE_ONLY )); then log "OK (configure-only): $WT/$BUILD_DIR"; return; fi
    cmake --build "$WT/$BUILD_DIR" --config Release --parallel "$JOBS" \
        --target llama-server llama-bench llama-batched-bench
    log "OK: $WT/$BUILD_DIR/bin/llama-server"
}

main "$@"
