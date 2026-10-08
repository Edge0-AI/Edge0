# scripts/vendor-build.ps1 — llama engine assembly for the Windows product (monorepo layout)
# The pristine upstream tree (../vendor/llama.cpp, pinned commit) is NEVER patched in place:
# this script replays the patch bands into an isolated build worktree (../wt/win), copies the
# edge0 serving pieces from serve/ into src/edge0 (patch #3's CMake glob compiles them into
# the llama target), then configures and builds the selected GPU backend.
#   -Backend hip     (default) ROCm/HIP for RDNA2 (gfx103x) and RDNA4 (gfx120x). Needs the AMD
#                    HIP SDK for Windows (6.1 or newer, sets HIP_PATH) and Ninja on PATH.
#   -Backend vulkan  Vulkan fallback (any AMD/NVIDIA/Intel GPU), Visual Studio generator.
# The vendor tree is submodule-free: it is MATERIALIZED on first run by cloning upstream at
# the pinned commit listed in vendor.llama.pin (repo root); afterwards it is just a checkout.
# Depot resolution (first match wins):
#   A. $env:EDGE0_DEPOT                                   explicit override
#   B. parent directory containing vendor.llama.pin       monorepo checkout (the normal path)
#   If neither matches, stop and explain — run from a full monorepo clone, or point
#   EDGE0_DEPOT at a depot that carries vendor.llama.pin.
# Set EDGE0_LLAMA_URL to clone from a mirror instead of GitHub.
# Usage: pwsh -File scripts/vendor-build.ps1 [-Backend hip|vulkan] [-GpuTargets "gfx1030;gfx1200;gfx1201"]
#                                            [-BuildDir build-hip] [-AssembleOnly]
#   -AssembleOnly = stages 1-3 only (checks + patch replay + serve copy), no compile.
# The app finds the result through EDGE0_BACKEND (hip|vulkan) or its depot-build default
# (see windows/app/src-tauri/src/paths.rs); EDGE0_BIN_DIR overrides everything.
param(
    [ValidateSet("hip", "vulkan")][string]$Backend = "hip",
    [string]$BuildDir = "",
    [ValidatePattern('^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$')][string]$GpuTargets = "gfx1030;gfx1200;gfx1201",
    [switch]$Clean,
    [switch]$AssembleOnly
)
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$DEPOT = if ($env:EDGE0_DEPOT) { $env:EDGE0_DEPOT } else {
    $parent = Split-Path -Parent $ROOT
    if (Test-Path (Join-Path $parent "vendor.llama.pin")) { $parent } else { $null }
}
if (-not $DEPOT) { throw "no depot: set EDGE0_DEPOT to a tree carrying vendor.llama.pin (the monorepo root), or run inside a full monorepo clone" }
$PINFILE = Join-Path $DEPOT "vendor.llama.pin"
if (-not (Test-Path $PINFILE)) { throw "vendor.llama.pin missing in depot $DEPOT" }
$PIN = (Get-Content $PINFILE -Raw).Trim().Split(" ")[0]   # pinned upstream commit (tag b11100) — ledger: patches/llama.cpp/README.md
$VENDOR = Join-Path $DEPOT "vendor/llama.cpp"
$WT     = Join-Path $DEPOT "wt/win"
# Patch bands resolve from the depot root ONLY — one source of truth. The old in-repo
# windows/patches/ mirror was retired 2026-09-30 (a drifting copy; depot carries the bands).
$PDIR = Join-Path $DEPOT "patches/llama.cpp"
if (-not (Test-Path (Join-Path $PDIR "common"))) { throw "patch bands missing at $PDIR/common — run inside the monorepo or point EDGE0_DEPOT at a depot carrying patches/llama.cpp" }
$BANDS = @("common", "windows") | ForEach-Object { Join-Path $PDIR "$_/*.patch" }

if (-not $BuildDir) { $BuildDir = if ($Backend -eq "hip") { "build-hip" } else { "build-vk" } }

# Materialize the pinned tree when absent (submodule-free supply).
if (-not (Test-Path (Join-Path $VENDOR ".git"))) {
    Write-Host "[0/4] materializing vendor: clone llama.cpp at pinned $PIN -> $VENDOR" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Split-Path $VENDOR) | Out-Null
    $LLAMA_URL = if ($env:EDGE0_LLAMA_URL) { $env:EDGE0_LLAMA_URL } else { "https://github.com/ggml-org/llama.cpp" }
    git clone --filter=blob:none $LLAMA_URL $VENDOR
    if ($LASTEXITCODE -ne 0) { throw "clone failed ($LLAMA_URL) — network? or set EDGE0_LLAMA_URL / EDGE0_DEPOT" }
    git -C $VENDOR checkout --detach $PIN
    if ($LASTEXITCODE -ne 0) { throw "pinned commit $PIN not found upstream (refresh the patch ledger)" }
}

# HIP SDK discovery: HIP_PATH must point at the SDK; its clang lives in lib\llvm\bin (SDK 6.x/7.x)
# or bin. Upstream builds with HIP's own clang, not MSVC, so both compilers are passed explicitly.
function Find-HipSdk {
    $root = $env:HIP_PATH
    if (-not $root -or -not (Test-Path $root)) {
        throw "HIP SDK not found: install the AMD HIP SDK for Windows (6.1 or newer; it sets HIP_PATH), or build with -Backend vulkan"
    }
    $clang = @((Join-Path $root "lib\llvm\bin\clang.exe"), (Join-Path $root "bin\clang.exe")) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $clang) { throw "no clang.exe under $root (looked in lib\llvm\bin and bin)" }
    $clangxx = Join-Path (Split-Path $clang) "clang++.exe"
    if (-not (Test-Path $clangxx)) { throw "no clang++.exe next to $clang" }
    return [pscustomobject]@{ Root = $root; Clang = $clang; ClangXX = $clangxx }
}

# The engine's own folder must hold the HIP runtime: the driver's amdhip64 DLL in System32
# is searched before PATH and can be a different version (see release.yml, issue #26929).
# rocBLAS also loads its kernel data from rocblas\library next to its DLL; only the libraries
# for the requested targets (plus arch-neutral files) are copied to keep the folder small.
function Copy-HipRuntime($hip, [string]$binDir, [string]$targets) {
    $sdkBin = Join-Path $hip.Root "bin"
    $patterns = @("amdhip64*.dll", "amd_comgr*.dll", "hipblas*.dll", "rocblas*.dll", "hiprtc*.dll", "libomp*.dll")
    foreach ($pat in $patterns) {
        Get-ChildItem (Join-Path $sdkBin $pat) -File -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName $binDir -Force }
    }
    $rbLib = Join-Path $sdkBin "rocblas\library"
    if (Test-Path $rbLib) {
        $dst = Join-Path $binDir "rocblas\library"
        New-Item -ItemType Directory -Force $dst | Out-Null
        $archs = $targets.Split(";")
        Get-ChildItem $rbLib -File | Where-Object {
            $n = $_.Name
            ($archs | Where-Object { $n -like "*$_*" }).Count -gt 0 -or $n -notmatch "gfx"
        } | ForEach-Object { Copy-Item $_.FullName $dst -Force }
    }
}

Push-Location $VENDOR
try {
    Write-Host "[1/4] checking pristine vendor tree (must sit exactly at the pin, zero local changes)" -ForegroundColor Cyan
    $head = (git rev-parse HEAD).Trim()
    if (-not $head.StartsWith($PIN)) { throw "vendor HEAD=$head is not the pinned $PIN — the pristine tree must never be patched in place; restore with: git checkout --detach $PIN" }
    if ((git status --porcelain).Length -gt 0) { throw "vendor worktree is dirty — the pristine source of truth tolerates no local edits" }

    Write-Host "[2/4] (re)creating build worktree $WT and replaying patch bands" -ForegroundColor Cyan
    if (-not (Test-Path $WT)) { git worktree add --detach $WT $PIN | Out-Null }
    # Unconditional reset before replay = idempotent
    git -C $WT reset --hard $PIN 2>&1 | Out-Null
    git -C $WT clean -fd src/edge0 tools/llama-bench 2>&1 | Out-Null
    Set-Location $WT
    foreach ($pat in $BANDS) {
        Get-ChildItem $pat | Sort-Object Name | ForEach-Object {
            Write-Host "  am $($_.Name)"
            git am --3way $_.FullName
            if ($LASTEXITCODE -ne 0) { throw "git am failed at $($_.Name) (upstream drift or band conflict — three-way by hand, see patches/llama.cpp/README.md)" }
        }
    }

    Write-Host "[3/4] copying serve/ pieces -> src/edge0/ (compiled in via patch #3 glob)" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Join-Path $WT "src/edge0") | Out-Null
    Copy-Item (Join-Path $ROOT "serve/*.cc") (Join-Path $WT "src/edge0/") -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $ROOT "serve/*.h")  (Join-Path $WT "src/edge0/") -Force -ErrorAction SilentlyContinue
    if ($AssembleOnly) {
        Write-Host "OK(assemble-only): worktree=$WT — patches replayed, serving pieces copied" -ForegroundColor Green
        return
    }

    $bd = Join-Path $WT $BuildDir
    if ($Backend -eq "vulkan") {
        Write-Host "[4/4] configure + build (Release/Vulkan, $BuildDir)" -ForegroundColor Cyan
        cmake -S $WT -B $bd "-DGGML_VULKAN=ON" "-DCMAKE_BUILD_TYPE=Release"
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (vulkan)" }
        cmake --build $bd --config Release
        if ($LASTEXITCODE -ne 0) { throw "build failed (vulkan)" }
        $exe = Join-Path $bd "bin\Release\llama-server.exe"
    } else {
        Write-Host "[4/4] configure + build (Release/HIP, targets $GpuTargets, $BuildDir)" -ForegroundColor Cyan
        $hip = Find-HipSdk
        $env:HIP_PATH = $hip.Root
        cmake -S $WT -B $bd -G Ninja "-DGGML_HIP=ON" "-DGPU_TARGETS=$GpuTargets" "-DCMAKE_C_COMPILER=$($hip.Clang)" "-DCMAKE_CXX_COMPILER=$($hip.ClangXX)" "-DCMAKE_BUILD_TYPE=Release"
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (hip): needs HIP SDK 6.1+, Ninja on PATH, and GPU targets the SDK supports ($GpuTargets)" }
        cmake --build $bd
        if ($LASTEXITCODE -ne 0) { throw "build failed (hip)" }
        $bin = Join-Path $bd "bin"
        Copy-HipRuntime $hip $bin $GpuTargets
        $exe = Join-Path $bin "llama-server.exe"
    }
    Write-Host "OK: $exe (backend $Backend)" -ForegroundColor Green
} finally {
    Pop-Location
}
