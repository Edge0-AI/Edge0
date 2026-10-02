# scripts/vendor-build.ps1 — llama engine assembly for the Windows product (monorepo layout)
# The pristine upstream tree (../vendor/llama.cpp, pinned commit) is NEVER patched in place:
# this script replays the patch bands into an isolated build worktree (../wt/win), copies the
# edge0 serving pieces from serve/ into src/edge0 (patch #3's CMake glob compiles them into
# the llama target), then configures and builds with Vulkan.
# The vendor tree is submodule-free: it is MATERIALIZED on first run by cloning upstream at
# the pinned commit listed in vendor.llama.pin (repo root); afterwards it is just a checkout.
# Depot resolution (first match wins):
#   A. $env:EDGE0_DEPOT                                   explicit override
#   B. parent directory containing vendor.llama.pin       monorepo checkout (the normal path)
#   If neither matches, stop and explain — run from a full monorepo clone, or point
#   EDGE0_DEPOT at a depot that carries vendor.llama.pin.
# Set EDGE0_LLAMA_URL to clone from a mirror instead of GitHub.
# Usage: pwsh -File scripts/vendor-build.ps1 [-BuildDir build-vk] [-AssembleOnly]
#   -AssembleOnly = stages 1-3 only (checks + patch replay + serve copy), no compile.
param(
    [string]$BuildDir = "build-vk",
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

    Write-Host "[4/4] configure + build (Release/Vulkan)" -ForegroundColor Cyan
    $bd = Join-Path $WT $BuildDir
    cmake -S $WT -B $bd -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build $bd --config Release
    Write-Host "OK: $bd/bin/Release/llama-server.exe" -ForegroundColor Green
} finally {
    Pop-Location
}
