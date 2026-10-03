# edge0/scripts/patch-green.ps1 — patch-band `git am` gate (final arbiter after an upstream refresh; seconds, no compile)
# The vendor tree is submodule-free: materialized on demand by cloning upstream at the pinned
# commit from vendor.llama.pin (repo root). Set EDGE0_LLAMA_URL to clone from a mirror.
# Usage: pwsh -File patch-green.ps1 [-Bands common,windows] [-Pin <sha>] [-Tree <expected-tree-hash>] [-Keep]
#        (-Pin defaults to vendor.llama.pin)
# Exit codes: 0 = GREEN (all bands replayed clean [+ tree parity]), 1 = RED.
param(
    [string]$Bands = "common,windows",
    [string]$Pin   = "",
    [string]$Tree  = "",
    [switch]$Keep
)
$ErrorActionPreference = "Stop"
$DEPOT  = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)   # = edge0 root
$VENDOR = Join-Path $DEPOT "vendor/llama.cpp"
$WT     = Join-Path $DEPOT "wt/greencheck"
if (-not $Pin) {
    if (-not (Test-Path (Join-Path $DEPOT "vendor.llama.pin"))) { throw "vendor.llama.pin missing at depot root (or pass -Pin)" }
    $Pin = (Get-Content (Join-Path $DEPOT "vendor.llama.pin") -Raw).Trim().Split(" ")[0]
}
if (-not (Test-Path (Join-Path $VENDOR ".git"))) {
    Write-Host "[gate] materializing vendor: clone llama.cpp at pinned $Pin -> $VENDOR" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Split-Path $VENDOR) | Out-Null
    $LLAMA_URL = if ($env:EDGE0_LLAMA_URL) { $env:EDGE0_LLAMA_URL } else { "https://github.com/ggml-org/llama.cpp" }
    git clone --filter=blob:none $LLAMA_URL $VENDOR
    if ($LASTEXITCODE -ne 0) { throw "clone failed ($LLAMA_URL)" }
    git -C $VENDOR checkout --detach $Pin
    if ($LASTEXITCODE -ne 0) { throw "pinned commit $Pin not found" }
}

Push-Location $VENDOR
try {
    if (Test-Path $WT) { Pop-Location; git -C $VENDOR worktree remove --force $WT | Out-Null; Push-Location $VENDOR }
    git worktree add --detach $WT $Pin | Out-Null
    Push-Location $WT
    $n = 0; $total = 0
    foreach ($band in $Bands.Split(",")) {
        $files = Get-ChildItem (Join-Path $DEPOT "patches/llama.cpp/$band/*.patch") | Sort-Object Name
        foreach ($f in $files) {
            $total++
            git am --3way $f.FullName *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "RED: git am failed at $band\$($f.Name)" -ForegroundColor Red
                git am --abort *> $null
                exit 1
            }
            $n++
        }
    }
    $actual = (git rev-parse "HEAD^{tree}").Trim()
    Write-Host "applied=$n/$total  tree=$actual"
    if ($Tree -and $actual -ne $Tree) {
        Write-Host "RED: tree parity failed -- expected $Tree, got $actual" -ForegroundColor Red
        exit 1
    }
    Write-Host "GREEN: band replay clean$(if ($Tree) { ' + tree parity' })" -ForegroundColor Green
} finally {
    Pop-Location
    if (-not $Keep) { git -C $VENDOR worktree remove --force $WT *> $null }
}
