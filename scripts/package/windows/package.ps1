# Builds the Windows package of Edge0 Chat, for x64 or Arm64: Edge0 Chat.exe, edge0-infer.exe, and edge0-strata with
# its CUDA engine and the Python runtime of its server, in one zip. README.md in this folder describes the machine and
# every step.
#
#   package.ps1 fetch  -Cache DIR [-Commit REV]
#   package.ps1 build  -Cache DIR -Out DIR [-Commit REV] [-Work DIR] [-KeepWork]
#   package.ps1 verify -Zip FILE [-Model DIR [-Device NAME]]
#
# Every command also takes -Arch x64|arm64 (default: the architecture of this machine) and -EnvScript FILE, a script
# run first that sets PATH and the other variables of a toolchain installed outside the usual places.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('fetch', 'build', 'verify')][string]$Command,
    [ValidateSet('x64', 'arm64')][string]$Arch,
    [string]$Cache,
    [string]$Out,
    [string]$Commit = 'HEAD',
    [string]$Work = 'C:\edge0-package',
    [switch]$KeepWork,
    [string]$Zip,
    [string]$Model,
    [string]$Device,
    [string]$EnvScript
)

Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

# Pinned inputs. Python 3.12.10 is the last 3.12 release with Windows binaries.
$PyVersion = '3.12.10'
$LlamaCommit = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
$Cfg = @{
    arm64 = @{
        Title = 'Arm64'; PyArch = 'arm64'; PySha = '3065efc3d382d1cda66757ac71ade11904fa6e350f5a97eb74811acd71ba5532'
        Wheel = 'win_arm64'; Rust = 'aarch64-pc-windows-msvc'; Machine = '0xAA64'; Crt = 'arm64'; CudaArchs = '121'
        GgmlCpu = @('-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm'); GgmlText = '`armv8.2-a+dotprod+fp16+i8mm`'
        # vcruntime140_1.dll is used only by x64 code.
        DropCrt = @('vcruntime140_1.dll')
    }
    x64   = @{
        Title = 'x64'; PyArch = 'amd64'; PySha = '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3'
        Wheel = 'win_amd64'; Rust = 'x86_64-pc-windows-msvc'; Machine = '0x8664'; Crt = 'x64'; CudaArchs = '75;86;89;120'
        GgmlCpu = @('-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON'); GgmlText = 'AVX2 with FMA and F16C'
        DropCrt = @()
    }
}
$StrataPaths = @('edge0-strata/edge0', 'edge0-strata/serve', 'edge0-strata/strata_control',
    'edge0-strata/strata_cli/__init__.py', 'edge0-strata/strata_cli/client.py', 'edge0-strata/tools/gguf_reader.py',
    'edge0-strata/tools/strata_tokenizer.py', 'edge0-strata/tools/dflash2_pack.py', 'edge0-strata/tools/mtp_gguf.py')
$Licenses = [ordered]@{
    'GGML-MIT.txt'                = 'edge0-infer/third_party/GGML_LICENSE'
    'llama.cpp-tokenizer-MIT.txt' = 'edge0-infer/src/app/text/LICENSE.llama.cpp'
    'cpp-httplib-MIT.txt'         = 'edge0-infer/third_party/cpp-httplib/LICENSE'
    'nlohmann-json-MIT.txt'       = 'edge0-infer/third_party/nlohmann-json/LICENSE.MIT'
    'Strata-MIT.txt'              = 'edge0-strata/LICENSE'
    'Project-Maya-MIT.txt'        = 'edge0-strata/third_party/project-maya/LICENSE'
    'Project-Maya-NOTICE.md'      = 'edge0-strata/third_party/project-maya/NOTICE.md'
    'libstrix-NOTICE.md'          = 'edge0-strata/third_party/libstrix/NOTICE.md'
}
$PyImports = 'numpy, jinja2, regex, yaml, tqdm, requests, PIL, psutil, certifi, truststore, charset_normalizer, idna, urllib3, markupsafe, colorama'
$Marker = '.edge0-windows-package'

function Die([string]$Message) { Write-Host "package.ps1: $Message" -ForegroundColor Red; exit 1 }
function Step([string]$Message) { Write-Host "== $Message" -ForegroundColor Cyan }

function Need([string[]]$Names) {
    $miss = @($Names | Where-Object { -not (Get-Command $_ -ErrorAction SilentlyContinue) })
    if ($miss.Count) { Die "not on PATH: $($miss -join ', ') (see README.md: Build machine)" }
}

# Runs a program with its output in a log file; on failure prints the end of the log and stops.
function Invoke-Logged([string]$Name, [string]$Exe, [string[]]$ArgList) {
    $log = Join-Path $script:Logs "$Name.log"
    Write-Host "   $Name (log: $log)"
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Exe @ArgList 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) {
        Select-String -Path $log -Pattern '^FAILED:|\): error|\berror:|fatal error|^error\b' | Select-Object -First 10 |
            ForEach-Object { Write-Host "   ! $($_.Line)" }
        Get-Content $log -Tail 15 | ForEach-Object { Write-Host "   | $_" }
        Die "$Name failed (exit $code); the log is $log"
    }
}

# Runs a program and returns its output lines (stderr included); stops if it fails.
function Get-Output([string]$Exe, [string[]]$ArgList) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $lines = @(& $Exe @ArgList 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) { $lines | Select-Object -Last 20 | ForEach-Object { Write-Host "   | $_" }; Die "$Exe $($ArgList -join ' ') failed (exit $code)" }
    return , $lines
}

function Remove-Tree([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        cmd.exe /c "rd /s /q `"$Path`"" | Out-Null
        if (Test-Path -LiteralPath $Path) { Die "cannot remove $Path" }
    }
}

function Write-Text([string]$Path, [string]$Text) { [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding $false)) }

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }

function Get-Pinned([string]$Url, [string]$Path, [string]$Sha) {
    if (-not (Test-Path -LiteralPath $Path) -or (Get-Sha256 $Path) -ne $Sha) {
        Invoke-Logged ('download-' + (Split-Path $Path -Leaf)) 'curl.exe' @('-fsSL', '--retry', '3', '-o', "$Path.part", $Url)
        Move-Item -Force "$Path.part" $Path
    }
    $got = Get-Sha256 $Path
    if ($got -ne $Sha) { Die "SHA-256 of $Path is $got, expected $Sha" }
}

function Resolve-Commit {
    $sha = (Get-Output 'git' @('-C', $script:Repo, 'rev-parse', '--verify', "$Commit^{commit}"))[0].Trim()
    if ($sha -notmatch '^[0-9a-f]{40}$') { Die "not a commit: $Commit" }
    return $sha
}

function Get-FullPath([string]$Path, [string]$Option) {
    if (-not $Path) { Die "$Option DIR is required" }
    if ($Path -match '\s') { Die "$Option must be a path without spaces: $Path" }
    return [IO.Path]::GetFullPath($Path).TrimEnd('\')
}

$HostArch = if ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment').PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
if (-not $Arch) { $Arch = $HostArch }
$C = $Cfg[$Arch]
if ($EnvScript) {
    if (-not (Test-Path -LiteralPath $EnvScript)) { Die "no such file: $EnvScript" }
    & $EnvScript
}
$script:Logs = Join-Path ([IO.Path]::GetTempPath()) 'edge0-package-logs'

# ---------------------------------------------------------------------------------------------------------------------
function Invoke-Fetch {
    Need @('git', 'curl.exe', 'tar.exe', 'python', 'cargo', 'npm.cmd')
    $script:Repo = (Get-Output 'git' @('-C', $PSScriptRoot, 'rev-parse', '--show-toplevel'))[0]
    $cache = Get-FullPath $Cache '-Cache'
    $sha = Resolve-Commit
    New-Item -ItemType Directory -Force $cache | Out-Null
    $script:Logs = Join-Path $cache 'logs'; New-Item -ItemType Directory -Force $script:Logs | Out-Null
    # Windows PowerShell drops the double quotes of native arguments, so the Python code here has none.
    $pyv = (Get-Output 'python' @('-c', 'import sys; print(*sys.version_info[:2], sep=chr(46))'))[0]
    if ([version]$pyv -lt [version]'3.12') { Die "python on PATH is $pyv; 3.12 or later reads the markers of requirements.txt right" }

    Step "Python $PyVersion embeddable package ($Arch)"
    $pyZip = Join-Path $cache "python-$PyVersion-embed-$($C.PyArch).zip"
    Get-Pinned "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-$($C.PyArch).zip" $pyZip $C.PySha

    Step "llama.cpp $LlamaCommit"
    $ll = Join-Path $cache "llama.cpp-$LlamaCommit.tar.gz"
    if (-not (Test-Path -LiteralPath $ll)) {
        Invoke-Logged 'download-llama.cpp' 'curl.exe' @('-fsSL', '--retry', '3', '-o', "$ll.part", "https://github.com/ggml-org/llama.cpp/archive/$LlamaCommit.tar.gz")
        Move-Item -Force "$ll.part" $ll
    }

    Step "sources of $sha"
    $tmp = Join-Path $cache 'tmp'
    Remove-Tree $tmp
    New-Item -ItemType Directory -Force "$tmp\src" | Out-Null
    Invoke-Logged 'git-archive' 'git' @('-C', $script:Repo, 'archive', '--format=tar', '-o', "$tmp\src.tar", $sha)
    Invoke-Logged 'untar-src' 'tar.exe' @('-xf', "$tmp\src.tar", '-C', "$tmp\src")

    Step "Python wheels ($($C.Wheel))"
    $wheels = Join-Path $cache "wheels-$Arch"
    Invoke-Logged "pip-download-$Arch" 'python' @('-m', 'pip', 'download', '--disable-pip-version-check', '--only-binary=:all:',
        '--platform', $C.Wheel, '--python-version', '3.12', '--implementation', 'cp', '-d', $wheels,
        '-r', "$tmp\src\edge0-strata\requirements.txt")

    Step 'Rust crates (cargo vendor)'
    Invoke-Logged 'cargo-vendor' 'cargo' @('vendor', '--locked', '--versioned-dirs', '--manifest-path',
        "$tmp\src\edge0-infer\examples\desktop\src-tauri\Cargo.toml", "$cache\cargo-vendor")

    Step 'npm packages'
    Push-Location "$tmp\src\edge0-infer\examples\desktop"
    try { Invoke-Logged 'npm-ci' 'npm.cmd' @('ci', '--cache', "$cache\npm", '--ignore-scripts', '--no-audit', '--no-fund') }
    finally { Pop-Location }

    Remove-Tree $tmp
    Write-Text (Join-Path $cache 'COMMIT') "$sha`n"
    Step "fetched into $cache"
}

# ---------------------------------------------------------------------------------------------------------------------
function Invoke-Build {
    if ($Arch -ne $HostArch) { Die "build the $Arch package on a $Arch machine; this one is $HostArch" }
    $compiler = if ($Arch -eq 'arm64') { 'clang-cl' } else { 'cl' }
    Need @('git', 'tar.exe', 'python', 'cmake', 'ninja', 'cl', 'link', $compiler, 'nvcc', 'glslc', 'cargo', 'rustc', 'node', 'npm.cmd')
    foreach ($v in 'CUDA_PATH', 'VULKAN_SDK', 'VCToolsRedistDir') {
        if (-not [Environment]::GetEnvironmentVariable($v)) { Die "$v is not set (see README.md: Build machine)" }
    }
    $script:Repo = (Get-Output 'git' @('-C', $PSScriptRoot, 'rev-parse', '--show-toplevel'))[0]
    $cache = Get-FullPath $Cache '-Cache'
    $out = Get-FullPath $Out '-Out'
    $work = Get-FullPath $Work '-Work'
    $sha = Resolve-Commit
    $cached = if (Test-Path "$cache\COMMIT") { (Get-Content "$cache\COMMIT" -Raw).Trim() } else { '' }
    if ($cached -ne $sha) { Die "$cache was fetched for '$cached', not ${sha}: run fetch for this commit" }
    $pyZip = Join-Path $cache "python-$PyVersion-embed-$($C.PyArch).zip"
    $ll = Join-Path $cache "llama.cpp-$LlamaCommit.tar.gz"
    $wheels = Join-Path $cache "wheels-$Arch"
    foreach ($p in $pyZip, $ll, $wheels, "$cache\cargo-vendor", "$cache\npm") {
        if (-not (Test-Path -LiteralPath $p)) { Die "missing in the cache: $p (run fetch -Arch $Arch)" }
    }
    if ((Get-Sha256 $pyZip) -ne $C.PySha) { Die "SHA-256 of $pyZip does not match" }
    $crt = Get-ChildItem -Directory (Join-Path $env:VCToolsRedistDir $C.Crt) -Filter 'Microsoft.VC14*.CRT' -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1
    if (-not $crt) { Die "no Microsoft.VC14*.CRT folder in $env:VCToolsRedistDir\$($C.Crt)" }

    if (Test-Path -LiteralPath $work) {
        if (-not (Test-Path -LiteralPath "$work\$Marker")) { Die "$work exists and was not made by this script; choose another -Work" }
        Remove-Tree $work
    }
    New-Item -ItemType Directory -Force $work, $out | Out-Null
    Write-Text "$work\$Marker" ''
    $script:Logs = Join-Path $out "logs-$Arch"
    Remove-Tree $script:Logs
    New-Item -ItemType Directory -Force $script:Logs | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()

    Step "clean checkout of $sha"
    $src = "$work\src"
    Invoke-Logged 'git-clone' 'git' @('clone', '-q', '--no-checkout', '-c', 'core.autocrlf=false', '-c', 'core.fileMode=false', $script:Repo, $src)
    Invoke-Logged 'git-checkout' 'git' @('-C', $src, 'checkout', '-q', '--detach', $sha)
    if (Get-Output 'git' @('-C', $src, 'status', '--porcelain')) { Die 'the checkout is not clean' }
    $version = (Get-Content -Raw "$src\edge0-infer\examples\desktop\src-tauri\tauri.conf.json" | ConvertFrom-Json).version
    $pkg = "edge0-chat-$version-windows-$Arch"
    $llama = "$work\llama.cpp"
    New-Item -ItemType Directory -Force $llama | Out-Null
    Invoke-Logged 'untar-llama.cpp' 'tar.exe' @('-xzf', $ll, '--strip-components=1', '-C', $llama)

    # Paths of this machine that must not end up in the binaries.
    $maps = [ordered]@{ "$work\" = '/build/'; "$cache\" = '/cache/' }
    if ($env:USERPROFILE -notmatch '\s') { $maps["$env:USERPROFILE\"] = '/home/' }
    $pmap = @(); $trim = @(); $remap = @()
    foreach ($k in $maps.Keys) {
        foreach ($p in $k, $k.Replace('\', '/')) {
            $pmap += "/clang:-ffile-prefix-map=$p=$($maps[$k])"
            $trim += "/d1trimfile:$p"
            $remap += "--remap-path-prefix=$p=$($maps[$k])"
        }
    }
    $hide = if ($Arch -eq 'arm64') { $pmap -join ' ' } else { $trim -join ' ' }
    $cflags = "/DWIN32 /D_WINDOWS /Brepro $hide"
    $cxxflags = "/DWIN32 /D_WINDOWS /EHsc /GR /Brepro $hide"
    $cl = (Get-Command cl).Source
    if ($Arch -eq 'arm64') {
        $cc = (Get-Command clang-cl).Source
        $ccArgs = @("-DCMAKE_C_COMPILER=$cc", "-DCMAKE_CXX_COMPILER=$cc", '-DCMAKE_OSX_ARCHITECTURES=arm64')
        $inferExtra = @("-DCMAKE_LINKER=$(Join-Path (Split-Path $cc) 'lld-link.exe')")
        $strataExtra = @('-DSTRATA_FORCE_ARM64=ON')
    } else {
        $ccArgs = @("-DCMAKE_C_COMPILER=$cl", "-DCMAKE_CXX_COMPILER=$cl")
        $inferExtra = @(); $strataExtra = @()
    }

    Step 'edge0-infer'
    $binfer = "$work\b\infer"
    Push-Location "$src\edge0-infer"
    try {
        Invoke-Logged 'infer-configure' 'cmake' (@('--preset', 'windows-vulkan', '-B', $binfer, '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded', "-DCMAKE_C_FLAGS=$cflags", "-DCMAKE_CXX_FLAGS=$cxxflags",
            '-DCMAKE_EXE_LINKER_FLAGS=/Brepro', '-DGGML_NATIVE=OFF', '-DGGML_BUILD_TESTS=OFF', '-DGGML_BUILD_EXAMPLES=OFF',
            '-DMOE_BUILD_TESTS=OFF', '-DMOE_BUILD_EXAMPLES=OFF', '-DMOE_BUILD_CLI=ON') + $ccArgs + $inferExtra + $C.GgmlCpu)
    } finally { Pop-Location }
    Invoke-Logged 'infer-build' 'cmake' @('--build', $binfer, '--target', 'edge0-infer')

    Step "edge0-strata (CUDA $($C.CudaArchs))"
    $bstrata = "$work\b\strata"
    $cudaFlags = '--use-local-env -D_WINDOWS -Xcompiler=/EHsc -Xcompiler=/GR -Xcompiler=/Brepro ' + (($trim | ForEach-Object { "-Xcompiler=$_" }) -join ' ')
    Invoke-Logged 'strata-configure' 'cmake' (@('-S', "$src\edge0-strata", '-B', $bstrata, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_CUDA_COMPILER=$((Get-Command nvcc).Source)", "-DCMAKE_CUDA_HOST_COMPILER=$cl", "-DCMAKE_CUDA_FLAGS=$cudaFlags",
        "-DCMAKE_C_FLAGS=$cflags", "-DCMAKE_CXX_FLAGS=$cxxflags",
        '-DCMAKE_EXE_LINKER_FLAGS=/Brepro /PDBALTPATH:%_PDB%', '-DCMAKE_SHARED_LINKER_FLAGS=/Brepro /PDBALTPATH:%_PDB%',
        "-DCMAKE_CUDA_ARCHITECTURES=$($C.CudaArchs)", '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_PORTABLE=ON',
        "-DSTRATA_GGML_DIR=$llama", '-DSTRATA_BUILD_TESTS=OFF') + $ccArgs + $strataExtra)
    Invoke-Logged 'strata-build' 'cmake' @('--build', $bstrata, '--target', 'strata', 'strata_kq')

    Step 'Edge0 Chat'
    $app = "$src\edge0-infer\examples\desktop"
    $cargoHome = "$work\cargo-home"
    New-Item -ItemType Directory -Force $cargoHome | Out-Null
    $vendor = "$cache\cargo-vendor".Replace('\', '/')
    Write-Text "$cargoHome\config.toml" "[source.crates-io]`nreplace-with = `"vendored`"`n[source.vendored]`ndirectory = `"$vendor`"`n[net]`noffline = true`n"
    $saved = @{}
    foreach ($v in 'CARGO_HOME', 'RUSTFLAGS', 'npm_config_cache', 'npm_config_offline', 'MOE_DESKTOP_EXAMPLE_APP_ID') {
        $saved[$v] = [Environment]::GetEnvironmentVariable($v)
    }
    $env:CARGO_HOME = $cargoHome
    $env:RUSTFLAGS = ($remap + @('-C', 'target-feature=+crt-static')) -join ' '
    $env:npm_config_cache = "$cache\npm"; $env:npm_config_offline = 'true'
    Remove-Item Env:MOE_DESKTOP_EXAMPLE_APP_ID -ErrorAction SilentlyContinue
    Push-Location $app
    try {
        Invoke-Logged 'chat-npm-ci' 'npm.cmd' @('ci', '--no-audit', '--no-fund')
        Invoke-Logged 'chat-build' 'npm.cmd' @('run', 'tauri', '--', 'build', '--no-bundle', '--ci', '--', '--locked')
    } finally { Pop-Location }

    Step "stage $pkg"
    $stage = "$work\stage\$pkg"
    New-Item -ItemType Directory -Force "$stage\licenses", "$stage\edge0-strata\build" | Out-Null
    function Put([string]$From, [string]$To) {
        $d = Join-Path $stage $To
        New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null
        Copy-Item -LiteralPath (Join-Path $src $From) -Destination $d -Force
    }
    Copy-Item "$app\src-tauri\target\release\edge0-chat-desktop.exe" "$stage\Edge0 Chat.exe"
    Copy-Item "$binfer\bin\edge0-infer.exe" "$stage\edge0-infer.exe"
    Copy-Item "$bstrata\strata.exe", "$bstrata\strata_kq.dll" "$stage\edge0-strata\build\"
    Put 'LICENSE' 'LICENSE'; Put 'NOTICE' 'NOTICE'; Put 'edge0-strata\LICENSE' 'edge0-strata\LICENSE'
    foreach ($k in $Licenses.Keys) { Put $Licenses[$k] "licenses\$k" }
    Copy-Item "$llama\LICENSE" "$stage\licenses\llama.cpp-MIT.txt"
    $tracked = Get-Output 'git' (@('-C', $src, 'ls-files', '--') + $StrataPaths)
    foreach ($f in $tracked) {
        if ($f -match '^edge0-strata/(serve|strata_control)/(.*/)?test_[^/]*\.py$') { continue }
        Put $f $f
    }
    Invoke-Logged 'notices' 'python' @("$PSScriptRoot\notices.py", '--title', "Edge0 Chat $version (Windows $($C.Title))",
        '--target', $C.Rust, '--cargo-manifest', "$app\src-tauri\Cargo.toml", '--vendor', "$cache\cargo-vendor",
        '--npm', $app, '--output', "$stage\licenses\THIRD-PARTY-NOTICES.txt")
    foreach ($v in $saved.Keys) { [Environment]::SetEnvironmentVariable($v, $saved[$v]) }

    Step "Python $PyVersion runtime"
    $sc = "$stage\edge0-strata\.venv\Scripts"
    $sp = "$sc\Lib\site-packages"
    New-Item -ItemType Directory -Force $sc | Out-Null
    Expand-Archive -LiteralPath $pyZip -DestinationPath $sc
    Write-Text "$sc\python312._pth" "python312.zip`r`n.`r`nLib\site-packages`r`n`r`nimport site`r`n"
    Invoke-Logged 'pip-install' 'python' @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', '--find-links', $wheels,
        '--no-deps', '--no-compile', '--only-binary=:all:', '--platform', $C.Wheel, '--python-version', '3.12', '--implementation', 'cp',
        '--target', $sp, '-r', "$src\edge0-strata\requirements.txt")
    Remove-Tree "$sp\bin"
    Get-ChildItem $sp -Recurse -Directory -Force | Where-Object { $_.Name -in 'tests', '__pycache__' } |
        Sort-Object { $_.FullName.Length } -Descending | ForEach-Object { Remove-Tree $_.FullName }
    Copy-Item "$($crt.FullName)\*.dll" $sc -Force
    foreach ($d in $C.DropCrt) { Remove-Item -Force -ErrorAction SilentlyContinue "$sc\$d" }
    Get-ChildItem $stage -Recurse -Directory -Force -Filter '__pycache__' | ForEach-Object { Remove-Tree $_.FullName }
    Get-ChildItem $stage -Recurse -File -Force -Include '*.pyc', '*.pyo', '*.pdb', '*.log' | Remove-Item -Force

    Step 'check modules, PE machine and build paths'
    Invoke-Logged 'check-modules' 'python' @("$PSScriptRoot\check.py", 'modules', "$src\edge0-strata", "$stage\edge0-strata")
    Invoke-Logged 'check-machine' 'python' @("$PSScriptRoot\check.py", 'machine', $stage, $C.Machine)
    Invoke-Logged 'check-paths' 'python' (@("$PSScriptRoot\check.py", 'paths', $stage, $work, $cache, $env:USERPROFILE,
        "\Users\$env:USERNAME\") + $(if ($script:Repo) { @($script:Repo) } else { @() }))
    if (Get-Output 'git' @('-C', $src, 'status', '--porcelain', '--untracked-files=no')) { Die 'the build changed tracked files' }

    Step 'README.md'
    $savedPath = $env:Path
    $env:Path = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:Path"
    $strataVersion = ((Get-Output "$stage\edge0-strata\build\strata.exe" @('--version')) -join ' ').Trim()
    $env:Path = $savedPath
    $nvcc = (Get-Output 'nvcc' @('--version')) -join "`n"
    if ($nvcc -notmatch 'release (\d+)\.(\d+), V(\S+)') { Die 'cannot read the nvcc version' }
    $cudaMajor = $Matches[1]; $nvccText = "$($Matches[1]).$($Matches[2]) ($($Matches[3]))"
    if ($Arch -eq 'arm64') {
        $clangLine = (Get-Output 'clang-cl' @('--version'))[0]
        if ($clangLine -notmatch 'version (\S+)') { Die 'cannot read the clang-cl version' }
        $compilerText = "clang-cl $($Matches[1]) with the MSVC libraries and linker"
        $pathFlags = '`-ffile-prefix-map` (C/C++), `/d1trimfile` (CUDA host code)'
    } else {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $banner = (& cl 2>&1 | ForEach-Object { "$_" }) -join ' '
        $ErrorActionPreference = $old
        if ($banner -notmatch 'Version (\S+)') { Die 'cannot read the MSVC version' }
        $compilerText = "MSVC $($Matches[1])"
        $pathFlags = '`/d1trimfile` (C/C++ and CUDA host code)'
    }
    $wheelList = (Get-ChildItem $sp -Directory -Filter '*.dist-info' | Sort-Object Name | ForEach-Object {
            $_.Name -replace '\.dist-info$', '' -replace '^(.*)-([^-]+)$', '$1 $2' }) -join ', '
    $cmakeVersion = ((Get-Output 'cmake' @('--version'))[0] -split ' ')[-1]
    $values = [ordered]@{
        VERSION = $version; COMMIT = $sha; ARCH_TITLE = $C.Title; PKG = $pkg; RUST_TARGET = $C.Rust
        RUST = ((Get-Output 'rustc' @('--version'))[0] -split ' ')[1]; NODE = (Get-Output 'node' @('--version'))[0].TrimStart('v')
        COMPILER = $compilerText; CMAKE = $cmakeVersion; GGML_CPU = $C.GgmlText; VULKAN_SDK = (Split-Path $env:VULKAN_SDK -Leaf)
        STRATA_VERSION = ($strataVersion -replace '^\D*', ''); NVCC = $nvccText; CUDA_MAJOR = $cudaMajor; CUDA_ARCHS = $C.CudaArchs
        LLAMA = $LlamaCommit; PYTHON = $PyVersion; WHEELS = $wheelList
        CRT = "$($C.Title), $(Split-Path $env:VCToolsRedistDir.TrimEnd('\') -Leaf)"; PATH_FLAGS = $pathFlags
    }
    $readme = Get-Content -Raw "$PSScriptRoot\user-README.md"
    $readme = [regex]::Replace($readme, '(?s)<!-- arch:(\w+) -->\r?\n(.*?)<!-- /arch -->\r?\n', {
            param($m) if ($m.Groups[1].Value -eq $Arch) { $m.Groups[2].Value } else { '' } })
    foreach ($k in $values.Keys) { $readme = $readme.Replace("{{$k}}", [string]$values[$k]) }
    if ($readme -match '\{\{[A-Z_]+\}\}') { Die "README placeholder left: $($Matches[0])" }
    Write-Text "$stage\README.md" $readme

    Step 'zip'
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $zipPath = "$out\$pkg.zip"
    Remove-Item -Force -ErrorAction SilentlyContinue $zipPath, "$zipPath.sha256"
    $base = Split-Path $stage
    $fs = [IO.File]::Open($zipPath, [IO.FileMode]::CreateNew)
    $za = New-Object IO.Compression.ZipArchive($fs, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in (Get-ChildItem $stage -Recurse -File -Force | Sort-Object FullName)) {
            $rel = $f.FullName.Substring($base.Length + 1).Replace('\', '/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($za, $f.FullName, $rel, [IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { $za.Dispose(); $fs.Dispose() }
    Write-Text "$zipPath.sha256" "$(Get-Sha256 $zipPath)  $pkg.zip`n"
    if (-not $KeepWork) { Remove-Tree $work }
    Step "done in $([int]$sw.Elapsed.TotalMinutes) min: $zipPath"
    Get-Content "$zipPath.sha256"
}

# ---------------------------------------------------------------------------------------------------------------------
function Invoke-Verify {
    if (-not $Zip -or -not (Test-Path -LiteralPath $Zip)) { Die '-Zip FILE is required' }
    $zipPath = (Resolve-Path -LiteralPath $Zip).Path
    $pkg = [IO.Path]::GetFileNameWithoutExtension($zipPath)
    if ($pkg -notmatch '^edge0-chat-.+-windows-(arm64|x64)$') { Die "not a package name: $pkg.zip" }
    $za = $Matches[1]; $zc = $Cfg[$za]
    if (Test-Path -LiteralPath "$zipPath.sha256") {
        $want = ((Get-Content "$zipPath.sha256" -Raw) -split '\s+')[0]
        if ((Get-Sha256 $zipPath) -ne $want) { Die "SHA-256 of $zipPath does not match $zipPath.sha256" }
        Write-Host 'SHA-256: ok'
    }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "edge0-verify-$PID"
    Remove-Tree $tmp
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $script:Logs = $tmp
    try {
        Step "extract $pkg.zip"
        Invoke-Logged 'extract' 'tar.exe' @('-xf', $zipPath, '-C', $tmp)
        $root = "$tmp\$pkg"; $st = "$root\edge0-strata"; $py = "$st\.venv\Scripts\python.exe"
        Step 'layout'
        foreach ($f in 'Edge0 Chat.exe', 'edge0-infer.exe', 'README.md', 'LICENSE', 'NOTICE', 'licenses\THIRD-PARTY-NOTICES.txt',
            'licenses\llama.cpp-MIT.txt', 'edge0-strata\LICENSE', 'edge0-strata\build\strata.exe', 'edge0-strata\build\strata_kq.dll',
            'edge0-strata\edge0\CONTRACT', 'edge0-strata\serve\server.py', 'edge0-strata\strata_cli\client.py',
            'edge0-strata\tools\strata_tokenizer.py', 'edge0-strata\.venv\Scripts\python.exe', 'edge0-strata\.venv\Scripts\python312._pth') {
            if (-not (Test-Path -LiteralPath "$root\$f")) { Die "missing: $f" }
        }
        foreach ($k in $Licenses.Keys) { if (-not (Test-Path -LiteralPath "$root\licenses\$k")) { Die "missing: licenses\$k" } }
        # numpy's wheel ships import libraries (.lib), so build outputs are looked for outside the Python runtime only.
        $bad = @(Get-ChildItem $root -Recurse -Force | Where-Object {
                $_.Name -eq '__pycache__' -or $_.Extension -in '.pdb', '.pyc', '.pyo', '.log', '.gguf', '.safetensors' -or
                ($_.Extension -in '.ilk', '.obj', '.lib', '.exp' -and $_.FullName -notmatch '\\\.venv\\') -or
                $_.FullName -match '\\edge0-strata\\(serve|strata_control)\\(.*\\)?test_[^\\]*\.py$' })
        if ($bad.Count) { $bad | ForEach-Object { Write-Host "   $($_.FullName.Substring($root.Length + 1))" }; Die 'files that do not belong in the package' }
        Write-Host 'layout: ok'

        if ($za -ne $HostArch) {
            Write-Host "This is a $za package on a $HostArch machine: only the layout was checked." -ForegroundColor Yellow
            return
        }
        Step "PE machine $($zc.Machine) and build paths"
        Invoke-Logged 'check-machine' $py @('-B', "$PSScriptRoot\check.py", 'machine', $root, $zc.Machine)
        Invoke-Logged 'check-paths' $py @('-B', "$PSScriptRoot\check.py", 'paths', $root, $env:USERPROFILE, "\Users\$env:USERNAME\")
        Write-Host 'PE machine and build paths: ok'

        $env:Path = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
        Step 'edge0-infer info'
        $info = Get-Output "$root\edge0-infer.exe" @('info')
        $info | Where-Object { $_ -match 'engine\.(version|source_commit|source_modified)|vulkan' } | ForEach-Object { Write-Host "   $_" }
        if (-not ($info -match 'engine\.source_modified: no')) { Die 'edge0-infer was not built from a clean tree' }

        Step 'edge0-strata'
        if ($env:CUDA_PATH) {
            $env:Path = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:Path"
            Write-Host ('   ' + ((Get-Output "$st\build\strata.exe" @('--version')) -join ' '))
        } else { Write-Host '   CUDA_PATH is not set: strata.exe --version skipped (it needs cublas)' -ForegroundColor Yellow }
        Get-Output $py @('-B', '-c', "import $PyImports; print('python packages: ok')") | ForEach-Object { Write-Host "   $_" }
        $smoke = "import sys; sys.path[:0] = [r'$st\tools', r'$st']; import strata_tokenizer, gguf_reader; " +
            "from strata_cli.client import normalize_legacy; import strata_control.drafting_fetch; print('edge0-strata modules: ok')"
        Get-Output $py @('-B', '-c', $smoke) | ForEach-Object { Write-Host "   $_" }
        Push-Location $st
        try { Get-Output $py @('-B', 'serve\server.py', '--help') | Out-Null; Write-Host '   server.py --help: ok' } finally { Pop-Location }

        if ($Model) {
            Step "serve one prompt: $Model$(if ($Device) { " on $Device" })"
            if (Get-Process -Name strata, edge0-infer -ErrorAction SilentlyContinue) { Die 'another model service is running' }
            $serveArgs = @('serve', '--model', $Model, '--port', '0')
            if ($Device) { $serveArgs += @('--device', $Device) }
            $log = "$tmp\serve.log"
            $p = Start-Process -FilePath "$root\edge0-infer.exe" -ArgumentList ($serveArgs | ForEach-Object { "`"$_`"" }) -NoNewWindow -PassThru `
                -RedirectStandardOutput $log -RedirectStandardError "$tmp\serve.err"
            try {
                $url = $null
                for ($i = 0; $i -lt 1800 -and -not $url; $i++) {
                    Start-Sleep -Seconds 1
                    $m = Select-String -Path $log, "$tmp\serve.err" -Pattern 'Listening: (http://127\.0\.0\.1:\d+)' | Select-Object -First 1
                    if ($m) { $url = $m.Matches[0].Groups[1].Value }
                    elseif ($p.HasExited) { Get-Content $log, "$tmp\serve.err" -Tail 20; Die 'serve exited' }
                }
                if (-not $url) { Die 'not ready in 1800 s' }
                $name = (Invoke-RestMethod "$url/v1/models").data[0].id
                $body = @{ model = $name; temperature = 0; max_tokens = 64
                    messages = @(@{ role = 'user'; content = 'What is 2 + 3? Reply with the number only.' }) } | ConvertTo-Json -Depth 5
                $answer = (Invoke-RestMethod -Method Post -ContentType 'application/json' -Body $body "$url/v1/chat/completions").choices[0].message.content
                Write-Host "   answer: '$answer'"
                if ($answer -notmatch '5') { Die 'unexpected answer' }
            } finally {
                if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; Get-Process -Name strata -ErrorAction SilentlyContinue | Stop-Process -Force }
            }
        }
        Step "verified: $zipPath"
    } finally { Set-Location $env:SystemRoot; Remove-Tree $tmp }
}

switch ($Command) {
    'fetch' { Invoke-Fetch }
    'build' { Invoke-Build }
    'verify' { Invoke-Verify }
}
