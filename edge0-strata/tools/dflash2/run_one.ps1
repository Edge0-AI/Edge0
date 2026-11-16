# DFlash2 experiment runner on strix. One engine process at a time; reads ids from C:\strata-test\dflash2\p\<name>.*.ids
# usage: run_one.ps1 -Mode gen|genf|genfm|tap|tcycle -Name <prompt> [-MaxNew N] [-Cycle 1,2,...]
param([string]$Mode, [string]$Name, [int]$MaxNew = 900, [string]$Cycle = "", [string]$Exe = "strata-hook.exe", [string]$Tag = "", [string]$Extra = "")
$ErrorActionPreference = "Continue"
$W = "C:\strata-test\dflash2"
$D = "C:\Users\Administrator\Documents\Strata"
$OWN = "C:\Users\Administrator\AppData\Local\Strata\terminal\0.1.46.1"
$env:LOCALAPPDATA = "C:\strata-test\la"
$env:APPDATA = "C:\strata-test\la"
New-Item -ItemType Directory -Force -Path $env:LOCALAPPDATA | Out-Null
New-Item -ItemType Directory -Force -Path "$W\out" | Out-Null
$env:STRATA_ARENA_PIN_GIB = "none"
$env:PATH = "$W\eng\rocm\bin;" + $env:PATH
$mtp = $D + "\mtp\rt"
$base = @("--pack", "$D\packs\iq3_s",
  "--native", "$D\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "--ple-gguf", "$D\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
  "--expert-profile", "$OWN\data\expert-profile.bin",
  "--expert-cache", "auto", "--prefill", "auto", "--max-context", "4096", "--kv", "int8", "--vram-reserve-mib", "700", "--greedy")
$exe = "$W\eng\$Exe"
$out = "$W\out\${Mode}_${Name}${Tag}.txt"
$err = "$W\out\${Mode}_${Name}${Tag}.err"
Remove-Item $out, $err -ErrorAction SilentlyContinue
if ($Mode -eq "gen") {
  $ids = (Get-Content "$W\p\$Name.prompt.ids" -Raw).Trim()
  $a = $base + @("--spec", "4", "--spec-min-p", "0.5", "--mtp", $mtp, "--max-new", "$MaxNew", "--stop-eos", "--tokens", $ids)
} elseif ($Mode -eq "genf" -or $Mode -eq "genfm") {
  # the production flags on a fixed text: the drafts are judged against the continuation in <name>.out.ids
  $ids = (Get-Content "$W\p\$Name.prompt.ids" -Raw).Trim()
  $n = (Get-Content "$W\p\$Name.out.ids" -Raw).Split(",").Count
  $a = $base + @("--spec", "4", "--spec-min-p", "0.5", "--mtp", $mtp, "--spec-follow", "$W\p\$Name.out.ids", "--max-new", "$n", "--tokens", $ids)
  if ($Mode -eq "genfm") { $a += @("--suffix-draft", "0") }
} elseif ($Mode -eq "tap") {
  $env:STRATA_TAP_FILE = "$W\out\tap_$Name.bin"
  Remove-Item $env:STRATA_TAP_FILE -ErrorAction SilentlyContinue
  $first = (Get-Content "$W\p\$Name.first.ids" -Raw).Trim()
  $n = (Get-Content "$W\p\$Name.rest.ids" -Raw).Split(",").Count
  $a = $base + @("--spec", "8", "--spec-oracle", "$W\p\$Name.rest.ids", "--spec-follow", "$W\p\$Name.rest.ids", "--max-new", "$n", "--tokens", $first)
} elseif ($Mode -eq "tcycle") {
  $env:STRATA_SPEC_T_CYCLE = $Cycle
  $first = (Get-Content "$W\p\$Name.first.ids" -Raw).Trim()
  $a = $base + @("--spec", "8", "--spec-oracle", "$W\p\$Name.rest.ids", "--spec-follow", "$W\p\$Name.rest.ids", "--max-new", "$MaxNew", "--tokens", $first)
} else { "bad mode"; exit 2 }
if ($Extra -ne "") { $a += $Extra.Split("_") }
Set-Location $OWN
$t0 = Get-Date
$p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
$p.WaitForExit()
"EXIT " + $p.ExitCode + " after " + [int]((Get-Date) - $t0).TotalSeconds + " s"
Get-Content $out | Select-String -Pattern "^decode|^speculation|^window|^accepted|^mtp|^suffix|^verify window" | ForEach-Object { $_.Line.Substring(0, [Math]::Min(300, $_.Line.Length)) }
