# bench_battery.ps1 — house-protocol battery on the live serve (stock 0.1.39)
# 4K/32K/104K/130K prompts, greedy, 256-token cap, unique nonce per run.
$ErrorActionPreference = "Continue"
$h = Invoke-RestMethod -Uri http://127.0.0.1:8080/health -TimeoutSec 5
$model = $h.model
"battery start $(Get-Date -Format 'HH:mm:ss') model=$model" | Out-File -Append -Encoding ascii C:\Strata\bench-battery.log
$base = "Explain what this function does, name its edge cases, and describe how you would test it. "
$sizes = @{ "4K" = 270; "32K" = 2175; "104K" = 6900; "130K" = 8630 }
foreach ($name in @("4K","32K","104K","130K")) {
  $rep = $sizes[$name]
  foreach ($i in 1..3) {
    $prompt = ("Run marker " + [guid]::NewGuid().ToString() + ". ") + ($base * $rep)
    $body = @{ model = $model; messages = @(@{ role = "user"; content = $prompt }); max_tokens = 256; temperature = 0 } | ConvertTo-Json -Depth 5
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
      $r = Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
      $sw.Stop()
      ("{0} run{1}: wall {2:N2}s ptok {3} gen {4}" -f $name, $i, $sw.Elapsed.TotalSeconds, $r.usage.prompt_tokens, $r.usage.completion_tokens) | Out-File -Append -Encoding ascii C:\Strata\bench-battery.log
    } catch {
      ("{0} run{1}: ERROR {2}" -f $name, $i, $_.Exception.Message) | Out-File -Append -Encoding ascii C:\Strata\bench-battery.log
    }
  }
}
"battery done $(Get-Date -Format 'HH:mm:ss')" | Out-File -Append -Encoding ascii C:\Strata\bench-battery.log
