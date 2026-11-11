# Community benchmark on RX 7900 XTX (Windows 11)

Measured on 2026-10-06 by nytasuk-commits. Strata engine 0.1.40, original Flash-Next IQ3_S, 262,144-token context, on Windows with the ready-made AMD engine (not a source build). Tested: three fresh runs each at ~100 and ~38K prompt tokens, plus six needle-recall checks at 32K and 128K. Main limitations: single machine, three runs per configuration, non-streaming requests (no time-to-first-token), and VRAM, PCIe link width and GPU power were not measured.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX, 24 GB (gfx1100). An AMD integrated GPU (gfx1036) is also present; setup listed it as not supported and used the 7900 XTX (GPU 0).
- CPU: AMD Ryzen 9 7950X3D 16-core (AVX-512 detected by setup)
- Installed RAM: 192 GB at 5200 MT/s (Strata status reported 191.2 GiB total)
- Storage: two Samsung 990 PRO 4TB NVMe SSDs. Windows is on one; the model files (`Strata-data`) are on the other.
- PCIe link: link speed and width not recorded. Strata's own startup probe measured 28.3 GB/s host to device (engine log of an earlier start on this machine).
- OS: Windows 11 Pro, version 10.0.26200 (build 26200)
- Driver: AMD display driver 32.0.31041.1004, dated 17/08/2026 (as shown in Task Manager)
- ROCm / HIP: runtime bundled with the Strata engine (`amdhip64_7.dll`); no separate ROCm install
- Strata commit: `1735d6471df29b42c26170efaac1f1446a58640f` (committed Tue 6 Oct 2026 03:12 +0200); engine version 0.1.40 (from `/v1/status`); ready-made AMD engine installed by `START-HERE.bat`, not built from source
- Background workloads: none that I started during the runs. LM Studio was unloaded; no Strata client was connected (`/v1/status` showed 0 requests before the first request). GPU power limit: not measured.

## Model and configuration

- Model: Qwen3.8-Flash-Next, original, GSQ-RCO quantization by ISTA-DASLab, size IQ3_S
- GGUF files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 bytes) and `...-00002-of-00002.gguf` (28,800,138,432 bytes). Both files read end to end with no errors before the runs. Repository revision: not recorded.
- Vision encoder: off. Custom packs or profiles: none. The pack was prepared by setup (`packs\iq3_s`) and the shipped `data/expert-profile.bin` was used.
- Context: 262,144. KV: int8 with `--kv-resident 32768` (KV streaming; setup reported 1.8 GB of the KV in RAM at 128K). Prefill: `auto:32768`. Expert cache: `auto`, which held 7,942 experts (15.06 GiB of VRAM) after startup. Low-RAM mode: not used. Conversation cache: 8192 MiB.
- MTP draft layer: on (`--spec 4 --spec-min-p 0.5`). Reasoning: server default. Sampling: server defaults (no sampling parameters sent). Calibration: not run (NVIDIA only per INSTALL.md). Experimental speed projection: off.
- hipBLASLt tuning table: `gfx1100-hipblaslt-100500.txt`, set through `STRATA_HIPBLASLT_TUNING` in the config.

```text
D:\AI\Strata\run-iq3_s.bat
```

Run config `strata-iq3_s.json` (also attached; no credentials in it):

```json
{
 "exe": "D:\\AI\\Strata\\engine\\strata.exe",
 "args": [
  "--pack", "D:\\AI\\Strata-data\\packs\\iq3_s",
  "--native", "D:\\AI\\Strata-data\\models\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "--ple-gguf", "D:\\AI\\Strata-data\\models\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
  "--expert-profile", "D:\\AI\\Strata\\data\\expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto:32768",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "D:\\AI\\Strata-data\\mtp\\rt",
  "--max-context", "262144",
  "--kv", "int8",
  "--kv-resident", "32768",
  "--conversation-cache-mib", "8192"
 ],
 "cwd": "D:\\AI\\Strata",
 "tokenizer": "D:\\AI\\Strata-data\\packs\\iq3_s\\tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_s",
 "log": "D:\\AI\\Strata\\strata-iq3_s.log",
 "lib_dirs": ["D:\\AI\\Strata\\engine\\rocm\\bin"],
 "port": 8080,
 "backend": "hip",
 "env": {
  "STRATA_HIPBLASLT_TUNING": "D:\\AI\\Strata\\tools\\hip\\gfx1100-hipblaslt-100500.txt"
 },
 "gpu": 0,
 "gpus_asked": true
}
```

`--prefill auto:32768` and `--conversation-cache-mib 8192` were added by hand to the config after setup, following the two tips setup printed. A later `START-HERE.bat --setup` resets both.

## Method

- Requests: non-streaming `POST /v1/chat/completions` with `max_tokens` 256. Each prompt starts with a fresh GUID (`Run id: ...`) so no earlier request shares its prefix.
- Timings: read from `/v1/status` `last_timings` straight after each request (`prompt_n`, `cache_n`, `prompt_ms`, `predicted_n`, `predicted_per_second`, draft counters). Total time is the client-side wall clock around the HTTP call, so it includes request overhead.
- Order: Strata was restarted, then one short warm-up request and one long warm-up request (neither counted), then 3 short runs and 3 long runs.
- Short prompt: "Write a short story about a lighthouse keeper." (about 100 tokens including the run id and template).
- Long prompt: a synthetic log of 1,100 numbered entries (seeded with `System.Random 42`), with one passphrase entry at position 550 and a two-part question at the end (about 38,170 tokens). The full script is in the appendix.
- Reuse and cache state: `cache_n` was 0 on every counted run, so all prompt tokens were read fresh. The expert cache was filled at startup and had served the two warm-up requests.
- Model loading is excluded. Memory: system RAM used, read from `/v1/status` after each run (69.0 to 70.2 GiB). VRAM during the runs: not measured (the Strata Monitor shows no GPU statistics for AMD).
- Nothing else was sent to Strata during the runs.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Short | 98 to 103 | 0 | 256 (cap) | 3 | 84.5 (77.7 to 85.9) | 77.5 (73.8 to 78.7) | not measured |
| Long, 38K | 38,170 to 38,174 | 0 | 131 to 138 | 3 | 1,612.0 (1,608.7 to 1,619.0) | 79.7 (66.7 to 83.2) | not measured |

The short-prompt "prompt tok/s" is dominated by fixed per-request overhead on a ~100-token prompt and is not a prefill speed. The long-prompt figure is.

Total wall-clock time: short median 4.5 s (4.4 to 4.8); long median 25.4 s (25.4 to 25.7).

Per-run data (also in `runs.csv`):

| Config | prompt_n | cache_n | prompt_ms | prompt tok/s | generated | decode tok/s | draft accepted % | total s | RAM used GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 103 | 0 | 1325.0 | 77.7 | 256 | 73.8 | 64.1 | 4.8 | 69.0 |
| short | 98 | 0 | 1160.1 | 84.5 | 256 | 78.7 | 65.9 | 4.4 | 69.2 |
| short | 99 | 0 | 1152.8 | 85.9 | 256 | 77.5 | 63.3 | 4.5 | 69.4 |
| long-38K | 38,174 | 0 | 23,579.2 | 1,619.0 | 138 | 66.7 | 74.1 | 25.7 | 69.5 |
| long-38K | 38,170 | 0 | 23,727.9 | 1,608.7 | 131 | 79.7 | 80.0 | 25.4 | 69.5 |
| long-38K | 38,173 | 0 | 23,680.8 | 1,612.0 | 138 | 83.2 | 78.5 | 25.4 | 70.2 |

The long runs generated 131 to 138 tokens, below the 256 cap. The stop reason was not recorded. Decode figures from samples that short vary more than the short-prompt ones.

### Additional observations (single runs, not part of the benchmark above)

These come from the Strata Monitor tab and the chat, with different prompts, so they are indicative only.

- **A model sharing the GPU shrinks the expert cache.** With a 17.74 GB model also loaded in LM Studio, the same coding prompt in chat gave 1,771 experts cached (3.4 GiB), 49.0 tok/s and a 52.1% VRAM hit rate (564 output tokens). After unloading it and rebooting: 8,243 experts (15.6 GB), 70.5 tok/s and 86.1% (1,053 output tokens). With the 38K log prompt in chat: 44.0 tok/s and 53.8% before, 66.2 tok/s and 80.3% after.
- **Conversation cache.** In Claude Code, starting a new session against a ~40K-token prefix took 28.1 s (0 reused) without `--conversation-cache-mib 8192`, and 1.0 s (40,408 of 40,413 reused) with it.
- **Prefill setting.** `--prefill auto:32768` against `auto` gave about 1,630 against 1,540 prompt tok/s on one run each, so within noise. The long-prompt benchmark above uses `auto:32768`.
- **Context at 128K against 256K.** The expert cache was 8,159 experts at 128K and 7,942 at 256K.

## Correctness and limitations

`tools/needle_bench.py` against the running server (also in `needles.json`):

```text
.\.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

| Length | Depth | Result | Prompt tokens | Time |
| --- | ---: | --- | ---: | ---: |
| 32K | 10% | found | 32,308 | 19 s |
| 32K | 50% | found | 32,308 | 18 s |
| 32K | 90% | found | 32,309 | 19 s |
| 128K | 10% | found | 125,705 | 78 s |
| 128K | 50% | found | 125,703 | 78 s |
| 128K | 90% | found | 125,703 | 61 s |

6 of 6 found. The script does not print reused-token counts. A needle test measures recall on those inputs, not overall model quality.

The 38K log prompt also returned the correct passphrase in chat, and the model completed an agent loop through Claude Code (reading files, listing directories, running a shell command) with prefix reuse above 99% on follow-up tool calls.

Limitations and untested:

- One machine, three runs per configuration, median and range only.
- No streaming measurements, so no time-to-first-token. Total latency is client-side.
- VRAM use, GPU load, temperature and power were not recorded (the Monitor shows no GPU statistics on AMD). PCIe link width and GPU power limit were not recorded.
- Answer quality and agentic reliability were not benchmarked. Images were not tested (off; AMD Windows reads pictures on the CPU only).
- Other sizes (Q2_0, IQ2_XS, IQ3_XXS), the Coder, Swift and Unsloth variants were not tested.
- Windows HIP is described in `AMD_HIP.md` as outside what that document validates. This report shows that `START-HERE.bat` set up and ran the AMD engine on Windows 11 with an RX 7900 XTX, which is a data point on that gap, not a validation of it.

## Appendix: benchmark script

PowerShell, run in a second window while Strata was running. It writes `runs.csv`.

```powershell
$u = "http://127.0.0.1:8080"
$null = New-Item -ItemType Directory -Force D:\AI\bench
$sb = New-Object System.Text.StringBuilder
$rnd = New-Object System.Random 42
1..1100 | ForEach-Object {
  if ($_ -eq 550) { [void]$sb.AppendLine("Entry 550: NOTE the deployment passphrase is copper-lantern-7731.") }
  else { [void]$sb.AppendLine("Entry ${_}: service-$($rnd.Next(1,50)) handled request $($rnd.Next(100000,999999)) in $($rnd.Next(5,900)) ms with status $(@('OK','OK','OK','RETRY','SLOW')[$rnd.Next(0,5)]) on node-$($rnd.Next(1,20)).") }
}
$long = $sb.ToString() + "`nQuestion: What is the deployment passphrase mentioned in the log above, and in one sentence what does the log mostly show?"
$short = "Write a short story about a lighthouse keeper."
function Run($label, $text) {
  $content = "Run id: $([guid]::NewGuid())`n" + $text
  $body = @{ model="x"; max_tokens=256; messages=@(@{role="user"; content=$content}) } | ConvertTo-Json -Depth 5
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $null = Invoke-RestMethod "$u/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
  $sw.Stop()
  $s = Invoke-RestMethod "$u/v1/status"
  $t = $s.last_timings
  [pscustomobject]@{ config=$label; prompt_n=$t.prompt_n; cache_n=$t.cache_n; prompt_ms=$t.prompt_ms; prompt_tps=$t.prompt_per_second; out_n=$t.predicted_n; decode_tps=$t.predicted_per_second; draft_acc_pct=[math]::Round(100*$t.draft_n_accepted/[math]::Max(1,$t.draft_n),1); total_s=[math]::Round($sw.Elapsed.TotalSeconds,1); ram_gib=$s.machine.ram.used_gib }
}
$null = Run "warmup-long" $long
$rows = @()
1..3 | ForEach-Object { $rows += Run "short" $short }
1..3 | ForEach-Object { $rows += Run "long-38K" $long }
$rows | Export-Csv D:\AI\bench\runs.csv -NoTypeInformation
$rows | Format-Table -AutoSize
```
