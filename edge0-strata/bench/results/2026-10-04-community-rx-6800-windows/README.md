# Community benchmark: RX 6800 (Windows), 0.1.39 against 0.1.38

Measured on 2026-10-04 by [did-technomancer](https://github.com/did-technomancer).

```text
GPU      AMD Radeon RX 6800 16 GB (gfx1030), also drives two monitors
PCIe     engine probe 28.2 GB/s (pcie_frac 0.55)
CPU      AMD Ryzen 7 5700X3D, 8 cores / 16 threads, AVX2, no AVX-512
RAM      64 GB DDR4-3200 (4x16)
Storage  NVMe SSD (PCIe 3.0), not the system drive
OS       Windows 11 Pro for Workstations 26H2 (build 26300), system-managed page file
Driver   AMD 32.0.21043.10005
ROCm     10.2.0a20260930, hipBLASLt 1.5.0 (no tuning table for gfx1030)
Load     no other workloads, no power limits
0.1.38   release strata-windows-x64-hip.zip (99f3dbd); the same commit built here gave the same numbers
0.1.39   tag v0.1.39 (6f32ec0) built here, tools\hip\build_windows.bat, STRATA_HIP_ARCHS=gfx1030
Model    ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ ed59f92: IQ2_XS, IQ3_XXS, Q2_0
```

tok/s. A cell with several values lists separate runs, median in brackets. "0.1.39s" = 0.1.39 with `STRATA_SH_STREAM=0`.

**Versions, IQ2_XS**

| Engine | Runs | English | Ukrainian | Code | Prompt 7.9K | Prompt 30.3K |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 0.1.38, release zip | 2 | 50.5 / 55.2 (52.9) | 40.8 / 42.3 (41.6) | 57.2 / 55.8 (56.5) | 319 / 326 | 309 / 310 |
| 0.1.38, built here | 1 | 54.2 | 41.7 | 56.6 | 319 | 310 |
| 0.1.39 | 2 | 42.6 / 42.5 (42.6) | 32.9 / 32.5 (32.7) | 49.2 / 50.2 (49.7) | 326 / 326 | 310 / 311 |
| 0.1.39 + `STRATA_SH_STREAM=0` | 4 | 56.4 / 57.0 / 55.5 / 54.2 (56.0) | 43.5 / 44.3 / 43.2 / 43.0 (43.4) | 62.7 / 62.9 / 60.5 / 61.5 (62.1) | 326 | 311 |

Expert cache 5,361-5,486 slots (7.2-7.4 GiB), 1,729-1,761 MiB of VRAM free with everything loaded.

Limitations:

- One PC; most configurations ran once (fewer than three); repeat runs differed by up to 9% (English), 4% (Ukrainian, code)
- The 0.1.39 release zip was not run
- Generated token counts per request were not saved; RAM was not measured
- Not tested: long prompts in `"parallel"` slots, the conversation cache, images, contexts above 128K on 0.1.39, IQ3_S

<details>
<summary>Configuration and method</summary>

```text
strata.exe --pack <data>\packs\iq2_xs
  --native <data>\models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
  --ple-gguf <data>\models\IQ2_XS\Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
  --expert-profile data\expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp <data>\mtp\rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2048
```

- Pack, MTP layer, expert profile and draft vocabularies: the bundled ones, as setup prepared them
- Low-RAM mode off; 7 pool workers; calibration and speed projection off; the CPU image encoder was loaded, not used
- [ab.py](ab.py): each arm restarts the server with one change; the expert cache as filled from the profile
- Requests: `/v1/chat/completions`, `reasoning_effort none`, `temperature 0`, every prompt starts with a random id
  (0 reused tokens)
- Decode: median of 3 English explanations (420-token cap), 2 Ukrainian (520), 1 Python module (700),
  `timings.predicted_per_second`
- Prompt: 7,858 and 30,348 tokens cut from `serve/server.py`, 40-token answer cap; "decode only" arms skip it
- Memory: expert cache slots and free VRAM from the engine's start-up log (a snapshot, not a peak); model loading is
  not in any timing
- [par.py](par.py): N streamed requests sent together, 300-token cap (246-287 generated); first token = the first
  non-empty answer delta, at the client

</details>

<details>
<summary>Versions, IQ3_XXS with `--draft-vocab cyrillic` (decode only, 1 run each)</summary>

| Engine | English | Ukrainian | Code | Expert cache slots |
| --- | ---: | ---: | ---: | ---: |
| 0.1.38, release zip | 42.2 | 45.7 | 42.8 | 3,895 |
| 0.1.39 + `STRATA_SH_STREAM=0` | 49.2 | 55.2 | 52.0 | 4,219 |

</details>

<details>
<summary>0.1.39 decode switches, turned off one at a time (IQ2_XS, decode only, 1 run each)</summary>

| Changed | English | Ukrainian | Code |
| --- | ---: | ---: | ---: |
| nothing | 42.5 | 32.5 | 50.2 |
| `STRATA_SH_STREAM=0` | 56.4 | 43.5 | 62.7 |
| `STRATA_IQ_STAGE_GRID=0` | 43.5 | 34.6 | 51.8 |
| `STRATA_PLE_BATCH=0` | 41.6 | 33.0 | 49.0 |
| `STRATA_OLD_IQ_MMVQ=1` | 43.3 | 32.5 | 47.9 |
| `STRATA_DEC_BATCH=0` | 40.0 | 31.8 | 45.4 |

</details>

<details>
<summary>Settings (1 run each; "—" is the same engine's base)</summary>

| Engine, model | Setting | English | Ukrainian | Code | Drafts accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.1.38, IQ2_XS | — (2 runs) | 50.5 / 55.2 | 40.8 / 42.3 | 57.2 / 55.8 | 63% / 65% |
| 0.1.38, IQ2_XS | `--draft-vocab cyrillic` | 56.6 | 61.4 | 57.3 | 75% |
| 0.1.38, IQ2_XS | `--pool-workers 11` | 50.6 | 40.1 | 57.6 | 62% |
| 0.1.38, IQ2_XS | `--pool-workers 15` | 37.7 | 28.6 | 39.1 | 63% |
| 0.1.39s, IQ2_XS | — | 57.0 | 44.3 | 62.9 | 63% |
| 0.1.39s, IQ2_XS | `--draft-vocab cyrillic` | 58.0 | 63.2 | 63.4 | 73% |
| 0.1.39s, IQ2_XS | `--spec 6` | 51.4 | 40.6 | 59.3 | 55% |
| 0.1.39s, IQ2_XS | `--spec-min-p 0.7` | 56.0 | 45.3 | 63.1 | 70% |
| 0.1.39s, IQ2_XS | `--pcie-frac 0` | 54.7 | 42.4 | 58.3 | 67% |
| 0.1.39s, IQ2_XS | `STRATA_ADAPT_NOWAIT=1` | 55.8 | 43.5 | 62.1 | 64% |
| 0.1.39s, IQ2_XS | `STRATA_RING_BYTES=0` | 58.2 | 43.3 | 61.6 | 63% |
| 0.1.39s, IQ3_XXS | — | 46.7 | 38.5 | 52.5 | 60% |
| 0.1.39s, IQ3_XXS | `--draft-vocab cyrillic` | 49.2 | 55.2 | 52.0 | 68% |
| 0.1.39s, IQ3_XXS | `STRATA_IQ256_GATHER=1` | 46.5 | 38.5 | 50.9 | 60% |

With the Cyrillic vocabulary 160-180 MiB more VRAM stayed free. `STRATA_RING_BYTES=0`: the 30.3K prompt read at
304 tok/s against 311.

</details>

<details>
<summary>Requests at once (0.1.39s, IQ2_XS, 1 round each)</summary>

| | Queue | `"parallel": 2` | `"parallel": 4` |
| --- | --- | --- | --- |
| 4 requests: all done in | 21.9 s | 23.5 s | 22.9 s |
| 4 requests: together | 46.6 tok/s | 46.0 tok/s | 46.6 tok/s |
| 4 requests: first token | 0.7 / 6.6 / 11.8 / 17.7 s | 0.8 / 1.6 / 12.3 / 13.6 s | 0.8 / 3.3 / 3.3 / 3.3 s |
| 2 requests: all done in | 11.9 s | 12.8 s | 13.3 s |
| 2 requests: first token | 0.8 / 7.0 s | 1.1 / 2.0 s | 1.0 / 1.9 s |
| 1 request: English / Ukrainian / code | 57.0 / 44.3 / 62.9 | 54.2 / 42.2 / 59.5 | 49.6 / 40.5 / 55.3 |
| 1 request: prompt 7.9K / 30.3K | 326 / 311 | 324 / 308 | 270 / 306 |
| Expert cache slots | 5,484 | 4,394 | 3,278 |

</details>

<details>
<summary>Model sizes (0.1.39s, default draft vocabulary; IQ2_XS 4 runs, the others 1)</summary>

| | Q2_0 | IQ2_XS | IQ3_XXS |
| --- | --- | --- | --- |
| English / Ukrainian / code | 55.8 / 45.1 / 65.6 | 54.2-57.0 / 43.0-44.3 / 60.5-62.9 | 46.7 / 38.5 / 52.5 |
| Prompt 7.9K / 30.3K | 336 / 321 | 326 / 311 | 324 / 309 |
| Expert cache slots | 5,656 (7.28 GiB) | 5,361-5,484 (7.2-7.4 GiB) | 4,218 (6.86 GiB) |

</details>

<details>
<summary>Context size (0.1.38, IQ2_XS, 2026-10-03)</summary>

| Context | VRAM reserve | Cache slots | Decode (3 runs) | Prompt 2.2K | Prompt 7.9K | Prompt 30.3K |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 8K | 700 MiB | 6,288 | 52.8 / 59.7 / 61.0 | 206 | not measured | not measured |
| 128K | 2,048 MiB | 5,583 | 47.1 / 53.0 / 54.9 | 223 | 327 | 306 |
| 256K | 2,048 MiB | 4,791 | 45.5 / 53.0 / 55.0 | 203 | 325 | 302 |
| 512K (yarn, x2) | 2,048 MiB | 4,420 | 46.9 / 48.5 / 50.2 | 205 | 328 | 304 |

Long prompts (five "secret codes" at 10/30/50/70/90% of the length, 1 run, 0 reused tokens):

| Context | Prompt tokens | Read time | Read tok/s | Decode | Codes found |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K | 118,410 | 375 s | 316 | 36.2 | 5 of 5 |
| 256K | 249,365 | 825 s | 302 | 48.7 | 5 of 5 |
| 512K | 512,194 | 1,968 s | 260 | 33.0 | 5 of 5 |

Reused prefix (128K, a 64,885-token file): first question 198.1 s (328 tok/s), five follow-ups of 25-33 new tokens
0.3-0.5 s each.

No request failed or was cancelled.

</details>

<details>
<summary>Correctness tasks (0.1.39s, reasoning off, 1 run, `strata_eval2.py hard js sql regex`)</summary>

| Set | Task | Q2_0 | IQ2_XS | IQ3_XXS |
| --- | --- | --- | --- | --- |
| Python | topo_sort | ✓ 8.0 s | ✓ 9.2 s | ✓ 9.8 s |
| Python | wildcard_match | ✓ 5.7 s | ✓ 6.6 s | ✓ 7.0 s |
| Python | min_window | ✓ 3.7 s | ✓ 4.0 s | ✓ 5.5 s |
| Python | json_pointer | ✗ 8.8 s | ✓ 18.9 s | ✓ 22.9 s |
| Python | nqueens | ✓ 3.3 s | ✓ 3.6 s | ✓ 5.8 s |
| Python | semver_sort | ✓ 14.6 s | ✓ 7.4 s | ✓ 22.7 s |
| Python | edit_distance | ✓ 3.5 s | ✓ 2.7 s | ✓ 4.4 s |
| Python | gather_limited | ✓ 3.9 s | ✓ 4.9 s | ✓ 6.2 s |
| Python | trie | ✓ 6.1 s | ✓ 6.7 s | ✓ 9.4 s |
| Python | bowling | ✗ 5.8 s | ✓ 5.8 s | ✓ 7.3 s |
| Python | palindrome | ✓ 3.0 s | ✓ 3.2 s | ✓ 4.8 s |
| Python | normpath | ✓ 6.7 s | ✓ 19.4 s | ✓ 11.0 s |
| JavaScript | groupBy | ✓ | ✓ | ✓ |
| JavaScript | deepEqual | ✗ | ✗ | ✓ |
| JavaScript | parseQuery | ✓ | ✓ | ✓ |
| JavaScript | chunk | ✓ | ✓ | ✓ |
| SQL | customers without orders | ✓ | ✓ | ✓ |
| SQL | the two customers with the highest total | ✓ | ✓ | ✓ |
| SQL | order total per customer | ✓ | ✓ | ✓ |
| regex | IPv4 | ✗ | ✗ | ✓ |
| regex | CSS color `#rrggbb` | ✓ | ✓ | ✓ |
| regex | date YYYY-MM-DD | ✗ | ✓ | ✓ |
| **Total** | | **17 of 22** | **20 of 22** | **22 of 22** |

</details>

<details>
<summary>Files</summary>

- [ab.py](ab.py): the arms
- [ab.jsonl](ab.jsonl): every arm's numbers; fields `arm`, `en` / `uk` / `code` (tok/s of each request),
  `read8k` / `read30k` (prompt tok/s), `expert_slots`, `cache_gib`, `vram_free_mib`, `draft_accept`; the last row
  (IQ3_XXS on 0.1.38) also `model` and `vocab`
- [run_all.py](run_all.py): the helpers `ab.py` imports, and the 2026-10-03 context-size runs
- `run_all.json`, `run_all-part1.json`, `run_all.log`: the context-size data
- [par.py](par.py): requests at once
- [par.jsonl](par.jsonl): per request `first_s`, `done_s`, `tok_s`, `out_n`
- [strata_eval.py](strata_eval.py), [strata_eval2.py](strata_eval2.py): the correctness tasks
- [correctness.json](correctness.json): the task results per size

</details>
