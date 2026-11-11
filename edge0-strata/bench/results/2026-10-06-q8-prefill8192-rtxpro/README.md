# Q8 prefill: 6.7x faster with 8,192-token chunks

**Unmodified Strata v0.1.40, Unsloth Qwen3.8-Flash-Next Q8_0, RTX PRO 6000
Blackwell Workstation Edition 96 GB, Ryzen 9 7950X, 128 GB RAM.** GPU power
limit: 400 W. Measured 2026-10-06.

![Q8 prefill and total request time with 1024 versus 8192 token chunks](overview.png)

The measured change is `--prefill 1024` to **`--prefill 8192`**. Both
configurations keep FP16 KV, MTP T4, ngram off, the same weights and pack,
15,472 GPU expert slots (75.25 GiB), a 56 GiB resident expert budget,
`--no-prefill-borrow`, and the same adaptation settings. No engine code was
changed or optimization history imported into this branch.

| Actual input | Starting prefill | 8192 prefill | Prefill speedup | Total request, starting -> 8192 |
|---|---:|---:|---:|---:|
| 32,768 tokens | 52.87 s | **7.90 s** | **6.70x** | 59.71 -> **14.85 s** |
| 131,072 tokens | 213.18 s | **31.98 s** | **6.67x** | 220.25 -> **39.13 s** |

| Actual input | Starting input tok/s | 8192 input tok/s | Starting output tok/s | 8192 output tok/s | Effective output tok/s, starting -> 8192 |
|---|---:|---:|---:|---:|---:|
| 32,768 | 619.8 | **4149.8** | 149.8 | 147.3 | 17.15 -> **68.98** |
| 131,072 | 614.9 | **4098.9** | 145.1 | 143.5 | 4.65 -> **26.18** |

Every request generated **1,024 tokens**. Total request time includes prefill
and excludes loading. Effective throughput is output count divided by engine
prefill plus decode time. Each timed request freshly read its whole prompt,
with zero prompt reuse. Allocated context was 40,960 for the 32K request and
139,264 for the 128K request.

## Why larger chunks helped

The initial 32K native Nsight trace recorded about **1.50 TB of host-to-device
copies during prefill**, at **28.55 GB/s**. A startup PCIe probe measured about
28.9 GB/s. Those bytes alone imply a 51.86-second transfer floor, close to
the 53.06-second prefill marker in that trace. Expert weights outside the
GPU cache were repeatedly streamed across 1,024-token chunks.

Larger chunks let more input tokens use each upload. The faster configuration's
hardware profile is still being collected; its new transfer volume and ceiling
are not claimed here. This result compares two explicit configurations, not
the default auto preset against a new engine implementation.

## Frozen configuration

[config.example.json](../../../configs/rtxpro-q8-prefill8192/config.example.json)
contains the runtime arguments. Edit its asset paths for your machine. Its
default allocated context is 139,264, covering the tested 128K input plus output.
[launch.py](../../../configs/rtxpro-q8-prefill8192/launch.py) accepts `--engine`
and `--context` overrides and passes additional arguments to the native engine:

```sh
python configs/rtxpro-q8-prefill8192/launch.py --dry-run
# For the native stdin/stdout protocol, not an HTTP listener:
sudo prlimit --pid $$ --memlock=unlimited:unlimited
python configs/rtxpro-q8-prefill8192/launch.py --serve
```

The launcher clears inherited Strata experiment variables and sets the recorded
environment. The benchmark overrides EOS to force 1,024 outputs; the runtime
example keeps normal stopping.

The exact engine tag is v0.1.40 at
`1cbcacbcae2953f3be9edc46369f0c875bc6ab8b`.
The native model is the six-shard Unsloth Q8_0 GGUF set. The existing compatibility
pack includes BF16 conversions for small projections. Both arms used the same
pack, profile, draft runtime and frozen request client. The coding task requested
a priority queue module and tests, padded with ordinary maintenance notes to
the recorded input count. Thinking was off and temperature was zero.

## Recorded checks

Both configurations completed at both input lengths and passed the small
`triangular(n)` generated-function check. Output streams first differed at
**token 17 for 32K** and **token 85 for 128K**. The long generated module was not
executed. These are single observations with a fixed output cap.

[results.csv](results.csv) holds chart data;
[measurements.json](measurements.json) holds the four timing records, binary
hashes, launch arguments and committed token IDs. Run `python plot.py` with
matplotlib to regenerate the PNG/SVG. This configuration experiment remains
under investigation.
