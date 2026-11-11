# Native benchmark matrix (v0.1.26)

Prompt speeds are engine tok/s medians; output lists the first-run value and the three-run median. `prompt tokens` is the median measured count. Draft acceptance is the mean of per-run accepted/offered values from engine logs.

## IQ3_XXS, one RTX 3090

| Context | Prompt tokens | Prompt tok/s | Decode first / median tok/s | TTFT median | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| 1K | 1,135 | 869.7 | 93.0 / 93.0 | 1.33 s | 0.738 |
| 4K | 4,200 | 1,772.2 | 91.7 / 89.2 | 2.41 s | 0.763 |
| 32K | 31,602 | 2,160.4 | 90.6 / 90.6 | 14.83 s | 0.739 |
| 128K | 129,913 | 2,173.6 | 79.3 / 79.3 | 60.53 s | 0.723 |
| 262K | 259,589 | 1,725.6 | 63.0 / 67.3 | 152.10 s | 0.697 |

## 32K model and GPU comparison

| Model | GPUs / split | Prompt tokens | Prompt tok/s median | Decode first / median tok/s | TTFT median | Draft acceptance |
|---|---|---:|---:|---:|---:|---:|
| Coder IQ1_M | 1x 3090 | 31,601 | 2,502.4 | 52.7 / 66.8 | 12.80 s | 0.760 |
| Coder IQ1_M | 2x 3090, auto K=19 | 31,601 | 2,704.3 | 88.5 / 98.6 | 11.85 s | 0.908 |
| IQ3_XXS | 2x 3090, auto K=24 | 31,603 | 1,158.9 | 83.7 / 96.7 | 26.90 s | 0.690 |
| IQ3_S | 2x 3090, auto K=24 | 31,604 | 749.3 | 83.1 / 106.3 | 42.27 s | 0.871 |

## Coder IQ1_M fixed layer split

One initial 32K run per K, then three-run repeats for K=22–30. Prompts are independent and adaptive-cache behavior varies; compare the medians cautiously.

| Split | Successful runs | Prompt tok/s median | Decode tok/s median | Notes |
|---|---:|---:|---:|---|
| Auto (K=19) | 3 | 2,704.3 | 98.6 | Layers 0–18 / 19–47 |
| K=18 | 1 | 2,324.6 | 83.1 | Repeat startup OOM while uploading a dense weight |
| K=22 | 3 | 2,951.6 | 103.3 | Layers 0–21 / 22–47 |
| K=24 | 3 | 3,123.2 | 100.2 | Layers 0–23 / 24–47 |
| K=26 | 3 | 3,205.2 | 93.5 | Layers 0–25 / 26–47 |
| K=28 | 3 | 3,034.9 | 96.6 | Layers 0–27 / 28–47 |
| K=30 | 3 | 2,920.6 | 96.1 | Layers 0–29 / 30–47 |

## Correctness

| Model / setup | Check | Result |
|---|---|---|
| IQ3_XXS, 1x 3090 | Needles 8K / 16K / 28K | 3/3 found |
| IQ3_XXS, 1x 3090 | 16K two-turn chat | 15,892 / 15,915 prompt tokens reused on turn two |
| IQ3_XXS, 2x 3090 auto K=24 | Needles 8K / 16K / 28K | 3/3 found |
| IQ3_XXS, 2x 3090 auto K=24 | 16K two-turn chat | 15,628 / 15,672 prompt tokens reused on turn two |
| IQ3_S, 2x 3090 auto K=24 | Needles 8K / 16K / 28K | 3/3 found |
| IQ3_S, 2x 3090 auto K=24 | 16K two-turn chat | 15,632 / 15,676 prompt tokens reused on turn two |
| Coder, 2x 3090 | Needles / two-turn | Not run |
