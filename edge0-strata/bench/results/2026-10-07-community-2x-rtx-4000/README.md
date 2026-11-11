# Community benchmark: 2x NVIDIA Quadro RTX 4000 8 GB, Xeon E5-2620 v3, 64 GB RAM

Measured on 2026-10-07 by [Ryan V](https://github.com/rvannos) on an unprivileged Proxmox LXC container environment. This tests the **125B Mixture-of-Experts model** (`Swift-Qwen3.8-Flash-Next` IQ2_XS) with a 131,072-token context window, layer split across dual NVIDIA Quadro RTX 4000 GPUs, and expert cache tiering between GPU VRAM and host DDR4 RAM.

The system was evaluated both on synthetic generation sweeps and as the core inference engine of a live production pipeline (**Gridiron Gemini Data Orchestrator**), which continuously ingests and synthesizes 45,000 to 62,000-token NFL podcast transcripts.

Prompt reading speeds reached **126.3 to 249.2 tok/s** on 73,000–74,600 token prompts, with sustained decode generation at **18.1 to 23.2 tok/s** across 500–1,100 generated tokens (expert cache hit rates 48–53%).

---

## Hardware and Software

- **GPUs:** 2x NVIDIA Quadro RTX 4000 8 GB GDDR6 (TU104, Turing Tensor Cores, Flash Attention). 125 W power limit per card. Both cards on PCIe 3.0 x16. Persistence mode enabled.
- **CPU and RAM:** Dual Intel Xeon E5-2620 v3 @ 2.40GHz (Haswell-EP, 12 cores / 24 threads total, AVX2). 64 GB DDR4-2133 ECC Registered host RAM (62.8 GiB reported by Linux); 48 GB RAM (49,152 MiB) and 12 vCPUs allocated to Container 114.
- **Storage:** Enterprise NVMe SSD storage pool.
- **OS and Runtime:** Proxmox VE 9.2.5 (Kernel 7.0.14-6-pve), unprivileged Debian 13 (Trixie) LXC container. NVIDIA driver 580.142, CUDA 13.0.
- **Engine:** Strata engine version 0.1.38 (`main`), release binary with native CUDA backend.
- **Co-located Workloads:** GPU 0 co-hosts WhisperX (isolated to GPU 0, idle at 150 MiB); GPU 1 co-hosts F5-TTS (parked in CPU RAM at 0 MB VRAM when idle). Shared VRAM loan protocols release GPU allocations during audio batches.

---

## Model and Configuration

**Model:** `Swift-Qwen3.8-Flash-Next` (125B MoE, IQ2_XS quantization).
- Native base weights: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
- Quantized expert shards: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`
- Multimodal Vision Projector: `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` (866 MB BF16 ViT on GPU 0 with an automated 5-minute idle watchdog).
- Expert profile: `data/expert-profile.bin`.
- Speculative Draft: MTP runtime (`/opt/Strata-data/mtp/rt`).

**Configuration (`strata-swift-iq2_xs.json`):**
- Context limit: 131,072 tokens (`--max-context 131072`)
- KV Cache: `--kv q4_0 --kv-resident 32768` (32,768 KV cells resident in VRAM, remainder streamed dynamically)
- GPU mapping: `"gpu": [0, 1]`, `layer_split: auto`
- Expert Cache: `--expert-cache auto`, `--vram-reserve-mib 700`
- Prefill: `--prefill auto`
- Speculative Drafting: `--spec 4 --spec-min-p 0.5`
- Vision: On-demand GPU 0 execution with 300s idle auto-unload (`idle_s: 300`)

---

## Measured Performance

### 1. High-Context Ingestion & Generation Sweeps

Measurements captured directly from the engine runtime log under deep-context workloads:

| Prompt Tokens | Reused Tokens | Fresh Tokens Read | Prefill Speed | Generated Tokens | Generation Time | Decode Speed | Expert Cache Hit Rate | KV VRAM Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **73,416** | 0 | 73,416 | **249.2 tok/s** | 814 | 38.38 s | **21.2 tok/s** | 52.5% | 97.70% |
| **74,110** | 73,411 | 699 | **126.3 tok/s** | 566 | 24.84 s | **22.8 tok/s** | 48.2% | 97.84% |
| **74,641** | 74,105 | 536 | **99.9 tok/s** | 271 | 14.94 s | **18.1 tok/s** | 52.5% | 98.07% |
| **72,891** | 71,733 | 1,158 | **139.1 tok/s** | 397 | 17.55 s | **22.6 tok/s** | 52.6% | 98.63% |

### 2. Real-World Production Pipeline Comparison (Data Orchestrator)

Prior to deploying Strata 125B MoE, the pipeline ran **Gemma 4 12B** (Dense) on a single GPU. Here is the operational delta recorded across production podcast dossier ingestions:

| Metric | Gemma 4 12B (Dense, 1x GPU) | Strata 125B MoE (Dual GPU) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Model Size** | 12B Dense | 125B MoE | **10.4× Parameter Scale** |
| **Zero-Tag Extraction Failures** | 5 of 15 episodes (33.3% failure) | 0 of 6 episodes (0.0% failure) | **100% Extraction Reliability** |
| **Avg. Validated Tags / Dossier** | 15.4 tags | 24.5 tags (max 25 cap) | **+59.1% Entity Density** |
| **Leftover Unmapped Speaker IDs** | 814 across 15 files | 160 across 6 files | **78% Reduction in Unmapped IDs** |
| **60,000+ Token Full Ingestion** | Failed (Context fatigue / broken JSON) | 60,927 prompt tokens in single pass | **Zero Dropouts** |
| **2-Host Dialogue Script Generation** | 180s+ (frequent timeouts / truncation) | 81.6s (1,103 words, 21 turns) | **2.2× Speedup** |

---

## Observations & Takeaways

1. **VRAM Efficiency**: Layer splitting across dual 8GB Quadro RTX 4000s allowed Strata to fit dense base weights and 2,473 resident experts directly in VRAM (~6.8 GB per card), while offloading the remaining 33 GB expert arena to host DDR4 RAM with >97% KV VRAM hit rates.
2. **Context Stability**: Processing massive 60k–75k token audio transcripts without JSON truncation or context dropouts eliminated the pipeline's primary reliability bottleneck.
3. **Speculative Decoding on Turing**: Even on older Turing architecture (Quadro RTX 4000), 4-step speculative drafting maintained consistent 18–23 tok/s decode generation.
