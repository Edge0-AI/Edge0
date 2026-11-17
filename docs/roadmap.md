# Roadmap

[← Edge0](../README.md)

### Q4 2026

**Platforms & systems**

- ✅ **edge0 unified inference engine open-sourced** (2026-11-04) —
  [`edge0-infer`](../edge0-infer/README.md): one engine and one native API
  across macOS, iOS, Android and Windows.
- **`edge0-8b` support in edge0-infer**.

**Models & algorithms**

Q4 works two fronts: bringing a next-generation architecture into the
framework, and turning latent reasoning into a real latency saving rather
than just an arithmetic one.

- **Next-gen architecture support (Qwen3.8-Flash class)** — run hybrid
  linear attention (GDN + QSA), gated multi-branch residual, and N-gram
  embedding topologies on edge0. These designs suit SSD streaming offload
  naturally: O(1)-state attention keeps long thinking from becoming a
  KV-cache problem, and lookup-only N-gram tables stream on demand. Goal:
  the tier runs on a single device and benchmarks within an acceptable
  gap of the fp16 base.
- **Latent thinking + batched expert pre-prediction** — make latent
  reasoning a *latency* saving, not only an arithmetic one. The core
  engineering problem: move expert routing from per-position to **once
  per block**, so one prediction covers every position and round of a
  block and **expert load volume decouples from the reasoning loop
  count** — plus cross-block prefetch that loads the next block's experts
  inside the current block's compute window. Progress is measured as
  **end-to-end thinking-phase time at matched accuracy** (never tokens/s).
- More model tiers and adapter releases on the existing pipeline.
