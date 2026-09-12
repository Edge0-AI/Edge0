# Experimental GGUF branch status

This branch adds local checkpoint selection, qwen4exp and qwen35moe text adapters,
and direct packed GGUF execution through custom MLX Metal kernels. Selected
experts execute in bounded batches, with packed-buffer caching and float32
activations and accumulation. Dependency pins and named safetensors tiers are
unchanged.

This branch is based directly on upstream main and does not include or depend
on persistent conversation-checkpoint PR #21. Integration is deferred to a
follow-up. The standalone regression run passed 138 tests, with one skipped and
two slow tests deselected. A real-checkpoint standalone smoke generated Hello!
and matched the combined-branch prefill logits. On the earlier combined branch,
both 512-token prefill plus 128-token generation trials completed; the process
lifetime peak physical footprint was 4.65 GiB. See
[validation results](benchmarks/gguf-packed-m5.json) and [usage](gguf.md).

This remains a prototype for discussion. Matched decoded-path performance
comparisons, compiler timing, and broader real-checkpoint numerical validation
remain unfinished. The separate PR message is being reviewed locally; no PR has
been opened.
