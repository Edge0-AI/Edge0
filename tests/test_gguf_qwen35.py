import numpy as np
from gguf_fixture import tiny_qwen35


def test_backbone_mapping_chunking(tmp_path):
    from edge0 import AutoEngine
    e = AutoEngine.from_pretrained(model_path=tiny_qwen35(tmp_path / "qwen35.gguf"),
        decoded_cache_bytes=128 * 1024, decode_chunk_bytes=4096)
    try:
        assert e.cfg.moe_spec.top_k == 2
        assert e.model.text.args.num_experts_per_tok == 2
        ids = [1, 3, 7, 4, 2, 256, 12, 10, 16]
        e.prefill(ids)
        logits = np.array(e.next_logits())
        from pathlib import Path
        reference = np.load(Path(__file__).parent / "fixtures" / "qwen35_logits.npy")
        # Existing MLX GDN uses RMS-based L2 normalization with epsilon inside
        # the root; ggml uses max(norm, epsilon). Tiny activations expose this.
        np.testing.assert_allclose(logits, reference, rtol=2e-3, atol=5e-4)
        e.reset()
        e.prefill(ids, chunk_size=1)
        np.testing.assert_allclose(np.array(e.next_logits()), logits, rtol=1e-4, atol=2e-5)
    finally:
        e.close()
