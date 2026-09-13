"""Custom GGUF engine using the shared generation and HTTP interfaces."""
from edge0.engine.base import Edge0Engine


class GGUFEngine(Edge0Engine):
    def __init__(self, model_dir, cfg):
        self.name = cfg.architecture
        self.source = self.model = None
        try:
            super().__init__(model_dir, cfg)
        except BaseException:
            self.close()
            raise

    def _build(self):
        from edge0.checkpoints.gguf import GGUFSource
        from edge0.checkpoints.tokenizer import load_tokenizer
        from edge0.streaming.gguf import GGUFWeights
        self.source = GGUFSource(self.cfg.model_dir)
        self._tok = load_tokenizer(self.source.metadata, self.cfg.tokenizer_path)
        from dataclasses import replace
        eos = self.cfg.gen.eos_ids or (self._tok.eos_token_id,)
        self.cfg = replace(self.cfg, gen=replace(self.cfg.gen, eos_ids=eos))
        if self.cfg.architecture == "qwen4exp":
            from edge0.backends.mlx._impl.qwen4exp import Model
        elif self.cfg.architecture == "qwen35moe":
            from edge0.backends.mlx.gguf_qwen35 import Model
        else:
            raise ValueError(f"unsupported GGUF architecture {self.cfg.architecture!r}")
        self.weights = GGUFWeights(self.source, self.cfg.weight_cache_bytes, self.cfg.weight_chunk_bytes)
        self.model = Model(self.weights, self.source.metadata)
        self.cache = self.model.make_cache()

    def encode_chat(self, messages, think=False):
        # No template fallback: silently substituting ChatML can corrupt prompts.
        text = self._tok.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=think)
        return self._tok.encode(text, add_special_tokens=False)

    def _forward(self, ids, intra_stage=True):
        logits = self.model(ids, self.cache, self.pos)
        from edge0.backends import core
        core.eval(logits)
        return logits

    def _step_pre(self, token_id):
        self.model.experts[0].prefetch()

    def _reset_state(self):
        self.cache = self.model.make_cache()
        self.model.history = []
        for expert in self.model.experts:
            expert.reset()

    def stats(self):
        return self.weights.cache.stats()

    def close(self):
        if hasattr(self, "weights"):
            self.weights.close()
        elif self.source is not None:
            self.source.close()
        self.cache = []
        self._last_logits = None
        self.model = None
