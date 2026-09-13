"""Checkpoint-derived profiles, deliberately separate from trained Edge0 tiers."""
from dataclasses import dataclass, field, replace, fields
from pathlib import Path
import json

from edge0.config import GenerationConfig
from edge0.models.base import ModelConfig
from edge0.moe.spec import MoESpec, QuantSpec
from edge0.streaming.options import LayerOptions


@dataclass
class CustomConfig(ModelConfig):
    architecture: str = ""
    metadata: dict = field(default_factory=dict)
    tokenizer_path: str | None = None
    # Packed device buffers, pending reads, and host staging share this budget.
    weight_cache_bytes: int | None = None
    weight_chunk_bytes: int | None = None
    decoded_cache_bytes: int | None = None
    decode_chunk_bytes: int | None = None

    def __post_init__(self):
        import warnings
        for old, new, default in (
            ('decoded_cache_bytes', 'weight_cache_bytes',
             3*1024**3 if self.architecture == 'qwen4exp' else 512*1024**2),
            ('decode_chunk_bytes', 'weight_chunk_bytes', 8*1024**2)):
            legacy, current = getattr(self, old), getattr(self, new)
            if legacy is not None:
                warnings.warn(f'{old} is deprecated; use {new}', DeprecationWarning, stacklevel=2)
                if current is not None and current != legacy:
                    raise ValueError(f'conflicting {old} and {new}')
                current = legacy
            setattr(self, new, default if current is None else current)
            setattr(self, old, None)


def config(path, tokenizer_path=None, **overrides):
    path = Path(path).expanduser().resolve(strict=True)
    if tokenizer_path is not None:
        tokenizer_path = str(Path(tokenizer_path).expanduser().resolve(strict=True))
    if path.is_file():
        from edge0.checkpoints.gguf import GGUFSource
        with GGUFSource(path) as source:
            m = source.metadata
        arch = m["general.architecture"]
        def required(key):
            full = arch + "." + key
            if full not in m:
                raise ValueError(f"missing GGUF architecture metadata {full!r}")
            return m[full]
        spec = MoESpec(num_experts=required("expert_count"),
            top_k=required("expert_used_count"),
            intermediate_size=required("expert_feed_forward_length"),
            shared_experts=int(m.get(arch + ".expert_shared_feed_forward_length", 0) > 0),
            quant=QuantSpec(bits=0, group_size=0, mode="gguf"))
        eos = m.get("tokenizer.ggml.eos_token_id")
        eos_ids = set(() if eos is None else (eos,))
        # Qwen end-of-generation markers, recognized by llama.cpp as well.
        for i, token in enumerate(m.get("tokenizer.ggml.tokens", [])):
            if token in ("<|im_end|>", "<|endoftext|>"):
                eos_ids.add(i)
        gen = GenerationConfig(temperature=m.get("general.sampling.temp", 1.0),
            top_k=m.get("general.sampling.top_k", 0),
            top_p=m.get("general.sampling.top_p", 1.0),
            eos_ids=tuple(sorted(eos_ids)), first_token_greedy=False)
    else:
        with (path / "config.json").open() as f:
            m = json.load(f)
        tc = m.get("text_config", m)
        arch = tc.get("model_type", m.get("model_type"))
        if arch not in ("qwen3_5_moe", "qwen3_5_moe_text"):
            raise ValueError(f"custom safetensors architecture {arch!r} is not implemented; use a named tier for existing Edge0 checkpoints")
        from edge0.models.edge0_35b import Qwen35Config
        baseline = Qwen35Config._defaults(str(path))
        quant = m.get("quantization", m.get("quantization_config", {}))
        if not all(k in quant for k in ("bits", "group_size")):
            raise ValueError("custom safetensors streaming requires MLX quantization bits and group_size in config.json")
        spec = replace(baseline.moe_spec, num_experts=tc["num_experts"],
            top_k=tc["num_experts_per_tok"], intermediate_size=tc["moe_intermediate_size"],
            norm_topk_prob=tc.get("norm_topk_prob", True),
            quant=QuantSpec(bits=quant["bits"], group_size=quant["group_size"], mode=quant.get("mode", "affine")))
        generation = {}
        if (path / "generation_config.json").exists():
            generation = json.loads((path / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", tc.get("eos_token_id", m.get("eos_token_id", [])))
        gen = GenerationConfig(eos_ids=tuple(eos if isinstance(eos, list) else [eos]),
            temperature=generation.get("temperature", 1.0), top_k=generation.get("top_k", 0),
            top_p=generation.get("top_p", 1.0), first_token_greedy=False)
    if not 1 <= spec.top_k <= spec.num_experts:
        raise ValueError("checkpoint expert_used_count must be between 1 and expert_count")
    cfg = CustomConfig(name=arch, model_dir=str(path), architecture=arch, metadata=m,
        tokenizer_path=tokenizer_path, moe_spec=spec, gen=gen,
        weight_cache_bytes=(3 * 1024**3 if arch == "qwen4exp" else 512 * 1024**2),
        options=LayerOptions(top_k=None, staged=False, full_layer_prefill=False),
        prefill_chunk=32)
    import warnings
    for old, new in (("decoded_cache_bytes", "weight_cache_bytes"),
                     ("decode_chunk_bytes", "weight_chunk_bytes")):
        if old in overrides:
            warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=2)
            value = overrides.pop(old)
            if new in overrides and overrides[new] != value:
                raise ValueError(f"conflicting {old} and {new}")
            overrides[new] = value
    for key, value in overrides.items():
        if key not in {f.name for f in fields(cfg)}:
            raise TypeError(f"unknown CustomConfig override {key!r}")
        cfg = replace(cfg, **{key: value})
    return cfg


def build(path, kind, **kwargs):
    cfg = config(path, **kwargs)
    if kind == "config":
        return cfg
    if Path(cfg.model_dir).is_file():
        from edge0.engine.gguf import GGUFEngine
        engine = GGUFEngine(cfg.model_dir, cfg)
        return engine.model if kind == "model" else engine
    from edge0.engine.qwen import Qwen35Engine, load_installed
    if kind == "model":
        model, _, _, _ = load_installed(cfg.model_dir, cfg)
        return model
    from edge0.backends.mlx.io import load_tokenizer
    tok = load_tokenizer(cfg.tokenizer_path or cfg.model_dir)
    engine = Qwen35Engine(cfg.model_dir, cfg, tokenizer=tok)
    engine.name = cfg.name
    return engine
