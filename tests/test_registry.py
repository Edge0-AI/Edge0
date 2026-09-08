"""Registry / AutoConfig / AutoModel resolution tests (no GPU needed)."""

from __future__ import annotations

import pytest

import edge0.models  # noqa: F401  (populates MODEL_REGISTRY on import)
from edge0 import AutoConfig
from edge0.registry import MODEL_REGISTRY, TYPE_ALIASES


def test_both_tiers_registered():
    assert set(MODEL_REGISTRY) == {"edge0-35b", "edge0-10b"}


def test_type_aliases_resolve():
    assert TYPE_ALIASES["qwen3_5_moe"] == "edge0-35b"
    assert TYPE_ALIASES["qwen3_5_moe_text"] == "edge0-35b"
    assert TYPE_ALIASES["bailing_hybrid"] == "edge0-10b"
    assert TYPE_ALIASES["bailing_moe_linear"] == "edge0-10b"


@pytest.mark.parametrize("name", ["edge0-35b", "edge0-10b"])
def test_auto_config_defaults(name):
    cfg = AutoConfig.from_pretrained(name=name)
    assert cfg.name == name
    assert cfg.moe_spec.num_experts > 0
    assert cfg.moe_spec.top_k in (4, 8)
    # qwen tier defaults to staged decode; the ling tier mirrors the
    # deployment production profile (STAGED_DECODE=0 — staged decode on
    # ling degrades output, see start_server.sh) with the prerouter
    # still staging each next token's expert set.
    assert cfg.options.staged is (name == "edge0-35b")
    assert cfg.prerouter is not None
    assert cfg.prerouter.weights_file.endswith(".safetensors")
    assert cfg.prerouter_top_k == cfg.moe_spec.top_k


def test_qwen35_profile():
    cfg = AutoConfig.from_pretrained(name="edge0-35b")
    assert cfg.moe_spec.num_experts == 256
    assert cfg.moe_spec.top_k == 4
    assert cfg.moe_spec.intermediate_size == 512
    assert cfg.moe_spec.norm_topk_prob is True
    assert cfg.moe_spec.shared_experts == 1
    assert cfg.moe_spec.quant.bits == 4
    assert cfg.moe_spec.quant.group_size == 64
    assert cfg.moe_spec.layout.value == "separate"
    assert "language_model.model.layers" in cfg.moe_spec.key_template
    assert cfg.options.staged_n == 4
    # demo production profile: on-demand prefill (QWEN_PREFILL_FULL=0),
    # no resident hot pins (QWEN_HOT=0)
    assert cfg.options.prefill_full_layers == 0
    assert cfg.options.full_layer_prefill is False
    assert cfg.options.hot_per_layer == 0
    assert cfg.prerouter.start_layer == 7
    assert cfg.prerouter.hidden == 512
    assert cfg.prerouter.dtype == "fp16"
    assert cfg.prerouter.feature_topk == "executed"
    assert cfg.gen.temperature == 0.7
    assert 248046 in cfg.gen.eos_ids
    assert cfg.port == 8085


def test_ling10b_profile():
    cfg = AutoConfig.from_pretrained(name="edge0-10b")
    assert cfg.moe_spec.num_experts == 128
    assert cfg.moe_spec.top_k == 8
    assert cfg.moe_spec.router.value == "sigmoid_group"
    assert cfg.moe_spec.routed_scaling == 2.5
    assert cfg.moe_spec.n_group == 8
    assert cfg.moe_spec.topk_group == 4
    assert cfg.options.staged_n == 8
    assert cfg.prerouter.start_layer == 7
    assert cfg.prerouter.owners == tuple(range(7, 23))
    assert cfg.prerouter.patch_call is False
    assert cfg.gen.repetition_penalty == 1.1
    assert 156895 in cfg.gen.eos_ids
    assert cfg.port == 8083


def test_override_and_reject():
    cfg = AutoConfig.from_pretrained(name="edge0-10b", prerouter=None,
                                     lora="", prerouter_top_k=0)
    assert cfg.prerouter is None
    assert cfg.lora == ""
    assert cfg.prerouter_top_k == 0
    with pytest.raises(TypeError, match="unknown"):
        AutoConfig.from_pretrained(name="edge0-10b", bogus_field=1)


def test_unknown_name_lists_registry():
    with pytest.raises(KeyError, match="edge0-35b"):
        AutoConfig.from_pretrained(name="edge0-99b")


def test_adapter_module_exposes_config():
    for name, mod in MODEL_REGISTRY.items():
        assert hasattr(mod, "Config")
        assert hasattr(mod, "build_model")
        assert hasattr(mod, "build_engine")
