"""Registry / AutoConfig / AutoModel resolution tests (no GPU needed)."""

from __future__ import annotations

import pytest

import edge0.models  # noqa: F401  (populates MODEL_REGISTRY on import)
from edge0 import AutoConfig
from edge0.registry import MODEL_REGISTRY, TYPE_ALIASES


def test_both_tiers_registered():
    assert set(MODEL_REGISTRY) == {"edge0-35b", "edge0-8b"}


def test_type_aliases_resolve():
    assert TYPE_ALIASES["qwen3_5_moe"] == "edge0-35b"
    assert TYPE_ALIASES["qwen3_5_moe_text"] == "edge0-35b"
    assert TYPE_ALIASES["bailing_hybrid"] == "edge0-8b"
    assert TYPE_ALIASES["bailing_moe_linear"] == "edge0-8b"


@pytest.mark.parametrize("name", ["edge0-35b", "edge0-8b"])
def test_auto_config_defaults(name):
    cfg = AutoConfig.from_pretrained(name=name)
    assert cfg.name == name
    assert cfg.moe_spec.num_experts > 0
    assert cfg.moe_spec.top_k in (4, 8)
    # Both tiers default to staged decode.  Staging is zero-drop only for
    # the layers whose ROUTE is a prerouter prediction (the scoped
    # consumers, li >= start_layer + 1); every layer below start_layer
    # runs the exact path, and the legacy history-filled staging for them
    # is opt-in via history_slots / --history-slots (see docs/prerouter.md).
    assert cfg.options.staged is True
    assert cfg.history_slots is False
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
    # only layers whose route is a prerouter prediction keep staged slots
    assert cfg.history_slots is False
    assert AutoConfig.from_pretrained(
        name="edge0-35b", history_slots=True).history_slots is True
    assert cfg.gen.temperature == 0.7
    assert 248046 in cfg.gen.eos_ids
    assert cfg.port == 8085


def test_ling8b_profile():
    cfg = AutoConfig.from_pretrained(name="edge0-8b")
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
    cfg = AutoConfig.from_pretrained(name="edge0-8b", prerouter=None,
                                     lora="", prerouter_top_k=0)
    assert cfg.prerouter is None
    assert cfg.lora == ""
    assert cfg.prerouter_top_k == 0
    with pytest.raises(TypeError, match="unknown"):
        AutoConfig.from_pretrained(name="edge0-8b", bogus_field=1)


def test_prefill_ondemand_switch_maps_onto_the_tier_preset():
    """`--prefill-ondemand` adjusts the tier's options without replacing the
    preset, and defaults to leaving it alone."""
    base = AutoConfig.from_pretrained(name="edge0-8b")
    assert base.prefill_ondemand is False
    assert base.options.full_layer_prefill is True   # E3b, the tier default

    ondemand = AutoConfig.from_pretrained(name="edge0-8b",
                                          prefill_ondemand=True)
    assert ondemand.options.full_layer_prefill is False
    assert ondemand.options.prefill_full_layers == 0  # everything else kept
    assert ondemand.options.prefill_hot == base.options.prefill_hot
    assert ondemand.options.warm_willneed == base.options.warm_willneed
    assert ondemand.moe_spec == base.moe_spec

    # edge0-35b already prefills on demand (prefill_hot=32): no-op, not a
    # preset rewrite.
    q35 = AutoConfig.from_pretrained(name="edge0-35b", prefill_ondemand=True)
    assert q35.options.full_layer_prefill is False
    assert q35.options.prefill_hot == 32


def test_demo_defaults():
    """The demo entry points run each tier's showcase configuration."""
    from edge0.registry import demo_kwargs

    # edge0-8b demos default to the gate-routed exact path (prerouter off).
    assert demo_kwargs(name="edge0-8b")["prerouter"] is None
    assert demo_kwargs("edge0-8b")["prerouter"] is None  # model_dir form
    # edge0-35b demos keep the tier config untouched.
    assert "prerouter" not in demo_kwargs(name="edge0-35b")
    # an explicit decision always wins.
    sentinel = object()
    assert demo_kwargs(name="edge0-8b",
                       prerouter=sentinel)["prerouter"] is sentinel


def test_unknown_name_lists_registry():
    with pytest.raises(KeyError, match="edge0-35b"):
        AutoConfig.from_pretrained(name="edge0-99b")


def test_adapter_module_exposes_config():
    for name, mod in MODEL_REGISTRY.items():
        assert hasattr(mod, "Config")
        assert hasattr(mod, "build_model")
        assert hasattr(mod, "build_engine")
