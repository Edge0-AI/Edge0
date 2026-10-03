"""MoESpec path resolution and key templating tests."""

from __future__ import annotations

import pytest

from edge0.moe.spec import (MoESpec, QuantSpec, RouterKind, WeightLayout)


class _FakeLayer:
    def __init__(self, mlp=None):
        self.mlp = mlp


class _FakeSwitchMLP:
    def __init__(self):
        self.name = "switch"


class _FakeBlock:
    def __init__(self):
        self.switch_mlp = _FakeSwitchMLP()


class _FakeModel:
    def __init__(self, n=4):
        self.language_model = type("LM", (), {})()
        self.language_model.model = type("M", (), {})()
        self.language_model.model.layers = [
            _FakeLayer(_FakeBlock()) for _ in range(n)
        ]
        # ling-style plain module list (no attribute access)
        self.layers = [_FakeLayer(_FakeBlock()) for _ in range(n)]


def _spec(**kw):
    base = dict(
        num_experts=8, top_k=2, intermediate_size=64,
        key_template="language_model.model.layers.{layer}.mlp.switch_mlp",
        block_path="language_model.model.layers.{layer}.mlp.switch_mlp",
    )
    base.update(kw)
    return MoESpec(**base)


def test_keys_templating():
    s = _spec()
    assert s.keys(3, "gate_proj", "weight") == (
        "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight")
    assert s.keys(12, "down_proj", "biases") == (
        "language_model.model.layers.12.mlp.switch_mlp.down_proj.biases")


def test_block_of_digit_segments():
    s = _spec()
    m = _FakeModel()
    block = s.block_of(m, 2)
    assert block.name == "switch"
    assert s.block_of(m, 0) is m.language_model.model.layers[0].mlp.switch_mlp


def test_layer_of_defaults_from_block_path():
    s = _spec()
    m = _FakeModel()
    layer = s.layer_of(m, 1)
    assert layer is m.language_model.model.layers[1]


def test_layer_of_plain_list_path():
    s = _spec(block_path="layers.{layer}.mlp",
              layer_path="layers.{layer}",
              key_template="model.layers.{layer}.mlp.experts")
    m = _FakeModel()
    assert s.block_of(m, 3) is m.layers[3].mlp
    assert s.layer_of(m, 2) is m.layers[2]


def test_bundle_projs_layouts():
    # Stack order matches the math signature directly: up, gate, down
    # (_swiglu(up, gate)) — same convention as the original
    # streaming_experts_qwen35.py ("up_proj", "gate_proj", "down_proj").
    sep = _spec()
    assert sep.bundle_projs == ("up_proj", "gate_proj", "down_proj")
    assert sep.fuse_gu is False
    fused = _spec(layout=WeightLayout.FUSED_GATE_UP)
    assert fused.bundle_projs == ("gate_up_proj", "down_proj")
    assert fused.fuse_gu is True


def test_quant_defaults():
    assert QuantSpec().bits == 4
    assert QuantSpec().group_size == 64
    assert QuantSpec().mode == "affine"


def test_router_kinds():
    assert RouterKind.SOFTMAX_TOPK.value == "softmax_topk"
    assert RouterKind.SIGMOID_GROUP.value == "sigmoid_group"
