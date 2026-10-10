"""Layer discovery and installation without tensor calculations."""

from types import SimpleNamespace

import pytest

from edge0.moe.spec import MoESpec
from edge0.streaming import install


def _spec():
    return MoESpec(
        num_experts=8, top_k=2, intermediate_size=64,
        block_path="layers.{layer}.mlp",
    )


def test_install_discovers_mixed_layers(monkeypatch):
    twin = object()
    resident = object()
    moe = SimpleNamespace(switch_mlp=resident)
    dense = SimpleNamespace()
    model = SimpleNamespace(layers=[
        SimpleNamespace(mlp=moe), SimpleNamespace(mlp=dense),
    ])
    calls = []

    def make_twin(shards, layer, spec, **kwargs):
        calls.append((shards, layer, spec))
        return twin

    monkeypatch.setattr(install, "StreamingSwitchGLU", make_twin)
    spec = _spec()
    twins = install.install_streaming_experts(model, [], spec)

    assert twins == [twin, None]
    assert calls == [([], 0, spec)]
    assert moe.switch_mlp is twin
    assert moe._edge0_resident_switch is resident
    assert not hasattr(dense, "switch_mlp")


def test_install_rejects_empty_layer_list():
    model = SimpleNamespace(layers=[])
    with pytest.raises(ValueError, match="resolves no layer 0"):
        install.install_streaming_experts(model, [], _spec())
