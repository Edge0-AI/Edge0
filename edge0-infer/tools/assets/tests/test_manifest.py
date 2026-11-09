import pytest

from edge0_35b.manifest import dimension, expand, load

pytestmark = pytest.mark.l1


def test_35b_manifest():
    p = load()['profiles']['35b']
    base = expand(p, 'base')
    assert len(base) == 703
    assert base['token_embd.weight']['shape'] == (2048, 248320)
    assert base['blk.3.attn_q.weight']['shape'] == (2048, 8192)
    assert base['blk.0.ssm_conv1d.weight']['shape'] == (4, 8192)
    for i in range(40):
        assert (f'blk.{i}.attn_q.weight' in base) == ((i + 1) % 4 == 0)
        assert (f'blk.{i}.ssm_ba.weight' in base) == ((i + 1) % 4 != 0)
    lora = expand(p, 'lora')
    assert len(lora) == 560
    assert lora['blk.0.ssm_ba.weight.lora_a']['shape'] == (2048, 32)
    assert lora['blk.0.ssm_ba.weight.lora_b']['shape'] == (32, 64)


def test_dimension_language_rejects_code():
    assert dimension('2 * width + 4', {'width': 3}) == 10
    with pytest.raises(ValueError):
        dimension('__import__("os")', {})
