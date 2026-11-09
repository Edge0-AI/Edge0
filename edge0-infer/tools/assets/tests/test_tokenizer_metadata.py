import pytest

from format_gguf import Reader

pytestmark = pytest.mark.l3


def test_tokenizer_metadata(environment):
    tiny = environment('MOE_TINY') / 'tiny-a/base.gguf'
    if not tiny.is_file():
        pytest.skip('tiny fixture is missing')
    base = environment('MOE_ASSETS_35B') / 'base.gguf'
    values = [{k: v for k, v in Reader(path).metadata.items() if k.startswith('tokenizer.')} for path in [tiny, base]]
    assert values[0] == values[1]
