from pathlib import Path

import pytest

from asset_common import read_json

pytestmark = pytest.mark.l1


def test_cases_file():
    value = read_json(Path(__file__).parents[3] / 'tests/fixtures/answers/cases.json')
    assert value['format_version'] == 1
    cases = value['cases']
    assert {c['id'] for c in cases} == {'chat-intro', 'plain-mixed', 'long-mixed'}
    assert len(cases) == 3
    for case in cases:
        assert isinstance(case['text'], str) and case['text']
        assert type(case['parse_special']) is bool
        assert type(case['decode_steps']) is int and case['decode_steps'] > 0
    assert value['smoke']['max_tokens'] == 64
