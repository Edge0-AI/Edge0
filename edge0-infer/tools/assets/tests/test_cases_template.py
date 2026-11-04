from pathlib import Path

from jinja2 import Environment
import pytest

from asset_common import read_json

pytestmark = pytest.mark.l3


def test_original_template(environment):
    template = (environment('MOE_SOURCE_35B') / 'chat_template.jinja').read_text()
    cases = read_json(Path(__file__).parents[3] / 'tests/fixtures/answers/cases.json')
    case = next(c for c in cases['cases'] if c['id'] == 'chat-intro')
    text = Environment().from_string(template).render(messages=[{'role': 'user', 'content': case['user_message']}], add_generation_prompt=True, enable_thinking=False)
    assert text == case['text'] == cases['smoke']['text']
