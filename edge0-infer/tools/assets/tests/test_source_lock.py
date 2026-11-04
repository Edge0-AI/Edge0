from pathlib import Path

import pytest

from asset_common import read_json, verify_source

pytestmark = pytest.mark.l3


def test_registered_source(environment):
    lock = read_json(Path(__file__).parents[1] / 'edge0_35b/source.lock.json')
    verify_source(environment('MOE_SOURCE_35B'), lock)
