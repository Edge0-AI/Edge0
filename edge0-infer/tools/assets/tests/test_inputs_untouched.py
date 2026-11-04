import pytest

from asset_common import read_json, snapshot

pytestmark = pytest.mark.l3


def test_inputs_untouched(environment):
    assets = environment('MOE_ASSETS_35B')
    source = environment('MOE_SOURCE_35B')
    provenance = read_json(assets / 'provenance.json')
    assert snapshot(source) == provenance['source_snapshot_before']
