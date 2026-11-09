from pathlib import Path
import subprocess
import sys

import pytest

from asset_common import read_json, snapshot

pytestmark = pytest.mark.l2


def test_tiny_determinism_and_refusal(environment):
    ref = environment('MOE_TINY')
    for name in ['tiny-a', 'tiny-b']:
        if not (ref / name).is_dir():
            pytest.skip(f'missing generated fixture: {name}')
    first, second = snapshot(ref / 'tiny-a'), snapshot(ref / 'tiny-b')
    expected = read_json(Path(__file__).parents[1] / 'edge0_35b/tiny.lock.json')
    assert first == second == expected['files']
    result = subprocess.run([sys.executable, '-I', str(Path(__file__).parents[1] / 'edge0_35b/make_tiny.py'), '--source', str(environment('MOE_SOURCE_35B')), '--out', str(ref / 'tiny-a')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'output already exists' in result.stderr
    assert snapshot(ref / 'tiny-a') == first
