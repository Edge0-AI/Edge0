import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    for level in ['l1', 'l2', 'l3']:
        config.addinivalue_line('markers', f'{level}: layered verification')


@pytest.fixture
def environment():
    def get(name):
        value = os.environ.get(name)
        if not value:
            pytest.skip(f'{name} is not configured')
        path = Path(value)
        if not path.exists():
            pytest.skip(f'{name} does not exist: {path}')
        return path
    return get
