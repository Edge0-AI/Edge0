"""The committed answer directory matches the in-use list; mutations are caught."""
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory import ANSWERS, IN_USE, problems


def test_committed_answers_match_inventory():
    assert problems() == []


@pytest.fixture
def copy(tmp_path):
    root = tmp_path / 'answers'
    shutil.copytree(ANSWERS, root)
    assert problems(root) == []
    return root


def test_missing_suite_fails(copy):
    shutil.rmtree(copy / IN_USE[0])
    assert problems(copy) == [f'missing suite: {IN_USE[0]}']


def test_extra_suite_fails(copy):
    shutil.copytree(copy / IN_USE[0], copy / 'tiny-base-cpu-flash-old')
    assert problems(copy) == ['not in the in-use list: tiny-base-cpu-flash-old']


def test_modified_file_fails(copy):
    path = copy / IN_USE[-1] / 'long-mixed/logits.bin'
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(bytes(data))
    found = problems(copy)
    assert len(found) == 1 and 'hash list does not match' in found[0], found
