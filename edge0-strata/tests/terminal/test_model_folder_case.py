"""strata launch finds a model folder whatever its letter case (#3996).

setup.py writes `models/IQ3_S` (the model name's case kept); the run-config tag is lower case (`iq3_s`). On a case-sensitive disk the
launch lookup missed the downloaded folder and offered the whole download again. The tests make the disk case-sensitive on any system by
answering `Path.is_dir` from the names the folder really lists.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_cli.launch import modelstate  # noqa: E402
from strata_cli.launch.catalog import Catalog  # noqa: E402

_real_is_dir = Path.is_dir


def _case_sensitive_is_dir(self):
    try:
        listed = {entry.name for entry in self.parent.iterdir()}
    except OSError:
        return False
    return self.name in listed and _real_is_dir(self)


class StubCatalog:
    """Just what model_folder and assess use."""
    tables = lambda self: ({}, {"qwen": {"tag": ""}}, None, None)          # noqa: E731
    tag_of = Catalog.tag_of
    model_folder = Catalog.model_folder

    def __init__(self, data: Path):
        self._data = data

    def data_dir(self):
        return self._data

    def rows(self):
        return [mock.Mock(family="qwen", model="IQ3_S", download_gb=83.6, images=False)]

    def installed_config(self, tag):
        return None


class ModelFolderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        patch = mock.patch.object(Path, "is_dir", _case_sensitive_is_dir)
        patch.start()
        self.addCleanup(patch.stop)
        self.cat = StubCatalog(self.data)

    def test_the_folder_setup_wrote_is_found(self):
        folder = self.data / "models" / "IQ3_S"
        folder.mkdir(parents=True)
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "IQ3_S")

    def test_a_lower_case_folder_is_found_too(self):
        (self.data / "models" / "iq3_s").mkdir(parents=True)
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "iq3_s")

    def test_a_folder_that_differs_only_in_case_is_found(self):
        (self.data / "models" / "Iq3_S").mkdir(parents=True)
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "Iq3_S")

    def test_nothing_there_yet_is_the_name_setup_will_write(self):
        (self.data / "models").mkdir(parents=True)
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "IQ3_S")
        self.assertEqual(self.cat.model_folder(self.data / "missing", "qwen", "IQ3_S").name, "IQ3_S")

    def test_a_downloaded_model_in_setups_folder_is_counted_by_the_launch_plan(self):
        folder = self.data / "models" / "IQ3_S"
        folder.mkdir(parents=True)
        (folder / "m-00001-of-00002.gguf").write_bytes(b"x" * 1000)
        state = modelstate.assess(self.cat, "qwen", "IQ3_S", data_dir=self.data)
        self.assertAlmostEqual(state.on_disk_gb, 1000 / 1e9)
        empty = modelstate.assess(self.cat, "qwen", "IQ3_S", data_dir=self.data / "other")
        self.assertEqual(empty.on_disk_gb, 0.0)


if __name__ == "__main__":
    unittest.main()
