"""Terminal polish (0.2.1): "already downloaded" instead of a fake download, and `strata monitor` finding a running launch."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from strata_cli.cli import main
from strata_cli.launch.runner import download_text


class AlreadyDownloaded(unittest.TestCase):
    def test_a_model_on_disk_is_not_called_a_download(self):
        dl = {"done_gb": 58.5, "total_gb": 58.5, "percent": 100.0}
        self.assertEqual(download_text(dl, True), "The model files are already on this computer (58.5 GB); checking them")

    def test_a_real_download_keeps_its_numbers(self):
        dl = {"done_gb": 12.0, "total_gb": 58.5, "percent": 20.5}
        self.assertEqual(download_text(dl, False), "Downloading 12.0 of 58.5 GB")
        self.assertEqual(download_text(dl, False, "Downloading the draft layer"), "Downloading the draft layer: 12.0 of 58.5 GB")


class MonitorFindsLaunch(unittest.TestCase):
    def _run(self, argv, info, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch("strata_cli.launch.cmd.launch_info", return_value=info), \
                mock.patch("strata_cli.monitor_commands.Endpoint") as endpoint:
            os.environ.pop("STRATA_URL", None) if not env else None
            endpoint.return_value.snapshot.return_value = {"schema_version": 1}
            out = io.StringIO()
            with redirect_stdout(out):
                main(argv)
        return endpoint

    def test_no_url_uses_the_running_launch_and_its_key(self):
        info = {"running": True, "url": "http://127.0.0.1:8080", "key": "launch-key"}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_API_KEY", None)
            endpoint = self._run(["monitor", "--once", "--quiet"], info)
        args = endpoint.call_args[0]
        self.assertEqual(args[0], "http://127.0.0.1:8080")
        self.assertEqual(args[1], "launch-key")

    def test_an_explicit_url_wins(self):
        info = {"running": True, "url": "http://127.0.0.1:8080", "key": "k"}
        endpoint = self._run(["monitor", "--once", "--quiet", "--url", "http://127.0.0.1:9999"], info)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:9999")

    def test_nothing_running_keeps_the_default_address(self):
        endpoint = self._run(["monitor", "--once", "--quiet"], None)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:8095")


if __name__ == "__main__":
    unittest.main()
