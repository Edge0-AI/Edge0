"""A launch keeps the log path the person wrote into the model's config (#3656)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_cli.launch import runner, settings as S  # noqa: E402
from strata_control import tune_knobs  # noqa: E402


def derive(base):
    return runner.derive_config(base, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS, log="C:/launch/q2_0.engine.log")


class OwnLog(unittest.TestCase):
    def test_the_setup_log_name_is_replaced_by_the_launch_log(self):
        for name in ("M:/Strata/strata-iq3_xxs.log", "/home/me/strata/strata-q2_0.log", r"M:\Strata\strata-iq3_s.log"):
            self.assertEqual(derive({"exe": "e", "args": [], "log": name})["log"], "C:/launch/q2_0.engine.log", name)
        self.assertEqual(derive({"exe": "e", "args": []})["log"], "C:/launch/q2_0.engine.log")

    def test_a_log_path_the_person_chose_is_kept(self):
        for name in ("M:/logs/history.log", "/var/log/strata/engine.log", r"D:\notes\iq3.txt"):
            self.assertEqual(derive({"exe": "e", "args": [], "log": name})["log"], name, name)
        self.assertEqual(runner.own_log_path({"log": "M:/Strata/strata-iq3_xxs.log"}), None)
        self.assertEqual(runner.own_log_path({"log": ""}), None)
        self.assertEqual(runner.own_log_path({"log": 5}), None)


if __name__ == "__main__":
    unittest.main()
