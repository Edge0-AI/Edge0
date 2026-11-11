"""#3895: STRATA_SH_STREAM=1 (the shared expert on its own stream) together with --kv-resident failed the engine on its first request on one
Windows RX 6900 XT ("verify: unspecified launch failure"). AMD runs without the fork by default. Calibrate does not measure the combination, the
tuner never picks it, and a person who set it by hand gets one warning line at the start. Nothing is refused.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate as CAL  # noqa: E402
from serve import server  # noqa: E402
from strata_control import tune_knobs  # noqa: E402
from tools.test_calibrate import BASE, FakeEngine  # noqa: E402


class CalibrateSkipsTheCombination(unittest.TestCase):
    def measure(self, platform, args):
        starts, envs, said = [], [], []

        def start(a, env=None):
            envs.append(dict(env or {}))
            return FakeEngine(a, lambda f, p, w: 50.0, 6, starts)

        with mock.patch.object(CAL.sys, "platform", platform):
            res = CAL.measure(args, [[1, 2, 3]] * 3, start, say=said.append, hip=True)
        return res, envs, said

    def test_windows_with_kv_streaming_does_not_measure_the_fork(self):
        args = CAL.with_arg(BASE, "--kv-resident", "32768")
        res, envs, said = self.measure("win32", args)
        self.assertEqual(res["env"], {})
        self.assertNotIn("sh_stream", res["report"])
        self.assertEqual(res["report"]["sh_stream_skipped"], "windows with --kv-resident")
        self.assertTrue(all(CAL.SH_STREAM not in e for e in envs))
        self.assertTrue(any("#3895" in line and "STRATA_SH_STREAM=1" in line for line in said))

    def test_linux_with_kv_streaming_and_windows_without_it_still_measure_it(self):
        for platform, args in (("linux", CAL.with_arg(BASE, "--kv-resident", "32768")), ("win32", BASE)):
            with self.subTest(platform=platform):
                res, envs, _ = self.measure(platform, args)
                self.assertEqual(set(res["report"]["sh_stream"]), {"0", "1"})
                self.assertNotIn("sh_stream_skipped", res["report"])


class TunerNeverPicksIt(unittest.TestCase):
    def ctx(self, platform, backend="hip", args=("--kv-resident", "32768")):
        context = tune_knobs.Context(backend=backend, arch="gfx1030", gpu_count=1, vram_mib=16384, ram_gib=64, model_gib=40.0, base_args=list(args))
        return context, mock.patch.object(tune_knobs.sys, "platform", platform)

    def test_windows_amd_with_kv_streaming(self):
        ctx, patch = self.ctx("win32")
        with patch:
            problems = tune_knobs.check_setting({}, {"STRATA_SH_STREAM": "1"}, ctx)
        self.assertEqual(len(problems), 1)
        self.assertIn("#3895", problems[0])

    def test_other_cases_stay_allowed(self):
        for platform, backend, args in (("linux", "hip", ("--kv-resident", "32768")), ("win32", "hip", ()), ("win32", "cuda", ("--kv-resident", "32768"))):
            ctx, patch = self.ctx(platform, backend, args)
            with patch, self.subTest(platform=platform, backend=backend, args=args):
                self.assertEqual(tune_knobs.check_setting({}, {"STRATA_SH_STREAM": "1"}, ctx), [])


class StartWarning(unittest.TestCase):
    CFG = {"backend": "hip", "args": ["--kv", "int8", "--kv-resident", "32768"]}

    def test_a_hand_set_fork_with_kv_streaming_on_windows_warns_once_per_start(self):
        with mock.patch.object(server.os, "name", "nt"):
            note = server.sh_stream_warning({"STRATA_SH_STREAM": "1"}, self.CFG)
        self.assertIn("#3895", note)
        self.assertIn("WARNING", note)
        self.assertIn("take STRATA_SH_STREAM out", note)

    def test_no_warning_without_the_combination(self):
        with mock.patch.object(server.os, "name", "nt"):
            self.assertIsNone(server.sh_stream_warning({}, self.CFG))
            self.assertIsNone(server.sh_stream_warning({"STRATA_SH_STREAM": "0"}, self.CFG))
            self.assertIsNone(server.sh_stream_warning({"STRATA_SH_STREAM": "1"}, {"backend": "hip", "args": ["--kv", "int8"]}))
        with mock.patch.object(server.os, "name", "posix"):
            self.assertIsNone(server.sh_stream_warning({"STRATA_SH_STREAM": "1"}, self.CFG))

    def test_child_env_prints_it_and_still_passes_the_variable(self):
        cfg = {**self.CFG, "env": {"STRATA_SH_STREAM": "1"}}
        with mock.patch.object(server.os, "name", "nt"), mock.patch("builtins.print") as shown, \
                mock.patch.object(server, "hip_env_guard", return_value=[]), mock.patch.object(server, "hip_auto_devices", return_value=None):
            env = server.child_env(cfg)
        self.assertEqual(env["STRATA_SH_STREAM"], "1")                       # nothing is taken away
        self.assertTrue(any("#3895" in str(call.args[0]) for call in shown.call_args_list if call.args))


if __name__ == "__main__":
    unittest.main()
