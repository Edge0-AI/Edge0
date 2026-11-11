"""`strata update` installs a newer Control; the Terminal on PATH hands over to it (it used to keep running the old one)."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from strata_cli import handover

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"


def make_control(components: Path, version: str, folder: str | None = None, whole: bool = True, point: bool = True) -> Path:
    folder = folder or version
    d = components / "control" / folder
    (d / "strata_control").mkdir(parents=True, exist_ok=True)
    (d / "VERSION").write_text(version + "\n", encoding="utf-8")
    (d / "strata.py").write_text("print('control', %r)\n" % version, encoding="utf-8")
    if whole:
        py = handover.bundled_python(d)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("stub", encoding="utf-8")
    if point:
        (components / "control" / "active.json").write_text(json.dumps({"version": version, "folder": folder}), encoding="utf-8")
    return d


class Handover(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-handover-")
        self.addCleanup(self.temp.cleanup)
        self.comp = Path(self.temp.name) / "components"
        self.comp.mkdir()
        self.env = {"STRATA_COMPONENTS_DIR": str(self.comp), handover.ENV_CHECKOUT: "1"}
        self.calls, self.warnings = [], []

    def go(self, running="0.1.46", env=None, argv=("status",)):
        def fake(*args, **kwargs):
            self.calls.append((args, kwargs))
            return 7
        env = {**self.env, **(env or {})}
        return handover.maybe_hand_over(list(argv), environ=env, running=running, execv=fake, run=fake, warn=self.warnings.append)

    def test_a_source_checkout_never_hands_over(self):
        make_control(self.comp, "0.1.46.1")
        env = {k: v for k, v in self.env.items() if k != handover.ENV_CHECKOUT}
        original = handover.running_from_checkout
        handover.running_from_checkout = lambda: True
        try:
            result = handover.maybe_hand_over(["status"], environ=env, running="0.1.46", execv=None,
                                              run=lambda *a, **k: self.fail("handed over"), warn=self.warnings.append)
        finally:
            handover.running_from_checkout = original
        self.assertIsNone(result)
        self.assertEqual(self.warnings, [])

    def test_a_newer_control_takes_over_with_its_own_interpreter_and_the_same_arguments(self):
        d = make_control(self.comp, "0.1.46.1")
        result = self.go(argv=("serve", "--port", "1"))
        self.assertEqual(len(self.calls), 1)
        args, kwargs = self.calls[0]
        python, command = (args[0], args[1]) if not WINDOWS else (args[0][0], args[0])
        self.assertEqual(Path(python), handover.bundled_python(d))
        self.assertEqual(command[1:], [str(d / "strata.py"), "serve", "--port", "1"])
        env = kwargs.get("env") or args[2]
        self.assertEqual(env["STRATA_HANDOVER"], "1")
        if WINDOWS:
            self.assertEqual(result, 7, "the exit code of the newer Control is passed through")
        self.assertEqual(self.warnings, [])

    def test_no_loop_when_already_handed_over(self):
        make_control(self.comp, "0.1.46.1")
        self.assertIsNone(self.go(env={"STRATA_HANDOVER": "1"}))
        self.assertIsNone(self.go(env={"STRATA_NO_HANDOVER": "1"}))
        self.assertEqual(self.calls, [])

    def test_a_broken_target_falls_back_with_one_warning(self):
        make_control(self.comp, "0.1.46.1", whole=False)
        self.assertIsNone(self.go())
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("0.1.46.1", self.warnings[0])
        self.assertIn("running 0.1.46 instead", self.warnings[0])

    def test_a_target_that_fails_to_start_falls_back(self):
        make_control(self.comp, "0.1.46.1")

        def boom(*a, **k):
            raise OSError(8, "Exec format error")
        result = handover.maybe_hand_over(["status"], environ=self.env, running="0.1.46", execv=boom, run=boom, warn=self.warnings.append)
        self.assertIsNone(result)
        self.assertEqual(len(self.warnings), 1)

    def test_an_older_or_equal_target_is_ignored(self):
        make_control(self.comp, "0.1.45")
        self.assertIsNone(self.go(running="0.1.46"))
        make_control(self.comp, "0.1.46")
        self.assertIsNone(self.go(running="0.1.46"))
        self.assertIsNone(self.go(running="0.1.46.1"))
        self.assertEqual((self.calls, self.warnings), ([], []))

    def test_the_version_in_the_folder_counts_not_the_pointers_claim(self):
        d = make_control(self.comp, "0.1.45")
        (self.comp / "control" / "active.json").write_text(json.dumps({"version": "9.9.9", "folder": "0.1.45"}), encoding="utf-8")
        self.assertIsNone(self.go(running="0.1.46"))
        self.assertTrue(d.is_dir())

    def test_a_pointer_cannot_leave_the_control_folder(self):
        outside = Path(self.temp.name) / "etc"
        (outside / "strata_control").mkdir(parents=True)
        (outside / "VERSION").write_text("9.0.0", encoding="utf-8")
        (outside / "strata.py").write_text("print('evil')", encoding="utf-8")
        py = handover.bundled_python(outside)
        py.parent.mkdir(parents=True)
        py.write_text("x", encoding="utf-8")
        (self.comp / "control").mkdir(exist_ok=True)
        for folder in ("../../etc", "..", "../etc", "a/b", "/etc", "..\\etc", ""):
            (self.comp / "control" / "active.json").write_text(json.dumps({"version": "9.0.0", "folder": folder}), encoding="utf-8")
            self.assertIsNone(self.go(), folder)
        self.assertEqual(self.calls, [])

    @unittest.skipIf(WINDOWS, "needs a symlink")
    def test_a_folder_that_is_a_link_out_of_the_control_folder_is_refused(self):
        real = make_control(Path(self.temp.name) / "elsewhere", "9.0.0", point=False)
        (self.comp / "control").mkdir(exist_ok=True)
        (self.comp / "control" / "9.0.0").symlink_to(real, target_is_directory=True)
        (self.comp / "control" / "active.json").write_text(json.dumps({"version": "9.0.0", "folder": "9.0.0"}), encoding="utf-8")
        self.assertIsNone(self.go())
        self.assertEqual(self.calls, [])

    def test_no_pointer_or_garbage_means_nothing_happens(self):
        self.assertIsNone(self.go())
        (self.comp / "control").mkdir()
        for text in ("{", "[]", "null", json.dumps({"folder": "x"})):
            (self.comp / "control" / "active.json").write_text(text, encoding="utf-8")
            self.assertIsNone(self.go(), text)
        self.assertEqual((self.calls, self.warnings), ([], []))

    @unittest.skipIf(WINDOWS, "a shell stub stands in for the bundled interpreter")
    def test_end_to_end_the_real_entry_point_hands_over_and_does_not_loop(self):
        newer = make_control(self.comp, "99.0.0")
        py = handover.bundled_python(newer)
        py.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable, encoding="utf-8")
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        (newer / "strata.py").write_text("import os,sys\nprint('control 99', os.environ.get('STRATA_HANDOVER'), sys.argv[1:])\nraise SystemExit(3)\n",
                                         encoding="utf-8")
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True,
                              env={**os.environ, "STRATA_COMPONENTS_DIR": str(self.comp), handover.ENV_CHECKOUT: "1"})
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertIn("control 99 1 ['--version']", done.stdout)
        # with the guard set the current Control runs, and says its own version
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True,
                              env={**os.environ, "STRATA_COMPONENTS_DIR": str(self.comp), "STRATA_HANDOVER": "1"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.startswith("strata "), done.stdout)

    def test_the_entry_point_falls_back_when_the_target_is_broken(self):
        make_control(self.comp, "99.0.0", whole=False)
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True,
                              env={**os.environ, "STRATA_COMPONENTS_DIR": str(self.comp), handover.ENV_CHECKOUT: "1"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.startswith("strata "), done.stdout)
        self.assertEqual(len(done.stderr.strip().splitlines()), 1, done.stderr)
        self.assertIn("99.0.0", done.stderr)


if __name__ == "__main__":
    unittest.main()
