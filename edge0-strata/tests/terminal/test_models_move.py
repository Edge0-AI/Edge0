"""Moving the models to another folder (strata models move): same drive, another drive, stop, a failed check, disk full, resume."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import models_move as mm


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MoveCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-move-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.old = self.base / "old"
        self.new = self.base / "new"
        self.files = {
            "models/gyro-s/a.gguf": b"A" * 300_000,
            "models/gyro-s/b.gguf": b"B" * 200_000,
            "packs/gyro-s/dense.bin": b"D" * 100_000,
            "models/coder/c.gguf": b"C" * 150_000,
            "mtp/rt/experts.bin": b"M" * 50_000,
        }
        for rel, data in self.files.items():
            f = self.old / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        # a finished download carries the pinned checksum in its mark
        (self.old / "models/gyro-s/a.gguf.done").write_text(f"done sha256 {sha(self.files['models/gyro-s/a.gguf'])}\n")
        (self.old / "notes.txt").write_text("not a model file")        # Strata moves its own folders only
        self.install = self.base / "install"
        self.install.mkdir()
        self.config = self.install / "strata-gyro-s.json"
        self.config.write_text(json.dumps({"args": ["--native", str(self.old / "models/gyro-s/a.gguf"), "--pack", str(self.old / "packs/gyro-s")],
                                           "mtp": str(self.old / "mtp/rt")}))
        self.total = sum(len(v) for v in self.files.values())

    def tree(self, root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != mm.STATE_FILE}

    def copy_mode(self):
        """Another drive: the files are copied, whatever this machine's volumes are."""
        return mock.patch.object(mm, "same_drive", return_value=False)

    def break_after(self, n):
        """A copy that works n times, then fails: a move cut off part way."""
        real, calls = mm._copy_verified, [0]

        def flaky(*args, **kw):
            calls[0] += 1
            if calls[0] > n:
                raise mm.MoveError("io", "cut off")
            return real(*args, **kw)
        return mock.patch.object(mm, "_copy_verified", side_effect=flaky)

    def cut_off(self, n=1):
        with self.copy_mode(), self.break_after(n):
            with self.assertRaises(mm.MoveError):
                self.run_move()

    def run_move(self, **kw):
        events = []
        summary = mm.move(self.old, self.new, on_progress=events.append, configs=[self.config], **kw)
        return summary, events


class PlanTests(MoveCase):
    def test_plan_numbers_and_choices(self):
        with self.copy_mode():
            p = mm.plan(self.old, self.new, free=lambda _p: 10**9)
        self.assertTrue(p["ok"])
        self.assertGreaterEqual(p["models_bytes"], self.total)
        self.assertGreater(p["needed_bytes"], p["models_bytes"])                    # the 5 % margin
        self.assertTrue(p["choices"]["move"]["possible"])
        self.assertEqual({m["name"] for m in p["models"]}, {"gyro-s", "coder", "mtp"})

    def test_not_enough_room_switches_move_off(self):
        with self.copy_mode():
            p = mm.plan(self.old, self.new, free=lambda _p: 1000)
        self.assertTrue(p["ok"])
        self.assertFalse(p["choices"]["move"]["possible"])
        self.assertIn("free", p["choices"]["move"]["reason"])
        self.assertTrue(p["choices"]["keep"]["possible"] and p["choices"]["empty"]["possible"])

    def test_same_drive_needs_no_room(self):
        p = mm.plan(self.old, self.new, free=lambda _p: 0)
        self.assertTrue(p["same_drive"])
        self.assertTrue(p["choices"]["move"]["possible"])

    def test_inside_and_same_are_refused(self):
        self.assertEqual(mm.plan(self.old, self.old / "sub")["issue"]["code"], "inside")
        self.assertEqual(mm.plan(self.old / "sub", self.old)["issue"]["code"], "inside")
        self.assertEqual(mm.plan(self.old, self.old)["issue"]["code"], "same")

    def test_a_folder_that_cannot_be_written_is_refused(self):
        with mock.patch.object(mm, "_write_test", return_value="EPERM: operation not permitted, mkdir 'x'"):
            p = mm.plan(self.old, self.new)
        self.assertEqual(p["issue"]["code"], "not_writable")
        self.assertIn("EPERM", p["issue"]["details"])

    def test_nothing_is_changed_by_a_plan(self):
        before = self.tree(self.old)
        mm.plan(self.old, self.new)
        self.assertEqual(self.tree(self.old), before)


class MoveTests(MoveCase):
    def check_whole(self):
        self.assertEqual({k for k in self.tree(self.new) if not k.endswith(".done")}, set(self.files))
        for rel, data in self.files.items():
            self.assertEqual((self.new / rel).read_bytes(), data)
        self.assertEqual(self.tree(self.old), {"notes.txt"})                       # only what is not Strata's own is left
        cfg = json.loads(self.config.read_text())
        self.assertEqual(Path(cfg["args"][1]), self.new / "models/gyro-s/a.gguf")
        self.assertEqual(Path(cfg["mtp"]), self.new / "mtp/rt")
        self.assertFalse((self.new / mm.STATE_FILE).exists())

    def test_same_drive_is_a_rename(self):
        with mock.patch.object(shutil, "copyfileobj", side_effect=AssertionError("copied")), mock.patch.object(mm, "_copy_verified", side_effect=AssertionError("copied")):
            summary, _events = self.run_move()
        self.assertTrue(summary["done"] and summary["same_drive"])
        self.check_whole()

    def test_another_drive_copies_checks_and_removes(self):
        with self.copy_mode():
            summary, events = self.run_move()
        self.assertTrue(summary["done"] and not summary["same_drive"])
        self.assertEqual(summary["files_moved"], len(self.files) + 1)               # + the .done mark
        self.check_whole()
        self.assertTrue(events and events[-1]["done"] <= self.total + 100)
        self.assertEqual(events[0]["event"], "progress")

    def test_stop_keeps_what_moved_and_removes_the_partial_copy(self):
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 6                                                   # a few chunks into the second file or so
        with self.copy_mode():
            summary, _events = self.run_move(cancelled=stop)
        self.assertTrue(summary["stopped"] and not summary["done"])
        self.assertEqual([p for p in self.new.rglob("*.part")], [])
        # every file is in exactly one place, nothing is lost
        for rel, data in self.files.items():
            places = [r for r in (self.old, self.new) if (r / rel).exists()]
            self.assertEqual(len(places), 1, rel)
            self.assertEqual((places[0] / rel).read_bytes(), data)
        self.assertTrue(summary["remaining"])
        # a model that moved whole now points at the new folder; one that did not stays where it is
        cfg = json.loads(self.config.read_text())
        for value in (cfg["args"][1], cfg["args"][3], cfg["mtp"]):
            self.assertTrue(Path(value).exists(), value)

    def test_stop_before_anything_leaves_the_old_folder_as_it_was(self):
        with self.copy_mode():
            summary, _ = self.run_move(cancelled=lambda: True)
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["files_moved"], 0)
        self.assertEqual({k for k in self.tree(self.old) if not k.endswith(".done")}, set(self.files) | {"notes.txt"})
        self.assertFalse((self.new / mm.STATE_FILE).exists())

    def test_a_copy_that_does_not_match_is_never_trusted(self):
        real = mm._sha256_of
        calls = {"n": 0}

        def lies(path, stop, tick=None):
            calls["n"] += 1
            return "0" * 64 if str(path).endswith(".part") and calls["n"] == 1 else real(path, stop, tick)
        with self.copy_mode(), mock.patch.object(mm, "_sha256_of", side_effect=lies):
            with self.assertRaises(mm.MoveError) as caught:
                self.run_move()
        self.assertEqual(caught.exception.code, "verify")
        self.assertEqual(list(self.new.rglob("*.part")), [])
        self.assertEqual({k for k in self.tree(self.old) if not k.endswith(".done")}, set(self.files) | {"notes.txt"})   # the original is untouched

    def test_a_file_that_differs_from_its_pinned_checksum_is_not_moved(self):
        (self.old / "models/gyro-s/a.gguf.done").write_text("done sha256 " + "1" * 64)
        with self.copy_mode():
            with self.assertRaises(mm.MoveError) as caught:
                self.run_move()
        self.assertEqual(caught.exception.code, "verify")
        self.assertTrue((self.old / "models/gyro-s/a.gguf").exists())
        self.assertFalse((self.new / "models/gyro-s/a.gguf").exists())

    def test_disk_full_stops_at_once_and_loses_nothing(self):
        real = mm._copy_verified
        seen = {"n": 0}

        def full(src, dst, size, stop, tick):
            seen["n"] += 1
            if seen["n"] == 3:
                part = dst.with_name(dst.name + ".part")
                part.parent.mkdir(parents=True, exist_ok=True)
                part.write_bytes(b"half")
                try:
                    raise OSError(28, "No space left on device", str(dst))
                except OSError as e:
                    mm._quiet_remove(part)
                    raise mm.MoveError("disk_full", "The drive ran out of space.", str(e)) from e
            return real(src, dst, size, stop, tick)
        with self.copy_mode(), mock.patch.object(mm, "_copy_verified", side_effect=full):
            with self.assertRaises(mm.MoveError) as caught:
                self.run_move()
        self.assertEqual(caught.exception.code, "disk_full")
        self.assertEqual(list(self.new.rglob("*.part")), [])
        for rel, data in self.files.items():
            places = [r for r in (self.old, self.new) if (r / rel).exists()]
            self.assertEqual(len(places), 1, rel)
        self.assertIn("remaining", caught.exception.summary)

    def test_a_real_disk_full_error_is_named(self):
        with mock.patch("builtins.open", side_effect=OSError(28, "No space left on device", "x")):
            with self.assertRaises(mm.MoveError) as caught:
                mm._copy_verified(self.old / "models/coder/c.gguf", self.new / "x.gguf", 1, lambda: False, lambda n: None)
        self.assertEqual(caught.exception.code, "disk_full")

    def test_a_crash_leaves_a_part_and_the_next_run_finishes(self):
        stale = self.new / "models/gyro-s/a.gguf.part"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"cut off")
        # a file that was copied and renamed, but whose original was not removed yet
        done = self.new / "models/coder/c.gguf"
        done.parent.mkdir(parents=True)
        done.write_bytes(self.files["models/coder/c.gguf"])
        with self.copy_mode():
            summary, _ = self.run_move()
        self.assertTrue(summary["done"])
        self.check_whole()

    def test_an_interrupted_move_is_remembered(self):
        self.cut_off(1)
        st = mm.interrupted(self.old)                                               # the old folder is still the data folder: it holds the pointer
        self.assertEqual(st["to"], str(self.new))
        self.assertEqual(st["state"], "failed")
        self.assertEqual(mm.interrupted(self.base / "elsewhere"), None)
        mm._write_state(self.old, state="running", pid=os.getpid() + 7, token=None, **{"from": str(self.old), "to": str(self.new)})
        self.assertEqual(mm.interrupted(self.old)["state"], "running")

    def test_a_move_that_is_running_now_is_not_interrupted(self):
        from strata_control import procs
        mm._write_state(self.old, state="running", pid=os.getpid() + 7, token="t", **{"from": str(self.old), "to": str(self.new)})
        with mock.patch.object(procs, "start_token", return_value="t"):
            self.assertIsNone(mm.interrupted(self.old))
        with mock.patch.object(procs, "start_token", return_value=None):
            self.assertIsNotNone(mm.interrupted(self.old))

    def test_nothing_moved_leaves_nothing_to_ask_about(self):
        self.cut_off(0)
        self.assertIsNone(mm.interrupted(self.old))
        self.assertFalse((self.new / mm.STATE_FILE).exists())

    def test_finish_resumes_the_move(self):
        self.cut_off(1)
        st = mm.interrupted(self.old)
        with self.copy_mode():
            summary = mm.move(Path(st["from"]), Path(st["to"]), configs=[self.config])
        self.assertTrue(summary["done"])
        self.check_whole()
        self.assertIsNone(mm.interrupted(self.old))
        self.assertFalse((self.old / mm.STATE_FILE).exists() or (self.new / mm.STATE_FILE).exists())

    def test_leave_keeps_the_old_folder_complete(self):
        self.cut_off(1)
        (self.new / "models/stray.gguf.part").write_bytes(b"x")                      # not listed by the move, but a .part of ours is
        (self.new / "models/gyro-s").mkdir(parents=True, exist_ok=True)
        (self.new / "models/gyro-s/a.gguf.part").write_bytes(b"cut")
        theirs = self.new / "models/own/mine.gguf"                                    # a file of the person's, not part of the move
        theirs.parent.mkdir(parents=True)
        theirs.write_bytes(b"mine")
        st = mm.interrupted(self.old)
        with self.copy_mode():
            result = mm.leave(Path(st["from"]), Path(st["to"]), configs=[self.config])
        self.assertGreaterEqual(result["restored"] + result["removed"], 1)
        for rel, data in self.files.items():
            self.assertEqual((self.old / rel).read_bytes(), data, rel)
            self.assertFalse((self.new / rel).exists(), rel)
        self.assertEqual(list(self.new.rglob("*.part")), [(self.new / "models/stray.gguf.part")])
        self.assertTrue(theirs.exists())
        self.assertIsNone(mm.interrupted(self.old))
        self.assertFalse((self.new / mm.STATE_FILE).exists())
        cfg = json.loads(self.config.read_text())
        self.assertTrue(all(str(self.new) not in str(v) for v in cfg.values()))
        self.assertIn(str(self.old / "models/gyro-s/a.gguf"), cfg["args"])

    def test_leave_after_a_crash_removes_the_duplicate(self):
        self.cut_off(1)
        dup = self.new / "models/coder/c.gguf"                                       # copied and renamed, original not yet removed
        dup.parent.mkdir(parents=True, exist_ok=True)
        dup.write_bytes(self.files["models/coder/c.gguf"])
        mm.leave(self.old, self.new)
        self.assertFalse(dup.exists())
        self.assertTrue((self.old / "models/coder/c.gguf").exists())

    def test_configs_are_rewritten_only_for_files_that_are_there(self):
        self.config.write_text(json.dumps({"args": ["--native", str(self.old / "models/gone.gguf")]}))
        with self.copy_mode():
            self.run_move()
        self.assertEqual(json.loads(self.config.read_text())["args"][1], str(self.old / "models/gone.gguf"))

    @unittest.skipUnless(os.environ.get("STRATA_TEST_OTHER_VOLUME"), "set STRATA_TEST_OTHER_VOLUME to a folder on another volume (e.g. a subst or USB drive)")
    def test_a_real_second_volume(self):
        other = Path(os.environ["STRATA_TEST_OTHER_VOLUME"]) / f"strata-move-{os.getpid()}"
        self.addCleanup(shutil.rmtree, other, True)
        self.new = other
        summary, _ = self.run_move()
        self.assertTrue(summary["done"])
        self.check_whole()


class CommandLineTests(MoveCase):
    def test_json_plan_and_run(self):
        from strata_cli import cli
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.copy_mode():
            code = cli.main(["models", "move", str(self.new), "--from", str(self.old), "--plan", "--output", "json"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out.getvalue())["plan"]["choices"]["move"]["possible"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.copy_mode():
            code = cli.main(["models", "move", str(self.new), "--from", str(self.old), "--install-dir", str(self.install), "--output", "jsonl"])
        self.assertEqual(code, 0, out.getvalue())
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith("{")]
        self.assertEqual(lines[-1]["event"], "result")
        self.assertTrue(lines[-1]["done"])
        self.assertTrue((self.new / "models/coder/c.gguf").exists())

    def run_cli(self, *argv):
        from strata_cli import cli
        from strata_control import discovery
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.copy_mode(), mock.patch.object(discovery, "data_dir", return_value=(self.old, "settings")):
            code = cli.main(["models", "move", *argv])
        return code, out.getvalue()

    def test_status_resume_and_leave(self):
        code, text = self.run_cli("--status", "--output", "json")
        self.assertEqual((code, json.loads(text)["interrupted"]), (0, None))
        self.assertEqual(self.run_cli("--resume", "--output", "json")[0], 5)
        self.cut_off(1)
        code, text = self.run_cli("--status", "--output", "json")
        self.assertEqual(json.loads(text)["interrupted"]["to"], str(self.new))
        self.assertIn("--resume", self.run_cli("--status")[1])
        code, text = self.run_cli("--leave", "--install-dir", str(self.install), "--output", "json")
        self.assertEqual(code, 0, text)
        self.assertTrue((self.old / "models/coder/c.gguf").exists())
        self.cut_off(1)
        code, text = self.run_cli("--resume", "--install-dir", str(self.install), "--output", "jsonl")
        self.assertEqual(code, 0, text)
        self.assertTrue([json.loads(l) for l in text.splitlines() if l.startswith("{")][-1]["done"])
        self.assertTrue((self.new / "models/coder/c.gguf").exists())
        self.assertFalse((self.old / "models/coder/c.gguf").exists())

    def test_launch_says_it_in_one_line(self):
        from strata_cli.launch import cmd
        from strata_control import discovery
        import contextlib
        import io
        self.cut_off(1)
        with mock.patch.object(discovery, "data_dir", return_value=(self.old, "settings")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cmd._interrupted_move_notice()
            self.assertEqual(len(out.getvalue().strip().splitlines()), 1)
            self.assertIn("models move --resume", out.getvalue())
            mm.leave(self.old, self.new)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cmd._interrupted_move_notice()
            self.assertEqual(out.getvalue(), "")

    def test_the_cancel_file_stops_the_run(self):
        from strata_cli import cli
        import contextlib
        import io
        cancel = self.base / "cancel"
        cancel.write_text("x")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.copy_mode():
            code = cli.main(["models", "move", str(self.new), "--from", str(self.old), "--cancel-file", str(cancel), "--output", "json"])
        self.assertEqual(code, 6)
        self.assertTrue(json.loads(out.getvalue().splitlines()[-1])["stopped"])
        self.assertTrue((self.old / "models/coder/c.gguf").exists())


if __name__ == "__main__":
    unittest.main()
