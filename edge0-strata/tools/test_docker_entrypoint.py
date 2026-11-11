"""docker/entrypoint.sh: environment variables become `strata launch` flags, the key never appears on a command line, an engine
that does not fit the card ends the start with a plain message, a started server is kept up, and SIGTERM runs `launch --stop`.

Runs the real script in a temporary tree with a stub for the `strata` command (POSIX sh needed: skipped on Windows).

    python -m unittest tools.test_docker_entrypoint
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")

STUB = r"""#!/bin/sh
# records every call; answers the few read-only queries the entrypoint makes
echo "$*" >> "$STUB_LOG"
case "$1 $2" in
  "components engine") if [ "${STUB_ENGINE:-ok}" = ok ]; then echo "{\"usable\": true, \"backend\": \"${STUB_BACKEND:-}\"}"; else echo '{"usable": false}'; echo "no engine for this card" >&2; fi; exit 0 ;;
esac
case "$*" in
  *"--list-presets --output json"*) if [ -n "${STUB_PRESET:-}" ]; then echo '{"presets": [{"name": "docker"}]}'; fi; exit 0 ;;
  *"--list-presets"*) if [ -n "${STUB_PRESET:-}" ]; then echo "docker   iq3_xxs   32K   fit to card   ${STUB_TUNED:-not tuned}"; fi; exit 0 ;;
  *"launch --status"*) echo '{"running": [{"name": "docker", "state": "ready"}]}'; exit 0 ;;
  *"--show-key"*) echo "KEY-FROM-PRESET-0123456789"; exit 0 ;;
  *"launch --stop"*) echo "stopped" >> "$STUB_LOG"; exit 0 ;;
  *"launch --no-tui"*) echo "key-env=${STRATA_LAUNCH_KEY:-}" >> "$STUB_LOG"; [ -n "${STUB_FAIL:-}" ] && exit 6; exit 0 ;;
esac
exit 0
"""


@unittest.skipIf(os.name == "nt" or SH is None, "needs POSIX sh")
class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "opt" / "terminal" / "bin").mkdir(parents=True)
        stub = self.tmp / "opt" / "terminal" / "bin" / "strata"
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
        self.log = self.tmp / "calls.log"
        text = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        text = (text.replace("/opt/strata", str(self.tmp / "opt")).replace("/data", str(self.tmp / "data"))
                .replace("/run/strata-phase", str(self.tmp / "phase")).replace("mkdir -p ", "mkdir -p " + str(self.tmp / "run") + " ", 1))
        self.script = self.tmp / "entry.sh"
        self.script.write_text(text, encoding="utf-8")

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("MODEL", "API_KEY", "CONTEXT", "VISION", "TUNE", "PORT", "GPU", "GPUS")}
        env.update(STUB_LOG=str(self.log), STRATA_CPUINFO=str(self.tmp / "cpuinfo"), **extra)
        (self.tmp / "cpuinfo").write_text("flags : fpu avx2 sse4_2\n")
        return env

    def start(self, **extra) -> subprocess.Popen:
        return subprocess.Popen([SH, str(self.script)], env=self.env(**extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_for(self, text: str, timeout: float = 20.0) -> str:
        end = time.time() + timeout
        while time.time() < end:
            if self.log.exists() and text in self.log.read_text():
                return self.log.read_text()
            time.sleep(0.1)
        self.fail(f"{text!r} not seen in {self.log.read_text() if self.log.exists() else '(no log)'}")

    def stop(self, p: subprocess.Popen) -> tuple[int, str, str]:
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=30)
        return p.returncode, out, err

    def launch_line(self, log: str) -> str:
        return next(line for line in log.splitlines() if line.startswith("launch --no-tui"))

    def test_settings_become_flags_and_sigterm_stops_cleanly(self):
        (self.tmp / "models").mkdir()
        (self.tmp / "models" / "m-00001-of-00002.gguf").write_text("x")
        p = self.start(MODEL="IQ3_XXS", CONTEXT="64k", PARALLEL="2", VISION="yes", TUNE="quick", GPUS="0,1", PORT="8123", PRESET="office",
                       API_KEY="a-long-enough-key-123456", MODEL_PATH=str(self.tmp / "models"), STRATA_SET="prefill=65536 STRATA_PIN_BY_NEED=1")
        log = self.wait_for("--show-key")
        line = self.launch_line(log)
        for want in ("--access lan", "--port 8123", "--model IQ3_XXS", "--model-path " + str(self.tmp / "models"), "--context 64k", "--parallel 2",
                     "--vision on", "--gpu 0,1", "--tune quick", "--set prefill=65536", "--set STRATA_PIN_BY_NEED=1",
                     "--save-preset office", "--api-key-env STRATA_LAUNCH_KEY"):
            self.assertIn(want, line)
        self.assertNotIn("a-long-enough-key", line)                      # the key is passed by environment, never on a command line
        self.assertIn("key-env=a-long-enough-key-123456", log)
        code, out, err = self.stop(p)
        self.assertEqual(code, 0, err)
        self.assertIn("launch --stop --preset office", self.log.read_text())
        self.assertIn("stopped", self.log.read_text())

    def test_saved_preset_is_reused_and_a_tuned_one_is_not_tuned_again(self):
        p = self.start(STUB_PRESET="1", STUB_TUNED="tuned 2026-10-17", TUNE="quick", MODEL="Q2_0")
        line = self.launch_line(self.wait_for("--show-key"))
        self.assertIn("--preset docker", line)
        self.assertIn("--model Q2_0", line)
        self.assertNotIn("--tune", line)
        self.stop(p)

    def test_untuned_preset_is_tuned_and_retune_forces_it(self):
        p = self.start(STUB_PRESET="1", TUNE="quick")
        self.assertIn("--tune quick", self.launch_line(self.wait_for("--show-key")))
        self.stop(p)
        self.log.unlink()
        p = self.start(STUB_PRESET="1", STUB_TUNED="tuned x", TUNE="quick", RETUNE="1")
        self.assertIn("--tune quick", self.launch_line(self.wait_for("--show-key")))
        self.stop(p)

    def test_no_settings_asks_for_the_recommended_model_and_warns(self):
        p = self.start()
        log = self.wait_for("--show-key")
        self.assertIn("--save-preset docker", self.launch_line(log))
        self.assertNotIn("--model ", self.launch_line(log))
        code, out, err = self.stop(p)
        self.assertIn("No MODEL was set", err)
        self.assertIn("made one", err)                                   # no API_KEY given: told how to show the generated key

    def test_wrong_engine_for_this_card_ends_the_start_with_a_plain_message(self):
        r = subprocess.run([SH, str(self.script)], env=self.env(STUB_ENGINE="none"), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 3)
        self.assertIn("does not match this computer's graphics card", r.stderr)
        self.assertIn("docs/DOCKER.md", r.stderr)
        self.assertNotIn("launch --no-tui", self.log.read_text())

    def test_failed_launch_ends_the_container_with_its_code(self):
        r = subprocess.run([SH, str(self.script)], env=self.env(STUB_FAIL="1", MODEL="Q2_0"), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 6)
        self.assertIn("stopped with code 6", r.stderr)

    def test_bad_vision_value_is_a_plain_error(self):
        r = subprocess.run([SH, str(self.script)], env=self.env(VISION="maybe"), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)
        self.assertIn("VISION=maybe is not one of", r.stderr)

    def test_short_key_and_old_variables_warn(self):
        p = self.start(API_KEY="short", HOST="0.0.0.0", MODEL="Q2_0")
        self.wait_for("--show-key")
        code, out, err = self.stop(p)
        self.assertIn("API_KEY is short", err)
        self.assertIn("HOST is from the older image", err)

    def test_a_changed_card_drops_the_saved_run_settings_of_the_old_card(self):
        install = self.tmp / "data" / "install"
        install.mkdir(parents=True)
        (install / "strata-q2_0.json").write_text('{"gpu": 3, "args": []}')
        (install / "strata-iq3_s.json").write_text('{"gpu": 1, "args": []}')
        p = self.start(MODEL="Q2_0", GPU="1")
        line = self.launch_line(self.wait_for("--show-key"))
        self.assertIn("--gpu 1", line)
        code, out, err = self.stop(p)
        self.assertIn("making them again", out)
        self.assertFalse((install / "strata-q2_0.json").exists())          # card 3 -> 1: made again from the files
        self.assertTrue((install / "strata-iq3_s.json").exists())          # already card 1: kept

    def test_all_cards_leaves_the_saved_run_settings_alone(self):
        install = self.tmp / "data" / "install"
        install.mkdir(parents=True)
        (install / "strata-q2_0.json").write_text('{"gpu": 3, "args": []}')
        p = self.start(MODEL="Q2_0", GPUS="all")
        self.wait_for("--show-key")
        self.stop(p)
        self.assertTrue((install / "strata-q2_0.json").exists())

    def test_a_volume_of_the_older_image_is_moved_not_downloaded_again(self):
        data = self.tmp / "data"
        (data / "models" / "Qwen3.8-Flash-Next-IQ3_XXS").mkdir(parents=True)
        shard = data / "models" / "Qwen3.8-Flash-Next-IQ3_XXS" / "a-00001-of-00002.gguf"
        shard.write_text("weights")
        (data / "models" / "Qwen3.8-Flash-Next-IQ3_XXS" / "a-00001-of-00002.gguf.done").write_text("x")
        (data / "packs" / "qwen3.8-flash-next-iq3_xxs").mkdir(parents=True)
        (data / "packs" / "qwen3.8-flash-next-iq3_xxs" / "experts.bin").write_text("pack")
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_text("mtp")
        ino = shard.stat().st_ino
        p = self.start(MODEL="IQ3_XXS")
        self.wait_for("--show-key")
        code, out, err = self.stop(p)
        new = data / "Strata-data"
        moved = new / "models" / "Qwen3.8-Flash-Next-IQ3_XXS" / "a-00001-of-00002.gguf"
        self.assertEqual(moved.read_text(), "weights")
        self.assertEqual(moved.stat().st_ino, ino)                          # renamed, not copied
        self.assertTrue((moved.parent / "a-00001-of-00002.gguf.done").exists())
        self.assertTrue((new / "packs" / "qwen3.8-flash-next-iq3_xxs" / "experts.bin").exists())
        self.assertTrue((new / "mtp" / "rt" / "experts.bin").exists())
        self.assertFalse((data / "models").exists())
        self.assertIn("moved them into", out)

    def test_a_file_already_in_the_new_place_is_never_overwritten(self):
        data = self.tmp / "data"
        (data / "models" / "t").mkdir(parents=True)
        (data / "models" / "t" / "f.gguf").write_text("old")
        (data / "Strata-data" / "models" / "t").mkdir(parents=True)
        (data / "Strata-data" / "models" / "t" / "f.gguf").write_text("new")
        p = self.start(MODEL="IQ3_XXS")
        self.wait_for("--show-key")
        self.stop(p)
        self.assertEqual((data / "Strata-data" / "models" / "t" / "f.gguf").read_text(), "new")   # never overwritten
        self.assertEqual((data / "models" / "t" / "f.gguf").read_text(), "old")

    def test_run_settings_that_name_another_images_engine_folder_are_made_again(self):
        install = self.tmp / "data" / "install"
        install.mkdir(parents=True)
        here = self.tmp / "opt" / "components" / "engine" / "0.1.46-cuda12"
        here.mkdir(parents=True)
        (here / "strata-vision").write_text("x")
        gone = self.tmp / "opt" / "components" / "engine" / "0.1.46-cuda13"
        (install / "strata-a.json").write_text('{"exe": "component:strata-engine", "vision": {"exe": "%s/strata-vision"}}' % gone)
        (install / "strata-b.json").write_text('{"exe": "component:strata-engine", "vision": {"exe": "%s/strata-vision"}}' % here)
        (install / "strata-c.json").write_text('{"exe": "component:strata-engine", "args": []}')
        (install / "strata-bad.json").write_text("not json")
        p = self.start(MODEL="IQ3_XXS")
        self.wait_for("--show-key")
        code, out, err = self.stop(p)
        self.assertFalse((install / "strata-a.json").exists())
        self.assertTrue((install / "strata-b.json").exists())
        self.assertTrue((install / "strata-c.json").exists())
        self.assertTrue((install / "strata-bad.json").exists())
        self.assertIn("strata-a.json", out)
        self.assertIn("does not have", out)

    def test_nvidia_card_without_gpu_access_ends_before_any_work(self):
        dev = self.tmp / "dev"
        dev.mkdir()
        r = subprocess.run([SH, str(self.script)], env=self.env(STUB_BACKEND="cuda", STRATA_DEV_DIR=str(dev), MODEL="Q2_0"),
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 3)
        self.assertIn("--gpus all", r.stderr)
        self.assertNotIn("launch --no-tui", self.log.read_text())
        (dev / "nvidia0").write_text("")                                  # with the device files the start goes on
        p = self.start(STUB_BACKEND="cuda", STRATA_DEV_DIR=str(dev), MODEL="Q2_0")
        self.wait_for("launch --no-tui")
        self.stop(p)

    def test_model_path_that_is_not_there_or_empty_ends_the_start_in_plain_words(self):
        missing = self.tmp / "nothing-mounted"
        r = subprocess.run([SH, str(self.script)], env=self.env(MODEL="IQ3_XXS", MODEL_PATH=str(missing)), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)
        self.assertIn("is not in the container", r.stderr)
        self.assertIn("-v /srv/gguf/IQ3_XXS:", r.stderr)
        empty = self.tmp / "empty"
        (empty / "IQ3_XXS").mkdir(parents=True)
        r = subprocess.run([SH, str(self.script)], env=self.env(MODEL="IQ3_XXS", MODEL_PATH=str(empty)), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)
        self.assertIn("holds no .gguf files", r.stderr)
        (empty / "IQ3_XXS" / "a-00001-of-00002.gguf").write_text("x")
        r = subprocess.run([SH, str(self.script)], env=self.env(MODEL="IQ3_XXS", MODEL_PATH=str(empty)), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Set MODEL_PATH=" + str(empty / "IQ3_XXS"), r.stderr)
        self.assertNotIn("launch --no-tui", self.log.read_text() if self.log.exists() else "")

    def test_own_files_without_a_model_name_do_not_say_recommended_model(self):
        (self.tmp / "g").mkdir()
        (self.tmp / "g" / "m-00001-of-00001.gguf").write_text("x")
        p = self.start(MODEL_PATH=str(self.tmp / "g"))
        self.wait_for("--show-key")
        code, out, err = self.stop(p)
        self.assertIn("Starting Strata: the model files in " + str(self.tmp / "g"), out)

    def test_other_commands_run_instead_of_the_server(self):
        r = subprocess.run([SH, str(self.script), "strata", "launch", "--plan", "--model", "q2_0"], env=self.env(), capture_output=True,
                           text=True, timeout=30)
        self.assertEqual(r.returncode, 0)
        self.assertIn("launch --plan --model q2_0", self.log.read_text())
        self.assertNotIn("--no-tui", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
