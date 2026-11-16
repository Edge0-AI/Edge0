"""Tests for setup on a Mac (metal/setup_mac.py and setup.py's hand-over): the Mac check, the pack recommended by
unified memory, what a bigger choice risks, the engine arguments with the Metal defaults and the config. Runs on any
PC: sysctl / sw_vers / xcode-select are mocked. No GPU, no network.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup  # noqa: E402
import setup_mac as M  # noqa: E402


def fake_run(values: dict, clt_rc=0):
    def run(cmd, **kw):
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, values.get(cmd[2], ""), "")
        if cmd[0] == "sw_vers":
            return subprocess.CompletedProcess(cmd, 0, values.get("macos", ""), "")
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, clt_rc, "/Library/Developer/CommandLineTools" if clt_rc == 0 else "",
                                               "")
        raise AssertionError(cmd)
    return run


M5 = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.memsize": str(128 * 2 ** 30), "macos": "27.2",
      "hw.perflevel0.physicalcpu": "12"}


class MacCheck(unittest.TestCase):
    def test_reads_chip_memory_macos_and_tools(self):
        with mock.patch.object(M.platform, "machine", return_value="arm64"):
            info = M.mac_info(fake_run(M5))
        self.assertEqual(info["chip"], "Apple M5 Max")
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["macos"], "27.2")
        self.assertTrue(info["arm64"])
        self.assertTrue(info["clt"])
        self.assertEqual(M.mac_problems(info), [])

    def test_an_intel_python_on_apple_silicon_is_still_an_apple_silicon_mac(self):     # #3894
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "machdep.cpu.brand_string": "Apple M1 Max"}))
        self.assertTrue(info["arm64"])
        self.assertTrue(info["rosetta_python"])
        self.assertNotIn("Intel", " ".join(M.mac_problems(info)))

    def test_intel_old_macos_and_missing_tools_are_named(self):
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "macos": "14.6", "machdep.cpu.brand_string": "Intel Core i9"}, clt_rc=2))
        p = " | ".join(M.mac_problems(info))
        self.assertIn("Apple Silicon", p)
        self.assertIn("macOS 15", p)
        self.assertIn("xcode-select --install", p)


class Recommendation(unittest.TestCase):
    def test_by_memory(self):
        # IQ3_S and UD-Q4_K_XL are the recommended sizes; IQ3_S is the default wherever it stays resident
        self.assertEqual(M.RECOMMENDED, ("IQ3_S", "UD-Q4_K_XL"))
        rec, notes = M.recommend_model(128)
        self.assertEqual(rec, "IQ3_S")
        self.assertTrue(any("UD-Q4_K_XL" in n and "higher quality" in n for n in notes))   # named next to it
        rec96, notes96 = M.recommend_model(96)
        self.assertEqual(rec96, "IQ3_S")
        self.assertFalse(notes96)                       # UD-Q4_K_XL does not fit 96 GB resident: no note
        self.assertEqual(M.recommend_model(64)[0], "Q2_0")      # no recommended size fits: the largest that does
        self.assertEqual(M.recommend_model(80)[0], "IQ3_XXS")
        rec, notes = M.recommend_model(32)
        self.assertEqual(rec, "Q2_0")
        self.assertTrue(notes and "stream" in notes[0] and "~14 tok/s" in notes[0])
        self.assertTrue(M.streams("Q2_0", 32) and M.streams("Q2_0", 48) and not M.streams("Q2_0", 64))

    def test_a_bigger_choice_is_kept_with_a_note(self):
        notes = M.model_note("IQ3_XXS", 64)
        self.assertTrue(any("stream from the SSD" in n for n in notes))
        self.assertEqual(M.model_note("Q2_0", 128), [])
        self.assertEqual(M.model_note("IQ3_S", 128), [])

    def test_context(self):
        self.assertEqual(M.recommend_context(64), 32768)
        self.assertEqual(M.recommend_context(128), 65536)


class Config(unittest.TestCase):
    def test_engine_args_carry_the_metal_defaults(self):
        args = M.engine_args(Path("/p/packs/q2_0"), [Path("/m/a-00001.gguf"), Path("/m/a-00002.gguf")],
                             Path("/d/mtp/rt"), 32768)
        self.assertEqual(args[args.index("--spec") + 1], "2")
        self.assertEqual(args[args.index("--native") + 1], str(Path("/m/a-00001.gguf")))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(Path("/m/a-00002.gguf")))
        self.assertEqual(args[args.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertIn("--expert-profile", args)
        self.assertNotIn("--hc-q8", args)                         # Q8_0 HC is the engine's own default on Metal
        st = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True)
        self.assertNotIn("--mtp", st)                             # streaming: one token per step, no draft layer
        self.assertEqual(st[st.index("--suffix-draft") + 1], "0")
        self.assertEqual(st[st.index("--spec") + 1], "2")
        # a big Mac that streams (UD-Q5/Q6 on 128 GB) keeps the draft layer, with a longer window
        big = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 65536, stream=True,
                            ram_gb=128)
        self.assertEqual(big[big.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertEqual(big[big.index("--spec") + 1], "3")
        self.assertEqual(big[big.index("--suffix-draft") + 1], "0")
        small = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True,
                              ram_gb=32)
        self.assertNotIn("--mtp", small)
        self.assertTrue(M.stream_mtp(False, 16) and M.stream_mtp(True, 128) and not M.stream_mtp(True, 64))

    def test_config_is_a_model_config_with_the_metal_backend(self):
        cfg = M.mac_config(Path("/r/build-metal/strata"), ["--pack", "x"], Path("/p"), "qwen3.8-flash-next-q2_0",
                           "Q2_0", 8080)
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["tokenizer"], str(Path("/p") / "tokenizer"))
        self.assertTrue(cfg["exe"] and isinstance(cfg["args"], list))


class Vision(unittest.TestCase):
    def test_the_projector_is_pinned_and_the_encoder_builds_with_metal(self):
        self.assertIn("mmproj-Qwen3.8-Flash-Next-BF16.gguf", M.VISION_SHA256)
        src = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"-DSTRATA_VISION_CUDA=OFF"', src)
        self.assertIn('cfg["args"] += ["--vision"]', src)


class ExpertsFile(unittest.TestCase):
    """The streaming Mac's <model>.experts: written only with room on the disk, else a warning and the GGUF."""

    def run_it(self, free_gb, done=False):
        import tempfile
        sys.path.insert(0, str(ROOT / "metal" / "tools"))
        import stream_experts
        d = Path(tempfile.mkdtemp())
        native = d / "m.gguf"
        native.write_bytes(b"")
        if done:
            (d / "m.gguf.experts").write_bytes(b"")
            (d / "m.gguf.experts.done").write_text("x")
        with mock.patch.object(stream_experts, "size_bytes", return_value=40e9), \
                mock.patch.object(M.S, "free_gb", return_value=free_gb), \
                mock.patch.object(M.S, "run") as run, mock.patch.object(M.S, "warn") as warn, \
                mock.patch.object(M.S, "mark") as mark, mock.patch.object(M.S, "say"), mock.patch.object(M.S, "ok"):
            M.experts_file(native)
        return run, warn, mark

    def test_fits_needs_the_margin(self):
        self.assertFalse(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB - 0.1))
        self.assertTrue(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB))

    def test_written_and_verified_with_room(self):
        run, warn, mark = self.run_it(100)
        self.assertIn("--verify", run.call_args[0][0])
        self.assertTrue(str(run.call_args[0][0][1]).endswith("stream_experts.py"))
        mark.assert_called_once()
        warn.assert_not_called()

    def test_too_little_disk_warns_and_goes_on(self):
        run, warn, mark = self.run_it(45)
        run.assert_not_called()
        mark.assert_not_called()
        self.assertIn("not written", warn.call_args[0][0])

    def test_done_is_not_written_again(self):
        run, warn, mark = self.run_it(100, done=True)
        run.assert_not_called()


class HandOver(unittest.TestCase):
    def test_darwin_hands_the_run_to_setup_mac(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--model", "Q2_0", "--yes"]):
            self.assertEqual(setup.main(), 0)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("metal") / "setup_mac.py")))
        self.assertEqual(cmd[2:], ["--model", "Q2_0", "--yes"])

    def test_other_platforms_never_reach_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin":', src)
        with mock.patch.object(setup, "mac_setup", side_effect=AssertionError("called")), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--inspect", "x"]), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(SystemExit):
                setup.main()

    def test_setup_sh_fetches_a_pinned_python_on_macos_only(self):
        sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('[ "$(uname -s)" = Darwin ]', sh)
        self.assertIn("83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8", sh)


class RepackGuard(unittest.TestCase):
    """No engine beside a repack: the locks, and the resident rule that matches the engine's arena_budget.h."""

    def test_resident_rule_matches_the_engine(self):
        # the resident cap: Metal's working set (3/4), and on a Mac of 64 GB or more the RAM less 30%
        self.assertAlmostEqual(M.gpu_budget_gb(128), 128 * 1.073741824 * 0.70, places=3)    # 96.2 GB, as arena_budget.h
        self.assertAlmostEqual(M.gpu_budget_gb(96), 96 * 1.073741824 * 0.70, places=3)
        self.assertAlmostEqual(M.gpu_budget_gb(32), 32 * 1.073741824 * 0.75, places=3)      # a small Mac: the working set
        self.assertFalse(M.streams("UD-Q4_K_XL", 128))          # 82.6 + 8 GB: resident (measured: nothing compressed)
        self.assertTrue(M.streams("UD-Q5_K_XL", 128))
        self.assertTrue(M.streams("UD-Q4_K_XL", 96))

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_a_live_lock_is_seen_a_stale_one_is_dropped(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(M.repack_running(d), [])
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))          # a live process that is not us
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / "m-metal.gguf.repack.lock").write_text("999999999")                # no such process
            self.assertEqual(M.repack_running(d), [])
            self.assertFalse((d / "m-metal.gguf.repack.lock").exists())             # the stale lock was removed
            (d / M.PREPARE_LOCK).write_text(str(os.getppid()))
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / M.PREPARE_LOCK).write_text(str(os.getpid()))                       # our own setup's lock does not stop us
            self.assertEqual(M.repack_running(d), [])

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_start_refuses_while_a_repack_runs(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gguf, exe = d / "m-metal.gguf", d / "strata"
            gguf.write_bytes(b"x")
            exe.write_bytes(b"x")
            cfg = d / "strata-x.json"
            cfg.write_text(json.dumps({"exe": str(exe), "args": ["--native", str(gguf)], "port": 8080}))
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))
            with mock.patch.object(M.subprocess, "call", side_effect=AssertionError("started")), \
                    mock.patch.object(M.S, "fail", side_effect=SystemExit("refused")) as fail:
                with self.assertRaises(SystemExit):
                    M.start(cfg, None, False)
            self.assertIn("repack is still running", fail.call_args[0][0])

    def test_the_repack_writes_in_bounded_pieces(self):
        src = (ROOT / "metal" / "tools" / "metal_repack.py").read_text(encoding="utf-8")
        self.assertIn("SYNC_EVERY = 1 << 30", src)                  # flushed every GiB
        self.assertIn("nocache(o)", src)                            # F_NOCACHE on the output
        self.assertIn("buffering=0", src)                           # write() is a write
        self.assertNotIn("mmap", src)                               # no dirty mapping of the output


M4 = {"machdep.cpu.brand_string": "Apple M4", "hw.memsize": str(32 * 2 ** 30), "macos": "27.2", "hw.perflevel0.physicalcpu": "4"}


class ControlMode(unittest.TestCase):
    """The app on a Mac (STRATA_CONTROL_INSTALL=1): model files and a run config naming the engine component. No source build, no
    metal/setup_mac.py, no Command Line Tools; the Mac rules are strata_control/mac_fit.py's, the module setup_mac.py uses too."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name).resolve() / "install"
        self.data = Path(self.tmp.name).resolve() / "data"
        self.models = Path(self.tmp.name).resolve() / "own"
        self.home.mkdir()
        self.models.mkdir()
        fam = setup.FAMILIES["qwen"]
        self.files = []
        for i in (1, 2):
            f = self.models / setup.model_file(fam, "Q2_0", i)
            f.write_bytes(b"x")
            f.with_name(f.name + ".done").write_text("whole")
            self.files.append(f)

    def run_setup(self, hw=M4, argv=("--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context", "8192"), clt_rc=0):
        from strata_control import engine_component, mac_fit
        ran = []

        def fake_tool(cmd, **kw):
            ran.append(cmd)
            pack = Path(cmd[cmd.index("--out") + 1])
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            (pack / "native_experts.txt").write_text("x")
            (pack / "tokenizer" / "vocab.json").write_text("{}")
        patches = [
            mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "_CONTROL_HOME", self.home),
            mock.patch.object(setup.sys, "platform", "darwin"),
            mock.patch.object(setup.sys, "argv", ["setup.py", *argv, "--data-dir", str(self.data), "--gguf-dir", str(self.models)]),
            mock.patch.object(mac_fit, "mac_info", lambda run=None, _real=mac_fit.mac_info: _real(fake_run(hw, clt_rc))),
            mock.patch.object(setup, "load_settings", lambda: {}), mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: Path(self.tmp.name) / "llama"),
            mock.patch.object(setup, "whole_shard", lambda s: True), mock.patch.object(setup, "check_shards", lambda s: None),
            mock.patch.object(setup, "run", fake_tool), mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "ok", lambda *a, **k: None),
            mock.patch.object(setup, "step", lambda *a, **k: None),
            mock.patch.object(setup, "warn", lambda *a, **k: self.warnings.append(" ".join(str(x) for x in a))),
            mock.patch.object(mac_fit.platform, "machine", return_value="arm64"),
            mock.patch.object(engine_component, "resolve", side_effect=engine_component.EngineMissing("no package here", None)),
            mock.patch.object(setup, "mac_setup", side_effect=AssertionError("setup_mac.py must not run in Control mode")),
            mock.patch.object(setup.subprocess, "call", side_effect=AssertionError("nothing is started or built")),
        ]
        self.warnings = []
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc = setup.main()
        return rc, ran

    def config(self, tag="q2_0"):
        import json
        return json.loads((self.home / f"strata-{tag}.json").read_text(encoding="utf-8"))

    def test_the_install_writes_a_run_config_for_the_engine_component(self):
        rc, ran = self.run_setup()
        self.assertEqual(rc, 0)
        cfg = self.config()
        self.assertEqual(cfg["exe"], "component:strata-engine")
        self.assertEqual(cfg["backend"], "metal")
        args = cfg["args"]
        # a 32 GB Mac streams Q2_0's experts from the SSD: one token per step, no draft layer, the GGUF mapped in place
        self.assertEqual(args[args.index("--native") + 1], str(self.files[0]))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(self.files[1]))
        self.assertIn("--mmap-experts", args)
        self.assertNotIn("--mtp", args)
        self.assertEqual(args[args.index("--suffix-draft") + 1], "0")
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "1"})
        self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / "q2_0" / "tokenizer"))
        self.assertEqual(len(ran), 1, "only the pack is made (tools/iq_pack.py): no draft layer, no build")
        self.assertTrue(ran[0][1].endswith("iq_pack.py"))
        self.assertFalse((ROOT / "build-metal").exists() and any((ROOT / "build-metal").glob("*.stamp")))

    def test_a_big_mac_keeps_the_pack_in_memory_and_adds_the_draft_layer(self):
        rc, ran = self.run_setup(hw=M5, argv=("--yes", "--no-start", "--model", "IQ3_XXS"))
        self.assertEqual(rc, 0)
        cfg = self.config("iq3_xxs")
        self.assertIn("--mtp", cfg["args"])
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "0"})
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")      # the default context by memory

    def test_no_command_line_tools_needed_and_the_unsloth_files_are_not_offered(self):
        rc, _ = self.run_setup(clt_rc=2)
        self.assertEqual(rc, 0)
        with self.assertRaises(SystemExit):
            self.run_setup(argv=("--yes", "--no-start", "--model", "UD-Q4_K_XL"))

    def test_gyro_is_offered_on_a_mac_as_one_file_with_a_compat_bf16_pack(self):
        fam = setup.FAMILIES["gyro"]
        g = self.models / setup.model_file(fam, "Gyro-S", 1)
        g.write_bytes(b"x")
        g.with_name(g.name + ".done").write_text("whole")
        rc, ran = self.run_setup(argv=("--yes", "--no-start", "--family", "gyro", "--model", "Gyro-S", "--context", "8192"))
        self.assertEqual(rc, 0)
        cfg = self.config("gyro-gyro-s")
        args = cfg["args"]
        self.assertEqual(args[args.index("--native") + 1], str(g), "the one file is the model")
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(g), "... and holds its n-gram table")
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "1"}, "29.7 GB of weights stream on a 32 GB Mac")
        self.assertNotIn("--mtp", args)
        self.assertTrue(ran[0][1].endswith("iq_pack.py"))
        self.assertIn("--compat-bf16", ran[0], "Gyro's pack converts the small Q6_K / Q5_K tensors (docs/GYRO.md)")
        self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / "gyro-gyro-s" / "tokenizer"))

    def test_gyro_m_on_a_small_mac_is_a_warning_not_a_refusal(self):
        fam = setup.FAMILIES["gyro"]
        g = self.models / setup.model_file(fam, "Gyro-M", 1)
        g.write_bytes(b"x")
        g.with_name(g.name + ".done").write_text("whole")
        rc, _ = self.run_setup(argv=("--yes", "--no-start", "--family", "gyro", "--model", "Gyro-M", "--context", "8192"))
        self.assertEqual(rc, 0)
        said = " ".join(self.warnings)
        self.assertIn("Gyro-M", said)
        self.assertIn("stream from the SSD", said)
        self.assertIn("not been measured on a Mac", said)

    def test_other_platforms_and_the_terminal_still_use_their_own_setup(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin" and not CONTROL:', src)
        self.assertIn('if sys.platform == "darwin" and CONTROL:', src)


class UpdateFlow(unittest.TestCase):
    """update.sh asks the same questions as a setup run but does not download model files: a model chosen there must
    be named, not dropped in silence (an M1 Max picked IQ3_S in update.sh and got no model and no message)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name).resolve() / "data"
        self.models = Path(self.tmp.name).resolve() / "own"
        self.data.mkdir()
        self.models.mkdir()
        self.said, self.warned = [], []

    def run_update(self, argv=("--update", "--model", "IQ3_S", "--vision", "no", "--yes")):
        with mock.patch.object(M, "mac_configs", lambda: []), \
                mock.patch.object(M.S, "data_folder_for", lambda a: (self.data, [])), \
                mock.patch.object(M, "mac_info", lambda run=None: {"chip": "Apple M1 Max", "ram_gb": 32.0,
                                                                   "macos": "26.6.2", "arm64": True, "clt": True}), \
                mock.patch.object(M, "mac_problems", lambda info: []), \
                mock.patch.object(M.S, "free_gb", lambda p: 200.0), \
                mock.patch.object(M.S, "step", lambda *a, **k: None), \
                mock.patch.object(M.S, "say", lambda *a, **k: self.said.append(" ".join(str(x) for x in a))), \
                mock.patch.object(M.S, "ok", lambda *a, **k: None), \
                mock.patch.object(M.S, "warn", lambda *a, **k: self.warned.append(" ".join(str(x) for x in a))), \
                mock.patch.object(M.S, "pip_install", lambda *a, **k: None), \
                mock.patch.object(M.S, "get_llama_cpp", lambda: self.data / "llama"), \
                mock.patch.object(M, "build_engine", lambda llama, jobs=None: self.data / "strata"), \
                mock.patch.object(M.S, "download", side_effect=AssertionError("an update must not download")):
            return M.main([*argv, "--data-dir", str(self.data), "--gguf-dir", str(self.models)])

    def test_a_model_chosen_in_an_update_is_named_not_dropped(self):
        self.assertEqual(self.run_update(), 0)
        self.assertIn("IQ3_S is not installed here", " ".join(self.warned))
        self.assertIn("does not download model files", " ".join(self.warned))
        self.assertIn("./setup.sh --setup", " ".join(self.said))

    def test_an_installed_model_keeps_the_update_quiet(self):
        for n in M.model_shard_names("IQ3_S"):
            f = self.models / n
            f.write_bytes(b"x")
            f.with_name(f.name + ".done").write_text("whole")
        tag = M.S.FAMILIES["qwen"]["tag"] + "IQ3_S"
        pack = self.data / "packs" / tag.lower()
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "native_experts.txt").write_text("x")
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        self.assertEqual(self.run_update(), 0)
        self.assertFalse([w for w in self.warned if "not installed here" in w], self.warned)


class Source(unittest.TestCase):
    """setup.py hands a Mac over before its own parser runs, so the Mac parser has to know --source:
    source_hint() recommends it on every platform (#2363)."""

    def test_the_source_flag_reaches_strata_source(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SOURCE", None)
        warnings = []
        # setup.data_folder() is mocked: the real one moves the model files of this checkout and of the data folder in
        # the user's settings.json into --data-dir, and saves that dir as the new data folder; with a temp dir that
        # deletes the models on exit and leaves settings.json pointing at a deleted directory
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup, "data_folder", lambda requested, *a, **k: (Path(tmp), [])), \
                mock.patch("subprocess.run", fake_run(M5)), \
                mock.patch.object(setup, "warn", lambda *a, **k: warnings.append(" ".join(str(x) for x in a))), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "ok", lambda *a, **k: None), \
                mock.patch.object(setup, "step", lambda *a, **k: None):
            M.main(["--source", "modelscope", "--yes", "--check", "--data-dir", tmp])
        self.assertEqual(os.environ.get("STRATA_SOURCE"), "modelscope")
        self.assertFalse([w for w in warnings if "--source" in w],
                         f"--source was not used: {warnings}")


if __name__ == "__main__":
    unittest.main()
