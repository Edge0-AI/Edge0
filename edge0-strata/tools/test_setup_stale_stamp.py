"""#2054: BUILD.json is trusted only beside a binary of the source's version.

A build that stopped (nvcc rejecting a newer host gcc) once left the old engine next to a stamp of the new source, and
every later update said "engine already built for this PC" for good.  `stale_binary` reads the version compiled into the
binary; `build_engine` refuses to stamp or install a binary of another version and compiles again when the installed
one is stale.

    python -m unittest tools.test_setup_stale_stamp
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GPU = {"arch": 86, "archs": [86], "name": "RTX 3060"}


def binary(version: str | None) -> bytes:
    return b"\x7fELF....tail_role_token=0 vram_elastic=0 engine=" + version.encode() + b"\n...." if version else b"old binary"


class StaleBinary(unittest.TestCase):
    def test_reads_the_compiled_in_version(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "source_version", lambda: "0.1.43"):
            exe = Path(d) / "strata"
            exe.write_bytes(binary("0.1.42"))
            self.assertEqual(setup.binary_engine_version(exe), "0.1.42")
            self.assertTrue(setup.stale_binary(exe))
            exe.write_bytes(binary("0.1.43"))
            self.assertFalse(setup.stale_binary(exe))

    def test_four_part_version_and_unknowns(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "source_version", lambda: "0.1.43.1"):
            exe = Path(d) / "strata"
            exe.write_bytes(binary("0.1.43.1"))
            self.assertFalse(setup.stale_binary(exe))
            exe.write_bytes(binary("0.1.43"))
            self.assertTrue(setup.stale_binary(exe))
            exe.write_bytes(binary(None))                 # a binary that carries no version: not judged
            self.assertFalse(setup.stale_binary(exe))
            self.assertFalse(setup.stale_binary(Path(d) / "missing"))


class BuildEngineStamp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / setup.EXE).write_bytes(binary("0.1.42"))
        self.old_stamp = {"source": "local", "version": "0.1.42", "archs": [86], "vision": "none",
                          "cuda_dirs": [], "src": "oldhash", "vision_src": None}
        (self.eng / "BUILD.json").write_text(json.dumps(self.old_stamp))
        self.built = binary("0.1.43")                     # what the fake compile leaves in the build folder
        self.compiles = 0
        patches = [mock.patch.object(setup, "ROOT", self.root),
                   mock.patch.object(setup, "source_hash", lambda parts: "newhash"),
                   mock.patch.object(setup, "source_version", lambda: "0.1.43"),
                   mock.patch.object(setup, "install_build_tools", lambda gpu, yes: ("nvcc", None)),
                   mock.patch.object(setup, "toolkit_root_defs", lambda nvcc: []),
                   mock.patch.object(setup, "host_compiler_defs", lambda cuda: []),
                   mock.patch.object(setup, "engine_defs", lambda archs, tk=13: []),
                   mock.patch.object(setup, "isa_floor_defs", lambda *a: []),
                   mock.patch.object(setup, "cpu_floor", lambda avx2: ""),
                   mock.patch.object(setup, "cpu_info", lambda: ("cpu", True)),
                   mock.patch.object(setup, "cmake_build", self.fake_build),
                   mock.patch.object(setup, "gpus", lambda: []),          # no real card of the box the test runs on (#4662: the build also covers every other card)
                   mock.patch.object(setup, "say", lambda *a, **k: None),
                   mock.patch.object(setup, "ok", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_build(self, src, bdir, target, defs, vcvars, bat):
        self.compiles += 1
        self.defs = list(defs)
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / setup.EXE).write_bytes(self.built)

    def stamp(self):
        return json.loads((self.eng / "BUILD.json").read_text())

    def test_a_good_build_is_stamped(self):
        setup.build_engine(GPU, "none", True, Path("llama"), toolkit=13)
        self.assertEqual(self.compiles, 1)
        self.assertEqual(self.stamp()["src"], "newhash")
        self.assertEqual(self.stamp()["version"], "0.1.43")
        self.assertEqual((self.eng / setup.EXE).read_bytes(), self.built)

    def test_the_q6k_expert_kernels_are_pinned_on(self):
        """PR #2515: a build/ folder first configured while STRATA_Q6K_EXPERTS defaulted to OFF keeps it in CMakeCache.txt;
        the configure line names it, so Qwen3.6 UD-Q4_K_XL (Q4_K gate/up, Q6_K down) still loads."""
        setup.build_engine(GPU, "none", True, Path("llama"), toolkit=13)
        self.assertIn("-DSTRATA_Q6K_EXPERTS=ON", self.defs)
        self.assertNotIn("-DSTRATA_Q6K_EXPERTS=OFF", self.defs)
        self.assertIn("-DSTRATA_MMQ_KQUANTS=ON", self.defs)  # flipped OFF -> ON in the same commit (465cd61d)

    def test_a_build_that_leaves_the_old_binary_is_not_stamped_or_installed(self):
        self.built = binary("0.1.42")                     # the build "succeeded" without producing the new engine
        with self.assertRaises(SystemExit):
            setup.build_engine(GPU, "none", True, Path("llama"), toolkit=13)
        self.assertEqual(self.stamp(), self.old_stamp)     # still the old hash: the next update compiles again
        self.assertEqual((self.eng / setup.EXE).read_bytes(), binary("0.1.42"))

    def test_a_poisoned_stamp_is_compiled_again(self):
        """The state #2054 found: the new source hash and version in BUILD.json, the old binary beside it."""
        (self.eng / "BUILD.json").write_text(json.dumps({**self.old_stamp, "version": "0.1.43", "src": "newhash"}))
        setup.build_engine(GPU, "none", True, Path("llama"), toolkit=13)
        self.assertEqual(self.compiles, 1)
        self.assertEqual((self.eng / setup.EXE).read_bytes(), self.built)

    def test_a_healthy_install_is_left_alone(self):
        (self.eng / setup.EXE).write_bytes(binary("0.1.43"))
        (self.eng / "BUILD.json").write_text(json.dumps({**self.old_stamp, "version": "0.1.43", "src": "newhash"}))
        setup.build_engine(GPU, "none", True, Path("llama"), toolkit=13)
        self.assertEqual(self.compiles, 0)


class UpdateInstalledEngine(unittest.TestCase):
    def test_a_poisoned_stamp_makes_the_update_compile(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(binary("0.1.42"))
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.43", "archs": [86],
                                                        "vision": "none", "src": "newhash"}))
            built = []
            with mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "source_hash", lambda parts: "newhash"), \
                    mock.patch.object(setup, "source_version", lambda: "0.1.43"), \
                    mock.patch.object(setup, "update_engine_gpu", lambda tk: dict(GPU)), \
                    mock.patch.object(setup, "get_llama_cpp", lambda: Path("llama")), \
                    mock.patch.object(setup, "build_engine", lambda *a, **k: built.append(a)), \
                    mock.patch.object(setup, "say", lambda *a, **k: None):
                setup.update_installed_engine("http://x", 13)
                self.assertEqual(len(built), 1)
                built.clear()
                (eng / setup.EXE).write_bytes(binary("0.1.43"))
                setup.update_installed_engine("http://x", 13)
                self.assertEqual(built, [])


if __name__ == "__main__":
    unittest.main()
