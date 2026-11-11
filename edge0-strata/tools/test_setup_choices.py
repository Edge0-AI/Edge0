"""Tests for setup.py's choices that depend on the PC (no GPU, no network, nothing installed): the image encoder of a
ready-made engine that has no code for the card (#331), --gguf-dir's shard count (#305), the experimental
Pascal/Volta build (#295), the CPU image encoder with the AMD backend (#304).

    python -m unittest tools.test_setup_choices
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class PrebuiltVision(unittest.TestCase):
    """#331: the 0.1.30/0.1.31 zip's encoder has no sm_75 code: an RTX 20 card gets the CPU encoder, not a compile."""
    META = {"version": "0.1.31", "archs": [75, 86, 89, 120], "ptx": True, "vision_archs": [86, 89, 120]}

    def test_a_card_the_encoder_has_no_code_for_gets_the_cpu_encoder(self):
        got, out = quiet(setup.prebuilt_vision, self.META, {"arch": 75}, "gpu")
        self.assertEqual(got, "cpu")
        self.assertIn("runs on the CPU instead", out)

    def test_covered_cards_keep_the_gpu_encoder(self):
        for arch in (86, 89, 120):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": arch}, "gpu")[0], "gpu")
        # a newer card than the newest encoder code: only with PTX in the zip
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89]}, {"arch": 120}, "gpu")[0],
                         "gpu")
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89], "ptx": False},
                               {"arch": 120}, "gpu")[0], "cpu")
        # 0.1.32's zip: the encoder built with 75-real too
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [75, 86, 89, 120]},
                               {"arch": 75}, "gpu")[0], "gpu")

    def test_other_choices_are_left_alone(self):
        for v in ("cpu", "none"):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": 75}, v), (v, ""))
        # an older BUILD.json without vision_archs: the engine's archs
        self.assertEqual(quiet(setup.prebuilt_vision, {"archs": [75, 86]}, {"arch": 75}, "gpu")[0], "gpu")


class VisionArchsAsAString(unittest.TestCase):
    """#3810: the packaging recipes write BUILD.json `vision_archs` as ONE string ("75;86;89;120"); 0.2.3 iterated it character by
    character and setup stopped with int(';') on the released Windows CUDA package.  A string and a list mean the same."""
    RELEASED = {"version": "0.2.3", "archs": [75, 86, 89, 120], "ptx": True, "cuda": "13.0", "vision": "gpu-cuda13",
                "vision_archs": "75;86;89;120", "portable": True}

    def test_the_released_package_manifest_is_read(self):
        for arch in (75, 86, 89, 120):
            self.assertEqual(quiet(setup.prebuilt_vision, self.RELEASED, {"arch": arch}, "gpu")[0], "gpu")

    def test_a_card_the_string_does_not_list_gets_the_cpu_reader(self):
        meta = {**self.RELEASED, "vision_archs": "86;89;120", "ptx": False}
        got, out = quiet(setup.prebuilt_vision, meta, {"arch": 75}, "gpu")
        self.assertEqual(got, "cpu")
        self.assertIn("runs on the CPU instead", out)
        self.assertEqual(quiet(setup.prebuilt_vision, {**meta, "ptx": True}, {"arch": 120}, "gpu")[0], "gpu")

    def test_a_string_and_a_list_agree(self):
        for value in ("75;86;89;120", [75, 86, 89, 120], ["sm_75", "86-real", "89", "compute_120"], "75,86 89;120"):
            self.assertEqual(setup.arch_numbers(value), [75, 86, 89, 120], value)
        self.assertEqual(setup.arch_numbers(None), [])
        self.assertEqual(setup.arch_numbers(""), [])
        self.assertEqual(setup.arch_names("gfx1100;gfx1201"), ["gfx1100", "gfx1201"])
        self.assertEqual(setup.arch_names(["gfx1100"]), ["gfx1100"])
        self.assertEqual(setup.arch_names(None), [])

    def test_the_other_readers_of_the_manifest_take_a_string_too(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "BUILD.json").write_text('{"archs": "60;61;70", "ptx": true}', encoding="utf-8")
            with mock.patch.object(setup, "engine_dir", return_value=Path(d)):
                self.assertEqual(setup.engine_archs(), ([60, 61, 70], True))
            (Path(d) / "BUILD.json").write_text('{"backend": "hip", "archs": ["gfx1100"], "vision_archs": "gfx1100;gfx1201"}', encoding="utf-8")
            self.assertEqual(setup.reader_archs(Path(d)), ["gfx1100", "gfx1201"])
            (Path(d) / "BUILD.json").write_text('{"backend": "hip", "archs": ["gfx1100", "gfx1201"]}', encoding="utf-8")
            self.assertEqual(setup.reader_archs(Path(d)), ["gfx1100", "gfx1201"])
        self.assertEqual(setup.reader_archs(Path(__file__).parent / "no-such-folder"), [])


class GgufDirShards(unittest.TestCase):
    """#305: --gguf-dir reads the shard count from the -of-N part of the name instead of assuming two."""

    def shards(self, names, family="qwen", model="IQ3_XXS"):
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                (Path(d) / n).write_bytes(b"")
            return [p.name for p in setup.gguf_dir_shards(Path(d), setup.FAMILIES[family], model)]

    def test_the_published_two_shards(self):
        names = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                 "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
        self.assertEqual(self.shards(names), names)
        self.assertEqual(self.shards(names[:1]), names)              # shard 2 missing: check_shards says so later
        swift = ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                 "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"]
        self.assertEqual(self.shards(swift + ["mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"], "swift", "Q2_0"), swift)

    def test_four_unsloth_shards(self):
        names = ["Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf" % i for i in range(1, 5)]
        self.assertEqual(self.shards(names + ["mmproj-F16.gguf"], model="Q4_K_XL"), names)
        self.assertEqual(self.shards(names[:1], model="Q4_K_XL"), names)   # the others named from the first

    def test_another_split_of_a_setup_size(self):
        names = ["my-IQ3_XXS-%05d-of-00003.gguf" % i for i in range(1, 4)]
        other = ["my-Q2_0-%05d-of-00002.gguf" % i for i in range(1, 3)]
        self.assertEqual(self.shards(names + other), names)           # the one with the size in its name
        self.assertEqual(self.shards(other, model="Q2_0"), other)

    def test_nothing_recognisable_keeps_the_published_names(self):
        want = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
        self.assertEqual(self.shards([]), want)
        self.assertEqual(self.shards(["a-00001-of-00002.gguf", "b-00001-of-00002.gguf"]), want)   # ambiguous


class GgufDirUnsupported(unittest.TestCase):
    """#444: --gguf-dir with GGUFs Strata cannot run (Unsloth's UD-IQ3_XXS, UD-Q2_K_XL) says so, names the files it
    runs (ISTA-DASLab's GSQ-RCO, the Coder's, Unsloth's UD-Q4_K_XL) and the --family/--model of the usable ones."""
    UD = ["Qwen3.8-Flash-Next-UD-IQ3_XXS-%05d-of-00003.gguf" % i for i in range(1, 4)] + \
         ["Qwen3.8-Flash-Next-UD-Q2_K_XL-%05d-of-00003.gguf" % i for i in range(1, 4)]

    def problem(self, names, family="qwen", model="IQ3_XXS"):
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                (Path(d) / n).write_bytes(b"")
            fam = setup.FAMILIES[family]
            first = setup.gguf_dir_shards(Path(d), fam, model)[0]
            p = setup.gguf_dir_problem(Path(d), first, fam, model)
            return p and (p[0].replace(d, "<D>"), p[1])

    def test_quant_names(self):
        for name, want in (("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf", "UD-IQ3_XXS"),
                           ("Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", "UD-Q2_K_XL"),
                           ("model-Q4_K_M.gguf", "Q4_K_M"),
                           ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", None),
                           ("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", None),
                           ("my-IQ3_XXS-00001-of-00003.gguf", None), ("mmproj-F16.gguf", None),
                           ("a-00001-of-00002.gguf", None)):
            self.assertEqual(setup.gguf_unsupported(name), want, name)

    def test_an_unsloth_ud_file_is_not_taken_for_a_gsq_rco_size(self):
        msg, hint = self.problem(self.UD)                   # IQ3_XXS: UD-IQ3_XXS has the size in its name
        self.assertEqual(msg, "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf is UD-IQ3_XXS, a GGUF Strata "
                              "cannot run")
        self.assertIn("ISTA-DASLab's GSQ-RCO files", hint)
        self.assertIn("Unsloth's UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 only", hint)

    def test_a_folder_without_the_choice_names_what_is_there(self):
        gsq = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-%05d-of-00002.gguf" % i for i in (1, 2)]
        msg, hint = self.problem(self.UD + gsq, "unsloth", "UD-Q4_K_XL")
        self.assertEqual(msg, "<D> has no Qwen3.8-Flash-Next (Unsloth) UD-Q4_K_XL file")
        self.assertIn("Not usable here: Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf, "
                      "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", hint)
        self.assertIn("Usable here: --family coder --model IQ1_M", hint)

    def test_usable_folders_are_left_alone(self):
        good = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-%05d-of-00002.gguf" % i for i in (1, 2)]
        self.assertIsNone(self.problem(good + self.UD))
        self.assertIsNone(self.problem(["my-IQ3_XXS-%05d-of-00003.gguf" % i for i in (1, 2, 3)]))
        self.assertIsNone(self.problem([]))                 # nothing there: check_shards' "missing", as before

    def test_the_size_of_another_family(self):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["64GB-1x32GB"]
        with tempfile.TemporaryDirectory() as d:
            code, out, _, _ = install(ram, found, ["--gguf-dir", d, "--family", "unsloth", "--model", "IQ3_XXS"])
        self.assertEqual(code, 1)
        self.assertIn("has no IQ3_XXS model file", out)
        self.assertIn("choose one of: UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 (or IQ3_XXS: --family qwen --model IQ3_XXS, --family swift --model "
                      "IQ3_XXS)", out)
        self.assertIn("Strata runs ISTA-DASLab's GSQ-RCO files", out)


class ReadOnlyGgufDir(unittest.TestCase):
    """--gguf-dir on a read-only folder (a share): whole shards without a finish mark still install, with a warning."""

    def test_a_finish_mark_that_cannot_be_written_is_a_warning(self):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["64GB-1x32GB"]
        real_mark = setup.mark

        def read_only_mark(path, text=""):
            if "gguf_share" in str(path):
                raise PermissionError(13, "Read-only file system")
            real_mark(path, text)

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "gguf_share"
            d.mkdir()
            for i in (1, 2):
                (d / f"Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{i:05d}-of-00002.gguf").write_bytes(b"")
            code, out, cfg, _ = install(ram, found, ["--gguf-dir", str(d), "--family", "qwen", "--model", "Q2_0",
                                                     "--no-start"],
                                        extra=[mock.patch.object(setup, "whole_shard", lambda s: True),
                                               mock.patch.object(setup, "mark", read_only_mark)])
        self.assertEqual(code, 0, out)
        self.assertIn("is whole but its finish mark cannot be written", out)


    def test_mark_itself_never_stops_on_a_read_only_folder(self):
        """#570: every finish mark (the SHA-256 result of a verified shard too), not only the one for a whole shard."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, "write_text",
                                                                     side_effect=PermissionError(13, "read-only")):
            setup.mark(Path(tmp) / "x.gguf", "sha256 abc")                   # no exception
            self.assertFalse(setup.done(Path(tmp) / "x.gguf"))


class VsRange(unittest.TestCase):
    """#985: Visual Studio 2026 (version 18) is looked for only with CUDA 13.3 or newer."""

    def asked(self, cuda_v):
        seen = []
        with mock.patch.object(setup.Path, "exists", lambda self: True),                 mock.patch.object(setup, "out", lambda cmd, *a, **k: seen.append(cmd) or "C:/VS"):
            setup.find_vcvars(cuda_v)
        return seen[0][seen[0].index("-version") + 1] if "-version" in seen[0] else None

    def test_the_range_follows_the_toolkit(self):
        self.assertIsNone(self.asked(None))      # #881: without CUDA vswhere gets no version range
        self.assertEqual(self.asked((13, 0)), "[17.0,18.0)")
        self.assertEqual(self.asked((13, 2)), "[17.0,18.0)")
        self.assertEqual(self.asked((13, 3)), "[17.0,19.0)")
        self.assertEqual(self.asked((14, 0)), "[17.0,19.0)")


    def test_vcvars_by_hand(self):
        """#881: STRATA_VCVARS names a vcvars64.bat directly, for an install vswhere cannot use."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bat = Path(d) / "vcvars64.bat"
            bat.write_text("rem")
            with mock.patch.dict(os.environ, {"STRATA_VCVARS": str(bat)}):
                self.assertEqual(setup.find_vcvars(), bat)
            with mock.patch.dict(os.environ, {"STRATA_VCVARS": str(Path(d) / "missing.bat")}):
                self.assertIsNone(setup.find_vcvars())

class ExperimentalSm60(unittest.TestCase):
    """#295: Pascal (6.x) and Volta (7.0) only with STRATA_EXPERIMENTAL_SM60=1, built with -DSTRATA_EXPERIMENTAL_SM60=ON
    and a CUDA 12.x toolkit; nothing changes without the variable."""

    def card(self, arch):
        return {"arch": arch, "vram_gb": 11.0, "index": 0, "name": "card"}

    def test_the_gate(self):
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": ""}):
            for arch in ("60", "61", "70"):
                p = setup.gpu_problem(self.card(arch))
                self.assertIn("not supported", p)
                self.assertIn("choose it with --gpu 0", p)              # the CUDA 12 engine (docs/OLDER_GPUS.md)
            self.assertNotIn("--gpu", setup.gpu_problem(self.card("52")))
            self.assertIsNone(setup.gpu_problem(self.card("75")))
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "1"}):
            for arch in ("60", "61", "70", "75", "120"):
                self.assertIsNone(setup.gpu_problem(self.card(arch)), arch)
            for arch in ("52", "72"):
                self.assertIsNotNone(setup.gpu_problem(self.card(arch)), arch)

    def test_the_build_flag(self):
        self.assertEqual(setup.engine_defs([61]), ["-DSTRATA_EXPERIMENTAL_SM60=ON"])
        self.assertEqual(setup.engine_defs([70, 86]), ["-DSTRATA_EXPERIMENTAL_SM60=ON"])
        self.assertEqual(setup.engine_defs([75, 86, 120]), [])

    def test_find_nvcc_below(self):
        with tempfile.TemporaryDirectory() as d:
            new, old = Path(d) / "13" / "nvcc", Path(d) / "12" / "bin" / "nvcc"
            for p in (new, old):
                p.parent.mkdir(parents=True)
                p.write_bytes(b"")
            versions = {str(new): "Cuda compilation tools, release 13.0, V13.0.88",
                        str(old): "Cuda compilation tools, release 12.9, V12.9.86"}
            with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.shutil, "which", lambda n: str(new)), \
                    mock.patch.dict(os.environ, {"CUDA_PATH": str(Path(d) / "12")}), \
                    mock.patch.object(setup, "out", lambda cmd: versions.get(cmd[0], "")):  # #414
                self.assertEqual(setup.find_nvcc(), (str(new), (13, 0)))
                self.assertEqual(setup.find_nvcc(below=(13, 0)), (str(old), (12, 9)))
                # #601: STRATA_NVCC is the only one considered; CUDA_HOME is a candidate like CUDA_PATH
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(old)}):
                    self.assertEqual(setup.find_nvcc(), (str(old), (12, 9)))
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(new)}):
                    got, text = quiet(setup.find_nvcc, below=(13, 0))
                    self.assertEqual(got, (None, None))
                    self.assertIn("needs one older than 13.0", text)
                with mock.patch.dict(os.environ, {"STRATA_NVCC": str(Path(d) / "missing")}):
                    got, text = quiet(setup.find_nvcc)
                    self.assertEqual(got, (str(new), (13, 0)))
                    self.assertIn("no such file", text)
            with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.shutil, "which", lambda n: None), \
                    mock.patch.dict(os.environ, {"CUDA_HOME": str(Path(d) / "12")}), \
                    mock.patch.object(setup, "out", lambda cmd: versions.get(cmd[0], "")):
                os.environ.pop("CUDA_PATH", None)
                got = setup.find_nvcc(below=(13, 0))
                self.assertEqual(got, (str(old), (12, 9)))

    def tools(self, archs, nvcc):
        seen = []

        def find(below=None):
            seen.append(below)
            return nvcc(below)

        with mock.patch.object(setup, "find_nvcc", find), mock.patch.object(setup, "find_vcvars", lambda cuda_v=None: "vcvars"), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n):
            got, _ = quiet(setup.install_build_tools, {"arch": str(archs[0]), "archs": archs}, True)
        return got, seen

    def test_pascal_takes_cuda_12(self):
        got, seen = self.tools([61], lambda below: ("nvcc12", (12, 9)) if below == (13, 0) else ("nvcc13", (13, 0)))
        self.assertEqual(got[0], "nvcc12")
        self.assertEqual(seen, [(13, 0)])
        got, seen = self.tools([86, 120], lambda below: ("nvcc13", (13, 0)))
        self.assertEqual((got[0], seen), ("nvcc13", [None]))           # the default: unchanged

    def test_cuda_132_on_sm120(self):
        """#892 / #968: nvcc 13.2 for an RTX 50 card: an older 13.x is taken when there is one, else a warning; never an
        error, and no other card or toolkit is touched."""
        def nvcc(below):
            return ("nvcc130", (13, 0)) if below == (13, 2) else ("nvcc132", (13, 2))

        got, seen = self.tools([120], nvcc)
        self.assertEqual(got[0], "nvcc130")
        self.assertEqual(seen, [None, (13, 2)])
        with mock.patch.dict(os.environ, {"STRATA_NVCC": "x"}):          # the user's own pick is kept
            got, seen = self.tools([120], nvcc)
            self.assertEqual((got[0], seen), ("nvcc132", [None]))
        only132 = lambda below: (None, None) if below == (13, 2) else ("nvcc132", (13, 2))
        with mock.patch.object(setup, "warn") as warn:                   # nothing older: the one found, with a warning
            got, _ = self.tools([120], only132)
        self.assertEqual(got[0], "nvcc132")
        self.assertIn("#892", warn.call_args[0][0])
        for archs in ([86], [89, 86]):                                   # not an sm_120 card: as before
            got, seen = self.tools(archs, nvcc)
            self.assertEqual((got[0], seen), ("nvcc132", [None]))

    def test_pascal_without_cuda_12_stops(self):
        for archs in ([61], [70, 120]):
            with self.subTest(archs=archs), self.assertRaises(SystemExit):
                self.tools(archs, lambda below: (None, None) if below else ("nvcc13", (13, 0)))


class HipVision(unittest.TestCase):
    """#304: --vision cpu with --backend hip builds the CPU image encoder beside the HIP engine."""

    @mock.patch.object(setup, "WIN", False)
    def test_the_choice(self):
        self.assertEqual(quiet(setup.hip_vision, "cpu"), ("cpu", ""))
        for asked in ("no", "none"):
            self.assertEqual(quiet(setup.hip_vision, asked), ("none", ""))
        for asked in (None, "yes", "gpu"):            # pictures are on by default; the engine folder's readers settle gpu vs cpu
            self.assertEqual(quiet(setup.hip_vision, asked), ("gpu", ""))

    def build(self, meta, vision, vexe=False):
        """build_engine_hip with an engine that is already built: only the encoder can be missing."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(b"engine")
            if vexe:
                (eng / setup.VEXE).write_bytes(b"vision")
            src, vsrc = "S", "V"
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "archs": ["gfx1201"], "src": src, **meta}))
            built = []

            def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
                built.append((target, defs))
                (bdir / "bin").mkdir(parents=True)
                (bdir / "bin" / setup.VEXE).write_bytes(b"vision")

            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "cmake_build", cmake_build), \
                    mock.patch.object(setup, "source_hash", lambda paths: vsrc if paths == setup.VISION_SOURCES else src):
                quiet(setup.build_engine_hip, {"arch": "gfx1201"}, "llama", vision)
            return built, json.loads((eng / "BUILD.json").read_text()), (eng / setup.VEXE).exists()

    def test_the_cpu_encoder_is_built_once(self):
        built, meta, have = self.build({}, "cpu")
        self.assertEqual([t for t, _ in built], ["strata-vision"])
        self.assertIn("-DSTRATA_VISION_CUDA=OFF", built[0][1])
        self.assertIn("-DSTRATA_PORTABLE=OFF", built[0][1])            # built for this PC (#411: portable is the default)
        self.assertEqual((meta["vision"], meta["vision_src"], have), ("cpu", "V", True))
        built, meta, _ = self.build({"vision": "cpu", "vision_src": "V"}, "cpu", vexe=True)
        self.assertEqual(built, [])                                    # built and unchanged: nothing to do
        built, meta, _ = self.build({"vision": "cpu", "vision_src": "old"}, "cpu", vexe=True)
        self.assertEqual([t for t, _ in built], ["strata-vision"])     # its source changed: again

    def test_without_images_nothing_changes(self):
        built, meta, have = self.build({}, "none")
        self.assertEqual((built, have), ([], False))
        self.assertNotIn("vision_src", meta)


class RotationalDisk(unittest.TestCase):
    """#605: a model on a rotational disk gets --ple-io ram when the n-gram table fits the RAM, else a warning."""

    def install(self, ram, disk, platform_ok=False, model="Q2_0"):
        from test_setup_golden import PROFILES, install
        _, found = PROFILES["64GB-1x32GB"]
        return install(ram, found, ["--family", "qwen", "--model", model, "--no-start"],
                       extra=[mock.patch.object(setup, "rotational_disk", lambda p: disk),
                              mock.patch.object(setup, "ple_ram_platform_ok", lambda: platform_ok)])

    def test_fits(self):
        code, out, cfg, _ = self.install(127.8, "sdb")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--ple-io") + 1], "ram")
        self.assertIn("rotational disk (sdb)", out)

    def test_does_not_fit(self):
        code, out, cfg, _ = self.install(63.7, "sdb")
        self.assertEqual(code, 0, out)
        self.assertNotIn("--ple-io", cfg["args"])
        self.assertIn("An SSD is recommended", out)

    def test_ssd(self):
        code, out, cfg, _ = self.install(127.8, None)
        self.assertNotIn("--ple-io", cfg["args"])
        self.assertNotIn("rotational", out)

    def test_ssd_ram_to_spare(self):
        # 147-verify (Linux): an SSD and RAM beyond the model's need + the table + 16 GB: the table in RAM too
        code, out, cfg, _ = self.install(127.8, None, platform_ok=True)
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--ple-io") + 1], "ram")
        self.assertIn("kept in RAM (--ple-io ram)", out)
        self.assertNotIn("rotational", out)

    def test_ssd_little_ram_keeps_the_ssd(self):
        # Q2_0 needs 48 GB + the 28.8 GB table + 16: a 64 GB / 92 GB PC keeps the SSD reads
        for ram in (63.7, 92.0):
            code, out, cfg, _ = self.install(ram, None, platform_ok=True)
            self.assertEqual(code, 0, out)
            self.assertNotIn("--ple-io", cfg["args"])

    def test_ssd_ram_rule_follows_the_model(self):
        # IQ3_S needs 62 GB: 96 GB is not enough to spare the table, 128 GB is
        for ram, want in ((95.8, False), (127.8, True)):
            code, out, cfg, _ = self.install(ram, None, platform_ok=True, model="IQ3_S")
            self.assertEqual(code, 0, out)
            self.assertEqual("--ple-io" in cfg["args"], want, (ram, cfg["args"]))

    def test_sysfs(self):
        if not sys.platform.startswith("linux"):
            self.assertIsNone(setup.rotational_disk(__file__))   # Windows: never
            return
        self.assertIn(setup.rotational_disk(__file__), (None,) + tuple(os.listdir("/sys/block")))


class VramReserve(unittest.TestCase):
    """#493: --vram-reserve-mib N - VRAM the engine leaves free for other programs - goes into the config only when
    given (the default config is tools/test_setup_golden.py's, unchanged), at setup and on a start."""

    def setUp(self):
        from test_setup_golden import PROFILES
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def install(self, *extra):
        from test_setup_golden import install
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra])

    def test_given_at_setup(self):
        code, out, cfg, _ = self.install("--vram-reserve-mib", "2048")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "2048")
        self.assertIn("VRAM kept free for other programs: 2048 MiB", out)
        code, out, cfg, _ = self.install("--vision", "no")
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # not given: the engine's default, as before
        code, out, cfg, _ = self.install("--vision", "cpu")
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # the CPU reader needs no reserve
        code, out, cfg, _ = self.install()
        self.assertEqual(cfg["args"][cfg["args"].index("--vram-reserve-mib") + 1], "700")   # pictures on by default, GPU reader on a 32 GB card: its 700 MiB

    def test_with_images_one_value(self):
        code, out, cfg, _ = self.install("--vision", "cpu", "--vram-reserve-mib", "1500")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "1500")

    def test_a_negative_value_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code, out, cfg, _ = self.install("--vram-reserve-mib", "-1")
        self.assertEqual(code, 2)
        self.assertIn("--vram-reserve-mib takes a number of MiB", err.getvalue())

    def test_kept_from_an_earlier_install(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"args": ["--max-context", "8192", "--vram-reserve-mib", "2048"]}))
            self.assertEqual(setup.choices_from_config(p)["vram_reserve_mib"], 2048)
            p.write_text(json.dumps({"args": ["--vision", "--vram-reserve-mib", "700"], "vision": {"gpu": True}}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])   # the images' own reserve
            p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])

    def test_given_on_a_start(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--kv", "int8"], "gpu": 0, "gpus_asked": True}))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: self.found), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup, "ensure_engine_for", lambda cards, path, cfg, yes: cfg), \
                    mock.patch.object(setup.subprocess, "call", call):
                for n in ("2048", "1024"):
                    _, out = quiet(setup.start, p, None, None, False, True, None, {"vram_reserve_mib": int(n)})
                    self.assertEqual(json.loads(p.read_text())["args"], ["--kv", "int8", "--vram-reserve-mib", n])
        self.assertIn("1024 MiB of VRAM kept free", out)
        self.assertTrue(call.called)


class SmallCardTip(unittest.TestCase):
    """#496: a card under 8 GB (a 6 GB laptop RTX 3060) gets a tip for when the start has no room for the expert cache;
    the engine chooses its own reserve there, so the config is the one any card gets.  An 8 GB card: no tip."""

    def install(self, vram, *extra):
        from test_setup_golden import card, install
        return install(63.7, [card(0, "NVIDIA GeForce RTX 3060 Laptop GPU", vram, "86")],
                       ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra])

    def test_a_6gb_card(self):
        code, out, cfg, _ = self.install(6.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # the engine decides (no fixed reserve)
        self.assertIn("no VRAM is left for the expert cache", out)
        self.assertIn("--draft-vocab en", out)
        self.assertIn("--mtp", cfg["args"])                          # the draft layer stays: the server needs it

    def test_an_8gb_card_has_no_tip(self):
        for vram in (8188 / 1024, 8.0):                               # nvidia-smi lists an 8 GB card as 8188 MiB
            code, out, cfg, _ = self.install(vram)
            self.assertEqual(code, 0, out)
            self.assertNotIn("no VRAM is left for the expert cache", out)
            self.assertNotIn("--vram-reserve-mib", cfg["args"])

    def test_a_given_reserve_is_kept(self):
        code, out, cfg, _ = self.install(6.0, "--vram-reserve-mib", "500")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "500")

    def test_the_tip(self):
        self.assertIn("an 8K context", " ".join(setup.small_card_note(32768, None)))
        self.assertIn("--draft-vocab en", " ".join(setup.small_card_note(32768, "cjk")))
        tip = " ".join(setup.small_card_note(8192, "en"))
        self.assertNotIn("--draft-vocab", tip)
        self.assertIn("close other programs", tip)


class DesktopReserveTip(unittest.TestCase):
    """#560 #516: an AMD card on a Linux desktop gets a recommended reserve (3072 MiB) - a tip, the config is not
    changed."""

    def test_desktop_detection(self):
        with mock.patch.object(setup.sys, "platform", "linux"):
            self.assertTrue(setup.linux_desktop({"WAYLAND_DISPLAY": "wayland-0"}))
            self.assertTrue(setup.linux_desktop({"DISPLAY": ":0"}))
            self.assertFalse(setup.linux_desktop({}))                 # a headless box / ssh
        with mock.patch.object(setup.sys, "platform", "win32"):
            self.assertFalse(setup.linux_desktop({"DISPLAY": ":0"}))  # Windows counts the desktop itself (#497)

    def test_the_tip(self):
        tip = " ".join(setup.desktop_reserve_note())
        self.assertIn("--vram-reserve-mib 3072", tip)
        self.assertIn("desktop", tip)


class CudaVision(unittest.TestCase):
    """#411: the image encoder setup compiles beside the CUDA engine is built for this PC, not portable."""

    def test_the_encoder_is_built_native(self):
        for vision in ("gpu", "cpu"):
            with self.subTest(vision), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                eng = root / "engine"
                eng.mkdir()
                (eng / setup.EXE).write_bytes(b"engine")                 # the engine is built: only the encoder
                (eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": [86], "src": "S"}))
                built = []

                def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
                    built.append((target, defs))
                    (bdir / "bin").mkdir(parents=True)
                    (bdir / "bin" / setup.VEXE).write_bytes(b"vision")

                with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "cmake_build", cmake_build), \
                        mock.patch.object(setup, "source_hash", lambda p: "V" if p == setup.VISION_SOURCES else "S"), \
                        mock.patch.object(setup, "install_build_tools", lambda gpu, yes: (str(root / "nvcc"), None)), \
                        mock.patch.object(setup, "source_version", lambda: "test"):
                    quiet(setup.build_engine, {"arch": "86"}, vision, True, "llama")
                self.assertEqual([t for t, _ in built], ["strata-vision"])
                self.assertIn("-DSTRATA_PORTABLE=OFF", built[0][1])
                self.assertIn(f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}", built[0][1])


class SwiftIq3s(unittest.TestCase):
    """#1651: Swift 1.5 has its own IQ3_S tier since 2026-10-08 (ukisai's Swift-1.5 GGUF repo) - the model choice
    must not hide it behind the qwen family.  Names and sizes checked against the Hub listing when written."""

    def test_iq3_s_is_a_swift_choice(self):
        self.assertIn("swift", setup.MODELS["IQ3_S"].get("families", ("qwen", "swift")))

    def test_the_published_file_names(self):
        fam = setup.FAMILIES["swift"]
        self.assertEqual(setup.model_shards(fam, "IQ3_S"), 2)
        self.assertEqual([setup.model_file(fam, "IQ3_S", i) for i in (1, 2)],
                         ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                          "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"])


class HostCompilers(unittest.TestCase):
    """#1645: CC / CXX / CUDAHOSTCXX name the build's compilers - handed to CMake as -D, so a build folder that is
    already configured picks them up too (CXX alone reaches CMake only on the first configure: the reporter's
    `CXX=g++-14` run died at the last step).  CUDAHOSTCXX is nvcc's host compiler and defaults to CXX."""

    def test_env_names_the_compilers(self):
        with mock.patch.dict(os.environ, {"CC": "gcc-14", "CXX": "g++-14", "CUDAHOSTCXX": "g++-14-alt"}):
            self.assertEqual(setup.host_compiler_defs(True), ["-DCMAKE_C_COMPILER=gcc-14",
                                                              "-DCMAKE_CXX_COMPILER=g++-14",
                                                              "-DCMAKE_CUDA_HOST_COMPILER=g++-14-alt"])

    def test_cudahostcxx_defaults_to_cxx_and_needs_cuda(self):
        with mock.patch.dict(os.environ, {"CC": "", "CXX": "g++-14", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(True), ["-DCMAKE_CXX_COMPILER=g++-14",
                                                              "-DCMAKE_CUDA_HOST_COMPILER=g++-14"])
            self.assertEqual(setup.host_compiler_defs(False), ["-DCMAKE_CXX_COMPILER=g++-14"])   # a C++-only project

    def test_unset_adds_nothing(self):
        with mock.patch.dict(os.environ, {"CC": "", "CXX": "", "CUDAHOSTCXX": ""}):
            self.assertEqual(setup.host_compiler_defs(True), [])

    def test_the_defs_reach_cmake(self):
        """The engine's and the CUDA encoder's configure both take them."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": [86], "src": "old"}))
            built = []

            def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
                built.append((target, defs))
                (bdir / "bin").mkdir(parents=True, exist_ok=True)
                (bdir / "bin" / setup.VEXE).write_bytes(b"vision")
                (bdir / setup.EXE).write_bytes(b"engine")

            env = {"CC": "gcc-14", "CXX": "g++-14", "CUDAHOSTCXX": ""}
            with mock.patch.dict(os.environ, env), mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "cmake_build", cmake_build), \
                    mock.patch.object(setup, "source_hash", lambda p: "V" if p == setup.VISION_SOURCES else "new"), \
                    mock.patch.object(setup, "install_build_tools", lambda gpu, yes: (str(root / "nvcc"), None)), \
                    mock.patch.object(setup, "source_version", lambda: "test"):
                quiet(setup.build_engine, {"arch": "86"}, "gpu", True, "llama")
        for target, defs in built:
            self.assertIn("-DCMAKE_CXX_COMPILER=g++-14", defs, target)
            self.assertIn("-DCMAKE_CUDA_HOST_COMPILER=g++-14", defs, target)


if __name__ == "__main__":
    unittest.main()


class CalibrateWithModel(unittest.TestCase):
    """#2036: --calibrate --model X tunes model X (it used to fall through to a plain re-setup); a model that is not
    installed is an error that says so; --calibrate with --setup or --check says it does nothing."""

    def run_main(self, *argv, installed=("iq3_xxs", "q2_0")):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfgs = []
        for tag in installed:
            p = Path(tmp.name) / f"strata-{tag}.json"
            p.write_text(json.dumps({"exe": "x", "args": ["--max-context", "8192"], "model_name": tag}))
            cfgs.append(p)
        tuned = []
        with mock.patch.object(sys, "argv", ["setup.py", *argv]), \
                mock.patch.object(setup, "installed_configs", lambda: cfgs), \
                mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None), \
                mock.patch.object(setup, "calibrate_config", lambda c: tuned.append(c.name) or True), \
                mock.patch.object(setup, "start", lambda *a, **k: 0), \
                mock.patch.object(setup, "data_folder", lambda d, *a, **k: (Path(tmp.name), [])), \
                mock.patch.object(setup, "step", side_effect=SystemExit("reached the plain setup")):
            try:
                code, out = quiet(setup.main)
            except SystemExit as e:
                code, out = e.code, ""
        return code, tuned

    def test_model_is_the_target(self):
        code, tuned = self.run_main("--calibrate", "--model", "IQ3_XXS", "--yes", "--no-start")
        self.assertEqual((code, tuned), (0, ["strata-iq3_xxs.json"]))

    def test_a_model_not_installed_is_an_error(self):
        code, tuned = self.run_main("--calibrate", "--model", "IQ3_S", "--yes", "--no-start")
        self.assertEqual((code, tuned), (1, []))

    def test_targets(self):
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for tag in ("iq3_xxs", "q2_0"):
                p = Path(d) / f"strata-{tag}.json"
                p.write_text(json.dumps({"args": []}))
                ps.append(p)
            self.assertEqual(setup.calibration_targets(ps, None, None), ps)
            self.assertEqual(setup.calibration_targets(ps, "Q2_0", None), [ps[1]])
            self.assertEqual(setup.calibration_targets(ps, "Q2_0", "qwen"), [ps[1]])
            self.assertEqual(setup.calibration_targets(ps, "Q4_K_M", None), [])
