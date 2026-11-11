"""Tests for the per-family ROCm wheel choice of setup.py (#1103, #1267), against the version lists that AMD's TheRock
indexes really offered on 2026-10-07 (tools/test_setup_rocm_indexes.json), and for the sm_120 compiler warning that
now covers CUDA 13.2.0 / 13.2.1 only (#968).  No network, no GPU.

    python -m unittest tools.test_setup_rocm_pins
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

LISTINGS = json.loads((ROOT / "tools" / "test_setup_rocm_indexes.json").read_text(encoding="utf-8"))["indexes"]


def page(family):
    """The index page's html as pip sees it, from the saved version list."""
    return "<html><body>" + "".join(f'<a href="../rocm-{v}.tar.gz">rocm-{v}.tar.gz</a><br/>' for v in LISTINGS[family]) + "</body></html>"


class RocmPick(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(setup, "ROCM_VERSION_OVERRIDE", None)
        p.start()
        self.addCleanup(p.stop)

    def want(self, arch):
        index = setup.ROCM_INDEXES[arch]
        return setup.rocm_wanted(index, page(setup.rocm_family(index)))

    def test_every_family_gets_a_version_its_index_has(self):
        for arch in setup.ROCM_INDEXES:
            if setup.rocm_family(setup.ROCM_INDEXES[arch]) not in LISTINGS:
                continue                                  # the gfx1150 opt-in (#2147): no saved version list for its index yet
            with self.subTest(arch=arch):
                version, note = self.want(arch)
                self.assertIn(version, LISTINGS[setup.rocm_family(setup.ROCM_INDEXES[arch])])
                self.assertIsNone(note, note)             # the shipped pins all exist: no warning

    def test_gfx103X_has_no_7_10_but_gets_the_7_13_pin(self):          # #1103
        self.assertFalse([v for v in LISTINGS["gfx103X-all"] if v.startswith("7.10")])
        self.assertEqual(self.want("gfx1030"), ("7.13.0a20260515", None))
        self.assertEqual(self.want("gfx1031"), ("7.13.0a20260515", None))

    def test_gfx120X_and_gfx110X_keep_the_tested_version(self):
        for arch in ("gfx1201", "gfx1200", "gfx1100", "gfx1101", "gfx1102"):
            self.assertEqual(self.want(arch), ("7.10.0a20251120", None))

    def test_gfx1151_is_off_the_7_10_wheel(self):                      # #1267
        version, note = self.want("gfx1151")
        self.assertEqual((version, note), ("7.14.0a20260608", None))
        self.assertGreaterEqual(setup.rocm_vkey(version)[:2], (7, 11))

    def test_pinned_version_gone_falls_back_to_the_same_line_with_a_warning(self):
        old = setup.ROCM_FAMILY_PINS["gfx103X-all"]
        with mock.patch.dict(setup.ROCM_FAMILY_PINS, {"gfx103X-all": "7.13.0a20260301"}):
            version, note = setup.rocm_wanted(setup.ROCM_INDEXES["gfx1030"], page("gfx103X-all"))
        self.assertEqual(version, "7.13.0a20260515")             # the newest of 7.13, not of 7.14
        self.assertIn("7.13.0a20260301", note)
        self.assertIn("same line", note)
        self.assertEqual(old, "7.13.0a20260515")

    def test_whole_line_gone_falls_back_to_the_same_major(self):
        # gfx103X-all has no 7.10 line at all: the default 7.10.0a20251120 would land on the newest 7.x there
        with mock.patch.dict(setup.ROCM_FAMILY_PINS, {"gfx103X-all": "7.10.0a20251120"}):
            version, note = setup.rocm_wanted(setup.ROCM_INDEXES["gfx1030"], page("gfx103X-all"))
        self.assertEqual(version, "7.14.0a20260612")
        self.assertIn("same major", note)

    def test_other_major_is_not_taken(self):
        version, note = setup.rocm_pick(["6.5.0rc20250610"], "7.10.0a20251120")
        self.assertEqual(version, "7.10.0a20251120")
        self.assertIn("nothing of its major", note)

    def test_unreadable_index_keeps_the_preferred_version(self):
        self.assertEqual(setup.rocm_pick([], "7.10.0a20251120"), ("7.10.0a20251120", None))
        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")):
            self.assertEqual(setup.rocm_wanted(setup.ROCM_INDEXES["gfx1030"]), ("7.13.0a20260515", None))

    def test_the_index_page_is_read_from_the_rocm_folder(self):
        seen = []

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return page("gfx103X-all").encode()

        def fake(req, timeout=0):
            seen.append(req.full_url)
            return Resp()
        with mock.patch.object(setup.urllib.request, "urlopen", fake):
            self.assertEqual(setup.rocm_wanted(setup.ROCM_INDEXES["gfx1030"])[0], "7.13.0a20260515")
        self.assertEqual(seen, ["https://rocm.nightlies.amd.com/v2/gfx103X-all/rocm/"])

    def test_the_override_is_exact_and_not_looked_up(self):
        with mock.patch.object(setup, "ROCM_VERSION_OVERRIDE", "7.12.0a20260311"), \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError("no lookup")):
            self.assertEqual(setup.rocm_wanted(setup.ROCM_INDEXES["gfx1030"]), ("7.12.0a20260311", None))
        with mock.patch.object(setup, "ROCM_VERSION_OVERRIDE", "7.10.0a20251120"):      # forced, but still warned about
            version, note = setup.rocm_wanted(setup.ROCM_INDEXES["gfx1151"])
        self.assertEqual(version, "7.10.0a20251120")
        self.assertIn("segfault", note)

    def test_gfx1151_on_the_7_10_wheel_warns(self):                    # #1267 with an override or a fallback
        version, note = None, None
        with mock.patch.dict(setup.ROCM_FAMILY_PINS, {"gfx1151": "7.10.0a20251120"}):
            version, note = setup.rocm_wanted(setup.ROCM_INDEXES["gfx1151"], page("gfx1151"))
        self.assertEqual(version, "7.10.0a20251120")
        self.assertIn("segfault", note)

    def test_version_order(self):
        v = setup.rocm_index_versions(page("gfx1151"))
        self.assertEqual(v, sorted(v, key=setup.rocm_vkey))
        self.assertEqual(v[-1], "7.14.0a20260612")
        self.assertLess(setup.rocm_vkey("7.9.0rc20251008"), setup.rocm_vkey("7.10.0a20251009"))   # 7.9 before 7.10


class Sm120Compiler(unittest.TestCase):
    """#968: only CUDA 13.2.0 / 13.2.1 (nvcc build 51) are warned about; 13.2.2 (build 86) is accepted."""

    def run_sm120(self, build, archs=(120,), cuda_v=(13, 2), env=None):
        with mock.patch.object(setup, "nvcc_build", return_value=build), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, None)), \
                mock.patch.object(setup, "warn") as warn, mock.patch.object(setup, "ok"), \
                mock.patch.dict(setup.os.environ, env or {}):
            got = setup.sm120_nvcc("nvcc", cuda_v, list(archs))
        return got, warn

    def test_13_2_0_and_13_2_1_warn(self):
        for build in (51, 78, None):                       # None: a build that cannot be read stays suspect
            got, warn = self.run_sm120(build)
            self.assertEqual(got, ("nvcc", (13, 2)))
            warn.assert_called_once()
            self.assertIn("13.2.0", warn.call_args[0][0])

    def test_13_2_2_is_accepted_silently(self):
        for build in (86, 90):
            got, warn = self.run_sm120(build)
            self.assertEqual(got, ("nvcc", (13, 2)))
            warn.assert_not_called()

    def test_13_2_2_keeps_the_newest_toolkit_not_an_older_one(self):
        with mock.patch.object(setup, "nvcc_build", return_value=86), \
                mock.patch.object(setup, "find_nvcc", side_effect=AssertionError("must not look for an older one")):
            self.assertEqual(setup.sm120_nvcc("nvcc", (13, 2), [120]), ("nvcc", (13, 2)))

    def test_not_sm120_or_not_13_2_is_untouched(self):
        for archs, v in (((86,), (13, 2)), ((120,), (13, 0)), ((120,), (13, 3))):
            got, warn = self.run_sm120(51, archs, v)
            self.assertEqual(got, ("nvcc", v))
            warn.assert_not_called()

    def test_build_number_is_read_from_nvcc_output(self):
        text = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.2, V13.2.86\nBuild cuda_13.2.r13.2/compiler.123_0\n"
        with mock.patch.object(setup, "out", return_value=text):
            self.assertEqual(setup.nvcc_build("nvcc"), 86)
        with mock.patch.object(setup, "out", return_value=""):
            self.assertIsNone(setup.nvcc_build("nvcc"))


if __name__ == "__main__":
    unittest.main()
