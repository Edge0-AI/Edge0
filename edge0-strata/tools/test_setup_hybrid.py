"""#642: setup recommends the expert pool's worker count on a hybrid CPU (the P-cores but the host loop's one, plus
half of the E-cores) and writes nothing new anywhere else; a calibration's measured count wins over the rule.

    python -m unittest tools.test_setup_hybrid
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402


def fake_sys(root: Path, caps: list, pmu=None, smt=False) -> None:
    """A /sys tree: cpu N's capacity (and, with smt, a sibling CPU per core), and the hybrid PMU lists when given."""
    cpu_dir = root / "devices" / "system" / "cpu"
    n = 0
    for core, cap in enumerate(caps):
        for _ in range(2 if smt and cap >= 1000 else 1):
            d = cpu_dir / f"cpu{n}"
            (d / "topology").mkdir(parents=True)
            (d / "topology" / "physical_package_id").write_text("0")
            (d / "topology" / "core_id").write_text(str(core))
            (d / "cpu_capacity").write_text(str(cap))
            n += 1
    if pmu:
        for name, lst in zip(("cpu_core", "cpu_atom", "cpu_lowpower"), pmu):
            (root / "devices" / name).mkdir(parents=True)
            (root / "devices" / name / "cpus").write_text(lst + "\n")


class LinuxCores(unittest.TestCase):
    """#798: Turbo Boost Max 3.0's "favored" P-cores have a slightly higher cpu_capacity."""

    def classes(self, caps, **kw):
        with tempfile.TemporaryDirectory() as d:
            fake_sys(Path(d), caps, **kw)
            cl = setup.linux_core_classes(d)
        if not cl or max(cl) == min(cl):
            return None
        p = sum(1 for c in cl if c == max(cl))
        return p, len(cl) - p

    def test_favored_cores_do_not_hide_the_p_cores(self):
        caps = [1024, 1012, 1024, 1012, 1012, 1012, 1012, 1012] + [768] * 16        # Core Ultra 7 270K Plus
        self.assertEqual(self.classes(caps), (8, 16))                               # by capacity: >= 90% of the largest
        self.assertEqual(self.classes(caps, pmu=("0-7", "8-23")), (8, 16))          # by the PMU lists
        self.assertEqual(setup.hybrid_pool_workers((8, 16)), 15)

    def test_lp_e_cores_are_efficiency_cores(self):
        """#3262: Core Ultra 7 255H (6P + 8E + 2 LP-E): the LP-E cores are in /sys/devices/cpu_lowpower/cpus, in neither
        of the other two lists; they were counted as P-cores (8P + 8E, so the engine's every-core default of 15 stayed)."""
        caps = [1024] * 6 + [774] * 8 + [312] * 2
        self.assertEqual(self.classes(caps, pmu=("0-5", "6-13", "14-15")), (6, 10))
        self.assertEqual(self.classes(caps), (6, 10))                               # no PMU lists: capacity says the same
        self.assertEqual(setup.hybrid_pool_workers((6, 10)), 10)                    # not None (8, 8 gave every core: 15)
        self.assertEqual(self.classes(caps, pmu=("0-5", "6-13")), (6, 10))          # a kernel without the third list: capacity

    def test_plain_capacities_and_plain_cpus(self):
        self.assertEqual(self.classes([1024] * 8 + [512] * 8), (8, 8))
        self.assertIsNone(self.classes([1024] * 8))                                 # no E-cores
        self.assertIsNone(self.classes([1024, 1012, 1012, 1012]))                   # favored cores only: not hybrid

    def test_smt_siblings_count_once(self):
        self.assertEqual(self.classes([1024, 1012] * 4 + [768] * 8, smt=True, pmu=("0-15", "16-23")), (8, 8))

    def test_no_capacity_files_and_no_pmu_says_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            fake_sys(Path(d), [1024, 768])
            (Path(d) / "devices/system/cpu/cpu1/cpu_capacity").unlink()
            self.assertIsNone(setup.linux_core_classes(d))


class HybridWorkers(unittest.TestCase):
    def test_rule(self):
        self.assertIsNone(setup.hybrid_pool_workers(None))
        self.assertEqual(setup.hybrid_pool_workers((8, 16)), 15)   # i9-14900K: the fork's measured 15
        self.assertEqual(setup.hybrid_pool_workers((6, 8)), 9)     # i5-13600K
        self.assertEqual(setup.hybrid_pool_workers((4, 8)), 7)     # Ryzen AI 9 HX 370 (Zen 5 + Zen 5c)
        self.assertIsNone(setup.hybrid_pool_workers((8, 8)))       # i7-13700K: all cores measured faster (AMD_HIP.md)
        self.assertIsNone(setup.hybrid_pool_workers((8, 4)))

    def test_recommend_keeps_a_given_count(self):
        with mock.patch.object(setup, "cpu_cores", lambda: (8, 16)), mock.patch.object(setup, "ok", lambda *a: None):
            self.assertEqual(setup.recommend_pool_workers(["--spec", "4"]), ["--spec", "4", "--pool-workers", "15"])
            given = ["--pool-workers", "12"]
            self.assertEqual(setup.recommend_pool_workers(given), given)
        with mock.patch.object(setup, "cpu_cores", lambda: None):
            self.assertEqual(setup.recommend_pool_workers(["--spec", "4"]), ["--spec", "4"])

    def test_install_on_a_hybrid_cpu(self):
        ram, found = PROFILES["96GB-1x16GB"]
        hybrid = [mock.patch.object(setup, "cpu_cores", lambda: (8, 16))]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"], extra=hybrid)
        self.assertEqual(code, 0, text)
        a = cfg["args"]
        self.assertEqual(a[a.index("--pool-workers") + 1], "15")
        self.assertIn("hybrid CPU (8 performance + 16 efficiency cores)", text)
        # the same PC without the hybrid CPU: the config is the one it always was (no --pool-workers)
        code, text, plain, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"])
        self.assertEqual(code, 0, text)
        self.assertNotIn("--pool-workers", plain["args"])
        i = a.index("--pool-workers")
        self.assertEqual(a[:i] + a[i + 2:], plain["args"])

    def test_calibration_wins(self):
        ram, found = PROFILES["96GB-1x16GB"]
        cal = {"settings": {"--pool-workers": "11"}, "date": "2026-10-03"}
        extra = [mock.patch.object(setup, "cpu_cores", lambda: (8, 16)),
                 mock.patch.object(setup, "saved_calibration", lambda cfg: cal)]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"], extra=extra)
        self.assertEqual(code, 0, text)
        a = cfg["args"]
        self.assertEqual(a[a.index("--pool-workers") + 1], "11")
        self.assertEqual(a.count("--pool-workers"), 1)


if __name__ == "__main__":
    unittest.main()
