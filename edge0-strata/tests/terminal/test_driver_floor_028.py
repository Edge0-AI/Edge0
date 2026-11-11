"""The CUDA 12 driver floor comes from the engine package, not from a constant of setup's own (#4247, #4188 part 1). Nothing here starts an
engine, downloads anything or reads the real profile."""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import setup as S
from strata_control import driver_floor as DF


def package(version: str, need: str | None, *, variant: str = "cuda12", platform: str = "windows-x86_64", backend: str = "cuda"):
    return SimpleNamespace(version=version, platform=platform, backend=backend, variant=variant, min_driver={"nvidia": need} if need else {})


class DriverFloor(unittest.TestCase):
    def test_the_floor_is_the_newest_cuda12_package_of_this_platform(self):
        packages = [package("0.1.49", "570.00"), package("0.1.50", "576.02"), package("0.1.50", "580.65", variant="cuda13"),
                    package("0.1.51", "575.51", platform="linux-x86_64"), package("0.1.52", None)]
        self.assertEqual(DF._floor_of(packages, "windows-x86_64", "cuda12"), "576.02")
        self.assertEqual(DF._floor_of(packages, "linux-x86_64", "cuda12"), "575.51")
        self.assertEqual(DF._floor_of(packages, "windows-x86_64", "cuda13"), "580.65")
        self.assertIsNone(DF._floor_of([package("0.1.50", "580.65", variant="cuda13")], "windows-x86_64", "cuda12"))
        self.assertIsNone(DF._floor_of([package("0.1.50", "580.65", backend="hip")], "windows-x86_64", "cuda13"))

    def test_an_installed_package_answers_before_the_feed_is_asked(self):
        with mock.patch("strata_control.engine_component.scan", return_value=([package("0.1.50", "576.02")], [])), \
                mock.patch.object(DF, "from_feed", side_effect=AssertionError("the feed must not be asked")):
            self.assertEqual(DF.cuda12_floor("windows-x86_64", roots=["/x"], feed_source="https://feed"),
                             ("576.02", "as the installed engine package says"))

    def test_the_feed_answers_when_no_package_is_installed(self):
        feed = SimpleNamespace(engine=(package("0.1.50", "576.02"), package("0.1.50", "580.65", variant="cuda13")))
        with mock.patch("strata_control.engine_component.scan", return_value=([], [])), \
                mock.patch("strata_control.feed.fetch", return_value=feed), \
                mock.patch("strata_control.signing.load_trusted_keys", return_value={"k": b"x"}):
            self.assertEqual(DF.cuda12_floor("windows-x86_64", roots=["/x"], feed_source="https://feed"),
                             ("576.02", "as the signed update feed says"))

    def test_nothing_readable_gives_the_published_figure_and_never_an_error(self):
        with mock.patch("strata_control.engine_component.scan", side_effect=OSError("no folder")), \
                mock.patch("strata_control.feed.fetch", side_effect=RuntimeError("offline")):
            got = DF.cuda12_floor("windows-x86_64", roots=["/x"], feed_source="https://feed")
        self.assertEqual(got[0], DF.FALLBACK)
        self.assertEqual(DF.major(DF.FALLBACK), 575)
        self.assertEqual(DF.cuda12_floor("linux-x86_64"), (DF.FALLBACK, got[1]), "no roots and no feed: the figure, without touching anything")

    def test_a_feed_that_hangs_does_not_hold_setup_up(self):
        def hang(*a, **k):
            time.sleep(3)
        t0 = time.monotonic()
        with mock.patch("strata_control.feed.fetch", side_effect=hang), mock.patch("strata_control.signing.load_trusted_keys", return_value={"k": b"x"}):
            self.assertIsNone(DF.from_feed("https://feed", "windows-x86_64", seconds=0.2))
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_major(self):
        self.assertEqual([DF.major(v) for v in ("576.02", "560", " 575.51 ", "", None, "x")], [576, 560, 575, 0, 0, 0])


class SetupUsesThePackagesFloor(unittest.TestCase):
    def floor(self, figure: str, where: str = "as the installed engine package says"):
        return mock.patch.object(S, "cuda12_min_driver", return_value=(DF.major(figure), figure, where))

    def test_setup_no_longer_says_the_cuda12_engine_works_from_528(self):
        """#4247 #4188: setup said 'it works with driver 528 or newer' while the feed package needed 576.02 and the update refused it."""
        with self.floor("576.02"):
            tk, why = S.cuda_choice([86], None, 556)
        self.assertEqual(tk, 12, "the choice itself is as before: a driver too old for CUDA 13 takes the CUDA 12 engine")
        self.assertNotIn("528", why)
        self.assertNotIn("works with driver", why)

    def test_a_driver_under_the_packages_floor_gets_a_warning_with_the_packages_own_number(self):
        with self.floor("576.02"):
            note = S.cuda12_driver_warning(556)
            self.assertIn("576.02", note)
            self.assertIn("as the installed engine package says", note)
            self.assertIn("556", note)
            self.assertIn("strata update", note)
            self.assertIsNone(S.cuda12_driver_warning(576))
            self.assertIsNone(S.cuda12_driver_warning(591))
            self.assertIsNone(S.cuda12_driver_warning(0), "driver unknown: nothing to say")

    def test_a_package_with_a_lower_floor_lowers_what_setup_says_too(self):
        with self.floor("550.54"):
            self.assertIsNone(S.cuda12_driver_warning(556))
            self.assertIn("550.54", S.cuda12_driver_warning(535))

    def test_the_linux_warning_adds_that_a_locally_built_engine_can_run_older(self):
        with self.floor("575.51"), mock.patch.object(S, "WIN", False):
            self.assertIn("compiles the engine here", S.cuda12_driver_warning(535))
        with self.floor("575.51"), mock.patch.object(S, "WIN", True):
            self.assertNotIn("compiles the engine here", S.cuda12_driver_warning(535))

    def test_the_choice_between_the_engines_is_unchanged(self):
        with self.floor("576.02"):
            self.assertEqual(S.cuda_choice([86], None, 580), (13, None))
            self.assertEqual(S.cuda_choice([86], None, 0), (13, None))
            self.assertEqual(S.cuda_choice([86], None, 500), (13, None), "under NVIDIA's own CUDA 12 line: setup stops with the driver message")
            self.assertEqual(S.cuda_choice([70], None, 0)[0], 12)
            self.assertEqual(S.cuda_choice([86], "12")[0], 12)

    def test_what_stops_setup_is_nvidias_own_line_not_a_promise(self):
        self.assertEqual(S.CUDA12_COMPAT_DRIVER, 528 if S.WIN else 525)
        self.assertFalse(hasattr(S, "CUDA12_MIN_DRIVER"))

    def test_cuda12_min_driver_reads_the_installed_package_and_remembers_it(self):
        with mock.patch.dict(S._CUDA12_FLOORS, clear=True), \
                mock.patch("strata_control.engine_component.scan", return_value=([package("0.1.50", "576.02", platform=_host())], [])):
            self.assertEqual(S.cuda12_min_driver(), (576, "576.02", "as the installed engine package says"))
            self.assertEqual(S.cuda12_min_driver(), (576, "576.02", "as the installed engine package says"))

    def test_a_plain_call_never_asks_the_feed_but_a_setup_run_may(self):
        with mock.patch.dict(S._CUDA12_FLOORS, clear=True), mock.patch("strata_control.engine_component.scan", return_value=([], [])), \
                mock.patch.object(DF, "from_feed", return_value="576.02") as feed, mock.patch.object(S.acquisition, "is_forbidden", return_value=False):
            self.assertEqual(S.cuda12_min_driver()[1], DF.FALLBACK)
            feed.assert_not_called()
            self.assertEqual(S.cuda12_min_driver(feed=True)[1], "576.02")
            self.assertEqual(S.cuda12_min_driver()[1], "576.02", "the answer from the feed is the one kept")
            self.assertEqual(feed.call_count, 1)


def _host() -> str:
    from strata_control import targets
    return targets.host_platform()


if __name__ == "__main__":
    unittest.main()
