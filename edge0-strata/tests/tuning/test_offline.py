"""Offline policy tests (checklist U08).

The checklist asks for two claims that are easy to conflate, and the whole point of these tests is
that they are separate:

* **runtime-only no-download** — the server fetches nothing while serving;
* **strict offline** — nothing in the process tree reaches the network, including children, tools and
  the agent runtime.

A passing runtime-only test says nothing about the second: the first is about the server's own code
paths, and the second is about every process it can start. So they are tested separately, and the
strict tests check the actual environment a child receives rather than the flag that was passed.

The limits are tested too. A policy that claims more than it enforces is worse than one that states
its boundary, so `limits()` is asserted to be non-empty and specific.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import offline as O  # noqa: E402


class TestUrlDecisions(unittest.TestCase):
    """One predicate decides every fetch, so "offline" is not a scattered set of ifs."""

    def test_a_remote_url_is_refused_when_offline(self):
        decision = O.check_url("https://huggingface.co/model.bin", offline=True)
        self.assertFalse(decision)
        self.assertIn("huggingface.co", decision.reason)

    def test_a_remote_url_is_allowed_when_not_offline(self):
        self.assertTrue(O.check_url("https://huggingface.co/model.bin", offline=False))

    def test_a_local_scheme_is_allowed_even_when_offline(self):
        """Talking to this machine is not a download, and forbidding it would break the local API."""
        for url in ("file:///tmp/image.png", "data:image/png;base64,AAAA"):
            decision = O.check_url(url, offline=True)
            self.assertTrue(decision, f"{url} must be allowed offline")

    def test_loopback_over_http_is_allowed_when_offline(self):
        for url in ("http://127.0.0.1:8095/v1/models", "http://localhost:8095/health",
                    "http://[::1]:8095/health"):
            self.assertTrue(O.check_url(url, offline=True), f"{url} is this machine")

    def test_a_bare_path_is_allowed_when_offline(self):
        """A profile usually names a local image as a path, not as a URL."""
        self.assertTrue(O.check_url("/models/photo.png", offline=True))
        self.assertTrue(O.check_url("photo.png", offline=True))

    def test_a_non_http_scheme_is_refused_when_offline(self):
        decision = O.check_url("ftp://example.com/x", offline=True)
        self.assertFalse(decision)
        self.assertIn("scheme", decision.reason)

    def test_require_url_raises_with_the_reason(self):
        with self.assertRaises(O.OfflineViolation) as ctx:
            O.require_url("https://example.com/x", offline=True)
        self.assertIn("would leave this machine", str(ctx.exception))
        self.assertEqual(ctx.exception.url, "https://example.com/x")

    def test_require_url_passes_a_local_url(self):
        O.require_url("http://127.0.0.1:1/x", offline=True)      # must not raise


class TestChildEnvironment(unittest.TestCase):
    """Strict offline must reach code we do not own, which means the environment."""

    def test_the_offline_markers_are_set_for_children(self):
        env = O.child_environment(offline=True, base={})
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["PIP_NO_INDEX"], "1")
        self.assertEqual(env["STRATA_OFFLINE"], "1")

    def test_proxies_are_emptied_so_an_accidental_call_fails_fast(self):
        env = O.child_environment(offline=True,
                                  base={"HTTPS_PROXY": "http://corp-proxy:8080"})
        self.assertEqual(env["HTTPS_PROXY"], "")
        self.assertEqual(env["NO_PROXY"], "*")

    def test_the_parent_environment_is_not_mutated(self):
        base = {"HTTPS_PROXY": "http://corp-proxy:8080", "PATH": "/usr/bin"}
        O.child_environment(offline=True, base=base)
        self.assertEqual(base["HTTPS_PROXY"], "http://corp-proxy:8080",
                         "the caller's environment must be untouched")

    def test_a_marker_from_a_previous_run_is_removed_when_not_offline(self):
        """Otherwise a later server mysteriously refuses to fetch anything."""
        env = O.child_environment(offline=False, base={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin"})
        self.assertNotIn("HF_HUB_OFFLINE", env)
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_the_real_environment_is_used_by_default(self):
        env = O.child_environment(offline=True)
        self.assertEqual(env["STRATA_OFFLINE"], "1")
        self.assertIn("PATH", env, "the child still needs a PATH")


class TestProfilePreflight(unittest.TestCase):
    """Refusing before the server starts beats failing on the first request."""

    def test_a_plain_profile_is_strictly_offline_capable(self):
        self.assertEqual(O.ensure_strictly_offline({"model": "x", "args": []}), [])

    def test_mcp_servers_are_named_as_a_reason(self):
        reasons = O.ensure_strictly_offline({"mcp_servers": {"a": {}, "b": {}}})
        self.assertTrue(reasons)
        self.assertIn("2 MCP server", reasons[0])

    def test_a_before_load_command_is_named_as_a_reason(self):
        reasons = O.ensure_strictly_offline({"before_load": "curl https://example.com/x"})
        self.assertTrue(reasons)
        self.assertIn("before_load", reasons[0])

    def test_remote_image_sources_are_named_as_a_reason(self):
        reasons = O.ensure_strictly_offline({"images": {"allowed_hosts": ["cdn.example.com"]}})
        self.assertTrue(reasons)
        self.assertIn("images.allowed_hosts", reasons[0])

    def test_a_local_image_section_is_not_a_reason(self):
        self.assertEqual(O.ensure_strictly_offline({"images": {"size": 512}}), [])


class TestTheLimitIsStated(unittest.TestCase):
    """A policy that claims more than it enforces is worse than one that names its boundary."""

    def test_the_limits_are_specific_and_non_empty(self):
        limits = O.limits()
        self.assertTrue(limits)
        joined = " ".join(limits).lower()
        self.assertIn("socket", joined, "the socket case is the one that matters")
        self.assertIn("sandbox", joined)

    def test_the_report_separates_the_two_claims(self):
        report = O.report(offline=True, cfg={"model": "x"})
        self.assertTrue(report["strict_offline"])
        self.assertTrue(report["runtime_only_no_download"],
                        "runtime-only is true independently of strict offline")
        self.assertTrue(report["enforced"])
        self.assertTrue(report["not_enforced"], "the limits travel with the claim")

    def test_the_report_of_a_non_offline_run_does_not_claim_enforcement(self):
        report = O.report(offline=False, cfg={})
        self.assertFalse(report["strict_offline"])
        self.assertEqual(report["not_enforced"], [])
        self.assertTrue(report["runtime_only_no_download"])

    def test_the_report_says_which_profile_parts_block_strict_offline(self):
        report = O.report(offline=True, cfg={"mcp_servers": {"a": {}}})
        self.assertTrue(report["profile_needs_network"])


class TestTheServerUsesThePolicy(unittest.TestCase):
    """A guard that is not wired in is decoration, so the wiring itself is asserted."""

    def setUp(self):
        self.source = (Path(__file__).resolve().parents[2] / "serve" / "server.py").read_text(
            encoding="utf-8")

    def test_the_image_path_consults_the_policy_or_the_flag(self):
        # The reviewed server already refuses a remote image URL when offline; what matters here is
        # that the refusal exists on both image paths rather than one.
        self.assertGreaterEqual(self.source.count("offline mode cannot download an image URL"), 2,
                                "both image paths must refuse a remote fetch")

    def test_child_env_stamps_the_policy(self):
        self.assertIn("offline_policy.OFFLINE_ENV", self.source,
                      "children must be given the offline environment")

    def test_every_child_environment_call_passes_the_policy(self):
        """A single call site that forgot the flag would leak network access to that child."""
        offenders = []
        for index, line in enumerate(self.source.splitlines(), start=1):
            if "child_env(" in line and "def child_env" not in line:
                if "offline=" not in line:
                    offenders.append(f"line {index}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "every child_env call must pass the server's offline state: "
                         + "; ".join(offenders))

    def test_the_strict_offline_preflight_runs_before_serving(self):
        self.assertIn("ensure_strictly_offline", self.source)


if __name__ == "__main__":
    unittest.main()
