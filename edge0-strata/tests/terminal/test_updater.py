"""The component updater end to end against a LOCAL feed: a folder of signed archives served by a loopback HTTP server, stub
engines behind real schema-2 packages, the real start path for the test start.

Proves: an update downloads (with resume), verifies signature and every hash, installs side by side, test-starts the new
engine from a staging folder through the owner runtime, and only then switches; a failing test start leaves the old engine
working; the previous version is kept and can be returned to; saved presets are marked, not touched; models are never
fetched; nothing runs while an engine serves; and a start/serve process (STRATA_NO_ACQUIRE) cannot reach any of it."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import acquisition, engine_component as ec, feed as feedmod, preset_marks, signing, updater
from strata_control.targets import host_platform
from tests.terminal import update_fixture as fx

PLATFORM = host_platform()
WINDOWS = os.name == "nt"


class UpdaterCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-updater-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.components = self.root / "components"
        self.state_base = self.root / "state"
        self.release = self.root / "release"
        self.release.mkdir()
        for name in ("STRATA_NO_ACQUIRE", "STRATA_API_KEY", "STRATA_REQUIRE_SIGNED_COMPONENTS", "STRATA_UPDATE_TRUSTED_KEYS"):
            patch = mock.patch.dict(os.environ)
            patch.start()
            self.addCleanup(patch.stop)
            os.environ.pop(name, None)
        env = mock.patch.dict(os.environ, {
            "STRATA_PROFILE_DIR": str(self.root / "profiles"), "STRATA_RUNTIME_DIR": str(self.root / "runtime"),
            "STRATA_COMPONENTS_DIR": str(self.components), "STRATA_ENGINE_BACKEND": "cuda",
            "FAKE_ENGINE_MARK": str(self.root / "ran.txt")})
        env.start()
        self.addCleanup(env.stop)
        for target, value in ((ec, "cuda_archs_present"), (ec, "nvidia_driver_version")):
            patch = mock.patch.object(target, value, return_value=["sm_86"] if value == "cuda_archs_present" else "610.47")
            patch.start()
            self.addCleanup(patch.stop)
        ec.register_signature_verifier(None)               # a verifier an earlier test file left behind (it holds the shipped keys only) must not decide here
        self.addCleanup(ec.register_signature_verifier, None)
        # the tests sign with throwaway keys that they hand over explicitly: the release key shipped in update_keys.json is not in play here
        # (ReleaseKeys below checks the shipped file itself)
        patch = mock.patch.object(signing, "BUILTIN_KEYS", self.root / "no-shipped-keys.json")
        patch.start()
        self.addCleanup(patch.stop)
        self.key = fx.make_key()
        self.trusted = fx.trusted_of(self.key)
        # the engine a test start runs is a child process: it reads the shipped release key too, so the test key is passed the way a
        # tester build passes its key
        self.test_keys = self.root / "test-trust.json"
        self.test_keys.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        patch = mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(self.test_keys)})
        patch.start()
        self.addCleanup(patch.stop)
        self.hardware = {"platform": PLATFORM, "backend": "cuda", "runtime": None, "nvidia_driver": "610.47",
                         "cuda_archs": ["sm_86"], "gfx": []}
        self.models = self.root / "models"
        self.models.mkdir()
        (self.models / "model.gguf").write_bytes(b"GGUF" + os.urandom(2048))
        self.model_digest = hashlib.sha256((self.models / "model.gguf").read_bytes()).hexdigest()
        self.preset = self.state_base / "tuning" / "presets" / "iq3-128k.json"
        self.preset.parent.mkdir(parents=True)
        self.preset.write_text('{"name": "IQ3_S 128K full speed", "settings": {"expert_cache": 1500}}\n')
        self.preset_bytes = self.preset.read_bytes()
        self._run_config()
        self.server = fx.FeedServer(self.release)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.sequence = 0
        self.addCleanup(self._stop_runtime)

    def _stop_runtime(self):
        from strata_control import runtime
        try:
            if runtime._read_state("updtest") is not None:
                runtime.stop("updtest", timeout=10)
        except Exception:                                      # noqa: BLE001
            pass

    def _run_config(self):
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (tokenizer / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
        (tokenizer / "merges.txt").write_text("a b")
        (tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": "/some/old/engine", "backend": "cuda", "tokenizer": str(tokenizer), "args": [],
                                           "cwd": str(self.root), "model_name": "fake", "host": "127.0.0.1", "port": 0}))

    # ------------------------------------------------------------------ helpers

    def install_current(self, version="0.1.45", **kwargs):
        folder = fx.build_package(self.root / f"inst-{version}", version, **kwargs)
        target = self.components / "engine" / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, target)
        return target

    def publish(self, *versions, sequence=None, key=None, **options):
        """Sign + archive each (version, engine_source[, kwargs]) and write a signed feed. Returns the package folders."""
        shutil.rmtree(self.release, ignore_errors=True)
        self.release.mkdir()
        out = self.release / "engine"
        out.mkdir()
        key = key or self.key
        self.sequence = sequence or self.sequence + 1
        folders = []
        for item in versions:
            version, source, extra = (item + ({},))[:3] if len(item) == 2 else item
            folder = fx.build_package(self.root / f"pub-{self.sequence}-{version}", version, engine_source=source, key=extra.pop("sign_with", key), **extra)
            folders.append(folder)
            from feed_tool import pack_engine
            pack_engine(folder, out)
        from feed_tool import build_feed
        build_feed(self.release, key=key, sequence=self.sequence, engine_dir=out, **options)
        return folders

    def ctx(self, **over):
        values = dict(components=self.components, source=self.server.url + "/strata-feed.json", trusted=self.trusted,
                      hardware=self.hardware, run_config=str(self.config), state_base=self.state_base, poll=0.05,
                      test_timeout=60, serving=lambda: [], environ=dict(os.environ))
        values.update(over)
        events = values.pop("events", None)
        context = updater.Context(**values)
        if events is not None:
            context.emit = events.append
        return context

    def engine_folders(self):
        folder = self.components / "engine"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []

    def resolved(self):
        return ec.resolve(self.components, backend="cuda").version

    def assert_user_data_untouched(self):
        self.assertEqual(self.preset.read_bytes(), self.preset_bytes, "a saved preset must not change")
        self.assertEqual(hashlib.sha256((self.models / "model.gguf").read_bytes()).hexdigest(), self.model_digest)

    def archive_requests(self):
        return [r for r in self.server.requests if r[1].endswith((".tar.gz", ".zip"))]


class EngineUpdate(UpdaterCase):
    def test_check_reads_only_the_feed(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        report = updater.check(self.ctx())
        self.assertEqual(report["engine"]["status"], "available")
        self.assertEqual((report["engine"]["installed"]["version"], report["engine"]["offered"]["version"]), ("0.1.45", "0.1.46"))
        self.assertEqual([r[1] for r in self.server.requests], ["/strata-feed.json", "/strata-feed.json.sig"])
        self.assertEqual(self.engine_folders(), [f"0.1.45-{PLATFORM}-cuda-cuda13"])

    def test_update_downloads_verifies_test_starts_then_switches(self):
        old = self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        events = []
        out = updater.run_update(self.ctx(events=events), control=False)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"], result["changed"]), (0, "updated", True), out)
        self.assertEqual(result["test_start"]["mode"], "full")
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertEqual(sorted(self.engine_folders()), [f"0.1.45-{PLATFORM}-cuda-cuda13", f"0.1.46-{PLATFORM}-cuda-cuda13"],
                         "the previous version stays for a manual rollback")
        # the test start ran the staged copy, not a file in the live folder
        ran = (self.root / "ran.txt").read_text().splitlines()[0]
        self.assertIn(os.sep + ".s" + os.sep, ran)
        phases = [e.get("phase") for e in events if e.get("event") == "phase"]
        phases = [p for i, p in enumerate(phases) if i == 0 or p != "test" or phases[i - 1] != "test"]     # progress lines of the test start repeat it
        self.assertEqual(phases, ["feed", "download", "verify", "verify", "test", "switch", "done"])
        progress = [e for e in events if e.get("event") == "progress"]
        self.assertTrue(progress and progress[-1]["percent"] == 100 and progress[-1]["done"] == progress[-1]["total"])
        self.assertEqual(sorted(p.name for p in (self.components / ".s").iterdir()), [], "staging is cleaned")
        self.assertEqual(list((self.components / ".updates" / "partial").iterdir()), [], "the archive is removed after the switch")
        state = json.loads((self.components / ".updates" / "state.json").read_text())
        self.assertEqual(state["engine"]["previous"], old.name)
        self.assertEqual(state["feed"]["sequence"], self.sequence)
        # presets are marked for the short re-check and kept as they were
        marks = preset_marks.pending(self.state_base)
        self.assertEqual(len(marks), 1)
        self.assertEqual((marks[0]["from"]["version"], marks[0]["to"]["version"]), ("0.1.45", "0.1.46"))
        self.assert_user_data_untouched()
        # nothing but the feed and the engine archive was requested: no model, no app
        self.assertEqual({r[1] for r in self.server.requests}, {"/strata-feed.json", "/strata-feed.json.sig", *{r[1] for r in self.archive_requests()}})
        self.assertEqual(len(self.archive_requests()), 1)
        # a second update with nothing newer changes nothing and downloads nothing
        again = updater.run_update(self.ctx(), control=False)
        self.assertEqual(again["results"][0]["status"], "current")
        self.assertEqual(len(self.archive_requests()), 1)

    def test_the_next_update_drops_versions_older_than_the_previous_one(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(), control=False)
        self.assertEqual(out["results"][0]["status"], "updated")
        self.assertEqual(self.engine_folders(), [f"0.1.46-{PLATFORM}-cuda-cuda13", f"0.1.47-{PLATFORM}-cuda-cuda13"])
        self.assertEqual(self.resolved(), "0.1.47")

    def test_a_failing_test_start_keeps_the_old_engine(self):
        self.install_current()
        self.publish(("0.1.47", fx.BAD_ENGINE))
        out = updater.run_update(self.ctx(), control=False)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"]), (6, "failed"), out)
        self.assertIn("did not start correctly on this computer", result["error"])
        self.assertIn("unchanged", result["error"])
        self.assertNotIn("server_exited", result["error"], "the engine's own words stay out of the sentence")
        self.assertIn("strata update --force", result["error"])
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(self.engine_folders(), [f"0.1.45-{PLATFORM}-cuda-cuda13"], "the failed engine was never put in the live folder")
        self.assertEqual(sorted(p.name for p in (self.components / ".s").iterdir()), [])
        self.assertEqual(preset_marks.pending(self.state_base), [])
        self.assert_user_data_untouched()
        # it is not offered again until forced, and the archive is not downloaded twice
        again = updater.run_update(self.ctx(), control=False)
        self.assertEqual(again["results"][0]["status"], "failed-before", again)
        self.assertIn("--force", again["results"][0]["note"])
        forced = updater.run_update(self.ctx(force=True), control=False)
        self.assertEqual(forced["results"][0]["status"], "failed")
        self.assertEqual(len(self.archive_requests()), 1, "the verified archive was kept and reused")
        # a fixed build of the same version is a different manifest and is tried
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        fixed = updater.run_update(self.ctx(), control=False)
        self.assertEqual(fixed["results"][0]["status"], "updated", fixed)
        self.assertEqual(self.resolved(), "0.1.47")

    def test_an_interrupted_download_resumes(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE, {"padding": 3_000_000}))
        self.server.cut[".tar.gz"] = 700_000
        first = updater.run_update(self.ctx(), control=False)
        self.assertEqual((first["exit_code"], first["results"][0]["status"]), (6, "failed"))
        self.assertIn("resumes it", first["results"][0]["error"])
        self.assertEqual(self.resolved(), "0.1.45", "an interrupted update keeps the old engine working")
        partial = list((self.components / ".updates" / "partial").iterdir())
        self.assertEqual(len(partial), 1)
        kept = partial[0].stat().st_size
        self.assertTrue(0 < kept <= 700_000, kept)
        second = updater.run_update(self.ctx(), control=False)
        self.assertEqual(second["results"][0]["status"], "updated", second)
        ranges = [r[2] for r in self.archive_requests()]
        self.assertEqual(ranges[0], None)
        self.assertEqual(ranges[1], f"bytes={kept}-", "the second try asked only for the missing bytes")
        self.assertEqual(self.resolved(), "0.1.46")

    def test_a_download_that_is_not_the_signed_hash_is_refused(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        archive = next((self.release / "engine").glob("*.tar.gz"))
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 0x55                                  # same size, one byte different
        archive.write_bytes(bytes(data))
        out = updater.run_update(self.ctx(), control=False)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"]), (5, "failed"))
        self.assertIn("SHA-256 in the signed feed", result["error"])
        self.assertEqual(list((self.components / ".updates" / "partial").iterdir()), [], "bytes that fail the hash are deleted")
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertFalse((self.root / "ran.txt").exists(), "a package that failed its hash must never run")

    def test_a_package_the_feed_did_not_describe_is_refused(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        entry = next((self.release / "engine").glob("*.entry.json"))
        data = json.loads(entry.read_text())
        data["manifest_sha256"] = "c" * 64
        entry.write_text(json.dumps(data))
        from feed_tool import build_feed
        build_feed(self.release, key=self.key, sequence=self.sequence + 1, engine_dir=self.release / "engine")
        out = updater.run_update(self.ctx(), control=False)
        self.assertEqual(out["exit_code"], 5)
        self.assertIn("manifest is not the one the signed feed names", out["results"][0]["error"])
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertFalse((self.root / "ran.txt").exists())

    def test_unsigned_and_wrongly_signed_packages_are_refused(self):
        self.install_current()
        other = fx.make_key()
        for label, kind in (("not signed", None), ("trust", other)):
            with self.subTest(label=label):
                shutil.rmtree(self.release, ignore_errors=True)
                self.release.mkdir()
                self.sequence += 1
                folder = fx.build_package(self.root / f"x-{self.sequence}", "0.1.46", key=kind)
                from feed_tool import build_feed, pack_engine
                (self.release / "engine").mkdir()
                pack_engine(folder, self.release / "engine")
                build_feed(self.release, key=self.key, sequence=self.sequence, engine_dir=self.release / "engine")
                out = updater.run_update(self.ctx(), control=False)
                self.assertEqual(out["exit_code"], 5, out)
                self.assertIn(label, out["results"][0]["error"])
                self.assertEqual(self.resolved(), "0.1.45")
                self.assertFalse((self.root / "ran.txt").exists())

    def test_a_downgrade_is_never_installed(self):
        self.install_current("0.1.45")
        self.publish(("0.1.44", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(), control=False)
        result = out["results"][0]
        self.assertEqual((result["status"], result["changed"]), ("current", False))
        self.assertIn("never downgrades", updater.check(self.ctx())["engine"]["note"])
        self.assertEqual(self.archive_requests(), [])
        self.assertEqual(self.resolved(), "0.1.45")

    def test_a_replayed_older_feed_is_refused(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE), sequence=7)
        self.assertEqual(updater.run_update(self.ctx(), control=False)["results"][0]["status"], "updated")
        self.publish(("0.1.46", fx.GOOD_ENGINE), sequence=3)
        with self.assertRaises(feedmod.FeedDowngrade):
            updater.run_update(self.ctx(), control=False)
        with self.assertRaises(feedmod.FeedDowngrade):
            updater.check(self.ctx())

    def test_nothing_is_replaced_while_an_engine_serves(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(serving=lambda: [{"profile": "app-1", "state": "ready"}]), control=False)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (7, "failed"))
        self.assertIn("app-1 is serving", out["results"][0]["error"])
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertFalse((self.root / "ran.txt").exists(), "no test start while another engine serves")
        # the download was not wasted: the next try reuses the verified archive
        done = updater.run_update(self.ctx(), control=False)
        self.assertEqual(done["results"][0]["status"], "updated")
        self.assertEqual(len(self.archive_requests()), 1)

    def test_waiting_for_the_engine_to_stop(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        answers = [[{"profile": "app-1"}], [{"profile": "app-1"}], [], []]
        events = []
        out = updater.run_update(self.ctx(serving=lambda: answers.pop(0) if answers else [], wait=30, events=events), control=False)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        self.assertTrue(any(e.get("event") == "waiting" for e in events))

    def test_manual_rollback_returns_to_the_previous_engine_and_is_not_undone_by_the_next_update(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        out = updater.rollback_engine(self.ctx())
        self.assertEqual((out["status"], out["from"], out["to"]), ("rolled-back", "0.1.46", "0.1.45"))
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(ec.status(self.components, backend="cuda")["pinned"], f"0.1.45-{PLATFORM}-cuda-cuda13")
        self.assertEqual(preset_marks.pending(self.state_base), [], "back on the engine the presets were measured on: nothing owed")
        again = updater.run_update(self.ctx(), control=False)
        self.assertEqual(again["results"][0]["status"], "skipped")
        self.assertEqual(self.resolved(), "0.1.45")
        forced = updater.run_update(self.ctx(force=True), control=False)
        self.assertEqual(forced["results"][0]["status"], "updated", forced)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assert_user_data_untouched()

    def test_rollback_without_a_previous_engine_says_so(self):
        self.install_current()
        with self.assertRaisesRegex(updater.UpdateError, "no previous engine"):
            updater.rollback_engine(self.ctx())

    def test_first_run_installs_the_engine_for_this_pc_without_a_model(self):
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.object(updater, "discover_run_config", return_value=None):          # whatever this machine has installed
            out = updater.run_update(self.ctx(run_config=None), control=False)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"]), (0, "updated"), out)
        self.assertEqual(result["test_start"]["mode"], "binary-only")
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertEqual(preset_marks.pending(self.state_base), [], "no earlier engine, so nothing to re-check")

    def test_a_broken_installed_engine_is_replaced_by_a_good_one(self):
        folder = self.install_current("0.1.46")
        library = next(p for p in folder.iterdir() if p.name.startswith(("cudart", "libcudart")))
        body = bytearray(library.read_bytes())
        body[5] ^= 0xFF
        library.write_bytes(bytes(body))
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        self.assertEqual(updater.check(self.ctx())["engine"]["status"], "install")
        out = updater.run_update(self.ctx(), control=False)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        self.assertEqual(self.resolved(), "0.1.46")

    def test_the_package_for_another_gpu_is_not_offered(self):
        self.publish(("0.1.46", fx.GOOD_ENGINE, {"archs": ("sm_90",)}))
        out = updater.run_update(self.ctx(hardware={**self.hardware, "cuda_archs": ["sm_61"]}), control=False)
        self.assertEqual(out["results"][0]["status"], "none-for-this-pc")
        self.assertIn("sm_61", " ".join(out["results"][0]["reasons"]))
        self.assertEqual(self.archive_requests(), [])

    def test_a_process_that_forbids_acquiring_cannot_update(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            with self.assertRaises(acquisition.AcquisitionRefused):
                updater.run_update(self.ctx(), control=False)
            with self.assertRaises(acquisition.AcquisitionRefused):
                updater.check(self.ctx())
        self.assertEqual(self.server.requests, [], "not even the feed is read where acquiring is forbidden")

    def test_a_started_engine_check_does_not_import_the_updater(self):
        """The start path (engine_component, runtime) must not reach the update machinery or the network code."""
        import subprocess
        code = ("import sys; import strata_control.engine_component, strata_control.runtime, strata_control.acquisition; "
                "import strata_cli.cli as c; c.parser(); "
                "print(sorted(m for m in sys.modules if m in ('strata_control.updater','strata_control.transfer','strata_control.feed')))")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(fx.ROOT))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "[]")


class StartPathSignature(UpdaterCase):
    """A start verifies a signed package with the keys this build trusts, without anything registering a verifier first."""

    def setUp(self):
        super().setUp()
        self.pub = self.root / "feed-key.pub.json"
        self.pub.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        patch = mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(self.pub)})
        patch.start()
        self.addCleanup(patch.stop)
        ec.register_signature_verifier(None)

    def test_signed_packages_verify_and_a_forged_signature_is_refused_on_every_start(self):
        folder = fx.build_package(self.root / "signed", "0.1.46", key=self.key)
        target = self.components / "engine" / folder.name
        target.parent.mkdir(parents=True)
        shutil.copytree(folder, target)
        self.assertEqual(ec.verify_signature(ec.read_manifest(target)), "verified")
        manifest = json.loads((target / ec.MANIFEST).read_text())
        manifest["notes"] = "changed after signing"
        (target / ec.MANIFEST).write_text(json.dumps(manifest))
        ec.register_signature_verifier(None)
        with self.assertRaises(ec.EngineMissing) as caught:
            ec.resolve(self.components, backend="cuda")
        self.assertIn("signature", str(caught.exception))

    def test_without_trusted_keys_a_signed_package_is_unverified_and_required_signing_refuses_it(self):
        folder = fx.build_package(self.root / "signed", "0.1.46", key=self.key)
        with mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": ""}):
            ec.register_signature_verifier(None)
            self.assertEqual(ec.verify_signature(ec.read_manifest(folder)), "unverified")
            with self.assertRaises(ec.IntegrityProblem):
                ec.verify_signature(ec.read_manifest(folder), {ec.ENV_REQUIRE_SIGNED: "1"})


class LightTestStart(unittest.TestCase):
    def test_the_test_start_drops_the_big_context_and_the_picture_reader(self):                # #3842
        args = ["--serve", "--pack", "p", "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768", "--vision",
                "--vision-tokens", "768", "--spec", "4"]
        self.assertEqual(updater.light_test_args(args), ["--serve", "--pack", "p", "--max-context", "4096", "--kv", "int8", "--spec", "4"])
        self.assertEqual(updater.light_test_args(["--max-context", "2048", "--vision", "cpu", "--mtp", "x"]),
                         ["--max-context", "2048", "--mtp", "x"])
        self.assertEqual(updater.light_test_args(["--vision"]), [])
        self.assertEqual(updater.light_test_args(["--max-context", "abc"]), ["--max-context", "abc"])


class LazyModel(UpdaterCase):
    """A server that is "ready" with its model unloaded has not generated anything yet; the test start forces the load and the probe."""

    def run_default(self, started, verified, tail=None):
        from strata_control import runtime
        self.install_current()
        folder = fx.build_package(self.root / "lazy", "0.1.46", key=self.key)
        stage = self.root / "stage"
        (stage / "engine").mkdir(parents=True)
        shutil.copytree(folder, stage / "engine" / folder.name)
        component = ec.read_manifest(stage / "engine" / folder.name)
        with mock.patch.object(runtime, "start", return_value=started), mock.patch.object(runtime, "verify", return_value=verified) as verify,                 mock.patch.object(runtime, "stop", return_value={}):
            with mock.patch.object(updater, "_log_tail", return_value=tail or ""):
                result = updater.default_test_start(stage, component, self.ctx())
        return result, verify

    def test_an_unloaded_ready_server_is_made_to_answer_before_the_engine_passes(self):
        result, verify = self.run_default({"ok": True, "state": "ready", "generation_ready": False},
                                          {"ok": True, "state": "ready", "generation_ready": True})
        self.assertEqual(result["generation_ready"], True)
        self.assertEqual(verify.call_args.kwargs["load"], True)

    def test_an_engine_that_cannot_answer_does_not_pass(self):
        with self.assertRaisesRegex(updater.TestStartFailed, "before it answered a test question"):
            self.run_default({"ok": True, "state": "ready", "generation_ready": False}, {"ok": False, "state": "failed", "error": "probe_generation_no_tokens"})
    def expired(self, phase, words, busy):
        # #3683: the engine log always mentions VRAM, so a wait that expired used to be blamed on a busy graphics card
        tail = " | strata serve: 358 MiB of VRAM free with everything loaded"
        with self.assertRaises(updater.TestStartFailed) as raised:
            self.run_default({"ok": False, "state": "starting", "error": "startup_wait_expired", "phase": phase}, {}, tail=tail)
        text = str(raised.exception)
        self.assertIn(words, text)
        self.assertEqual(raised.exception.busy, busy)
        self.assertNotIn("graphics card or memory looks busy", text)
        self.assertNotIn("358 MiB", text)
        self.assertIn("test-start-", text)                                # where the kept log is

    def test_an_expired_start_with_the_model_loaded_says_it_did_not_answer(self):
        self.expired("verifying_generation", "did not answer one token", False)

    def test_an_expired_start_with_an_unmatched_ready_file_says_so(self):
        self.expired("ready_file_not_matched", "could not match it to the engine process", False)

    def test_an_expired_start_that_was_still_loading_says_so(self):
        self.expired("loading", "still loading the model", True)

    def test_the_test_start_shows_progress_while_it_waits(self):
        self.install_current()
        folder = fx.build_package(self.root / "lazy", "0.1.46", key=self.key)
        stage = self.root / "stage"
        (stage / "engine").mkdir(parents=True)
        shutil.copytree(folder, stage / "engine" / folder.name)
        component = ec.read_manifest(stage / "engine" / folder.name)
        events = []
        ctx = self.ctx()
        ctx.emit = events.append

        def fake_start(name, **kw):
            for status in ({"state": "starting", "phase": "loading"}, {"state": "starting", "phase": "loading"},
                           {"state": "starting", "phase": "verifying_generation"}, {"state": "ready"}):
                kw["on_status"](status)
            return {"ok": True, "state": "ready", "generation_ready": True}

        from strata_control import runtime
        with mock.patch.object(runtime, "start", side_effect=fake_start), mock.patch.object(runtime, "stop", return_value={}):
            updater.default_test_start(stage, component, ctx)
        lines = [e["message"] for e in events if e.get("phase") == "test"]
        self.assertEqual(len(lines), 2)                                       # one per phase change, not one per poll
        self.assertIn("loading the model", lines[0])
        self.assertIn("answer one token", lines[1])


class OtherComponents(UpdaterCase):
    def build_with_app(self, version="0.2.0", tamper=False):
        public, secret = fx.minisign_pair()
        installer = os.urandom(5000)
        (self.release / "app").mkdir(parents=True, exist_ok=True)
        (self.release / "app" / "Strata_setup.bin").write_bytes(installer)
        sig = fx.minisign_sign(installer + (b"x" if tamper else b""), secret)
        app = {"version": version, "notes": "Faster starts.", "pub_date": "2026-10-15T00:00:00Z",
               "platforms": {feedmod.TAURI_TARGETS[PLATFORM]: {"url": "app/Strata_setup.bin", "signature": sig,
                                                              "sha256": hashlib.sha256(installer).hexdigest(), "size": len(installer)}}}
        (self.root / "tauri.json").write_text(json.dumps(app))
        from feed_tool import build_feed
        (self.release / "engine").mkdir(exist_ok=True)
        build_feed(self.release, key=self.key, sequence=self.sequence + 1, app_json=self.root / "tauri.json")
        self.sequence += 1
        return public, installer

    def test_the_app_installer_is_downloaded_and_its_signature_checked(self):
        public, installer = self.build_with_app()
        ctx = self.ctx(app_version="0.1.0", environ={**os.environ, "STRATA_APP_UPDATE_PUBKEY": public})
        report = updater.check(ctx)
        self.assertEqual((report["app"]["status"], report["app"]["offered"]), ("available", "0.2.0"))
        out = updater.run_update(ctx, engine=False, control=False, app=True)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"]), (0, "downloaded"), out)
        self.assertEqual(Path(result["path"]).read_bytes(), installer)
        self.assertEqual((self.release / "app" / "latest.json").exists(), True, "the Tauri updater's file is written beside the feed")
        self.assertEqual(self.engine_folders(), [], "an app-only update downloads 0 engine bytes")
        self.assertEqual(self.archive_requests(), [])
        self.assertTrue(all(r[1].startswith(("/strata-feed", "/app/")) for r in self.server.requests))

    def test_an_installer_with_a_bad_signature_is_deleted(self):
        public, _ = self.build_with_app(tamper=True)
        ctx = self.ctx(app_version="0.1.0", environ={**os.environ, "STRATA_APP_UPDATE_PUBKEY": public})
        out = updater.run_update(ctx, engine=False, control=False, app=True)
        self.assertEqual(out["exit_code"], 5)
        self.assertIn("signature", out["results"][0]["error"])
        self.assertFalse((self.components / ".updates" / "app").exists() and any((self.components / ".updates" / "app").iterdir()))

    def test_no_app_key_no_app_download(self):
        self.build_with_app()
        ctx = self.ctx(app_version="0.1.0", environ={k: v for k, v in os.environ.items() if k != "STRATA_APP_UPDATE_PUBKEY"})
        with mock.patch.object(updater, "app_public_key", return_value=None):
            out = updater.run_update(ctx, engine=False, control=False, app=True)
        self.assertEqual(out["exit_code"], 5)
        self.assertEqual([r for r in self.server.requests if r[1].startswith("/app/")], [])

    def test_control_is_staged_run_once_and_pointed_at(self):
        bundle = self.root / "control-bundle"
        (bundle / "strata_control").mkdir(parents=True)
        (bundle / "strata.py").write_text("print('strata 0.1.46 (python stub)')\n")
        (bundle / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(bundle, self.release / "control", version="0.1.46", platform=PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        active = updater.active_control(self.components)
        self.assertEqual(active["version"], "0.1.46")
        self.assertTrue((self.components / "control" / "0.1.46" / "strata.py").is_file())
        again = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual(again["results"][0]["status"], "current")

    def test_a_control_that_cannot_run_is_not_switched_to(self):
        bundle = self.root / "control-bad"
        bundle.mkdir()
        (bundle / "strata.py").write_text("import sys; sys.exit(9)\n")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(bundle, self.release / "control", version="0.1.46", platform=PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (6, "failed"))
        self.assertIsNone(updater.active_control(self.components))
        self.assertFalse((self.components / "control" / "0.1.46").exists())

    @unittest.skipIf(WINDOWS, "symbolic links need no privilege only on POSIX")
    def test_a_control_bundle_may_link_inside_itself_and_nowhere_else(self):
        bundle = self.root / "control-links"
        (bundle / "python" / "lib").mkdir(parents=True)
        (bundle / "strata.py").write_text("print('strata 0.1.46 (stub)')\n")
        (bundle / "python" / "lib" / "libpython3.12.so").write_bytes(b"lib")
        os.symlink("libpython3.12.so", bundle / "python" / "lib" / "libpython3.so")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        entry = pack_control(bundle, self.release / "control", version="0.1.46", platform=PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual(out["results"][0]["status"], "updated", out)
        link = self.components / "control" / "0.1.46" / "python" / "lib" / "libpython3.so"
        self.assertTrue(link.is_symlink() and os.readlink(link) == "libpython3.12.so")
        self.assertEqual(link.read_bytes(), b"lib")

    @unittest.skipIf(WINDOWS, "symbolic links need no privilege only on POSIX")
    def test_a_control_archive_with_a_link_out_of_the_package_is_refused(self):
        import tarfile
        from feed_tool import build_feed
        (self.release / "control").mkdir()
        archive = self.release / "control" / "evil.tar.gz"
        good = self.root / "strata.py"
        good.write_text("print('strata 0.1.46')\n")
        for target in ("/etc/passwd", "../../outside", "a/../../../outside"):
            with self.subTest(target=target):
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(good, arcname="strata.py")
                    info = tarfile.TarInfo("python/evil")
                    info.type, info.linkname = tarfile.SYMTYPE, target
                    tar.addfile(info)
                data = archive.read_bytes()
                (self.release / "control" / "evil.entry.json").write_text(json.dumps({
                    "version": "0.1.46", "platform": PLATFORM, "format": "tar.gz", "url": "control/evil.tar.gz", "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "installed_bytes": 100, "entry": "strata.py"}))
                build_feed(self.release, key=self.key, sequence=self.sequence + 1, control_dir=self.release / "control")
                self.sequence += 1
                out = updater.run_update(self.ctx(), engine=False, control=True)
                self.assertEqual(out["exit_code"], 5, out)
                self.assertRegex(out["results"][0]["error"], "symbolic link")
                self.assertIsNone(updater.active_control(self.components))


class Cli(UpdaterCase):
    """`strata update`, `strata components update|rollback` against a feed in a folder, with the test key file the way the
    owner's local setup uses it (STRATA_UPDATE_TRUSTED_KEYS)."""

    def setUp(self):
        super().setUp()
        self.pub = self.root / "feed-key.pub.json"
        self.pub.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        patch = mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(self.pub)})
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(updater, "detect_hardware", return_value=dict(self.hardware))
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(updater, "serving_now", return_value=[])
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *args):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from strata_cli.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def common(self):
        return ["--feed", str(self.release), "--components-dir", str(self.components), "--model-config", str(self.config)]

    def test_cached_update_notice_is_dropped_when_versions_change(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        self.assertEqual(self.cli("update", "--check", *self.common())[0], 0)
        notice = "update available (engine 0.1.46): run `strata update`"
        self.assertEqual(ec.cached_notice(self.components), notice)
        state_file = self.components / ec.UPDATE_STATE
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["last_check"]["engine_installed"], "0.1.45")
        # a new app brings a new Control: the old result is not shown again
        with mock.patch.object(ec, "runtime_version", return_value="9.9.9"):
            self.assertIsNone(ec.cached_notice(self.components))
        # the installed engine changed since the check (an update from elsewhere, a rollback)
        with mock.patch.object(ec, "newest_installed_version", return_value="0.1.44"):
            self.assertIsNone(ec.cached_notice(self.components))
        # an old file without the recorded versions: still never offers an engine that is not newer
        saved["last_check"] = {"at": "x", "engine": "0.1.45", "app": None}
        state_file.write_text(json.dumps(saved), encoding="utf-8")
        self.assertIsNone(ec.cached_notice(self.components))
        saved["last_check"] = {"at": "x", "engine": "0.1.46", "app": None}
        state_file.write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(ec.cached_notice(self.components), notice)

    def test_check_update_and_rollback_from_the_terminal(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        code, text, _ = self.cli("update", "--check", *self.common())
        self.assertEqual(code, 0)
        self.assertIn("update available: 0.1.45 -> 0.1.46", text)
        self.assertEqual(ec.cached_notice(self.components), "update available (engine 0.1.46): run `strata update`")
        code, listing, _ = self.cli("components", "engine", "--components-dir", str(self.components))
        self.assertIn("update available (engine 0.1.46): run `strata update`", listing)
        code, text, _ = self.cli("update", "--check", "--json", *self.common())
        self.assertEqual((code, json.loads(text)["engine"]["status"]), (0, "available"))
        code, text, _ = self.cli("update", *self.common())
        self.assertEqual(code, 0, text)
        self.assertIn("0.1.45 -> 0.1.46   updated (tested: started, answered one token, stopped)", text)
        self.assertIn("Saved tuning presets: marked for the short re-check (nothing was deleted).", text)
        self.assertIn("strata components rollback", text)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertIsNone(ec.cached_notice(self.components))
        code, text, _ = self.cli("update", *self.common())
        self.assertEqual(code, 0)
        self.assertIn("up to date", text)
        code, text, _ = self.cli("components", "rollback", *self.common())
        self.assertEqual(code, 0, text)
        self.assertIn("Engine rolled back: 0.1.46 -> 0.1.45", text)
        self.assertEqual(self.resolved(), "0.1.45")
        code, out, _ = self.cli("components", "rollback", "--json", *self.common())
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["to"], "0.1.46", "rolling back twice goes forward again")

    def test_a_failed_test_start_exits_6_and_says_the_old_engine_is_unchanged(self):
        self.install_current()
        self.publish(("0.1.47", fx.BAD_ENGINE))
        code, text, err = self.cli("update", *self.common())
        self.assertEqual(code, 6, text + err)
        self.assertIn("NOT changed", text)
        self.assertIn("unchanged", text)
        self.assertEqual(self.resolved(), "0.1.45")

    def test_jsonl_streams_progress_and_ends_with_the_result(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        code, text, _ = self.cli("update", "--output", "jsonl", *self.common())
        records = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(code, 0)
        self.assertTrue(any(r.get("event") == "progress" and r["percent"] == 100 for r in records))
        self.assertTrue(records[-1]["ok"] and records[-1]["results"][0]["status"] == "updated")

    def test_an_untrusted_or_forbidden_update_is_exit_5_and_fetches_nothing(self):
        self.install_current()
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        with mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": ""}):
            code, _, err = self.cli("update", *self.common())
            self.assertEqual(code, 5)
            self.assertIn("trusts no update key", err)
        wrong = self.root / "wrong.json"
        wrong.write_text(json.dumps({"keys": [signing.public_part(fx.make_key())]}))
        with mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(wrong)}):
            code, _, err = self.cli("update", *self.common())
            self.assertEqual(code, 5)
            self.assertIn("does not trust", err)
        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            code, _, err = self.cli("update", *self.common())
            self.assertEqual(code, 5)
            self.assertIn("STRATA_NO_ACQUIRE", err)
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(self.archive_requests(), [])


class AnotherCardFamilyNote(unittest.TestCase):
    """A feed whose only HIP engine is built for another AMD card family says so in one plain line."""

    def test_the_note_names_both_families_and_the_build_to_ask_for(self):
        import tempfile
        from types import SimpleNamespace
        pkg = SimpleNamespace(label="0.1.45.1 linux-x86_64-hip-rx7900", platform="linux-x86_64", backend="hip", min_runtime=None,
                              archs=("gfx1100", "gfx1101", "gfx1102"), min_driver={})
        hardware = {"platform": "linux-x86_64", "backend": "hip", "runtime": None, "gfx": ["gfx1201"], "cuda_archs": [], "nvidia_driver": None}
        with tempfile.TemporaryDirectory() as d:
            ctx = SimpleNamespace(components=Path(d), hardware=hardware)
            report, offered = updater._choose_engine(SimpleNamespace(engine=[pkg]), ctx, SimpleNamespace(section=lambda name: {}))
        self.assertIsNone(offered)
        self.assertEqual(report["status"], "none-for-this-pc")
        self.assertEqual(report["note"], "This build of Strata is for AMD RX 7000 cards. This computer has an AMD RX 9000 card (gfx1201). Ask for the RX 9000 build.")


class FailedStartRecords(LazyModel):
    """#3654: a failed test start used to leave only the last 64 KiB of the engine log, because the sandbox that held the host, child and
    ready records was deleted at once. The records are kept (without tokens or keys) next to the log."""

    def failing_start(self):
        def start(name, **kw):
            runtime_dir = Path(os.environ["STRATA_RUNTIME_DIR"])           # the sandbox's, set by default_test_start
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / f"{name}.json").write_text(json.dumps({"pid": 4321, "token": "a1b2c3d4" * 8, "state": "starting"}), encoding="utf-8")
            (runtime_dir / f"{name}-{'a1b2c3d4' * 8}.child.json").write_text(json.dumps({"host_token": "a1b2c3d4" * 8, "child_pid": 111}), encoding="utf-8")
            (runtime_dir / f"{name}-9f.ready.json").write_text(json.dumps({"pid": 222, "api_key": "sekret-key"}), encoding="utf-8")
            (runtime_dir / f"{name}.server.log").write_text("server: model loaded\nserver: pid 222\n", encoding="utf-8")
            (runtime_dir / f"{name}.host.log").write_text("", encoding="utf-8")
            return {"ok": False, "state": "starting", "error": "startup_wait_expired", "phase": "ready_file_not_matched"}
        return start

    def test_the_runtime_records_of_a_failed_start_are_kept_without_tokens_or_keys(self):
        from strata_control import runtime
        self.install_current()
        folder = fx.build_package(self.root / "lazy", "0.1.46", key=self.key)
        stage = self.root / "stage"
        (stage / "engine").mkdir(parents=True)
        shutil.copytree(folder, stage / "engine" / folder.name)
        component = ec.read_manifest(stage / "engine" / folder.name)
        self.config.write_text(json.dumps({**json.loads(self.config.read_text()), "api_key": "sekret-key"}), encoding="utf-8")
        with mock.patch.object(runtime, "start", side_effect=self.failing_start()), mock.patch.object(runtime, "stop", return_value={}):
            with self.assertRaises(updater.TestStartFailed):
                updater.default_test_start(stage, component, self.ctx())
        kept = updater.updates_dir(self.components) / "logs" / f"test-start-{component.version}-failed"
        names = sorted(p.name for p in kept.iterdir())
        self.assertIn("run.json", names)
        self.assertTrue(any(n.endswith(".child.json") for n in names), names)
        self.assertTrue(any(n.endswith(".ready.json") for n in names), names)
        self.assertIn("updtest.server.log", names)
        text = "".join(p.read_text(encoding="utf-8") for p in kept.iterdir())
        self.assertNotIn("sekret-key", text)
        self.assertNotIn("a1b2c3d4a1b2c3d4", text, "no full token is kept")
        self.assertIn('"child_pid": 111', text)                              # the PIDs that show a mismatch are
        self.assertIn("server: pid 222", text)
        self.assertFalse((self.components / ".updates").exists() and any(p.name.startswith("t-") for p in (self.components / ".updates").rglob("*") if p.is_dir()),
                         "the sandbox itself is still removed")

    def test_only_the_last_few_failed_starts_are_kept(self):
        logs = updater.updates_dir(self.components) / "logs"
        for i in range(5):
            folder = logs / f"test-start-0.1.4{i}-failed"
            folder.mkdir(parents=True)
            (folder / "run.json").write_text("{}", encoding="utf-8")
            os.utime(folder, (1_700_000_000 + i, 1_700_000_000 + i))
        from types import SimpleNamespace
        sandbox = self.root / "sandbox"
        sandbox.mkdir()
        (sandbox / "run.json").write_text("{}", encoding="utf-8")
        updater._keep_records(sandbox, SimpleNamespace(version="0.1.49"), self.ctx())
        left = sorted(p.name for p in logs.glob("test-start-*-failed"))
        self.assertEqual(len(left), 3, left)
        self.assertIn("test-start-0.1.49-failed", left)


if __name__ == "__main__":
    unittest.main()
