"""Tests for setup.py's reproducible installs (#214): the engine of the checkout's own release first (the latest as
the fallback), Hugging Face files at pinned revisions (a gone revision stops the download), the pinned
requirements file, and an existing install left as it is.  Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_pins
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def not_found(url):
    return urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class HuggingFacePins(unittest.TestCase):
    def test_every_model_url_is_pinned(self):
        for name, fam in setup.FAMILIES.items():
            if fam.get("ready"):                       # ready-made packs: pinned per file (test_ready_packs_pinned)
                continue
            for key in ("hf", "mmproj_hf"):
                with self.subTest(family=name, key=key):
                    self.assertRegex(fam[key], SHA)
                    self.assertNotIn("/resolve/main/", fam[key])
        self.assertRegex(setup.HF, SHA)

    def test_ready_packs_pinned(self):
        """NVFP4: every size names a repository of NVFP4_REPOS, pinned by a commit; every file has a size and a
        SHA-256, and the pack folder, dense GGUF, n-gram table and embedding are each there."""
        sizes = [m for m, d in setup.MODELS.items() if d.get("repo")]
        self.assertEqual(sorted(sizes), ["NVFP4", "NVFP4-NVIDIA", "NVFP4-QAD"])
        for m in sizes:
            if setup.MODELS[m].get("prepare"):         # NVIDIA's checkpoint: pinned files, prepared on the PC
                t = setup.NVFP4_REPOS[setup.MODELS[m]["repo"]]
                self.assertRegex(t["revision"], r"^[0-9a-f]{40}$")
                for path, (role, size, sha) in t["files"].items():
                    self.assertEqual(role, "src")
                    self.assertGreater(size, 0)
                    self.assertRegex(sha, r"^[0-9a-f]{64}$")
                self.assertEqual(sorted(setup.ready_paths(m, setup.Path("M"))), ["dense", "mtp", "pack", "ple"])
                continue
            t = setup.NVFP4_REPOS[setup.MODELS[m]["repo"]]
            self.assertRegex(t["revision"], r"^[0-9a-f]{40}$")
            for path, (role, size, sha) in t["files"].items():
                with self.subTest(model=m, path=path):
                    self.assertIn(role, ("pack", "dense", "ple", "embd"))
                    self.assertGreater(size, 0)
                    self.assertRegex(sha, r"^[0-9a-f]{64}$")
            p = setup.ready_paths(m, setup.Path("M"))
            self.assertEqual(sorted(p), ["dense", "embd", "pack", "ple"])
            self.assertEqual(p["pack"].name, "pack")
            self.assertIn("experts.bin", t["files"] and [x.split("/")[-1] for x in t["files"]])

    def test_unpinned(self):
        url = setup.FAMILIES["swift"]["hf"] + "x.gguf"
        self.assertEqual(setup.hf_unpinned(url),
                         "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/x.gguf")
        self.assertEqual(setup.hf_unpinned("https://example.com/a/b"), "https://example.com/a/b")

    def test_a_gone_revision_stops_the_download(self):
        """The pinned commit answers 404: setup stops with a plain message and fetches no current file."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            raise not_found(req.full_url)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen),                 mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}):
            setup.os.environ.pop("STRATA_ALLOW_UNPINNED", None)
            dst = Path(d) / "m.gguf"
            with self.assertRaises(SystemExit):
                _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertFalse(dst.exists())
        self.assertEqual([m for m, _ in seen], ["HEAD"])
        self.assertTrue(all("/resolve/main/" not in u for _, u in seen))

    def test_the_opt_out_downloads_the_current_file(self):
        """STRATA_ALLOW_UNPINNED=1 (a tester's choice): the repository's current file, announced, off by default."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            if "/resolve/main/" not in req.full_url:
                raise not_found(req.full_url)
            return Response(b"model bytes", status=200)

        # the pinned Hugging Face path itself (setup's --source huggingface; auto may pick ModelScope)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen),                 mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface", "STRATA_ALLOW_UNPINNED": "1"}),                 mock.patch.object(setup, "_unpinned_said", []):
            dst = Path(d) / "m.gguf"
            _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
        self.assertIn("STRATA_ALLOW_UNPINNED=1", out)
        self.assertIn("not at the pinned revision any more", out)
        self.assertEqual([m for m, _ in seen], ["HEAD", "HEAD", "GET"])
        self.assertTrue(all("/resolve/main/" in u for _, u in seen[1:]))

    def test_a_complete_part_is_finished_without_a_request(self):
        """A .part with every byte (setup stopped between the last byte and the rename): renamed, not resumed with a
        range past its end - the server answers that with 416, which download() retried 30 times, 10 s apart."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.headers.get("Range")))
            if req.get_method() == "HEAD":
                return Response(b"model bytes")
            raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup.time, "sleep", lambda s: None):
            dst = Path(d) / "m.gguf"
            dst.with_name("m.gguf.part").write_bytes(b"model bytes")
            quiet(setup.download, "https://example.com/m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
            self.assertTrue(setup.done(dst))
        self.assertEqual(seen, [("HEAD", None)])

    def test_mtp_fetch_is_pinned_and_stops_when_the_revision_is_gone(self):
        import mtp_fetch
        self.assertRegex(mtp_fetch.REPO, SHA)
        self.assertIn(mtp_fetch.REVISION, mtp_fetch.REPO)
        pinned = mtp_fetch.REPO

        def urlopen(req, timeout=None):
            raise not_found(req.full_url)

        try:
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen),                     mock.patch.dict(mtp_fetch.os.environ, {"STRATA_ALLOW_UNPINNED": ""}):
                with self.assertRaises(SystemExit) as stop:
                    mtp_fetch.resolve_repo()
                self.assertIn("no longer on Hugging Face", str(stop.exception))
                self.assertEqual(mtp_fetch.REPO, pinned)
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen),                     mock.patch.dict(mtp_fetch.os.environ, {"STRATA_ALLOW_UNPINNED": "1"}),                     contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mtp_fetch.resolve_repo(), "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/")
            self.assertIn("STRATA_ALLOW_UNPINNED=1", err.getvalue())
        finally:
            mtp_fetch.REPO = pinned

    def test_hf_endpoint(self):
        # #495: HF_ENDPOINT (a mirror) serves the same pinned revision; a trailing slash and blanks are dropped
        repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        with mock.patch.dict(setup.os.environ, {}, clear=False):
            setup.os.environ.pop("HF_ENDPOINT", None)
            self.assertEqual(setup.hf(repo), f"https://huggingface.co/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
            for value in ("https://hf-mirror.com", "https://hf-mirror.com/", " https://hf-mirror.com/ "):
                setup.os.environ["HF_ENDPOINT"] = value
                url = setup.hf(repo)
                self.assertEqual(url, f"https://hf-mirror.com/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
                self.assertRegex(url, SHA)
                self.assertEqual(setup.hf_unpinned(url + "x.gguf"), f"https://hf-mirror.com/{repo}/resolve/main/x.gguf")
            setup.os.environ["HF_ENDPOINT"] = ""
            self.assertEqual(setup.hf_endpoint(), "https://huggingface.co")

    def test_mtp_fetch_honours_hf_endpoint(self):
        import importlib
        import mtp_fetch
        try:
            with mock.patch.dict(mtp_fetch.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}):
                importlib.reload(mtp_fetch)
                self.assertTrue(mtp_fetch.REPO.startswith("https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/"))
                self.assertRegex(mtp_fetch.REPO, SHA)
                self.assertTrue(mtp_fetch.pinned())                 # the tensors' SHA-256 checks still apply
        finally:
            importlib.reload(mtp_fetch)
        self.assertTrue(mtp_fetch.REPO.startswith("https://huggingface.co/"))

    def test_step5_names_the_folder(self):
        # #495: where setup expects the model files, and how to give it files downloaded by hand
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["96GB-1x16GB"]
        with mock.patch.dict(setup.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
            code, out, cfg, _ = install(ram, found, ["--family", "qwen", "--model", "IQ3_XXS", "--no-start"])
        self.assertEqual(code, 0)
        text = out.replace("\\", "/")
        self.assertRegex(text, r"The model files go in .*/models/IQ3_XXS")
        self.assertIn("put them here with their original names (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                      text)
        self.assertIn("--gguf-dir", text)
        self.assertIn("Downloading from https://hf-mirror.com (HF_ENDPOINT)", text)


@unittest.skipIf(setup.is_arm64() and sys.platform.startswith("linux"),
                 "arm64 Linux takes its ready-made engine from the signed feed, not from the flat x86 zips these tests build "
                 "(tests/terminal/test_linux_aarch64.py covers that path)")
class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "engine").mkdir()
        self.patches = [mock.patch.object(setup, "ROOT", self.root),
                        mock.patch.object(setup, "source_version", lambda: "0.1.31")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def fake_download(self, got, wrote=None):
        def download(url, dst, what=None):
            got.append(url)
            if wrote is not None:
                wrote.append(Path(dst))
            with zipfile.ZipFile(dst, "w") as z:
                z.writestr("BUILD.json", json.dumps({"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [89]}))
                z.writestr(setup.EXE, b"engine")
        return download

    def fake_digest(self, wrote):
        """The SHA-256 of whatever the fake download just wrote, computed for real.

        `get_prebuilt()` now checks the archive against GitHub's size and SHA-256 before it unpacks
        anything, so a test that mocks the download has to supply a hash or it stops at the check. These
        tests are about WHICH release and WHICH asset are chosen, not about the bytes; what happens when
        the hash is wrong has its own file (tools/test_setup_engine_hash.py).
        """
        def digest(asset, base):
            data = Path(wrote[-1]).read_bytes()
            return len(data), hashlib.sha256(data).hexdigest()
        return digest

    def run_get(self, published):
        heads, got, wrote = [], [], []

        def urlopen(req, timeout=None):
            heads.append(req.full_url)
            if not any(req.full_url.startswith(p) for p in published):
                raise not_found(req.full_url)
            return Response()

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup, "download", self.fake_download(got, wrote)), \
                mock.patch.object(setup, "engine_digest", self.fake_digest(wrote)):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        return eng, out, heads, got

    def test_bases(self):
        self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                         ["https://github.com/Niko1221/Strata/releases/download/v0.1.31/", setup.PREBUILT_URL])
        self.assertEqual(setup.prebuilt_bases("https://mirror.example/x"), ["https://mirror.example/x/"])

    def test_the_checkout_s_release_first(self):
        tag = "https://github.com/Niko1221/Strata/releases/download/v0.1.31/"
        eng, out, heads, got = self.run_get([tag, setup.PREBUILT_URL])
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [tag + setup.PREBUILT_ASSET])
        self.assertEqual(len(heads), 1)

    def test_latest_when_it_is_not_published(self):
        eng, out, heads, got = self.run_get([setup.PREBUILT_URL])
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual(got, [setup.PREBUILT_URL + setup.PREBUILT_ASSET])
        self.assertIn("No ready-made engine for v0.1.31", out)

    def test_a_refused_archive_is_not_kept(self):
        """PR #324: a refused archive (too old, or no code for the GPU) kept its zip and .done mark, and every later
        run reused it ("already downloaded") instead of the published one."""
        for meta in ({"version": "0.1.0", "archs": [89]},
                     {"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [120]}):
            with self.subTest(meta=meta):
                wrote = []

                def download(url, dst, what=None):
                    wrote.append(Path(dst))
                    with zipfile.ZipFile(dst, "w") as z:
                        z.writestr("BUILD.json", json.dumps(meta))
                        z.writestr(setup.EXE, b"engine")
                    setup.mark(dst)

                with mock.patch.object(setup.urllib.request, "urlopen", lambda req, timeout=None: Response()), \
                        mock.patch.object(setup, "download", download), \
                        mock.patch.object(setup, "engine_digest", self.fake_digest(wrote)):
                    eng, _ = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
                self.assertIsNone(eng)
                z = self.root / "engine" / setup.PREBUILT_ASSET
                self.assertFalse(z.exists())
                self.assertFalse(z.with_name(z.name + ".done").exists())

    def test_an_archive_that_does_not_unpack_is_not_kept(self):
        """#397, for an archive that does not unpack (not a zip, or a damaged one): it kept its zip and .done mark,
        so every later run failed on it, even after the right one was published."""
        with tempfile.TemporaryDirectory() as folder:  # a --prebuilt folder, through the real download()
            asset = Path(folder) / setup.PREBUILT_ASSET
            asset.write_bytes(b"<html>not a zip</html>")
            with self.assertRaises(zipfile.BadZipFile):
                quiet(setup.get_prebuilt, folder, {"arch": 89}, "gpu")
            with zipfile.ZipFile(asset, "w") as z:
                z.writestr("BUILD.json", json.dumps({"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [89]}))
                z.writestr(setup.EXE, b"engine")
            eng, _ = quiet(setup.get_prebuilt, folder, {"arch": 89}, "gpu")
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"engine")

    def test_an_installed_engine_is_kept(self):
        (self.root / "engine" / "BUILD.json").write_text(json.dumps(
            {"version": ".".join(map(str, setup.MIN_ENGINE)), "archs": [89]}))
        (self.root / "engine" / setup.EXE).write_bytes(b"old")

        def urlopen(req, timeout=None):
            raise AssertionError("asked the network for an installed engine")

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            eng, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {"arch": 89}, "gpu")
        self.assertEqual(eng, self.root / "engine")
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"old")


class Requirements(unittest.TestCase):
    def test_every_package_is_pinned(self):
        lines = setup.requirement_lines()
        names = {setup.req_name(x) for x in lines}
        for p in setup.PY_PACKAGES:
            self.assertIn(p, names)
        for x in lines:
            self.assertIn("==", x, x)
        self.assertEqual(setup.req_name('numpy==2.5.3; python_version >= "3.12"'), "numpy")
        self.assertEqual(setup.req_name("charset_normalizer==3"), "charset-normalizer")

    def test_build_helpers_are_pinned_but_absent_from_runtime_install(self):
        runtime = {setup.req_name(x) for x in setup.requirement_lines()}
        build = setup.requirement_lines(setup.BUILD_REQUIREMENTS)
        self.assertEqual({setup.req_name(x) for x in build}, {"cmake", "ninja"})
        self.assertTrue(all("==" in x for x in build))
        self.assertTrue(runtime.isdisjoint({"cmake", "ninja"}))

    def test_local_build_installs_only_missing_helper(self):
        installed = False
        requests = []
        commands = []

        def find_tool(name):
            if name == "cmake":
                return "/usr/bin/cmake"
            return "/venv/bin/ninja" if installed else None

        def pip_install(packages, what):
            nonlocal installed
            requests.append((packages, what))
            installed = True

        def run(command, **kwargs):
            commands.append(command)
            return mock.Mock(returncode=0)

        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "find_tool", side_effect=find_tool), \
                mock.patch.object(setup, "_tool_version", return_value=(True, (3, 28, 1))), \
                mock.patch.object(setup, "pip_install", side_effect=pip_install), \
                mock.patch.object(setup, "run", side_effect=run):
            setup.cmake_build(ROOT, ROOT / "build-test", "strata", [], None, "unused.bat")
        self.assertEqual(requests, [(["ninja==1.13.2"], "CMake and Ninja build tools")])
        self.assertEqual(commands[0][:4], ["/usr/bin/cmake", "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=/venv/bin/ninja"])
        self.assertEqual(commands[1][:3], ["/usr/bin/cmake", "--build", str(ROOT / "build-test")])

    def pip(self, stamp, packages, installed=()):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            if stamp is not None:
                (Path(d) / ".strata-pip.json").write_text(json.dumps(stamp))
            with mock.patch.object(setup.sys, "prefix", d), \
                    mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)), \
                    mock.patch.object(setup, "_installed", lambda name: name in installed):
                quiet(setup.pip_install, packages, "the packages")
            after = json.loads((Path(d) / ".strata-pip.json").read_text()) if ran else stamp
        return [c for cmd in ran for c in cmd if "==" in c or c in setup.PY_PACKAGES], after

    def test_a_stamp_written_by_another_system_is_not_believed(self):                    # #3634
        lines = setup.requirement_lines()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".strata-pip.json").write_text(json.dumps(lines))
            (Path(d) / ".strata-pip.json.os").write_text("win32" if setup.sys.platform != "win32" else "linux")
            ran = []
            with mock.patch.object(setup.sys, "prefix", d), mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)):
                quiet(setup.pip_install, lines, "the packages")
            self.assertTrue(ran)                                                          # installed again, here
            self.assertEqual((Path(d) / ".strata-pip.json.os").read_text(), setup.sys.platform)
            ran.clear()
            with mock.patch.object(setup.sys, "prefix", d), mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)):
                quiet(setup.pip_install, lines, "the packages")
            self.assertFalse(ran)                                                         # and then skipped as before

    def test_fresh_install_gets_every_pin(self):
        lines = setup.requirement_lines()
        ran, stamp = self.pip(None, lines)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertEqual(self.pip(stamp, lines)[0], [])                       # the second run: nothing

    def test_an_install_from_before_the_pins_is_left_alone(self):
        lines = setup.requirement_lines()
        legacy = sorted(setup.PY_PACKAGES) + ["nvidia-cublas==13.0.2.14"]
        deps = {"markupsafe", "certifi", "charset-normalizer", "idna", "urllib3", "colorama"}
        # the packages it has stay as they are; only truststore (#3722: the operating system's certificates) is new, and it is wanted
        self.assertEqual(self.pip(legacy, lines, installed=deps)[0], ['truststore==0.10.5; python_version >= "3.10"'])
        deps.add("truststore")
        self.assertEqual(self.pip(legacy, lines, installed=deps)[0], [])
        missing = [p for p in legacy if p != "psutil"]                         # an older list without psutil
        ran, _ = self.pip(missing, lines, installed=deps)
        self.assertEqual(ran, ["psutil==7.2.2"])

    def test_a_changed_pin_is_installed(self):
        lines = setup.requirement_lines()
        old = [x.replace("tqdm==4.70.1", "tqdm==4.60.0") for x in lines]
        ran, _ = self.pip(old, lines)
        self.assertEqual(ran, ["tqdm==4.70.1"])


if __name__ == "__main__":
    unittest.main()
