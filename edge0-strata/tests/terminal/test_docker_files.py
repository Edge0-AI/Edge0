"""The Docker files stay consistent: shell scripts parse, Dockerfiles use only what the legacy builder knows, the compose examples carry what
docs/DOCKER.md says they carry, and every setting the document lists is one the entry point reads."""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
SH = shutil.which("sh")
BASH = shutil.which("bash")      # the full path: on Windows a bare "bash" can start the WSL launcher in System32 instead of Git bash
CYGPATH = shutil.which("cygpath")


def sh_path(p) -> str:
    """The path as the shell reads it. Git bash on Windows wants /c/Users/..., not C:\\Users\\... (tar even takes `C:` for a host)."""
    if CYGPATH is None:
        return str(p)
    return subprocess.run([CYGPATH, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()


class Scripts(unittest.TestCase):
    @unittest.skipIf(SH is None, "needs sh")
    def test_shell_scripts_parse(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh", "publish.sh"):
            r = subprocess.run([SH, "-n", sh_path(DOCKER / name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")

    @unittest.skipIf(BASH is None, "needs bash")
    def test_smoke_test_parses_with_bash(self):           # it is a bash script (process substitution), not POSIX sh
        r = subprocess.run([BASH, "-n", sh_path(DOCKER / "smoke-test.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipIf(SH is None or shutil.which("tar") is None, "needs sh and tar")
    def test_build_script_exits_zero_after_assembling_a_context(self):
        """--ctx-only stops after the context is made (no docker needed). The EXIT trap once returned 1 there, so a good run reported failure."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            (pkg / "engine-component.json").write_text('{"entrypoint": "strata"}', encoding="utf-8")
            (pkg / "strata").write_text("#!/bin/sh\n", encoding="utf-8")
            (pkg / "strata").chmod(0o755)
            ctx = Path(tmp) / "ctx"
            r = subprocess.run([SH, sh_path(DOCKER / "build.sh"), "--flavor", "cuda", "--engine", sh_path(pkg), "--ctx", sh_path(ctx), "--ctx-only"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((ctx / "Dockerfile").is_file())
            self.assertTrue((ctx / "terminal-src" / "docs" / "TERMINAL_LAUNCH.md").is_file())
            self.assertTrue((ctx / "engine" / "pkg" / "engine-component.json").is_file())
            # the bundle reads tools/ modules by path as well (INSTALLER_SCRIPTS), and the DeepSeek ones are nested: every name must be in the
            # context, or package_terminal.py stops the build at "installer tool missing: tools/<name>.py" (the image build after v0.2.4.1 was broken this way)
            pt = ast.parse((ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8"))
            tools: set[str] = set()
            for node in pt.body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("INSTALLER_TOOLS", "INSTALLER_SCRIPTS") for t in node.targets):
                    tools.update(ast.literal_eval(node.value))
            self.assertTrue({"dsv41/pack_gguf", "dsv41/tune"} <= tools)          # the nested ones this assertion exists for
            for name in sorted(tools):
                self.assertTrue((ctx / "terminal-src" / "tools" / f"{name}.py").is_file(), f"build.sh did not copy tools/{name}.py into the context")

    def test_scripts_use_unix_line_ends(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh", "publish.sh"):
            self.assertNotIn(b"\r", (DOCKER / name).read_bytes(), name)       # build.sh strips CR for a CRLF checkout, the files themselves stay LF


class Dockerfiles(unittest.TestCase):
    FILES = (DOCKER / "Dockerfile", ROOT / "Dockerfile")

    def test_only_legacy_builder_syntax(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"<<\s*-?['\"]?\w+", f"{f.name}: a heredoc needs BuildKit")
            self.assertNotIn("--mount=", text, f.name)
            self.assertNotRegex(text, r"COPY\s+--chmod", f.name)

    def test_runtime_contract(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertIn("HEALTHCHECK", text, f.name)
            self.assertIn("STOPSIGNAL SIGTERM", text, f.name)
            self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]', text, f.name)
            self.assertIn("VOLUME [\"/data\"]", text, f.name)
            self.assertIn("STRATA_COMPONENTS_DIR=/opt/strata/components", text, f.name)

    def test_component_image_checks_the_package_before_use(self):
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("package_engine.py", text)
        self.assertIn("binary_deps.py", text)
        self.assertIn("ldd", text)                       # the finished image resolves every library of the package

    def test_binary_deps_and_package_engine_exist_where_the_dockerfile_copies_them(self):
        for name in ("package_engine.py", "binary_deps.py", "package_terminal.py"):
            self.assertTrue((ROOT / "packaging" / name).is_file(), name)


class Compose(unittest.TestCase):
    def load(self, name: str) -> dict:
        data = yaml.safe_load((DOCKER / "compose" / name).read_text(encoding="utf-8"))
        self.assertIn("strata", data["services"], name)
        return data

    def test_common_settings(self):
        for name in ("docker-compose.nvidia.yml", "docker-compose.amd.yml", "docker-compose.intel.yml"):
            svc = self.load(name)["services"]["strata"]
            self.assertIn("strata-data:/data", svc["volumes"], name)
            self.assertEqual(svc["ulimits"]["memlock"], {"soft": -1, "hard": -1}, name)
            self.assertEqual(svc["shm_size"], "2gb", name)
            self.assertEqual(svc["stop_grace_period"], "2m", name)
            self.assertTrue(svc["ports"][0].startswith("${STRATA_BIND:-127.0.0.1}:"), f"{name}: the port must default to this computer only")
            for var in ("MODEL", "MODEL_PATH", "CONTEXT", "PARALLEL", "VISION", "TUNE", "API_KEY"):
                self.assertIn(var, svc["environment"], f"{name}: {var}")
            self.assertIn("strata-data", self.load(name)["volumes"], name)

    def test_gpu_access_per_vendor(self):
        nv = self.load("docker-compose.nvidia.yml")["services"]["strata"]
        dev = nv["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(dev["driver"], "nvidia")
        self.assertIn("gpu", dev["capabilities"])
        amd = self.load("docker-compose.amd.yml")["services"]["strata"]
        self.assertIn("/dev/kfd:/dev/kfd", amd["devices"])
        self.assertIn("/dev/dri:/dev/dri", amd["devices"])
        self.assertIn("video", amd["group_add"])
        self.assertTrue(any("render" in str(g).lower() or "RENDER" in str(g) for g in amd["group_add"]))
        intel = self.load("docker-compose.intel.yml")["services"]["strata"]
        self.assertIn("/dev/dri:/dev/dri", intel["devices"])
        self.assertNotIn("/dev/kfd:/dev/kfd", intel["devices"])


@unittest.skipIf(SH is None, "needs sh")
class LegacyModelsGuard(unittest.TestCase):
    """#3421: model files left in the older image's /data/models (a folder the person mounted there, or a failed move) are not read by this image,
    so the start would download the model again. It stops and says where the files are; MODEL_PATH names them, ALLOW_DOWNLOAD=1 goes on."""

    def run_guard(self, folder, model_path="", allow=None):
        import os
        import tempfile
        entry = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
        body = entry[entry.index("legacy_models_guard() {"):]
        body = body[:body.index("\n}\n") + 3]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "guard.sh"
            script.write_text("warn() { printf '[!] %s\\n' \"$*\" >&2; }\ndie() { printf '[X] %s\\n' \"$*\" >&2; exit \"${2:-1}\"; }\n" + body
                              + f"\nlegacy_models_guard '{model_path}' '{sh_path(folder)}'\necho CONTINUED\n", encoding="utf-8", newline="\n")
            env = dict(os.environ)
            env.pop("ALLOW_DOWNLOAD", None)
            if allow is not None:
                env["ALLOW_DOWNLOAD"] = allow
            return subprocess.run([SH, sh_path(script)], capture_output=True, text=True, env=env)

    def test_files_in_the_older_place_stop_the_start_with_the_folder_named(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            shards = Path(tmp) / "IQ3_XXS"
            shards.mkdir()
            (shards / "model-00001-of-00002.gguf").write_bytes(b"GGUF")
            done = self.run_guard(Path(tmp))
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertNotIn("CONTINUED", done.stdout)
            self.assertIn(shards.name, done.stderr)
            self.assertIn("MODEL_PATH", done.stderr)
            self.assertIn("ALLOW_DOWNLOAD=1", done.stderr)
            self.assertIn("Nothing was downloaded", done.stderr)

    def test_model_path_allow_download_an_empty_folder_and_a_missing_one_go_on(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "IQ3_XXS").mkdir()
            (Path(tmp) / "IQ3_XXS" / "model-00001-of-00002.gguf").write_bytes(b"GGUF")
            named = self.run_guard(Path(tmp), model_path="/models")
            self.assertEqual((named.returncode, "CONTINUED" in named.stdout), (0, True), named.stderr)
            allowed = self.run_guard(Path(tmp), allow="1")
            self.assertEqual((allowed.returncode, "CONTINUED" in allowed.stdout), (0, True), allowed.stderr)
            self.assertIn("[!]", allowed.stderr)
            self.assertIn("ALLOW_DOWNLOAD=1", allowed.stderr)
        with tempfile.TemporaryDirectory() as empty:
            nothing = self.run_guard(Path(empty))
            self.assertEqual((nothing.returncode, "CONTINUED" in nothing.stdout), (0, True), nothing.stderr)
        missing = self.run_guard(Path("/no/such/folder"))
        self.assertEqual((missing.returncode, "CONTINUED" in missing.stdout), (0, True), missing.stderr)

    def test_the_entry_point_runs_the_guard_before_any_flag_is_built(self):
        entry = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertLess(entry.index('legacy_models_guard "$model_path"'), entry.index('--model "$MODEL"'))


class Document(unittest.TestCase):
    def test_every_documented_setting_is_read_by_the_entry_point(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        section = doc.split("## Settings", 1)[1].split("###", 1)[0]
        entry = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
        names: set[str] = set()
        for line in section.splitlines():
            if line.startswith("| `"):
                names.update(re.findall(r"`([A-Z][A-Z_]+)`", line.split("|")[1]))
        self.assertTrue({"MODEL", "CONTEXT", "API_KEY", "TUNE", "VISION", "MODEL_PATH", "STRATA_SET"} <= names, names)
        for name in sorted(names):
            self.assertIn(name, entry, f"docs/DOCKER.md lists {name} but docker/entrypoint.sh does not read it")

    def test_tags_in_the_document_match_the_compose_defaults(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        for tag in ("0.2.7-cuda13", "0.2.7-cuda12", "0.2.7-rocm-gfx120x", "0.2.7-rocm-gfx110x", "0.2.7-rocm-gfx1151", "0.2.7-sycl"):
            self.assertIn(tag, doc)
        for name, tag in (("docker-compose.nvidia.yml", "ghcr.io/niko1221/strata:0.2.7-cuda13"), ("docker-compose.amd.yml", "ghcr.io/niko1221/strata:0.2.7-rocm-gfx120x"),
                          ("docker-compose.intel.yml", "ghcr.io/niko1221/strata:0.2.7-sycl")):
            self.assertIn(tag, (DOCKER / "compose" / name).read_text(encoding="utf-8"))

    def test_the_user_doc_says_images_are_published_from_0_2_1(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertNotIn("TODO(owner)", doc)
        self.assertNotIn("not available yet", doc)
        self.assertIn("published from version 0.2.1 on", doc)
        self.assertIn("Publishing the images (for maintainers)", doc)


class Publishing(unittest.TestCase):
    """The scripts and the workflow that publish the images (docker/publish.sh, docker/fetch-engines.py, .github/workflows/docker-images.yml)."""
    WORKFLOW = ROOT / ".github" / "workflows" / "docker-images.yml"

    @unittest.skipIf(SH is None, "needs sh")
    def test_publish_script_parses_and_defaults_to_not_pushing(self):
        path = DOCKER / "publish.sh"
        self.assertEqual(subprocess.run([SH, "-n", sh_path(path)], capture_output=True, text=True).returncode, 0)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("\r", text)
        self.assertIn('push=""', text)                      # no --push, no docker push
        pushes = re.findall(r'^[^#\n]*docker push "', text, re.M)
        self.assertEqual(len(pushes), 2, pushes)             # the version tag and latest-<flavor>, both inside `if [ -n "$push" ]`

    def test_every_flavor_of_the_document_is_one_the_fetch_script_knows(self):
        src = (DOCKER / "fetch-engines.py").read_text(encoding="utf-8")
        flavors = re.findall(r'^    "([a-z0-9-]+)": \("', src, re.M)
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertTrue({"cuda13", "cuda12", "rocm-gfx120x", "rocm-gfx110x", "rocm-gfx1151", "sycl"} <= set(flavors), flavors)
        for flavor in flavors:
            self.assertIn(f"0.2.7-{flavor}", doc)

    def test_build_script_copies_everything_the_terminal_package_needs(self):
        """docker/build.sh lists the Terminal's files by hand; the list must cover package_terminal.py's payload (it once missed docs/TERMINAL_LAUNCH.md)."""
        build = (DOCKER / "build.sh").read_text(encoding="utf-8")
        listed = re.search(r'^paths="([^"]+)"', build, re.M).group(1).split()
        pt = ast.parse((ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8"))
        payload = None
        for node in pt.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PAYLOAD" for t in node.targets):
                payload = ast.literal_eval(node.value)
        self.assertTrue(payload)
        for item in payload:
            self.assertTrue(any(item == p or item.startswith(p.rstrip("/") + "/") for p in listed), f"build.sh does not copy {item}")

    def test_the_source_build_passes_the_commit_to_the_terminal_stage(self):
        """.git is not in the build context (.dockerignore), so the root Dockerfile's terminal stage gets the commit as a build argument, the way
        docker/Dockerfile does; without it package_terminal.py stops (#3644). The documented source-build commands pass it."""
        for dockerfile in (ROOT / "Dockerfile", DOCKER / "Dockerfile"):
            text = dockerfile.read_text(encoding="utf-8")
            self.assertRegex(text, r"(?m)^ARG STRATA_COMMIT=", dockerfile.name)
            self.assertIn("ENV STRATA_BUILD_COMMIT=${STRATA_COMMIT}", text, dockerfile.name)
        self.assertIn(".git", (ROOT / ".dockerignore").read_text(encoding="utf-8").split())
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertEqual(doc.count("docker build -t strata:src"), doc.count("--build-arg STRATA_COMMIT=$(git rev-parse HEAD)"))
        self.assertIn("/data/install/", (ROOT / ".dockerignore").read_text(encoding="utf-8"))

    def test_workflow_is_manual_and_pushes_only_on_request(self):
        data = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        triggers = data.get("on", data.get(True))            # YAML 1.1 reads a bare `on` as True
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        inputs = triggers["workflow_dispatch"]["inputs"]
        self.assertIn("release_tag", inputs)
        self.assertFalse(inputs["push"]["default"])
        self.assertEqual(data["permissions"], {"contents": "read"})
        self.assertEqual(data["jobs"]["image"]["permissions"], {"contents": "read", "packages": "write"})
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ghcr.io/niko1221/strata", text)
        self.assertIn("docker/publish.sh", text)
        self.assertNotIn("${{ secrets.GITHUB_TOKEN }}\" |", text)   # the token goes through an env variable, not into the command line


if __name__ == "__main__":
    unittest.main()
