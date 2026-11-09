"""Seed violations must fail with source locations."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class LayerTests(unittest.TestCase):
    def check(self, path, content, rule=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools/checks").mkdir(parents=True)
            shutil.copyfile(ROOT / "tools/checks/layers.toml", root / "tools/checks/layers.toml")
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            result = subprocess.run([sys.executable, "-I", str(ROOT / "tools/checks/check_layers.py"), str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1 if rule else 0, result.stdout + result.stderr)
            if rule:
                self.assertIn(f"{path}:1:", result.stdout)
                self.assertIn(rule, result.stdout)

    def test_system_headers(self):
        for header in ("sys/sysctl.h", "mach/mach_host.h", "objc/message.h", "objc/runtime.h", "os/proc.h", "dlfcn.h"):
            self.check("src/platform/system.cpp", f'#include <{header}>\n')
            self.check("src/app/system.cpp", f'#include <{header}>\n', "module-include")

    def test_clean(self):
        self.check("src/weights/x.cpp", '#include "exec/x.h"\n#include <vector>\n')

    def test_llama_header_in_any_directory(self):
        # No directory is exempt any more, including a standalone tool.
        for path in ("tools/oracle/x.cpp", "tools/cli/x.cpp", "tests/x/x.cpp"):
            with self.subTest(path=path):
                self.check(path, '#include "llama.h"\n', "llama-header")

    def test_llama(self):
        self.check("src/app/x.cpp", '#include "llama.h"\n', "llama-header")

    def test_weights_app(self):
        self.check("src/weights/x.cpp", '#include "app/x.h"\n', "module-include")

    def test_public_ggml(self):
        self.check("include/edge0/x.h", '#include "ggml.h"\n', "public-include")

    def test_cli_ggml(self):
        self.check("tools/cli/x.cpp", '#include "ggml.h"\n', "public-include")

    def test_integration_internal(self):
        self.check("tests/integration/x.cpp", '#include "platform/x.h"\n', "public-include")

    def test_link(self):
        self.check("src/weights/CMakeLists.txt", "target_link_libraries(moe_weights INTERFACE moe_app)\n", "target-link")

    def test_private_internal(self):
        self.check("src/weights/x.cpp", '#include "exec/internal/x.h"\n', "module-include")

    def test_cmake_fragment(self):
        self.check("cmake/x.cmake", "target_link_libraries(moe_weights INTERFACE moe_app)\n", "target-link")
        self.check("cmake/x.cmake", "target_link_libraries(moe_weights INTERFACE moe_exec)\n")

    def test_source_extensions(self):
        for suffix in (".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx"):
            with self.subTest(suffix=suffix):
                self.check("src/app/x" + suffix, '#include "llama.h"\n', "llama-header")
                self.check("tools/oracle/x" + suffix, '#include "llama.h"\n', "llama-header")

    def test_build_directories(self):
        for directory in ("build", "build-debug"):
            with self.subTest(directory=directory):
                self.check(f"src/app/{directory}/x.cpp", '#include "llama.h"\n', "llama-header")
                self.check(f"src/app/{directory}/x.cpp", '#include "edge0/engine.h"\n')
                self.check(f"{directory}/x.cpp", '#include "llama.h"\n')

    def test_tests_root(self):
        self.check("tests/x.cpp", '#include "platform/backend.h"\n', "public-include")
        self.check("tests/x.cpp", '#include "edge0/engine.h"\n#include "tests/support/check.h"\n#include <vector>\n')

    def test_c_gguf_allowed(self):
        self.check("src/model/x.cpp", '#include "gguf.h"\n')

    def test_platform_system_headers(self):
        for header in ("unistd.h", "fcntl.h", "sys/stat.h", "sys/types.h", "sys/resource.h", "mach/mach.h"):
            self.check("src/platform/io.cpp", f"#include <{header}>\n")
            for owner in ("weights", "exec", "session", "app"):
                self.check(f"src/{owner}/io.cpp", f"#include <{header}>\n", "module-include")
        self.check("src/platform/io.cpp", "#include <sys/socket.h>\n", "module-include")
        self.check("tests/unit/platform/io.cpp", "#include <unistd.h>\n", "module-include")

    def test_console_setup_exception_is_narrow(self):
        for header in ("windows.h", "io.h", "fcntl.h"):
            self.check("tools/cli/main.cpp", f"#include <{header}>\n")
            self.check("tools/cli/run.cpp", f"#include <{header}>\n", "public-include")
        self.check("tools/cli/main.cpp", "#include <psapi.h>\n", "public-include")
        self.check("tests/unit/app/template_tools_probe.cpp", "#include <io.h>\n")
        self.check("tests/unit/app/template_tools_probe.cpp", "#include <windows.h>\n", "module-include")

    def test_model_runtime_dependencies(self):
        self.check("src/models/api/x.cpp", '#include "session/storage.h"\n#include "exec/execution.h"\n')
        self.check("src/model/x.cpp", '#include "exec/execution.h"\n')
        self.check("src/model/x.cpp", '#include "session/storage.h"\n', "module-include")

    def test_app_execution_dependency(self):
        self.check("src/app/x.cpp", '#include "exec/budget.h"\n')
        self.check("src/app/x.cpp", '#include "weights/storage.h"\n', "module-include")


    def test_model_registration(self):
        include = '#include "models/edge0_35b/compute.h"\n'
        self.check("src/app/generation.cpp", include, "model-include")
        self.check("src/bootstrap.cpp", include, "model-include")
        self.check("src/app/adapters.cpp", include)
        self.check("src/models/edge0_35b/forward.cpp", include)
        self.check("tests/unit/app/generation.cpp", include)

    def test_portal_headers(self):
        for path, header, allowed in (
            ('tools/cli/serve.cpp', 'serve_protocol.h', True),
            ('tools/cli/serve.cpp', 'tools/cli/serve_protocol.h', True),
            ('tests/unit/cli/protocol.cpp', 'tools/cli/serve_protocol.h', True),
            ('tests/integration/test.cpp', 'tools/cli/serve_protocol.h', False),
            ('tools/cli/serve.cpp', 'src/app/private.h', False),
            ('tests/unit/cli/test.cpp', 'src/app/private.h', False),
            ('tools/cli/serve.cpp', 'httplib.h', True),
            ('tests/unit/cli/test.cpp', 'nlohmann/json.hpp', True),
            ('tools/cli/chat.cpp', 'nlohmann/json.hpp', False),
            ('include/edge0/test.h', 'httplib.h', False),
            ('src/app/test.cpp', 'httplib.h', False),
        ):
            with self.subTest(path=path, header=header), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'tools/checks').mkdir(parents=True)
                (root / 'tools/cli').mkdir(parents=True)
                (root / 'tools/cli/serve_protocol.h').write_text('')
                shutil.copyfile(ROOT / 'tools/checks/layers.toml', root / 'tools/checks/layers.toml')
                shutil.copyfile(ROOT / 'tools/checks/third_party.toml', root / 'tools/checks/third_party.toml')
                file = root / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(f'#include "{header}"\n')
                result = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/checks/check_layers.py'), str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if allowed else 1, result.stdout + result.stderr)


    def test_example_headers(self):
        self.check('examples/ios/Bridge.m', '#import "src/app/engine.h"\n', 'public-include')
        self.check('examples/ios/Bridge.m', '#import <Foundation/Foundation.h>\n')
        for directive in ('include', 'import'):
            for header in ('src/app/engine.h', 'ggml.h', 'UIKit/UIKit.h', '../android/bridge.h', 'edge0/../../src/app/engine.h'):
                self.check('examples/ios/bridge.mm', f'#{directive} "{header}"\n', 'public-include')
        self.check('examples/android/bridge.cpp', '#include <jni.h>\n#include <android/log.h>\n')
        self.check('examples/ios/bridge.mm', '#import <Foundation/Foundation.h>\n')
        self.check('examples/native_chat.cpp', '#include "edge0/generation.h"\n')

    def test_example_local_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tools/checks').mkdir(parents=True)
            shutil.copyfile(ROOT / 'tools/checks/layers.toml', root / 'tools/checks/layers.toml')
            app = root / 'examples/ios'
            (app / 'nested').mkdir(parents=True)
            (app / 'Bridge.h').write_text('#import <Foundation/Foundation.h>\n')
            source = app / 'nested/Bridge.mm'
            source.write_text('#import "Bridge.h"\n')
            def run():
                return subprocess.run([sys.executable, '-I', str(ROOT / 'tools/checks/check_layers.py'), str(root)], capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            other = root / 'examples/android'
            other.mkdir()
            (other / 'Other.h').write_text('')
            for include in ('../../android/Other.h', 'Escape.h'):
                if include == 'Escape.h':
                    (app / include).symlink_to(other / 'Other.h')
                source.write_text(f'#import "{include}"\n')
                result = run()
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn('public-include', result.stdout)

    def test_example_cmake(self):
        self.check('examples/android/CMakeLists.txt', 'target_link_libraries(bridge PRIVATE edge0::edge0 android log)\n')
        self.check('examples/ios/CMakeLists.txt', 'target_link_libraries(bridge PRIVATE ggml)\n', 'example-link')
        self.check('examples/ios/CMakeLists.txt', 'add_subdirectory(../.. edge0)\n')
        self.check('examples/ios/CMakeLists.txt', 'add_subdirectory(../../src engine)\n', 'example-subdirectory')

    def test_app_generated_directories(self):
        for directory in ('node_modules', '.gradle', '.cxx', '.kotlin', '.build', 'DerivedData', 'build', 'target', 'dist'):
            self.check(f'examples/ios/{directory}/bad.cpp', '#include "llama.h"\n')
