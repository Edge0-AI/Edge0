"""Check include and target dependencies against layers.toml."""
from pathlib import Path
import re
import sys
import tomllib
import importlib.util

_spec = importlib.util.spec_from_file_location("checks_common", Path(__file__).with_name("common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
cmake_commands, files = _common.cmake_commands, _common.files

# Process entry points outside src/platform/ that must configure the Windows console
# and standard streams themselves (UTF-8 code pages, binary stdio). Only these files,
# and only these headers.
CONSOLE_SETUP = {
    "tools/cli/main.cpp": {"windows.h", "io.h", "fcntl.h"},
    "tests/unit/app/template_tools_probe.cpp": {"io.h", "fcntl.h"},
}

STANDARD = set("algorithm array atomic bitset cassert cctype cerrno cfenv cfloat chrono cinttypes climits clocale cmath codecvt complex condition_variable csetjmp csignal cstdarg cstddef cstdint cstdio cstdlib cstring ctime cuchar cwchar cwctype deque exception execution filesystem forward_list fstream functional future initializer_list iomanip ios iosfwd iostream istream iterator limits list locale map memory memory_resource mutex new numeric optional ostream queue random ratio regex scoped_allocator set shared_mutex sstream stack stdexcept streambuf string string_view system_error thread tuple type_traits typeindex typeinfo unordered_map unordered_set utility valarray variant vector assert.h ctype.h errno.h float.h inttypes.h limits.h math.h setjmp.h signal.h stdarg.h stdbool.h stddef.h stdint.h stdio.h stdlib.h string.h time.h wchar.h wctype.h".split())


def check(root):
    config = tomllib.loads((root / "tools/checks/layers.toml").read_text())
    table = config["modules"]
    registration = config.get("registration", {}).get("files", [])
    errors = []
    libraries = _common.third_party(root)

    def module(path):
        return next((m for m in sorted(table, key=len, reverse=True) if path.startswith(m + "/")), None)

    def error(path, line, rule):
        errors.append(f"layers: {path}:{line}: {rule}")

    for file, path in files(root):
        if file.suffix in _common.SOURCE_SUFFIXES:
            text = file.read_text()
            owner = module(path.removeprefix("src/")) if path.startswith("src/") else None
            if path.startswith("tests/unit/"):
                owner = module(path.removeprefix("tests/unit/"))
            public = path.startswith(("include/edge0/", "tools/cli/", "tests/integration/", "tests/device/", "tests/unit/cli/", "examples/"))
            public = public or (path.startswith("tests/") and len(Path(path).parts) == 2)
            for match in re.finditer(r'^\s*#\s*(?:include|import)\s*[<"]([^>"]+)[>"]', text, re.M):
                inc = match[1]
                line = text[:match.start()].count("\n") + 1
                if Path(inc).name.startswith("llama"):
                    error(path, line, "llama-header: forbidden llama header")
                target = module(inc.removeprefix("src/"))
                if (path.startswith("src/") and target and target.startswith("models/")
                        and target != "models/api" and owner != target and path not in registration):
                    error(path, line, "model-include: concrete model headers only from the registration point")
                if not public and owner is None:
                    continue
                support = path.startswith("tests/") and inc.startswith("tests/support/")
                if _common.portal_header(root, file, path, inc):
                    continue
                if any(inc in library.get("headers", []) and _common.dependency_allowed(path, library)
                       for library in libraries.values()):
                    continue
                public_header = inc.startswith("edge0/") and ".." not in Path(inc).parts
                if inc in STANDARD or public_header or support:
                    continue
                if path.startswith("examples/"):
                    parts = Path(path).parts
                    if len(parts) >= 3 and parts[1] in {"desktop", "android", "ios"}:
                        app = (root / "examples" / parts[1]).resolve()
                        if any(candidate.is_file() and candidate.resolve().is_relative_to(app)
                               for candidate in (file.parent / inc, app / inc)):
                            continue
                    platform_headers = {"android": {"jni.h", "android/log.h"},
                                        "ios": {"Foundation/Foundation.h"}}
                    if inc in platform_headers.get(parts[1], set()):
                        continue
                if path.startswith("src/platform/") and inc in {"unistd.h", "fcntl.h", "sys/stat.h", "sys/types.h", "sys/uio.h", "sys/resource.h", "mach/mach.h", "sys/sysctl.h", "mach/mach_host.h", "objc/message.h", "objc/runtime.h", "os/proc.h", "dlfcn.h", "windows.h", "psapi.h"}:
                    continue
                if path.startswith("tools/cli/ios/") and inc in {"UIKit/UIKit.h", "Foundation/Foundation.h", "os/proc.h", "pthread.h", "unistd.h"}:
                    continue
                if inc in CONSOLE_SETUP.get(path, ()):
                    continue
                if public:
                    error(path, line, "public-include: only standard and public headers allowed")
                elif target:
                    if (target != owner and target not in table[owner]) or (target != owner and "/internal/" in inc):
                        error(path, line, "module-include: forbidden dependency")
                elif not (Path(inc).name.startswith("ggml") or inc == "gguf.h"):
                    error(path, line, "module-include: unrecognized header")
        if file.suffix == ".cmake" or path == "CMakeLists.txt" or (path.startswith(("src/", "examples/")) and file.name == "CMakeLists.txt"):
            names = {"moe_" + m.replace("/", "_"): m for m in table}
            for command, args, line in cmake_commands(file.read_text()):
                if path.startswith("examples/") and args:
                    if command == "target_link_libraries":
                        if any(dep not in {"PRIVATE", "PUBLIC", "INTERFACE", "edge0::edge0", "android", "log"} for dep in args[1:]):
                            error(path, line, "example-link: only edge0::edge0 and listed system libraries allowed")
                    if command == "add_subdirectory" and (file.parent / args[0]).resolve() != root.resolve():
                        error(path, line, "example-subdirectory: only the repository root allowed")
                if command != "target_link_libraries" or not args or args[0] not in names:
                    continue
                owner = names[args[0]]
                for dep in args[1:]:
                    if dep in names and names[dep] not in table[owner]:
                        error(path, line, "target-link: forbidden module dependency")
    print("\n".join(errors) if errors else "layers: passed")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
