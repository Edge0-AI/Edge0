"""The engine as a CMake subproject: a host project configures it with add_subdirectory() without leaks.

Configures the minimal host project in tests/unit/build/subproject in a temporary directory and
checks what it records: no CLI, test or example targets, no BUILD_SHARED_LIBS in the host cache, a
host library without an explicit type stays static, no runtime output directory, the build identity
header inside the engine's binary directory, preset GGML defaults that keep a host override, and
C++17 reaching a consumer that asks for C++11 (only that consumer's object file is compiled).
The iOS case (no MOE_IOS_TEAM) and the Android case (NDK toolchain given only with -D) are skipped
when the iOS SDK or $ANDROID_NDK is not available.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CMAKE, GENERATOR, MAKE_PROGRAM, C_COMPILER, CXX_COMPILER, ROOT = sys.argv[1:7]
del sys.argv[1:7]
ROOT = Path(ROOT)
HOST = ROOT / 'tests' / 'unit' / 'build' / 'subproject'


def without(*names):
    return {key: value for key, value in os.environ.items() if key not in names}


class SubprojectTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-subproject-')
        self.addCleanup(holder.cleanup)
        self.work = Path(holder.name)

    def configure(self, name, arguments, env=None):
        build, probes = self.work / name, self.work / f'{name}.probes'
        command = [CMAKE, '-S', str(HOST), '-B', str(build), f'-DEDGE0_SOURCE_DIR={ROOT}',
                   f'-DPROBE_OUTPUT={probes}', *arguments]
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        values = dict(line.split('=', 1) for line in probes.read_text().splitlines() if '=' in line)
        return build, values

    def build_consumer(self, build, env=None):
        result = subprocess.run([CMAKE, '--build', str(build), '--target', 'probe_consumer'],
                                capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])

    def assert_no_leaks(self, values):
        for target in ('edge0-infer', 'moe_example_native_chat', 'moe_test_app', 'moe_public_header_check'):
            self.assertEqual(values[f'target.{target}'], 'absent', target)
        self.assertEqual(values['host_lib_type'], 'STATIC_LIBRARY')
        self.assertEqual(values['cache.BUILD_SHARED_LIBS'], '<unset>')
        self.assertEqual(values['engine.CMAKE_RUNTIME_OUTPUT_DIRECTORY'], '')
        header = Path(values['engine.MOE_BUILD_IDENTITY_HEADER'])
        self.assertEqual(header.parents[1], Path(values['engine.binary_dir']) / 'generated')
        for option in ('GGML_BLAS', 'GGML_ACCELERATE', 'GGML_OPENMP'):
            self.assertEqual(values[f'cache.{option}'], 'OFF', option)

    def generator_arguments(self):
        arguments = ['-G', GENERATOR]
        if MAKE_PROGRAM:
            arguments.append(f'-DCMAKE_MAKE_PROGRAM={MAKE_PROGRAM}')
        return arguments

    def test_native_host(self):
        # GGML_LLAMAFILE=ON stands for any host override of a defaulted GGML option.
        build, values = self.configure('native', [*self.generator_arguments(), '-DCMAKE_BUILD_TYPE=Release',
                                                  f'-DCMAKE_C_COMPILER={C_COMPILER}',
                                                  f'-DCMAKE_CXX_COMPILER={CXX_COMPILER}', '-DGGML_LLAMAFILE=ON'])
        self.assert_no_leaks(values)
        self.assertEqual(values['cache.GGML_LLAMAFILE'], 'ON')
        self.assertEqual(values['cache.GGML_METAL'], 'ON' if sys.platform == 'darwin' else 'OFF')
        self.build_consumer(build)

    def test_ios_without_signing_team(self):
        if sys.platform != 'darwin' or not shutil.which('xcodebuild'):
            self.skipTest('not executed: Xcode not available')
        sdk = subprocess.run(['xcrun', '--sdk', 'iphoneos', '--show-sdk-path'], capture_output=True, text=True)
        if sdk.returncode != 0:
            self.skipTest('not executed: iOS SDK not available')
        _, values = self.configure('ios', ['-G', 'Xcode', '-DCMAKE_SYSTEM_NAME=iOS', '-DCMAKE_OSX_ARCHITECTURES=arm64',
                                           '-DCMAKE_OSX_DEPLOYMENT_TARGET=17.0'],
                                   env=without('MOE_IOS_TEAM', 'MOE_IOS_BUNDLE_ID'))
        self.assert_no_leaks(values)
        self.assertEqual(values['cache.GGML_METAL'], 'ON')
        self.assertEqual(values['cache.GGML_NATIVE'], 'OFF')
        self.assertEqual(values['cache.GGML_CPU_ARM_ARCH'], 'armv8.4-a+dotprod+fp16')

    def test_android_toolchain_from_command_line(self):
        ndk = os.environ.get('ANDROID_NDK', '')
        toolchain = Path(ndk) / 'build' / 'cmake' / 'android.toolchain.cmake'
        if not ndk or not toolchain.is_file():
            self.skipTest('not executed: ANDROID_NDK not set')
        if GENERATOR not in ('Ninja', 'Unix Makefiles'):
            self.skipTest(f'not executed: generator {GENERATOR} cannot target Android')
        # As an IDE build does: the toolchain arrives only through -D, $ANDROID_NDK is not set.
        env = without('ANDROID_NDK')
        build, values = self.configure('android', [*self.generator_arguments(), '-DCMAKE_BUILD_TYPE=Release',
                                                   f'-DCMAKE_TOOLCHAIN_FILE={toolchain}', '-DANDROID_ABI=arm64-v8a',
                                                   '-DANDROID_PLATFORM=android-28'], env=env)
        self.assert_no_leaks(values)
        self.assertEqual(values['cache.GGML_METAL'], 'OFF')
        self.assertEqual(values['cache.GGML_NATIVE'], 'OFF')
        self.assertEqual(values['cache.GGML_CPU_ARM_ARCH'], 'armv8.6-a+dotprod+i8mm+fp16')
        self.build_consumer(build, env=env)


if __name__ == '__main__':
    unittest.main(verbosity=2)
