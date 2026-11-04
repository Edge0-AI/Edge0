"""Exercise dependency admission independently of installed App toolchains."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[3]


def table(name, values):
    return '[' + name + ']\n' + ''.join(json.dumps(k) + ' = ' + json.dumps(v) + '\n' for k, v in values.items())


class AppDepsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write('tools/checks/app_dependencies.toml', (ROOT / 'tools/checks/app_dependencies.toml').read_text())
        self.manifest = tomllib.loads((self.root / 'tools/checks/app_dependencies.toml').read_text())
        npm = copy.deepcopy(self.manifest['desktop']['npm'])
        self.write_json('examples/desktop/package.json', npm)
        packages = {'': copy.deepcopy(npm)}
        for values in npm.values():
            packages.update({'node_modules/' + name: {'version': version} for name, version in values.items()})
        self.write_json('examples/desktop/package-lock.json', {'packages': packages})
        cargo = self.manifest['desktop']['cargo']
        text = '[package]\nname = "example"\nversion = "0.1.0"\n'
        dependencies = {}
        for section in ('dependencies', 'build-dependencies'):
            text += table(section, {k: '=' + v for k, v in cargo[section].items()})
            dependencies.update(cargo[section])
        for target, sections in cargo['target'].items():
            text += table('target.' + json.dumps(target) + '.dependencies', {k: '=' + v for k, v in sections['dependencies'].items()})
            dependencies.update(sections['dependencies'])
        self.write('examples/desktop/src-tauri/Cargo.toml', text)
        lock = 'version = 4\n[[package]]\nname = "example"\nversion = "0.1.0"\ndependencies = ' + json.dumps(list(dependencies)) + '\n'
        for name, version in dependencies.items():
            lock += '[[package]]\n' + f'name = "{name}"\nversion = "{version}"\n'
        self.write('examples/desktop/src-tauri/Cargo.lock', lock)
        gradle = self.manifest['android']['gradle']
        text = ''
        for section, identity in (('libraries', 'module'), ('plugins', 'id')):
            for i, (name, version) in enumerate(gradle[section].items()):
                text += table(section + '.entry' + str(i), {identity: name, **({'version': version} if version != 'bom' else {})})
        self.write('examples/android/gradle/libs.versions.toml', text)
        self.write('examples/android/gradle/wrapper/gradle-wrapper.properties',
                   'distributionUrl=https\\://services.gradle.org/distributions/gradle-' + gradle['wrapper'] + '-bin.zip\n'
                   'distributionSha256Sum=' + 'a' * 64 + '\n')
        self.write('examples/ios/ChatCore/Package.swift', 'let dependencies: [Package.Dependency] = []\n')

    def write(self, path, text):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)

    def write_json(self, path, value):
        self.write(path, json.dumps(value))

    def run_check(self, expected=None):
        result = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/checks/check_app_deps.py'), str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1 if expected else 0, result.stdout + result.stderr)
        if expected:
            self.assertIn(expected, result.stdout)
            self.assertIn('app-deps: examples/', result.stdout)

    def test_clean(self):
        self.run_check()

    def test_npm_mutations(self):
        original = copy.deepcopy(self.manifest['desktop']['npm'])
        for mutation, expected in (
                (lambda p: p['dependencies'].update(extra='1.0.0'), 'extra: unregistered'),
                (lambda p: p['dependencies'].pop('react'), 'react: missing'),
                (lambda p: p['dependencies'].update(react='19.3.0'), "react: version '19.3.0'"),
                (lambda p: p['dependencies'].update(react='^19.3.1'), "react: version '^19.3.1'")):
            with self.subTest(expected=expected):
                value = copy.deepcopy(original); mutation(value)
                self.write_json('examples/desktop/package.json', value)
                self.run_check(expected)

    def test_npm_lock(self):
        path = 'examples/desktop/package-lock.json'
        value = json.loads((self.root / path).read_text())
        value['packages']['node_modules/react']['version'] = '0.0.0'
        self.write_json(path, value)
        self.run_check('react: locked version')

    def test_cargo_lock(self):
        path = self.root / 'examples/desktop/src-tauri/Cargo.lock'
        path.write_text(path.read_text().replace('version = "2.12.5"', 'version = "0.0.0"'))
        self.run_check('locked direct version')

    def test_cargo_range(self):
        path = self.root / 'examples/desktop/src-tauri/Cargo.toml'
        path.write_text(path.read_text().replace('=2.12.5', '2.12.5'))
        self.run_check('un-pinned:2.12.5')

    def test_gradle_coordinate(self):
        self.write('examples/android/app/build.gradle.kts', 'implementation("unlisted:library:1.0")\n')
        self.run_check('unlisted:library:1.0: direct coordinate')

    def test_ios_package(self):
        self.write('examples/ios/ChatCore/Package.swift', '.package(url: "https://example.invalid/lib", from: "1.0.0")\n')
        self.run_check('ios package: unregistered')

    def test_generated_files_ignored(self):
        self.write('examples/android/build/other/build.gradle.kts', 'implementation("bad:lib:1")')
        self.write('examples/ios/.build/Package.swift', '.package(url: "bad")')
        self.run_check()

    def test_missing_lock(self):
        (self.root / 'examples/desktop/src-tauri/Cargo.lock').unlink()
        self.run_check('cannot read')
