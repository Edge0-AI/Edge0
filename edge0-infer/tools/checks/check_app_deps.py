"""Compare example App direct dependencies and resolved versions to their allowlist."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tomllib

_spec = importlib.util.spec_from_file_location('checks_common', Path(__file__).with_name('common.py'))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def check(root):
    manifest = root / 'tools/checks/app_dependencies.toml'
    allowed = tomllib.loads(manifest.read_text())
    errors = []

    def error(path, app, name, reason):
        file = root / path
        text = file.read_text() if file.is_file() else ''
        offset = text.find(name)
        line = text[:offset].count('\n') + 1 if offset >= 0 else 1
        errors.append(f'app-deps: {path}:{line}: {app} {name}: {reason}')

    def read(path, loader):
        try:
            return loader((root / path).read_text())
        except (OSError, ValueError) as exc:
            error(path, path.split('/')[1], 'manifest', f'cannot read: {exc}')
            return {}

    def compare(path, app, actual, expected):
        for name in sorted(actual.keys() | expected.keys()):
            if name not in expected:
                error(path, app, name, 'unregistered dependency')
            elif name not in actual:
                error(path, app, name, 'missing dependency')
            elif actual[name] != expected[name]:
                error(path, app, name, f'version {actual[name]!r}; expected {expected[name]!r}')

    desktop = 'examples/desktop/'
    if (root / desktop).exists():
        path = desktop + 'package.json'
        package = read(path, json.loads)
        expected = allowed['desktop']['npm']
        for section in ('dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies'):
            compare(path, 'desktop npm ' + section, package.get(section, {}), expected.get(section, {}))
        lock_path = desktop + 'package-lock.json'
        lock = read(lock_path, json.loads).get('packages', {})
        for section, dependencies in expected.items():
            compare(lock_path, 'desktop npm root ' + section, lock.get('', {}).get(section, {}), dependencies)
            for name, version in dependencies.items():
                resolved = lock.get('node_modules/' + name, {}).get('version')
                if resolved != version:
                    error(lock_path, 'desktop npm', name, f'locked version {resolved!r}; expected {version!r}')
        path = desktop + 'src-tauri/Cargo.toml'
        cargo = read(path, tomllib.loads)
        expected = allowed['desktop']['cargo']

        def cargo_sections(value):
            result = {key: value.get(key, {}) for key in ('dependencies', 'build-dependencies', 'dev-dependencies')}
            for target, config in value.get('target', {}).items():
                for kind in ('dependencies', 'build-dependencies', 'dev-dependencies'):
                    result[f'target.{target}.{kind}'] = config.get(kind, {})
            return result

        actual_sections, expected_sections = cargo_sections(cargo), cargo_sections(expected)
        direct = {}
        for section in actual_sections.keys() | expected_sections.keys():
            actual = {}
            for name, config in actual_sections.get(section, {}).items():
                version = config if isinstance(config, str) else config.get('version', '')
                if isinstance(config, dict) and any(k in config for k in ('git', 'path', 'workspace', 'package')):
                    error(path, 'desktop cargo', name, 'alternate dependency source or alias forbidden')
                # Cargo needs '=' to pin a version; a bare version is a compatible range.
                actual[name] = version[1:] if version.startswith('=') else 'un-pinned:' + version
            compare(path, 'desktop cargo ' + section, actual, expected_sections.get(section, {}))
            direct.update(expected_sections.get(section, {}))
        lock_path = desktop + 'src-tauri/Cargo.lock'
        packages = read(lock_path, tomllib.loads).get('package', [])
        root_name = cargo.get('package', {}).get('name')
        root_package = next((p for p in packages if p.get('name') == root_name), {})
        root_deps = root_package.get('dependencies', [])
        for name, version in direct.items():
            entries = [p for p in packages if p.get('name') == name]
            references = [d.split() for d in root_deps if d.split()[0] == name]
            resolved = references[0][1] if len(references) == 1 and len(references[0]) > 1 else (
                entries[0].get('version') if len(references) == 1 and len(entries) == 1 else None)
            if resolved != version or not any(p.get('version') == resolved for p in entries):
                error(lock_path, 'desktop cargo', name, f'locked direct version {resolved!r}; expected {version!r}')

    android = 'examples/android/'
    if (root / android).exists():
        path = android + 'gradle/libs.versions.toml'
        catalog = read(path, tomllib.loads)
        expected = allowed['android']['gradle']
        versions = catalog.get('versions', {})
        for section, identity in (('libraries', 'module'), ('plugins', 'id')):
            actual = {}
            for alias, config in catalog.get(section, {}).items():
                if not isinstance(config, dict):
                    error(path, 'android', alias, 'use structured version-catalog entries')
                    continue
                name = config.get(identity, '')
                version = config.get('version', 'bom')
                if isinstance(version, dict):
                    version = versions.get(version.get('ref'), '')
                if name in actual:
                    error(path, 'android', name, 'duplicate catalog entry')
                actual[name] = version
            compare(path, 'android ' + section, actual, expected[section])
        for file, relative in _common.files(root):
            if relative.startswith(android) and file.name.endswith(('.gradle', '.gradle.kts')):
                text = file.read_text()
                for match in re.finditer(r'["\']([\w.\-]+:[\w.\-]+(?::[^"\']*)?)["\']', text):
                    error(relative, 'android', match[1], 'direct coordinate forbidden; use the version catalog')
                for match in re.finditer(r'\bid\s*\(\s*["\']([^"\']+)["\']\s*\)\s*version', text):
                    error(relative, 'android', match[1], 'direct plugin version forbidden; use the version catalog')
        path = android + 'gradle/wrapper/gradle-wrapper.properties'
        text = (root / path).read_text() if (root / path).exists() else ''
        version = re.search(r'gradle-([\d.]+)-(?:bin|all)\.zip', text)
        if not version or version[1] != expected['wrapper']:
            error(path, 'android', 'Gradle', 'wrapper version differs from allowlist')
        if not re.search(r'^distributionSha256Sum=[a-f0-9]{64}$', text, re.M):
            error(path, 'android', 'Gradle', 'missing wrapper distribution checksum')

    ios = root / 'examples/ios'
    if ios.exists():
        for file, relative in _common.files(root):
            if relative.startswith('examples/ios/') and file.name == 'Package.swift':
                if re.search(r'\.package\s*\(', file.read_text()):
                    error(relative, 'ios', 'package', 'unregistered dependency; iOS uses system frameworks only')
    print('\n'.join(errors) if errors else 'app-deps: passed')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()))
