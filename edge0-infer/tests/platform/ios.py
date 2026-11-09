"""iOS App transport. Device metadata is never included in error messages."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import uuid
from android import Unavailable, manifest_files, sync_assets


class IOS:
    def __init__(self, execute=subprocess.run, environ=None):
        self.env = dict(os.environ if environ is None else environ)
        self.execute = execute
        if self.env.get('MOE_DEVICECTL'):
            path = self.env['MOE_DEVICECTL']
            # Windows cannot execute a .py shebang as a process image.
            self.command = [sys.executable, path] if path.endswith('.py') else [path]
        else:
            self.command = ['xcrun', 'devicectl']
        self.device = self.env.get('MOE_IOS_DEVICE')
        team = self.env.get('MOE_IOS_TEAM')
        if not self.device or not team:
            raise Unavailable('MOE_IOS_DEVICE and MOE_IOS_TEAM are required')
        self.bundle = self.env.get('MOE_IOS_BUNDLE_ID') or f'dev.edge0.{team}.edge0-infer'

    def argv(self, *args):
        return [*self.command, *map(str, args)]

    def call(self, *args, check=True):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'result.json'
            try:
                result = self.execute(self.argv(*args, '--json-output', output), capture_output=True,
                                      stdin=subprocess.DEVNULL, env=self.env)
            except OSError as error:
                raise Unavailable('devicectl is unavailable') from error
            if result.returncode:
                if check:
                    raise RuntimeError(f'devicectl {args[0]} {args[1]} failed (exit {result.returncode})')
                return None
            return json.loads(output.read_text())['result']

    def reachable(self):
        details = self.call('device', 'info', 'details', '--device', self.device, check=False)
        if not details:
            raise Unavailable('iOS device is unreachable')
        if details.get('connectionProperties', {}).get('transportType') != 'wired':
            raise Unavailable('iOS requires a data cable')
        lock = self.call('device', 'info', 'lockState', '--device', self.device, check=False)
        if not lock or lock.get('passcodeRequired', True):
            raise Unavailable('iOS device is locked or lock state is unavailable')
        return details

    def domain(self):
        return ['--device', self.device, '--domain-type', 'appDataContainer', '--domain-identifier', self.bundle]

    @staticmethod
    def safe_path(remote):
        path = PurePosixPath(remote)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in ('assets', 'inv'):
            raise ValueError('Remote path must be under assets/ or inv/')
        return 'Documents/' + str(path)

    def install(self, app):
        self.call('device', 'install', 'app', '--device', self.device, str(app))

    def push(self, local, remote):
        self.call('device', 'copy', 'to', *self.domain(), '--source', str(local), '--destination', self.safe_path(remote))

    def pull(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.call('device', 'copy', 'from', *self.domain(), '--source', self.safe_path(remote), '--destination', str(local))

    def files(self, remote):
        result = self.call('device', 'info', 'files', *self.domain(), '--subdirectory', self.safe_path(remote), check=False)
        if result is None:
            return None
        return result.get('files', [])

    def size(self, remote):
        path = PurePosixPath(remote)
        listing = self.files(str(path.parent))
        return next((item['metadata']['size'] for item in listing or [] if item['relativePath'] == path.name), None)

    def digest(self, remote):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'file'
            try:
                self.pull(remote, path)
            except RuntimeError:
                return None
            return hashlib.sha256(path.read_bytes()).hexdigest()

    def asset_roots(self):
        roots = []
        if self.env.get('MOE_TINY'):
            roots += [(Path(self.env['MOE_TINY']) / name, 'assets/' + name) for name in ('tiny-a', 'tiny-b')]
        if self.env.get('MOE_ASSETS_35B'):
            roots.append((Path(self.env['MOE_ASSETS_35B']), 'assets/35b'))
        return [(path.resolve(), remote) for path, remote in roots]

    def launch(self, argv):
        return self.argv('device', 'process', 'launch', '--device', self.device,
                         '--console', '--terminate-existing', self.bundle, *argv)

    def clean(self):
        marker = self.env.get('MOE_IOS_TERMINATION_REPORT')
        if marker and Path(marker).exists():
            raise RuntimeError('iOS process terminated; failed invocation must be preserved')
        environment = dict(self.env, DEVICECTL_CHILD_MOE_HOST_CLEAN_ONLY='1', DEVICECTL_CHILD_MOE_INVOCATION_DIR='')
        result = self.execute(self.launch([]), env=environment, stdin=subprocess.DEVNULL, capture_output=True)
        # devicectl's own status may be zero when the application fails.
        from ios_cli import console_result
        _, code = console_result(result.stdout)
        if result.returncode or code != 0:
            raise RuntimeError('iOS host cleanup failed')


class Invocation:
    def __init__(self, device):
        self.device = device
        self.name = uuid.uuid4().hex
        self.remote = 'inv/' + self.name
        self.uploads = {}; self.checked = set(); self.outputs = []

    def path(self, value):
        path = Path(value).resolve()
        for root, remote in self.device.asset_roots():
            if path == root or root in path.parents:
                if root not in self.checked:
                    if self.device.digest(remote + '/assets.tsv') != hashlib.sha256((root / 'assets.tsv').read_bytes()).hexdigest():
                        raise RuntimeError('Asset manifest differs; run ios_push.py --tiny explicitly')
                    self.checked.add(root)
                suffix = path.relative_to(root).as_posix()
                return remote if suffix == '.' else remote + '/' + suffix
        if not path.exists():
            return value
        if path not in self.uploads:
            remote = self.remote + f'/input-{len(self.uploads)}'
            self.device.push(path, remote)
            self.uploads[path] = remote
        return self.uploads[path]

    def arguments(self, argv):
        result = []; index = 0
        while index < len(argv):
            value = argv[index]
            if value == '--out':
                if index + 1 == len(argv): raise ValueError('Missing --out value')
                local = Path(argv[index + 1])
                if local.exists(): raise ValueError('Output directory already exists')
                remote = self.remote + '/out'
                self.outputs.append((remote, local)); result += [value, remote]; index += 2
                continue
            if index and argv[index - 1] in ('--model', '--tokens', '--turns', '--replay', '--user-file', '--check'):
                value = self.path(value)
            result.append(value); index += 1
        return result

    def pull_outputs(self):
        for remote, local in self.outputs:
            listing = self.device.files(remote)
            if not listing or not any(not item.get('resources', {}).get('isDirectory') for item in listing):
                continue
            for item in listing:
                suffix = PurePosixPath(item['relativePath'])
                if suffix.is_absolute() or '..' in suffix.parts or item.get('resources', {}).get('isSymbolicLink'):
                    raise RuntimeError('Invalid output path')
            # A single recursive transfer preserves bytes and hierarchy, without one
            # devicectl startup per layer dump. The App has already exited.
            self.device.pull(remote, local)
