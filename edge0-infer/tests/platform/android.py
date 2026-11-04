"""Android transport, manifest checks and per-invocation file mapping.

MOE_ADB can select a test double; production defaults to adb. No command is run
on import. All remote arguments are quoted for the device shell.
"""
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid


class Unavailable(RuntimeError):
    pass


class Android:
    def __init__(self, execute=subprocess.run, environ=None):
        self.env = dict(os.environ if environ is None else environ)
        self.execute = execute
        self.adb = self.env.get('MOE_ADB', 'adb')
        self.serial = self.env.get('ANDROID_SERIAL')
        self.root = self.env.get('MOE_ANDROID_ROOT', '')
        path = PurePosixPath(self.root)
        if not path.is_absolute() or len(path.parts) < 3 or '..' in path.parts:
            raise ValueError('MOE_ANDROID_ROOT must be an absolute, dedicated device directory')
        self.root = str(path)

    def argv(self, *args):
        return [self.adb, *(['-s', self.serial] if self.serial else []), *map(str, args)]

    def call(self, *args, check=True):
        try:
            result = self.execute(self.argv(*args), capture_output=True, text=True, stdin=subprocess.DEVNULL)
        except OSError as error:
            raise Unavailable(f'Android transport unavailable: {error}') from error
        if check and result.returncode:
            raise RuntimeError(f'Android command failed ({result.returncode}): {result.stderr.strip()}')
        return result

    def reachable(self):
        result = self.call('get-state', check=False)
        if result.returncode or result.stdout.strip() != 'device':
            raise Unavailable('Android device is unreachable or not authorized')

    def shell(self, *args, check=True):
        return self.call('shell', shlex.join(map(str, args)), check=check)

    def push(self, local, remote):
        self.shell('mkdir', '-p', str(PurePosixPath(remote).parent))
        self.call('push', str(local), remote)

    def pull(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self.call('pull', remote, str(local))

    def digest(self, remote):
        result = self.shell('sha256sum', remote, check=False)
        return result.stdout.split()[0] if not result.returncode and result.stdout.split() else None

    def size(self, remote):
        result = self.shell('stat', '-c', '%s', remote, check=False)
        try:
            return int(result.stdout.strip()) if not result.returncode else None
        except ValueError:
            return None

    def forward(self, port):
        value = self.call('forward', 'tcp:0', f'tcp:{port}').stdout.strip()
        if not value.isdecimal() or not 0 < int(value) <= 65535:
            raise RuntimeError('Invalid forwarded port')
        return value

    def unforward(self, port):
        self.call('forward', '--remove', f'tcp:{port}')

    def asset_roots(self):
        result = []
        if self.env.get('MOE_TINY'):
            result += [(Path(self.env['MOE_TINY']) / name, f'{self.root}/{name}') for name in ('tiny-a', 'tiny-b')]
        if self.env.get('MOE_ASSETS_35B'):
            result.append((Path(self.env['MOE_ASSETS_35B']), f'{self.root}/35b'))
        return [(path.resolve(), remote) for path, remote in result]


class AndroidApp(Android):
    """Debug App private storage, accessed as its uid through run-as."""
    def __init__(self, package, execute=subprocess.run, environ=None):
        env = dict(os.environ if environ is None else environ)
        env.setdefault('MOE_ANDROID_ROOT', '/data/local/tmp/edge0-example-transfer')
        super().__init__(execute=execute, environ=env)
        self.staging_root = self.root
        self.package = package
        self.root = 'files/assets'

    def reachable(self):
        super().reachable()
        self.shell('true')

    def shell(self, *args, check=True):
        return super().shell('run-as', self.package, *args, check=check)

    def push(self, local, remote):
        # Only a single file is staged at a time. Publish assets.tsv last via sync_assets.
        stage = f'{self.staging_root}/app-transfer-{uuid.uuid4().hex}'
        super().shell('mkdir', '-p', self.staging_root)
        self.shell('mkdir', '-p', str(PurePosixPath(remote).parent))
        try:
            self.call('push', str(local), stage)
            super().shell('chmod', '644', stage)
            self.shell('cp', stage, remote)
        finally:
            super().shell('rm', '-f', stage, check=False)


def manifest_files(directory):
    directory = Path(directory)
    text = (directory / 'assets.tsv').read_text()
    result = {}
    for row in csv.DictReader(io.StringIO(text), delimiter='\t'):
        name = row['filename']
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid asset manifest filename')
        result[name] = int(row['bytes'])
    for name in ('assets.tsv', 'model.json', 'provenance.json'):
        if (directory / name).is_file():
            result[name] = (directory / name).stat().st_size
    return result


def sync_assets(device, local, remote, check=False):
    local = Path(local)
    digest = hashlib.sha256((local / 'assets.tsv').read_bytes()).hexdigest()
    same_manifest = device.digest(remote + '/assets.tsv') == digest
    differences = []
    files = manifest_files(local)
    # Publish the manifest last, so an interrupted transfer cannot claim completion.
    for name in sorted(files, key=lambda name: name == 'assets.tsv'):
        source, target = local / name, remote + '/' + name
        if source.stat().st_size != files[name]:
            raise ValueError(f'Local asset size mismatch: {name}')
        same = same_manifest and device.size(target) == files[name]
        if not same:
            differences.append(name)
            if not check:
                device.push(source, target)
    return differences


class Invocation:
    def __init__(self, device):
        self.device = device
        self.remote = f'{device.root}/calls/{uuid.uuid4().hex}'
        self.outputs = []
        self.checked = set()
        self.uploads = {}
        self.pulled = {}

    def __enter__(self):
        self.device.shell('mkdir', '-p', self.remote)
        return self

    def __exit__(self, *unused):
        self.device.shell('rm', '-rf', self.remote)

    def path(self, value):
        path = Path(value).resolve()
        for root, remote in self.device.asset_roots():
            if path == root or root in path.parents:
                if root not in self.checked:
                    expected = hashlib.sha256((root / 'assets.tsv').read_bytes()).hexdigest()
                    if self.device.digest(remote + '/assets.tsv') != expected:
                        raise RuntimeError('Asset manifest differs; run android_push.py --tiny explicitly')
                    self.checked.add(root)
                suffix = path.relative_to(root).as_posix()
                return remote if suffix == '.' else remote + '/' + suffix
        if not path.exists():
            return value
        if path not in self.uploads:
            target = f'{self.remote}/input-{len(self.uploads)}'
            self.device.push(path, target)
            self.uploads[path] = target
        return self.uploads[path]

    def arguments(self, argv):
        result, index = [], 0
        while index < len(argv):
            value = argv[index]
            if value == '--out':
                if index + 1 == len(argv):
                    raise ValueError('Missing --out value')
                local = Path(argv[index + 1])
                if local.exists():
                    raise ValueError('Output directory already exists')
                target = self.remote + '/out'
                self.outputs.append((target, local))
                result += [value, target]
                index += 2
                continue
            # Only path-valued arguments are mapped. Literal text must remain text.
            if index and argv[index - 1] in ('--model', '--tokens', '--turns', '--replay', '--user-file', '--check'):
                value = self.path(value)
            result.append(value)
            index += 1
        return result

    def pull_outputs(self):
        for remote, local in self.outputs:
            listing = self.device.shell('find', remote, '-type', 'f', check=False)
            if listing.returncode:
                continue
            for filename in listing.stdout.splitlines():
                path = PurePosixPath(filename)
                base = PurePosixPath(remote)
                if base not in path.parents or '..' in path.parts:
                    raise RuntimeError('Invalid remote output path')
                digest = self.device.digest(filename)
                if digest is not None and self.pulled.get(filename) == digest:
                    continue
                self.device.pull(filename, local / path.relative_to(base).as_posix())
                self.pulled[filename] = digest
