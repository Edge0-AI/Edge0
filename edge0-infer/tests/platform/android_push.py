"""Explicit, resumable Android asset installation; --check never writes remotely."""
import argparse
import hashlib
import os
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from android import Android, AndroidApp, Unavailable, sync_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/android-cpu/bin/edge0-infer'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--tiny', action='store_true', help='Also synchronize tiny-a and tiny-b (requires MOE_TINY)')
    parser.add_argument('--app', help='Application package: synchronize assets only')
    args = parser.parse_args()
    try:
        env = dict(os.environ)
        if args.app:
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+', args.app):
                raise ValueError('Invalid application package')
        device = AndroidApp(args.app, environ=env) if args.app else Android(environ=env)
        device.reachable()
        if not device.env.get('MOE_ASSETS_35B'):
            raise ValueError('MOE_ASSETS_35B is required')
        if args.tiny and not device.env.get('MOE_TINY'):
            raise ValueError('--tiny requires MOE_TINY')
        if not args.tiny:
            device.env.pop('MOE_TINY', None)
        roots = device.asset_roots()
        changed = False
        if not args.app:
            target = device.root + '/edge0-infer'
            changed = device.digest(target) != hashlib.sha256(args.binary.read_bytes()).hexdigest()
            if changed and not args.check:
                device.push(args.binary, target)
                device.shell('chmod', '755', target)
            print('binary:', 'different' if changed else 'consistent')
        for source, remote in roots:
            different = sync_assets(device, source, remote, args.check)
            changed |= bool(different)
            print(source.name + ':', ', '.join(different) if different else 'consistent')
        return int(args.check and changed)
    except Unavailable as error:
        print(f'SKIP: {error}', file=sys.stderr)
        return 77
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
