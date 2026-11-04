"""Install the iOS App without uninstalling it; resumably synchronize its assets."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios import IOS, Unavailable, sync_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, default=Path('build/ios/bin/Release/edge0-infer.app'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--tiny', action='store_true', help='Also synchronize tiny-a and tiny-b (requires MOE_TINY)')
    args = parser.parse_args()
    try:
        device = IOS(); device.reachable()
        if not device.env.get('MOE_ASSETS_35B'):
            raise ValueError('MOE_ASSETS_35B is required')
        if args.tiny and not device.env.get('MOE_TINY'):
            raise ValueError('--tiny requires MOE_TINY')
        if not args.tiny:
            device.env.pop('MOE_TINY', None)
        roots = device.asset_roots()
        if not args.check:
            device.install(args.app)
            print('App installed without uninstalling', flush=True)
        changed = False
        for local, remote in roots:
            start = time.monotonic()
            differences = sync_assets(device, local, remote, args.check)
            changed |= bool(differences)
            print(f'{remote}: {"different: " + ", ".join(differences) if differences else "consistent"}; '
                  f'{time.monotonic()-start:.2f}s', flush=True)
        return int(args.check and changed)
    except Unavailable as error:
        print(f'SKIP: {error}', file=sys.stderr); return 77
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
