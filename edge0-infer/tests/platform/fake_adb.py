#!/usr/bin/env python3
"""Local-only adb double. Every test gives it its own disposable root."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

args = sys.argv[1:]
if args[:1] == ['-s']:
    args = args[2:]
if os.environ.get('FAKE_ADB_READ_CONTROL_STDIN') and args[:2] != ['shell', '-T']:
    if sys.stdin.buffer.read():
        raise SystemExit('Auxiliary adb command consumed user stdin')
with open(os.environ['FAKE_ADB_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\n')
if os.environ.get('FAKE_ADB_OFFLINE'):
    print('offline', file=sys.stderr)
    raise SystemExit(1)
if args[0] == 'get-state':
    print('device')
elif args[0] == 'push':
    source, dest = map(Path, args[1:])
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
elif args[0] == 'pull':
    source, dest = map(Path, args[1:])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
elif args[0] == 'forward':
    if args[1] != '--remove':
        print('43210')
elif args[0] == 'shell':
    import shlex
    command = args[-1]
    words = shlex.split(command)
    if words[0] == 'sha256sum':
        path = Path(words[1])
        if not path.is_file():
            raise SystemExit(1)
        print(hashlib.sha256(path.read_bytes()).hexdigest(), words[1])
    elif words[:3] == ['stat', '-c', '%s']:
        path = Path(words[3])
        if not path.exists():
            raise SystemExit(1)
        print(path.stat().st_size)
    else:
        # All paths belong to a TemporaryDirectory chosen by the test.
        raise SystemExit(subprocess.call(['/bin/sh', '-c', command]))
else:
    raise SystemExit('Unexpected fake adb operation: ' + repr(args))
