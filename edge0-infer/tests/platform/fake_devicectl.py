#!/usr/bin/env python3
"""Filesystem-backed devicectl test double; never contacts a device."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

args = sys.argv[1:]
root = Path(os.environ['FAKE_IOS_ROOT'])
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps(dict(args=args, env={k:v for k,v in os.environ.items() if k.startswith('DEVICECTL_CHILD_')}))+'\n')
def option(name): return args[args.index(name)+1]

def remote(value):
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('escape')
    return path

result = {}
if os.environ.get('FAKE_IOS_UNREACHABLE'): raise SystemExit(1)
if args[:3] == ['device','info','details']:
    result = dict(connectionProperties=dict(transportType=os.environ.get('FAKE_IOS_TRANSPORT','wired')),
                  hardwareProperties=dict(productType='SyntheticPhone'),deviceProperties=dict(osVersionNumber='1'))
elif args[:3] == ['device','info','lockState']:
    result = dict(passcodeRequired=bool(os.environ.get('FAKE_IOS_LOCKED')))
elif args[:3] == ['device','info','files']:
    path = remote(option('--subdirectory'))
    if not path.is_dir(): raise SystemExit(1)
    result = dict(files=[dict(relativePath=str(p.relative_to(path)),metadata=dict(size=p.stat().st_size),
                              resources=dict(isDirectory=p.is_dir())) for p in path.rglob('*')])
elif args[:3] == ['device','copy','to']:
    source=Path(option('--source'));target=remote(option('--destination'));target.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir(): shutil.copytree(source,target,dirs_exist_ok=True)
    else: shutil.copyfile(source,target)
elif args[:3] == ['device','copy','from']:
    source=remote(option('--source'));target=Path(option('--destination'))
    if not source.exists(): raise SystemExit(1)
    if source.is_dir(): shutil.copytree(source,target,dirs_exist_ok=True)
    else: shutil.copyfile(source,target)
elif args[:3] == ['device','process','launch']:
    if os.environ.get('FAKE_IOS_WAIT'):
        def stop(number, frame):
            print('Application exited with signal '+str(number),flush=True);raise SystemExit(0)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        (root/'launched').touch()
        while True: time.sleep(.05)
    env=lambda k:os.environ.get('DEVICECTL_CHILD_'+k)
    if env('MOE_HOST_CLEAN_ONLY'):
        shutil.rmtree(root/'Documents/inv',ignore_errors=True)
    else:
        invocation=env('MOE_INVOCATION_DIR')
        inv=root/'Documents/inv'
        if inv.exists():
            for p in inv.iterdir():
                if p.name!=invocation:
                    if p.is_dir():shutil.rmtree(p)
                    else:p.unlink()
        if env('MOE_STDERR_FILE'):remote('Documents/'+env('MOE_STDERR_FILE')).write_bytes(b'example stderr\n')
        if env('MOE_HOST_PROBE'):remote('Documents/'+env('MOE_HOST_PROBE')).write_text('{"before": {}, "after": {}}')
        if '--out' in args:
            out=remote('Documents/'+option('--out'));out.mkdir(parents=True);(out/'result.txt').write_text('result')
            (out/'nested').mkdir();(out/'nested/bytes.bin').write_bytes(bytes(range(256)))
    print('12:34:56  Acquired tunnel connection to device.')
    print('Launched application example')
    print('Waiting for the application to terminate...')
    print('example output', end='' if os.environ.get('FAKE_IOS_SIGNAL') else '\n')
    print('example stderr',file=sys.stderr)
    if os.environ.get('FAKE_IOS_SIGNAL'):
        print('App terminated due to signal '+os.environ['FAKE_IOS_SIGNAL']+'.')
    else:
        print('Application exited with exit code '+os.environ.get('FAKE_IOS_EXIT','0'))
    raise SystemExit(0)
elif args[:3] != ['device','install','app']: raise SystemExit(1)
if '--json-output' in args:Path(option('--json-output')).write_text(json.dumps(dict(result=result)))
