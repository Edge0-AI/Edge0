"""Exercise a real background/foreground transition without changing other Apps' data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--during-generation', action='store_true', help='Start while an explicit background-active exercise is generating')
    parser.add_argument('--output', type=Path, default=Path('examples/ios/build/device-evidence'))
    args = parser.parse_args()
    device = os.environ.get('MOE_IOS_DEVICE')
    team = os.environ.get('MOE_IOS_TEAM')
    bundle = os.environ.get('MOE_IOS_EXAMPLE_BUNDLE_ID') or (f'dev.edge0.{team}.edge0-chat' if team else None)
    if not device or not bundle:
        parser.error('MOE_IOS_DEVICE and MOE_IOS_TEAM or MOE_IOS_EXAMPLE_BUNDLE_ID are required')
    args.output.mkdir(parents=True, exist_ok=True)

    def call(*command):
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / 'result.json'
            result = subprocess.run(['xcrun', 'devicectl', *command, '--json-output', str(result_file)], capture_output=True)
            if result.returncode:
                raise RuntimeError(f'Device request failed: {command[0]} {command[1]}')
            return json.loads(result_file.read_text())['result']

    def pull(name, target):
        call('device', 'copy', 'from', '--device', device, '--domain-type', 'appDataContainer',
             '--domain-identifier', bundle, '--source', 'Documents/validation/' + name, '--destination', str(target))

    def process_id():
        rows = call('device', 'info', 'processes', '--device', device).get('runningProcesses', [])
        matches = [row['processIdentifier'] for row in rows if Path(row.get('executable', '')).name == 'edge0-chat']
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one running example App process')
        return matches[0]

    pull('state.json', args.output / 'before-background.json')
    before = json.loads((args.output / 'before-background.json').read_text())
    if before['generating'] != args.during_generation or not before['messages']:
        raise RuntimeError('Complete a device exercise before this check')
    pid_before = process_id()
    call('device', 'process', 'launch', '--device', device, 'com.apple.Preferences')
    time.sleep(2)
    pull('state.json', args.output / 'while-background.json')
    background = json.loads((args.output / 'while-background.json').read_text())
    if background.get('lifecycle') != 'background' or before['session'] != background['session']:
        raise RuntimeError('No real background lifecycle event was observed')
    call('device', 'process', 'launch', '--device', device, bundle)
    deadline = time.monotonic() + 240
    while True:
        time.sleep(1)
        pull('state.json', args.output / 'after-foreground.json')
        current = json.loads((args.output / 'after-foreground.json').read_text())
        if current.get('lifecycle') == 'foreground' and not current['generating']:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Foreground exercise did not complete')
    pull('foreground.png', args.output / 'foreground.png')
    after = json.loads((args.output / 'after-foreground.json').read_text())
    retained = before['messages'][:-1] if args.during_generation else before['messages']
    if pid_before != process_id() or before['session'] != after['session'] or retained != after['messages'][:len(retained)]:
        raise RuntimeError('App process, model owner, or complete history changed')
    if not after.get('foreground_followup_complete'):
        raise RuntimeError('Run the device exercise with --visual-checks to verify a continued turn')
    if args.during_generation and not after.get('background_cancellation_safe'):
        raise RuntimeError('Active generation did not report a safe cancellation')
    print(f'Background/foreground passed: same process and session, {len(retained)} complete original messages retained exactly; follow-up completed.')
    print('Foreground screenshot copied. Inspect it separately.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
