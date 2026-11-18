"""Run the same CLI scenarios on a platform (Android device or local Windows build); no cross-compiled unit tests are executed."""
import argparse
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'tests/platform/android-cli'
INTEGRATION = ROOT / 'tests/integration'

# Per-platform command runner and host-side probe binaries. Android forwards every scenario to a
# device through the android-cli wrapper and runs host probes from the cpu build; Windows runs the
# scenarios locally against the windows build, so both the CLI and the probes come from it.
PLATFORMS = {
    'ios': dict(cli=ROOT / 'tests/platform/ios-cli', bindir=ROOT / 'build/cpu/bin', exe='', backend='metal', file_stdin=True, skip_observe=('long-mixed',)),
    'android': dict(cli=CLI, bindir=ROOT / 'build/cpu/bin', exe=''),
    'windows': dict(cli=ROOT / 'build/windows/bin/edge0-infer.exe', bindir=ROOT / 'build/windows/bin', exe='.exe'),
}


# Force UTF-8 mode in the harness processes (-X utf8, which survives -I) so that subprocess text
# capture and read_text/write_text default to UTF-8 regardless of the OS locale. Without it, model
# generated non-ASCII text is decoded with the console code page (for example GBK on a Chinese
# Windows) and dropped. On macOS/Android the locale is already UTF-8, so this is a no-op.
UTF8 = ['-I', '-X', 'utf8']


def scenarios(cli, bindir, exe='', backend='cpu', file_stdin=False, skip_observe=()):
    def script(name, *args):
        return [sys.executable, *UTF8, str(INTEGRATION / name), '--cli', str(cli), *args]
    def probe(name):
        return str(bindir / (name + exe))
    return {
        'tiny-generate': script('generate_answers_test.py', '--record-only', '--cross-kernel', '--backend', backend),
        'tiny-forward': [script('forward_answers_test.py', '--record-only', '--cross-kernel', '--backend', backend),
                         script('forward_answers_test.py', '--record-only', '--cross-kernel', '--backend', backend, '--split-last')],
        '35b-numerical': script('single_turn_35b_test.py', '--backend', backend, '--mode', 'numerical', '--cross-kernel', '--record-only',
                                *[arg for name in skip_observe for arg in ('--skip-observe', name)]),
        '35b-describe': script('single_turn_35b_test.py', '--backend', backend, '--mode', 'describe'),
        '35b-memory': script('single_turn_35b_test.py', '--backend', backend, '--mode', 'memory'),
        'tiny-multi-turn': script('multi_turn_test.py', '--backend', backend),
        '35b-multi-turn': script('multi_turn_test.py', '--backend', backend, '--profile', '35b', '--derive-counts'),
        'tiny-cancel': script('cancel_test.py', '--backend', backend),
        '35b-cancel': script('cancel_test.py', '--backend', backend, '--profile', '35b', '--prompt-file',
                             str(ROOT / 'tests/fixtures/bench/library-report-prompt-head.txt')),
        'thinking-tools': script('thinking_tools_test.py', '--backend', backend, '--probe', probe('moe_thinking_rerender_probe')),
        'chat': script('chat_cli_test.py', '--backend', backend, *(['--file-stdin'] if file_stdin else [])),
        'budget': script('budget_cli_test.py', '--backend', backend),
        '35b-serve': [sys.executable, *UTF8, str(INTEGRATION / 'serve_sdk_runner.py'), str(INTEGRATION / 'serve_35b_test.py'),
                      '--cli', str(cli), '--backend', backend, '--decode', probe('moe_portal_decode')],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', required=True, choices=sorted(PLATFORMS))
    parser.add_argument('--only', help='Comma-separated scenario names')
    parser.add_argument('--list', action='store_true', help='Print commands without contacting a device')
    args = parser.parse_args()
    cases = scenarios(**PLATFORMS[args.platform])
    cases['info-system'] = [sys.executable, *UTF8, str(ROOT / 'tests/platform/info_system_test.py'),
                            '--cli', str(PLATFORMS[args.platform]['cli']), '--platform', args.platform]
    names = args.only.split(',') if args.only else list(cases)
    if any(name not in cases for name in names):
        parser.error('Unknown scenario; use --list')
    failed = skipped = 0
    termination = None
    if args.platform == 'ios' and not args.list:
        temporary = tempfile.TemporaryDirectory()
        termination = Path(temporary.name) / 'terminated'
        os.environ['MOE_IOS_TERMINATION_REPORT'] = str(termination)
    for name in names:
        if args.platform == 'ios' and name == '35b-serve':
            print('35b-serve: outside iOS scope', flush=True)
            continue
        commands = cases[name] if isinstance(cases[name][0], list) else [cases[name]]
        start = time.monotonic()
        code = 0
        for command in commands:
            print(name + ': ' + ' '.join(command), flush=True)
            if args.list:
                continue
            code = subprocess.run(command, cwd=ROOT).returncode
            if termination is not None and termination.exists():
                print(termination.read_text(), flush=True)
                return 1
            if code:
                break
        if args.list:
            continue
        status = 'passed' if code == 0 else 'not executed' if code == 77 else 'failed'
        failed += status == 'failed'; skipped += status == 'not executed'
        print(f'{name}: {status} ({time.monotonic()-start:.2f}s, exit={code})', flush=True)
    if args.platform == 'ios' and not args.list:
        sys.path.insert(0, str(ROOT / 'tests/platform'))
        from ios import IOS, Unavailable
        try:
            device = IOS()
            device.reachable()
            device.clean()
        except Unavailable as error:
            print(f'cleanup not executed: {error}', flush=True)
            skipped += 1
        except RuntimeError as error:
            print(f'cleanup failed: {error}', flush=True)
            failed += 1
    return 1 if failed else 77 if skipped else 0


if __name__ == '__main__':
    raise SystemExit(main())
