"""Launch the opt-in device exercise and print only identity-free App evidence."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='35b')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--status-retake-check', action='store_true', help='One short request, only light/dark Status captures, then unload')
    parser.add_argument('--bad-assets-name', help='Inspect one explicitly prepared invalid fixture without loading a model')
    parser.add_argument('--model-switch-check', action='store_true', help='Switch from existing model to tiny-a and back with fresh chats and status checks')
    parser.add_argument('--background-active-check', action='store_true', help='Background an active generation, return and verify safe continuation')
    parser.add_argument('--background-output', type=Path, default=Path('examples/ios/build/background-evidence'))
    parser.add_argument('--focused-native-checks', action='store_true', help='Short Markdown and Thinking turns with unload/reload')
    parser.add_argument('--history-relaunch-check', action='store_true', help='Load persisted history after process relaunch and continue one turn')
    parser.add_argument('--status-pages-checks', action='store_true', help='History, message actions, native status and actual light/dark page captures')
    parser.add_argument('--presentation-checks', action='store_true', help='Focused chooser, reasoning disclosure and locked switch screenshots; no fixtures')
    parser.add_argument('--thinking-label-checks', action='store_true', help='Only locked On/Off label screenshots; no fixtures')
    parser.add_argument('--decode-comparison', action='store_true', help='One rainbow paragraph, max96; no fixtures, screenshots or other checks')
    parser.add_argument('--revision-checks', action='store_true', help='Focused model-label, thinking-lock, stop-feedback and short-input metric checks')
    parser.add_argument('--visual-checks', action='store_true', help='Also request Markdown and capture the presented window')
    parser.add_argument('--max-tokens', type=int, help='Optional exercise-only generation limit for random tiny fixtures')
    parser.add_argument('--failure-log', type=Path, default=Path('examples/ios/build/device-failure.log'))
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.decode_comparison:
        if args.max_tokens not in (None, 96):
            parser.error('--decode-comparison requires max-tokens 96')
        if args.revision_checks or args.visual_checks:
            parser.error('--decode-comparison cannot be combined with other checks')
        args.max_tokens = 96
    team = os.environ.get('MOE_IOS_TEAM')
    device = os.environ.get('MOE_IOS_DEVICE')
    bundle = os.environ.get('MOE_IOS_EXAMPLE_BUNDLE_ID') or (f'dev.edge0.{team}.edge0-chat' if team else None)
    if not team or not device:
        parser.error('MOE_IOS_TEAM and MOE_IOS_DEVICE are required')
    command = ['xcrun', 'devicectl', 'device', 'process', 'launch', '--device', device,
               '--console', '--terminate-existing', bundle, '--smoke', args.model,
               '--rounds', str(args.rounds)]
    if args.status_retake_check:
        command += ['--status-retake-check']
    if args.bad_assets_name:
        command += ['--bad-assets-name', args.bad_assets_name]
    if args.model_switch_check:
        command += ['--model-switch-check']
    if args.background_active_check:
        command += ['--background-active-check']
    if args.focused_native_checks:
        command += ['--focused-native-checks']
    if args.history_relaunch_check:
        command += ['--history-relaunch-check']
    if args.status_pages_checks:
        command += ['--status-pages-checks']
    if args.presentation_checks:
        command += ['--presentation-checks']
    if args.thinking_label_checks:
        command += ['--thinking-label-checks']
    if args.decode_comparison:
        command += ['--decode-comparison']
    if args.revision_checks:
        command += ['--revision-checks']
    if args.visual_checks:
        command += ['--visual-checks']
    if args.max_tokens:
        command += ['--smoke-max-tokens', str(args.max_tokens)]
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / 'console.log'
        with log.open('wb') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        printed = set()
        started = time.monotonic()
        try:
            while time.monotonic() - started < args.timeout:
                content = log.read_text(errors='replace')
                for line in content.splitlines():
                    if ('EXAMPLE_' in line or 'EXAMPLE_SMOKE' in line or 'example chat:' in line or 'EXAMPLE_CAPTURE' in line or 'EXAMPLE_LIFECYCLE' in line or 'EXAMPLE_ASSETS' in line or 'EXAMPLE_SWITCH' in line) and line not in printed:
                        # Engine load logs contain no filesystem paths or signing identifiers.
                        print(line, flush=True)
                        printed.add(line)
                if args.status_retake_check and 'EXAMPLE_STATUS_RETAKE complete' in content:
                    print('Status light/dark retake completed; model unloaded.', flush=True)
                    return 0
                if args.bad_assets_name and 'EXAMPLE_BAD_ASSETS complete' in content:
                    print('Invalid fixture shown only in Unavailable with a reason; inspect actual Models capture.', flush=True)
                    return 0
                if args.model_switch_check and 'EXAMPLE_MODEL_SWITCH complete' in content:
                    print('Two-way model switch, fresh conversations and status checks completed.', flush=True)
                    return 0
                if args.background_active_check and 'EXAMPLE_ACTIVE ready' in content:
                    subprocess.run([sys.executable, '-I', 'examples/ios/background_check.py', '--during-generation', '--output', str(args.background_output)], check=True)
                    print('Active generation cancelled on background and resumed safely.', flush=True)
                    return 0
                if args.focused_native_checks and 'EXAMPLE_FOCUSED complete' in content:
                    print('Focused native exercise complete; inspect actual captures separately.', flush=True)
                    return 0
                if args.history_relaunch_check and 'EXAMPLE_STATUS relaunch complete' in content:
                    print('Persisted history and totals restored after relaunch; continued turn completed.', flush=True)
                    return 0
                if args.status_pages_checks and 'EXAMPLE_STATUS complete' in content:
                    print('Status page exercise completed; screenshots await proposer review.', flush=True)
                    return 0
                if args.thinking_label_checks and 'EXAMPLE_THINKING_LABELS complete' in content:
                    print('Focused On/Off label checks completed; inspect captured windows.', flush=True)
                    return 0
                if args.presentation_checks and 'EXAMPLE_PRESENTATION complete' in content:
                    print('Focused presentation checks completed; inspect captured windows.', flush=True)
                    return 0
                if args.decode_comparison and 'EXAMPLE_DECODE complete' in content:
                    print('One decode comparison completed.', flush=True)
                    return 0
                if args.revision_checks and 'EXAMPLE_REVISION complete' in content:
                    print('Focused revision checks passed; inspect actual window captures separately.', flush=True)
                    return 0
                if any('EXAMPLE_SMOKE resumed=' in line and 'state valid: true complete' in line for line in content.splitlines()):
                    if f'EXAMPLE_SMOKE rounds={args.rounds} completed' not in content or 'EXAMPLE_SMOKE cancel=cancelled · state valid: true' not in content:
                        raise RuntimeError('Missing required rounds or cancelled result')
                    print('Device exercise passed; inspect the UI checklist separately.', flush=True)
                    return 0
                if 'EXAMPLE_SMOKE failed' in content or 'EXAMPLE_REVISION failed' in content or 'EXAMPLE_STATUS failed' in content or 'EXAMPLE_FOCUSED failed' in content or 'EXAMPLE_ACTIVE failed' in content or 'EXAMPLE_BAD_ASSETS failed' in content or 'EXAMPLE_MODEL_SWITCH failed' in content or 'EXAMPLE_STATUS_RETAKE failed' in content:
                    raise RuntimeError('App reported a failed exercise turn')
                if process.poll() is not None:
                    raise RuntimeError('Device console ended before the exercise completed')
                if 'App terminated due to signal' in content:
                    raise RuntimeError('App was terminated; preserve the failed run and report')
                time.sleep(0.25)
            raise RuntimeError('Device exercise timed out')
        except BaseException:
            args.failure_log.parent.mkdir(parents=True, exist_ok=True)
            args.failure_log.write_bytes(log.read_bytes())
            print('Raw failed console preserved in the ignored build directory.', flush=True)
            raise
        finally:
            # Leave the App in the foreground for visual review. Detach the host console.
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
