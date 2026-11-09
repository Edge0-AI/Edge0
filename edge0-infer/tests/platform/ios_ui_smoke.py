"""Require successful UI model loading and 30 seconds of subsequent liveness."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import time
from ios import IOS


def app_pid(device):
    result = device.call('device', 'info', 'processes', '--device', device.device)
    matches = [row['processIdentifier'] for row in result.get('runningProcesses', [])
               if Path(row.get('executable', '')).name == 'edge0-infer']
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one running App process')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leave-running', action='store_true',
                        help='Relaunch without a console after the check for manual acceptance')
    args = parser.parse_args()
    device = IOS()
    device.reachable()
    with tempfile.TemporaryDirectory() as directory:
        console = Path(directory) / 'console.log'
        with console.open('wb') as output:
            process = subprocess.Popen(device.launch([]), env=device.env,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while True:
                    content = console.read_text(errors='replace')
                    if 'chat ui: load failed:' in content:
                        raise RuntimeError('UI model loading failed')
                    if 'App terminated due to signal' in content:
                        raise RuntimeError('App terminated by signal; stop and report')
                    if process.poll() is not None:
                        raise RuntimeError('Console ended before successful UI startup')
                    if 'chat ui: model loaded' in content:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('No successful UI load message within 120 seconds')
                    time.sleep(0.25)
                print('Observed stderr: chat ui: model loaded', flush=True)
                pid = app_pid(device)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Console ended during the 30-second liveness check')
                    time.sleep(0.25)
                if app_pid(device) != pid:
                    raise RuntimeError('App process changed during the liveness check')
                print('Same UI process alive 30 seconds after model loaded', flush=True)
                device.call('device', 'process', 'terminate', '--device', device.device, '--pid', str(pid))
                process.wait(timeout=15)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
    if args.leave_running:
        device.call('device', 'process', 'launch', '--device', device.device,
                    '--terminate-existing', device.bundle)
        print('App relaunched without console for manual acceptance', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
