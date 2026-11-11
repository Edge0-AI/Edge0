"""Exercise a real console Ctrl-C in an isolated, hidden Windows console."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_support import assert_no_leftovers, environment, strata_args  # noqa: E402


def run_console(cli, model, engine, record, mode, eof=False):
    result = Path(str(record) + '-console.json')
    log = Path(str(record) + '-console.log')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    command = [sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--cli', str(Path(cli).resolve()),
               '--model', str(model), '--engine', str(engine), '--record', str(record), '--mode', mode,
               '--result', str(result)]
    if eof:
        command.append('--eof')
    with log.open('w', encoding='utf-8') as output:
        helper = subprocess.run(command, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup,
                                stdout=output, stderr=output, env=environment(record), timeout=150)
    assert result.exists(), (helper.returncode, log.read_text(encoding='utf-8'))
    report = json.loads(result.read_text(encoding='utf-8'))
    assert helper.returncode == 0 and report.get('ok'), report
    assert_no_leftovers(record)
    print(f'PASS console Ctrl-C: {mode}, eof={eof}, exit={report["exit_code"]}, no leftovers')


def main():
    parser = argparse.ArgumentParser()
    for option in ('cli', 'model', 'engine', 'record', 'mode', 'result'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--eof', action='store_true')
    args = parser.parse_args()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong)
    # A handler is not inherited by the child; the inheritable ignore-Ctrl-C flag would hide the test signal.
    handler = callback_type(lambda event: int(event == 0))
    kernel.SetConsoleCtrlHandler.argtypes = [callback_type, ctypes.c_int]
    kernel.SetConsoleCtrlHandler.restype = ctypes.c_int
    kernel.GenerateConsoleCtrlEvent.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel.GenerateConsoleCtrlEvent.restype = ctypes.c_int
    assert kernel.SetConsoleCtrlHandler(handler, True), ctypes.get_last_error()
    command = [args.cli, args.mode, '--model', args.model, *strata_args(args.engine)]
    if args.mode == 'serve':
        command += ['--port', '0']
    report = {'ok': False}
    process = None
    lines = []
    ready = threading.Event()
    try:
        # Deliberately share this console and process group: Ctrl-C is broadcast to this console only.
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding='utf-8', env=environment(args.record))
        def read(stream, readiness=False):
            for line in stream:
                lines.append(line)
                if readiness and (line.startswith('Listening:') or line.startswith('Ready:')):
                    ready.set()
        readers = [threading.Thread(target=read, args=(process.stdout,), daemon=True),
                   threading.Thread(target=read, args=(process.stderr, True), daemon=True)]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + 60
        while not ready.wait(0.05):
            assert process.poll() is None and time.monotonic() < deadline, ''.join(lines)
        assert kernel.GenerateConsoleCtrlEvent(0, 0), ctypes.get_last_error()
        if args.eof:
            process.stdin.close()
        report['exit_code'] = process.wait(60)
        for reader in readers:
            reader.join(5)
        assert report['exit_code'] == (0 if args.mode == 'serve' else 130), report
        assert_no_leftovers(args.record)
        report['ok'] = True
    except BaseException as error:
        report['error'] = repr(error)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(10)
        report['output'] = ''.join(lines)
        Path(args.result).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
