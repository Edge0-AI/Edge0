"""Streaming CLI transport; completed-request evidence precedes its summary."""
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
from android import Android, Invocation, Unavailable


class ChatInput:
    """Match chat command classification while preserving non-path bytes."""

    def __init__(self, invocation):
        self.invocation = invocation
        self.collecting = False

    def line(self, data):
        whitespace = b' \t\r\n\v\f'
        trimmed = data.strip(whitespace)
        if trimmed == b'"""':
            self.collecting = not self.collecting
            return data
        if self.collecting or not trimmed.startswith(b'/file') or len(trimmed) <= 5:
            return data
        if trimmed[5] not in whitespace:
            return data
        start = 5
        while start < len(trimmed) and trimmed[start] in b' \t\r\n':
            start += 1
        path = os.fsdecode(trimmed[start:])
        if not path or not Path(path).is_file():
            return data
        offset = len(data) - len(data.lstrip(whitespace))
        mapped = os.fsencode(self.invocation.path(path))
        return data[:offset + start] + mapped + data[offset + len(trimmed):]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        device = Android()
        device.reachable()
        with Invocation(device) as invocation:
            remote_args = invocation.arguments(argv)
            pidfile = invocation.remote + '/pid'
            # exec retains the shell PID, so only this invocation receives signals.
            script = 'echo $$ > ' + shlex.quote(pidfile) + '; exec ' + shlex.join([device.root + '/edge0-infer', *remote_args])
            process = subprocess.Popen(device.argv('shell', '-T', script), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            forwarded = []
            errors = []
            pending = []
            signal_lock = threading.Lock()
            done = threading.Event()

            def relay_signal(number, frame):
                with signal_lock:
                    pending.append(number)

            previous = {sig: signal.signal(sig, relay_signal) for sig in (signal.SIGINT, signal.SIGTERM)}

            def signals():
                while not done.wait(.02):
                    with signal_lock:
                        if not pending:
                            continue
                        number = pending[0]
                    result = device.shell('cat', pidfile, check=False)
                    pid = result.stdout.strip()
                    if not result.returncode and pid.isdecimal():
                        device.shell('kill', '-' + signal.Signals(number).name[3:], pid, check=False)
                        with signal_lock:
                            pending.pop(0)

            def input_stream():
                try:
                    chat = ChatInput(invocation) if argv and argv[0] == 'chat' else None
                    pending_input = b''
                    # Avoid holding a buffered stdin lock during interpreter
                    # shutdown when a caller waits for exit before closing input.
                    while not done.is_set():
                        ready, _, _ = select.select([sys.stdin.fileno()], [], [], .05)
                        if not ready:
                            continue
                        block = os.read(sys.stdin.fileno(), 65536)
                        if not block:
                            if chat and pending_input:
                                process.stdin.write(chat.line(pending_input))
                            break
                        if chat:
                            pending_input += block
                            while b'\n' in pending_input:
                                line, pending_input = pending_input.split(b'\n', 1)
                                process.stdin.write(chat.line(line + b'\n'))
                        else:
                            process.stdin.write(block)
                        process.stdin.flush()
                except BrokenPipeError:
                    pass
                except Exception as error:
                    errors.append(error)
                    relay_signal(signal.SIGTERM, None)
                finally:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass

            def output_stream(source, target):
                try:
                    if not argv or argv[0] != 'serve':
                        while True:
                            block = os.read(source.fileno(), 65536)
                            if not block:
                                return
                            target.write(block); target.flush()
                    else:
                        for line in iter(source.readline, b''):
                            text = line.decode('utf-8', errors='replace')
                            match = re.search(r'Listening: http://[^:]+:(\d+)', text)
                            if match:
                                port = device.forward(match[1])
                                forwarded.append(port)
                                line = (text[:match.start(1)] + port + text[match.end(1):]).encode()
                            if text.startswith('{'):
                                try:
                                    summary = json.loads(text)
                                except ValueError:
                                    summary = {}
                                if 'stop_reason' in summary:
                                    invocation.pull_outputs()
                            target.write(line); target.flush()
                except Exception as error:
                    errors.append(error)
                    relay_signal(signal.SIGTERM, None)

            workers = [threading.Thread(target=output_stream, args=(process.stdout, sys.stdout.buffer)),
                       threading.Thread(target=output_stream, args=(process.stderr, sys.stderr.buffer))]
            input_worker = threading.Thread(target=input_stream)
            input_worker.start()
            signal_worker = threading.Thread(target=signals, daemon=True)
            signal_worker.start()
            try:
                for worker in workers:
                    worker.start()
                code = process.wait()
                for worker in workers:
                    worker.join()
                invocation.pull_outputs()
            finally:
                done.set(); signal_worker.join(); input_worker.join()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                for port in forwarded:
                    device.unforward(port)
            if errors:
                raise RuntimeError(str(errors[0]))
            return code if code >= 0 else 128 - code
    except Unavailable as error:
        print(f'SKIP: {error}', file=sys.stderr)
        return 77
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
