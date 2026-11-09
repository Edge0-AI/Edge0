"""File-input CLI transport for the iOS App, with exact output bytes."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from ios import IOS, Invocation, Unavailable
from android_cli import ChatInput

STATUS = re.compile(rb'^(?:\d\d:\d\d:\d\d\s+(?:Acquired tunnel connection|Enabling developer disk image services|Acquired usage assertion)|Launched application |Waiting for the application to terminate)')
EXIT = re.compile(rb'^(?:App(?:lication)?|Process|The app).*?(?:exit code\s*[:=]?\s*(-?\d+)|signal\s*[:=]?\s*(\d+))', re.I)


def console_result(data):
    output = []; code = None
    # CLI run need not end with LF: devicectl appends this trailer directly.
    trailer = re.search(rb'The app terminated with the exit code (-?\d+)\.\r?\n?\Z|(?:The app|App) (?:was )?terminated (?:with|by|due to)(?: the| a)? signal (\d+)\.\r?\n?\Z', data)
    if trailer:
        code = int(trailer[1]) if trailer[1] is not None else 128 + int(trailer[2])
        data = data[:trailer.start()]
    for line in data.splitlines(keepends=True):
        end = EXIT.search(line)
        if end:
            code = int(end[1]) if end[1] is not None else 128 + int(end[2])
        elif not STATUS.match(line):
            # The device PTY inserts CR before every LF, including existing CRLF.
            # Removing that inserted CR restores the original bytes.
            output.append(line.replace(b'\r\n', b'\n'))
    return b''.join(output), code


def execute(device, argv, data, probe_target=None):
    marker = device.env.get('MOE_IOS_TERMINATION_REPORT')
    if marker and Path(marker).exists():
        return 137, b'', b'iOS process was terminated previously; stop and report before any further invocation\n'
    invocation = Invocation(device)
    args = invocation.arguments(argv)
    if argv and argv[0] == 'chat':
        chat = ChatInput(invocation)
        data = b''.join(chat.line(line) for line in data.splitlines(keepends=True))
    with tempfile.TemporaryDirectory() as temp:
        local = Path(temp)
        input_file = local / 'stdin'; input_file.write_bytes(data)
        device.push(input_file, invocation.remote + '/stdin')
        environment = dict(device.env)
        environment.update(DEVICECTL_CHILD_MOE_INVOCATION_DIR=invocation.name,
                           DEVICECTL_CHILD_MOE_STDIN_FILE=invocation.remote + '/stdin',
                           DEVICECTL_CHILD_MOE_HOST_PROBE=invocation.remote + '/probe.json')
        process = subprocess.Popen(device.launch(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment)
        def forward(number, frame):
            if process.poll() is None: process.send_signal(number)
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            stdout, transport_errors = process.communicate()
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
        output, code = console_result(stdout)
        errors = transport_errors.replace(b'\r\n', b'\n')
        if code is None:
            # Do not pretend a transport success proves that the App completed.
            code = process.returncode if process.returncode > 0 else 1
            errors += b'iOS application exit status unavailable\n'
        terminated = code >= 128
        if terminated:
            # Persist the stop condition before any fallible output retrieval.
            if marker:
                Path(marker).write_text(f'iOS App terminated by signal {code - 128}; stop and report.\n')
            failure = Path(device.env.get('MOE_IOS_FAILURE_DIR',
                str(Path(__file__).resolve().parents[2] / 'build/ios-failures'))) / invocation.name
            failure.mkdir(parents=True, exist_ok=True)
            (failure / 'termination.txt').write_text(f'signal={code - 128}\ninvocation={invocation.name}\n')
            try:
                device.pull(invocation.remote + '/probe.json', failure / 'probe.json')
            except (OSError, RuntimeError, ValueError):
                errors += b'iOS failed invocation probe could not be retrieved; remote invocation preserved\n'
        try:
            invocation.pull_outputs()
            if probe_target:
                device.pull(invocation.remote + '/probe.json', probe_target)
        except (OSError, RuntimeError, ValueError):
            if not terminated:
                raise
            errors += b'iOS failed invocation outputs could not be retrieved; remote invocation preserved\n'
        return code, output, errors


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'serve':
        print('SKIP: serve is outside the iOS scope', file=sys.stderr); return 77
    try:
        device = IOS(); device.reachable()
        data = b'' if sys.stdin.isatty() else sys.stdin.buffer.read()
        code, output, errors = execute(device, argv, data, os.environ.get('MOE_IOS_PROBE_OUTPUT'))
        sys.stdout.buffer.write(output); sys.stderr.buffer.write(errors)
        return code
    except Unavailable as error:
        print(f'SKIP: {error}', file=sys.stderr); return 77
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr); return 1
