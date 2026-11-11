"""Functional delegate smoke check (standard library only); write a JSON result and stderr logs."""
import argparse
import ctypes
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


def process_tree():
    if os.name != 'nt':
        output = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
        return {int(row.split()[0]): int(row.split()[1]) for row in output.splitlines()}
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD), ('pid', wintypes.DWORD),
                    ('heap', ctypes.c_size_t), ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
                    ('parent', wintypes.DWORD), ('priority', wintypes.LONG), ('flags', wintypes.DWORD),
                    ('exe', wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ('Process32FirstW', 'Process32NextW'):
        method = getattr(kernel, name)
        method.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        method.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), 'CreateToolhelp32Snapshot')
    result = {}
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            result[entry.pid] = entry.parent
            found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return result


class Session:
    def __init__(self, command, stderr_path):
        self.stderr_path = stderr_path
        self.lines = []
        self.output = []
        self.condition = threading.Condition()
        self.owned = set()
        self.observer_error = None
        self.done = threading.Event()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        encoding='utf-8', errors='replace',
                                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        def read(stream, target):
            for line in stream:
                with self.condition:
                    target.append(line)
                    self.condition.notify_all()
        def watch():
            try:
                while not self.done.is_set():
                    tree = process_tree()
                    roots = {self.process.pid, *self.owned}
                    while True:
                        children = {pid for pid, parent in tree.items() if parent in roots}
                        if children <= roots:
                            break
                        roots |= children
                    self.owned |= roots - {self.process.pid}
                    self.done.wait(0.1)
            except Exception as error:
                self.observer_error = repr(error)
        self.readers = [threading.Thread(target=read, args=(self.process.stdout, self.output), daemon=True),
                        threading.Thread(target=read, args=(self.process.stderr, self.lines), daemon=True)]
        for reader in self.readers:
            reader.start()
        self.watcher = threading.Thread(target=watch, daemon=True)
        self.watcher.start()

    def ready(self, mode):
        pattern = r'^Listening: http://127\.0\.0\.1:(\d+) model=(.+)$' if mode == 'serve' else r'^Ready:'
        deadline = time.monotonic() + 1800
        with self.condition:
            while True:
                for line in self.lines:
                    match = re.search(pattern, line.strip())
                    if match:
                        return match
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Service did not become ready: ' + ''.join(self.lines[-10:]))
                self.condition.wait(0.2)

    def close(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
        try:
            code = self.process.wait(90)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(30)
        self.done.set()
        self.watcher.join(5)
        for reader in self.readers:
            reader.join(5)
        self.stderr_path.write_text(''.join(self.lines), encoding='utf-8')
        deadline = time.monotonic() + 10
        while True:
            left = sorted(self.owned & process_tree().keys())
            if not left or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        return {'exit_code': code, 'observed_descendants': sorted(self.owned), 'remaining_pids': left,
                'observer_error': self.observer_error,
                'clean': code == 0 and not left and bool(self.owned) and self.observer_error is None}


def request(port, alias, prompt, stream):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=900)
    body = {'model': alias, 'messages': [{'role': 'user', 'content': prompt}], 'stream': stream,
            'temperature': 0, 'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}}
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        content = response.read().decode('utf-8')
        if stream:
            events = [json.loads(line[6:]) for line in content.splitlines() if line.startswith('data: {')]
            text = ''.join(choice.get('delta', {}).get('content', '') for event in events for choice in event.get('choices', []))
            finish = next((choice['finish_reason'] for event in reversed(events) for choice in event.get('choices', [])
                           if choice.get('finish_reason')), None)
            protocol = response.getheader('Content-Type', '').startswith('text/event-stream') and 'data: [DONE]' in content
        else:
            choice = json.loads(content)['choices'][0]
            text, finish = choice['message']['content'], choice.get('finish_reason')
            protocol = response.getheader('Content-Type', '').startswith('application/json')
        return {'status': response.status, 'answer': text, 'finish': finish, 'protocol_ok': protocol,
                'ok': response.status == 200 and finish == 'stop' and protocol}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--strata-dir', required=True)
    parser.add_argument('--strata-python')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-option', action='append', default=[])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared = ['--model', args.model, '--strata-dir', args.strata_dir, '--device', args.device]
    if args.strata_python:
        shared += ['--strata-python', args.strata_python]
    for option in args.model_option:
        shared += ['--model-option', option]
    report = {'ok': False, 'checks': {'paris': False, 'count_1_to_5': False, 'green_second_turn': False}}
    try:
        session = Session([args.cli, 'serve', *shared, '--port', '0'], output.with_suffix('.serve.stderr.log'))
        try:
            ready = session.ready('serve')
            port, alias = int(ready[1]), ready[2]
            paris = request(port, alias, 'What is the capital of France? Answer in one word.', False)
            count = request(port, alias, 'List the numbers from 1 to 5, separated by commas.', True)
            report['requests'] = [paris, count]
            report['checks']['paris'] = paris['ok'] and bool(re.search(r'\bparis\b', paris['answer'], re.I))
            report['checks']['count_1_to_5'] = count['ok'] and re.findall(r'\b\d+\b', count['answer']) == ['1', '2', '3', '4', '5']
        finally:
            report['serve_cleanup'] = session.close()
        session = Session([args.cli, 'chat', *shared], output.with_suffix('.chat.stderr.log'))
        try:
            session.ready('chat')
            session.process.stdin.write('/think off\nMy favourite colour is green. Remember it and reply in one short sentence.\n'
                                        'What is my favourite colour? Answer in one short sentence.\n/quit\n')
            session.process.stdin.flush()
            session.process.stdin.close()
            session.process.wait(1800)
        finally:
            report['chat_cleanup'] = session.close()
            report['chat_stdout'] = ''.join(session.output)
        answers = [line for line in report['chat_stdout'].splitlines() if line.strip()]
        completed = sum('[edge0-strata]' in line and 'end_of_generation' in line for line in session.lines)
        report['checks']['green_second_turn'] = completed == 2 and len(answers) >= 2 and bool(re.search(r'\bgreen\b', answers[-1], re.I))
        report['ok'] = all(report['checks'].values()) and report['serve_cleanup']['clean'] and report['chat_cleanup']['clean']
    except Exception as error:
        report['error'] = repr(error)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
