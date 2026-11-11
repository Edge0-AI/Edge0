"""Exercise one 145B model/format through serve, or the default two-turn chat.

Read only the child engine's argv and relevant environment to prove selected options.
Wait for the selected GPU and available RAM before launch; stop only the process started here.
"""
import argparse, http.client, json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path

import engine_info


def children(pid):
    out = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[1]) == pid:
                    out.append(int(entry.name))
            except OSError:
                pass
    return out


def descendants(pid):
    found, todo = [], [pid]
    while todo:
        for child in children(todo.pop()):
            found.append(child)
            todo.append(child)
    return found


def comm(pid):
    try:
        return Path(f'/proc/{pid}/comm').read_text().strip()
    except OSError:
        return '?'


def status_kb(pid, key):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(key + ':'):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def gib(kb):
    return round(kb / 1048576, 1)


def engine_evidence(root, model, port):
    tree = descendants(root)
    print('process tree:', [(p, comm(p)) for p in tree], flush=True)
    def exe(p):
        try:
            return os.path.basename(os.readlink(f'/proc/{p}/exe'))
        except OSError:
            return ''
    engines = [p for p in tree if exe(p) == 'strata']
    if len(engines) != 1:
        print('ENGINE NOT FOUND', engines, flush=True)
        return None
    pid = engines[0]
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    argv = [a.decode() for a in argv if a]
    ple_io = argv[argv.index('--ple-io') + 1] if '--ple-io' in argv else None
    env = [e.decode() for e in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if e.startswith((b'STRATA_LATENT_', b'STRATA_LAT_FUSED_Q3K='))]
    # The model directory's entries may be symbolic links; the kernel shows the resolved target in maps and fd,
    # so compare against each entry's real path and report it under the entry's name.
    registered = {}
    for folder, _, names in os.walk(model, followlinks=True):
        for name in names:
            entry = os.path.join(folder, name)
            if os.path.isfile(entry):
                registered[os.path.realpath(entry)] = os.path.relpath(entry, model)
    maps = sorted({registered[line.split(None, 5)[5]] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                   if len(line.split(None, 5)) == 6 and line.split(None, 5)[5] in registered})
    fds = set()
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target in registered:
            fds.add(registered[target])
    rss = status_kb(pid, 'VmRSS')
    tree_rss = sum(status_kb(p, 'VmRSS') for p in [root] + tree)
    _, _, body = get(port, '/v1/status')
    try:
        ram = json.loads(body).get('machine', {}).get('ram')
    except ValueError:
        ram = None
    evidence = {'engine_pid': pid, 'ple_io_argv': ple_io, 'latent_env': env, 'argv': argv,
                'mapped_model_files': maps, 'open_model_files': sorted(fds),
                'engine_vmrss_gib': gib(rss), 'engine_vmhwm_gib': gib(status_kb(pid, 'VmHWM')),
                'engine_vmlck_gib': gib(status_kb(pid, 'VmLck')), 'tree_vmrss_gib': gib(tree_rss), 'v1_status_machine_ram': ram}
    for key, value in evidence.items():
        print(f'engine {key}: {value}', flush=True)
    return evidence


def get(port, path):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    c.request('GET', path)
    r = c.getresponse()
    return r.status, r.getheader('Content-Type'), r.read().decode()


def chat_request(port, alias, content, stream):
    body = {'model': alias, 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': content}]}
    if stream:
        body['stream'] = True
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse()
    kind = r.getheader('Content-Type')
    if not stream:
        data = json.loads(r.read())
        return r.status, kind, data['choices'][0]['message'].get('content') or '', data['choices'][0].get('finish_reason')
    text, finish, done = '', None, False
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            done = True
            continue
        for choice in json.loads(payload).get('choices', []):
            text += choice.get('delta', {}).get('content') or ''
            finish = choice.get('finish_reason') or finish
    return r.status, kind, text, (finish if done else 'no [DONE]')


class Reader:
    def __init__(self, stream):
        self.lines, self.changed = [], threading.Condition()
        threading.Thread(target=self.run, args=(stream,), daemon=True).start()

    def run(self, stream):
        for line in stream:
            with self.changed:
                self.lines.append(line.rstrip('\n'))
                self.changed.notify_all()
        with self.changed:
            self.lines.append(None)
            self.changed.notify_all()

    def wait_for(self, pattern, timeout):
        deadline = time.time() + timeout
        with self.changed:
            while True:
                for line in self.lines:
                    if line is None:
                        return None
                    m = re.search(pattern, line)
                    if m:
                        return m
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.changed.wait(left)


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM if process.args[1] == 'serve' else signal.SIGINT)
    try:
        return process.wait(180)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--option', action='append', default=[])
    p.add_argument('--chat', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.gpu < 0:
        p.error('--gpu must be nonnegative')
    model = Path(a.model)
    options = [x for o in a.option for x in ('--model-option', o)]
    command = [a.cli, 'chat' if a.chat else 'serve', '--model', a.model, '--device', f'cuda:{a.gpu}', *options]
    if not a.chat:
        command += ['--port', '0']
    subprocess.run(['bash', str(Path(__file__).with_name('wait_resources.sh')), 'direct', str(a.gpu)], check=True)
    print('command:', ' '.join(command), flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE if a.chat else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print('pid', process.pid, flush=True)
    err = Reader(process.stderr)
    out = Reader(process.stdout)
    report = {'command': command, 'options': a.option, 'mode': 'chat' if a.chat else 'serve'}
    ok = True
    owned = []
    try:
        if a.chat:
            if not err.wait_for(r'^Ready:', 900):
                print('NOT READY', flush=True)
                return 1
            log = err.wait_for(r'\(log: (.+)\)', 1)[1]
            port = int(re.search(r'ready: http://127\.0\.0\.1:(\d+)/v1', Path(log).read_text())[1])
            alias = None
        else:
            m = err.wait_for(r'^Listening: http://127\.0\.0\.1:(\d+) model=(\S+)$', 900)
            if not m:
                print('NOT READY', flush=True)
                return 1
            print('ready line:', m[0], flush=True)
            port, alias = int(m[1]), m[2]
        good, info = engine_info.check(f'http://127.0.0.1:{port}', 0)
        print('\n'.join(info), flush=True)
        report['engine_info'] = info
        if not good:
            print('RESULT FAILED: engine information unavailable', flush=True)
            return 3
        report['engine'] = engine_evidence(process.pid, model, port)
        owned = descendants(process.pid)
        report['owned_pids'] = [process.pid, *owned]
        assert report['engine'] is not None
        selected = dict(o.split('=', 1) for o in a.option)
        model_name = json.loads((model / 'model.json').read_text())['model_name']
        assert model_name in ('edge0-145b-q3k', 'edge0-145b-fp4', 'edge0-120b-q3k', 'edge0-120b-fp4')
        experts = model_name.rsplit('-', 1)[1]
        du = selected.get('du', 'q4' if model_name.startswith('edge0-120b-') else 'q8')
        argv = report['engine']['argv']
        removed = {'--pool-workers', '--pcie-mode', '--pcie-frac', '--vram-reserve-mib', '--kv-resident'}
        assert not removed.intersection(argv), argv
        assert argv[argv.index('--max-context') + 1] == '131072', argv
        report['portable_template_verified'] = True
        env = dict(item.split('=', 1) for item in report['engine']['latent_env'])
        assert argv[argv.index('--pack') + 1] == str(model), argv
        assert env['STRATA_LATENT_A'] == str(model / 'experts.bin'), env
        for flag, filename in (('--native', 'model-skeleton.gguf'), ('--native-dense-gguf', 'model-skeleton.gguf'),
                               ('--expert-profile', 'profile.bin')):
            assert argv[argv.index(flag) + 1] == str(model / filename), argv
        assert env['STRATA_LATENT_LABELS'] == str(model / 'labels.bin'), env
        assert env['STRATA_LATENT_DENSE'] == str(model / f'latent_dense_{du}.bin'), env
        assert env['STRATA_LATENT_DENSE_FMT'] == du[1:], env
        assert env['STRATA_LAT_FUSED_Q3K'] == ('1' if experts == 'q3k' else '0'), env
        report['selection_verified'] = {'model_name': model_name, 'experts': experts, 'du': du}
        if a.chat:
            turns = ['/think off', 'My favourite colour is green. Please remember it and reply in one short sentence.',
                     'What is my favourite colour? Answer in one short sentence.', '/quit']
            process.stdin.write('\n'.join(turns) + '\n')
            process.stdin.close()
            process.wait(900)
            out.wait_for(r"(?!x)x", 30)  # matches nothing: returns when stdout ends
            stdout = '\n'.join(l for l in out.lines if l is not None)
            report['chat_input'] = turns
            report['chat_stdout'] = stdout
            report['chat_exit_code'] = process.returncode
            print('chat stdout:', stdout, sep='\n', flush=True)
            print('chat exit', process.returncode, flush=True)
            ok &= process.returncode == 0 and 'green' in stdout.lower()
        else:
            report['requests'] = []
            for content, stream in (('What is the capital of France? Answer in one word.', False),
                                    ('List the numbers from 1 to 5, separated by commas.', True)):
                status, kind, text, finish = chat_request(port, alias, content, stream)
                report['requests'].append({'stream': stream, 'question': content, 'status': status, 'content_type': kind,
                                           'answer': text, 'finish': finish})
                print(f'POST /v1/chat/completions stream={stream} status={status} type={kind} finish={finish} answer={text!r}',
                      flush=True)
                ok &= status == 200 and bool(text.strip()) and finish == 'stop'
            ok &= 'paris' in report['requests'][0]['answer'].lower() and '1' in report['requests'][1]['answer'] and \
                '5' in report['requests'][1]['answer']
    finally:
        owned = sorted(set(owned + descendants(process.pid)))
        code = stop(process)
        report['exit_code'] = code
        print('exit code', code, flush=True)
        left = [pid for pid in owned if Path(f'/proc/{pid}').exists()]
        report['remaining_owned_pids'] = left
        ok &= not left
        report['gpu_after'] = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
        print('descendants left:', left or 'none', flush=True)
        report['stderr_tail'] = [l for l in err.lines[-40:] if l is not None]
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    ok &= code == 0
    print('RESULT', 'OK' if ok else 'FAILED', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
