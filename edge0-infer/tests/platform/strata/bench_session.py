"""Benchmark one Edge0 145B configuration through `edge0-infer serve` with edge0-strata's own benchmark.

Standard library only; runs on Linux and Windows. One call measures one configuration:

1. checks that the GPU has no compute process and (where the driver reports it) little memory in use;
2. starts `edge0-infer serve --port 0` with the given --model-option and --strata-arg values;
3. records the engine's description endpoints (/health, /v1/status, /props, /metrics) and its expert cache size;
4. runs edge0-strata's `tools/strata_bench.py --url <serve>` unchanged (its default suite, --runs per test);
5. stops serve, checks that no process it started is left, waits until the GPU memory falls back, then cools down;
6. appends one JSON line (machine, code, command, engine description, the benchmark's results.json) to --record.

The record is the raw data: one line per configuration, appended, never rewritten. Logs and the benchmark's
own results folder are kept next to the record, in <record>.d/<label>-<time>/.

Pass values that start with "-" in the --name=value form, for example
  --strata-arg=--pool-workers --strata-arg=24 --model-option=du=q8

Exit status: 0 when the benchmark finished and serve stopped cleanly; 3 when the GPU was busy (nothing started);
1 otherwise. A line is appended in every case except a busy GPU.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_info  # noqa: E402
from smoke_session import Session  # noqa: E402

SCHEMA = 'edge0-145b-bench/1'


def now():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec='seconds')


def run(command, **kwargs):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                timeout=60, **kwargs)
        return result.returncode, result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, repr(error)


def gpu_state(index):
    """The GPU's name, driver, memory and compute processes as nvidia-smi reports them (None where it does not)."""
    state = {'index': index}
    code, out = run(['nvidia-smi', '-i', str(index), '--query-gpu=name,driver_version,memory.used,memory.total',
                     '--format=csv,noheader,nounits'])
    if code != 0:
        state['error'] = out.strip()
        return state
    fields = [field.strip() for field in out.strip().split(',')]
    number = lambda text: int(text) if text.isdigit() else None
    state.update(name=fields[0], driver=fields[1], memory_used_mib=number(fields[2]), memory_total_mib=number(fields[3]))
    code, out = run(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid,process_name,used_memory',
                     '--format=csv,noheader'])
    state['compute_processes'] = [line.strip() for line in out.splitlines() if line.strip()] if code == 0 else None
    code, out = run(['nvidia-smi'])
    match = re.search(r'CUDA Version:\s*([0-9.]+)', out or '')
    state['cuda_driver_api'] = match[1] if match else None
    return state


def memory_bytes():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                        ('available', ctypes.c_ulonglong), ('page_total', ctypes.c_ulonglong),
                        ('page_available', ctypes.c_ulonglong), ('virtual_total', ctypes.c_ulonglong),
                        ('virtual_available', ctypes.c_ulonglong), ('extended', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(status)
        return status.total if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cpu_name():
    if os.name == 'nt':
        return os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def machine(gpu):
    code, nvcc = run(['nvcc', '--version'])
    release = re.search(r'release ([0-9.]+), V([0-9.]+)', nvcc or '') if code == 0 else None
    return {'system': platform.platform(), 'architecture': platform.machine(), 'cpu': cpu_name(),
            'logical_cpus': os.cpu_count(), 'memory_bytes': memory_bytes(), 'gpu': gpu,
            'cuda_toolkit': release[2] if release else None}


def code_state(strata_dir, cli):
    repository = Path(strata_dir).resolve().parent
    state = {'repository_commit': None, 'repository_dirty': None, 'cli': str(cli)}
    code, out = run(['git', '-C', str(repository), 'rev-parse', 'HEAD'])
    if code == 0:
        state['repository_commit'] = out.strip()
        code, out = run(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=no'])
        state['repository_dirty'] = bool(out.strip()) if code == 0 else None
    code, out = run([str(cli), 'info'])
    state['cli_info'] = out.splitlines()[:5] if code == 0 else None
    return state


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        return {'unreadable': repr(error)}


def busy(state, limit_mib):
    if 'error' in state:
        return 'nvidia-smi failed: ' + state['error']
    if state.get('compute_processes'):
        return 'compute processes on the GPU: ' + '; '.join(state['compute_processes'])
    used = state.get('memory_used_mib')
    if used is not None and used > limit_mib:
        return f'{used} MiB of GPU memory in use (more than {limit_mib})'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli', required=True, help='edge0-infer executable')
    parser.add_argument('--model', required=True, help='Edge0 145B model directory')
    parser.add_argument('--strata-dir', required=True, help='edge0-strata directory')
    parser.add_argument('--strata-python', help='its Python (default: the .venv inside it)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-option', action='append', default=[], help='name=value, repeatable')
    parser.add_argument('--strata-arg', action='append', default=[], help='one engine word, repeatable')
    parser.add_argument('--label', required=True, help='a name for this configuration, e.g. 120b-q3k')
    parser.add_argument('--record', required=True, help='JSON Lines file the result is appended to')
    parser.add_argument('--runs', type=int, default=3, help="the benchmark's runs per test (default 3)")
    parser.add_argument('--idle-mib', type=int, default=500, help='most GPU memory in use to count as idle')
    parser.add_argument('--cooldown', type=int, default=120, help='seconds to wait after the GPU memory falls back')
    args = parser.parse_args()

    strata = Path(args.strata_dir).resolve()
    python = Path(args.strata_python) if args.strata_python else (
        strata / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    bench = strata / 'tools' / 'strata_bench.py'
    gpu = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    record = Path(args.record).resolve()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    keep = record.with_name(record.name + '.d') / f'{args.label}-{stamp}'

    before = gpu_state(gpu)
    reason = busy(before, args.idle_mib)
    if reason:
        print(f'GPU {gpu} is busy, nothing started: {reason}', file=sys.stderr)
        sys.exit(3)
    keep.mkdir(parents=True)

    model_name = json.loads((Path(args.model) / 'model.json').read_text(encoding='utf-8'))['model_name']
    if model_name not in ('edge0-145b-q3k', 'edge0-145b-fp4', 'edge0-120b-q3k', 'edge0-120b-fp4'):
        parser.error('unregistered release model_name')
    entry = {'model_name': model_name, 'experts': model_name.rsplit('-', 1)[1], 'schema': SCHEMA, 'label': args.label, 'started': now(), 'machine': machine(before),
             'code': code_state(strata, args.cli), 'model': str(Path(args.model).resolve()),
             'model_options': args.model_option, 'strata_args': args.strata_arg, 'runs': args.runs, 'files': str(keep)}
    command = [args.cli, 'serve', '--model', args.model, '--strata-dir', str(strata), '--device', args.device,
               '--port', '0']
    if args.strata_python:
        command += ['--strata-python', args.strata_python]
    for option in args.model_option:
        command += ['--model-option', option]
    for word in args.strata_arg:
        command += ['--strata-arg', word]
    entry['serve_command'] = command

    ok = False
    session = Session(command, keep / 'serve-stderr.log')
    try:
        port = int(session.ready('serve')[1])
        url = f'http://127.0.0.1:{port}'
        entry['ready'] = now()
        found, errors = engine_info.read(url)
        entry['engine'] = {'expert_cache_mib': [value for name, value in found if name.endswith('expert_cache_mib')],
                           'prefill_cap': [value for name, value in found if name.endswith('prefill_cap')],
                           'unreadable': errors}
        for path in ('/health', '/v1/status', '/props', '/metrics'):
            entry['engine'][path] = get_json(url + path)
        print(f'{args.label}: ready on port {port}; expert_cache_mib {entry["engine"]["expert_cache_mib"]}', flush=True)
        out = keep / 'bench'
        bench_command = [str(python), str(bench), '--url', url, '--runs', str(args.runs), '--out', str(out)]
        with open(keep / 'bench.log', 'w', encoding='utf-8') as log:
            code = subprocess.run(bench_command, cwd=strata, stdout=log, stderr=subprocess.STDOUT).returncode
        results = sorted(out.rglob('results.json'))
        entry['bench'] = {'command': bench_command, 'exit_code': code,
                          'results': json.loads(results[-1].read_text(encoding='utf-8')) if results else None}
        ok = code == 0 and bool(results)
    except Exception as error:
        entry['error'] = repr(error)
    finally:
        entry['stop'] = session.close()
        entry['ended'] = now()
    ok = ok and entry['stop']['clean']

    deadline = time.monotonic() + 300
    while True:
        after = gpu_state(gpu)
        used = after.get('memory_used_mib')
        if not after.get('compute_processes') and (used is None or used <= args.idle_mib) or time.monotonic() >= deadline:
            break
        time.sleep(5)
    entry['gpu_after'] = after
    entry['ok'] = ok
    with open(record, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'{args.label}: {"ok" if ok else "FAILED"}; appended to {record}; files in {keep}', flush=True)
    if after.get('compute_processes') or (after.get('memory_used_mib') or 0) > args.idle_mib:
        print(f'GPU {gpu} did not fall back to idle within 300 s: {after}', file=sys.stderr)
        ok = False
    else:
        time.sleep(args.cooldown)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
