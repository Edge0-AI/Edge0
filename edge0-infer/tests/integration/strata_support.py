"""Fixtures for delegate-mode tests: a fake edge0-strata directory and a 145B-shaped model directory."""
import ctypes
import signal
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
FILES = ['latent_dense_q8.bin', 'latent_dense_q4.bin', 'model-skeleton.gguf', 'ple-fp8.gguf',
         'profile.bin', 'experts.bin', 'labels.bin', 'dense.bin', 'index.txt', 'native_experts.txt']
# Only model_name is used from the algorithm team's service configuration.
MODEL_JSON = '{"model_name":"edge0-145b-q3k","exe":"ignored","args":["ignored"],"env":{"ignored":"value"}}'
ALIAS = 'edge0_145b'
TEMPLATE = {
    'exe': '${STRATA_DIR}/build/strata${EXE_SUFFIX}',
    'lib_dirs': ['${CUDA_LIB_DIR}'],
    'args': ['--pack', '${MODEL_DIR}', '--ple-io', '${PLE_IO}', '--max-context', '131072'],
    'cwd': '${STRATA_DIR}', 'tokenizer': '${MODEL_DIR}/tokenizer', 'model_name': 'edge0-145b',
    'env': {'STRATA_LATENT_DENSE': '${DU_FILE}', 'STRATA_LATENT_DENSE_FMT': '${DU_FMT}',
            'STRATA_LATENT_A': '${MODEL_DIR}/experts.bin', 'STRATA_LATENT_LABELS': '${MODEL_DIR}/labels.bin', 'STRATA_LAT_FUSED_Q3K': '${LAT_FUSED_Q3K}'},
}
# The desktop App's readiness rule (examples/desktop/src-tauri/src/service.rs, ready_url).
APP_READY = re.compile(r'^Listening: http://(?:127\.0\.0\.1|0\.0\.0\.0):(\d+) model=(\S.*)$')
# Only the Metal template declares this. The fake tools/mtp_gguf.py writes the output unless FAKE_MTP_MODE says otherwise.
PREPARE = [{
    'output': '${MODEL_DIR}/mtp-q2_0.gguf',
    'script': 'tools/mtp_gguf.py',
    'args': ['--rt', '${MODEL_DIR}/mtp', '--out', '${MODEL_DIR}/mtp-q2_0.gguf'],
}]


def write_templates(engine, body):
    """CUDA templates as given; Metal templates are the same body plus the prepare step."""
    models = Path(engine) / 'edge0' / 'models'
    models.mkdir(parents=True, exist_ok=True)
    metal = dict(body)
    metal['prepare'] = PREPARE
    text = json.dumps(body, indent=1)
    metal_text = json.dumps(metal, indent=1)
    for name in ('edge0-145b.json', 'edge0-120b.json'):
        (models / name).write_text(text)
        (models / name.replace('.json', '-metal.json')).write_text(metal_text)


def make_engine(root, contract='5'):
    from fake_strata import MTP_SCRIPT
    engine = Path(root) / 'edge0-strata'
    (engine / 'serve').mkdir(parents=True)
    (engine / 'tools').mkdir(parents=True)
    shutil.copy(HERE / 'fake_strata.py', engine / 'serve' / 'server.py')
    (engine / 'tools' / 'mtp_gguf.py').write_text(MTP_SCRIPT)
    (engine / 'edge0').mkdir(parents=True)
    (engine / 'edge0' / 'CONTRACT').write_text(contract + '\n')
    write_templates(engine, TEMPLATE)
    return engine


def make_model(root, skip=(), model_name='edge0-145b-q3k'):
    model = Path(root) / 'edge0-145b'
    model.mkdir(parents=True)
    if 'model.json' not in skip:
        (model / 'model.json').write_text(json.dumps({"model_name": model_name, "exe": "ignored", "args": ["--pack", "wrong"],
                                                         "env": {"STRATA_LAT_FUSED_Q3K": "wrong"}, "log": "ignored"}))
    for name in FILES:
        if name not in skip:
            (model / name).parent.mkdir(parents=True, exist_ok=True)
            (model / name).write_bytes(name.encode())
    for folder in ('mtp', 'tokenizer', *(['scales'] if model_name.endswith('-q3k') else [])):
        if folder not in skip:
            (model / folder).mkdir(parents=True, exist_ok=True)
            (model / folder / 'part.bin').write_bytes(b'12345')
    return model


def strata_args(engine):
    return ['--strata-dir', str(engine), '--strata-python', sys.executable]


def temp_dir(record):
    """The TMPDIR given to edge0-infer for this record: its private directories must not outlive it."""
    return Path(str(record) + '-tmp')


def environment(record, **fake):
    env = dict(os.environ)
    env.pop('EDGE0_STRATA_DIR', None)
    env['PYTHONUTF8'] = '1'
    env['FAKE_STRATA_RECORD'] = str(record)
    temp_dir(record).mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        env[key] = str(temp_dir(record))
    env.setdefault('FAKE_STRATA_CHUNK_DELAY', '0.01')
    for key, value in fake.items():
        env['FAKE_STRATA_' + key.upper()] = str(value)
    return env


def process_record(record, timeout=10):
    path = Path(record) / 'process.json'
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.read_text().endswith('\n'):
            return json.loads(path.read_text().splitlines()[0])
        time.sleep(0.05)
    raise AssertionError('fake strata never started')


def gone(pid, timeout=10):
    """True once the process (or, with a negative pid, the process group) no longer exists or is a zombie."""
    deadline = time.time() + timeout
    if os.name == 'nt':
        assert pid > 0
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.GetExitCodeProcess.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            assert ctypes.get_last_error() == 87, ('OpenProcess', pid, ctypes.get_last_error())
            return True
        try:
            while time.time() < deadline:
                code = ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(handle, ctypes.byref(code)), ctypes.get_last_error()
                if code.value != 259:
                    return True
                time.sleep(0.05)
            return False
        finally:
            kernel.CloseHandle(handle)
    while time.time() < deadline:
        try:
            if pid < 0:
                os.killpg(-pid, 0)
            else:
                os.kill(pid, 0)
                status = Path(f'/proc/{pid}/status')
                if status.exists() and 'State:\tZ' in status.read_text():
                    return True
                if sys.platform == 'darwin':
                    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
                    if state.startswith('Z') or not state:
                        return True
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def assert_no_leftovers(record, temp_empty=True):
    """No fake process is left, and (unless edge0-infer was killed outright) no private temporary directory."""
    if temp_empty and temp_dir(record).exists():
        left = sorted(p.name for p in temp_dir(record).iterdir())
        assert not left, ('temporary directories left behind', left)
    process = process_record(record)
    assert gone(process['pid']), ('fake strata still running', process)
    assert gone(process['child']), ('fake engine child still running', process)
    if process.get('pgid'):
        assert gone(-process['pgid']), ('process group still alive', process)


class Running:
    """edge0-infer serve in delegate mode, read until its Listening line."""

    def __init__(self, cli, model, engine, record, extra=(), env=None, timeout=60, default_python=False):
        self.lines = []
        self.changed = threading.Condition()
        self.started = time.time()
        self.process = subprocess.Popen([str(cli), 'serve', '--model', str(model), '--port', '0', *(['--strata-dir', str(engine)] if default_python else strata_args(engine)), *extra],
                                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        env=env or environment(record))
        self.port = None
        self.listening_at = None

        def read():
            for line in self.process.stderr:
                with self.changed:
                    self.lines.append(line)
                    match = APP_READY.match(line.rstrip('\n'))
                    if match and self.port is None:
                        self.port = int(match[1])
                        self.alias = match[2]
                        self.listening_at = time.time()
                    self.changed.notify_all()
            with self.changed:
                self.changed.notify_all()
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        with self.changed:
            self.changed.wait_for(lambda: self.port is not None or self.process.poll() is not None, timeout=timeout)
        assert self.port, ''.join(self.lines)

    def text(self):
        with self.changed:
            return ''.join(self.lines)

    def wait_for(self, predicate, timeout=20):
        with self.changed:
            return self.changed.wait_for(lambda: predicate(''.join(self.lines)), timeout=timeout)

    def stop(self, sig=None):
        import signal
        if self.process.poll() is None:
            self.process.send_signal(sig or (signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM))
        code = self.process.wait(90)
        self.reader.join(10)
        return code
