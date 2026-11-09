import hashlib
from http.client import IncompleteRead
import io
import json

import pytest

from edge0_35b import fetch_source as fetch

REPO, REV = 'org/model', 'a' * 40
BIG = b'large-file-content' * 64
SMALL = b'{"small": true}'
ENDPOINT = 'https://fake.invalid'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Response:
    def __init__(self, status: int, body: bytes, headers: dict, cut: int | None = None):
        self.status, self.headers, self.cut = status, headers, cut
        self.stream = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size=-1):
        if self.cut is not None:
            remaining = self.cut - self.stream.tell()
            if remaining <= 0:
                raise IncompleteRead(b'', 1)
            size = remaining if size is None or size < 0 else min(size, remaining)
        return self.stream.read(size)


class Remote:
    """A fake Hugging Face endpoint in place of the network layer; no sockets are opened."""

    def __init__(self):
        self.files = {'model.safetensors': BIG, 'config.json': SMALL}
        self.lfs = {'model.safetensors'}
        self.lfs_sha = {'model.safetensors': sha(BIG)}
        self.honor_range = True
        self.cut = {}  # file name -> number of body bytes delivered before the connection drops
        self.served = []  # (name, Range header or None, bytes in the body)
        self.endpoint = ENDPOINT

    def info(self):
        siblings = []
        for name, data in self.files.items():
            s = {'rfilename': name, 'size': len(data)}
            if name in self.lfs:
                s['lfs'] = {'sha256': self.lfs_sha[name], 'size': len(data)}
            siblings.append(s)
        return {'sha': REV, 'private': False, 'gated': False, 'siblings': siblings}

    def get(self, url, timeout, headers=None):
        path = url[len(ENDPOINT):]
        if path.startswith(f'/api/models/{REPO}/revision/{REV}'):
            return Response(200, json.dumps(self.info()).encode(), {})
        prefix = f'/{REPO}/resolve/{REV}/'
        name = path[len(prefix):] if path.startswith(prefix) else None
        if name not in self.files:
            raise fetch.FetchError(f'{url}: HTTP Error 404')
        body, status, extra = self.files[name], 200, {}
        wanted = (headers or {}).get('Range')
        if wanted and self.honor_range:
            first, _, last = wanted[len('bytes='):].partition('-')
            if int(first) >= len(body):
                # What the real network layer raises for HTTP 416 Range Not Satisfiable.
                raise fetch.FetchError(f'{url}: HTTP Error 416', 416)
            end = int(last) + 1 if last else len(body)
            extra = {'Content-Range': f'bytes {first}-{end - 1}/{len(body)}'}
            body, status = body[int(first):end], 206
        self.served.append((name, wanted, len(body)))
        return Response(status, body, extra, self.cut.pop(name, None))


@pytest.fixture
def remote(monkeypatch):
    state = Remote()
    monkeypatch.setattr(fetch, '_get', state.get)
    return state


def lock_for(remote):
    return {'repository': REPO, 'revision': REV, 'files': {n: sha(d) for n, d in remote.files.items()}}


@pytest.mark.l1
def test_probe_passes_without_downloading_large_files(remote):
    report = fetch.probe(lock_for(remote), remote.endpoint, 5)
    assert report['status'] == 'passed', report
    assert report['files']['model.safetensors'] == {'method': 'lfs-sha256', 'result': 'passed', 'range_status': 206}
    assert report['files']['config.json']['method'] == 'download-sha256'
    assert ('model.safetensors', 'bytes=0-0', 1) in remote.served
    assert all(n != len(BIG) for _, _, n in remote.served)


@pytest.mark.l1
def test_probe_rejects_lfs_digest_mismatch(remote):
    remote.lfs_sha['model.safetensors'] = sha(b'other')
    report = fetch.probe(lock_for(remote), remote.endpoint, 5)
    assert report['status'] == 'failed'
    assert any('model.safetensors' in e and 'LFS' in e for e in report['errors'])


@pytest.mark.l1
def test_probe_rejects_missing_remote_file(remote):
    lock = lock_for(remote)
    lock['files']['tokenizer.json'] = sha(b'x')
    report = fetch.probe(lock, remote.endpoint, 5)
    assert report['status'] == 'failed'
    assert any(e.startswith('tokenizer.json: missing') for e in report['errors'])


@pytest.mark.l1
def test_probe_rejects_small_file_content_mismatch(remote):
    lock = lock_for(remote)
    remote.files['config.json'] = b'{"changed": true}'
    report = fetch.probe(lock, remote.endpoint, 5)
    assert report['status'] == 'failed'
    assert any(e.startswith('config.json: downloaded sha256') for e in report['errors'])


@pytest.mark.l1
def test_probe_rejects_ignored_range_request(remote):
    remote.honor_range = False
    report = fetch.probe(lock_for(remote), remote.endpoint, 5)
    assert report['status'] == 'failed'
    assert any('range request returned 200' in e for e in report['errors'])


@pytest.mark.l1
def test_download_selected_files(remote, tmp_path):
    out = tmp_path / 'source'
    fetch.download(lock_for(remote), remote.endpoint, 5, out, ['config.json'])
    assert sorted(p.name for p in out.iterdir()) == ['config.json']
    assert (out / 'config.json').read_bytes() == SMALL
    assert not out.with_name('source.partial').exists()


@pytest.mark.l1
def test_download_refuses_existing_output(remote, tmp_path):
    out = tmp_path / 'source'
    out.mkdir()
    (out / 'keep').write_text('preserve')
    with pytest.raises(fetch.FetchError, match='output already exists'):
        fetch.download(lock_for(remote), remote.endpoint, 5, out)
    assert sorted(p.name for p in out.iterdir()) == ['keep']
    assert not remote.served


@pytest.mark.l1
def test_download_corruption_keeps_verified_files_only(remote, tmp_path):
    lock = lock_for(remote)
    remote.files['model.safetensors'] = BIG[:-1] + b'!'
    out = tmp_path / 'source'
    partial = out.with_name('source.partial')
    with pytest.raises(fetch.FetchError, match='model.safetensors: downloaded sha256') as failure:
        fetch.download(lock, remote.endpoint, 5, out, ['config.json', 'model.safetensors'])
    assert f'partial download kept at {partial}' in str(failure.value)
    assert not out.exists()
    assert sorted(p.name for p in partial.iterdir()) == ['config.json']
    # The remote is repaired: the rerun skips the verified file and publishes.
    remote.files['model.safetensors'] = BIG
    remote.served.clear()
    report = fetch.download(lock, remote.endpoint, 5, out, ['config.json', 'model.safetensors'])
    assert report == {'skipped': ['config.json'], 'resumed': [], 'downloaded': ['model.safetensors']}
    assert [n for n, _, _ in remote.served] == ['model.safetensors']
    assert (out / 'model.safetensors').read_bytes() == BIG and not partial.exists()


@pytest.mark.l1
def test_interrupted_download_resumes(remote, tmp_path):
    lock = lock_for(remote)
    out = tmp_path / 'source'
    partial = out.with_name('source.partial')
    remote.cut['model.safetensors'] = 300
    with pytest.raises(fetch.FetchError, match='rerun the same command to resume'):
        fetch.download(lock, remote.endpoint, 5, out, ['config.json', 'model.safetensors'])
    assert not out.exists()
    assert (partial / 'config.json').read_bytes() == SMALL
    assert (partial / 'model.safetensors.part').read_bytes() == BIG[:300]
    remote.served.clear()
    report = fetch.download(lock, remote.endpoint, 5, out, ['config.json', 'model.safetensors'])
    assert report == {'skipped': ['config.json'], 'resumed': ['model.safetensors'], 'downloaded': []}
    assert remote.served == [('model.safetensors', 'bytes=300-', len(BIG) - 300)]
    assert (out / 'model.safetensors').read_bytes() == BIG and (out / 'config.json').read_bytes() == SMALL
    assert sorted(p.name for p in out.iterdir()) == ['config.json', 'model.safetensors']
    assert not partial.exists()


@pytest.mark.l1
def test_resume_restarts_when_ranges_are_not_supported(remote, tmp_path):
    lock = lock_for(remote)
    out = tmp_path / 'source'
    remote.cut['model.safetensors'] = 300
    with pytest.raises(fetch.FetchError):
        fetch.download(lock, remote.endpoint, 5, out, ['model.safetensors'])
    remote.honor_range = False
    remote.served.clear()
    fetch.download(lock, remote.endpoint, 5, out, ['model.safetensors'])
    assert remote.served == [('model.safetensors', 'bytes=300-', len(BIG))]
    assert (out / 'model.safetensors').read_bytes() == BIG


@pytest.mark.l1
def test_full_length_corrupted_part_restarts_in_the_same_run(remote, tmp_path):
    lock = lock_for(remote)
    out = tmp_path / 'source'
    partial = out.with_name('source.partial')
    partial.mkdir()
    (partial / 'model.safetensors.part').write_bytes(b'!' * len(BIG))
    report = fetch.download(lock, remote.endpoint, 5, out, ['model.safetensors'])
    assert report == {'skipped': [], 'resumed': [], 'downloaded': ['model.safetensors']}
    # The range from the part's length is refused (416); the file is fetched again from the start.
    assert remote.served == [('model.safetensors', None, len(BIG))]
    assert (out / 'model.safetensors').read_bytes() == BIG and not partial.exists()


@pytest.mark.l1
def test_corrupted_finished_file_is_downloaded_again(remote, tmp_path):
    lock = lock_for(remote)
    out = tmp_path / 'source'
    partial = out.with_name('source.partial')
    partial.mkdir()
    (partial / 'config.json').write_bytes(b'damaged')
    report = fetch.download(lock, remote.endpoint, 5, out, ['config.json'])
    assert report == {'skipped': [], 'resumed': [], 'downloaded': ['config.json']}
    assert (out / 'config.json').read_bytes() == SMALL


@pytest.mark.l1
def test_publishes_only_after_every_file_matches(remote, tmp_path):
    lock = lock_for(remote)
    out = tmp_path / 'source'
    remote.cut['model.safetensors'] = 10
    with pytest.raises(fetch.FetchError):
        fetch.download(lock, remote.endpoint, 5, out)
    assert not out.exists()
    fetch.download(lock, remote.endpoint, 5, out)
    assert sorted(p.name for p in out.iterdir()) == ['config.json', 'model.safetensors']


@pytest.mark.l1
def test_partial_of_another_selection_is_refused(remote, tmp_path):
    out = tmp_path / 'source'
    partial = out.with_name('source.partial')
    partial.mkdir()
    (partial / 'other.bin').write_bytes(b'x')
    with pytest.raises(fetch.FetchError, match='another selection'):
        fetch.download(lock_for(remote), remote.endpoint, 5, out)
    assert (partial / 'other.bin').read_bytes() == b'x'


@pytest.fixture(params=['timeout', 'incomplete'])
def broken_response(request, monkeypatch):
    class Response:
        status = 206

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            if request.param == 'timeout':
                raise TimeoutError('body read timed out')
            raise IncompleteRead(b'', 1)

    monkeypatch.setattr(fetch, '_get', lambda *args, **kwargs: Response())


@pytest.mark.l1
@pytest.mark.parametrize('phase', ['metadata', 'files'])
def test_probe_reports_body_read_failure(broken_response, monkeypatch, phase):
    state = Remote()
    if phase == 'files':
        monkeypatch.setattr(fetch, 'revision_info', lambda *args: state.info())
    report = fetch.probe(lock_for(state), 'https://unused.invalid', 5)
    assert report['status'] == 'failed'
    if phase == 'metadata':
        assert report['errors'][0].startswith('revision metadata unavailable:')
    else:
        assert len(report['errors']) == 2
        assert all(f['result'] == 'failed' for f in report['files'].values())


@pytest.mark.l1
def test_download_body_read_failure_keeps_partial(broken_response, tmp_path):
    out = tmp_path / 'source'
    with pytest.raises(fetch.FetchError) as failure:
        fetch.download(lock_for(Remote()), 'https://unused.invalid', 5, out)
    assert isinstance(failure.value.__cause__, (TimeoutError, IncompleteRead))
    assert not out.exists()
    assert out.with_name('source.partial').is_dir()


SMALL_FILES = ['chat_template.jinja', 'config.json', 'generation_config.json', 'model.safetensors.index.json']


@pytest.mark.l3
def test_real_source_probe_and_small_downloads(tmp_path):
    """Needs network access to Hugging Face; a failure here is a failure, not a skip."""
    lock = json.loads(fetch.LOCK.read_text())
    report = fetch.probe(lock, 'https://huggingface.co', 120)
    assert report['status'] == 'passed', report['errors']
    assert len(report['files']) == len(lock['files']) == 11
    large = [n for n, f in report['files'].items() if f['method'] == 'lfs-sha256']
    assert len(large) == 7 and all(report['files'][n]['range_status'] == 206 for n in large)
    out = tmp_path / 'small'
    fetch.download(lock, 'https://huggingface.co', 120, out, SMALL_FILES)
    for name in SMALL_FILES:
        assert sha((out / name).read_bytes()) == lock['files'][name]
    assert sorted(p.name for p in out.iterdir()) == sorted(SMALL_FILES)


@pytest.mark.l1
def test_network_layer_keeps_http_status(monkeypatch):
    import urllib.error

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 416, 'Range Not Satisfiable', {}, None)

    monkeypatch.setattr(fetch.urllib.request, 'urlopen', refuse)
    with pytest.raises(fetch.FetchError) as failure:
        fetch._get(ENDPOINT + '/x', 5, {'Range': 'bytes=10-'})
    assert failure.value.status == 416


class RecordingOpen:
    """Stands in for urllib.request.urlopen and keeps every request it receives."""

    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return Response(200, b'', {})


@pytest.mark.l1
def test_requests_send_an_explicit_user_agent_and_keep_range(monkeypatch):
    opened = RecordingOpen()
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', opened)
    fetch._get(ENDPOINT + '/file', 5, {'Range': 'bytes=0-0'})
    fetch._get(ENDPOINT + '/file', 5)
    for request in opened.requests:
        assert request.get_header('User-agent') == fetch.USER_AGENT
        assert not request.get_header('User-agent').startswith(('Python-urllib', 'curl', 'Mozilla'))
    assert opened.requests[0].get_header('Range') == 'bytes=0-0'


@pytest.mark.l1
def test_caller_user_agent_takes_precedence(monkeypatch):
    opened = RecordingOpen()
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', opened)
    fetch._get(ENDPOINT + '/file', 5, {'user-agent': 'caller/2', 'Range': 'bytes=4-'})
    assert opened.requests[0].get_header('User-agent') == 'caller/2'
    assert opened.requests[0].get_header('Range') == 'bytes=4-'


@pytest.mark.l1
def test_user_agent_survives_a_redirect(monkeypatch):
    # Mirrors redirect small files to another address that checks the User-Agent.
    opened = RecordingOpen()
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', opened)
    fetch._get(ENDPOINT + '/file', 5)
    original = opened.requests[0]
    redirected = fetch.urllib.request.HTTPRedirectHandler().redirect_request(
        original, io.BytesIO(), 307, 'Temporary Redirect', {}, ENDPOINT + '/api/resolve-cache/file')
    assert redirected.get_header('User-agent') == fetch.USER_AGENT
