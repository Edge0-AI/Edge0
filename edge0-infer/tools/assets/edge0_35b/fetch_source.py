#!/usr/bin/env python3
"""Probe or download the registered 35B source files from Hugging Face.

`probe` checks that the registered revision is public, that every registered
file exists with the registered sha256 (LFS metadata for large files, the
downloaded content for small ones), and that each large file can start
downloading (a one-byte range request). It never downloads a large file.
`download` writes the selected files into a new directory and verifies them;
an interrupted download resumes when the same command runs again.
"""

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

LOCK = Path(__file__).with_name('source.lock.json')
SMALL_LIMIT = 1 << 20
CHUNK = 1 << 20
# Some mirrors reject urllib's default User-Agent ("Python-urllib/3.x") on the
# addresses they redirect small files to, so every request names this tool.
USER_AGENT = 'edge0-assets-fetch/1'


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status when the server answered with an error


def _get(url: str, timeout: float, headers: dict | None = None):
    # A User-Agent the caller passes takes precedence over the default.
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, **(headers or {})})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise FetchError(f'{url}: {exc}', exc.code) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f'{url}: {exc}') from exc


def _file_url(endpoint: str, lock: dict, name: str) -> str:
    return f"{endpoint}/{lock['repository']}/resolve/{lock['revision']}/{urllib.parse.quote(name)}"


def revision_info(endpoint: str, lock: dict, timeout: float) -> dict:
    url = f"{endpoint}/api/models/{lock['repository']}/revision/{lock['revision']}?blobs=true"
    with _get(url, timeout) as response:
        return json.load(response)


def probe(lock: dict, endpoint: str, timeout: float) -> dict:
    report = {'status': 'passed', 'repository': lock['repository'], 'revision': lock['revision'], 'errors': [], 'files': {}}

    def fail(message):
        report['status'] = 'failed'
        report['errors'].append(message)

    try:
        info = revision_info(endpoint, lock, timeout)
    except (FetchError, ValueError, OSError, HTTPException) as exc:
        fail(f'revision metadata unavailable: {exc}')
        return report
    if info.get('sha') != lock['revision']:
        fail(f"revision mismatch: remote {info.get('sha')}")
    if info.get('private') or info.get('gated'):
        fail('revision is not publicly readable')
    remote = {s.get('rfilename'): s for s in info.get('siblings', [])}
    for name, digest in lock['files'].items():
        entry = {'method': None, 'result': 'failed'}
        report['files'][name] = entry
        sibling = remote.get(name)
        if sibling is None:
            fail(f'{name}: missing from the remote revision')
            continue
        lfs = sibling.get('lfs') or {}
        try:
            if lfs.get('sha256'):
                entry['method'] = 'lfs-sha256'
                if lfs['sha256'] != digest:
                    fail(f'{name}: remote LFS sha256 differs from the registered digest')
                    continue
                with _get(_file_url(endpoint, lock, name), timeout, {'Range': 'bytes=0-0'}) as response:
                    entry['range_status'] = response.status
                    body = response.read(2)
                if response.status != 206 or len(body) != 1:
                    fail(f'{name}: range request returned {response.status} with {len(body)} bytes')
                    continue
            else:
                entry['method'] = 'download-sha256'
                if sibling.get('size', 0) > SMALL_LIMIT:
                    fail(f'{name}: non-LFS file larger than {SMALL_LIMIT} bytes; not downloaded')
                    continue
                with _get(_file_url(endpoint, lock, name), timeout) as response:
                    data = response.read(SMALL_LIMIT + 1)
                if len(data) > SMALL_LIMIT:
                    fail(f'{name}: download exceeded {SMALL_LIMIT} bytes')
                    continue
                if hashlib.sha256(data).hexdigest() != digest:
                    fail(f'{name}: downloaded sha256 differs from the registered digest')
                    continue
        except (FetchError, OSError, HTTPException) as exc:
            fail(f'{name}: {exc}')
            continue
        entry['result'] = 'passed'
    return report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, timeout: float, part: Path) -> bool:
    """Download into `part`, continuing from its current length when the server honors ranges.

    Returns whether the download continued an existing `part`. A range starting at or past
    the end of the file (HTTP 416: `part` is already full length but its digest is wrong, or
    longer than the file) drops `part` and downloads from the start in the same run.
    """
    offset = part.stat().st_size if part.is_file() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    try:
        response = _get(url, timeout, headers)
    except FetchError as exc:
        if not offset or exc.status != 416:
            raise
        part.unlink()
        offset, response = 0, _get(url, timeout)
    with response:
        resumed = False
        expected, total = None, None
        if response.status == 206:
            content_range = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                        response.headers.get('Content-Range') or '')
            if not content_range:
                raise FetchError('invalid Content-Range in download response')
            start, end, total = map(int, content_range.groups())
            if start != offset or end < start or end >= total:
                raise FetchError('invalid Content-Range in download response')
            expected, resumed = end - start + 1, bool(offset)
        elif response.status != 200:
            raise FetchError(f'unexpected download status: {response.status}')
        length = response.headers.get('Content-Length')
        if length is not None:
            if not length.isdecimal() or (expected is not None and int(length) != expected):
                raise FetchError('invalid Content-Length in download response')
            expected = int(length)
        received = 0
        with part.open('ab' if resumed else 'wb') as f:
            for chunk in iter(lambda: response.read(CHUNK), b''):
                f.write(chunk)
                received += len(chunk)
        # read(size) may return clean EOF for a truncated Content-Length body.
        # Keep that verified prefix available to the next range request instead
        # of letting the digest mismatch path delete gigabytes of progress.
        if (expected is not None and received != expected) or \
                (total is not None and offset + received != total):
            raise FetchError('incomplete response; download can be resumed')
    return resumed


def download(lock: dict, endpoint: str, timeout: float, out: Path, names: list[str] | None = None) -> dict:
    """Download into `<out>.partial` and publish it as `out` once every file matches the lock.

    A failed or interrupted run keeps `<out>.partial`; running the same command
    again skips verified files, continues `<name>.part` files with range
    requests (from the start when the server ignores them) and downloads files
    whose digest does not match again.
    """
    names = names if names else list(lock['files'])
    unknown = [n for n in names if n not in lock['files']]
    if unknown:
        raise FetchError(f'not registered: {", ".join(unknown)}')
    if out.exists() or out.is_symlink():
        raise FetchError(f'output already exists: {out}')
    partial = out.with_name(out.name + '.partial')
    if partial.is_symlink() or (partial.exists() and not partial.is_dir()):
        raise FetchError(f'partial output is not a directory: {partial}')
    expected = set(names) | {n + '.part' for n in names}
    if partial.is_dir() and {p.name for p in partial.iterdir()} - expected:
        raise FetchError(f'{partial} holds files of another selection; delete it and rerun')
    partial.mkdir(parents=True, exist_ok=True)
    report = {'skipped': [], 'resumed': [], 'downloaded': []}
    try:
        for name in names:
            final, part = partial / name, partial / (name + '.part')
            if final.is_file():
                if _sha256(final) == lock['files'][name]:
                    report['skipped'].append(name)
                    continue
                final.unlink()
            if part.is_file() and _sha256(part) == lock['files'][name]:
                # Completed earlier but not yet renamed.
                os.rename(part, final)
                report['resumed'].append(name)
                continue
            resumed = _fetch(_file_url(endpoint, lock, name), timeout, part)
            (report['resumed'] if resumed else report['downloaded']).append(name)
            if _sha256(part) != lock['files'][name]:
                part.unlink()
                raise FetchError(f'{name}: downloaded sha256 differs from the registered digest')
            os.rename(part, final)
        if out.exists():
            raise FetchError(f'output appeared during download: {out}')
        os.rename(partial, out)
    except Exception as exc:
        raise FetchError(f'{exc}; partial download kept at {partial}; rerun the same command to resume') from exc
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--endpoint', default='https://huggingface.co', help='Hugging Face endpoint or mirror')
    parser.add_argument('--timeout', type=float, default=60.0, help='seconds per request')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('probe', help='verify the registered source without downloading large files')
    get = commands.add_parser('download', help='download registered files into a new directory')
    get.add_argument('--out', type=Path, required=True)
    get.add_argument('--files', nargs='+', help='registered file names (default: all)')
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    endpoint = args.endpoint.rstrip('/')
    if args.command == 'probe':
        report = probe(lock, endpoint, args.timeout)
        print(json.dumps(report, indent=2, sort_keys=True))
        sys.exit(0 if report['status'] == 'passed' else 1)
    try:
        report = download(lock, endpoint, args.timeout, args.out.absolute(), args.files)
    except (FetchError, OSError, HTTPException) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(json.dumps(report, sort_keys=True))
    print(f'Downloaded and verified {args.out}')


if __name__ == '__main__':
    main()
