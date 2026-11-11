"""Fetching update files: small documents, and large archives with resume and progress.

This is the only network code of the update path. It is reached from `strata update` / `strata components update` and
the app's explicit "Update" step, never from a start or a serve: ``require_acquisition`` is called by the caller before
anything here runs, and it refuses under ``STRATA_NO_ACQUIRE``.

A *source* is where the feed lives: an ``https://`` URL (or ``http://`` to a loopback address, which is how a local test
feed is served), a ``file://`` URL, or a plain folder / file path. Everything the feed points at is resolved against it
by `resolve_ref`, which refuses a reference that leaves the source.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import threading
import email.utils
import datetime as _dt
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from strata_control import tls

tls.install()      # #3722: the operating system's certificates, like the browser

CHUNK = 1 << 20
SMALL_LIMIT = 8 << 20
ENV_ALLOW_HTTP = "STRATA_UPDATE_ALLOW_HTTP"


class TransferError(RuntimeError):
    """Something could not be fetched. `resumable` says whether a partial file was kept for the next attempt."""

    def __init__(self, message: str, *, resumable: bool = False):
        super().__init__(message)
        self.resumable = resumable


class ChecksumMismatch(TransferError):
    """The bytes arrived but are not the bytes the signed feed names. The partial file was deleted."""


class Cancelled(TransferError):
    """The caller asked to stop. The partial file is kept."""

    def __init__(self, message: str = "cancelled"):
        super().__init__(message, resumable=True)


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def kind_of(source: str) -> str:
    """"http", "https", "file" or "path"."""
    parsed = urllib.parse.urlparse(source)
    if len(parsed.scheme) == 1:                     # a Windows drive letter
        return "path"
    if parsed.scheme in ("http", "https", "file"):
        return parsed.scheme
    if parsed.scheme:
        raise TransferError(f"the update source scheme {parsed.scheme!r} is not supported (https, a loopback http, file or a folder)")
    return "path"


def check_url_policy(url: str, environ=None) -> None:
    environ = os.environ if environ is None else environ
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http":
        if _is_loopback(parsed.hostname or "") or environ.get(ENV_ALLOW_HTTP) == "1":
            return
        raise TransferError(f"{url} is plain http; updates are fetched over https (a loopback address is the only exception)")
    raise TransferError(f"{url} is not an https address")


def _local_path(source: str) -> Path:
    if source.startswith("file:"):
        parsed = urllib.parse.urlparse(source)
        path = urllib.request.url2pathname(parsed.path)
        if os.name == "nt" and path.startswith(("/", "\\")) and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(source)


def base_of(source: str) -> str:
    """The folder / URL prefix that references in the feed are relative to."""
    kind = kind_of(source)
    if kind in ("http", "https"):
        return source.rsplit("/", 1)[0] + "/"
    path = _local_path(source)
    return str(path.parent if path.suffix else path)


def resolve_ref(base: str, ref: str, environ=None) -> str:
    """The absolute location a feed reference points at. A relative reference stays inside the base; an absolute one must
    pass the URL policy (https, or a loopback http) and, for a local base, may not be a remote address at all."""
    if not isinstance(ref, str) or not ref or "\0" in ref or "\\" in ref:
        raise TransferError(f"the feed contains an unusable file reference {ref!r}")
    parsed = urllib.parse.urlparse(ref)
    kind = kind_of(base)
    if parsed.scheme:
        if kind in ("path", "file"):
            raise TransferError(f"the local feed refers to a remote address ({ref}); a local feed may only name files inside its own folder")
        check_url_policy(ref, environ)
        return ref
    if ref.startswith("/") or ".." in ref.split("/"):
        raise TransferError(f"the feed reference {ref!r} leaves the feed's own location")
    if kind in ("http", "https"):
        return urllib.parse.urljoin(base if base.endswith("/") else base + "/", ref)
    root = _local_path(base).resolve()
    target = (root / ref).resolve()
    if root != target and root not in target.parents:
        raise TransferError(f"the feed reference {ref!r} leaves the feed's own folder")
    return str(target)


class _PolicyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url_policy(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_PolicyRedirects)
_USER_AGENT = "strata-updater/1"


_SERVER_DATE: list = []


def server_date():
    """The time the update server named in the `Date` header of its last answer (a timezone-aware datetime), or None. It lets a feed that
    looks expired be told from a PC whose clock is wrong."""
    return _SERVER_DATE[-1] if _SERVER_DATE else None


def _remember_date(response) -> None:
    try:
        parsed = email.utils.parsedate_to_datetime(response.headers.get("Date") or "")
        if parsed is not None:
            _SERVER_DATE[:] = [parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)]
    except (TypeError, ValueError):
        pass


FIRST_TRY_SECONDS = 12.0


def _open_small(request, timeout: float, limit: int) -> bytes:
    with _opener.open(request, timeout=timeout) as response:
        _remember_date(response)
        return response.read(limit + 1)


def read_small(location: str, *, limit: int = SMALL_LIMIT, timeout: float = 30) -> bytes:
    """A small document (the feed, its signature). Larger than `limit` is refused, not truncated.
    A web address gets one more try when the first one times out or the connection fails: about one request in 25 to github.com hung for the whole
    timeout on a PC whose next 24 answered in half a second (measured on a 3060 PC: the first-run card said "offline" a moment before the install's
    own read worked). The first try therefore waits at most 12 s; the second the full `timeout`."""
    kind = kind_of(location)
    _SERVER_DATE.clear()                                 # only the answer of this read counts
    try:
        if kind in ("http", "https"):
            check_url_policy(location)
            request = urllib.request.Request(location, headers={"User-Agent": _USER_AGENT, "Cache-Control": "no-cache"})
            try:
                data = _open_small(request, min(timeout, FIRST_TRY_SECONDS), limit)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as first:
                if tls.is_certificate_error(first):
                    raise
                _SERVER_DATE.clear()
                data = _open_small(request, timeout, limit)
        else:
            path = _local_path(location)
            if path.stat().st_size > limit:
                raise TransferError(f"{location} is larger than {limit} bytes")
            data = path.read_bytes()
    except TransferError:
        raise
    except urllib.error.HTTPError as exc:
        raise TransferError(f"the update server answered {exc.code} for {location}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        hint = ""
        if tls.is_certificate_error(exc):
            raise TransferError(tls.explain(tls.host_of(location))) from None
        raise TransferError(f"could not read {location} ({type(exc).__name__}: {reason}){hint}") from None
    if len(data) > limit:
        raise TransferError(f"{location} is larger than {limit} bytes")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(location: str, partial: Path, *, size: int, sha256: str,
             progress: Callable[[int, int], None] | None = None,
             cancel: threading.Event | None = None, timeout: float = 60) -> Path:
    """Fetch `location` into `partial` (resuming what is there) and return it once it has exactly `size` bytes and the
    SHA-256 `sha256`. An interrupted transfer keeps the partial file (`TransferError.resumable`); a complete file with the
    wrong hash is deleted, because resuming cannot repair it."""
    partial = Path(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:
        partial.unlink()
        have = 0
    if have < size:
        _transfer(location, partial, have, size, progress, cancel, timeout)
    elif progress:
        progress(size, size)
    final = partial.stat().st_size
    if final != size:
        raise TransferError(f"{location} delivered {final} of {size} bytes; the partial file is kept and the next try resumes it",
                            resumable=True)
    if sha256_file(partial) != sha256.lower():
        partial.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{location} does not match the SHA-256 in the signed feed; the download was deleted")
    return partial


def _transfer(location, partial: Path, have: int, size: int, progress, cancel, timeout) -> None:
    kind = kind_of(location)
    done = have
    try:
        if kind in ("http", "https"):
            check_url_policy(location)
            headers = {"User-Agent": _USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(location, headers=headers)
            with _opener.open(request, timeout=timeout) as response:
                if have and not _resumes_at(response, have, size):
                    if response.status == 206:               # another range: its bytes are not the file's, ask again from the start
                        partial.unlink(missing_ok=True)
                        return _transfer(location, partial, 0, size, progress, cancel, timeout)
                    have = done = 0                          # the server ignored the range and sent the whole file: start over, never append to a prefix
                _copy(response, partial, "wb" if not have else "ab", done, size, progress, cancel)
        else:
            path = _local_path(location)
            with path.open("rb") as source:
                source.seek(have)
                _copy(source, partial, "ab" if have else "wb", have, size, progress, cancel)
    except Cancelled:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            partial.unlink(missing_ok=True)
            raise TransferError(f"the server could not resume {location}; the partial file was removed, try again",
                                resumable=False) from None
        raise TransferError(f"the update server answered {exc.code} for {location}") from None
    except (urllib.error.URLError, OSError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        if tls.is_certificate_error(exc):
            raise TransferError(tls.explain(tls.host_of(location)), resumable=True) from None
        got = partial.stat().st_size if partial.exists() else 0
        reason = getattr(exc, "reason", exc)
        raise TransferError(f"the download of {location} was interrupted after {got} of {size} bytes "
                            f"({type(exc).__name__}: {reason}); the partial file is kept and the next try resumes it",
                            resumable=True) from None


_CONTENT_RANGE = re.compile(r"\s*bytes (\d+)-(\d+)/(\d+|\*)\s*$")


def _resumes_at(response, have: int, size: int) -> bool:
    """Whether a resumed answer is a 206 whose Content-Range starts at byte `have` of a file of `size` bytes."""
    if response.status != 206:
        return False
    m = _CONTENT_RANGE.match(response.headers.get("Content-Range") or "")
    return bool(m) and int(m.group(1)) == have and (m.group(3) == "*" or int(m.group(3)) == size)


def _copy(source, partial: Path, mode: str, done: int, size: int, progress, cancel) -> None:
    with partial.open(mode) as out:
        while True:
            if cancel is not None and cancel.is_set():
                out.flush()
                raise Cancelled()
            block = source.read(CHUNK)
            if not block:
                break
            if done + len(block) > size:
                raise TransferError("the server sent more bytes than the feed says; the download was stopped")
            out.write(block)
            done += len(block)
            if progress:
                progress(done, size)
        out.flush()
