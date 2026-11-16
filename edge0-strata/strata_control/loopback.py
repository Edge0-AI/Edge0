"""HTTP calls to this computer's own servers (127.0.0.1, localhost, ::1) never go through a proxy.

`urllib.request.urlopen` follows the system proxy (HTTP_PROXY, the Windows registry, a PAC file). A school or office PC sends
`http://127.0.0.1:PORT/health` to that proxy, which cannot reach it, and Strata then reports a healthy server as "did not answer".
Everything that talks to a local server uses `urlopen` below; any other address is passed on unchanged.
"""

from __future__ import annotations

import urllib.request
from urllib.parse import urlsplit

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LOCAL = {"127.0.0.1", "localhost", "::1"}


def is_loopback(url: str) -> bool:
    try:
        return (urlsplit(url).hostname or "").lower() in _LOCAL
    except ValueError:
        return False


def urlopen(request, timeout: float | None = None):
    url = request if isinstance(request, str) else request.full_url
    if is_loopback(url):
        return _OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)
