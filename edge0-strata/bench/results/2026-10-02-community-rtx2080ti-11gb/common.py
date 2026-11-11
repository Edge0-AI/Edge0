"""Shared helpers for the scripts in this folder: talk to a running Strata server, wait until it is idle,
follow its logs, parse the engine's per-request timing lines and strip private details from what is saved.

The API key is read at run time from the STRATA_API_KEY environment variable or from the server's config
file (--key-config, its "api_key"); it is never written to any output."""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from pathlib import Path

PROMPT_RE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+), (\d+) checkpoints")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)")
KV_RE = re.compile(r"KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM")
SUFFIX_RE = re.compile(r"suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted")


def api_key(key_config: str | None) -> str:
    key = os.environ.get("STRATA_API_KEY", "")
    if not key and key_config:
        key = json.loads(Path(key_config).read_text()).get("api_key") or ""
    return key


class Server:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get(self, path: str, timeout: float = 30) -> dict:
        req = urllib.request.Request(self.url + path, headers=self.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def totals(self) -> int:
        """Requests finished since the server started (GET /metrics): a jump of more than one around a measured
        request means someone else's request ran in between."""
        return int(self.get("/metrics")["totals"]["requests"])

    def wait_idle(self, settle_s: float = 3.0, poll_s: float = 1.0) -> dict:
        """Block until GET /status reports busy=false and queued=0 for `settle_s` seconds in a row."""
        quiet_since = None
        while True:
            s = self.get("/status")
            if not s.get("busy") and not s.get("queued"):
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since >= settle_s:
                    return s
            else:
                if quiet_since is not None:
                    print("  server busy (foreign request?), waiting ...", flush=True)
                quiet_since = None
            time.sleep(poll_s)

    def loaded(self) -> bool:
        return bool(self.get("/health").get("loaded"))


class LogTail:
    """Byte offset into a log file; `since()` returns the lines appended after `mark()`."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.offset = 0

    def mark(self):
        self.offset = self.path.stat().st_size

    def since(self) -> list[str]:
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        return data.decode("utf-8", errors="replace").splitlines()


def parse_engine(lines: list[str]) -> list[dict]:
    """One dict per 'strata serve: prompt ...' line, with the hit-rate / KV-streaming / suffix lines after it."""
    out = []
    for line in lines:
        m = PROMPT_RE.search(line)
        if m:
            g = m.groups()
            out.append({"prompt_tokens": int(g[0]), "reused": int(g[1]), "read": int(g[2]), "prompt_ms_log": int(g[3]),
                        "prompt_tok_s_log": float(g[4]), "generated": int(g[5]), "decode_ms_log": int(g[6]),
                        "decode_tok_s_log": float(g[7]), "drafts_accepted": int(g[8]), "drafts_proposed": int(g[9]),
                        "checkpoints": int(g[10]), "cancelled": "(cancelled)" in line, "line": line.strip()})
            continue
        if not out:
            continue
        for key, rx in (("hit", HIT_RE), ("kv", KV_RE), ("suffix", SUFFIX_RE)):
            m = rx.search(line)
            if m:
                if key == "hit":
                    out[-1].update(hit_rate_pct=float(m[1]), hit_line=line.strip())
                elif key == "kv":
                    out[-1].update(kv_vram_pct=float(m[1]), kv_block_reads=int(m[2]), kv_ram_mib=float(m[3]),
                                   kv_line=line.strip())
                else:
                    out[-1].update(suffix_windows=int(m[1]), suffix_accepted=int(m[2]), suffix_proposed=int(m[3]))
    return out


_HOST = socket.gethostname()
_HOME = str(Path.home())


def scrub(text: str) -> str:
    """Remove the install directory (STRATA_SCRUB_PREFIX, shown as <workspace>), the home directory, user name,
    a host name and LAN addresses from a log excerpt."""
    prefix = os.environ.get("STRATA_SCRUB_PREFIX")    # the directory that holds Strata and Strata-data
    if prefix:
        text = text.replace(prefix.rstrip("/") + "/", "<workspace>/")
    text = text.replace(_HOME, "<home>")
    text = text.replace(os.environ.get("USER", "\0"), "<user>")
    if _HOST and len(_HOST) > 8:                     # a short host name can be an ordinary word in the log
        text = text.replace(_HOST, "<host>")
    text = re.sub(r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b", "<lan-ip>", text)
    return text


def scrub_lines(lines: list[str]) -> list[str]:
    return [scrub(l) for l in lines if "from other devices" not in l]


def sse_lines(resp):
    """Yield (t_seconds, raw_line) for each SSE line of an HTTP response."""
    for raw in resp:
        yield time.perf_counter(), raw.decode("utf-8", errors="replace").rstrip("\r\n")
