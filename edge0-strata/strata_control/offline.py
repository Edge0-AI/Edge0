"""Offline policy: what "no download" means, and for whom (checklist U08).

The checklist asks for two claims that are easy to conflate, so this module keeps them apart:

* **runtime-only no-download** — the server, while serving, fetches nothing. The engine runs from
  local files; no model, library or asset is fetched mid-request. This can be true while the machine
  has a network and while other parts of the product do download.
* **strict offline** — nothing in the process tree reaches the network at all, including children,
  tool subprocesses and the agent runtime. This must hold even though a tool or a model file might
  want to fetch something.

They are tested separately because a passing runtime-only test says nothing about the second: the
first is about the server's own code paths, and the second is about every process it can start.

How each is enforced:

**Environment, for children.** Every child the server starts is given variables that the common
runtimes honour — Hugging Face, Transformers, pip, and the proxy variables that make an accidental
HTTP call fail fast rather than hang. This is the mechanism that reaches code we do not control.

**A guard, for our own fetches.** `check_url` is the single place the product decides whether a URL
may be fetched, so "offline" is one predicate rather than a scattered set of `if offline` checks. A
test asserts the server's image path calls it, because a guard that is not wired in is decoration.

**A preflight, for the profile.** A profile that names MCP servers, a `before_load` command or remote
image sources cannot be strict-offline, and saying so before the server starts is more useful than
failing on the first request.

**What this cannot do.** It cannot stop a child that ignores the environment and opens a socket
directly, because that would need a sandbox this product does not have on all three platforms.
`limits()` states that plainly, and the qualification kit asks for the network-isolated run that
would settle it. Recording the limit is the honest option; claiming enforcement would not be.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

#: Environment for children. The first group is what the Python ML stack reads; the second makes an
#: accidental HTTP call fail immediately instead of hanging on a timeout; the third tells tools that
#: ask that they are offline.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "NO_PROXY": "*",
    "no_proxy": "*",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "STRATA_OFFLINE": "1",
}

#: Schemes that are local by construction and therefore always allowed.
LOCAL_SCHEMES = ("file", "data", "blob")

#: Hosts that mean "this machine" and never leave it.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0")


@dataclass(frozen=True)
class Decision:
    """Whether a URL may be fetched, and the reason either way."""

    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class OfflineViolation(RuntimeError):
    """A fetch was attempted under a policy that forbids it."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"offline policy forbids fetching {url!r}: {reason}")
        self.url = url
        self.reason = reason


def check_url(url: str, *, offline: bool) -> Decision:
    """May this URL be fetched?

    A local scheme or a loopback host is allowed even under strict offline: talking to this machine
    is not a download, and forbidding it would break the local API the product is built on.
    """
    if not offline:
        return Decision(True, "not offline")
    if not url:
        return Decision(False, "no URL")
    parsed = urlparse(str(url))
    scheme = (parsed.scheme or "").lower()
    if scheme in LOCAL_SCHEMES:
        return Decision(True, f"the {scheme} scheme is local")
    if scheme in ("", "http", "https"):
        host = (parsed.hostname or "").lower()
        if scheme == "" and not host:
            # A bare path is a local file, which is how a profile usually names an image.
            return Decision(True, "a local path, not a URL")
        if host in LOCAL_HOSTS:
            return Decision(True, f"{host} is this machine")
        return Decision(False, f"{scheme or 'http'} to {host or 'an unnamed host'} would leave this machine")
    return Decision(False, f"the {scheme} scheme is not local")


def require_url(url: str, *, offline: bool) -> None:
    """`check_url`, as an exception. Used where a refusal must stop the operation."""
    decision = check_url(url, offline=offline)
    if not decision.allowed:
        raise OfflineViolation(url, decision.reason)


def child_environment(*, offline: bool, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a child process gets.

    Under strict offline the variables are *set*, not merely left alone: a child that would otherwise
    read the parent's proxy settings must be told not to use them.
    """
    env = dict(base if base is not None else os.environ)
    if offline:
        env.update(OFFLINE_ENV)
    else:
        # Not offline: remove the markers so a previous run's policy cannot leak into this one, which
        # would make a later server mysteriously refuse to fetch.
        for key in OFFLINE_ENV:
            env.pop(key, None)
    return env


def profile_requirements(cfg: Mapping[str, Any]) -> list[str]:
    """The parts of a profile that need the network, named so a refusal can explain itself."""
    needs: list[str] = []
    servers = cfg.get("mcp_servers") or cfg.get("mcpServers")
    if servers:
        count = len(servers) if hasattr(servers, "__len__") else 1
        needs.append(f"the profile declares {count} MCP server(s), which are started as subprocesses "
                     "and may fetch tools")
    for key in ("before_load", "beforeLoad"):
        if cfg.get(key):
            needs.append(f"the profile declares a {key} command, which runs before the model loads")
    for key in ("images", "vision"):
        section = cfg.get(key)
        if isinstance(section, dict):
            for field in ("remote_sources", "allowed_hosts", "url"):
                if section.get(field):
                    needs.append(f"the profile's {key}.{field} permits network sources")
    return needs


def ensure_strictly_offline(cfg: Mapping[str, Any]) -> list[str]:
    """Why this profile cannot be strict-offline. Empty means it can.

    Refusing before the server starts is more useful than failing on the first request, and it keeps
    the promise precise: a profile that permits a network source is not offline, whatever the flag
    says.
    """
    return profile_requirements(cfg)


def limits() -> list[str]:
    """What this policy does not enforce. Stated so it is not over-read as a sandbox."""
    return [
        "a child that ignores the environment and opens a socket directly is not stopped; that needs "
        "an OS sandbox, which this product does not have on all three platforms",
        "a library already loaded with its own network stack is not intercepted",
        "DNS resolution is not blocked, so a lookup may still occur before a connection fails",
    ]


def report(*, offline: bool, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What the policy is doing, for the logs and the evidence record."""
    requirements = profile_requirements(cfg or {})
    return {
        "strict_offline": bool(offline),
        "runtime_only_no_download": True,
        "child_environment": sorted(OFFLINE_ENV) if offline else [],
        "profile_needs_network": requirements,
        "enforced": (["our own fetches go through check_url",
                      "every child is given the offline environment"] if offline else
                     ["serving itself fetches nothing; downloads elsewhere are permitted"]),
        "not_enforced": limits() if offline else [],
    }
