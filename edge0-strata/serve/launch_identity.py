"""Private proof of the configuration and arguments read by an owned server.

The owner passes a fresh secret through the child environment, then compares
this MAC with the value on the authenticated telemetry endpoint.  The digest
does not expose a reusable hash of the run config's credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import json


def attestation(key: bytes, config_bytes: bytes, argv: list[str]) -> str:
    if len(key) != 32 or not isinstance(config_bytes, bytes) or \
            not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("invalid launch attestation inputs")
    args = json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    message = b"strata-owned-launch-v1\0" + len(config_bytes).to_bytes(8, "big") + config_bytes + \
        len(args).to_bytes(8, "big") + args
    return hmac.new(key, message, hashlib.sha256).hexdigest()
