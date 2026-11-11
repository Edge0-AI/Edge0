"""Ed25519 signatures for the update feed and the engine components, with no third-party package.

The packaged Control ships a pinned CPython with hash-locked wheels; adding a crypto wheel for one signature check would
widen that lock for no gain. Ed25519 (RFC 8032) is small enough to carry here: verification is what every installed PC
runs, signing is what the owner's release machine runs (`packaging/feed_tool.py`). Both are checked against the RFC's
test vectors in tests/terminal/test_update_feed.py.

Two kinds of key exist and must not be mixed up:

* the **feed/component key** (this module): an Ed25519 key the owner keeps offline. It signs the update feed index and every
  engine package manifest. Installed copies of Control carry the public half in ``update_keys.json`` and nothing else
  can add to it except the explicit, documented override ``STRATA_UPDATE_TRUSTED_KEYS`` (a file of extra public keys;
  the test suite and a local feed use it, and the CLI says so when it is in effect).
* the **app updater key** (Tauri's minisign key): signs the desktop installer. ``minisign_verify`` below checks such a
  signature so the Terminal can verify an app installer it downloads, the same way the Tauri updater does.

A public key is ``{"key_id": "<16 hex>", "alg": "ed25519", "public": "<base64 of the 32 bytes>"}``. ``key_id`` is the first 8 bytes
of SHA-256 of the public key, so a signature names the key that made it without anyone choosing the name.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path

ALG = "ed25519"
ENV_EXTRA_KEYS = "STRATA_UPDATE_TRUSTED_KEYS"
BUILTIN_KEYS = Path(__file__).with_name("update_keys.json")


class SignatureError(ValueError):
    """A signature, a key or a signed document is not what it claims to be."""


# ---------------------------------------------------------------------------------------------- RFC 8032 Ed25519

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    xx = (y * y - 1) * _inv(_D * y * y + 1) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _I % _P
    if (x * x - xx) % _P != 0:
        return None
    if x == 0 and sign:
        return None
    if x & 1 != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _add(a, b):
    a1 = (a[1] - a[0]) * (b[1] - b[0]) % _P
    b1 = (a[1] + a[0]) * (b[1] + b[0]) % _P
    c1 = 2 * a[3] * b[3] * _D % _P
    d1 = 2 * a[2] * b[2] % _P
    e, f, g, h = b1 - a1, d1 - c1, d1 + c1, b1 + a1
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(scalar: int, point):
    result = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _equal(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _compress(point) -> bytes:
    zi = _inv(point[2])
    x, y = point[0] * zi % _P, point[1] * zi % _P
    return int(y | ((x & 1) << 255)).to_bytes(32, "little")


def _decompress(data: bytes):
    if len(data) != 32:
        return None
    value = int.from_bytes(data, "little")
    y, sign = value & ((1 << 255) - 1), value >> 255
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _sha512(*parts: bytes) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def _secret_expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise SignatureError("an Ed25519 seed is 32 bytes")
    h = _sha512(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_from_seed(seed: bytes) -> bytes:
    a, _ = _secret_expand(seed)
    return _compress(_mul(a, _G))


def sign_raw(seed: bytes, message: bytes) -> bytes:
    a, prefix = _secret_expand(seed)
    public = _compress(_mul(a, _G))
    r = int.from_bytes(_sha512(prefix, message), "little") % _L
    big_r = _compress(_mul(r, _G))
    h = int.from_bytes(_sha512(big_r, public, message), "little") % _L
    s = (r + h * a) % _L
    return big_r + int(s).to_bytes(32, "little")


def verify_raw(public: bytes, message: bytes, signature: bytes) -> bool:
    """RFC 8032 verification (cofactorless; non-canonical encodings and s >= L are refused)."""
    if len(public) != 32 or len(signature) != 64:
        return False
    point_a = _decompress(public)
    point_r = _decompress(signature[:32])
    if point_a is None or point_r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(_sha512(signature[:32], public, message), "little") % _L
    return _equal(_mul(s, _G), _add(point_r, _mul(h, point_a)))


# ---------------------------------------------------------------------------------------------- keys

def key_id_of(public: bytes) -> str:
    return hashlib.sha256(public).digest()[:8].hex()


def generate_key() -> dict:
    """A new key pair as the dictionary `write_key_files` stores. The seed is the secret: it is never printed."""
    seed = secrets.token_bytes(32)
    public = public_from_seed(seed)
    return {"alg": ALG, "key_id": key_id_of(public),
            "public": base64.b64encode(public).decode("ascii"), "seed": base64.b64encode(seed).decode("ascii")}


def public_part(key: dict) -> dict:
    return {"alg": ALG, "key_id": key["key_id"], "public": key["public"]}


def _b64(value, what: str, length: int) -> bytes:
    try:
        raw = base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError):
        raise SignatureError(f"{what} is not valid base64") from None
    if len(raw) != length:
        raise SignatureError(f"{what} must be {length} bytes")
    return raw


def parse_public(entry: dict) -> tuple[str, bytes]:
    if not isinstance(entry, dict) or entry.get("alg") != ALG:
        raise SignatureError("a public key entry needs alg ed25519")
    public = _b64(entry.get("public"), "a public key", 32)
    key_id = key_id_of(public)
    if entry.get("key_id") != key_id:
        raise SignatureError(f"the key_id {entry.get('key_id')!r} does not belong to that public key")
    return key_id, public


def load_trusted_keys(extra: str | os.PathLike | None = None, *, environ=None) -> dict[str, bytes]:
    """{key_id: public key bytes}: the keys shipped in ``update_keys.json`` plus a file of extra keys when
    ``STRATA_UPDATE_TRUSTED_KEYS`` (or `extra`) names one. A malformed entry is an error, never skipped, so a typo cannot
    quietly turn into "trust nothing" or "trust something else"."""
    environ = os.environ if environ is None else environ
    keys: dict[str, bytes] = {}
    for path in (BUILTIN_KEYS, extra or environ.get(ENV_EXTRA_KEYS)):
        if not path:
            continue
        p = Path(path)
        if p == BUILTIN_KEYS and not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise SignatureError(f"the key file {p} cannot be read ({type(exc).__name__})") from None
        entries = data.get("keys") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise SignatureError(f"the key file {p} has no list of keys")
        for entry in entries:
            key_id, public = parse_public(entry)
            keys[key_id] = public
    return keys


def extra_keys_in_effect(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return bool(environ.get(ENV_EXTRA_KEYS))


# ---------------------------------------------------------------------------------------------- signing documents

def sign_bytes(key: dict, payload: bytes) -> dict:
    """``{"alg", "key_id", "value"}`` over exactly `payload`."""
    seed = _b64(key.get("seed"), "the secret seed", 32)
    if public_from_seed(seed) != base64.b64decode(key["public"]) or key_id_of(public_from_seed(seed)) != key["key_id"]:
        raise SignatureError("the secret seed does not match the public key in the key file")
    return {"alg": ALG, "key_id": key["key_id"], "value": base64.b64encode(sign_raw(seed, payload)).decode("ascii")}


def verify_bytes(trusted: dict[str, bytes], payload: bytes, signature: dict) -> str:
    """The key_id that signed `payload`; raises SignatureError when the signature is missing, from an unknown key, or bad."""
    if not isinstance(signature, dict) or signature.get("alg") != ALG:
        raise SignatureError("the signature is not an ed25519 signature")
    key_id = signature.get("key_id")
    public = trusted.get(key_id) if isinstance(key_id, str) else None
    if public is None:
        raise SignatureError(f"the signature was made by a key this PC does not trust ({key_id!r})")
    raw = _b64(signature.get("value"), "the signature value", 64)
    if not verify_raw(public, payload, raw):
        raise SignatureError("the signature does not match the signed content")
    return key_id


def install_component_verifier(trusted: dict[str, bytes]) -> None:
    """Hook the engine component's signature check (engine_component.register_signature_verifier) to `trusted`.
    A manifest then counts as verified only if one of these keys signed its canonical payload."""
    from strata_control import engine_component

    def verifier(payload: bytes, signature: dict, component) -> None:
        try:
            verify_bytes(trusted, payload, signature)
        except SignatureError as exc:
            raise engine_component.IntegrityProblem(str(exc)) from None

    engine_component.register_signature_verifier(verifier)


def ensure_component_verifier() -> bool:
    """Register the verifier with the shipped keys (and the explicit extra-keys file) unless one is registered already.
    Returns False when there are no keys to check against: signed packages are then reported unverified, never verified."""
    from strata_control import engine_component
    if engine_component._verifier is not None:
        return True
    trusted = load_trusted_keys()
    if not trusted:
        return False
    install_component_verifier(trusted)
    return True


# ---------------------------------------------------------------------------------------------- minisign (Tauri updater)

def minisign_public(text_or_b64: str) -> tuple[bytes, bytes]:
    """(key id, public key) from a minisign public key: the Tauri ``pubkey`` value (base64 of the .pub file) or the file text."""
    text = text_or_b64.strip()
    if "untrusted comment" not in text:
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except (ValueError, UnicodeError):
            raise SignatureError("the minisign public key is not valid") from None
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("untrusted comment")]
    if not lines:
        raise SignatureError("the minisign public key has no key line")
    raw = _b64(lines[0].strip(), "the minisign public key", 42)
    if raw[:2] != b"Ed":
        raise SignatureError("the minisign public key is not an Ed25519 key")
    return raw[2:10], raw[10:42]


def minisign_verify(public_text: str, signature_text: str, data_chunks) -> None:
    """Check a minisign signature (the Tauri updater's format) over `data_chunks` (an iterable of bytes).

    Legacy ``Ed`` signs the data itself, ``ED`` signs its BLAKE2b-512 digest (what Tauri produces); both are accepted.
    The signature can be the Tauri updater JSON value (base64 of the .sig file) or the file text."""
    key_id, public = minisign_public(public_text)
    text = signature_text.strip()
    if "untrusted comment" not in text:
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except (ValueError, UnicodeError):
            raise SignatureError("the app signature is not valid base64") from None
    lines = [line for line in text.splitlines() if line.strip()]
    body = [line for line in lines if not line.startswith(("untrusted comment", "trusted comment"))]
    if not body:
        raise SignatureError("the app signature has no signature line")
    raw = _b64(body[0].strip(), "the app signature", 74)
    algorithm, signer, signature = raw[:2], raw[2:10], raw[10:74]
    if signer != key_id:
        raise SignatureError("the app signature was made by a different key than the one this build trusts")
    if algorithm == b"ED":
        digest = hashlib.blake2b(digest_size=64)
        for chunk in data_chunks:
            digest.update(chunk)
        message = digest.digest()
    elif algorithm == b"Ed":
        message = b"".join(data_chunks)
    else:
        raise SignatureError("the app signature uses an unknown algorithm")
    if not verify_raw(public, message, signature):
        raise SignatureError("the app installer does not match its signature")
    # the global signature covers the trusted comment: it is checked when present so a tampered comment is refused too
    comment = next((line[len("trusted comment: "):] for line in lines if line.startswith("trusted comment: ")), None)
    if comment is not None and len(body) > 1:
        global_sig = _b64(body[1].strip(), "the app signature's global part", 64)
        if not verify_raw(public, signature + comment.encode("utf-8"), global_sig):
            raise SignatureError("the app signature's comment was changed")
