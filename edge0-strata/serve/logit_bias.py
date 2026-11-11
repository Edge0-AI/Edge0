"""Request-scoped token biases. Normalize before HTTP streaming starts."""
import json
import math
import re
import threading
from collections import OrderedDict


def _normalize(value, vocab_size=None):
    if value is None:
        return {}
    if isinstance(value, dict):
        entries = value.items()
        pairs = False
    elif isinstance(value, list):
        entries = value
        pairs = True
    else:
        raise ValueError("logit_bias: expected an object or a list of [token_id, bias] pairs")
    out = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("logit_bias: each entry must be [token_id, bias]")
        token, bias = entry
        if isinstance(token, bool) or not isinstance(token, (str, int)) or not re.fullmatch(r"[0-9]+", str(token)):
            raise ValueError("logit_bias: token IDs must be non-negative integers")
        token = int(token)
        if token > 2147483647 or (vocab_size is not None and token >= vocab_size):
            raise ValueError("logit_bias: token ID outside the model vocabulary")
        if token in out:
            raise ValueError("logit_bias: duplicate token ID")
        if pairs and bias is False:
            bias = -100.0
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100 or not math.isfinite(bias):
            raise ValueError("logit_bias: biases must be finite numbers between -100 and 100; false is a pair-list ban")
        out[token] = float(bias)
    if vocab_size is not None and sum(b == -100 for b in out.values()) == vocab_size:
        raise ValueError("logit_bias: cannot ban the entire vocabulary")
    return out


# A long list (a standing ban of a whole script is tens of thousands of IDs) is sent with every request and used to be
# normalized five times per request: in the OpenAI route, twice in validate_request and twice in engine_key (sampling_keys
# runs for the cache key and again for the GEN line), about 170 ms for 65,929 entries.  A list is now normalized once:
#  - the same object again in one request is answered from `_last`;
#  - an equal list in a later request is found by its JSON text, an exact key that tells False from 0 and "1" from 1.
# Only a list that normalized without an error is stored, so a bad list still raises on every call.  Short lists skip the
# cache: they cost less than the key.
_CACHE_MIN = 64
_CACHE_MAX = 4
_cache = OrderedDict()   # (vocab_size, JSON text) -> [normalized dict, engine key or None]
_last = None             # [raw value, its length, {vocab_size: entry}]: the object the previous call was given
_lock = threading.Lock()


def _entry(value, vocab_size):
    global _last
    if not isinstance(value, (dict, list)) or len(value) < _CACHE_MIN:
        return [_normalize(value, vocab_size), None]
    last = _last
    if last is not None and last[0] is value and last[1] == len(value):
        hit = last[2].get(vocab_size)
        if hit is not None:
            return hit
    else:
        last = None
    try:
        key = (vocab_size, json.dumps(value, separators=(",", ":")))
    except (TypeError, ValueError):
        key = None   # something JSON cannot write: _normalize raises for it
    entry = None
    if key is not None:
        with _lock:
            entry = _cache.get(key)
            if entry is not None:
                _cache.move_to_end(key)
    if entry is None:
        entry = [_normalize(value, vocab_size), None]
        if key is not None:
            with _lock:
                _cache[key] = entry
                while len(_cache) > _CACHE_MAX:
                    _cache.popitem(last=False)
    if key is not None:
        if last is None:
            last = [value, len(value), {}]
            _last = last
        last[2][vocab_size] = entry
    return entry


def normalize(value, vocab_size=None):
    return dict(_entry(value, vocab_size)[0])


def engine_key(value):
    entry = _entry(value, None)
    if not entry[0]:
        return ""
    if entry[1] is None:
        entry[1] = " logit_bias=" + ",".join(f"{i}:{b:g}" for i, b in sorted(entry[0].items()))
    return entry[1]


def validate_request(req, engine, vocab_size):
    """Fail closed for old engines, unsupported backends and continuous batching."""
    normalized = normalize(req.get("logit_bias"), vocab_size)
    if normalized:
        if getattr(engine, "info", {}).get("logit_bias") != 1:
            raise ValueError("logit_bias: this engine does not support token biases; use a supporting CUDA/HIP build")
        if getattr(engine, "batch", 0):
            raise ValueError("logit_bias: continuous batching is not supported; run without --batch")
    return normalized
