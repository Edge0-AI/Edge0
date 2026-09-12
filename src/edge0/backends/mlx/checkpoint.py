"""Lossless MLX safetensors codec. No dynamic classes or executable payloads."""
import mlx.core as mx
from mlx_lm.models.cache import KVCache, ArraysCache


def write(engine, put, phase='ready'):
    family_state = engine._checkpoint_family_state()
    def nbytes(value):
        if isinstance(value, mx.array):
            return value.nbytes
        if isinstance(value, dict):
            return sum(nbytes(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(nbytes(v) for v in value)
        return 0
    estimate = sum(c.nbytes for c in engine.cache) + nbytes(family_state) + nbytes(engine.next_logits())
    if estimate > engine.conversation_cache.config.budget_bytes:
        from edge0.conversation.store import _Oversized
        raise _Oversized('checkpoint exceeds disk budget')
    tensors = {}
    def encode(value):
        if isinstance(value, mx.array):
            key = str(len(tensors))
            tensors[key] = value
            return {'tensor': key}
        if isinstance(value, (list, tuple)):
            return {'list': [encode(v) for v in value]}
        if isinstance(value, dict):
            return {'dict': [[encode(k), encode(v)] for k, v in value.items()]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'unsupported checkpoint value: {type(value)}')
    layers = []
    for cache in engine.cache:
        if type(cache) is KVCache:
            blocks = []
            keys, values = cache.state
            for start in range(0, cache.offset, engine.conversation_cache.config.interval):
                end = min(start + engine.conversation_cache.config.interval, cache.offset)
                data = {'keys': keys[..., start:end, :], 'values': values[..., start:end, :]}
                blocks.append(put(lambda path, data=data: mx.save_safetensors(str(path), data)))
            layers.append({'kind': 'kv', 'offset': cache.offset, 'blocks': blocks})
        elif type(cache) is ArraysCache:
            layers.append({'kind': 'arrays', 'state': encode(cache.state),
                           'left_padding': encode(cache.left_padding), 'lengths': encode(cache.lengths)})
        else:
            raise ValueError(f'unsupported cache class {type(cache)}')
    state = encode(family_state)
    logits = encode(engine.next_logits())
    snapshot = put(lambda path: mx.save_safetensors(str(path), tensors))
    return dict(layers=layers, state=state, logits=logits, snapshot=snapshot,
                pos=engine.pos, phase=phase)


def read(manifest, paths):
    tensors = mx.load(str(paths[manifest['snapshot']]))
    def decode(value):
        if isinstance(value, dict):
            if 'tensor' in value:
                return tensors[value['tensor']]
            if 'list' in value:
                return [decode(v) for v in value['list']]
            if 'dict' in value:
                return {decode(k): decode(v) for k, v in value['dict']}
            raise ValueError('invalid state')
        return value
    caches = []
    for layer in manifest['layers']:
        if layer['kind'] == 'kv':
            blocks = [mx.load(str(paths[k])) for k in layer['blocks']]
            cache = KVCache()
            cache.state = tuple(mx.concatenate([b[k] for b in blocks], axis=2) for k in ('keys', 'values'))
            if cache.offset != layer['offset'] or cache.offset != manifest['pos']:
                raise ValueError('invalid KV offset')
        elif layer['kind'] == 'arrays':
            state = decode(layer['state'])
            cache = ArraysCache(len(state))
            cache.state = state
            cache.left_padding = decode(layer['left_padding'])
            cache.lengths = decode(layer['lengths'])
        else:
            raise ValueError('unknown layer cache')
        caches.append(cache)
    logits = decode(manifest['logits'])
    state = decode(manifest['state'])
    mx.eval(logits, *[c.state for c in caches])
    return caches, logits, state, manifest['pos'], manifest['phase']
