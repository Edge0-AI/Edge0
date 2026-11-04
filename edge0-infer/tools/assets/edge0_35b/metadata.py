"""GGUF metadata matching the pinned Edge0 conversion rules."""

from asset_common import read_json
from format_gguf import Value
from edge0_35b.manifest import load


def model_metadata(p: dict) -> dict:
    m = {'general.architecture': Value(8, 'qwen3next')}
    m.update({key: Value(4, p[symbol]) for symbol, key in load()['metadata_keys'].items()})
    m.update({
        'qwen3next.context_length': Value(4, 262144),
        'qwen3next.feed_forward_length': Value(4, p['n_ff']),
        'qwen3next.attention.value_length': Value(4, p['head_dim']),
        'qwen3next.attention.layer_norm_rms_epsilon': Value(6, 1e-6),
        'qwen3next.rope.freq_base': Value(6, 10000000.0),
        'qwen3next.ssm.inner_size': Value(4, p['state_dim'] * p['n_value_head']),
        'qwen3next.attention.recurrent_layers': Value(9, [(i + 1) % p['full_attention_interval'] != 0 for i in range(p['n_layer'])], 7),
    })
    return m


def lora_metadata(rank: int) -> dict:
    return {'general.architecture': Value(8, 'qwen3next'),
            'general.type': Value(8, 'adapter'), 'adapter.type': Value(8, 'lora'),
            'adapter.lora.alpha': Value(6, float(2 * rank))}


def tokenizer_metadata(source, vocab_size=248320) -> dict:
    # https://github.com/Edge0-AI/Edge0/blob/c7ee73d5dfb3729de452bdb1fec7df7aebe42806/windows/tools/repack_r3.py#L154-L186
    tokenizer = read_json(source / 'tokenizer.json')
    tokens = [None] * vocab_size
    for token, index in tokenizer['model']['vocab'].items():
        tokens[index] = token
    types = [1] * vocab_size
    for item in tokenizer['added_tokens']:
        tokens[item['id']] = item['content']
        types[item['id']] = 3
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = f'<0x{255 - i % 256:02X}>'
    return {
        'tokenizer.ggml.model': Value(8, 'gpt2'),
        'tokenizer.ggml.tokens': Value(9, tokens, 8),
        'tokenizer.ggml.merges': Value(9, [' '.join(m) for m in tokenizer['model']['merges']], 8),
        'tokenizer.ggml.scores': Value(9, [0.0] * vocab_size, 6),
        'tokenizer.ggml.token_type': Value(9, types, 5),
        'tokenizer.ggml.added_tokens_ids': Value(9, [a['id'] for a in tokenizer['added_tokens']], 4),
        'tokenizer.ggml.bos_token_id': Value(4, 248044),
        'tokenizer.ggml.eos_token_id': Value(4, 248046),
        'tokenizer.ggml.padding_token_id': Value(4, 248044),
        'tokenizer.ggml.eot_token_id': Value(4, 248046),
        'tokenizer.ggml.add_bos_token': Value(7, False),
        'tokenizer.ggml.add_eos_token': Value(7, False),
        'tokenizer.ggml.byte_fallback': Value(7, True),
        'tokenizer.ggml.pre': Value(8, 'qwen35'),
        'tokenizer.chat_template': Value(8, (source / 'chat_template.jinja').read_text(encoding='utf-8')),
    }


def prerouter_metadata(p: dict, alpha: float) -> dict:
    """p holds the base profile plus prerouter_hidden and prerouter_first_owner."""
    return {
        'general.architecture': Value(8, 'qwen3next'), 'general.type': Value(8, 'prerouter'),
        'prerouter.top_k': Value(4, p['top_k']), 'prerouter.feed_forward_length': Value(4, p['prerouter_hidden']),
        'prerouter.owners': Value(9, list(range(p['prerouter_first_owner'], p['n_layer'] - 1)), 4),
        'prerouter.lora_rank': Value(4, p['lora_rank']), 'prerouter.lora_alpha': Value(6, float(alpha)),
    }
