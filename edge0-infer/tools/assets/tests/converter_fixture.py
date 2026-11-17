"""Small, deterministic MLX-format inputs for the repository's real converter."""
import json
from pathlib import Path
import struct

import numpy as np


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    types = {np.dtype('<u4'): 'U32', np.dtype('<f4'): 'F32', np.dtype('<f2'): 'F16'}
    header, offset = {}, 0
    for name, array in tensors.items():
        header[name] = {'dtype': types[array.dtype], 'shape': list(array.shape),
                        'data_offsets': [offset, offset + array.nbytes]}
        offset += array.nbytes
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with path.open('wb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for array in tensors.values():
            stream.write(array.tobytes())


def make_source(path: Path, *, vocab: int = 64, hidden: int = 64, layers: int = 40) -> None:
    path.mkdir(parents=True)
    head = 64
    config = {
        'linear_num_key_heads': 1, 'linear_num_value_heads': 2,
        'num_hidden_layers': layers, 'full_attention_interval': 4,
        'max_position_embeddings': 256, 'hidden_size': hidden,
        'moe_intermediate_size': 64, 'num_attention_heads': hidden // head,
        'num_key_value_heads': 1, 'head_dim': head, 'rms_norm_eps': 1e-6,
        'rope_parameters': {'rope_theta': 10000.0}, 'num_experts': 4,
        'shared_expert_intermediate_size': 64, 'linear_conv_kernel_dim': 4,
        'linear_key_head_dim': head, 'linear_value_head_dim': head,
        'partial_rotary_factor': 1.0, 'vocab_size': vocab,
    }
    (path / 'config.json').write_text(json.dumps({'text_config': config}))
    (path / 'tokenizer.json').write_text(json.dumps({
        'model': {'vocab': {'a': 0, 'b': 1}, 'merges': []}, 'added_tokens': [],
    }))
    (path / 'chat_template.jinja').write_text('{{ messages[0]["content"] }}')
    tensors = {}

    def quant(base: str, rows: int, width: int, *, experts=None, bits=4):
        lead = (rows,) if experts is None else (experts, rows)
        tensors[base + '.weight'] = np.full(lead + (width * bits // 32,), 0x76543210, '<u4')
        tensors[base + '.scales'] = np.full(lead + (width // 64,), 0.125, '<f4')
        tensors[base + '.biases'] = np.full(lead + (width // 64,), -0.25, '<f4')

    def raw(name: str, shape, value=1):
        tensors[name] = np.full(shape, value, '<f4')

    quant('language_model.model.embed_tokens', vocab, hidden)
    quant('language_model.lm_head', vocab, hidden)
    raw('language_model.model.norm.weight', (hidden,))
    for layer in range(layers):
        base = f'language_model.model.layers.{layer}.'
        raw(base + 'input_layernorm.weight', (hidden,))
        raw(base + 'post_attention_layernorm.weight', (hidden,))
        if (layer + 1) % 4 == 0:
            for family, rows, width in [('q_proj', 2 * hidden, hidden), ('k_proj', head, hidden),
                                         ('v_proj', head, hidden), ('o_proj', hidden, hidden)]:
                quant(base + 'self_attn.' + family, rows, width)
            raw(base + 'self_attn.q_norm.weight', (head,))
            raw(base + 'self_attn.k_norm.weight', (head,))
        else:
            for family, rows, width in [('in_proj_qkv', 4 * head, hidden),
                                         ('in_proj_z', 2 * head, hidden),
                                         ('out_proj', hidden, 2 * head),
                                         ('in_proj_b', 2, hidden), ('in_proj_a', 2, hidden)]:
                quant(base + 'linear_attn.' + family, rows, width)
            raw(base + 'linear_attn.conv1d.weight', (4 * head, 4, 1))
            raw(base + 'linear_attn.dt_bias', (2,), -2)
            raw(base + 'linear_attn.A_log', (2,), 0)
            raw(base + 'linear_attn.norm.weight', (head,))
        mlp = base + 'mlp.'
        quant(mlp + 'gate', 4, hidden, bits=8)
        quant(mlp + 'shared_expert_gate', 1, hidden, bits=8)
        for family, rows, width in [('gate_proj', 64, hidden), ('up_proj', 64, hidden),
                                     ('down_proj', hidden, 64)]:
            quant(mlp + 'switch_mlp.' + family, rows, width, experts=4)
            quant(mlp + 'shared_expert.' + family, rows, width)
    write_safetensors(path / 'model.safetensors', tensors)
    lora = {}
    # Fifty independent adapter pairs satisfy the entrypoint's existing minimum.
    for layer in range(layers):
        for family in (['gate_proj', 'up_proj'] if layer < 10 else ['gate_proj']):
            base = f'language_model.model.layers.{layer}.mlp.shared_expert.{family}'
            lora[base + '.lora_A'] = np.full((16, hidden), 0.25, '<f2')
            lora[base + '.lora_B'] = np.full((64, 16), 0.125, '<f2')
    write_safetensors(path / 'lora_edge0_35b.safetensors', lora)
