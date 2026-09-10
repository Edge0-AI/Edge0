#!/usr/bin/env python3
"""Read only selected checkpoint slices. Compare to Edge0's actual Python gate,
mlx-lm SwitchGLU (gather_qmm) and Edge0 shared MLP in explicit Float32 mode.
Requires an installed Edge0 checkout; no tokenizer/model download or full model load.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from edge0.backends.mlx._impl.bailing_hybrid import BailingGate, BailingMLP, ModelArgs
from mlx_lm.models.switch_layers import SwitchGLU

p = argparse.ArgumentParser()
p.add_argument('model', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--layer', type=int, default=1)
p.add_argument('--expert', type=int, default=0)
p.add_argument('--seed', type=int, default=20260909)
p.add_argument('--input', type=Path, help='Optional JSON array: one captured hidden-state vector')
a = p.parse_args()
config = json.loads((a.model / 'config.json').read_text())
assert config['quantization'] == {'bits': 4, 'group_size': 64, 'mode': 'affine'}
args = ModelArgs.from_dict(config)
file = a.model / 'model.safetensors'
with file.open('rb') as f:
    header_bytes = f.read(struct.unpack('<Q', f.read(8))[0])
header = json.loads(header_bytes)
base = len(header_bytes) + 8

def tensor(name, expert=None):
    h = header[name]
    dtype = {'U32': '<u4', 'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}[h['dtype']]
    shape = h['shape']
    offset = base + h['data_offsets'][0]
    if expert is not None:
        assert 0 <= expert < shape[0]
        offset += expert * (h['data_offsets'][1] - h['data_offsets'][0]) // shape[0]
        shape = shape[1:]
    # A temporary mapping of precisely this tensor/expert, copied before release.
    view = np.memmap(file, mode='r', dtype=dtype, offset=offset, shape=tuple(shape))
    data = np.array(view)
    del view
    if h['dtype'] == 'BF16':
        data = (data.astype(np.uint32) << 16).view(np.float32)
    elif h['dtype'] != 'U32':
        data = data.astype(np.float32)
    return mx.array(data)

prefix = f'model.layers.{a.layer}.mlp'
xvalues = (json.loads(a.input.read_text()) if a.input else
           np.random.default_rng(a.seed).normal(0, 0.5, args.hidden_size).astype(np.float32).tolist())
assert len(xvalues) == args.hidden_size and np.isfinite(xvalues).all()
x = mx.array(xvalues, mx.float32).reshape(1, 1, -1)
gate = BailingGate(args)
gate.weight = tensor(prefix + '.gate.weight')
if args.moe_router_enable_expert_bias:
    gate.expert_bias = tensor(prefix + '.gate.expert_bias')
indices, weights = gate(x)
mx.eval(indices, weights)
ids = np.array(indices).reshape(-1).tolist()

def linear(name, expert=None):
    return {part: tensor(name + '.' + part, expert) for part in ('weight', 'scales', 'biases')}

def qmm(v, params):
    return mx.quantized_matmul(v, params['weight'], params['scales'], params['biases'],
                               transpose=True, group_size=64, bits=4)

selected = {proj: linear(prefix + '.experts.' + proj, a.expert)
            for proj in ('up_proj', 'gate_proj', 'down_proj')}
up = qmm(x, selected['up_proj'])
g = qmm(x, selected['gate_proj'])
y = qmm(nn.silu(g) * up, selected['down_proj'])
# Only K experts are instantiated, then the actual upstream gather path executes
# them with local indices. No 128-expert resident object is constructed.
experts = SwitchGLU(args.hidden_size, args.moe_intermediate_size, len(ids))
nn.quantize(experts, group_size=64, bits=4)
for proj in ('up_proj', 'gate_proj', 'down_proj'):
    parts = [linear(prefix + '.experts.' + proj, expert) for expert in ids]
    module = getattr(experts, proj)
    for part in ('weight', 'scales', 'biases'):
        setattr(module, part, mx.stack([v[part] for v in parts]))
routed = experts(x, mx.arange(len(ids)).reshape(1, 1, -1))
shared = BailingMLP(args, args.moe_shared_expert_intermediate_size * args.num_shared_experts)
nn.quantize(shared, group_size=64, bits=4)
for proj in ('up_proj', 'gate_proj', 'down_proj'):
    for part, value in linear(prefix + '.shared_experts.' + proj).items():
        setattr(getattr(shared, proj), part, value)
output = (routed * weights[..., None]).sum(axis=-2) + shared(x)
mx.eval(up, g, y, output)
def floats(v): return np.array(v).reshape(-1).tolist()
result = dict(layer=a.layer, expert=a.expert, input=xvalues, up=floats(up), gate=floats(g),
              expertOutput=floats(y), moeOutput=floats(output), indices=ids, weights=floats(weights),
              provenance=dict(precision='float32; BF16 checkpoint coefficients promoted',
                  input_kind='captured' if a.input else 'synthetic Gaussian, real checkpoint weights',
                  seed=a.seed, header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                  model_bytes=file.stat().st_size,
                  mlx=importlib.metadata.version('mlx'), mlx_lm=importlib.metadata.version('mlx-lm'),
                  reference='Edge0 BailingGate/BailingMLP + mlx-lm SwitchGLU'))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(f'Wrote {a.output}; layer={a.layer}, expert={a.expert}, routed={ids}')
