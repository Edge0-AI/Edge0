"""Expand the shared tensor contract without evaluating arbitrary expressions."""

import ast
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name('tensor_manifest.json')
TYPES = {'F32': 0, 'F16': 1, 'Q4_1': 3}


def load():
    return json.loads(MANIFEST.read_text())


def dimension(expression: str, parameters: dict[str, int]) -> int:
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return parameters[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            a, b = visit(node.left), visit(node.right)
            return a + b if isinstance(node.op, ast.Add) else a * b
        raise ValueError(f'invalid dimension expression: {expression}')
    result = visit(ast.parse(expression, mode='eval').body)
    if type(result) is not int or result <= 0:
        raise ValueError(f'invalid dimension: {expression}')
    return result


def expand(parameters: dict[str, int], role: str) -> dict:
    result = {}
    for entry in load()['tensors']:
        if entry['role'] != role:
            continue
        layers = [None] if entry['layers'] == 'none' else range(parameters['n_layer'])
        for i in layers:
            full = i is not None and (i + 1) % parameters['full_attention_interval'] == 0
            if entry['layers'] == 'attention' and not full:
                continue
            if entry['layers'] == 'delta' and full:
                continue
            name = entry['name'].format(i=i)
            if name in result:
                raise ValueError(f'duplicate tensor: {name}')
            result[name] = {
                'shape': tuple(dimension(d, parameters) for d in entry['shape']),
                'type': TYPES[entry['representation']], 'alignment': entry['alignment'],
            }
    return result


def metadata_parameters(metadata: dict) -> dict[str, int]:
    m = load()
    p = {}
    for symbol, key in m['metadata_keys'].items():
        value = metadata[key].data
        if type(value) is not int or value <= 0:
            raise ValueError(f'invalid hyperparameter {key}')
        p[symbol] = value
    p['n_vocab'] = len(metadata['tokenizer.ggml.tokens'].data)
    return p


def expand_prerouter(parameters: dict[str, int]) -> dict:
    """Expand prerouter tensors; parameters include the prerouter profile values."""
    owners = range(parameters['prerouter_first_owner'], parameters['n_layer'] - 1)
    result = {}
    for entry in load()['prerouter']['tensors']:
        for i in owners:
            result[entry['name'].format(i=i)] = {
                'shape': tuple(dimension(d, parameters) for d in entry['shape']),
                'type': TYPES[entry['representation']], 'alignment': entry['alignment'],
            }
    return result


def prerouter_profile(name: str) -> dict[str, int]:
    return load()['prerouter']['profiles'][name]
