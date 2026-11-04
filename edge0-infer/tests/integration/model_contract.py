"""Independent fixed-profile expansion and slice contract checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def run(cli, root, *extra):
    result = subprocess.run([str(cli), 'inspect', '--model', str(root), *extra], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def expand(manifest, profile):
    result = set()
    for rule in manifest['tensors']:
        for i in [None] if rule['layers'] == 'none' else range(profile['n_layer']):
            attention = i is not None and (i + 1) % profile['full_attention_interval'] == 0
            if rule['layers'] == 'attention' and not attention:
                continue
            if rule['layers'] == 'delta' and attention:
                continue
            shape = []
            for expression in rule['shape']:
                assert re.fullmatch(r'[a-z_0-9+*() ]+', expression)
                shape.append(eval(expression, {'__builtins__': {}}, profile))
            shape += [1] * (4 - len(shape))
            result.add((rule['role'], rule['name'].replace('{i}', str(i)), rule['representation'].lower(), tuple(shape)))
    pr = manifest['prerouter']
    values = dict(profile)
    values.update(pr['profiles']['35b' if profile['n_layer'] == 40 else 'tiny'])
    if profile['n_layer'] not in (8, 40):
        values['prerouter_first_owner'] = 0
    for rule in pr['tensors']:
        for i in range(values['prerouter_first_owner'], profile['n_layer'] - 1):
            shape = [eval(expr, {'__builtins__': {}}, values) for expr in rule['shape']]
            result.add(('prerouter', rule['name'].replace('{i}', str(i)), rule['representation'].lower(), tuple(shape + [1] * (4-len(shape)))))
    return result


def verify(report, manifest, profile):
    for key, value in profile.items():
        assert report['parameters'][key] == value, (key, report['parameters'][key], value)
    actual = {(t['role'], t['name'], t['type'], tuple(t['shape'])) for t in report['tensor_requirements']}
    expected = expand(manifest, profile)
    assert actual == expected, (actual - expected, expected - actual)
    assert len(actual) == len(report['tensor_requirements'])
    attention = [i for i in range(profile['n_layer']) if (i + 1) % profile['full_attention_interval'] == 0]
    assert report['layer_groups']['attention'] == attention
    assert report['layer_groups']['deltanet'] == [i for i in range(profile['n_layer']) if i not in attention]
    assert len(report['expert_slices']) == profile['n_layer'] * profile['n_expert'] * 3
    files = {f['filename']: f['bytes'] for f in report['files']}
    groups = {}
    for s in report['expert_slices']:
        assert s['offset'] + s['length'] <= files[s['file']]
        groups.setdefault((s['layer'], s['projection']), []).append(s)
    for (layer, projection), slices in groups.items():
        slices.sort(key=lambda s: s['expert'])
        assert [s['expert'] for s in slices] == list(range(profile['n_expert']))
        for left, right in zip(slices, slices[1:]):
            assert left['offset'] + left['length'] == right['offset']
        tensor = next(t for t in report['tensor_requirements'] if t['role'] == 'base' and t['name'] == f'blk.{layer}.ffn_{projection}_exps.weight')
        product = 1
        for n in tensor['shape']:
            product *= n
        assert sum(s['length'] for s in slices) == product // 32 * 20
    print(f'profile exact; requirements={len(actual)}; slices={len(report["expert_slices"])}; layout={attention}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', type=Path)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--profile', choices=['tiny', '35b', 'custom'], required=True)
    p.add_argument('--independent-reader', action='store_true')
    p.add_argument('--digest-only', action='store_true')
    p.add_argument('--custom-tool', type=Path)
    args = p.parse_args()
    manifest = json.loads((args.root / 'tools/assets/edge0_35b/tensor_manifest.json').read_text())
    if args.profile == 'custom':
        profile = dict(n_layer=5, n_expert=7, full_attention_interval=3, n_embd=96, n_head=5, n_head_kv=3,
                       head_dim=64, rope_dim=48, n_key_head=3, n_value_head=6, state_dim=16, conv_kernel=5,
                       n_ff=128, n_shared=160, top_k=4, n_vocab=41, lora_rank=7)
        output = subprocess.check_output([args.custom_tool, '--dump'], text=True)
        actual = set()
        for line in output.splitlines():
            role, name, kind, *shape = line.split('\t')
            actual.add((role, name, kind, tuple(map(int, shape))))
        assert actual == expand(manifest, profile)
        print(f'custom independent dimensions: {len(actual)} exact requirements')
        return
    env = 'MOE_TINY' if args.profile == 'tiny' else 'MOE_ASSETS_35B'
    if not os.environ.get(env):
        print(f'not executed: {env} unset')
        sys.exit(77)
    assets = Path(os.environ[env]) / 'tiny-a' if args.profile == 'tiny' else Path(os.environ[env])
    before = {f.name: (f.stat().st_size, f.stat().st_mtime_ns, f.stat().st_ino) for f in assets.iterdir() if f.is_file()}
    if args.digest_only:
        start = time.monotonic()
        report = json.loads(run(args.cli, assets, '--json', '--digest'))
        assert report['integrity'] == 'digest' and all(f['digest_verified'] for f in report['files'])
        print(f'35B digest passed; elapsed_seconds={time.monotonic() - start:.3f}')
    else:
        report = json.loads(run(args.cli, assets, '--json'))
        assert report['integrity'] == 'size_only' and not any(f['digest_verified'] for f in report['files'])
        verify(report, manifest, manifest['profiles'][args.profile])
        if args.independent_reader:
            spec = importlib.util.spec_from_file_location('format_gguf', args.root / 'tools/assets/format_gguf.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as error:
                print(f"not executed: independent reader dependency missing: {error}")
                sys.exit(77)
            reader = module.Reader(assets / 'base.gguf')
            tensors = reader.tensors
            for s in report['expert_slices']:
                t = tensors[f'blk.{s["layer"]}.ffn_{s["projection"]}_exps.weight']
                length = t.size // manifest['profiles'][args.profile]['n_expert']
                assert s['length'] == length
                assert s['offset'] == reader.data_offset + t.offset + s['expert'] * length
            print(f'independent GGUF reader: {len(report["expert_slices"])} offsets equal')
        else:
            text = run(args.cli, assets)
            assert f'model.layers: {manifest["profiles"][args.profile]["n_layer"]}\n' in text
            if args.profile == 'tiny':
                assert 'model.layers.attention: 3 7\n' in text and 'model.experts: 16 used=4\n' in text
                digest = json.loads(run(args.cli, assets, '--json', '--digest'))
                assert digest['integrity'] == 'digest' and all(f['digest_verified'] for f in digest['files'])
            else:
                result = subprocess.run(['/usr/bin/time', '-l', str(args.cli), 'inspect', '--model', str(assets)], capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                elapsed = float(re.search(r'([\d.]+) real', result.stderr)[1])
                rss = int(re.search(r'(\d+)\s+maximum resident set size', result.stderr)[1])
                assert elapsed <= 10 and rss <= 512 * 1024**2, (elapsed, rss)
                print(f'35B default elapsed_seconds={elapsed}; max_rss_bytes={rss}')
    after = {f.name: (f.stat().st_size, f.stat().st_mtime_ns, f.stat().st_ino) for f in assets.iterdir() if f.is_file()}
    assert before == after, 'external asset identity changed during test'


if __name__ == '__main__':
    main()
