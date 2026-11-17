"""Private-engine image/steering cache isolation gate; dry-run unless --run.

Synthetic image embeddings exercise GENI, image identity and M-RoPE without an
image encoder. Synthetic nonzero control vectors exercise actual add/project
kernels. This validates cache correctness, not vision or steering quality.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from conversation_cache_parity import STATE_KEYS, engine_args, load_tokenizer, require, state_hashes
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
import numpy as np


def fixtures(output, width):
    vector = np.sin(np.arange(width, dtype=np.float32) * .17)
    vector *= .25 / np.linalg.norm(vector)
    writer = GGUFWriter()
    writer.add('general.architecture', 'controlvector')
    writer.add('controlvector.model_hint', 'qwen4exp')
    writer.add_f32('direction.1', vector)
    writer.add_f32('direction.20', vector)
    writer.write(output / 'control.gguf')
    rows = np.sin(np.arange(4 * width, dtype=np.float32) * .013).reshape(4, width) * .2
    for name, nx, ny, values in [('image-a', 2, 2, rows), ('image-b', 2, 2, -rows),
                                 ('image-grid', 1, 4, rows)]:
        with (output / (name + '.sve')).open('xb') as f:
            f.write(struct.pack('<5i', 0x31455653, 4, nx, ny, width))
            f.write(values.astype('<f4').tobytes())


def verify(results, scenario):
    baseline, candidate = results['baseline'], results['candidate']
    expected = (['image-A', 'text-B', 'image-A-return', 'image-B', 'image-A-again',
                 'grid-changed', 'image-A-final'] if scenario == 'image' else
                ['off', 'on', 'text-B', 'off-return', 'on-return', 'off-again'])
    for records in (baseline, candidate):
        require([r['name'] for r in records] == expected, 'incomplete isolation sequence')
    for before, after in zip(baseline, candidate):
        require(len(before['ids']) == len(after['ids']) == 1, 'missing isolation output')
        require(before['finish'] in ('length', 'stop') and after['finish'] in ('length', 'stop'),
                'isolation request did not finish normally')
        require(before['ids'] == after['ids'], 'isolation output differs')
        require(set(STATE_KEYS) <= before['state'].keys() and set(STATE_KEYS) <= after['state'].keys(),
                'incomplete isolation state')
        require(before['state'] == after['state'], 'isolation main-model state differs')
    for key in ('expert_slots', 'kv', 'kv_resident', 'context', 'spec', 'mtp_max', 'lookup', 'cvec'):
        require(results['engine_info']['baseline'][key] == results['engine_info']['candidate'][key],
                f'isolation engine setting differs: {key}')
    misses, hits = (([0, 1, 3, 5], [2, 4, 6]) if scenario == 'image' else ([0, 1, 2], [3, 4, 5]))
    require(all(candidate[i]['reused'] == 0 for i in misses), 'incompatible identity reused state')
    require(all(candidate[i]['reused'] > 0 for i in hits), 'compatible parked state was not restored')
    changed = 3 if scenario == 'image' else 1
    require(baseline[0]['state'] != baseline[changed]['state'], 'fixture did not change actual model state')
    if scenario == 'image':
        require(baseline[0]['state'] != baseline[5]['state'], 'grid fixture did not change M-RoPE state')
    else:
        require(results['engine_info']['candidate']['cvec'] != 0, 'control vector not loaded')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scenario', choices=('image', 'add', 'project'), required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    # the fixtures are passed to an engine that runs in the config's cwd, not this shell's
    args.output = args.output.resolve()
    if not args.run:
        print(f'Dry run: {args.scenario} isolation; no model loaded or files created.')
        return
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    native = cfg['args'][cfg['args'].index('--native') + 1]
    geometry = GGUFFile(native).metadata
    width = geometry['qwen4exp.embedding_length']
    require(geometry['qwen4exp.block_count'] > 20, 'fixture requires layers 1 and 20')
    p = Path(cfg['tokenizer'])
    tok = load_tokenizer(p)
    tpl = ChatTemplate(p / 'chat_template.jinja')
    def encode(text):
        return tok.encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
    image_tokens = '<|vision_start|>' + '<|image_pad|>' * 4 + '<|vision_end|>'
    tail = '\n'.join(f'Record {i}: blue square and green triangle.' for i in range(64))
    A = encode('Conversation A: ' + (image_tokens if args.scenario == 'image' else '') + '\n' + tail)
    B = encode('Unrelated conversation B: name a color.\n' + tail)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    fixtures(args.output, width)
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    results = {'engine_info': {}}
    for label, budget in [('baseline', 0), ('candidate', 8192)]:
        log = args.output / (label + '.log')
        command = engine_args(cfg, budget, 1)
        if args.scenario == 'image':
            command += ['--vision']
        else:
            command += ['--control-vector', str(args.output / 'control.gguf'), '--cvec-mode', args.scenario]
        engine = StrataEngine(str(args.engine.resolve()), command, cwd=cfg.get('cwd'), log=str(log), env=env)
        records = []
        results['engine_info'][label] = dict(engine.info)
        def generate(name, ids, image=None, steering=True):
            embeddings = str(args.output / (image + '.sve')) if image else None
            out = [t for t in engine.generate(ids, 1, {'experimental_speed_projection': steering},
                                             threading.Event(), embeddings=embeddings) if t is not None]
            records.append({'name': name, 'ids': out, **engine.last})
        try:
            if args.scenario == 'image':
                generate('image-A', A, 'image-a')
                generate('text-B', B)
                generate('image-A-return', A, 'image-a')
                generate('image-B', A, 'image-b')
                generate('image-A-again', A, 'image-a')
                generate('grid-changed', A, 'image-grid')
                generate('image-A-final', A, 'image-a')
            else:
                generate('off', A, steering=False)
                generate('on', A)
                generate('text-B', B)
                generate('off-return', A, steering=False)
                generate('on-return', A)
                generate('off-again', A, steering=False)
        finally:
            engine.close()
        hashes = state_hashes(log.read_text(encoding='utf-8'))
        require(len(hashes) == len(records), 'missing isolation state hashes')
        for record, state in zip(records, hashes):
            record['state'] = state
        results[label] = records
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    verify(results, args.scenario)
    print(f'PASS: {args.scenario} identity isolation, output and main-model state parity')
    print(f'Results: {args.output / "results.json"}')


if __name__ == '__main__':
    main()
