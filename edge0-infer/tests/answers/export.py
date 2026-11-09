#!/usr/bin/env python3
"""Export one answer suite with this repository's reference CLI.

For each fixed case the CLI tokenizes the case text, generates greedily with
`run --tokens`, then computes the input plus the generated tokens twice with
`forward` (plain, and with `--observe` for routing). The suite is written to a
temporary directory next to the target and replaces the target only after
every case succeeded. Exit code 77 means the requested device is unavailable.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_format as fmt
from judge import observed_routes

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'tests/fixtures/answers/cases.json'
SUITE = re.compile(r'^(tiny|35b)-(base|lora)-(cpu|metal)-flash(-split)?$')
GENERATION = {'temperature': 0, 'presence_penalty': 0}


class Unavailable(Exception):
    """The requested device is not available (CLI exit code 77)."""


def call(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        raise Unavailable(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f'command failed ({result.returncode}): {command[1]}\n{result.stdout}{result.stderr}')
    return result


def parse_suite(name: str) -> dict:
    match = SUITE.match(name)
    if not match:
        raise ValueError(f'invalid suite name: {name}')
    return dict(family=match.group(1), lora=match.group(2) == 'lora', backend=match.group(3),
                chunking='split-last' if match.group(4) else 'whole')


def git(*args: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def engine_identity() -> dict:
    commit = git('rev-parse', 'HEAD')
    clean = git('status', '--porcelain', '--untracked-files=no') == ''
    return dict(engine_commit=commit, engine_tracked_clean=clean)


def exporter_digests() -> dict:
    here = Path(__file__).resolve().parent
    return {name: fmt.sha256(here / name) for name in ('export.py', 'answer_format.py', 'judge.py')}


def model_options(lora: bool) -> list[str]:
    options = ['--model-option', 'routing=actual']
    if not lora:
        options += ['--model-option', 'lora=off']
    return options


def export_case(cli: str, model: Path, spec: dict, case: dict, work: Path) -> dict:
    """Run one case and return its tokens, full logits and routing observation."""
    common = ['--model', str(model), '--device', spec['backend'], *model_options(spec['lora'])]
    tokenize = [cli, 'tokenize', '--model', str(model)]
    if case['parse_special']:
        tokenize.append('--parse-special')
    prompt = [int(t) for t in call(tokenize + [case['text']]).stdout.split()]
    if not prompt:
        raise ValueError(f'empty input for case {case["id"]}')
    (work / 'input.json').write_text(json.dumps({'input': prompt}))
    run = work / 'run'
    call([cli, 'run', *common, '--tokens', str(work / 'input.json'), '--temperature', '0', '--presence-penalty', '0',
          '--max-tokens', str(case['decode_steps']), '--out', str(run)])
    generated_record = fmt.read_json(run / 'tokens.json')
    if generated_record['input'] != prompt:
        raise ValueError(f'run changed the input of case {case["id"]}')
    generated = generated_record['generated']
    (work / 'sequence.json').write_text(json.dumps({'input': prompt, 'generated': generated}))
    outputs = {}
    for name, extra in [('plain', []), ('observed', ['--observe'])]:
        command = [cli, 'forward', *common, '--tokens', str(work / 'sequence.json'), '--out', str(work / name), *extra]
        if spec['chunking'] == 'split-last':
            command.append('--split-last')
        call(command)
        outputs[name] = work / name
    shape = fmt.read_json(outputs['plain'] / 'tokens.json')['logits_shape']
    plain = np.fromfile(outputs['plain'] / 'logits.f32', dtype='<f4')
    if plain.size != shape[0] * shape[1] or shape[0] != len(prompt) + len(generated) or not np.isfinite(plain).all():
        raise ValueError(f'invalid forward logits for case {case["id"]}')
    observation_equal = (outputs['plain'] / 'logits.f32').read_bytes() == (outputs['observed'] / 'logits.f32').read_bytes()
    if spec['backend'] == 'cpu' and not observation_equal:
        raise ValueError(f'observation changed CPU logits for case {case["id"]}')
    routes = observed_routes(outputs['observed'])
    layers = sorted({layer for _, layer in routes})
    steps = sorted({step for step, _ in routes})
    if layers != list(range(len(layers))) or not steps or steps[0] != -1:
        raise ValueError(f'incomplete routing observation for case {case["id"]}')
    counts = []
    for step in steps:
        sizes = {routes[(step, layer)][0].shape for layer in layers}
        if len(sizes) != 1:
            raise ValueError(f'routing shapes differ within step {step}')
        counts.append([step, sizes.pop()[0]])
    ids = {key: value[0] for key, value in routes.items()}
    experts = {value[1].shape[1] for value in routes.values()}
    top_k = {value.shape[1] for value in ids.values()}
    if len(experts) != 1 or len(top_k) != 1:
        raise ValueError('inconsistent routing dimensions')
    tokens = dict(input=prompt, generated=generated, stop_reason=generated_record['stop_reason'],
                  chunking=spec['chunking'], logits_shape=shape, route_steps=counts)
    dimensions = dict(vocabulary=shape[1], route_layers=len(layers), route_top_k=top_k.pop(), route_experts=experts.pop())
    return dict(tokens=tokens, logits=plain.reshape(shape), ids=ids, dimensions=dimensions,
                observation_bitwise_equal=observation_equal)


def write_case(target: Path, result: dict, probes: list[int]) -> None:
    target.mkdir(parents=True)
    tokens = result['tokens']
    (target / 'tokens.json').write_text(json.dumps(tokens, sort_keys=True) + '\n')
    fmt.encode_logits(result['logits'], probes).tofile(target / 'logits.bin')
    layers = result['dimensions']['route_layers']
    (target / 'routes.bin').write_bytes(fmt.encode_routes(result['ids'], tokens['route_steps'], layers))


def export(cli: str, model: Path, suite: str, out: Path, cases_file: Path = CASES) -> dict:
    """Export `suite` into `out` (replaced only on success); return the run record."""
    spec = parse_suite(suite)
    model = Path(model)
    if not (model / 'assets.tsv').is_file():
        raise FileNotFoundError(f'missing assets: {model}')
    cases = fmt.read_json(cases_file)['cases']
    if [c['id'] for c in cases] != list(fmt.CASES):
        raise ValueError('cases file does not list the fixed cases in order')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(format_version=fmt.FORMAT_VERSION, suite=suite, assets_sha256=fmt.catalog_digests(model),
                    lora=spec['lora'], backend=spec['backend'], chunking=spec['chunking'],
                    model_options={'routing': 'actual', **({} if spec['lora'] else {'lora': 'off'})},
                    generation=GENERATION, **engine_identity(), exporter_sha256=exporter_digests(), top=fmt.TOP,
                    probe_seed=fmt.PROBE_SEED, cases=[c['id'] for c in cases], cases_sha256=fmt.sha256(cases_file))
    record = dict(exported_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), cases={})
    staging = Path(tempfile.mkdtemp(prefix=f'.{out.name}.', dir=out.parent))
    try:
        with tempfile.TemporaryDirectory() as scratch:
            for case in cases:
                started = time.monotonic()
                work = Path(scratch) / case['id']
                work.mkdir()
                result = export_case(cli, model, spec, case, work)
                if 'probes' not in identity:
                    # Probe positions depend only on the seed and the vocabulary size.
                    identity.update(result['dimensions'], probes=fmt.probe_positions(result['dimensions']['vocabulary']))
                elif any(identity[k] != v for k, v in result['dimensions'].items()):
                    raise ValueError('model dimensions differ between cases')
                write_case(staging / case['id'], result, identity['probes'])
                record['cases'][case['id']] = dict(seconds=round(time.monotonic() - started, 1),
                                                   observation_bitwise_equal=result['observation_bitwise_equal'])
                shutil.rmtree(work)
        fmt.validate_identity(identity)
        (staging / 'identity.json').write_text(json.dumps(identity, indent=1, sort_keys=True) + '\n')
        fmt.write_sums(staging)
        (staging / 'run.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
        fmt.suite_identity(staging)
        replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def replace(staging: Path, out: Path) -> None:
    previous = None
    if out.exists():
        previous = Path(tempfile.mkdtemp(prefix=f'.{out.name}.old.', dir=out.parent))
        os.rename(out, previous / out.name)
    os.rename(staging, out)
    if previous is not None:
        shutil.rmtree(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli', required=True, help='edge0-infer executable')
    parser.add_argument('--model', type=Path, required=True, help='asset directory')
    parser.add_argument('--suite', required=True, help='suite name, for example tiny-lora-cpu-flash-split')
    parser.add_argument('--out', type=Path, required=True, help='suite directory to create or replace')
    parser.add_argument('--cases', type=Path, default=CASES)
    args = parser.parse_args()
    try:
        record = export(args.cli, args.model, args.suite, args.out, args.cases)
    except Unavailable as exc:
        print(f'SKIP: {exc}')
        return 77
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
