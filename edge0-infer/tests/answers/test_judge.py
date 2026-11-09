"""Counterexamples for the answer judgement, built from small synthetic data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer_format as fmt
from judge import judge, judge_generation, judge_suite, tolerances

VOCAB, LAYERS, EXPERTS, K = 100, 2, 8, 2
INPUT, GENERATED = [1, 2, 3, 4], [5, 6, 7]
STEPS = [(-1, 3), (0, 1), (1, 1)]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Suite:
    def __init__(self, root: Path, backend='cpu', chunking='split-last'):
        rng = np.random.default_rng(7)
        self.root, self.backend, self.chunking = root, backend, chunking
        rows = len(INPUT) + len(GENERATED)
        self.logits = rng.normal(0, 3, (rows, VOCAB)).astype('<f4')
        # Greedy rows: each generated token is the clear argmax of its row.
        for i, token in enumerate(GENERATED):
            self.logits[len(INPUT) - 1 + i, token] = 20
        counts = dict(STEPS)
        self.probs = {key: softmax(rng.normal(0, 2, (counts[key[0]], EXPERTS))).astype('<f4') for key in self.keys()}
        self.ids = {key: np.argsort(-p, axis=-1)[:, :K] for key, p in self.probs.items()}
        self.identity = dict(
            format_version=1, suite='synthetic', assets_sha256={'base.gguf': '0' * 64}, lora=True, backend=backend,
            chunking=chunking, model_options={'routing': 'actual'}, generation={'temperature': 0},
            engine_commit='abc', engine_tracked_clean=True, exporter_sha256={'export.py': '1' * 64}, top=fmt.TOP,
            probe_seed=fmt.PROBE_SEED, probes=fmt.probe_positions(VOCAB), cases=['case'], cases_sha256='2' * 64,
            vocabulary=VOCAB, route_layers=LAYERS, route_top_k=K, route_experts=EXPERTS)
        self.write_answer()

    def keys(self):
        return [(step, layer) for step, _ in STEPS for layer in range(LAYERS)]

    @property
    def case(self):
        return self.root / 'answer/case'

    def tokens(self, **extra):
        return dict(input=INPUT, generated=GENERATED, stop_reason='length', chunking=self.chunking,
                    logits_shape=[len(INPUT) + len(GENERATED), VOCAB], **extra)

    def write_answer(self):
        self.case.mkdir(parents=True)
        (self.root / 'answer/identity.json').write_text(json.dumps(self.identity))
        (self.case / 'tokens.json').write_text(json.dumps(self.tokens(route_steps=[list(s) for s in STEPS])))
        fmt.encode_logits(self.logits, self.identity['probes']).tofile(self.case / 'logits.bin')
        (self.case / 'routes.bin').write_bytes(fmt.encode_routes(self.ids, STEPS, LAYERS))
        fmt.write_sums(self.root / 'answer')

    def ours(self, logits=None, ids=None, generated=None, **changes):
        """Write a plain run, an observed run and a generation run; return their directories."""
        logits = self.logits if logits is None else logits
        ids = self.ids if ids is None else ids
        result = []
        for name in ['ours', 'observed', 'generated']:
            root = self.root / name
            if root.exists():
                for p in sorted(root.rglob('*'), reverse=True):
                    p.unlink() if p.is_file() else p.rmdir()
            (root / 'layers').mkdir(parents=True)
            tokens = self.tokens()
            if name == 'generated' and generated is not None:
                tokens['generated'] = generated
            tokens.update(changes)
            (root / 'tokens.json').write_text(json.dumps(tokens))
            logits.astype('<f4').tofile(root / 'logits.f32')
            index = []
            for (step, layer), probs in self.probs.items():
                for kind, data, type_ in [('probs', probs, 0), ('topk', ids[(step, layer)].astype('<i4'), 26)]:
                    file = f'{len(index)}.bin'
                    data.tofile(root / 'layers' / file)
                    index.append(dict(name=f'ffn_moe_{kind}-{layer}', file=file, step=step, position=0, type=type_,
                                      shape=[data.shape[1], data.shape[0], 1, 1]))
            (root / 'layers/index.json').write_text(json.dumps(index))
            result.append(root)
        return result


@pytest.fixture
def suite(tmp_path):
    return Suite(tmp_path)


def forward(suite, **kwargs):
    ours, observed, _ = suite.ours(**kwargs)
    return judge(ours, observed, suite.case, suite.backend)


def generation(suite, **kwargs):
    *_, generated = suite.ours(**kwargs)
    return judge_generation(generated, suite.case, suite.backend)


def test_unchanged_output_passes(suite):
    report = forward(suite)
    assert report['status'] == 'passed' and not report['tie_case'], report
    assert report['logits']['max_abs'] == 0
    assert generation(suite) == dict(status='passed', tie_cases=0, matched=len(GENERATED))
    assert judge_suite([report])['status'] == 'passed'


def flip_route(suite, gap):
    """Swap the second choice of step -1, layer 0, position 0 for the third, at a given probability gap."""
    probs = suite.probs[(-1, 0)]
    order = np.argsort(-probs[0])
    probs[0, order[2]] = probs[0, order[1]] - gap
    probs[0] /= probs[0].sum()
    ids = {key: value.copy() for key, value in suite.ids.items()}
    ids[(-1, 0)][0, 1] = order[2]
    return ids


def test_non_tied_route_difference_fails(suite):
    report = forward(suite, ids=flip_route(suite, 0.05))
    assert report['status'] == 'failed'
    assert report['first_failure']['layer'] == 0 and report['first_failure']['step'] == -1
    assert report['first_failure']['worst_difference']['gap'] > 1e-4


def test_tied_route_difference_passes_once_per_suite(suite):
    report = forward(suite, ids=flip_route(suite, 0))
    assert report['status'] == 'passed' and report['tie_case'], report
    assert judge_suite([report])['status'] == 'passed'
    assert judge_suite([report, report])['status'] == 'failed'


def test_cross_kernel_rejects_excess_route_gap(suite):
    ours, observed, _ = suite.ours(ids=flip_route(suite, 5e-3))
    report = judge(ours, observed, suite.case, 'cpu', cross_kernel=True)
    assert report['status'] == 'failed'
    assert judge_suite([report, report])['status'] == 'failed'


def test_logit_difference_beyond_limit_fails(suite):
    logits = suite.logits.copy()
    top = int(np.argsort(-logits[0])[5])
    logits[0, top] += 2e-3
    assert forward(suite, logits=logits)['status'] == 'failed'
    logits[0, top] = suite.logits[0, top] + 5e-4
    assert forward(suite, logits=logits)['status'] == 'passed'


def test_cross_kernel_limits_are_opt_in():
    assert tolerances('cpu') == (1e-4, 1e-3)
    assert tolerances('cpu', False) == (1e-4, 1e-3)
    assert tolerances('metal') == (1e-3, 1e-2)
    assert tolerances('cpu', True) == tolerances('metal', True) == (1e-3, 1e-2)
    with pytest.raises(ValueError):
        tolerances('invalid', True)


def test_cross_kernel_forward_still_rejects_excess_error(suite):
    for error, verdict in [(5e-3, 'passed'), (2e-2, 'failed')]:
        logits = suite.logits.copy()
        logits[0, np.argmax(logits[0])] += error
        ours, observed, _ = suite.ours(logits=logits)
        assert judge(ours, observed, suite.case, 'cpu')['status'] == 'failed'
        assert judge(ours, observed, suite.case, 'cpu', cross_kernel=True)['status'] == verdict


def test_cross_kernel_route_tie_keeps_suite_limit(suite):
    ours, observed, _ = suite.ours(ids=flip_route(suite, 5e-4))
    assert judge(ours, observed, suite.case, 'cpu')['status'] == 'failed'
    report = judge(ours, observed, suite.case, 'cpu', cross_kernel=True)
    assert report['status'] == 'passed' and report['tie_case']
    assert judge_suite([report, report])['status'] == 'failed'


def test_cross_kernel_generation_tie_is_opt_in(suite):
    row = len(INPUT) - 1
    suite.logits[row, 9] = suite.logits[row, GENERATED[0]] - 5e-4
    fmt.encode_logits(suite.logits, suite.identity['probes']).tofile(suite.case / 'logits.bin')
    fmt.write_sums(suite.root / 'answer')
    _, _, generated = suite.ours(generated=[9, *GENERATED[1:]])
    assert judge_generation(generated, suite.case, 'cpu')['status'] == 'failed'
    report = judge_generation(generated, suite.case, 'cpu', cross_kernel=True)
    assert report['status'] == 'passed' and report['tie_cases'] == 1


def test_probe_value_beyond_limit_fails(suite):
    logits = suite.logits.copy()
    rows = fmt.encode_logits(suite.logits, suite.identity['probes'])
    probe = next(p for p in suite.identity['probes'] if p not in set(rows['tokens'][0].tolist()))
    logits[0, probe] += 2e-3
    report = forward(suite, logits=logits)
    assert report['status'] == 'failed' and report['logits']['probe_max_abs'] > 1e-3


def test_non_tied_argmax_difference_fails(suite):
    logits = suite.logits.copy()
    row = len(INPUT) - 1
    logits[row, 9] = 25
    report = forward(suite, logits=logits)
    assert report['status'] == 'failed'
    assert report['logits']['non_tied_argmax_differences'] == 1


def test_generated_token_difference(suite):
    generated = [GENERATED[0], int(np.argsort(-suite.logits[len(INPUT)])[3]), GENERATED[2]]
    report = generation(suite, generated=generated)
    assert report['status'] == 'failed' and report['position'] == 1 and report['gap'] > 1e-4


def test_generated_token_tie_passes(suite):
    row = len(INPUT)
    second = int(np.argsort(-suite.logits[row])[1])
    suite.logits[row, second] = suite.logits[row, GENERATED[1]]
    for p in (suite.root / 'answer').rglob('*'):
        if p.is_file():
            p.unlink()
    for p in sorted((suite.root / 'answer').rglob('*'), reverse=True):
        p.rmdir()
    suite.write_answer()
    report = generation(suite, generated=[GENERATED[0], second, GENERATED[2]])
    assert report['status'] == 'passed' and report['tie_cases'] == 1, report


def test_generated_token_outside_top_list_fails(suite):
    row = len(INPUT)
    outside = int(np.argsort(-suite.logits[row])[fmt.TOP + 5])
    report = generation(suite, generated=[GENERATED[0], outside, GENERATED[2]])
    assert report['status'] == 'failed' and 'outside answer top' in report['reason']


@pytest.mark.parametrize('field, value', [('backend', 'metal'), ('chunking', 'whole')])
def test_identity_backend_or_chunking_mismatch(tmp_path, field, value):
    suite = Suite(tmp_path, **{field: value})
    # The engine output is a CPU split-last run; only the answer identity differs.
    ours, observed, generated = suite.ours(chunking='split-last')
    with pytest.raises(ValueError):
        judge(ours, observed, suite.case, 'cpu')
    with pytest.raises(ValueError):
        judge_generation(generated, suite.case, 'cpu')


def test_own_chunking_must_match(suite):
    with pytest.raises(ValueError, match='chunking'):
        forward(suite, chunking='whole')


def test_different_input_is_an_error(suite):
    with pytest.raises(ValueError, match='input'):
        forward(suite, input=[1, 2, 3, 9])
    with pytest.raises(ValueError, match='input'):
        generation(suite, input=[1, 2, 3, 9])


def test_stop_reason_mismatch_is_an_error(suite):
    with pytest.raises(ValueError, match='stop_reason'):
        generation(suite, stop_reason='end_of_text')


def test_record_generation_length_and_stop_differences(suite):
    _, _, generated = suite.ours(generated=GENERATED[:1], stop_reason='end_of_text')
    with pytest.raises(ValueError):
        judge_generation(generated, suite.case, 'cpu')
    result = judge_generation(generated, suite.case, 'cpu', record_only=True)
    assert result['status'] == 'failed' and result['matched'] == 1
    assert result['ours_count'] == 1 and result['answer_count'] == 3


def test_record_generation_invalid_tokens_still_fail(suite):
    _, _, generated = suite.ours(generated=[VOCAB])
    with pytest.raises(ValueError, match='outside vocabulary'):
        judge_generation(generated, suite.case, 'cpu', record_only=True)
    _, _, generated = suite.ours(input=[99])
    with pytest.raises(ValueError, match='input'):
        judge_generation(generated, suite.case, 'cpu', record_only=True)


@pytest.mark.parametrize('name', ['case/logits.bin', 'case/routes.bin', 'case/tokens.json', 'identity.json'])
def test_tampered_file_is_an_error(suite, name):
    path = suite.root / 'answer' / name
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='hash list'):
        forward(suite)


def test_identity_requires_every_field(suite):
    fmt.validate_identity(suite.identity)
    for key in fmt.IDENTITY_FIELDS:
        broken = {k: v for k, v in suite.identity.items() if k != key}
        with pytest.raises(ValueError, match='missing'):
            fmt.validate_identity(broken)


def test_top_list_order():
    row = np.array([1, 5, 5, 3, 5, 0], dtype='<f4')
    assert fmt.top_tokens(row, 4).tolist() == [1, 2, 4, 3]


def write_catalog(model: Path, files: dict) -> None:
    lines = ['role\tbytes\tsha256\tfilename']
    for name, data in files.items():
        (model / name).write_bytes(data)
        lines.append(f'{name.split(".")[0]}\t{len(data)}\t{fmt.sha256(model / name)}\t{name}')
    (model / 'assets.tsv').write_text('\n'.join(lines) + '\n')


def test_asset_identity_covers_payload_only(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    write_catalog(model, {'base.gguf': b'weights', 'provenance.json': b'{"produced_at": "1"}'})
    identity = {'assets_sha256': fmt.catalog_digests(model)}
    assert set(identity['assets_sha256']) == {'base.gguf'}
    # Re-made assets: same payload, new provenance (and so a new catalog digest) -> same identity.
    write_catalog(model, {'base.gguf': b'weights', 'provenance.json': b'{"produced_at": "2"}'})
    fmt.check_assets(identity, model)
    # A different payload digest -> identity check fails.
    write_catalog(model, {'base.gguf': b'other weights', 'provenance.json': b'{"produced_at": "2"}'})
    with pytest.raises(ValueError, match='asset identity'):
        fmt.check_assets(identity, model)
