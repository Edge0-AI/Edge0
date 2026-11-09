"""Hand-computed checks for row alignment, saved-value differences and expert-set comparison."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'answers'))
import answer_format as fmt
from single_turn_35b_test import differences

VOCAB = 40
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    ours, observed, suite = root / 'ours', root / 'observed', root / 'answer'
    answer = suite / 'case'
    for path in [ours, observed / 'layers', answer]:
        path.mkdir(parents=True)
    probes = fmt.probe_positions(VOCAB)
    identity = dict(format_version=1, suite='synthetic', assets_sha256={'base.gguf': '0' * 64}, lora=True, backend='cpu',
                    chunking='split-last', model_options={}, generation={}, engine_commit='abc', engine_tracked_clean=True,
                    exporter_sha256={'export.py': '1' * 64}, top=fmt.TOP, probe_seed=fmt.PROBE_SEED, probes=probes,
                    cases=['case'], cases_sha256='2' * 64, vocabulary=VOCAB, route_layers=1, route_top_k=4, route_experts=9)
    (suite / 'identity.json').write_text(json.dumps(identity))
    full = np.tile(-np.arange(VOCAB, dtype='<f4'), (4, 1))  # distinct values, token 0 first
    (answer / 'tokens.json').write_text(json.dumps(dict(input=[0, 1], generated=[0, 1], stop_reason='length',
        chunking='split-last', logits_shape=[4, VOCAB], route_steps=[[-1, 2]])))
    fmt.encode_logits(full, probes).tofile(answer / 'logits.bin')
    (answer / 'routes.bin').write_bytes(np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8).tobytes())
    fmt.write_sums(suite)
    # Our run starts at the last input row: rows 1 and 2 of the answer.
    own = full[1:3].copy()
    own[0, 0] += 1.0
    own[1, probes[-1]] += 0.5
    (ours / 'tokens.json').write_text(json.dumps(dict(input=[0, 1], generated=[0, 0], logits_shape=[2, VOCAB])))
    own.tofile(ours / 'logits.f32')
    index = []
    for name, data, type_ in [('ffn_moe_topk-0', np.array([[3, 2, 1, 0], [4, 5, 6, 8]], dtype='<i4'), 26),
                              ('ffn_moe_probs-0', np.full((2, 9), 1 / 9, dtype='<f4'), 0)]:
        data.tofile(observed / 'layers' / f'{len(index)}.bin')
        index.append(dict(name=name, file=f'{len(index)}.bin', step=-1, position=0, type=type_,
                          shape=[data.shape[1], data.shape[0], 1, 1]))
    (observed / 'layers/index.json').write_text(json.dumps(index))
    result = differences(ours, observed, answer)
    assert result['token_prefix_match'] == 1 and result['expected_tokens'] == 2
    assert result['route_positions'] == 2 and result['route_different_positions'] == 1
    assert result['route_difference_ratio'] == .5
    assert result['row_logits_max_abs'] == [1., .5], result['row_logits_max_abs']
    assert result['selected_token_logit_abs_difference'] == [1., 0.], result['selected_token_logit_abs_difference']
    assert result['numerical_threshold'] is None
print('hand-computed cross-backend metrics passed')
