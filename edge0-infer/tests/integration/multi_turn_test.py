"""Exact multi-turn replay, accounting, and checkpoint mutation checks."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def call(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, result.stderr
    return result


def compare(folder, replay):
    original = json.loads((folder / 'tokens.json').read_text())
    blocks = json.loads((folder / 'blocks.json').read_text())['blocks']
    fresh = json.loads((replay / 'tokens.json').read_text())
    count = len(original['generated'])
    selected = [i for i, b in enumerate(blocks) if b['sample']][-count:]
    assert len(selected) == count
    width = original['logits_shape'][1] * 4
    a = (folder / 'logits.f32').read_bytes()
    all_rows = (replay / 'logits.f32').read_bytes()
    b = b''.join(all_rows[i*width:(i+1)*width] for i in selected)
    assert len(a) == len(b) == count*width
    tokens = [fresh['generated'][i] for i in selected]
    if a != b:
        offset = next(i for i in range(0, len(a), 4) if a[i:i+4] != b[i:i+4])
        print(json.dumps({'first_different_float': offset//4, 'original': struct.unpack('<f',a[offset:offset+4])[0],
                          'replay': struct.unpack('<f',b[offset:offset+4])[0]}), flush=True)
    return a == b and original['generated'] == tokens


def derive_counts(case, expectations, max_tokens=4):
    """Derive counts from token files, never from reported reuse/compute counts."""
    derived = []
    previous_e = previous_boundary = None
    for index, (method, _, _, saved) in enumerate(expectations, 1):
        folder = case / str(index)
        tokens = json.loads((folder / 'tokens.json').read_text())
        result = json.loads((folder / 'result.json').read_text())
        n, generated = len(tokens['input']), len(tokens['generated'])
        assert n > 0 and 1 <= generated <= max_tokens
        assert result['input_count'] == n
        if result['stop_reason'] == 'end_of_generation':
            assert result['tail_is_end']
            # generated_count excludes end-of-generation tokens while tokens['generated'] keeps the
            # terminal EOG, so an EOG-terminated turn reports exactly one fewer. The compute/e
            # formulas below still use the sampled length.
            assert result['generated_count'] == generated - 1, (result['generated_count'], generated)
        else:
            assert result['stop_reason'] == 'length' and generated == max_tokens
            assert not result['tail_is_end']
            assert result['generated_count'] == generated
        k = previous_e if method == 'keep' else previous_boundary if method == 'restore' else 0
        assert k is not None and 0 <= k < n
        computed = n - k + generated - 1
        assert result['e'] == n + generated - 1
        derived.append((method, k, computed, saved))
        previous_e, previous_boundary = n + generated - 1, n - 1
    return derived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu','metal','vulkan'])
    parser.add_argument('--prefill-backend', choices=['cpu', 'metal', 'vulkan'])
    parser.add_argument('--profile', default='tiny', choices=['tiny','35b'])
    parser.add_argument('--derive-counts', action='store_true', help='Derive expected counts from this platform\'s generated tokens')
    args = parser.parse_args()
    reference = Path(os.environ.get('MOE_TINY','/nonexistent'))
    answers = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers'
    model = reference/'tiny-a' if args.profile == 'tiny' else Path(os.environ.get('MOE_ASSETS_35B','/nonexistent'))
    if not (model/'assets.tsv').is_file():
        return 77
    inputs = [[1,2,3,4,5]] if args.profile == 'tiny' else [json.loads((answers / f'35b-lora-cpu-flash-split/{case}/tokens.json').read_text())['input'] for case in ['chat-intro','plain-mixed']]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for input_index, initial in enumerate(inputs):
            n = len(initial)
            scenarios = [
                ('continue',[{'input':initial},{'continue':[6,7]},{'continue':[8,9]}],
                 [('first',0,n+3,True),('keep',n+3,6,True),('keep',n+9,6,True)]),
                ('regenerate',[{'input':initial},{'same':True},{'same':True}],
                 [('first',0,n+3,True),('restore',n-1,4,False),('restore',n-1,4,False)]),
                ('checkpoint',[{'input':initial},{'prefix':n+2,'append':[6,7]}],
                 [('first',0,n+3,True),('restore',n-1,8,True)]),
            ]
            if args.profile == 'tiny':
                scenarios += [
                    ('cold',[{'input':initial},{'input':[9,8,7,6,5]}],[('first',0,n+3,True),('cold',0,8,True)]),
                    ('aligned',[{'input':initial},{'continue':list(range(10,40))},{'input':initial}],
                     [('first',0,n+3,True),('keep',n+3,34,True),('cold',0,n+3,True)]),
                    ('reset',[{'input':initial},{'reset':True},{'input':initial}],
                     [('first',0,n+3,True),('first',0,n+3,True)]),
                ]
            for route in ['actual','predicted']:
                base = [args.cli,'run','--model',str(model),'--device',args.backend,'--temperature', '0', '--presence-penalty', '0','--max-tokens','4','--model-option','routing='+route]
                if args.prefill_backend:
                    base += ['--prefill-device', args.prefill_backend]
                for name, turns, expected in scenarios:
                    case = root/f'{input_index}-{route}-{name}'
                    turns_file = root/'turns.json'; turns_file.write_text(json.dumps({'turns':turns}))
                    call(base+['--turns',str(turns_file),'--out',str(case)])
                    if args.derive_counts:
                        expected = derive_counts(case, expected)
                        print(json.dumps({'route': route, 'case': name, 'input': input_index,
                            'expected': expected, 'generated': [json.loads((case/str(i)/'tokens.json').read_text())['generated']
                            for i in range(1, len(expected)+1)]}), flush=True)
                    for index, (method,k,computed,saved) in enumerate(expected,1):
                        folder=case/str(index)
                        result=json.loads((folder/'result.json').read_text())
                        # This fixed 35B continuation ends immediately with the end-of-text EOG token.
                        # Three suffix tokens are forwarded; the sampled EOG is not forwarded.
                        early_end = args.profile == '35b' and input_index == 1 and route == 'actual' and name == 'continue' and index == 3
                        if early_end and not args.derive_counts:
                            computed = 3
                            end_id = int(call([args.cli,'tokenize','--model',str(model),'--parse-special','<|endoftext|>']).stdout.strip())
                            generated = json.loads((folder/'tokens.json').read_text())['generated']
                            assert generated == [end_id] and result['stop_reason'] == 'end_of_generation' and result['tail_is_end']
                        actual=(result['reuse_method'],result['reused_tokens'],result['computed_tokens'],result['checkpoint_saved'])
                        assert actual==(method,k,computed,saved),(name,index,actual,(method,k,computed,saved))
                        assert result['checkpoint_boundary']==result['input_count']-1
                        replay=case/f'replay-{index}'
                        call(base+['--replay',str(folder/'blocks.json'),'--out',str(replay)])
                        assert compare(folder,replay),(args.backend,route,name,index)
                    if name in ['regenerate','reset']:
                        for index in range(2,len(expected)+1):
                            assert (case/'1/logits.f32').read_bytes()==(case/str(index)/'logits.f32').read_bytes()
                            assert json.loads((case/'1/tokens.json').read_text())['generated']==json.loads((case/str(index)/'tokens.json').read_text())['generated']
                    if name=='checkpoint':
                        mutation=case/'mutation'
                        call(base+['--turns',str(turns_file),'--skip-checkpoint-restore','--out',str(mutation)])
                        replay=case/'mutation-replay'
                        call(base+['--replay',str(mutation/'2/blocks.json'),'--out',str(replay)])
                        assert not compare(mutation/'2',replay),'Mutation was not detected'
                        print(json.dumps({'backend':args.backend,'route':route,'mutation_detected':True}),flush=True)
                    print(json.dumps({'backend':args.backend,'route':route,'case':name,'input':input_index,'passed':True}),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
