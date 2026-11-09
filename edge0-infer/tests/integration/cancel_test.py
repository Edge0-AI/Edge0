"""Exact cancellation boundaries, rollback mutations, and continuation replay."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_metric import process_peak


def call(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        raise SystemExit(77)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            try:
                summary=json.loads(line)
            except ValueError:
                continue
            if isinstance(summary,dict) and 'stop_reason' in summary:
                keys=('stop_reason','e','input_complete','cancel_requested','rolled_back_tokens',
                      'reused_tokens','computed_tokens','state_valid','memory_snapshot')
                print(json.dumps({'failed_run_turn':{key:summary[key] for key in keys}}),flush=True)
    assert result.returncode == 0, (command, result.stderr)
    return result


def compare(folder, replay):
    original = json.loads((folder/'tokens.json').read_text())
    blocks = json.loads((folder/'blocks.json').read_text())['blocks']
    fresh = json.loads((replay/'tokens.json').read_text())
    count = len(original['generated'])
    selected = [i for i, block in enumerate(blocks) if block['sample']][-count:] if count else []
    assert len(selected) == count
    width = original['logits_shape'][1]*4
    a = (folder/'logits.f32').read_bytes()
    rows = (replay/'logits.f32').read_bytes()
    b = b''.join(rows[i*width:(i+1)*width] for i in selected)
    assert len(a) == len(b) == count*width
    if a != b:
        offset = next(i for i in range(0,len(a),4) if a[i:i+4] != b[i:i+4])
        print(json.dumps({'first_different_float': offset//4,
                          'original': struct.unpack('<f',a[offset:offset+4])[0],
                          'replay': struct.unpack('<f',b[offset:offset+4])[0]}), flush=True)
    return a == b and original['generated'] == [fresh['generated'][i] for i in selected]


def expected(stop, e, complete, rolled, method, k, computed, saved, checkpoint):
    return dict(zip(('stop_reason','e','input_complete','rolled_back_tokens','reuse_method',
                     'reused_tokens','computed_tokens','checkpoint_saved','checkpoint_boundary'),
                    (stop,e,complete,rolled,method,k,computed,saved,checkpoint)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu','metal'])
    parser.add_argument('--profile', default='tiny', choices=['tiny','35b'])
    parser.add_argument('--prompt-file', type=Path, default=Path(__file__).resolve().parents[1]/'fixtures/bench/library-report-prompt.txt')
    args = parser.parse_args()
    reference = Path(os.environ.get('MOE_TINY','/nonexistent'))
    model = reference/'tiny-a' if args.profile == 'tiny' else Path(os.environ.get('MOE_ASSETS_35B','/nonexistent'))
    if not (model/'assets.tsv').is_file():
        return 77
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        base=[args.cli,'run','--model',str(model),'--device',args.backend,'--temperature', '0', '--presence-penalty', '0','--max-tokens','4']
        normal=expected('length',14,True,0,'first',0,14,True,10)
        x=list(range(1,12))
        scenarios=[
            ('before_compute',[{'input':x},{'input':[30]*11,'cancel':{'at':'before_compute'}},{'input':[30]*11}],
             [normal,expected('cancelled',14,False,0,'cold',0,0,False,10),expected('length',14,True,0,'cold',0,14,True,10)]),
            ('after_reuse',[{'input':x},{'same':True,'cancel':{'at':'after_reuse'}},{'same':True}],
             [normal,expected('cancelled',10,False,0,'restore',10,0,False,10),expected('length',14,True,0,'keep',10,4,False,10)]),
            ('cold',[{'input':x},{'input':[30]*11,'cancel':{'at':'after_reuse'}},{'same':True}],
             [normal,expected('cancelled',0,False,0,'cold',0,0,False,None),normal]),
            ('between_blocks',[{'input':x,'cancel':{'at':'before_block','block':2}},{'same':True}],
             [expected('cancelled',4,False,0,'first',0,4,False,None),expected('length',14,True,0,'keep',4,10,True,10)]),
            ('before_save',[{'input':x,'cancel':{'at':'before_block','block':5}},{'same':True}],
             [expected('cancelled',10,False,0,'first',0,10,False,None),expected('length',14,True,0,'keep',10,4,True,10)]),
            ('in_block',[{'input':x,'cancel':{'at':'in_block','block':1,'segment':3}},{'same':True}],
             [expected('cancelled',2,False,2,'first',0,2,False,None),expected('length',14,True,0,'keep',2,12,True,10)]),
            ('before_step',[{'input':x,'cancel':{'at':'before_step','token':2}},{'continue':[21]}],
             [expected('cancelled',12,True,0,'first',0,12,True,10),expected('length',17,True,0,'keep',12,5,True,13)]),
            ('changed_content',[{'input':x},{'continue':[30]*8,'cancel':{'at':'before_block','block':2}},{'prefix':15,'append':[40,41]}],
             [normal,expected('cancelled',18,False,0,'keep',14,4,False,10),expected('length',20,True,0,'restore',10,10,True,16)]),
        ]
        routes=['actual','predicted']
        if args.profile=='35b':
            prompt=args.prompt_file
            # Use the same chat rendering as the benchmark input.
            tokens=call([args.cli,'prompt','--model',str(model),'--user',prompt.read_text(),'--tokens']).stdout
            x=[int(value) for value in tokens.splitlines()[-1].split()]
            rendered=tokens.rsplit('\n', 2)[0]
            encoded=call([args.cli,'tokenize','--model',str(model),'--parse-special',rendered]).stdout
            assert x == [int(value) for value in encoded.split()], 'Rendered prompt token count/content differs'
            n=len(x)
            assert n > 1
            print(json.dumps({'prompt_file': prompt.name, 'input_count': n, 'cancel_segment': 20}), flush=True)
            base[-1]='12'
            routes=[None]
            scenarios=[
                ('in_block',[{'input':x,'cancel':{'at':'in_block','block':0,'segment':20}},{'same':True}],
                 [expected('cancelled',0,False,n-1,'first',0,0,False,None),expected('length',n+11,True,0,'first',0,n+11,True,n-1)]),
                ('before_step',[{'input':x,'cancel':{'at':'before_step','token':8}},{'continue':[21]}],
                 [expected('cancelled',n+7,True,0,'first',0,n+7,True,n-1),expected('length',n+20,True,0,'keep',n+7,13,True,n+8)]),
            ]
        failures=[]
        for route in routes:
            configured=base+(['--prefill-chunk','2','--model-option','routing='+route] if route else [])
            for name,turns,expectations in scenarios:
                def run(suffix, extra=(), mutation=False):
                    folder=root/f'{route}-{name}-{suffix}'
                    script=root/'turns.json';script.write_text(json.dumps({'turns':turns}))
                    call(configured+list(extra)+['--turns',str(script),'--out',str(folder)])
                    for index,wanted in enumerate(expectations,1):
                        turn=folder/str(index)
                        result=json.loads((turn/'result.json').read_text())
                        actual={key:result[key] for key in wanted}
                        assert actual==wanted,(name,index,actual,wanted)
                        assert result['state_valid'] and result['cancel_requested']==(wanted['stop_reason']=='cancelled')
                        blocks=json.loads((turn/'blocks.json').read_text())['blocks']
                        assert sum(b['length'] for b in blocks)==wanted['e']
                        if turns[index-1].get('cancel'):
                            point=turns[index-1]['cancel']
                            assert all(result['cancel_point'][key]==value for key,value in point.items())
                        if args.profile=='35b':
                            metric, source, peak = process_peak(result['memory_snapshot'])
                            print(json.dumps({'memory_metric': metric, 'source': source, 'peak_bytes': peak}),flush=True)
                            assert peak<=4*1024**3
                        replay=folder/f'replay-{index}'
                        call(configured+['--replay',str(turn/'blocks.json'),'--out',str(replay)])
                        same=compare(turn,replay)
                        if mutation and index==len(expectations):
                            assert not same,'Mutation was not detected'
                        else:
                            assert same,(route,name,index,'replay differs')
                    print(json.dumps({'backend':args.backend,'route':route,'case':name,'variant':suffix,'passed':True}),flush=True)
                    return folder
                try:
                    run('normal')
                    if name=='in_block' and args.profile=='tiny':
                        run('mutation',['--skip-block-rollback'],True)
                        ahead_options=['--model-option','prefill_read=ahead','--model-option','prefill_ahead_min_tokens=1']
                        ahead=run('ahead',ahead_options)
                        before=root/f'{route}-ahead-before'
                        script=root/'before.json'
                        script.write_text(json.dumps({'turns':[{'input':x,'cancel':{'at':'before_block','block':1}}]}))
                        call(configured+ahead_options+['--turns',str(script),'--out',str(before)])
                        a=json.loads((ahead/'1/result.json').read_text())
                        b=json.loads((before/'1/result.json').read_text())
                        assert a['budget_current']==b['budget_current'],('ahead ledger',a['budget_current'],b['budget_current'])
                        print(json.dumps({'route':route,'ahead_ledger_restored':a['budget_current']}),flush=True)
                except AssertionError as error:
                    failure={'route':route,'case':name,'error':str(error)[-800:]}
                    failures.append(failure)
                    print(json.dumps(failure),flush=True)
        assert not failures,[(failure['route'],failure['case']) for failure in failures]
    return 0


if __name__=='__main__':
    raise SystemExit(main())
