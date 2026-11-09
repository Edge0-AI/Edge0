"""One observation per cancellation-latency and snapshot-cost prediction."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def call(command, text=None):
    result=subprocess.run(command,input=text,capture_output=True,text=True)
    assert result.returncode==0,(command,result.stderr)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cli',required=True)
    args=parser.parse_args()
    model=os.environ['MOE_ASSETS_35B']
    prompt=(Path(__file__).resolve().parents[1]/'fixtures/bench/library-report-prompt.txt').read_text()
    common=['--context','4096','--presence-penalty','0','--model',model,'--device','metal','--max-tokens','32']
    report={}
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for label,option,value in [('prefill','--sigint-after-seconds','3'),('decode','--sigint-after-fragments','8')]:
            folder=root/label
            # The interactive command accepts a single line; preserve every word.
            text=prompt.replace('\n',' ')+'\n/quit\n'
            result=call([args.cli,'chat',*common,option,value,'--out',str(folder)],text)
            summary=json.loads((folder/'1/result.json').read_text())
            assert summary['stop_reason']=='cancelled'
            latency=re.search(r'Cancellation latency: ([0-9.e+-]+) s',result.stderr)
            assert latency,result.stderr
            report[label]={'latency_seconds':float(latency[1]),'input_count':summary['input_count'],
                           'generated_count':summary['generated_count'],'rolled_back_tokens':summary['rolled_back_tokens']}
        rendered=call([args.cli,'prompt','--model',model,'--user',prompt,'--tokens']).stdout
        tokens=[int(value) for value in rendered.splitlines()[-1].split()]
        assert len(tokens)==3084
        for enabled in (False,True):
            turn={'input':tokens}
            if enabled:
                turn['cancel']={'after_fragments':1000}
            script=root/'turns.json';script.write_text(json.dumps({'turns':[turn]}))
            folder=root/str(enabled)
            call([args.cli,'run',*common,'--temperature','0','--turns',str(script),'--out',str(folder)])
            summary=json.loads((folder/'1/result.json').read_text())
            report['with_token' if enabled else 'without_token']={
                'first_token_seconds':summary['request_summary']['first_token_seconds'],
                'block_snapshot_seconds':summary['block_snapshot_seconds']}
        report['first_token_relative_difference']=report['with_token']['first_token_seconds']/report['without_token']['first_token_seconds']-1
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
