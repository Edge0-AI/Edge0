"""Real-model history reuse, physical peak and recorded conversation output."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cli',required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args()
    model=Path(os.environ.get('MOE_ASSETS_35B','/nonexistent'))
    if not (model/'assets.tsv').exists():return 77
    samples=json.loads((Path(__file__).resolve().parents[1]/'fixtures/multi_turn/conversations.json').read_text(encoding='utf-8'))
    records=[]
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for i,sample in enumerate(samples):
            turns=root/'turns.json';turns.write_text(json.dumps({'turns':[{'user':message} for message in sample['messages']]}))
            output=root/str(i)
            result=subprocess.run([args.cli,'run','--model',str(model),'--device','metal','--temperature','0','--max-tokens','128','--turns',str(turns),'--out',str(output)],capture_output=True,text=True)
            if result.returncode==77:return 77
            assert result.returncode==0,result.stderr
            previous=None;record={'name':sample['name'],'turns':[]}
            for index,message in enumerate(sample['messages'],1):
                folder=output/str(index)
                summary=json.loads((folder/'result.json').read_text())
                record['turns'].append({'user':message,'answer':(folder/'answer.txt').read_text(),'summary':summary})
                records_snapshot=records+[record]
                args.record.parent.mkdir(parents=True,exist_ok=True)
                args.record.write_text(json.dumps(records_snapshot,ensure_ascii=False,indent=2))
                assert summary['loading']['model_options']['routing']=='predicted'
                assert summary['memory_snapshot']['physical']['values']['peak_bytes']<=4*1024**3
                if previous is not None:
                    assert summary['reuse_method']=='keep' and summary['reused_tokens']==previous['e'],(sample['name'],index,summary['reuse_method'],summary['reused_tokens'],previous['e'])
                assert summary['checkpoint_saved'] and summary['checkpoint_boundary']==summary['input_count']-1
                previous=summary
            records.append(record)
            print(json.dumps({'case':sample['name'],'passed':True}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
