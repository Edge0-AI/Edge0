"""Fresh ordinary-RoPE / YaRN retrieval using an unmodified Strata executable.

Run from a configured Strata checkout, using its Python environment:
python bench/results/2026-10-09-rope-yarn-long-context/probe.py \
  --config strata.json --exe build/strata --tokens 1048576 --rope yarn --output /new/run
Each invocation starts and closes an isolated engine. No server or saved cache.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,child_env
from serve.frontend import ChatTemplate
import strata_tokenizer as ST


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--exe',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tokens',required=True,type=int)
    p.add_argument('--rope',required=True,choices=('none','yarn'))
    p.add_argument('--capacity',type=int,default=1048832)
    a=p.parse_args()
    if a.tokens+256>a.capacity:p.error('leave 256 tokens of capacity for output/internal headroom')
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());tp=Path(cfg['tokenizer'])
    vocab=json.loads((tp/'vocab.json').read_text());v=[None]*len(vocab)
    for text,i in vocab.items():v[i]=text
    tok=ST.Tokenizer(v,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()))
    template=ChatTemplate(tp/'chat_template.jinja')
    def prompt(n):
        content=('The first code is CEDAR-731.\n'+' apple'*(n//2)+
                 '\nThe middle code is MARBLE-482.\n'+' orange'*(n-n//2)+
                 '\nThe last code is QUARTZ-956.\nReturn all three codes in order separated by |. Nothing else.')
        return tok.encode(template.render([{'role':'user','content':content}],tools=None,enable_thinking=False),parse_special=True)
    lo,hi=0,a.tokens
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(prompt(mid))<=a.tokens:lo=mid
        else:hi=mid-1
    ids=prompt(lo)
    assert len(ids)==a.tokens
    token_json=json.dumps(ids)
    (a.output/'tokens.json').write_text(token_json)
    args=[];raw=cfg['args'];i=0
    replace={'--max-context','--kv','--kv-resident','--mtp','--spec','--conversation-cache-mib','--rope-scaling',
             '--rope-scale','--rope-freq-scale','--yarn-orig-ctx','--prefill','--suffix-draft'}
    while i<len(raw):
        if raw[i] in replace:i+=2
        else:args.append(raw[i]);i+=1
    args+=['--kv','fp16','--spec','2','--suffix-draft','0','--conversation-cache-mib','0',
           '--prefill','8192','--max-context',str(a.capacity),'--rope-scaling',a.rope]
    if a.rope=='yarn':args+=['--rope-scale','4','--yarn-orig-ctx','262144']
    env=child_env(cfg);env['STRATA_PREFILL_CPU_SHARE']='0'
    for key in ('STRATA_LOGPOS','STRATA_MIGRATION_LOGITS','STRATA_MIGRATION_LOGITS_FROM'):env.pop(key,None)
    result=dict(input_tokens=len(ids),input_sha256=hashlib.sha256(token_json.encode()).hexdigest(),
                rope=a.rope,capacity=a.capacity,argv=args,expected='CEDAR-731|MARBLE-482|QUARTZ-956',
                executable_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),samples_per_condition=1)
    engine=None;samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            sample={'time':time.time()}
            try:
                if engine and engine.proc:
                    for line in Path('/proc',str(engine.proc.pid),'status').read_text().splitlines():
                        if line.startswith(('VmHWM:','VmRSS:','VmSwap:')):
                            key,_,value=line.partition(':');sample[key+'_kib']=int(value.split()[0])
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
                sample['gpu_mib']=int(gpu[0]);sample['gpu_utilization']=int(gpu[1])
            except Exception as exc:sample['monitor_error']=str(exc)
            samples.append(sample);stop.wait(1)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        start=time.perf_counter()
        engine=StrataEngine(str(a.exe.resolve()),args,cwd=str(ROOT),log=str(a.output/'engine.log'),env=env)
        result['startup_s']=time.perf_counter()-start
        first=None;out=[];start=time.perf_counter()
        for token in engine.generate(ids,64,{'temperature':0},threading.Event()):
            if token is not None:
                if first is None:first=time.perf_counter()
                out.append(token)
        result.update(ttft_s=first-start if first else None,total_request_s=time.perf_counter()-start,
                      output_tokens=len(out),answer=tok.decode(out).replace('<|im_end|>',''),reused_tokens=engine.reused)
        result['correct_code_pairs']=sum(x==y for x,y in zip(result['expected'].split('|'),result['answer'].split('|')))
        result['complete']=True
    except Exception as exc:
        result.update(complete=False,error=repr(exc))
        raise
    finally:
        if engine:engine.close()
        stop.set();worker.join(timeout=5)
        result['peak_gpu_gib']=max((s.get('gpu_mib',0) for s in samples),default=0)/1024
        result['peak_engine_rss_gib']=max((s.get('VmHWM_kib',0) for s in samples),default=0)/1048576
        log=a.output/'engine.log'
        if log.exists():
            matches=re.findall(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)',log.read_text(errors='replace'))
            if matches:
                m=matches[-1];result.update(native_prompt_tokens=int(m[0]),native_reused_tokens=int(m[1]),prefill_s=float(m[3])/1000,prefill_tps=float(m[4]),decode_tps=float(m[7]))
        (a.output/'resources.json').write_text(json.dumps(samples,indent=2))
        (a.output/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='argv'}),flush=True)


if __name__=='__main__':main()
