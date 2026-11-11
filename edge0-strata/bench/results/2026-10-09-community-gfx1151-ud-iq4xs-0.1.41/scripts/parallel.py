#!/usr/bin/env python3
"""Two simultaneous, distinct ~19.6K prompts for issue #1356 (not its unpublished original prompts).

Reuse the public V100 report's synthetic-Python filler/template and sampling.
Save exact requests, partial SSE, usage, elapsed time and server metrics.
"""
import argparse
import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate, openai_to_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=pathlib.Path, required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d=a.pack/'tokenizer'; vocab=json.loads((d/'vocab.json').read_text())
    tokens=[None]*len(vocab)
    for token, number in vocab.items(): tokens[number]=token
    tok=Tokenizer(tokens,(d/'merges.txt').read_text().splitlines(),json.loads((d/'token_type.json').read_text()))
    template=ChatTemplate(d/'chat_template.jinja')
    filler='\n'.join(f'def task_{i:05d}(value: int) -> int: return (value * {(i%97)+1} + {i}) % 100003' for i in range(12000))
    ending='\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.'
    def payload(text):
        return dict(model='strata',messages=[dict(role='user',content=text)],temperature=0,
                    reasoning_effort='none',max_tokens=256,stream=True,stream_options=dict(include_usage=True))
    requests=[]
    for n in range(2):
        prefix=f'Benchmark nonce: parallel-1356-trial-{n}.\nReview this synthetic Python module:\n'
        lo,hi=0,len(filler)
        def count(body):
            m,t,k=openai_to_messages(body);return len(tok.encode(template.render(m,t,**k),parse_special=True))
        while lo<hi:
            mid=(lo+hi+1)//2
            if count(payload(prefix+filler[:mid]+ending))<=19600:lo=mid
            else:hi=mid-1
        body=payload(prefix+filler[:lo]+ending); requests.append(body)
        (a.out/f'request-{n}.json').write_text(json.dumps(body))
        (a.out/f'count-{n}.json').write_text(json.dumps({'prompt_tokens':count(body)}))
    import threading
    barrier=threading.Barrier(2)
    def ask(n):
        chunks=[];row={'slot':n};barrier.wait();start=time.monotonic()
        try:
            req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(requests[n]).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as response, (a.out/f'sse-{n}.jsonl').open('w') as f:
                for line in response:
                    if not line.startswith(b'data:'):continue
                    data=line[5:].strip()
                    if data==b'[DONE]':break
                    chunk=json.loads(data);chunks.append(chunk);f.write(json.dumps(chunk)+'\n');f.flush()
                    if 'first_token_s' not in row and any(c.get('delta',{}).get('content') for c in chunk.get('choices',[])):
                        row['first_token_s']=time.monotonic()-start
            row['text']=''.join(c.get('delta',{}).get('content') or '' for ch in chunks for c in ch.get('choices',[]))
            row['usage']=next((ch['usage'] for ch in reversed(chunks) if ch.get('usage')),None)
            row['finish_reason']=next((c['finish_reason'] for ch in reversed(chunks) for c in ch.get('choices',[]) if c.get('finish_reason')),None)
        except Exception as ex:
            row['error']=f'{type(ex).__name__}: {ex}'
        row['elapsed_s']=time.monotonic()-start
        (a.out/f'result-{n}.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: rows=list(ex.map(ask,range(2)))
    with urllib.request.urlopen(a.url+'/metrics',timeout=10) as response:metrics=json.load(response)
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return int(any(r.get('error') or not r.get('text') for r in rows))


if __name__=='__main__':sys.exit(main())
