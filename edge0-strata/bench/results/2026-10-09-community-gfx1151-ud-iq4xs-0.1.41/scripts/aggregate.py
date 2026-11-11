#!/usr/bin/env python3
"""Derive report statistics from retained per-run data; never discard failed requests."""
import argparse
import collections
import csv
import json
import pathlib
import re
import statistics


def stats(values):
    values=[x for x in values if x is not None]
    return dict(n=len(values),median=statistics.median(values),min=min(values),max=max(values)) if values else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for p in sorted((a.data/'quality').glob('*.jsonl')):
        records.extend(json.loads(line) for line in p.read_text().splitlines())
    quality=[r for r in records if r['category']!='speed']
    report={'quality':{},'speed':{},'failures':[],'repeatability':{},'parallel':{},'needles':[], 'skipped':[]}
    for rep in sorted({r['rep'] for r in quality}):
        subset=[r for r in quality if r['rep']==rep];cats={}
        for cat in sorted({r['category'] for r in subset}):
            rows=[r for r in subset if r['category']==cat]
            cats[cat]={'passed':sum(r['passed'] is True for r in rows),'n':len(rows),'request_errors':sum(bool(r['error']) for r in rows)}
        report['quality'][str(rep)]={'categories':cats,'passed':sum(r['passed'] is True for r in subset),'n':len(subset),'client_elapsed_s':sum((r['res'] or {}).get('t_total',0) for r in subset),'prompt_tokens':sum((r['res'] or {}).get('prompt_tokens') or 0 for r in subset),'generated_tokens':sum((r['res'] or {}).get('completion_tokens') or 0 for r in subset)}
    by_rep={rep:{r['id']:r for r in quality if r['rep']==rep} for rep in [0,1]}
    common=by_rep[0].keys() & by_rep[1].keys()
    def generated(r):
        s=r['res'] or {};return {k:s.get(k) for k in ['text','reasoning','tool_calls','finish_reason']}
    report['repeatability']={'compared':len(common),'identical':sum(generated(by_rep[0][i])==generated(by_rep[1][i]) for i in common),'pass_fail_changed':[i for i in sorted(common) if by_rep[0][i]['passed']!=by_rep[1][i]['passed']]}
    for r in quality:
        if r['passed'] is not True:
            report['failures'].append({**{k:r[k] for k in ['rep','id','category','grade','error']},
                                       'finish_reason':(r['res'] or {}).get('finish_reason'),
                                       'generated_tokens':(r['res'] or {}).get('completion_tokens')})
    report['speed_request_errors']=[{k:r[k] for k in ['rep','id','error']} for r in records if r['category']=='speed' and r['error']]
    per_run=[]
    for task in sorted({r['id'] for r in records if r['category']=='speed'}):
        rows=[r for r in records if r['id']==task and not r['error']]
        values=collections.defaultdict(list)
        for r in rows:
            s=r['res'];t=s['server_timings'];row=dict(arm='default-32k',label=task,run=r['rep']+1,prompt_tokens=s['prompt_tokens'],reused=t['cache_n'],generated=s['completion_tokens'],prefill_tok_s=t['prompt_per_second'],decode_tok_s=t['predicted_per_second'],client_ttft_s=s['t_first'],client_elapsed_s=s['t_total'],drafted=t['draft_n'],accepted=t['draft_n_accepted'],finish_reason=s['finish_reason'])
            per_run.append(row)
            for k,v in row.items():
                if isinstance(v,(int,float)):values[k].append(v)
        report['speed']['default-32k/'+task]={k:stats(v) for k,v in values.items()}
    for arm in ['default-long','fast-long']:
        p=a.data/arm/'results.json'
        if not p.exists():continue
        rows=json.loads(p.read_text())
        for r in rows:
            if r['label']=='warmup':continue
            e=r['engine']; row=dict(arm=arm,label=r['label'],run=int(r['label'].rsplit('-',1)[1]),prompt_tokens=e['prompt_tokens'],reused=e.get('reused',0),generated=e['engine_generated'],prefill_tok_s=(e['prompt_tokens']-(e.get('reused') or 0))/(e['prompt_ms']/1000),decode_tok_s=e['engine_generated']/(e['decode_ms']/1000),client_ttft_s=r['client_ttft_s'],client_elapsed_s=r['client_elapsed_s'],drafted=e.get('drafts_offered'),accepted=e.get('drafts_accepted'),finish_reason=r['finish_reason'])
            per_run.append(row)
        targets=sorted({r['label'].split('-')[1] for r in rows if r['label']!='warmup'},key=int)
        for target in targets:
            subset=[r for r in per_run if r['arm']==arm and r['label'].startswith('tokens-'+target+'-')]
            report['speed'][arm+'/'+target]={k:stats([r[k] for r in subset]) for k in ['prompt_tokens','reused','generated','prefill_tok_s','decode_tok_s','client_ttft_s','client_elapsed_s']}
    for f in ['needles.json','needles-native.json']:
        p=a.data/f
        if p.exists():report['needles']+=json.loads(p.read_text())
    p=a.data/'default-262k-engine.log'
    if p.exists():
        timings=[dict(prompt_tokens=int(m[0]),reused_tokens=int(m[1]),fresh_tokens=int(m[2]),
                      prompt_ms=int(m[3]),generated_tokens=int(m[4]),decode_ms=int(m[5]))
                 for m in re.findall(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms',p.read_text())]
        report['needle_engine_timings']=timings
        completed=[r for r in report['needles'] if 'prompt_tokens' in r]
        if len(completed)==len(timings) and all(r['prompt_tokens']==t['prompt_tokens'] for r,t in zip(completed,timings)):
            for r,t in zip(completed,timings):r['engine']=t
    p=a.data/'needle.log'
    if p.exists():
        report['skipped']=[{'length':m[0],'server_max_context':int(m[1]),'source':'needle.log'}
                           for m in re.findall(r'(\w+): skipped \(the server.s context is (\d+)\)',p.read_text())]
    p=a.data/'events.jsonl'
    if p.exists():
        report['stages']=[json.loads(x) for x in p.read_text().splitlines() if json.loads(x)['event']=='result']
    p=a.data/'batch-exact.json'
    if p.exists():
        d=json.loads(p.read_text());report['parallel']['batch_exact']={'slots':len(d['solo']),'identical':[x==y for x,y in zip(d['solo'],d['batch'])],'solo_token_counts':[len(x) for x in d['solo']],'batch_token_counts':[len(x) for x in d['batch']]}
    for d in sorted(a.data.glob('parallel-qfuse*-borrow*')):
        if not d.is_dir():continue
        report['parallel'][d.name]=[json.loads(p.read_text()) for p in sorted(d.glob('result-*.json'))]
    p=a.data/'telemetry.jsonl'
    if p.exists():
        rows=[json.loads(x) for x in p.read_text().splitlines()];mem={}
        for phase in sorted({r['phase'] for r in rows}):
            ss=[r for r in rows if r['phase']==phase];mem[phase]={'samples':len(ss),'minimum_MemAvailable_bytes':min(r['MemAvailable'] for r in ss),'SwapFree_first_bytes':ss[0]['SwapFree'],'SwapFree_last_bytes':ss[-1]['SwapFree'],'gpu_max':{k:max(r['gpu'].get(k,0) for r in ss) for k in ss[0]['gpu']}}
        report['memory']=mem
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if per_run:
        with (a.out/'speed.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(per_run[0]),lineterminator='\n');w.writeheader();w.writerows(per_run)
    print(json.dumps({k:v for k,v in report.items() if k in ['quality','repeatability','parallel']},indent=2))


if __name__=='__main__':main()
