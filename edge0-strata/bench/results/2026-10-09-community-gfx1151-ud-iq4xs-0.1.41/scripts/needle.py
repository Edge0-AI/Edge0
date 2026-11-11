#!/usr/bin/env python3
"""Run the unchanged official needle probe, also preserving its exact requests/responses."""
import importlib.util
import json
import os
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('needle_bench',ROOT/'tools/needle_bench.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
original=bench.ask
out=pathlib.Path(os.environ['NEEDLE_REQUEST_DIR']);out.mkdir(parents=True,exist_ok=True)
index=0

def ask(url,key,prompt,timeout):
    global index
    index+=1
    p=out/f'probe-{index}.json'
    record={'request':{'model':'strata','max_tokens':40,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':prompt}]}}
    p.write_text(json.dumps(record))
    try:
        answer,n,seconds=original(url,key,prompt,timeout)
        record.update(answer=answer,prompt_tokens=n,seconds=seconds)
        return answer,n,seconds
    except Exception as ex:
        record['error']=f'{type(ex).__name__}: {ex}'
        raise
    finally:p.write_text(json.dumps(record))

bench.ask=ask
if __name__=='__main__':raise SystemExit(bench.main())
