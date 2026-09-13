#!/usr/bin/env python3
"""Fetch a small, pinned subset of public Edge0 weights with verified HTTP ranges.
Never downloads a full checkpoint. Only the declared public byte ranges are fetched.
"""
from pathlib import Path
import argparse
import hashlib, json, struct
import requests
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=Path('edge0-samples'))
OUT = parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)
REPO='Edge0/Edge0-35B-A3B-preview'
REV='1ff9f4478890faec0368c5463b621d1036d5b518'
ROOT=f'https://huggingface.co/{REPO}/resolve/{REV}/'
S=requests.Session()

def get_json(name):
    p=OUT/name
    if p.exists(): return json.loads(p.read_text())
    r=S.get(ROOT+name,timeout=30); r.raise_for_status()
    j=r.json(); p.write_text(json.dumps(j,indent=2)); return j

def byte_range(name,start,length):
    # A range-specific query prevents a proxy from replaying a different cached range.
    u=ROOT+name+f'?sc_range={start}-{start+length-1}'
    with S.get(u,headers={'Range':f'bytes={start}-{start+length-1}'},timeout=40,stream=True) as r:
        r.raise_for_status()
        expected=f'bytes {start}-{start+length-1}/'
        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(expected):
            raise RuntimeError(f'Host did not honor requested range: {r.status_code} {r.headers.get("Content-Range")}')
        data=r.raw.read(length+1)
        if len(data)!=length: raise RuntimeError(f'Incorrect range size: {len(data)} != {length}')
        return data

index=get_json('model.safetensors.index.json')
get_json('config.json')
headers={}
ledger=[]
# Fixed layers/experts selected before inspecting values. Total payload is under 20 MB.
for layer in (0,20,39):
    for expert in (0,127,255):
        for proj in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                key=f'language_model.model.layers.{layer}.mlp.switch_mlp.{proj}.{part}'
                shard=index['weight_map'][key]
                if shard not in headers:
                    hp=OUT/(shard+'.header.json')
                    if hp.exists():
                        saved=json.loads(hp.read_text()); hlen=saved['header_bytes']; header=saved['tensors']
                    else:
                        hlen=struct.unpack('<Q',byte_range(shard,0,8))[0]
                        if hlen>16*1024*1024: raise RuntimeError('Unexpected large tensor header')
                        header=json.loads(byte_range(shard,8,hlen))
                        hp.write_text(json.dumps({'header_bytes':hlen,'tensors':header},indent=2))
                    headers[shard]=(hlen,header)
                hlen,header=headers[shard]
                t=header[key]; shape=t['shape']; start,stop=t['data_offsets']
                if shape[0]!=256 or (stop-start)%256: raise RuntimeError('Unexpected expert layout')
                size=(stop-start)//256; offset=8+hlen+start+expert*size
                filename=f'L{layer:02d}_E{expert:03d}_{proj}_{part}.bin'
                p=OUT/filename
                if not p.exists(): p.write_bytes(byte_range(shard,offset,size))
                data=p.read_bytes()
                if len(data)!=size: raise RuntimeError(f'Invalid cached payload {p}')
                ledger.append({'layer':layer,'expert':expert,'projection':proj,'part':part,'key':key,'shard':shard,'range_start':offset,'bytes':size,'dtype':t['dtype'],'shape':shape[1:],'file':filename,'sha256':hashlib.sha256(data).hexdigest()})
        print(f'SAMPLED layer={layer} expert={expert}',flush=True)
        (OUT/'manifest.json').write_text(json.dumps({'repo':REPO,'revision':REV,'samples':ledger},indent=2))
print(json.dumps({'experts':9,'payload_bytes':sum(x['bytes'] for x in ledger),'manifest':str(OUT/'manifest.json')},indent=2))
