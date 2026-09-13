#!/usr/bin/env python3
"""Exact real-weight smoke and paired warm/materialize benchmark.

Uses nine pinned Edge0 experts fetched by fetch_pagewarm_samples.py.
Pass --cpu-eager for CPU validation without the MLX JIT. This benchmark
measures explicit warming plus materialization, not full-model token rate.
"""
from pathlib import Path
import argparse
import hashlib, json, sys, tempfile, time
import numpy as np
from safetensors.numpy import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--samples', type=Path, required=True)
parser.add_argument('--json', type=Path, required=True)
parser.add_argument('--cpu-eager', action='store_true')
args=parser.parse_args()
args.json.parent.mkdir(parents=True, exist_ok=True)
import mlx.core as mx
if args.cpu_eager:
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
from edge0.streaming.layer import StreamingSwitchGLU
from edge0.streaming.mmap import SafetensorsMmap
from edge0.streaming.options import LayerOptions
from edge0.moe.spec import MoESpec, QuantSpec

sample=args.samples
mf=json.loads((sample/'manifest.json').read_text())
pairs=sorted({(r['layer'],r['expert']) for r in mf['samples']})
tensors={}
for proj in ('gate_proj','up_proj','down_proj'):
    for part in ('weight','scales','biases'):
        arrays=[]
        for layer,expert in pairs:
            r=next(r for r in mf['samples'] if (r['layer'],r['expert'],r['projection'],r['part'])==(layer,expert,proj,part))
            data=(sample/r['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            arrays.append(np.frombuffer(data,dtype='<u4' if part=='weight' else '<u2').reshape(r['shape']))
        tensors[f'layers.0.mlp.switch_mlp.{proj}.{part}']=np.stack(arrays)

def old_warm(lay,experts):
    for e in experts:
        for proj in ('gate_proj','up_proj','down_proj'):
            for part in ('weight','scales','biases'):
                name=f'{lay._prefix}.{proj}.{part}'
                raw=lay._shard_for(name).raw(name)
                per=raw.size//lay.num_experts
                _=raw[e*per:(e+1)*per].sum()

result={'repo':mf['repo'],'weight_revision':mf['revision'],'sample_experts':[list(p) for p in pairs],'python':sys.version.split()[0],'numpy':np.__version__,'mlx':mx.__version__,'device':str(mx.default_device()),'cpu_eager':args.cpu_eager,'weight_bytes':sum(t.nbytes for t in tensors.values())}
with tempfile.TemporaryDirectory(prefix='real-pagewarm-',dir=args.json.parent) as td:
    path=Path(td)/'experts.safetensors'; save_file(tensors,str(path))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    mm=SafetensorsMmap(str(path))
    spec=MoESpec(num_experts=len(pairs),top_k=4,intermediate_size=512,quant=QuantSpec(bits=4,group_size=64),key_template='layers.{layer}.mlp.switch_mlp',block_path='layers.{layer}.mlp.switch_mlp')
    options=LayerOptions(use_compile=False,load_threads=1,prefetch_threads=1)
    layer=StreamingSwitchGLU([mm],0,spec,options)
    # Independent snapshots of every materialized tensor before/after warming.
    before={}
    for e in range(len(pairs)):
        bundle=layer._build(e); mx.eval(*bundle.values())
        before[e]={k:np.asarray(v.view(mx.uint16) if v.dtype==mx.bfloat16 else v).copy() for k,v in bundle.items()}
    layer.warm_pages(range(len(pairs))); mm.seq_read()
    tensor_checks=0
    for e in range(len(pairs)):
        bundle=layer._build(e); mx.eval(*bundle.values())
        for k,v in bundle.items():
            current=np.asarray(v.view(mx.uint16) if v.dtype==mx.bfloat16 else v)
            assert np.array_equal(current,before[e][k]); tensor_checks+=1
    result['materialized_tensors_exact']=tensor_checks
    # Same actual quantized gather/SwiGLU path before and after warming.
    rng=np.random.default_rng(10913)
    equal_outputs=0
    for i in range(12):
        x=mx.array(rng.normal(size=(1,1,2048)).astype(np.float16))
        ids=mx.array(rng.choice(len(pairs),size=(1,4),replace=False).astype(np.int32))
        a=layer(x,ids); mx.eval(a)
        layer.warm_pages(range(len(pairs)))
        b=layer(x,ids); mx.eval(b)
        assert mx.array_equal(a,b).item(); equal_outputs+=1
    result['exact_swiglu_output_checks']=equal_outputs
    selected=[0,3,5,8]
    def work(warm):
        warm(layer,selected)
        bundles=[layer._build(e) for e in selected]
        mx.eval(*[v for b in bundles for v in b.values()])
    # Direct materialization avoids claiming any cache hit as a build gain.
    work(old_warm); work(lambda l,e:l.warm_pages(e))
    rows=[]
    for trial in range(31):
        row={}
        for side in rng.permutation(['baseline','candidate']):
            fn=old_warm if side=='baseline' else lambda l,e:l.warm_pages(e)
            start=time.perf_counter_ns()
            for _ in range(3): work(fn)
            row[str(side)]=(time.perf_counter_ns()-start)/3
        rows.append(row)
    a=np.array([r['baseline'] for r in rows]); b=np.array([r['candidate'] for r in rows]); reductions=1-b/a
    boot=np.median(reductions[rng.integers(0,len(rows),(5000,len(rows)))],axis=1)
    result['warm_and_materialize']={'pairs':31,'loops':3,'selected_experts':selected,'baseline_median_ms':float(np.median(a)/1e6),'candidate_median_ms':float(np.median(b)/1e6),'median_paired_time_reduction':float(np.median(reductions)),'bootstrap_95pct':np.quantile(boot,[0.025,0.975]).tolist(),'samples_ns':rows}
    result['file_unchanged']=hashlib.sha256(path.read_bytes()).hexdigest()==sha
    result['scope']='Actual materialization of nine published experts and 12 exact SwiGLU output comparisons with synthetic inputs. Paired latency covers explicit page warming followed by four expert builds and MLX eval. It is not full-model inference, token throughput, or an end-to-end quality benchmark.'
    layer.close(); del bundle,before,tensors
    mm.close()
out=args.json
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='warm_and_materialize'},indent=2))
print(json.dumps({k:v for k,v in result['warm_and_materialize'].items() if k!='samples_ns'},indent=2))
