import json
import re
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = OUT / 'p4-roofline'
base = json.loads((DATA / 'results.json').read_text())
control = json.loads((DATA / 'control/results.json').read_text())
auto = json.loads((DATA / 'auto/results.json').read_text())
roofs = [json.loads(l) for l in (DATA / 'roofs.jsonl').read_text().splitlines()]
host = [json.loads(l) for l in (DATA / 'host-roofs.jsonl').read_text().splitlines()]
assert base['complete'] and control['complete'] and auto['complete']
bandwidth = {k: median([r['GB_s'] for r in roofs if r['kind'] == k]) for k in ['vram_read', 'h2d', 'd2h']}
bandwidth['host_read'] = median(r['GB_s'] for r in host)

def phase_data(row):
    line = next(l for l in row['log'].splitlines() if 'GPU timeline' in l)
    total = float(re.search(r'GPU timeline (\d+) ms', line)[1])
    parts = re.findall(r'([\w +/]+) (\d+) \([\d.]+%\)', line.split('ms:', 1)[1])
    phases = {k.strip(): float(v) for k, v in parts}
    io = re.search(r'OS read ([\d.]+) MB from storage \((\d+) major faults\) for ([\d.]+) MB of expert reads', row['log'])
    return dict(total_ms=total, phases_ms=phases, disk_MB=float(io[1]) if io else None,
                major_faults=int(io[2]) if io else None, logical_expert_MB=float(io[3]) if io else None)

summary = {'bandwidth_GB_s': bandwidth, 'baseline': [], 'control': [], 'notes': []}
for n in [512,4096,8192]:
    rows = [r for r in base['runs'] if r['phase'] == 'measured' and r['tokens'] == n]
    assert len(rows) == 3
    rates = [n*1000/r['timing']['prompt_ms'] for r in rows]
    data = [phase_data(r) for r in rows]
    phases = {k: median(d['phases_ms'][k] for d in data) for k in data[0]['phases_ms']}
    total = median(d['total_ms'] for d in data)
    # This scenario leaves every non-expert phase at its measured runtime.
    expert_names = ['host grouping','gather','wait copy','dequant','gemm gate/up','gemm down','combine']
    expert = sum(phases.get(k,0) for k in expert_names)
    wait = phases.get('wait copy',0)
    stream_group = wait + phases.get('dequant',0)
    summary['baseline'].append(dict(tokens=n,rates=rates,median_tps=median(rates),min_tps=min(rates),max_tps=max(rates),
        prompt_ms_median=median(r['timing']['prompt_ms'] for r in rows),phases_ms=phases,total_ms=total,
        logical_expert_MB_median=median(d['logical_expert_MB'] for d in data),disk_MB=[d['disk_MB'] for d in data],
        major_faults=[d['major_faults'] for d in data],
        wait_only_zero_speedup=total/(total-wait),
        wait_and_dequant_zero_speedup=total/(total-stream_group),
        expert_phase_zero_speedup=total/(total-expert),
        fixed_nonexpert_ms=total-expert))

# Routed experts only: 48 layers * top-10 * three (2560 x 640) products, multiply+add = 2 ops.
ops_per_token = 48*10*3*2560*640*2
for setting, row in [('512', r) for r in control['runs']] + [('auto', r) for r in auto['runs']]:
    if row['phase'] != 'measured': continue
    m = re.search(r'roofline: tokens (\d+) H2D_bytes (\d+) copies (\d+) unique_bytes (\d+) unique_experts (\d+)',row['log'])
    assert m, row['log'][-2000:]
    n, b, copies, unique, experts = map(int,m.groups())
    work = n*ops_per_token
    item = dict(prefill_setting=setting,prompt_tokens=row['tokens'],prefill_tokens=n,h2d_bytes=b,copies=copies,unique_bytes=unique,unique_experts=experts,
        repetition_ratio=b/unique,avoidable_repeated_bytes=b-unique,
        h2d_floor_s=b/(bandwidth['h2d']*1e9),unique_h2d_floor_s=unique/(bandwidth['h2d']*1e9),
        prompt_ms=row['timing']['prompt_ms'],tps=row['tokens']*1000/row['timing']['prompt_ms'],
        routed_ops=work,h2d_ops_per_byte=work/b,unique_h2d_ops_per_byte=work/unique,
        routed_equiv_TOP_s=work/(row['timing']['prompt_ms']/1000)/1e12,
        phases=phase_data(row))
    item['copy_only_saving_at_peak_s'] = item['h2d_floor_s']-item['unique_h2d_floor_s']
    summary['control'].append(item)
summary['ops_per_token_routed_only'] = ops_per_token
summary['notes'] = [
 'Baseline: main fb58e0d, one P4, IQ3_XXS, int8 KV, 512-token prefill chunks, mmap experts, three repetitions, CPU share default.',
 'Controls: same base plus host-only byte counters, CPU expert share disabled, one observation per length and chunk policy. Auto selected 1792 tokens.',
 'Logical file-tier reads are not physical disk bytes or measured GPU DRAM traffic.',
 'CUDA-event phase times include host-issue/copy stalls; dequant includes quantized group gathering, not just dequantization.',
 'Host staging in the engine log is cumulative and overlapping, not an additive per-request component.',
 'Routed-equivalent ops omit dense, attention, recurrent, and quantization work; the 22-TOPS roof is an optimistic INT8 hardware budget, not a measured IQ3 kernel ceiling.',
 'Unique H2D bytes assume each observed streamed expert is copied once across the request. This models reuse only, not an implemented layer-major run.',
 'Zero-phase scenarios hold remaining phase time fixed. They are conditional sensitivity calculations, not universal hardware bounds.',
]
(OUT/'roofline-summary.json').write_text(json.dumps(summary,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
fig.suptitle('Tesla P4 prefill: repeated transfers matter, but do not explain the whole request',fontsize=15,fontweight='bold')
ax=axs[0]
labels=['512 tokens','4K tokens','8K tokens']
groups=[('Wait for copy',['wait copy'],'#d58c3d'),('Gather / dequant / issue',['dequant','gather','host grouping'],'#efbc71'),('Expert math + combine',['gemm gate/up','gemm down','combine'],'#447fbd'),('Other model stages',None,'#8cacc4')]
bottom=np.zeros(3)
for title,names,color in groups:
    vals=[]
    for r in summary['baseline']:
        ms=sum(r['phases_ms'].get(k,0) for k in names) if names else r['total_ms']-sum(r['phases_ms'].get(k,0) for _,ns,_ in groups if ns for k in ns)
        vals.append(ms/r['total_ms']*100)
    ax.bar(labels,vals,bottom=bottom,label=title,color=color,width=.6)
    for i,v in enumerate(vals):
        if v>5:ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',color='white' if color=='#447fbd' else '#172535')
    bottom+=np.array(vals)
ax.set_ylabel('Share of profiled compute-stream timeline (%)')
ax.set_title('Measured baseline, 512-token chunks\n3 repeats; all expert data served from page cache',fontsize=11)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9,frameon=False)
ax.set_ylim(0,104)

ax=axs[1]
I=np.logspace(0,5,300)
ax.loglog(I,np.minimum(22,bandwidth['h2d']*I/1000),label=f"PCIe roof: {bandwidth['h2d']:.2f} GB/s",color='#d58c3d',lw=2)
ax.loglog(I,np.minimum(22,bandwidth['vram_read']*I/1000),label=f"Same bytes read locally: {bandwidth['vram_read']:.1f} GB/s",color='#447fbd',ls='--')
ax.axhline(22,color='#666666',lw=1,ls=':',label='22 INT8 TOP/s: vendor peak')
for r,c in zip([r for r in summary['control'] if r['prefill_setting']=='512'],['#28875c','#814bae']):
    x,y=r['h2d_ops_per_byte'],r['routed_equiv_TOP_s']
    xx=r['unique_h2d_ops_per_byte']
    ax.scatter([x],[y],s=55,color=c,zorder=4)
    ax.annotate('',xy=(xx,y),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=c,lw=2))
    ax.annotate(f"{r['prompt_tokens']//1024}K: {r['repetition_ratio']:.1f}x fewer bytes\nif each streamed expert is copied once",xy=(x,y),xytext=(4,12 if r['prompt_tokens']==4096 else -32),textcoords='offset points',fontsize=8,color=c)
ax.set_xlabel('Useful routed-expert operations / byte\nPoints use measured H2D bytes; arrows model reuse only')
ax.set_ylabel('Routed-expert equivalent TOP/s over full prefill')
ax.set_title('Bandwidth roofs and GPU-only expert controls\nNot a full-model FLOP or measured VRAM-traffic roofline',fontsize=11)
ax.set_xlim(1,1e5);ax.set_ylim(.01,40);ax.grid(True,which='major',alpha=.15)
ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.savefig(OUT/'p4-prefill-roofline.png',dpi=180)
fig.savefig(OUT/'p4-prefill-roofline.svg')
print(json.dumps({'bandwidth':bandwidth,'baseline':[{k:r[k] for k in ['tokens','median_tps','expert_phase_zero_speedup']} for r in summary['baseline']], 'controls':[{k:r[k] for k in ['prefill_setting','prompt_tokens','tps','h2d_bytes','unique_bytes','repetition_ratio']} for r in summary['control']]},indent=2))
