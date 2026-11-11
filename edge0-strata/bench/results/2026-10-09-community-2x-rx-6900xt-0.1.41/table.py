import json,os,sys
E=os.path.dirname(os.path.abspath(__file__))
arms=[d for d in ['split-stock','split-stock-kc','split-sw-kc','split-prs-kc','split-prs-kc-dec'] if os.path.exists(f'{E}/{d}/summary.json')]
print(f"{'arm':18s}" + ''.join(f"{t:>22s}" for t in ('4K','32K','128K')))
for a in arms:
    s=json.load(open(f'{E}/{a}/summary.json')); row=f"{a:18s}"
    for k in ('4096','32768','128000'):
        if k in s:
            p=s[k]['prefill_tok_s']; d=s[k]['decode_tok_s']
            row+=f"{p['median']:8.0f} [{p['min']:.0f}-{p['max']:.0f}] {d['median']:5.1f}".rjust(22)
        else: row+=f"{'-':>22s}"
    print(row)
