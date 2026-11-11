#!/usr/bin/env python3
import json, os, subprocess, time, urllib.request, statistics

# Set STRATA_API_KEY in the environment; nothing is committed.

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = os.environ.get("STRATA_API_KEY", "")
OUT = "/mnt/1TB/src/Strata/bench-titan"
os.makedirs(OUT, exist_ok=True)

def pcie():
    o = subprocess.run(["nvidia-smi","--query-gpu=index,pcie.link.gen.current,pcie.link.width.current",
                        "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
    return o.replace("\n","; ")

def req(prompt, max_tokens=256):
    body = json.dumps({"model":"strata","messages":[{"role":"user","content":prompt}],
                       "max_tokens":max_tokens,"temperature":0,"stream":False}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json",
        "Authorization":"Bearer "+KEY})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=900) as f:
        d = json.loads(f.read())
    wall=time.time()-t0
    tm=d.get("timings",{}); u=d.get("usage",{})
    return {"prompt_n":tm.get("prompt_n"),"prompt_ms":tm.get("prompt_ms"),
            "prompt_tps":tm.get("prompt_per_second"),"gen_n":tm.get("predicted_n"),
            "gen_ms":tm.get("predicted_ms"),"gen_tps":tm.get("predicted_per_second"),
            "ttft_s":round(wall-(tm.get("predicted_ms",0) or 0)/1000,2),
            "total_s":round(wall,2),"usage_prompt":u.get("prompt_tokens"),
            "usage_completion":u.get("completion_tokens"),
            "finish":d["choices"][0].get("finish_reason")}

BASE = ("Sparse mixture-of-experts routing sends each token through a small subset of "
 "feed-forward experts. The gate projects hidden states into per-expert scores, applies a "
 "top-k mask and renormalizes the surviving weights. Recurrent delta layers maintain linear "
 "state across the sequence while sparse attention indexes only a subset of keys. "
 "Quantized weights trade numerical precision for memory bandwidth at every matmul. ")

def make(n_tokens, variant):
    s = BASE + (" Document revision marker %d. " % variant)
    per = len(s)//4
    reps = max(1, n_tokens//per)
    return s*reps

results={}
for length in (4096, 32768, 128000):
    runs=[]
    for v in range(3):
        p = make(length, v)
        r = req(p)
        r["variant"]=v; r["pcie_during"]=pcie()
        runs.append(r)
        print(length, "run", v+1, "prompt_tps", round(r["prompt_tps"] or 0,1),
              "gen_tps", round(r["gen_tps"] or 0,2), "ttft", r["ttft_s"], flush=True)
    results[str(length)]=runs

summary={}
for k,v in results.items():
    pt=[x["prompt_tps"] for x in v if x["prompt_tps"]]
    gt=[x["gen_tps"] for x in v if x["gen_tps"]]
    tt=[x["ttft_s"] for x in v]
    summary[k]={"prompt_tps":{"runs":[round(x,1) for x in pt],
                              "median":round(statistics.median(pt),1) if pt else None,
                              "min":round(min(pt),1) if pt else None,"max":round(max(pt),1) if pt else None},
                "decode_tps":{"runs":[round(x,2) for x in gt],
                              "median":round(statistics.median(gt),2) if gt else None,
                              "min":round(min(gt),2) if gt else None,"max":round(max(gt),2) if gt else None},
                "ttft_s":{"runs":tt,"median":round(statistics.median(tt),2) if tt else None}}

json.dump({"summary":summary,"runs":results}, open(OUT+"/titan-results.json","w"), indent=1)
print("=== SUMMARY ===")
print(json.dumps(summary, indent=1))
