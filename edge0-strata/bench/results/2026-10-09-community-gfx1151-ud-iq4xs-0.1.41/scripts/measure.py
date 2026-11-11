#!/usr/bin/env python3
"""Run the fixed EVO-X2 report protocol. Requires exclusive GPU use, a prepared pack/MTP, and HIP build.

This script controls only the Strata processes it starts. Stop other GPU servers first.
"""
import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[4]
HERE=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
for name in ['model','build','sdk','out']:ap.add_argument('--'+name,type=pathlib.Path,required=True)
ap.add_argument('--port',type=int,default=18741)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
url=f'http://127.0.0.1:{a.port}'
base_env={**os.environ,'PYTHONUNBUFFERED':'1','ROCM_PATH':str(a.sdk),'HIP_PATH':str(a.sdk),'HIP_PLATFORM':'amd','LD_LIBRARY_PATH':':'.join(str(a.sdk/x) for x in ['lib','lib/rocm_sysdeps/lib','lib/llvm/lib']),'STRATA_HIPBLASLT_TUNING':str(ROOT/'tools/hip/gfx1151-hipblaslt-100401.txt')}
# Avoid inheriting unrelated engine experiments from the caller.
base_env={k:v for k,v in base_env.items() if not k.startswith('STRATA_') or k=='STRATA_HIPBLASLT_TUNING'}
FAST=dict(STRATA_PF_FUSED='1',STRATA_PF_GEMM='1',STRATA_HC_UPMIX='1',STRATA_PA_FAST='1',STRATA_HIP_WMMA='1',STRATA_SELECT_WMMA='1',STRATA_HC_Q8='1',STRATA_PF_SWITCH_MIN_T='4096',STRATA_PREFILL_STREAM_MIN='128')
server=None;server_log=None;phase='initial';events=[];stop_event=threading.Event()
def gtt_bytes():
    return sum(int(p.read_text()) for p in pathlib.Path('/sys/class/drm').glob('card*/device/mem_info_gtt_used'))
initial_gtt=gtt_bytes()

def event(kind,**fields):
    row=dict(timestamp=time.time(),phase=phase,event=kind,**fields);events.append(row)
    with (a.out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def group_stop(p):
    if p is None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=15)

def stop_server():
    global server,server_log
    group_stop(server);server=None
    if server_log:server_log.close();server_log=None
    # Do not start the next GPU process until KFD no longer attributes mappings to this engine.
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        alive=[]
        for p in pathlib.Path('/sys/class/kfd/kfd/proc').glob('*'):
            try:
                cmd=(pathlib.Path('/proc')/p.name/'cmdline').read_bytes()
                if str(a.build).encode() in cmd:alive.append(p.name)
            except FileNotFoundError:continue
        if not alive and gtt_bytes() <= initial_gtt + 512*1024**2:return
        time.sleep(1)
    raise RuntimeError('Strata GPU mappings/GTT were not released')

def get(path):
    with urllib.request.urlopen(url+path,timeout=10) as response:return json.load(response)

def config(name,ctx,fast=False,parallel=1,borrow=True):
    args=['--pack',str(a.model/'pack'),'--native',str(a.model/'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'),'--expert-profile',str(ROOT/'data/expert-profile.bin'),'--expert-cache','auto','--resident-budget-gib','55','--prefill','16384' if fast else 'auto','--spec','4','--spec-min-p','0.5','--mtp',str(a.model/'mtp/rt'),'--max-context',str(ctx),'--kv','int8']
    if ctx>=65536:args+=['--kv-resident','32768']
    if fast:args+=['--lookup-chain','3','--mtp-q4','all']
    if not borrow:args+=['--no-prefill-borrow']
    cfg=dict(exe=str(a.build/'strata'),args=args,cwd=str(ROOT),tokenizer=str(a.model/'pack/tokenizer'),model_name='strata',log=str(a.out/(name+'-engine.log')),host='127.0.0.1',port=a.port,parallel=parallel,open_browser=False)
    path=a.out/(name+'-config.json');path.write_text(json.dumps(cfg,indent=2));return path

def start(name,ctx,fast=False,parallel=1,borrow=True,qfuse=None):
    global server,server_log,phase
    stop_server();phase=name
    cfg=config(name,ctx,fast,parallel,borrow)
    env={**base_env,**(FAST if fast else {})}
    if qfuse is not None:env['STRATA_QFUSE']=str(qfuse)
    (a.out/(name+'-env.json')).write_text(json.dumps({k:v for k,v in env.items() if k.startswith('STRATA_')},indent=2))
    server_log=(a.out/(name+'-server.log')).open('w')
    server=subprocess.Popen([sys.executable,str(ROOT/'serve/server.py'),'--engine','strata','--config',str(cfg),'--host','127.0.0.1','--port',str(a.port)],cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
    event('server-start',pid=server.pid,config=str(cfg))
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if server.poll() is not None:raise RuntimeError(f'{name}: server exited {server.returncode}')
        try:
            data=get('/health')
            if data.get('status') in ['ok','healthy','ready']:
                (a.out/(name+'-health.json')).write_text(json.dumps(data,indent=2));event('server-ready');return cfg
        except (OSError,ValueError):pass
        time.sleep(2)
    raise TimeoutError(name+' startup')

def run(label,script,args,timeout=7200,extra=None,allowed=(0,)):
    cmd=[sys.executable,str(script),*map(str,args)];event('command',label=label,command=cmd)
    with (a.out/(label+'.log')).open('w') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env={**base_env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:group_stop(p);rc=124
        except BaseException:
            group_stop(p)
            raise
        finally:group_stop(p)
    event('result',label=label,returncode=rc)
    if rc not in allowed:raise RuntimeError(f'{label} failed: {rc}')
    return rc

def telemetry():
    with (a.out/'telemetry.jsonl').open('w') as f:
        while not stop_event.is_set():
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
            gpu={}
            for d in pathlib.Path('/sys/class/drm').glob('card*/device'):
                for key in ['mem_info_gtt_used','mem_info_vram_used','gpu_busy_percent']:
                    p=d/key
                    if p.exists():gpu[str(d.parent.name)+'/'+key]=int(p.read_text().strip())
            row=dict(timestamp=time.time(),phase=phase,MemAvailable=mem['MemAvailable'],SwapFree=mem['SwapFree'],gpu=gpu)
            f.write(json.dumps(row)+'\n');f.flush();stop_event.wait(2)

thread=threading.Thread(target=telemetry,daemon=True);thread.start()
def interrupted(signum,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupted)
try:
    # Cold process, speed repeats in the same order/state as the public P40 study.
    start('default-32k',32768)
    shared=['--engine','openai','--base',url,'--model','strata','--arm','evo-x2-ud-iq4xs','--think','off','--out',str(a.out/'quality')]
    run('p40-speed',HERE/'quality.py',shared+['--suite','speed','--reps','3'])
    run('quality-69x2',HERE/'quality.py',shared+['--suite','reasoning,code,tools,complete,longctx','--reps','2'])
    # Same public tokenizer-counted 4K/32K/128K workload in both arms, each with its own warmup.
    public=ROOT/'bench/results/2026-10-04-community-v100-16gb-ram/benchmark.py'
    for fast in [False,True]:
        name='fast-long' if fast else 'default-long';start(name,131072,fast=fast)
        run(name+'-speed',public,['--root',ROOT,'--pack',a.model/'pack','--url',url,'--out',a.out/name,'--runs','3'])
    start('default-262k',262144)
    run('needle',HERE/'needle.py',['--url',url,'--lengths','32k,128k,262k','--depths','10,50,90','--out',a.out/'needles.json'],extra={'NEEDLE_REQUEST_DIR':str(a.out/'needle-requests')},allowed=(0,1))
    # '262k' means 262*1024; the official script skips it at a 262144-token native limit.
    # 256k names that native limit correctly, without changing the script's length/depth construction.
    run('needle-native',HERE/'needle.py',['--url',url,'--lengths','256k','--depths','10,50,90','--out',a.out/'needles-native.json'],extra={'NEEDLE_REQUEST_DIR':str(a.out/'needle-native-requests')},allowed=(0,1))
    stop_server();phase='batch-exact'
    cfg=config('batch-exact',131072,parallel=1,borrow=False)
    run('batch-exact',ROOT/'tools/batch_test.py',['--exe',a.build/'strata','--config',cfg,'--batch','2','--n','2','--max-new','150','--dump',a.out/'batch-exact.json','--extra','--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow'],timeout=1200,extra={'BATCH_TEST_LOG':str(a.out/'batch-exact-engine.log'),'STRATA_QFUSE':'1'},allowed=(0,2,124))
    # All four cells retained. QFUSE off is diagnostic, never silently made the baseline.
    for qfuse in [1,0]:
        for borrow in [True,False]:
            name=f'parallel-qfuse{qfuse}-borrow{int(borrow)}';start(name,131072,parallel=2,borrow=borrow,qfuse=qfuse)
            run(name,HERE/'parallel.py',['--pack',a.model/'pack','--url',url,'--out',a.out/name],timeout=750,allowed=(0,1,124))
    event('all-measurements-complete')
finally:
    stop_event.set();thread.join(timeout=5);stop_server()
