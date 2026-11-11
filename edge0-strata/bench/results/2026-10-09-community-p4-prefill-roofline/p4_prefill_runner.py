import json,os,re,subprocess,sys,threading,time,traceback
from pathlib import Path
ROOT=Path(os.environ.get('ROOF_ROOT','/home/bench/src/strata-pr1656-parent-p4-20261009'))
OUT=Path(os.environ.get('ROOF_OUT','/home/bench/fleet-downloads/issue1670-p4-roofline'))
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.server import StrataEngine,child_env,engine_args
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate

def save(data):
 p=OUT/'results.tmp';p.write_text(json.dumps(data,indent=2));p.replace(OUT/'results.json')

cfg=json.loads(Path(os.environ['ROOF_CONFIG']).read_text())
cfg.update(exe=str(ROOT/'build/strata'),cwd=str(ROOT),gpu=0,log=str(OUT/'engine.log'),model_name='qwen3.8-flash-next-gsq-rco-iq3_xxs')
cfg.pop('layer_split',None)
a=cfg['args']
for flag in ['--shared-expert-arena','--pipeline-windows']:
 if flag in a:i=a.index(flag);del a[i:i+2]
if '--trim-stage-weights' in a:a.remove('--trim-stage-weights')
for flag,value in {'--max-context':'16384','--kv-resident':'16384','--prefill':'512'}.items():a[a.index(flag)+1]=value
if '--mmap-experts' not in a:a+=['--mmap-experts']
cfg['env']={'STRATA_PREFILL_TIMING':'1','STRATA_IQ_MT_MIN':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','STRATA_ATTN_MERGE_V2':'1'}
if os.environ.get('ROOF_CONTROL'):
 cfg['env'].update(STRATA_PREFILL_ROOFLINE='1',STRATA_PREFILL_CPU_SHARE='0')
 if os.environ.get('ROOF_PREFILL'):
  a[a.index('--prefill')+1]=os.environ['ROOF_PREFILL']
(OUT/'config.json').write_text(json.dumps(cfg,indent=2))
args=engine_args(cfg);env=child_env(cfg)
tok=Tokenizer.from_gguf(Path(a[a.index('--native')+1]))
template=ChatTemplate(Path(cfg['tokenizer'])/'chat_template.jinja')
pre,post=template.render([{'role':'user','content':'Background notes:\nFILLER_HERE\nEnd of notes.\nTask: Explain how a Linux administrator diagnoses CPU, memory, disk and network bottlenecks.'}],enable_thinking=False).split('FILLER_HERE')
aa,bb=tok.encode(pre,parse_special=True),tok.encode(post,parse_special=True)
filler=tok.encode('Archived maintenance notes contain background information about ordinary testing and documentation.\n')
prompts={str(n):aa+(filler*((n-len(aa)-len(bb)+len(filler)-1)//len(filler)))[:n-len(aa)-len(bb)]+bb for n in [512,4096,8192]}
plan=[dict(tokens=512,phase='warmup',repeat=-1)]+[dict(tokens=n,phase='measured',repeat=i) for n in [512,4096,8192] for i in range(3)]
if os.environ.get('ROOF_CONTROL'):
 plan=[dict(tokens=512,phase='warmup',repeat=-1)]+[dict(tokens=n,phase='measured',repeat=0) for n in [4096,8192]]
report={'config':cfg,'args':args,'prompts':prompts,'plan':plan,'runs':[],'complete':False,'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
engine=None
try:
 save(report);engine=StrataEngine(cfg['exe'],args,cfg['cwd'],cfg['log'],env);report['engine_info']=engine.info;save(report)
 for entry in plan:
  print('START',entry,flush=True);offset=Path(cfg['log']).stat().st_size;start=time.monotonic();tokens=[]
  timer=threading.Timer(900,engine.proc.kill);timer.start()
  try:
   for token in engine.generate(prompts[str(entry['tokens'])],1,{'temperature':0},threading.Event()):
    if token is not None:tokens.append(token)
  finally:timer.cancel()
  row=dict(entry,output=tokens,wall_s=time.monotonic()-start,timing=dict(engine.last),log=Path(cfg['log']).read_bytes()[offset:].decode(errors='replace'))
  report['runs'].append(row);save(report);print('RESULT',json.dumps({k:v for k,v in row.items() if k!='log'}),flush=True)
  assert row['timing'].get('reused',0)==0 and tokens
 report['complete']=True
except BaseException:
 report['error']=traceback.format_exc();raise
finally:
 if engine:engine.close()
 save(report)
