"""Real HTTP chat validation, with emitted token IDs captured at the engine boundary."""
import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__)
for name in ['source','config','baseline','candidate','mtp','output']:
 ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
source=a.source.resolve()
sys.path[:0]=[str(source),str(source/'tools')]
from serve.server import StrataEngine, Service, engine_args, child_env, serve
from serve.frontend import ChatTemplate
import strata_tokenizer as ST

cfg=json.loads(a.config.read_text())
cfg={k:v for k,v in cfg.items() if k in ['args','tokenizer','gpu','lib_dirs','env']}
args=list(cfg['args'])
for flag in ['--conversation-cache-spill-dir','--conversation-cache-disk-mib','--mtp']:
 if flag in args:i=args.index(flag);del args[i:i+2]
args=[a for a in args if a!='--conversation-cache-disk-only']
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text());tokens=[None]*len(vocab)
for t,i in vocab.items():tokens[i]=t
tok=ST.Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()))
template=ChatTemplate(tp/'chat_template.jinja')
rows=[]
out=a.output;out.mkdir(parents=True,exist_ok=True)
class AuditEngine(StrataEngine):
 def generate(self,*a,**kw):
  self.emitted=[]
  for token in super().generate(*a,**kw):
   if token is not None:self.emitted.append(token)
   yield token

for mode in ['baseline','off','mtp']:
 exe=(a.baseline if mode=='baseline' else a.candidate).resolve()
 current=list(args)
 if mode=='mtp':current+=['--mtp',str(a.mtp.resolve())];current[current.index('--spec')+1]='4'
 local=dict(cfg,args=current,exe=str(exe),cwd=str(source),log=str(out/(mode+'.log')))
 engine=AuditEngine(local['exe'],engine_args(local),cwd=local['cwd'],log=local['log'],env=child_env(local))
 service=Service(engine,tok,template,model_name='bias-validation')
 server=serve(service,port=0)
 def call(label,prompt='Write the Chinese word for hello. Only the word.',expect=200,**kw):
  req={'model':'bias-validation','messages':[{'role':'user','content':prompt}],'max_tokens':8,'temperature':0,'reasoning_effort':'none',**kw}
  data=json.dumps(req).encode();started=time.perf_counter()
  http=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(http,timeout=180) as response:status=response.status;raw=response.read().decode()
  except urllib.error.HTTPError as e:
   with e:status=e.code;raw=e.read().decode()
  row={'mode':mode,'label':label,'status':status,'seconds':time.perf_counter()-started,'request_bytes':len(data),
       'token_ids':list(engine.emitted) if status==200 else [],'response':raw,'engine':dict(engine.last) if status==200 else {}}
  rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
  print(mode,label,status,row['token_ids'],flush=True)
  assert status==expect,(label,status,raw)
  return row
 try:
  first=call('unchanged-greedy')
  sampled=call('unchanged-sampled',prompt='Write one sentence about Python.',temperature=0.7,top_k=20,seed=42)
  if mode=='baseline':continue
  assert engine.info.get('logit_bias')==1
  banned=first['token_ids'][0]
  one=call('ban-object',logit_bias={str(banned):-100})
  two=call('ban-pairs',logit_bias=[[banned,False]])
  assert banned not in one['token_ids'] and banned not in two['token_ids']
  assert one['token_ids']==two['token_ids']
  boosted=tok.encode(' banana',parse_special=False)[0]
  positive=call('positive-bias',logit_bias={str(boosted):100},max_tokens=3)
  assert positive['token_ids'][0]==boosted
  # Changing requests must not inherit a prior mask, including after a rejected request.
  call('invalid-id',expect=400,logit_bias={str(len(tokens)):1},stream=True)
  call('invalid-value',expect=400,logit_bias={'0':101},stream=True)
  again=call('cleared-bias')
  assert again['token_ids']==first['token_ids']
  empty=call('empty-bias',logit_bias={})
  assert empty['token_ids']==first['token_ids']
  large=set(range(103214));large.add(banned)
  if len(large)<103215:large.add(103214)
  bulk=call('103215-bans',logit_bias=[[i,False] for i in sorted(large)],max_tokens=8)
  assert not set(bulk['token_ids']) & large
  streamed=call('stream-ban',logit_bias={str(banned):-100},stream=True)
  assert banned not in streamed['token_ids']
  sampled_ban=call('sampled-ban',logit_bias={str(banned):-100},temperature=0.8,seed=123)
  assert banned not in sampled_ban['token_ids']
 finally:
  server.shutdown();server.server_close();engine.close()

for label in ['unchanged-greedy','unchanged-sampled']:
 before=next(x for x in rows if x['mode']=='baseline' and x['label']==label)
 after=next(x for x in rows if x['mode']=='off' and x['label']==label)
 assert before['token_ids']==after['token_ids'],('default-output-changed',label)
(out/'complete.json').write_text(json.dumps({'status':'passed','requests':len(rows),'token_109266':tok.decode([109266]),
 'baseline':str(a.baseline),'candidate':str(a.candidate),'config':str(a.config)},indent=2,ensure_ascii=False))
