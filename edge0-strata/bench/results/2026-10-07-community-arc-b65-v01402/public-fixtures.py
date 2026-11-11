#!/usr/bin/env python3
"""Bounded prefill diagnosis; content lives only in RAM and process memory."""
import argparse, hashlib, importlib.util, json, os, re, signal, sys, threading, time
from pathlib import Path

R = Path(__file__).resolve().parent
SOURCE = Path(os.environ['STRATA_ROOT'])

def digest(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['matched'])
    a = ap.parse_args()
    runid = os.environ.get('STRATA_TEST_ID','public-fixtures')
    ram = Path(os.environ['STRATA_TEST_RAM'])
    out = Path(os.environ['BENCH_OUTPUT'])
    out.mkdir(exist_ok=False)
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'tools')]
    from serve.server import StrataEngine, ENGINE_REQUEST
    from serve.frontend import ChatTemplate
    from strata_tokenizer import Tokenizer
    spec = importlib.util.spec_from_file_location('frozen', R / 'frozen-community.py')
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    cfg = json.loads(Path(os.environ.get('STRATA_TEST_PROFILE',str(R/'profile.json'))).read_text())
    cfg['exe']=os.path.expandvars(cfg['exe']);cfg['cwd']=os.path.expandvars(cfg['cwd'])
    cfg['args']=[os.path.expandvars(x) for x in cfg['args']]
    tok = Tokenizer.from_gguf(Path(cfg['args'][cfg['args'].index('--native') + 1]))
    tpl = ChatTemplate(Path(cfg['args'][cfg['args'].index('--pack') + 1]) / 'tokenizer/chat_template.jinja')
    def shaped(task, target):
        rendered = tpl.render([{'role':'system','content':'Follow the requested format and provide a thorough answer.'},
                               {'role':'user','content':'PAD_MARKER\n'+task}], enable_thinking=False)
        before, after = rendered.split('PAD_MARKER')
        aa, bb = tok.encode(before, parse_special=True), tok.encode(after, parse_special=True)
        ids = aa + tok.encode(frozen.FILLER)[:target-len(aa)-len(bb)] + bb
        assert len(ids) == target
        return ids
    public = [int(x) for x in (R / 'benchy-long.ids').read_text().strip().split(',')]
    assert len(public) == 2185
    bench7000 = (public[:2000] * 4)[:7000]
    our7000 = shaped(frozen.TASKS[0], 7000)
    our512 = shaped(frozen.TASKS[0], 512)
    warm = shaped(frozen.WARMUP, 512)
    canary = tok.encode(tpl.render([{'role':'user','content':'Reply only with CANARY_MAPLE_73.'}], enable_thinking=False), parse_special=True)
    env = dict(os.environ); env['STRATA_TRACE']='1'; assert 'STRATA_DBG_NAN' not in env
    short=[int(x) for x in (R/'benchy-short.ids').read_text().strip().split(',')]; assert len(short)==20
    class Engine(StrataEngine):
        def _parse_done(self, line):
            self.done_line = line
            return super()._parse_done(line)
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    engine = None; rows = []; failure = None; life = {}
    streams = {}
    log = ram / 'native.log'
    def save():
        (out / 'requests.json').write_text(json.dumps(rows, indent=2)+'\n')
        (out / 'lifecycle.json').write_text(json.dumps(life, indent=2)+'\n')
    def io():
        return {k:int(v) for k,v in (x.split(':',1) for x in Path(f'/proc/{engine.proc.pid}/io').read_text().splitlines())}
    def ask(ids, cap, label, expected=None):
        assert not cancel.is_set()
        offset = log.stat().st_size; before = io(); engine.done_line = None
        arrivals = []; output = []; start = time.monotonic()
        for t in engine.generate(ids, cap, {'temperature':0.,'seed':17}, cancel):
            if t is not None: arrivals.append(time.monotonic()); output.append(t)
        elapsed = time.monotonic()-start; after=io(); n=dict(engine.last)
        text = tok.decode(output).replace('<|im_end|>','').replace('<|endoftext|>','').strip()
        assert engine.done_line and n['generated']==len(output) and n['prompt_tokens']==len(ids)
        assert n['prompt_read']==len(ids) and n['reused']==0 and output and not cancel.is_set()
        if expected: assert text==expected
        with log.open() as f: f.seek(offset); raw=f.read()
        assert not re.search(r'\b[1-9]\d* (?:of \d+ logits )?non-finite\b|\bnonfinite [1-9]\d*',raw)
        allowed = [x for x in raw.splitlines() if ENGINE_REQUEST.search(x) or
                   re.match(r'^strata (?:trace: (?:read \d|lent \d|refilled \d|prompt chunk)|prefill timing:|prefill timing \(|serve: prompt|generate: prompt)',x)]
        row={'label':label,'input_tokens':len(ids),'output_cap':cap,'output_tokens':len(output),
             'input_ids_sha256':digest(ids),'output_ids_sha256':digest(output),'native':n,
             'native_done_line':engine.done_line,'prompt_tps':len(ids)*1000/n['prompt_ms'],
             'decode_tps':len(output)*1000/n['decode_ms'],'ttft_s':arrivals[0]-start,
             'total_s':elapsed,'process_io_delta':{k:after[k]-before[k] for k in before},
             'numeric_trace':allowed,'valid':True}
        prefixes={}
        for k in (64,128,256,512,640):
            if len(output)>=k:
                prefixes[str(k)]={'ids_sha256':digest(output[:k]),'post_first_tps':None if k<2 else (k-1)/(arrivals[k-1]-arrivals[0]),'after_prompt_elapsed_ms':(arrivals[k-1]-start)*1000-n['prompt_ms']}
        row['prefixes']=prefixes
        if label.startswith('public2185'):
            if 'public2185' in streams:
                row['matches_previous_prefix']=streams['public2185']==output[:len(streams['public2185'])]
                assert row['matches_previous_prefix']
            else: streams['public2185']=output
        row['target']=label.rsplit('-r',1)[0]
        for prior in rows:
            if prior.get('target')==row['target']:
                assert prior['output_ids_sha256']==row['output_ids_sha256']
                row['matches_previous_repeat']=True
        rows.append(row);save()
        print(json.dumps({k:row[k] for k in ['label','input_tokens','output_tokens','prompt_tps','ttft_s','total_s']}),flush=True)
    try:
        life={'mode':a.mode,'binary_sha256':hashlib.file_digest(Path(cfg['exe']).open('rb'),'sha256').hexdigest(),
              'args':cfg['args'],'diagnostic_env':{k:env[k] for k in ['STRATA_TRACE','STRATA_STAGER_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
              'intermediate_tensor_diagnostic':env.get('STRATA_DBG_NAN'), 'instrumented':False}
        life['launches']=[]
        for rep in range(1,4):
            assert not cancel.is_set();streams.clear()
            log=ram/('native-'+str(rep)+'.log')
            start=time.monotonic()
            engine=Engine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=str(log),env=env)
            launch={'repeat':rep,'startup_ready_s':time.monotonic()-start,'info':engine.info}
            life['launches'].append(launch);save()
            ask(public,256,'public2185-256-r'+str(rep))
            ask(short,256,'public20-256-r'+str(rep))
            ask(canary,64,'canary-end-r'+str(rep),'CANARY_MAPLE_73')
            p=engine.proc;start=time.monotonic();p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=75)
            launch.update(native_exit=p.returncode,quit_exit_s=time.monotonic()-start);assert p.returncode==0
            engine.close();engine=None
            for _ in range(5):time.sleep(1);assert not cancel.is_set()
            raw=log.read_text()
            launch['startup_geometry']=[x for x in raw.splitlines() if re.match(r'^strata (?:generate|serve): (?:GPU \d+:|expert cache \d|\d+ of \d+ experts missing|\d+ MiB of VRAM free|prompt path|session is up)',x)]
            save()
        life['native_exit']=0
    except BaseException as e:
        failure={'type':type(e).__name__,'reason':str(e)}
    finally:
        if engine is not None:
            try:
                p=engine.proc;start=time.monotonic();p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=75)
                life.update(native_exit=p.returncode,quit_exit_s=time.monotonic()-start)
                assert p.returncode==0
                engine.close();engine=None
                for _ in range(5): time.sleep(1);assert not cancel.is_set()
            except BaseException as e:
                failure=failure or {'type':type(e).__name__,'reason':str(e)}
        save()
        (out/'result.json').write_text(json.dumps({'failure':failure,'requests':len(rows),'native_exit':life.get('native_exit'),'content_persisted':False},indent=2)+'\n')
    if failure:raise SystemExit(1)

if __name__=='__main__':main()
