"""Record the same English greeting request before and after a token ban."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ap = argparse.ArgumentParser(description=__doc__)
for name in ['source', 'config', 'candidate', 'output']:
    ap.add_argument('--' + name, type=Path, required=True)
a = ap.parse_args()
source = a.source.resolve()
sys.path[:0] = [str(source), str(source / 'tools')]
from serve.server import StrataEngine, Service, engine_args, child_env, serve
from serve.frontend import ChatTemplate
import strata_tokenizer as ST

cfg = json.loads(a.config.read_text())
cfg = {k: v for k, v in cfg.items() if k in ['args', 'tokenizer', 'gpu', 'lib_dirs', 'env']}
args = list(cfg['args'])
for flag in ['--conversation-cache-spill-dir', '--conversation-cache-disk-mib', '--mtp']:
    if flag in args:
        i = args.index(flag)
        del args[i:i+2]
args = [x for x in args if x != '--conversation-cache-disk-only']
cfg['args'] = args
tp = Path(cfg['tokenizer'])
vocab = json.loads((tp / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
tok = ST.Tokenizer(tokens, (tp / 'merges.txt').read_text().split('\n'),
                   json.loads((tp / 'token_type.json').read_text()))
a.output.mkdir(parents=True, exist_ok=True)

class AuditEngine(StrataEngine):
    def generate(self, *args, **kwargs):
        self.emitted = []
        for token in super().generate(*args, **kwargs):
            if token is not None:
                self.emitted.append(token)
            yield token

engine = AuditEngine(str(a.candidate.resolve()), engine_args(cfg), cwd=str(source),
                     log=str(a.output / 'engine.txt'), env=child_env(cfg))
server = serve(Service(engine, tok, ChatTemplate(tp / 'chat_template.jinja'),
                       model_name='bias-validation'), port=0)
base = dict(model='bias-validation', messages=[dict(role='user', content=
    'Reply with one English greeting, one word only.')], max_tokens=8,
    temperature=0, reasoning_effort='none')
rows = []
def call(label, **extra):
    request = dict(base, **extra)
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
        data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read())
    row = dict(label=label, request=request, response=result, token_ids=list(engine.emitted))
    rows.append(row)
    (a.output / 'rows.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(label, row['token_ids'], result['choices'][0]['message']['content'], flush=True)
    return row

try:
    hello = tok.encode('Hello', parse_special=False)
    print('Hello tokenizer IDs:', hello, flush=True)
    before = call('before')
    assert len(hello) == 1 and hello[0] in before['token_ids'], 'Choose a prompt whose baseline emits Hello'
    banned = hello[0]
    after = call('after-object', logit_bias={str(banned): -100})
    pairs = call('after-pairs', logit_bias=[[banned, False]])
    cleared = call('cleared')
    assert banned not in after['token_ids'] and banned not in pairs['token_ids']
    assert after['token_ids'] == pairs['token_ids']
    assert cleared['token_ids'] == before['token_ids']
    (a.output / 'complete.json').write_text(json.dumps(dict(status='passed', requests=len(rows),
        token_id=banned, token_text=tok.decode([banned])), indent=2))
finally:
    server.shutdown()
    server.server_close()
    engine.close()
