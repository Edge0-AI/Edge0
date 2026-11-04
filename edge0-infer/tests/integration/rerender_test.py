"""Tokenizer-level exact history preservation and whitespace counterexamples."""
import json
import os
from pathlib import Path
import subprocess
import sys
model=Path(os.environ.get('MOE_TINY','/nonexistent'))/'tiny-a'
if not (model/'assets.tsv').exists():raise SystemExit(77)
def render(messages):
    result=subprocess.run([sys.argv[1],str(model)],input='1\n'+''.join(m['role']+' '+m['content'].encode().hex()+'\n' for m in messages),text=True,capture_output=True,check=True)
    return list(map(int,result.stdout.split()))
def encode(text):
    return list(map(int,subprocess.check_output([sys.argv[2],'tokenize','--model',str(model),text],text=True).split()))
first=[dict(role='user',content='Hello')]
prompt=render(first)
for answer in ['Hello, world!','\u4f60\u597d\uff0c\u4e16\u754c\u3002','```python\nprint(1)\n```','Line one.\nLine two!','  Answer  ','\nAnswer\n']:
    cached=prompt+encode(answer)
    next_tokens=render(first+[dict(role='assistant',content=answer),dict(role='user',content='Next')])
    equal=next_tokens[:len(cached)]==cached
    assert equal==(answer==answer.strip()),(answer,equal)
print('exact rerender tokens and whitespace counterexamples passed')
