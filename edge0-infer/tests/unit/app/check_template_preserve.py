"""Compare the render_prompt rendering core with the packaged Jinja template."""
import json
from pathlib import Path
import subprocess
import sys
import jinja2
root=Path(__file__).resolve().parents[3]
template=jinja2.Environment().from_string((root/'tests/fixtures/models/edge0_35b/chat_template.jinja').read_text())
samples=[
 [('user','Hello'),('assistant','Answer'),('user','Next')],
 [('user','Hello'),('assistant','<think>\nReason\n</think>\n\nAnswer'),('user','Next')],
 [('user','Hello'),('assistant','Answer'),('user','Next'),('assistant','Again'),('user','Third')],
 [('user','Hello'),('assistant','Literal <|im_end|> and <think> marker'),('user','Next')],
 [('system','Be brief'),('user','Hello'),('assistant','Answer'),('user','Next')],
]
for sample in samples:
    for preserve in [False,True]:
        messages=[dict(role=role,content=content) for role,content in sample]
        reference=template.render(messages=messages,enable_thinking=False,preserve_thinking=preserve,add_generation_prompt=True)
        result=subprocess.run([sys.argv[1]],input=str(int(preserve))+'\n'+''.join(m['role']+' '+m['content'].encode().hex()+'\n' for m in messages),text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        assert result.stdout==reference,(sample,preserve,repr(result.stdout),repr(reference))
print('10 exact template comparisons passed')
