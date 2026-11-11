import json,sys,re,urllib.request,time,glob
URL="http://127.0.0.1:8085/v1/chat/completions"
def ask(msgs,max_tokens=256):
    body={"messages":msgs,"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"content-type":"application/json"}),timeout=900)
    d=json.load(r); d["wall"]=time.time()-t
    return d
def show(tag,d):
    tm=d["timings"];u=d["usage"]
    print(f"[{tag}] prompt_n={tm['prompt_n']} cache_n={tm['cache_n']} prompt {tm['prompt_per_second']:.1f} tok/s | decode {tm['predicted_n']} tok {tm['predicted_per_second']:.1f} tok/s | drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')} | wall {d['wall']:.1f}s | finish {d['choices'][0]['finish_reason']}")
def code(text):
    m=re.search(r"```python\n(.*?)```",text,re.S); return m.group(1) if m else text
def run_tests(src,tests):
    g={}
    try:
        exec(src,g); exec(tests,g); return "PASS"
    except Exception as e: return f"FAIL {e!r}"
mode=sys.argv[1]
if mode=="short":
    for i in range(2):
        d=ask([{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number"}],256); show(f"fib run{i}",d)
    print(d["choices"][0]["message"]["content"])
    print("fib tests:",run_tests(code(d["choices"][0]["message"]["content"]),"fn=[v for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and k!='exec'][-1]\nassert [fn(i) for i in range(10)]==[0,1,1,2,3,5,8,13,21,34]; assert fn(50)==12586269025"))
    d=ask([{"role":"user","content":"Write a Python function gcd(a, b) that returns the greatest common divisor using Euclid's algorithm. Only the code."}],200); show("gcd",d)
    c=d["choices"][0]["message"]["content"]; print(c)
    print("gcd tests:",run_tests(code(c),"assert gcd(48,18)==6; assert gcd(17,5)==1; assert gcd(0,9)==9; assert gcd(270,192)==6"))
    d=ask([{"role":"user","content":"What is the output of this Python code? print(sum(i*i for i in range(1,11)))  Answer with just the number."}],20); show("known",d); print(d["choices"][0]["message"]["content"], "(expected 385)")
if mode=="long":
    src="".join(open(f).read() for f in sorted(glob.glob("<strata checkout>/tools/*.py"))[:6])[:int(sys.argv[2])]
    msgs=[{"role":"user","content":"Here is some Python source:\n\n"+src+"\n\nIn one paragraph, say what this code is for, then write a short Python function that reverses a string."}]
    for i in range(2):
        d=ask(msgs,256); show(f"long run{i}",d)
    print(d["choices"][0]["message"]["content"][:1200])
if mode=="long2":
    fs=sorted(glob.glob("<strata checkout>/tools/*.py"))
    src="".join(open(f).read() for f in fs[6:12])[:7000]
    msgs=[{"role":"user","content":"Here is some Python source:\n\n"+src+"\n\nWrite a Python implementation of a binary search tree class with insert, search, delete and in-order traversal methods, with docstrings."}]
    d=ask(msgs,256); show("long2 BST",d)
    d2=ask(msgs+[{"role":"assistant","content":d["choices"][0]["message"]["content"]},{"role":"user","content":"Now add a height() method."}],256); show("long2 turn2",d2)
    print(d["choices"][0]["message"]["content"][:600])
