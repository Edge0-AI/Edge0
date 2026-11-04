"""Validate JSON/TSV agreement, HF attribution, and the per-group piece digests."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

def check(root):
    def read(name):return json.loads((root/name).read_bytes())
    corpus=read('corpus.json');assert len(corpus)>=400
    assert (root/'corpus.tsv').read_text()==''.join(r['id']+'\t'+r['text'].encode().hex()+'\n' for r in corpus)
    golden=read('tokenize_golden.json');rows=golden['rows']
    assert len(rows)==2*len(corpus)
    assert (root/'tokenize_golden.tsv').read_text()==''.join(f"{r['id']}\t{int(r['parse_special'])}\t{r['text'].encode().hex()}\t{' '.join(map(str,r['llama']))}\n" for r in rows)
    diffs=[]
    for r in rows:
        assert r['llama']==r['hf_no_normalizer'],r['id']
        if r['hf_original']!=r['hf_no_normalizer']:
            assert r['attribution']=='normalization'
            assert r['normalizer_output']!=r['text'];diffs.append(r)
        else:assert r['attribution']=='none'
    assert golden['differences']==diffs and golden['difference_count']==len(diffs)
    assert golden['equal_count']==len(rows)-len(diffs)
    pieces=read('pieces_golden.json');lines=(root/'pieces_golden.tsv').read_text().splitlines()
    assert len(lines)==pieces['count']==248320
    for special in [0,1]:
        # Per-group digests only: any byte change in the TSV changes its group's digest.
        group=bytearray();groups=[];eog=[]
        for i,line in enumerate(lines):
            f=line.split('\t');assert int(f[0])==i
            b=bytes.fromhex(f[1+special]);group+=struct.pack('<II',i,len(b))+b
            if (i+1)%1024==0 or i+1==len(lines):groups.append(hashlib.sha256(group).hexdigest());group.clear()
            if f[3]=='1':eog.append(i)
        assert pieces['group_size']==1024 and groups==pieces['groups'][special]
        assert eog==pieces['eog']
    utf=read('utf8_golden.json');assert len(utf)>=40
    assert (root/'utf8_golden.tsv').read_text()==''.join(r['input']+'\t'+r['output']+'\n' for r in utf)
    for r in utf:assert bytes.fromhex(r['input']).decode('utf-8','replace').encode().hex()==r['output']
    templates=read('template_golden.json')['rows']
    assert (root/'template_golden.tsv').read_text()==''.join(r['id']+'\t'+r['text'].encode().hex()+'\t'+' '.join(map(str,r['tokens']))+'\t'+';'.join(m['role']+':'+m['content'].encode().hex() for m in r['messages'])+'\n' for r in templates)
    for r in templates:
        assert ''.join(f['text'] for f in r['fragments'])==r['text']
        if r['hf_no_normalizer'] is not None:assert r['tokens']==r['hf_no_normalizer']
    answers=root.parent/'answers'
    for case in json.loads((answers/'cases.json').read_bytes())['cases']:
        expected=json.loads((answers/'tiny-base-cpu-flash'/case['id']/'tokens.json').read_bytes())['input']
        matches=[r for r in rows if r['text']==case['text'] and r['parse_special']]
        assert matches and all(r['llama']==expected for r in matches),case['id']
        if case['id']=='chat-intro':
            intro=next(r for r in templates if r['id']=='chat-intro')
            assert intro['text']==case['text'] and intro['tokens']==expected
    original=read('hf_original.json')['rows'];diagnostic=read('hf_no_normalizer.json')['rows']
    for r,o,d in zip(rows,original,diagnostic):
        assert r['id']==o['id']==d['id'] and r['hf_original']==o['tokens'] and r['hf_no_normalizer']==d['tokens']
    assert len(rows)==len(original)==len(diagnostic)
    print(f'JSON/TSV and digests passed: corpus={len(corpus)} differences={len(diffs)} pieces={len(lines)}')

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('executable',nargs='?');a=p.parse_args();check(a.root)
    if a.executable:subprocess.run([a.executable],check=True)
if __name__=='__main__':main()
