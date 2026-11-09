"""Check the pinned source manifest, including a mutation counterexample."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

def blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def check(root):
    for line in (root/'src/app/text/UPSTREAM').read_text().splitlines():
        if not line or line.startswith('#'):continue
        local,upstream,commit,expected,mode,current=line.split('\t')
        data=(root/local).read_bytes()
        if mode in ['verbatim','include-only']:
            actual=blob(data)
            if actual!=current:raise ValueError(f'{local}: recorded={current} calculated={actual}')
            if mode=='include-only':
                assert data.count(b'#include "app/text/unicode-data.h"')==1
                data=data.replace(b'#include "app/text/unicode-data.h"',b'#include "unicode-data.h"')
            actual=blob(data)
            if actual!=expected:raise ValueError(f'{local}: recorded={expected} calculated={actual}')
        else:
            header=data[:600].decode()
            assert 'Copyright (c) 2023-2026 The ggml authors' in header,local
            assert 'adapted' in header and commit+'/'+upstream in header,local
        assert commit=='7ab4ee7baad2d920464cbacfad4f4b07cf111fd2'
    assert b'MIT License' in (root/'src/app/text/LICENSE.llama.cpp').read_bytes()

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();check(a.root)
    with tempfile.TemporaryDirectory(prefix='moe-text-upstream-') as name:
        root=Path(name);shutil.copytree(a.root/'src/app/text',root/'src/app/text')
        file=root/'src/app/text/unicode-data.cpp';file.write_bytes(file.read_bytes()+b' ')
        try:check(root)
        except ValueError as e:print('mutation rejected:',e)
        else:raise AssertionError('mutation accepted')
    print('upstream: passed')
if __name__=='__main__':main()
