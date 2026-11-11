#!/usr/bin/env python3
"""Prepare the pinned Q2_0 draft pack with upstream hash verification."""
import argparse
from pathlib import Path
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--assets', type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.source / 'tools'))
import mtp_fetch as m
assert m.REVISION == 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
m.resolve_repo = lambda: None  # No fallback to a floating revision.
m.fetch(str(a.assets / 'mtp-source'), None)
assert not m.verify(str(a.assets / 'mtp-source'))
subprocess.run([sys.executable, str(a.source / 'tools/mtp_pack.py'), '--src', str(a.assets / 'mtp-source'),
                '--experts', 'q2_0', '--out', str(a.assets / 'mtp-q2_0.gguf')], check=True)
subprocess.run([sys.executable, str(a.source / 'tools/mtp_rt.py'), '--gguf', str(a.assets / 'mtp-q2_0.gguf'),
                '--out', str(a.assets / 'mtp-rt')], check=True)
