#!/usr/bin/env python3
"""Runs benchmark.py unmodified against a server that needs an API key.

benchmark.py has no option for an Authorization header, so this wrapper adds
`Authorization: Bearer <key>` to every urllib request and then runs the
harness file as __main__. The key is read from a file and never printed.

usage: run_harness.py KEY_FILE BENCHMARK_PY [benchmark.py arguments ...]
"""
import runpy
import sys
import urllib.request

key = open(sys.argv[1]).read().strip()
harness = sys.argv[2]
original = urllib.request.urlopen


def urlopen(request, *args, **kwargs):
    if isinstance(request, str):
        request = urllib.request.Request(request)
    if not request.has_header("Authorization"):
        request.add_header("Authorization", "Bearer " + key)
    return original(request, *args, **kwargs)


urllib.request.urlopen = urlopen
sys.argv = [harness] + sys.argv[3:]
runpy.run_path(harness, run_name="__main__")
