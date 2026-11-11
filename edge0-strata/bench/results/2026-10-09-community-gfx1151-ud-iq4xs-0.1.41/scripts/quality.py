#!/usr/bin/env python3
"""Run the unchanged P40 workload/graders, isolating generated Python with bubblewrap.

All arguments are forwarded to the original bench.py. Requires Linux and bwrap.
"""
import importlib.util
import pathlib
import resource
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'bench/results/2026-10-05-community-2x-p40/scripts/bench.py'
spec = importlib.util.spec_from_file_location('p40_bench', SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
original_run = subprocess.run


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))


def isolated_run(command, **kwargs):
    if len(command) == 3 and command[:2] == [sys.executable, '-I']:
        command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
                   '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
                   '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
                   '--tmpfs', '/tmp', '--ro-bind', command[2], '/solution.py',
                   '--chdir', '/tmp', '/usr/bin/python3', '-I', '/solution.py']
        kwargs['preexec_fn'] = limits
    return original_run(command, **kwargs)


# Only the code grader invokes subprocess.run. Task construction and grading stay upstream's.
bench.subprocess.run = isolated_run
if __name__ == '__main__':
    bench.main()
