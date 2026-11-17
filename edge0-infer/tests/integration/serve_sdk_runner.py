"""Use pinned offline SDKs; unavailable packages are an explicit skip."""
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--with', dest='packages', action='append')
parser.add_argument('script')
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = ['uv', 'run', '--no-project', '--offline']
for package in args.packages or ['openai==3.26.1']:
    command += ['--with', package]
command += ['python', '-I']
probe = subprocess.run([*command, '-c', 'pass'], capture_output=True, text=True)
if probe.returncode:
    print('SKIP: pinned offline SDK unavailable')
    raise SystemExit(77)
raise SystemExit(subprocess.call([*command, args.script, *args.arguments]))
