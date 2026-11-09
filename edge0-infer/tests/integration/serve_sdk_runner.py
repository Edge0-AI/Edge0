"""Use the pinned offline SDK; unavailable packages are an explicit skip."""
import subprocess
import sys
command = ['uv', 'run', '--no-project', '--offline', '--with', 'openai==3.26.1', 'python', '-I']
probe = subprocess.run([*command, '-c', 'import openai; assert openai.__version__ == "3.26.1"'], capture_output=True, text=True)
if probe.returncode:
    print('SKIP: pinned offline SDK unavailable')
    raise SystemExit(77)
raise SystemExit(subprocess.call([*command, *sys.argv[1:]]))
