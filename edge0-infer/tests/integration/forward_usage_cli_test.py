"""The supported split-last mode is discoverable from forward usage."""
import subprocess
import sys
result = subprocess.run([sys.argv[1], 'forward'], capture_output=True, text=True)
assert result.returncode != 0 and '--split-last' in result.stderr, result.stderr
