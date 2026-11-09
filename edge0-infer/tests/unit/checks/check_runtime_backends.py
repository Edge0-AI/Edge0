"""Keep runtime mechanisms independent of concrete backends."""
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
pattern = re.compile(r'metal|cuda|vulkan|ggml_backend_is_|ggml-cpu\.h|ggml-metal\.h', re.I)
errors = []
for module in ('exec', 'weights', 'session'):
    for path in (root / 'src' / module).rglob('*'):
        if path.is_file():
            for line, text in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(text):
                    errors.append(f'{path.relative_to(root)}:{line}: concrete backend reference')
print('\n'.join(errors) if errors else 'runtime backend independence: passed')
sys.exit(bool(errors))
