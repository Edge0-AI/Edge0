"""Model loading must not build or execute graphs (T11)."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pattern = re.compile(r"create_graph|build_plan|submit|collect_completion|graph_compute")
errors = []
for path in sorted((root / "src/model").rglob("*")):
    if path.is_file():
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(text):
                errors.append(f"{path.relative_to(root)}:{line}: model computation forbidden")
print("\n".join(errors) if errors else "model no compute: passed (0 matches)")
sys.exit(bool(errors))
