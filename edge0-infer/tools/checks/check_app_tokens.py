"""Report literal colors in example App UI code outside each App's design-token file."""
from pathlib import Path
import re
import sys

# (directory, suffixes, token file) per App; paths are relative to the repository root.
SCANS = (
    ('examples/desktop/src/', ('.ts', '.tsx', '.css'), 'examples/desktop/src/theme.css'),
    ('examples/android/app/src/main/kotlin/', ('.kt',), None),
    ('examples/ios/App/', ('.swift',), 'examples/ios/App/Tokens.swift'),
)
ANDROID_TOKENS = 'ui/theme/Tokens.kt'
GENERATED = {'node_modules', 'dist', 'build', '.gradle', '.build'}

RULES = (
    ('hex-color', re.compile(r'(?<![\w&#])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![\w-])')),
    ('css-color-function', re.compile(r'\b(?:rgba?|hsla?)\s*\(')),
    ('compose-color-literal', re.compile(r'\bColor\s*\(\s*0x')),
    ('swift-color-components', re.compile(r'\b(?:UI|NS)?Color\s*\(\s*red\s*:')),
)


def skipped(relative, token_file):
    name = relative.rsplit('/', 1)[-1]
    if relative == token_file or relative.endswith('/' + ANDROID_TOKENS):
        return True
    # Unit tests may assert on literal token values.
    return re.search(r'\.test\.tsx?$|\.spec\.tsx?$|Tests?\.(kt|swift)$', name) is not None


def check(root):
    errors = []
    for directory, suffixes, token_file in SCANS:
        base = root / directory
        if not base.is_dir():
            continue
        for file in sorted(base.rglob('*')):
            relative = file.relative_to(root).as_posix()
            if (not file.is_file() or file.is_symlink() or not file.name.endswith(suffixes)
                    or GENERATED & set(file.relative_to(base).parts) or skipped(relative, token_file)):
                continue
            for number, line in enumerate(file.read_text(encoding='utf-8').splitlines(), 1):
                for rule, pattern in RULES:
                    for match in pattern.finditer(line):
                        errors.append(f'app-tokens: {relative}:{number}: {rule}: {match[0]!r}; use the design-token file')
    print('\n'.join(errors) if errors else 'app-tokens: passed')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()))
