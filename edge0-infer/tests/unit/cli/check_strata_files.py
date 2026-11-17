"""Keep each delegated model's file list in the registry equal to the model directory table on its user documentation page
(edge0_145b: docs/edge0-145b.md, edge0_120b: docs/edge0-120b.md).

Not checked: an older table written differently from the one with the expected header, and rows that continue the
table after a blank line; both need unusual documentation that readers see once it is rendered."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
docs = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'docs'
source = (root / 'tools' / 'cli' / 'serve_delegate_registry.cpp').read_text()
# Drop // and /* */ comments (string literals are kept as they are), so a commented-out entry does not count.
source = re.sub(r'("(?:\\.|[^"\\\n])*")|//[^\n]*|/\*.*?\*/', lambda m: m.group(1) or ' ', source, flags=re.S)

# Registry: read each model's literal file list independently.
errors = []
registries = {}
for model in ('edge0_145b', 'edge0_120b'):
    marker = '{"' + model + '"'
    if source.count(marker) != 1:
        errors.append(f'{model}: expected exactly one registration')
        continue
    start = source.index(marker)
    files_open = source.index('{{', start)
    depth, groups = 0, []
    for index in range(files_open, len(source)):
        if source[index] == '{':
            depth += 1
            if depth == 2:
                group_start = index
        elif source[index] == '}':
            depth -= 1
            if depth == 1:
                groups.append(source[group_start:index + 1])
            elif depth == 0:
                break
    entries = [m.groups() for m in (re.fullmatch(r'\{"([^" ]+)",\s*(true|false),\s*"([^" ]*)"\}', g.strip()) for g in groups) if m]
    if len(entries) != len(groups):
        errors.append(f'{model}: file entries must be literal name, directory, condition triples')
    registry = {}
    for name, directory, condition in entries:
        key = name + ('/' if directory == 'true' else '')
        if key in registry:
            errors.append(f'{model}: duplicate file {key}')
        registry[key] = condition
    if not registry:
        errors.append(f'{model}: no registry entries found')
    registries[model] = registry

# Documentation: the table whose header is "| Path | Content | Required |" on each model's page.
HEADER = '| Path | Content | Required |'


def read_table(doc):
    lines = doc.read_text().splitlines()
    headers = [number for number, line in enumerate(lines) if line == HEADER]
    if len(headers) != 1:
        errors.append(f'{doc.name}: the model directory table header must appear exactly once, found {len(headers)} '
                      f'(expected header, exactly: {HEADER})')
        return None
    table = {}
    for line in lines[headers[0] + 2:]:
        if not line.startswith('|'):
            break
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        required = cells[-1]
        if required == 'yes':
            condition = ''
        elif required == 'for Q3_K folders':
            condition = 'experts=q3k'
        else:
            match = re.fullmatch(r'for `([a-z-]+=[a-z0-9]+)`', required)
            if not match:
                errors.append(f'{doc.name}: unrecognised Required cell: {required}')
                continue
            condition = match.group(1)
        for name in re.findall(r'`([^`]+)`', cells[0]):
            if name in table:
                errors.append(f'{doc.name}: {name} is listed more than once')
            table[name] = condition
    return table


# Compare every registration with its own page, so divergence of only one model cannot pass unnoticed.
for model, registry in registries.items():
    doc = docs / (model.replace('_', '-') + '.md')
    if not doc.is_file():
        errors.append(f'{model}: {doc.name} not found')
        continue
    table = read_table(doc)
    for name in sorted(registry.keys() | table.keys()) if table is not None else []:
        if name not in table:
            errors.append(f'{model}: {name}: missing from {doc.name}')
        elif name not in registry:
            errors.append(f'{model}: {name}: missing from the registry')
        elif registry[name] != table[name]:
            errors.append(f'{model}: {name}: condition differs: {registry[name]} vs {table[name]}')
print('\n'.join(errors) if errors else f'strata file list: passed ({len(registries)} models)')
sys.exit(bool(errors))
