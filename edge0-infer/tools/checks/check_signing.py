"""Reject checked-in signing identities and generated Apple project artifacts."""
import os
from pathlib import Path
import re
import subprocess
import sys

KEY = r'(?:DEVELOPMENT_TEAM|PRODUCT_BUNDLE_IDENTIFIER|CODE_SIGN_IDENTITY)'
ASSIGNMENT = re.compile(KEY + r'[\s"\']*(?:[:=]\s*|\s+)["\']?([^\n;"\']+)')
GENERATED = {'.xcodeproj', '.xcworkspace', '.mobileprovision'}


def inspect(path, text, secrets=()):
    errors = []
    if any(Path(part).suffix in GENERATED or part == 'xcuserdata' for part in Path(path).parts):
        errors.append(f'signing: {path}:1: generated-signing-artifact')
    for number, line in enumerate(text.splitlines(), 1):
        for match in ASSIGNMENT.finditer(line):
            value = match[1].strip()
            if value == 'Apple Development':
                continue  # Generic certificate class, not a developer identity.
            if value and not any(marker in value for marker in ('$', '<', 'PLACEHOLDER', 'YOUR_', 'ENV', 'os.environ')):
                errors.append(f'signing: {path}:{number}: literal-signing-value')
        if re.search(r'(?:TeamIdentifier|Team ID|teamID)\s*[:=]\s*[A-Z0-9]{10}\b', line):
            errors.append(f'signing: {path}:{number}: literal-team-identity')
        if re.search(r'(?:Signing Identity|Apple Development:)\s*.*[\w.+-]+@[\w.-]+', line):
            errors.append(f'signing: {path}:{number}: signing-account')
        if re.search(r'\bdev\.(?:edge0|moe)\.[A-Z0-9]{10}\.edge0-(?:infer|chat)\b', line):
            errors.append(f'signing: {path}:{number}: literal-bundle-identity')
        if re.search(r'\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{16}\b', line):
            errors.append(f'signing: {path}:{number}: device-identifier')
        if any(secret and secret in line for secret in secrets):
            errors.append(f'signing: {path}:{number}: configured-private-identity')
    return errors


def check(root):
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            capture_output=True, check=True)
    secrets = [os.environ.get(key, '') for key in ('MOE_IOS_TEAM', 'MOE_IOS_BUNDLE_ID', 'MOE_IOS_DEVICE')]
    errors = []
    for path in sorted(set(os.fsdecode(result.stdout).split('\0')) - {''}):
        file = root / path
        if file.is_symlink() or not file.is_file() or path == 'AGENTS.local.md':
            continue
        # Binary signing artifacts must be rejected without decoding their contents.
        if any(Path(part).suffix in GENERATED or part == 'xcuserdata' for part in Path(path).parts):
            errors += inspect(path, '')
            continue
        with file.open('rb') as stream:
            if b'\0' in stream.read(8192):
                continue
        try:
            text = file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        errors += inspect(path, text, secrets)
    print('\n'.join(errors) if errors else 'signing: passed')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(check(Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()))
