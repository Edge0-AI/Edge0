#!/usr/bin/env python3
"""Build a server config: the current install config plus extra args/env.

The same builder scripts/ab-run.sh uses inline, factored out so the report's own
configs (the soak's, interleave500's B arm) can be reproduced from the script.

Usage:
  build_cfg.py OUT [--arg NAME VALUE]... [--env NAME=VALUE]...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'strata-coder-iq1_m.json'


def main():
    out = sys.argv[1]
    extras = sys.argv[2:]
    cfg = json.loads(BASE.read_text())
    args = list(cfg['args'])
    env = dict(cfg.get('env') or {})
    i = 0
    while i < len(extras):
        if extras[i] == '--env':
            k, _, v = extras[i + 1].partition('=')
            env[k] = v
            i += 2
        elif extras[i] == '--arg':
            args += [extras[i + 1], extras[i + 2]]
            i += 3
        else:
            raise SystemExit(f'unknown extra {extras[i]!r}')
    cfg['args'] = args
    if env:
        cfg['env'] = env
    Path(out).write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'{out}: +args {extras}')


if __name__ == '__main__':
    main()
