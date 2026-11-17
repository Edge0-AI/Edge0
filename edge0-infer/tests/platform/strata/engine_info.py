"""Precondition check for a measurement on a running edge0-strata: the size of its GPU expert cache.

Usage: engine_info.py --url http://127.0.0.1:<port> [--min-expert-cache-mib 10848]

edge0-strata sizes its expert cache (`--expert-cache auto`) from the GPU memory that is free when it loads, so a
smaller value than usual means another process held GPU memory at that moment and the run is not comparable.
Reads /metrics, /v1/status and /props (directly or through `edge0-infer serve`, which passes them through), prints
every `expert_cache_mib` and `prefill_cap` found, and exits 0 when every expert_cache_mib found is at least the
minimum, 1 when one is smaller or none is found (2 when the endpoints cannot be read). Without
--min-expert-cache-mib it only prints. A larger cache than the minimum is fine (less GPU memory in use by the
desktop, for example).
"""
import argparse
import json
import sys
import urllib.request


def find(value, names, path='', found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names and not isinstance(item, (dict, list)):
                found.append((path + key, item))
            find(item, names, path + key + '.', found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            find(item, names, f'{path}{index}.', found)
    return found


def read(url, names=('expert_cache_mib', 'prefill_cap')):
    found, errors = [], []
    for path in ('/metrics', '/v1/status', '/props'):
        try:
            with urllib.request.urlopen(url.rstrip('/') + path, timeout=30) as response:
                found += [(path + ' ' + name, value) for name, value in find(json.load(response), names)]
        except (OSError, ValueError) as error:
            errors.append(f'{path}: {error}')
    return found, errors


def check(url, minimum=None):
    """Returns (ok, report lines): ok when every expert_cache_mib found is >= minimum (always, without one)."""
    found, errors = read(url)
    lines = [f'{name} = {value}' for name, value in found] + [f'unreadable {error}' for error in errors]
    caches = {value for name, value in found if name.endswith('expert_cache_mib')}
    if minimum is None:
        return True, lines
    ok = bool(caches) and min(caches) >= minimum
    lines.append(f'expert_cache_mib {sorted(caches)} {"is at least" if ok else "is missing or BELOW"} the minimum {minimum}')
    return ok, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', required=True)
    parser.add_argument('--min-expert-cache-mib', type=int)
    args = parser.parse_args()
    ok, lines = check(args.url, args.min_expert_cache_mib)
    print('\n'.join(lines))
    if args.min_expert_cache_mib is None and (not lines or all(line.startswith('unreadable') for line in lines)):
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
