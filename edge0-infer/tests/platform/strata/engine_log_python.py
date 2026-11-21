#!/usr/bin/python3
"""A Python for `edge0-infer serve --strata-python` that keeps edge0-strata's engine log.

`edge0-infer serve` hands edge0-strata's server a config without a "log" key, so the engine's own stderr (the
Metal engine's "strata metal: ..." lines) goes nowhere. Pass this file as --strata-python: it adds
"log": $EDGE0_TEST_ENGINE_LOG to the config named after --config, then runs the real Python
($EDGE0_TEST_REAL_PYTHON, default: .venv/bin/python of the edge0-strata directory beside the server script) with
the same arguments. EDGE0_TEST_CONFIG_ENV ("NAME=value,NAME=value") also sets those variables in the config's
"env", which edge0-strata's server gives the engine over its own environment: a variable the template sets (such
as STRATA_METAL_PROMPT_CHUNK) can only be changed there. Nothing else changes. Without both variables the config is
left as it is. With EDGE0_TEST_ENGINE_LOG the final configuration is also written as config.json beside the log.
"""
import json
import os
import sys


def main():
    argv = sys.argv[1:]
    log = os.environ.get('EDGE0_TEST_ENGINE_LOG')
    extra = [item.split('=', 1) for item in os.environ.get('EDGE0_TEST_CONFIG_ENV', '').split(',') if '=' in item]
    if (log or extra) and '--config' in argv:
        path = argv[argv.index('--config') + 1]
        with open(path, encoding='utf-8') as f:
            config = json.load(f)
        if log:
            config['log'] = os.path.abspath(log)
        config.setdefault('env', {}).update({k: v for k, v in extra})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        if log:   # the configuration the server runs with, beside the engine log (for the record)
            with open(os.path.join(os.path.dirname(os.path.abspath(log)), 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
    real = os.environ.get('EDGE0_TEST_REAL_PYTHON')
    if not real:
        server = next((a for a in argv if a.endswith(os.path.join('serve', 'server.py'))), None)
        if server is None:
            sys.exit('engine_log_python.py: set EDGE0_TEST_REAL_PYTHON (no serve/server.py among the arguments)')
        real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(server))), '.venv', 'bin', 'python')
    os.execv(real, [real] + argv)


if __name__ == '__main__':
    main()
