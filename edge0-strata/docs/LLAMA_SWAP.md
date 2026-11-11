# Running Strata behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts several model servers behind one address, starts the one a
request names and stops the others to free the GPU. This page is for people who already run llama-swap and want
Strata as one of its models. It needs a [source checkout](SOURCE_INSTALL.md), not the desktop installer.

## Quick start

1. Set up Strata without starting it: `./setup.sh --no-start`. This writes `strata-<model>.json` in the checkout.
2. Add an entry to llama-swap's config. The start script that setup writes (`run-<model>.sh`) fixes the port at 8080,
   so call `serve/server.py` with llama-swap's port instead:

```yaml
healthCheckTimeout: 300            # model loading takes minutes; give it room

models:
  strata:
    cmd: >-
      sh -c 'cd /path/to/Strata && exec .venv/bin/python serve/server.py
      --engine strata --config strata-<model>.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # the host name your clients use, see below
    checkEndpoint: /health
    ttl: 1800
```

3. If clients reach llama-swap by a host name, keep the `STRATA_ALLOWED_HOSTS` line and set it to that name
   ([Host not allowed](#host-is-not-allowed)).

## The entry

- `exec` makes the Python server the process llama-swap stops. On SIGTERM the server stops its engine and the
  GPU memory is freed.
- `checkEndpoint: /health` works because Strata's `/health` answers only once the model is loaded.
- Strata needs the card's memory to itself. If other models share the GPU, put them in a llama-swap group or let
  swapping unload them.

## Host is not allowed

A client that reaches the proxy by name (`http://my-server:8040`) gets:

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

Strata only answers to the host names it knows, so a web page cannot reach it by pointing a DNS name at 127.0.0.1.
llama-swap passes the client's Host header through. Requests by IP address and to `localhost` always pass, so a quick
local test can work while the same setup fails from another machine. Any one of these fixes it:

| Fix | Notes |
| --- | --- |
| `STRATA_ALLOWED_HOSTS=my-server` in the entry's `env` | Comma-separated names; ports are ignored. |
| `"allowed_hosts": ["my-server"]` in `strata-<model>.json` | Same effect, stored in the config. |
| An `"api_key"` in the config | Turns the host check off. Every client must then send the key. |

## Strata's own pages through llama-swap

llama-swap forwards `/upstream/<model>/...` unchanged. Strata's status and health pages stay reachable, and calling one
also loads the model without sending a chat request:

```bash
curl http://my-server:8040/upstream/strata/health
curl http://my-server:8040/upstream/strata/v1/status
```

## Claude Code and other Anthropic-API clients

Claude Code and the Claude Agent SDK send `"output_config": {"effort": "high"}` with every request unless told
otherwise, so Strata thinks at its highest level. Set `CLAUDE_CODE_EFFORT_LEVEL=medium` (or `low`) in the client's
environment to change it.

## History / older measurements

> **Measured on:** Tesla V100, Coder IQ1_M, Strata 0.1.39, llama-swap default stop behaviour, model files on a SATA SSD.

| Item | Result |
| --- | --- |
| Model load time | 80 s; 105 s while another download used the disk |
| Engine gone and V100 memory freed after llama-swap's stop | within 2 s |

> **Measured on:** one real coding task, Strata 0.1.39 setup, one run per level. This is not a general result.

| `CLAUDE_CODE_EFFORT_LEVEL` | Time to a correct result |
| --- | --- |
| `medium` | 6.5 min |
| `high` | 9.7 min |
| `low` | 10.1 min (more trial-and-fix steps) |
