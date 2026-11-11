# Model asset tools

These tools turn a model's original weights into an **asset directory** the engine can load: GGUF files plus three description files (`assets.tsv` lists every file with its size and sha256, `model.json` names the model, `provenance.json` records how the assets were made). The flow is the same for every model; only the tools under `tools/assets/<model>/` differ. Today there is one model, `edge0_35b`, and the examples below use it.

The engine has no model flag: it reads `architecture` from `model.json` and picks the matching model compiled into the program. `edge0-infer info` prints one `model.<n>: <id>` line per compiled model.

## Setup

Run every command from the engine's top-level directory (`edge0-infer/`). Create the Python environment once:

```sh
uv sync --project tools/assets
```

The commands below run scripts with `uv run --project tools/assets --offline --no-sync python -I`, shortened here to a shell function (bash or zsh):

```sh
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }
```

Tests and examples read three directories from environment variables; set them to wherever you keep the data (weights never go into Git):

| Variable | Directory |
| --- | --- |
| `MOE_SOURCE_35B` | original 35B weights (downloaded by `fetch_source.py`) |
| `MOE_ASSETS_35B` | converted 35B assets (made by `make_assets.py`) |
| `MOE_TINY` | tiny fixtures `tiny-a` and `tiny-b` (see below) |

## Downloading and converting the 35B

[Getting started](../../docs/getting-started.md#get-a-model) walks through probing the source, downloading, converting and checking the 35B, then running it. `check` trusts the catalog, so data that was changed together with its digest still passes. To prove that assets are a faithful re-make or untouched, use `compare` against another trusted copy: it compares tensor by tensor and should report every file equal except `provenance.json`.

```sh
assets_py tools/assets/check_assets.py compare "$MOE_ASSETS_35B" <trusted-assets>
```

## Tiny fixtures

The tiny fixture is a small random-weight model with the same structure as the 35B. Integration tests (level L2) use it, so they run without real weights. It is generated from three text files of the 35B source (`tokenizer.json`, `chat_template.jinja`, `generation_config.json`, about 20 MB together); `tiny-b` is a second generation that the tests compare with `tiny-a` to prove the generator is deterministic.

Without the full original weights, download only those three files. A partial download cannot later be completed in the same directory, so give it a directory of its own and point `MOE_SOURCE_35B` at it:

```sh
export MOE_SOURCE_35B=<new directory for the source text files>
assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B" --files tokenizer.json chat_template.jinja generation_config.json
```

Then generate both fixtures; `tiny-a` and `tiny-b` must not exist yet:

```sh
assets_py tools/assets/edge0_35b/make_tiny.py --source "$MOE_SOURCE_35B" --out "$MOE_TINY/tiny-a"
assets_py tools/assets/edge0_35b/make_tiny.py --source "$MOE_SOURCE_35B" --out "$MOE_TINY/tiny-b"
```

## Tests

The `l1` (no weights) and `l2` (tiny fixture) tests run under ctest. The `l2` test checks `tiny-a` and `tiny-b` against `edge0_35b/tiny.lock.json` and that the generator refuses to overwrite `tiny-a`; it needs `$MOE_TINY` and an existing `$MOE_SOURCE_35B` directory, and without either it is reported as skipped. The `l3` tests need the network or the real weights and run by hand, with the three variables above set.

```sh
ctest --preset cpu --no-tests=error -R "^moe_test_asset_tools_" --output-on-failure
uv run --project tools/assets --offline --no-sync pytest -q -m l3 tools/assets/tests
```

## Adding a model

Write the model's tools in `tools/assets/<model>/`: a fetch script with its source lock, a converter, and a `check.py` with the model's roles and tensor checks. Then register it in `MODELS` in `check_assets.py`. A directory whose `model.json` names an unregistered model fails the check with `unsupported_model`, listing the registered ids. The steps above stay the same; only the script paths change.
