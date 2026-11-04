# Testing

The tests are CTest tests. Each one carries a level label, `L0` to `L4`, ordered from cheapest to most expensive; tests that need a GPU or other device also carry the label `device`. Every property is tested at the cheapest level that can decide it.

## Levels

| Level | What it checks | Needs |
| --- | --- | --- |
| `L0` static | Public headers compile on their own, language and documentation checks, dependency and layering rules, vendored source checks. Runs in a few seconds. | Nothing |
| `L1` unit | Pure logic: tokenizer, prefix matching, sampling, slot selection, CLI parsing, the asset and measurement tools | The Python environment of the asset tools |
| `L2` integration | Several modules together on a tiny random-weight model with the same structure as the 35B; single-turn computation is compared with the reference answers in [`tests/fixtures/answers/`](../tests/fixtures/answers/README.md) | `$MOE_TINY`; `$MOE_SOURCE_35B` for the asset tools' fixture test |
| `L3` real assets | The 35B on real assets: logits, routing, expert loads, memory peaks | `$MOE_ASSETS_35B` |
| `L4` acceptance | Fixed scenarios through the CLI only, per platform | `$MOE_ASSETS_35B`, the target device |

`L3`, `L4` and `device` tests run one at a time even when CTest runs others in parallel, because they assert memory peaks or share the GPU. A test that cannot run in the current environment, such as a Metal test without a Metal device or a check that needs Git metadata, exits with code 77 and is reported as skipped, not as passed.

## Setup

1. Create the Python environment that the Python tests use (it lives in the ignored `tools/assets/.venv`):

   ```sh
   uv sync --project tools/assets
   ```

2. For `L2`, generate the tiny fixtures as described in the [asset tools guide](../tools/assets/README.md#tiny-fixtures) and export their directory as `MOE_TINY`. This needs only three small text files of the 35B source, not the full weights; keep `MOE_SOURCE_35B` exported: the asset tools' fixture test needs it and is otherwise skipped.
3. For `L3` and `L4`, convert the 35B as described in [Getting started](getting-started.md#get-the-model-assets) and export `MOE_ASSETS_35B`.

## Running

```sh
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"                                    # no assets needed
ctest --preset cpu --no-tests=error -L "L0|L1|L2" -LE device     # with $MOE_TINY

cmake --preset mac-metal && cmake --build --preset mac-metal
ctest --preset mac-metal --no-tests=error -L "device|L3"         # Metal and real assets
```

These commands, with both presets built, are the full regression on macOS. Tests are registered only when `MOE_BUILD_TESTS` is ON, the default when this repository is the top-level project; it defaults to OFF when the repository is [added to another CMake project](getting-started.md#use-from-another-cmake-project). `ctest -N -L <level>` lists the tests of a level without running them.

`tools/run_checks.py` configures, builds and runs the chosen levels in one step and lists every test as passed, failed or not executed, with the reason:

```sh
uv run --no-project python -I tools/run_checks.py --preset cpu --levels L0,L1,L2
```

## Platform and performance tests

- [Platform CLI scenarios](../tests/platform/README.md) run the `L4` scenarios on Android and iOS devices, through wrappers that stand in for `edge0-infer` on the host, and on Windows directly.
- [Measurement tools](../tests/perf/README.md) produce the performance reports described in [Benchmarking](benchmarking.md); measurements are not part of the test levels.
