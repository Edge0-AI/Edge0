# What this folder leaves out

Following the convention of the other community folders (raw dumps, logs and binaries are trimmed),
this folder keeps every number's source but not every byte that produced it. Left out:

- **The engine binaries** (`build-906/strata`, 25 MB each for 0.1.40.4 and 0.1.41) and the CMake build
  trees. Their sha256 and the exact build commands are in [BUILD.json](BUILD.json); the full build logs
  are kept (`build-0.1.40.4.log`, `build-0.1.41.log`) because they carry the compiler's verdict on the
  host patches.
- **The per-request raw dumps** the harness writes while it runs: `<label>-request.json` and
  `<label>-raw.json` for each of the 10 requests of each sweep (the rendered prompt, every SSE chunk,
  and the collected text), about 3.5 MB per sweep. The numbers computed from them are in
  `results-*.json` (per run) and `summary-*.json` (medians and ranges); the request bodies are
  reproducible byte-for-byte from the script's nonce scheme.
- **The model files** (39.2 GB + 28.8 GB). Filenames, byte sizes and sha256 are in
  [model-sha256.txt](model-sha256.txt) and [provenance.json](provenance.json); both match the
  repository's published LFS oids.
- **The expert pack, MTP draft layer and tokenizer** (~1.5 GB + ~0.9 GB). The commands that built them
  are in [provenance.json](provenance.json).

Kept on purpose (because the maintainers read them): the **complete engine logs** of both rounds —
`engine-0.1.40.4.log` and `engine-0.1.41.log` — including the startup decision lines (expert cache
sizing, PCIe probe, prefill chunk, KV streaming) and every request's `cache hit_rate`.

The first sweep of each version ran with a telemetry sampler that had a parsing bug: the per-second
GPU readings came out empty (that sampler file is left out). The
second sweep (`-b`) ran with the fixed sampler and its telemetry is the one quoted in the README.
