# POST /v1/vram: what it does and what it answers (opt-in, one NVIDIA GPU, #533, #1034)

This page is written from `serve/server.py` and the engine's `VRAM` command (`src/program/generate.cpp`,
`src/core/expert_cache.cpp`), and the cases the unit tests in `serve/test_vram.py` pin. A line marked **(code)** is
read from the code and has no test of its own; treat it as a statement of intent, not a promise from a measurement.
The measurements are in [DETAILS.md](DETAILS.md) (RTX 5070, Q2_0).

## What it is

The expert cache is normally one big `cudaMalloc`. With `"vram_elastic": true` in the config (engine flag
`--vram-elastic`, segment size `"vram_segment_mib"`, default 512, at least 64) it is allocated in segments instead.
`POST /v1/vram` then keeps a number of MiB of VRAM free for another program by unmapping the last segments, or maps
them again when more is free. Nothing resizes on its own. Only the expert cache moves: the dense weights, the K/V, the
MTP head, the CUDA context and the prompt buffers stay, so Strata never gives back "everything".

The engine switches the feature off at start, with one line in its log (`--vram-elastic is off: ...`), when it is not
a `--serve` engine, with a layer split, with helper caches on other GPUs, with a segment below 64 MiB, and on HIP
(NVIDIA only). If the driver offers no virtual memory management the start fails with an error naming it **(code)**.
Without the flag nothing changes: the same answers as without it.

## The request

```
POST /v1/vram
Content-Type: application/json          (anything else: 415)
{"reserve_mib": 8000}                    (a whole number of MiB, 0 or more; null = the reserve the engine started with)
```

- The server's API key applies as for the other `/v1` routes. A request that carries a foreign `Origin` is refused
  (403) unless the origin is in `trusted_origins`.
- `reserve_mib` is the free VRAM you want left on the card after the call (it counts what any program holds, not only
  Strata). Below 0, a string, a boolean or a fraction: 400.
- `null` and `0` are the grow-back requests: `null` keeps the `--vram-reserve-mib` the engine started with, `0` keeps
  nothing free on purpose.

## What comes back

| Case | Status | Body |
|---|---|---|
| Done (or nothing to do) | 200 | `{"status": "ok", "reserve_mib", "expert_slots", "expert_slots_full", "expert_cache_mib", "expert_cache_full_mib", "vram_free_mib", "prompt_chunk"}` |
| The model is unloaded | 200 | `{"status": "not loaded", "reserve_mib": N, "note": "applied when the model loads"}` |
| Bad `reserve_mib` | 400 | `invalid_request_error` |
| The engine refuses the command | 400 | `invalid_request_error`, the engine's reason in `message`. The engine stays up. |
| Not from Strata's own page, or wrong content type | 403 / 415 | |
| A request is still running or queued (waited up to 300 s) | 409 | `model_busy` |
| The engine ended, or did not answer in 120 s | 503 | `server_error` |

**Partial success is a 200.** The JSON does not say the reserve was missed; compare `vram_free_mib` with your
`reserve_mib` (and `expert_cache_mib` with `expert_cache_full_mib`):

- **Shrink.** The engine frees the shortfall, but never goes below its floor: 128 slots plus the smallest prompt-path
  loan (a 256-token chunk). At the floor the call still answers 200 with `vram_free_mib` below the reserve. The log
  line says `(the cache keeps its smallest size: the prompt path's buffers)`. Segments are given back whole, and the
  engine keeps enough segments to cover what must stay, so the free VRAM can end a segment (512 MiB by default)
  short of the reserve **(code)**.
- **Grow.** It maps segments until the VRAM that is free beyond the reserve is used, or the cache is full. If the driver
  has no VRAM for the next segment the cache stays smaller and the call still answers 200; the reason is only in the
  server window (`(the driver has no VRAM for another expert-cache segment)`).
- The figures are measured after the call (`cudaMemGetInfo`), so another program that allocates a moment later is not in them.

## When the engine says no (400, engine stays up)

- `VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)`.
- `VRAM: not with the resident low-RAM mode` and `VRAM: not with --peer-device`.
- `VRAM: not while batch slots are decoding` (with `--parallel`/`--batch`; the server also takes the engine's control
  lock first, so a prompt that is being admitted finishes before the resize).
- `VRAM: a prompt's loan is still out`.
- A device error before the resize: `VRAM: <cuda error>`.

## Failure inside a resize **(code)**

Two failures can happen after the cache has changed. In both the engine answers `ERR` (HTTP 400) and keeps serving,
but I would not trust its cache table any more:

- the driver refuses to release a segment (`the driver refused to release an expert-cache segment`). The slot count
  is already updated; the residency table of the experts is not repaired in this path;
- refilling a slot on the way back fails (`VRAM: refilling the cache failed: ...`). Segments are mapped, some experts
  are not back in them.

The safe response to either is to unload and load the model (`POST /v1/unload`, then `POST /v1/load`), which starts a
fresh engine. Wrong answers are not expected from a smaller cache (a missing expert is computed on the CPU, as for any
expert outside the cache), but I have not run a fault-injection test of these two paths.

## Giving the VRAM back and taking it again

1. `{"reserve_mib": 8000}`: the cache shrinks until 8000 MiB are free (or to its floor). The experts of the segments
   given back are computed on the CPU, so answers keep coming, slower (RTX 5070, Q2_0: decode 44 to 33 tok/s with 4.3
   GiB given back; the call took 78 ms).
2. The other program runs and ends.
3. `{"reserve_mib": null}` (or a smaller number): the cache grows back as far as the free VRAM allows. Growing back puts
   the same experts in the same slots (token for token the answers from before the shrink in the DETAILS measurement);
   where the adaptive tier already brought an expert back, the layer's most-routed missing expert takes the slot **(code)**.
   Check `expert_cache_mib` against `expert_cache_full_mib` to know it is whole again.

## Unload, restart and crash

- The server remembers the last `reserve_mib` **in memory** and sends it again after the engine has started again
  (idle unload, `POST /load`, or a restart after a crash). The engine sizes its cache at that start from what is free
  *then*, and the reserve is applied only after it is up. If it fails, the server window says
  `the VRAM reserve (N MiB) was not applied: ...` and the engine runs with the cache it got.
- If another program holds the VRAM at load time, `min_free_vram_mib` (config) decides: the load waits up to 15 s for
  that much to be free, then answers 503 `the GPU is in use by another program` and the model stays unloaded.
- A server restart forgets the reserve: the engine starts with its full cache.
- Calling it on an unloaded model changes nothing on the GPU; it only sets the reserve for the next load.
- `GET /v1/status` has a `vram` object with the last figures the engine reported (empty until the first successful
  call) and `elastic`.

## The image encoder on demand

With `"elastic": true` in the config's `"vision"` section (and `"vram_elastic": true`; without it the server says so
and starts the encoder with the model, as before) the GPU encoder is not started with the server or with a load:

- **A picture's turn** (the same FIFO turn an encode always takes): if the encoder is not running, the engine's
  `VRAM` command first keeps `"elastic_mib"` (default 2048) free on top of the reserve - the last `POST /v1/vram`, or 0
  when none was made - and then the encoder starts. If the cache cannot shrink (the resident low-RAM mode, an engine
  without `--vram-elastic`, a layer split) it starts anyway, as a non-elastic encoder would, and the log says why; on a
  layer split the server says at start that the encoder's VRAM has to be kept free with `--vram-reserve-mib`. An
  encoder that fails to start (out of VRAM) ends the request with a `503`; the next picture tries again.
- **Idle** for `"elastic_idle_s"` (default 120) without a picture: a thread stops the encoder between requests and
  sends `VRAM` with that reserve again, so the cache grows back to the size it had (0, not the start's reserve: that
  would leave it a segment or more smaller).
- **`POST /v1/vram` while the encoder runs** keeps the asked reserve on top of the encoder's share; the encoder's stop
  then goes back to the asked reserve.
- An unload of the model stops the encoder too. The log lines start with `[strata] images:` **(code; the unit tests in
  `serve/test_vision_elastic.py` pin each case)**.

`--vram-elastic` turns `--kv-grow` off (the engine says so at start): with both asked, the whole K/V is in VRAM.

## What to build on

For automation: treat only `200` with `status: "ok"` as a call that ran, then check `vram_free_mib >= reserve_mib`
yourself; retry a `409` later; on `503` or a `400` from the failure paths above, unload and load; keep your own record
of what you asked for, because a server restart forgets it.
