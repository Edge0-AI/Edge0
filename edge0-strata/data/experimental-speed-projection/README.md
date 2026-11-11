# experimental-speed-projection (EXPERIMENTAL, off by default)

`Qwen3.8-Flash-Next-experimental-speed-projection.gguf` is a 480 KB control vector for Qwen3.8-Flash-Next: one unit
direction of 2,560 values per layer (`direction.1` .. `direction.47`, GGUF `controlvector` format). The JSON beside it
is the configuration its authors selected and their scores.

**What it does.** Turned on, Strata removes each direction from the residual stream after layers 4-44
(`h -= (h . v) v` on every hyper-connection stream), the same operation as llama.cpp's `--cvec-mode project` in the
package's patches. The package's own documentation describes the vector as a **refusal-direction projection**: the
model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and removing refusals removes a
safety behaviour - you are responsible for what the model writes with it on. It also shifts the model's predictions
slightly on ordinary text (measured here: `bench/results/2026-09-27-esp/`). It is not an optimization in the engine:
on the same text it costs 0.2-0.4% per token; a chat's speed with it on depends on the text the model writes.

**Turning it on.** `START-HERE.bat --setup` asks (default: off), or pass `--experimental-speed-projection on`. With it
loaded, the web app's Sampling drawer and the API field `"experimental_speed_projection": false` switch it off per
request. See `docs/DETAILS.md`, "Experimental speed projection".

**Origin and license.** A local package of the vector (the published original uses filenames containing "refusal";
the bytes are the same), made from Qwen3.8-Flash-Next activations, so the Qwen Community License 1.0 of the model
applies to it. Made for the original Qwen3.8-Flash-Next (not Swift 1.5).
