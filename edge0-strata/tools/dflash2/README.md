# DFlash2 acceptance experiment

Research tooling, not shipped with the app. It measures how many tokens per round the incoai DFlash2 drafter for
Qwen3.8-Flash-Next would accept on Strata's quantized target. The measurement is offline, from hidden states the
engine dumps. It is for developers who want to repeat or extend the experiment.

The scripts keep the paths of the original run (`C:\dflash2-work` on the development PC, `C:\strata-test\dflash2` on
the test box). Edit the paths before you reuse them. The method and results are in the project's internal notes
and are not part of this tree.

## Engine variables

Both are off unless set.

| Variable | Effect |
| --- | --- |
| `STRATA_TAP_FILE=<file>` | Writes `pos, token, taps` records. `Verifier::record_window` copies each verified row's completed layer output (the mean of the 4 hyper-connection streams, the quantity the drafter was trained on) and `Verifier::run` appends it. |
| `STRATA_TAP_LAYERS` | Layers to tap, default `5,15,25,35,45`. Used with `STRATA_TAP_FILE`. |
| `STRATA_SPEC_T_CYCLE=1,2,...,8` | CLI bench only. The verify window size cycles; the report line `window cost (ms/round)` gives the cost of each width. |

## Scripts

| Script | Purpose |
| --- | --- |
| `mk_prompts.py` | Builds chat-template prompt ids with the server's own tokenizer and template. |
| `orch.py <prompt> <max_new> [gen,genf,genfm,tap]` | Drives `run_one.ps1` on the test box: free generation (ids and MTP statistics), the same text with MTP only, and the teacher-forced tap dump with oracle drafts (`--spec 8 --spec-oracle --spec-follow`, a 2-token prompt, because a native pack never starts the speculative loop from a 1-token prompt). |
| `dflash2_ref.py` | Plain-torch port of z-lab's `DFlash2DraftModel`, with the target's embedding and head dequantized from the GGUF. `test_port.py` checks it against the original. |
| `run_eval.py` | Walks the tap file with a few block sizes. |
| `run_all2.py` | Evaluates every position, resumable: per-anchor correctness and the selector's confidences. |
| `analyze2.py`, `final_tables.py` | Policies, and speed estimates from the measured window costs. |
