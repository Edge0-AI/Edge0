# The idle GPUs (proposal 2507-0): what the existing switches do on NVIDIA

Engine: the released 0.1.46 (Strata 0.2.1, `a7bdbfdabc9a`, cuda12 Linux build), no engine change. Model: Qwen3.8-Flash-Next IQ3_XXS, setup-like flags
(`--expert-cache auto --prefill auto --spec 4 --kv int8`, layer split auto on z440, resident experts on P2003). Greedy, thinking off.
Harness: `tools/idle_gpu_ab.py` (the 0.2.1 split A/B harness plus an `nvidia-smi` sampler, 5 samples/s per card, over every decode, prompt and batch window).
Fresh engine per arm run, rotating arm order, 5 pairs, medians, "pairs faster" = pairs in which the arm beat the baseline, plus an A-versus-A noise arm (`default2`).
Boxes: z440 (2x RTX A4000 16 GB, Xeon E5-2643 v3, layer split 0-24 / 25-47), P2003 (RTX 3060 12 GB). Raw: `z440/`, `p2003/` (results.jsonl, one engine log per arm, summary.txt).

## How idle are they

Utilisation = `nvidia-smi` utilization.gpu (share of time some kernel is running on the card), median over runs.

| Window | z440 card 0 / card 1 | mean idle | P2003 (one card) |
|---|---|---|---|
| decode, story / code / facts (512 tokens) | 45 / 58, 45 / 58, 44 / 61 % | 47-49 % | 94-95 % (157 W of 170 W) |
| prompt read 16K | 55 / 61 % | 42 % | 96 % |
| prompt read 58K | 75 / 78 % | 24 % | not run (32K context) |
| four chats (`--batch 4`, default) | 46 / 51 % | 51 % | not run |

On the two-card split the two utilisations add up to about 100 %: the cards take turns, one is always waiting for the other, as the issue says (it measured ~40 % idle on a 4080 + 3090). The single card is not idle (95 %), so there is no gap to close on P2003.
Caveat: the figure counts a card as busy when any kernel is resident, including a kernel that spins on a hand-off flag (see `overlap` below).

## Does an existing switch close the gap (z440, default flags, decode tok/s, medians of 5 pairs)

| Switch | story | code | facts | prompt 16K | prompt 58K | card utilisation in decode |
|---|---|---|---|---|---|---|
| default | 82.1 | 97.8 | 93.6 | 1647 | 2956 | 45 / 58 % |
| `STRATA_SPLIT_OVERLAP=1` | 82.4 (+0.3 %, 2/5) | 99.8 (+2.1 %, 4/5) | 93.4 (-0.3 %, 2/5) | +0.1 % | +0.1 % | 45 / **94** % |
| `--pipeline-windows 2` | 81.3 (-1.0 %, 2/5) | 103.0 (+5.4 %, 5/5) | 94.8 (+1.2 %, 4/5) | +0.3 % | +0.2 % | 76 / 75 % |
| A versus A (noise) | -0.1 % | +2.8 % (5/5) | 0.0 % | 0.0 % | +0.1 % | 45 / 58 % |

Second interleaved set (default as baseline again; 5 pairs):

| Switch | story | code | facts | prompt 16K | prompt 58K |
|---|---|---|---|---|---|
| `--pipeline-windows 2` + `STRATA_MTP_STAGE0=1` (what the engine's note asks for) | **-13.2 %** (0/5) | -5.5 % (1/5) | -7.1 % (0/5) | -3.4 % | -4.7 % |
| `--pipeline-windows 1` | -1.4 % | -0.4 % | -1.5 % | -0.2 % | -0.2 % |
| A versus A | -0.4 % | -0.1 % | -2.1 % | -0.2 % | -0.1 % |

Four concurrent chats (`--batch 4`, total tok/s over two passes of 200 tokens, 5 pairs): default 78.0 (the 0.2.1 default on two NVIDIA stages is one serial group), `--batch-groups 2` 66.0 (**-15.3 %**, 0/5 pairs), noise -0.4 %.
Time to first token 0.19 s against 0.27 s. Card utilisation 46 / 51 % against 95 / 94 %: the pipeline keeps both cards busy and is still slower, which repeats the 0.2.1 result (serial +12 to +16 % on 2x A4000).

Reading: the `overlap` flag removes nothing from the critical path on NVIDIA (the AMD gain came from a costly sync and graph launch; here they are cheap) and its busy 94 % on card 1 is a kernel waiting for the flag. `--pipeline-windows 2` does fill both cards (75 %) but a guessed window is wasted work: +5 % on code against +2.8 % of noise, nothing on prose, and -5 to -13 % once the draft layer moves to the first card.

## Exactness (z440, the identity-gate flags `--expert-cache 1500 --adapt-every 100000 --pcie-frac 0 --suffix-draft 0` and `STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_PCIE_FRAC_DEFAULT=old STRATA_PF_FUSED=0`, SHA-256 of the greedy token streams)

- `STRATA_SPLIT_OVERLAP=1`: story, code, facts and the 4K prompt answer identical to default (as the docs say).
- `--pipeline-windows 2`: the code answer differs from default; with `STRATA_MTP_STAGE0=1` story and code differ; `--pipeline-windows 1`: the code answer differs.
- `--batch-groups 2`: the four-chat token streams differ from the serial group.
- Default flags (adaptation on) are not repeatable run to run (14 to 20 distinct hashes over 20 runs), so only the gate-flag runs are an identity statement.

## Recommendation

Nothing to switch on by default for NVIDIA layer splits: no existing switch gives more than noise except `--pipeline-windows 2` on code (+5 %, not identical, and worse on prose or with the draft on card 0). Keep `STRATA_SPLIT_OVERLAP` opt-in and drop its "not measured on NVIDIA" line in docs/MULTI_GPU.md (measured: no change). The idle half is structural: a serial layer split has one card working at a time, and the only fill (a second window in flight) costs as much as it gains at this acceptance rate. No A/B for default-worthiness and no identity gate were warranted.
