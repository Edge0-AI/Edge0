# What 0.1.40.1 does with a `<tool_call>` the model writes, and with requests waiting for a dead engine

Measured on 2026-10-06 by [Dmitry-B](https://github.com/Dmitry-B), on the machine described in
[2026-10-03-community-rtx4090-iq3xxs-200k](../2026-10-03-community-rtx4090-iq3xxs-200k/README.md)
(RTX 4090, `qwen3.8-flash-next-iq3_xxs`, context 204800). Results-only report about the two fixes of the hotfix
release v0.1.40.1 (`82f46a8`): the rules for a `<tool_call>` written by the model (#804, #1058) and requests waiting
during an engine restart (#1012). The engine binary is the one built for 0.1.40 and is byte-identical to it
(`196504f2822cbf15fb49b873e0000f41`); only the Python server changed.

## What was run

| File | What it is |
| --- | --- |
| `probe_parser_corpus.py`, `corpus_results.json` | the author's own corpus `serve/fixtures/rcall_specimens.json` replayed through the same parser the server uses |
| `probe_toolcall.py`, `results.json` | live chat requests with a declared tool, to see what the API returns end to end |
| `probe_restart_waiters.py`, `results_restart.json` | three requests waiting for the engine, the engine killed while reading a prompt |

## 1. The corpus of quoted and stranded calls, replayed

`serve/fixtures/rcall_specimens.json` holds 37 specimens: 16 must become tool calls, 21 must stay text. Each was
fed to `serve.frontend.OutputParser` — the class the server itself uses — at six feed widths (1, 2, 3, 7, random
1-9 characters, and the whole text at once) and with `stream_tools` on and off: the same matrix as
`serve/test_reasoning_rescue.py`, but recorded per specimen instead of pass/fail.

```
образцов 37, должны дать вызов: 16, должны остаться текстом: 21, провалились: 0
```

37/37, 444 parser runs, no mismatch, and for every specimen the streamed result equals the whole-text result.
Per-specimen detail: [corpus_results.json](corpus_results.json).

## 2. Live requests with a tool declared

`get_weather` was declared in `tools`; each prompt either shows a call as an example (in a code fence or inline
code) or asks for a real call. `reasoning_effort` is stated per case, because the rules are about where the text
sits — in the thinking or in the visible answer. `totals.tool_calls_from_reasoning` was read before and after the
whole set.

| Case | Expected | Returned | Where the call text stayed | Verdict |
| --- | --- | --- | --- | --- |
| `1-fenced-example-in-answer` (thinking on) | no call | 1 call | `reasoning_content` | not conclusive — see below |
| `2-inline-code-in-answer` (thinking off) | no call | 1 call | nowhere (the model emitted a top-level call) | not conclusive |
| `3-fenced-example-in-thinking` (thinking high) | no call | 0 | `content`, `reasoning_content` | matches |
| `4-turn-cut-by-max-tokens` (`max_tokens=60`, thinking high) | no call | 0 | `reasoning_content` | matches |
| `5-real-call-top-level-after-sentence` (thinking off) | call | 1 call | — | matches |
| `6-plain-real-call` | call | 1 call | — | matches |
| `7-streaming-split-fence` (stream, fence split across chunks) | no call | 0 | `content`, `reasoning_content` | matches |
| `8-fenced-example-in-answer-no-thinking` | no call | 0 | `content` | matches |

`totals.tool_calls_from_reasoning` stayed **0 → 0** across the whole set: none of these prompts made the model
strand a call in its thinking, so the rescue path never had anything to rescue. The calls in cases 1, 2, 5 and 6
came through the ordinary path.

That is also why cases 1 and 2 are recorded as inconclusive rather than as failures. In case 1 the model put the
example inside a fenced block in its reasoning (the text is still there in `reasoning_content`) and produced a
separate top-level call; in case 2, with thinking off, it answered with a top-level call instead of copying the
inline code. Both are model behaviour, not the rule being broken — the rule was applied to what the model actually
wrote, and the counter shows no rescue was performed. Raw text of every answer: [results.json](results.json).

What is shown end to end: a `<tool_call>` inside a code fence in the visible answer stays text, including when the
answer streams and the fence is split across chunks (cases 7, 8), a call written in the thinking of a turn cut by
`max_tokens` stays text (case 4), and ordinary calls keep working (cases 5, 6).

## 3. Requests waiting for an engine that dies (#1012)

Three requests with ~60 000-token prompts were started together; 15 s later `pkill` killed the engine process
(two pids, `39647` and `39659`).

| Request | HTTP | First streamed delta | Request ended | Deltas |
| --- | --- | --- | --- | --- |
| `waiter-1` | 200 | none | 16.6 s after start | 0 |
| `waiter-2` | 200 | 40.4 s | 40.5 s after start | 3 |
| `waiter-3` | 200 | 40.3 s | 40.4 s after start | 3 |

The server started a new engine 25.5 s after the kill and `/health` reported `loaded: true` at the same point.
Two waiters continued with the new engine and finished at ~40 s; the one whose stream had already been opened was
closed at 16.6 s with no content. Nothing hung: no request waited 300 s, and no `list.remove(x): x not in list`
appeared. For comparison, this is the case that on 0.1.40 ended about 300 s later with 1 and 5 tokens — see
[2026-10-06-community-rtx4090-opt-ins](../2026-10-06-community-rtx4090-opt-ins/README.md).

Server-side detail, including the engine command line it restarted with: [results_restart.json](results_restart.json).

## Unit tests on this machine, for the same commit

```
tools.test_setup_golden tools.test_setup_amd tools.test_setup_choices tools.test_setup_config
tools.test_setup_hotfix_tag                                                     -> 87 tests, OK
serve.test_responses serve.test_security serve.test_server serve.test_slots serve.test_runconfig
serve.test_mcp serve.test_monitor serve.test_frontend serve.test_structured serve.test_detok
serve.test_reasoning_tools serve.test_reasoning_loop_recovery serve.test_reasoning_rescue
serve.test_restart_waiters serve.test_parallel serve.test_vram                  -> 430 tests, OK (skipped=5)
serve.test_control_cancel serve.test_fatal_recovery                             -> 10 tests, OK
```

527 tests, 0 failures. On 0.1.40 the same set was 492; the difference is the two new modules of this release.

## Limitations

- The corpus replay exercises the parser, not the HTTP layer. The end-to-end part (section 2) did not produce a
  real stranded call on this model, so the rescue path was not observed through the API — only the "must stay text"
  half of the rules was.
- The three waiters sent identical filler text, so the engine reused the prefix cache (the log shows
  `prompt 141045 tokens = 140329 reused + 716 read`). That is fine for a hang test and wrong for a speed test;
  no throughput conclusion is drawn from it.
- `waiter-1` ended with HTTP 200 and no content rather than the 503 the release notes describe: its stream had
  already been opened, so the server could not answer 503 any more. Reported as measured.
- One machine, one model, one engine restart. Cancellation mid-turn and the batch-window `ERR` reporting were not
  tested.
