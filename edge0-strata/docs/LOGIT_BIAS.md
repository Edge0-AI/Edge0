# Per-request token biases

`POST /v1/chat/completions` accepts `logit_bias` with a supporting CUDA or HIP
engine, without continuous batching. The same bias applies to every generated
token in the request, including tokens checked together during speculative
decoding. It does not force single-token verification windows.

## Example: Hello becomes Hi

We use `Hello` because it is a familiar word, it is exactly one token in the
tested tokenizer, and the unbiased model naturally chooses it for this prompt.
The tokenizer maps `Hello` to **token ID 9419**. That number identifies the token;
it is not its logit or probability. The value **-100** is the requested bias,
which this implementation treats as a hard exclusion.

Before: send this request without a bias. The measured response was **Hello**.

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none"
}
```

After: add only `logit_bias`. The measured response was **Hi** (token 12675).

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none",
  "logit_bias": {"9419": -100}
}
```

The equivalent llama.cpp-style pair list is `"logit_bias": [[9419, false]]`;
it also produced **Hi**. Removing the field on the next request restored **Hello**.
Nothing is banned by default. This is a per-request example, not a built-in list.
The `model` value above is the test server's alias; use your server's model name.
Numeric pair values are also supported, for example `[[42, 2.5]]`.
Token IDs belong to the loaded model's tokenizer; the example ID is not portable
between arbitrary models. Leading spaces and letter case can also change token IDs.
The four English requests were measured on ISTA IQ3_XXS; a
[reproduction script](../tools/logit_bias_english_example.py) is included.

## Semantics and limits

Biases must be finite numbers in `[-100, 100]`. Negative values discourage a
token, positive values encourage it, and **-100 excludes it** (`false` in a
pair list has the same meaning). The bias is added before repetition/frequency/
presence penalties, temperature, and candidate filtering. Raw model logits
are not overwritten. Banning a token does not ban a word or language: other
token sequences may produce the same text.

Omitting the field, `null`, `{}`, or `[]` clears the previous request's bias.
Biases are request sampling state; they are not stored in a conversation cache.
They do not change the model weights or prompt KV. A fixed dense vocabulary
vector is uploaded when the bias changes and reused within the request.

Malformed entries, duplicate IDs in pair lists, IDs outside the vocabulary,
out-of-range biases, or banning the entire vocabulary return HTTP 400. Validation
on the chat route occurs before SSE streaming headers are sent. Nonempty biases
also return HTTP 400 with older engines, the current SYCL engine, or continuous
batching (`--batch`), rather than being silently ignored. The engine advertises
`logit_bias=1` in its `INFO` response when this implementation is present.

MTP target verification applies the bias; the draft model is not biased, so a
restrictive list may reduce draft acceptance. Ordinary MTP was tested. The
experimental coupled/rejection-sampling combinations and multi-GPU execution
were not validated in this campaign. There is no startup bias-file option in
this change: send the field with each request.
