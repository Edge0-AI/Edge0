# Known Limitations

Behavior of the current release that users and integrators should be aware of.

## Multi-turn reuse sometimes recomputes part of the conversation

**What you see.** In a multi-turn conversation the engine normally computes only the new message of each turn and reuses everything before it. Sometimes a turn instead restores a checkpoint and computes part of the previous answer again: its generation summary shows `"checkpoint_restored":true` and more `computed_tokens`, and the first token takes longer. The answer is still correct; only the extra prefill costs time. With thinking enabled, answers are longer, so this happens more often.

**Why.** Reuse needs the tokens of the new turn to match the tokens already computed, one by one. For each turn the application sends the whole conversation as text, and the engine renders and tokenizes it again. The previous answer, however, was computed as the tokens the model generated one at a time. The same text can be split into tokens in more than one way, and the model does not always generate the split that the tokenizer produces from the text. Where the two sequences first differ, the computed state no longer matches the conversation, so the engine restores the checkpoint it saved at the end of the previous prompt and computes the previous answer and the new message again from there (see [Multi-turn reuse](architecture.md#multi-turn-reuse)). The longer an answer is, the more chances there are for such a difference.
