"""Local Qwen byte-level BPE tokenizers reconstructed from GGUF metadata."""
from pathlib import Path

# llama.cpp 6c84c7d5d8833c6e0df69628f75a0f599797934e, llama-vocab.cpp.
QWEN35_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def load_tokenizer(metadata, override=None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if override is not None:
        path = Path(override).expanduser().resolve(strict=True)
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if not tok.chat_template or tok.eos_token_id is None:
            raise ValueError("--tokenizer-path must supply a chat template and EOS token ID")
        tokens = metadata.get("tokenizer.ggml.tokens")
        if tokens is not None:
            vocab = tok.get_vocab()
            if any(vocab.get(token) != i for i, token in enumerate(tokens)):
                raise ValueError("--tokenizer-path vocabulary IDs disagree with GGUF metadata")
        return tok
    required = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
                "tokenizer.ggml.eos_token_id", "tokenizer.chat_template")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"GGUF tokenizer metadata missing {', '.join(missing)}; provide --tokenizer-path /local/tokenizer")
    if (metadata.get("tokenizer.ggml.model"), metadata.get("tokenizer.ggml.pre")) != ("gpt2", "qwen35"):
        raise ValueError("GGUF tokenizer requires gpt2/qwen35 pretokenization; provide --tokenizer-path")
    from tokenizers import Tokenizer, Regex, AddedToken, models, pre_tokenizers, decoders, processors
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    if len(tokens) != len(types) or len(set(tokens)) != len(tokens):
        raise ValueError("invalid GGUF tokenizer vocabulary or token types")
    vocab = {text: i for i, text in enumerate(tokens)}
    merges = []
    for merge in metadata["tokenizer.ggml.merges"]:
        pair = merge.split(" ")
        if len(pair) != 2 or any(t not in vocab for t in pair) or "".join(pair) not in vocab:
            raise ValueError(f"invalid GGUF BPE merge {merge!r}")
        merges.append(tuple(pair))
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False))
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(t, normalized=False, special=True)
                                for t, kind in zip(tokens, types) if kind in (3, 4)])
    kwargs = {}
    for key, name in (("eos", "eos"), ("bos", "bos"), ("padding", "pad")):
        tid = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if tid is not None:
            if type(tid) is not int or not 0 <= tid < len(tokens):
                raise ValueError(f"invalid GGUF {key} token ID")
            kwargs[f"{name}_token"] = tokens[tid]
    if metadata.get("tokenizer.ggml.add_bos_token", False):
        bid = metadata.get("tokenizer.ggml.bos_token_id")
        if bid is None:
            raise ValueError("GGUF requests BOS insertion but has no BOS token ID")
        backend.post_processor = processors.TemplateProcessing(
            single=f"{tokens[bid]} $A", special_tokens=[(tokens[bid], bid)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend,
        chat_template=metadata["tokenizer.chat_template"], clean_up_tokenization_spaces=False, **kwargs)
    # Fail before loading weights if the checkpoint's template cannot render text.
    tok.apply_chat_template([{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True)
    return tok
