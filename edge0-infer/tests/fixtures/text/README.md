# Frozen text fixtures

These files are tokenizer and chat template reference data for the 35B tokenizer.
They were exported once, offline, from:

- llama.cpp at commit `7ab4ee7baad2d920464cbacfad4f4b07cf111fd2` (tokenization of the
  `*_golden` rows and the `llama` columns, using the 35B `base.gguf` with the `qwen35`
  pre-tokenizer);
- Hugging Face `tokenizers` 0.21.4 on the 35B `tokenizer.json` (`hf_original.json`, and
  `hf_no_normalizer.json` with the normalizer removed as a diagnostic);
- the unmodified 35B Jinja chat template (`template_golden.*`).

The programs that produced them were retired together with the llama.cpp reference
export on 2026-10-21. Running the tests needs none of those tools: the files are frozen
data, and their internal consistency (JSON and TSV forms, digests, counts) is checked
by the text tests.

`pieces_golden.json` holds digests of `pieces_golden.tsv`: one SHA-256 per group of 1024
tokens. Per-token digests were removed on 2026-10-30 to keep the tree small; any byte
change in the TSV still changes its group's digest, so the check still detects it, but
locates it only to the group.

Rules for changing them:

- Do not edit the files by hand.
- A change must be produced by a tool in this repository, and every changed row must be
  listed and explained in the change that makes it.
