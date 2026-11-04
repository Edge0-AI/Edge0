# GGML source

`ggml/` is an unmodified copy of the `ggml/` tree from
[llama.cpp](https://github.com/ggml-org/llama.cpp) at commit
`7ab4ee7baad2d920464cbacfad4f4b07cf111fd2` (GGML 0.24.0).
`GGML_ORIGIN` records the commit, dates, tree hash and license blob hash.
`GGML_LICENSE` preserves that commit's root MIT license verbatim.

Do not edit or prune the vendored files.

To upgrade:

1. Export `ggml/` and `LICENSE` from the new fixed llama.cpp commit in the
   external reference checkout. Replace the whole tree and license.
2. Update `commit`, `commit_date`, `tree`, `ggml_version`, `license_blob` and
   `imported` in `GGML_ORIGIN`.
3. Rerun L0 through L2.
4. Record the upgrade in a separate commit identifying both old and new
   upstream commits. Git history preserves the previous metadata for rollback.
