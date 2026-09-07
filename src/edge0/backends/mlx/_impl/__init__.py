"""Vendored base model implementations (licensing in NOTICE).

* ``qwen3_next.py`` — mlx-lm 0.31.0's Qwen3.5-MoE model, byte-identical
  to the upstream file except the import block, which was rewritten to
  absolute ``mlx_lm.models`` paths so the file can live outside the
  mlx_lm package.  Pinned to mlx-lm==0.31.0 by pyproject.
* ``bailing_hybrid.py`` — the ling 3.0 backbone vendored from the
  production deployment, with the prerouter machinery renamed from the
  legacy identifiers (``*_gate*`` -> ``prerouter*``) and no other
  changes.
"""
