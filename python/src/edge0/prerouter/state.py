"""Prerouter decode-time state (cross-token, cross-layer double buffer).

After each decode step call ``swap()``:
  ``logits_prev[owner] = logits[owner]``   (owner = producing layer N)
  ``oh_prev[layer]     = oh[layer]``       (per-layer executed one-hots)
``pred_inds`` / ``pred_scores`` are keyed by the CONSUMING layer (N+1)
and hold the selection made at the step boundary; each is consumed
exactly once on the next step by that layer, then overwritten.
"""

from __future__ import annotations


class PrerouterState:
    def __init__(self, n_layers: int, num_experts: int, start_layer: int,
                 owners=None):
        self.n = n_layers
        self.num_experts = num_experts
        self.start = start_layer                # first consuming layer
        if owners is None:
            owners = list(range(start_layer - 1, n_layers - 1))
        self.owners = list(owners)
        self.logits: list = [None] * n_layers        # owner output, this token
        self.logits_prev: list = [None] * n_layers   # owner output, prev token
        self.pred_inds: list = [None] * n_layers     # keyed by consumer N+1
        self.pred_scores: list = [None] * n_layers   # keyed by consumer N+1
        self.oh: list = [None] * n_layers            # executed one-hots, now
        self.oh_prev: list = [None] * n_layers       # executed one-hots, prev

    def swap(self):
        self.logits_prev = self.logits
        self.logits = [None] * self.n
        self.oh_prev = self.oh
        self.oh = [None] * self.n

    def reset(self):
        self.logits = [None] * self.n
        self.logits_prev = [None] * self.n
        self.pred_inds = [None] * self.n
        self.pred_scores = [None] * self.n
        self.oh = [None] * self.n
        self.oh_prev = [None] * self.n
