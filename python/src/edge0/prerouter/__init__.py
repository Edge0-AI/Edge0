"""Prerouter: trained cross-token routing predictor heads + step-boundary
staging.

Public API::

    from edge0.prerouter import (PrerouterSpec, PrerouterState,
                                 PrerouterHead, install_prerouter,
                                 CrossTokenStager, LingPrerouterStager)
"""

from edge0.prerouter.heads import PrerouterHead, select_from_logits, topk_onehot
from edge0.prerouter.install import install_prerouter
from edge0.prerouter.spec import PrerouterSpec
from edge0.prerouter.stager import (
    CrossTokenStager,
    LingPrerouterStager,
    PrerouterStager,
)
from edge0.prerouter.state import PrerouterState

__all__ = [
    "PrerouterSpec",
    "PrerouterState",
    "PrerouterHead",
    "PrerouterStager",
    "CrossTokenStager",
    "LingPrerouterStager",
    "install_prerouter",
    "select_from_logits",
    "topk_onehot",
]
