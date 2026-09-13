"""Opt-in persistent inference checkpoints (no tensor imports)."""
from .store import CacheConfig, CheckpointStore

__all__ = ['CacheConfig', 'CheckpointStore']
