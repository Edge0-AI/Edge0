"""Local, format-neutral checkpoint sources (no backend imports or downloads)."""

from .source import CheckpointSource, TensorInfo, open_checkpoint

__all__ = ["CheckpointSource", "TensorInfo", "open_checkpoint"]
