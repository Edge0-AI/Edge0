"""Adapter installation (parallel LoRA + prerouter heads)."""

from edge0.adapters.lora import LoraLinear, install_lora

__all__ = ["LoraLinear", "install_lora"]
