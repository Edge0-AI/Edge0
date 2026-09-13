"""GGML block layouts and direct GPU dispatch, indexed by tensor encoding."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Encoding:
    kind: int
    name: str
    block_elements: int
    block_bytes: int

    def packed_bytes(self, elements):
        if elements < 0 or elements % self.block_elements:
            raise ValueError(f'{self.name}: incomplete GGML block')
        return elements // self.block_elements * self.block_bytes

    def matmul(self, buffers, x, ids, rows):
        from edge0.backends.mlx.gguf_packed import matmul
        return matmul(buffers, self.kind, x, ids, rows)

    def rows(self, buffer, rows, width):
        from edge0.backends.mlx.gguf_packed import decode_rows
        return decode_rows(buffer, self.kind, rows, width)


ENCODINGS = MappingProxyType({kind: Encoding(kind, name, block, size)
    for kind, name, block, size in (
        (0, 'F32', 1, 4), (30, 'BF16', 1, 2), (8, 'Q8_0', 32, 34),
        (12, 'Q4_K', 256, 144), (13, 'Q5_K', 256, 176),
        (14, 'Q6_K', 256, 210), (16, 'IQ2_XXS', 256, 66),
        (19, 'IQ1_S', 256, 50), (20, 'IQ4_NL', 32, 18))})
QUANT_SIZES = {kind: (e.block_elements, e.block_bytes) for kind, e in ENCODINGS.items()}
