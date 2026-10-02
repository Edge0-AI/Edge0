"""Generation / sampling configuration (transformers-style)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Sampling knobs for one model tier.

    Attributes:
        temperature: softmax temperature (0 disables).
        top_p: nucleus truncation (<1.0 enables).
        top_k: top-k truncation (>0 enables).
        repetition_penalty: HF-style penalty applied to the history of
            generated tokens.
        max_new_tokens: default cap for generated tokens.
        eos_ids: token ids that end generation.
        seed: optional RNG seed.
        first_token_greedy: always argmax the FIRST generated token
            (production-verified pattern): after the think opener a
            randomly-sampled first token can derail the whole block into
            '!' loops.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    eos_ids: tuple[int, ...] = ()
    seed: int | None = None
    first_token_greedy: bool = True

    def is_eos(self, token_id: int) -> bool:
        return token_id in self.eos_ids
