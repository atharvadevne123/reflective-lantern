"""Training data augmentation utilities.

Provides text and numerical augmentation strategies for ML training pipelines.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

__all__ = [
    "AugmentationConfig",
    "augment_text",
    "synonym_replace",
    "random_deletion",
    "random_swap",
    "jitter_numerics",
    "augment_batch",
]


@dataclass
class AugmentationConfig:
    """Configuration for an augmentation pipeline."""

    synonym_prob: float = 0.1
    deletion_prob: float = 0.1
    swap_prob: float = 0.1
    numeric_jitter_pct: float = 0.05
    seed: Optional[int] = None
    synonyms: dict = field(default_factory=dict)


def synonym_replace(tokens: List[str], synonyms: dict, prob: float, rng: random.Random) -> List[str]:
    """Replace tokens with synonyms at random."""
    result = []
    for token in tokens:
        if rng.random() < prob and token in synonyms:
            result.append(rng.choice(synonyms[token]))
        else:
            result.append(token)
    return result


def random_deletion(tokens: List[str], prob: float, rng: random.Random) -> List[str]:
    """Delete tokens randomly; always keeps at least one."""
    if len(tokens) == 1:
        return tokens[:]
    kept = [t for t in tokens if rng.random() > prob]
    return kept if kept else [rng.choice(tokens)]


def random_swap(tokens: List[str], prob: float, rng: random.Random) -> List[str]:
    """Swap adjacent token pairs randomly."""
    result = tokens[:]
    for i in range(len(result) - 1):
        if rng.random() < prob:
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def jitter_numerics(text: str, pct: float, rng: random.Random) -> str:
    """Add Gaussian jitter to numeric values found in *text*."""
    def _jitter(m: re.Match) -> str:
        """Replace the matched number with a Gaussian-jittered version."""
        val = float(m.group())
        noise = rng.gauss(0, abs(val) * pct) if val != 0 else rng.gauss(0, pct)
        return str(round(val + noise, 6))

    return re.sub(r"-?\d+(?:\.\d+)?", _jitter, text)


def augment_text(text: str, config: AugmentationConfig) -> str:
    """Apply all configured augmentations to *text* and return the result."""
    rng = random.Random(config.seed)
    tokens = text.split()
    tokens = synonym_replace(tokens, config.synonyms, config.synonym_prob, rng)
    tokens = random_deletion(tokens, config.deletion_prob, rng)
    tokens = random_swap(tokens, config.swap_prob, rng)
    result = " ".join(tokens)
    if config.numeric_jitter_pct > 0:
        result = jitter_numerics(result, config.numeric_jitter_pct, rng)
    return result


def augment_batch(
    samples: Sequence[str],
    config: AugmentationConfig,
    n_augments: int = 1,
    transform: Optional[Callable[[str, AugmentationConfig], str]] = None,
) -> List[str]:
    """Augment every sample *n_augments* times and return all results."""
    fn = transform or augment_text
    return [fn(s, config) for s in samples for _ in range(n_augments)]
