"""Training data augmentation utilities.

Provides text and numerical augmentation strategies for ML training pipelines.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "AugmentationConfig",
    "augment_batch",
    "augment_text",
    "jitter_numerics",
    "random_deletion",
    "random_swap",
    "synonym_replace",
]


@dataclass
class AugmentationConfig:
    """Configuration for an augmentation pipeline."""

    synonym_prob: float = 0.1
    deletion_prob: float = 0.1
    swap_prob: float = 0.1
    numeric_jitter_pct: float = 0.05
    seed: int | None = None
    synonyms: dict = field(default_factory=dict)


def synonym_replace(tokens: list[str], synonyms: dict, prob: float, rng: random.Random) -> list[str]:
    """Replace tokens with synonyms at random.

    Args:
        tokens: Input word list.
        synonyms: Mapping from token to list of replacement candidates.
        prob: Per-token probability of replacement (0–1).
        rng: Seeded random source for reproducibility.

    Returns:
        New token list with some tokens substituted.
    """
    result = []
    for token in tokens:
        if rng.random() < prob and token in synonyms:
            result.append(rng.choice(synonyms[token]))
        else:
            result.append(token)
    return result


def random_deletion(tokens: list[str], prob: float, rng: random.Random) -> list[str]:
    """Delete tokens randomly; always keeps at least one token.

    Args:
        tokens: Input word list.
        prob: Per-token deletion probability (0–1).
        rng: Seeded random source for reproducibility.

    Returns:
        Shortened token list with at least one element.
    """
    if len(tokens) == 1:
        return tokens[:]
    kept = [t for t in tokens if rng.random() > prob]
    return kept if kept else [rng.choice(tokens)]


def random_swap(tokens: list[str], prob: float, rng: random.Random) -> list[str]:
    """Swap adjacent token pairs at random.

    Args:
        tokens: Input word list.
        prob: Per-pair swap probability (0–1).
        rng: Seeded random source for reproducibility.

    Returns:
        Token list with some adjacent pairs swapped.
    """
    result = tokens[:]
    for i in range(len(result) - 1):
        if rng.random() < prob:
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def jitter_numerics(text: str, pct: float, rng: random.Random) -> str:
    """Add Gaussian jitter to every numeric value found in *text*.

    Args:
        text: Input string possibly containing integers or floats.
        pct: Standard-deviation fraction of each value's magnitude.
        rng: Seeded random source for reproducibility.

    Returns:
        Modified string with jittered numeric values.
    """

    def _jitter(m: re.Match) -> str:
        val = float(m.group())
        noise = rng.gauss(0, abs(val) * pct) if val != 0 else rng.gauss(0, pct)
        return str(round(val + noise, 6))

    return re.sub(r"-?\d+(?:\.\d+)?", _jitter, text)


def augment_text(text: str, config: AugmentationConfig) -> str:
    """Apply the full augmentation pipeline to *text*.

    Applies synonym replacement, random deletion, random swap, and numeric
    jitter in sequence, using the parameters from *config*.

    Args:
        text: Input string to augment.
        config: Augmentation parameters and synonym mapping.

    Returns:
        Augmented string.
    """
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
    transform: Callable[[str, AugmentationConfig], str] | None = None,
) -> list[str]:
    """Augment every sample *n_augments* times, returning all augmented variants.

    Args:
        samples: Original text samples to augment.
        config: Augmentation parameters.
        n_augments: Number of augmented copies to produce per sample.
        transform: Custom augmentation function; defaults to :func:`augment_text`.

    Returns:
        Flat list of ``len(samples) * n_augments`` augmented strings.
    """
    fn = transform or augment_text
    return [fn(s, config) for s in samples for _ in range(n_augments)]
