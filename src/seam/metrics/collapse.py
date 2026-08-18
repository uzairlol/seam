"""Memory collapse and behavioral metrics engine for SEAM."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from seam.agents.decoding import OllamaClient

_HEADER_PATTERNS = [
    r"===\s*[^=]+\s*===",
    r"Rule\s*#?\d+:",
    r"Observation:",
    r"Action Taken:",
    r"Reward Received:",
    r"=== Shared Peer Memories ===",
]
_HEADER_RE = re.compile("|".join(_HEADER_PATTERNS), re.IGNORECASE)


def _clean_memory_text(text: str) -> str:
    """Remove boilerplate section headers and structural labels prior to metric computation."""
    cleaned = _HEADER_RE.sub("", text).strip()
    return " ".join(cleaned.split()) if cleaned else text.strip()


def compute_ngram_counts(text: str, n: int = 2) -> Counter[tuple[str, ...]]:
    """Extract n-gram counts from tokenized text.

    Args:
        text: Target text string.
        n: N-gram order (default 2 for bigrams).

    Returns:
        Counter of n-gram tuples.
    """
    cleaned_text = _clean_memory_text(text)
    tokens = cleaned_text.lower().split()
    if len(tokens) < n:
        return Counter()
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_self_bleu(memory_sequence: list[str], max_n: int = 2) -> float:
    """Compute mean Self-BLEU score across a sequence of memory texts.

    Higher Self-BLEU indicates higher lexical repetition/collapse across rounds.

    Args:
        memory_sequence: List of memory text strings over successive rounds.
        max_n: Maximum n-gram order to include (default 2).

    Returns:
        Scalar Self-BLEU score in [0.0, 1.0].
    """
    clean_seq = [_clean_memory_text(m) for m in memory_sequence if m.strip()]
    clean_seq = [m for m in clean_seq if m]
    if len(clean_seq) <= 1:
        return 1.0

    scores: list[float] = []
    for i, candidate in enumerate(clean_seq):
        references = [m for j, m in enumerate(clean_seq) if j != i]
        cand_tokens = candidate.lower().split()
        if not cand_tokens:
            continue

        precision_product = 1.0
        valid_ngrams = 0

        for n in range(1, max_n + 1):
            cand_ngrams = compute_ngram_counts(candidate, n)
            if not cand_ngrams:
                continue

            ref_max_counts: dict[tuple[str, ...], int] = {}
            for ref in references:
                ref_counts = compute_ngram_counts(ref, n)
                for ngram, count in ref_counts.items():
                    ref_max_counts[ngram] = max(ref_max_counts.get(ngram, 0), count)

            clipped_matches = sum(
                min(count, ref_max_counts.get(ngram, 0))
                for ngram, count in cand_ngrams.items()
            )
            total_cand = sum(cand_ngrams.values())
            precision = clipped_matches / total_cand if total_cand > 0 else 0.0
            precision_product *= max(precision, 1e-4)
            valid_ngrams += 1

        if valid_ngrams > 0:
            bleu = math.exp(math.log(precision_product) / valid_ngrams)
            scores.append(bleu)

    return sum(scores) / len(scores) if scores else 0.0


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vector representations.

    Args:
        vec_a: First float vector.
        vec_b: Second float vector.

    Returns:
        Float cosine similarity in [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_embedding_similarity(
    memory_sequence: list[str],
    client: OllamaClient | None = None,
) -> list[float]:
    """Compute cosine similarity between successive memory embeddings.

    Args:
        memory_sequence: Sequence of memory strings.
        client: Optional OllamaClient for fetching text embeddings.

    Returns:
        List of float similarity scores between round t and round t+1.
    """
    if len(memory_sequence) < 2:
        return []

    similarities: list[float] = []
    if client is not None:
        embeddings = [client.embed(m) for m in memory_sequence]
        for i in range(len(embeddings) - 1):
            sim = compute_cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
    else:
        # Deterministic fallback using token overlap cosine proxy
        for i in range(len(memory_sequence) - 1):
            set_a = set(memory_sequence[i].lower().split())
            set_b = set(memory_sequence[i + 1].lower().split())
            if not set_a or not set_b:
                similarities.append(0.0)
            else:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                similarities.append(jaccard)

    return similarities


def compute_action_entropy(actions: list[str], window_size: int = 10) -> list[float]:
    """Compute Shannon action entropy over rolling windows of actions.

    Lower entropy indicates action repetition/collapse.

    Args:
        actions: Sequence of action strings.
        window_size: Rolling window size (default 10).

    Returns:
        List of entropy values per window.
    """
    if len(actions) < window_size:
        window_size = max(1, len(actions))

    entropies: list[float] = []
    for i in range(len(actions) - window_size + 1):
        window = actions[i : i + window_size]
        counts = Counter(window)
        total = len(window)
        ent = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                ent -= p * math.log2(p)
        entropies.append(ent)

    return entropies


def compute_memory_length(memory_sequence: list[str]) -> list[int]:
    """Return word count per memory state in sequence.

    Args:
        memory_sequence: Sequence of memory text strings.

    Returns:
        List of integer word counts.
    """
    return [len(m.split()) for m in memory_sequence]
