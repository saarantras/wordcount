"""G² (Dunning log-likelihood) keyness computation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from wordcount.analyzer import CorpusStats
from wordcount.corpora import RefCorpus, wordfreq_lookup

_LAPLACE = 0.5  # smoothing for zero counts

# Significance thresholds (signed sqrt(G²) ≈ z-score)
SIG_P05 = 1.96
SIG_P001 = 3.29


@dataclass
class KeynessScore:
    lemma: str
    pos: str
    user_count: int
    user_freq_per_10k: float
    scores: dict[str, float] = field(default_factory=dict)
    # scores[corpus_name] = signed sqrt(G²); positive = over-represented


@dataclass
class KeynessResults:
    # Each list is sorted: over-represented first (descending score),
    # under-represented last (ascending score, i.e. most negative first).
    by_pos: dict[str, list[KeynessScore]] = field(default_factory=dict)
    user_total_tokens: int = 0
    user_analyzed_tokens: int = 0
    user_file_count: int = 0
    corpora_used: list[str] = field(default_factory=list)
    primary_corpus: str = ""


def _g2_signed(o11: int, n1: int, o21: int, n2: int) -> float:
    """Return signed sqrt(G²): positive = over-represented in corpus 1."""
    o12 = n1 - o11
    o22 = n2 - o21
    n = n1 + n2

    def expected(row_sum: int, col_sum: int) -> float:
        return max(row_sum * col_sum / n, _LAPLACE)

    e11 = expected(o11 + o21, o11 + o12)
    e12 = expected(o12 + o22, o11 + o12)
    e21 = expected(o11 + o21, o21 + o22)
    e22 = expected(o12 + o22, o21 + o22)

    def cell(o: int, e: float) -> float:
        if o <= 0:
            return 0.0
        return o * math.log(o / e)

    g2 = max(0.0, 2.0 * (cell(o11, e11) + cell(o12, e12) + cell(o21, e21) + cell(o22, e22)))
    sign = 1 if (o11 / n1) >= (o21 / n2) else -1
    return sign * math.sqrt(g2)


def compute_keyness(
    user_stats: CorpusStats,
    ref_corpora: dict[str, RefCorpus | None],
    min_freq: int = 5,
) -> KeynessResults:
    """Compute keyness scores for all (lemma, pos) pairs in the user corpus.

    ref_corpora maps corpus_name → RefCorpus (or None for 'modern'/wordfreq).
    """
    corpora_names = list(ref_corpora.keys())
    primary = corpora_names[0] if corpora_names else ""

    by_pos: dict[str, list[KeynessScore]] = defaultdict(list)
    n1 = user_stats.analyzed_tokens

    for (lemma, pos), count in user_stats.lemma_pos_counts.items():
        if count < min_freq:
            continue

        freq_per_10k = count / n1 * 10_000 if n1 else 0.0
        scores: dict[str, float] = {}

        for corpus_name, ref in ref_corpora.items():
            if corpus_name == "modern" or ref is None:
                # wordfreq path: virtual 1M-token corpus
                wf = wordfreq_lookup(lemma)
                o21 = round(wf * 1_000_000)
                n2 = 1_000_000
            else:
                o21 = ref.lemma_pos_counts.get((lemma, pos), 0)
                n2 = ref.total_tokens

            scores[corpus_name] = _g2_signed(
                o11=count,
                n1=n1,
                o21=o21,
                n2=n2,
            )

        by_pos[pos].append(
            KeynessScore(lemma, pos, count, freq_per_10k, scores)
        )

    # Sort each POS group by primary corpus score descending
    # (over-represented first, under-represented last)
    sorted_by_pos: dict[str, list[KeynessScore]] = {}
    for pos, items in by_pos.items():
        sorted_by_pos[pos] = sorted(
            items,
            key=lambda k: k.scores.get(primary, 0.0),
            reverse=True,
        )

    return KeynessResults(
        by_pos=sorted_by_pos,
        user_total_tokens=user_stats.total_tokens,
        user_analyzed_tokens=user_stats.analyzed_tokens,
        user_file_count=user_stats.file_count,
        corpora_used=corpora_names,
        primary_corpus=primary,
    )
