"""spaCy NLP pipeline: tokenise, lemmatise, POS-tag a corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# POS tags always excluded regardless of --include-stopwords
_ALWAYS_SKIP_POS = frozenset({"PUNCT", "SPACE", "SYM", "X", "NUM", "PROPN"})


@dataclass
class CorpusStats:
    total_tokens: int
    analyzed_tokens: int
    file_count: int
    lemma_pos_counts: Counter = field(default_factory=Counter)


def _load_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )


def _should_include(token, include_stopwords: bool) -> bool:
    if token.is_space or token.is_punct or token.like_num:
        return False
    if token.pos_ in _ALWAYS_SKIP_POS:
        return False
    if not include_stopwords and token.is_stop:
        return False
    lemma = token.lemma_.lower()
    if len(lemma) < 2:
        return False
    if not lemma.isalpha():
        return False
    return True


def analyze_corpus(
    texts: list[str],
    include_stopwords: bool = False,
    show_progress: bool = True,
) -> CorpusStats:
    """Process a list of plain-text strings through spaCy and return frequency counts."""
    nlp = _load_nlp()
    counts: Counter = Counter()
    total = 0
    analyzed = 0

    if show_progress:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
    else:
        from contextlib import nullcontext
        progress_ctx = nullcontext()

    with progress_ctx as progress:
        if show_progress:
            task = progress.add_task("Analysing corpus…", total=len(texts))

        for doc in nlp.pipe(texts, batch_size=50):
            for token in doc:
                total += 1
                if _should_include(token, include_stopwords):
                    counts[(token.lemma_.lower(), token.pos_)] += 1
                    analyzed += 1
            if show_progress:
                progress.advance(task)

    return CorpusStats(
        total_tokens=total,
        analyzed_tokens=analyzed,
        file_count=len(texts),
        lemma_pos_counts=counts,
    )
