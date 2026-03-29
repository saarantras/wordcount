"""Reference corpora: NLTK Brown (cached) and wordfreq."""

from __future__ import annotations

import hashlib
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BROWN_ALL_CATEGORIES: list[str] = [
    "adventure",
    "belles_lettres",
    "editorial",
    "fiction",
    "government",
    "hobbies",
    "humor",
    "learned",
    "lore",
    "mystery",
    "news",
    "religion",
    "reviews",
    "romance",
    "science_fiction",
]

CORPUS_GROUPS: dict[str, list[str]] = {
    "newspaper": ["news", "editorial", "reviews"],
    "literary": [
        "fiction",
        "romance",
        "mystery",
        "adventure",
        "science_fiction",
        "humor",
        "belles_lettres",
    ],
    "general": BROWN_ALL_CATEGORIES,
    # "modern" → handled via wordfreq_lookup, not Brown
}


@dataclass
class RefCorpus:
    name: str
    total_tokens: int
    lemma_pos_counts: dict[tuple[str, str], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# wordfreq
# ---------------------------------------------------------------------------

def wordfreq_lookup(lemma: str) -> float:
    """Return word_frequency(lemma, 'en') as a fraction (0.0 if unknown)."""
    from wordfreq import word_frequency
    return word_frequency(lemma, "en")


# ---------------------------------------------------------------------------
# Brown corpus (cached)
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    from platformdirs import user_cache_dir
    d = Path(user_cache_dir("wordcount", "wordcount"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fingerprint(group: str) -> str:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        meta = nlp.meta
        key = f"{spacy.__version__}:{meta['name']}:{meta['version']}:{group}"
    except OSError:
        key = f"unknown:unknown:unknown:{group}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _cache_path(group: str) -> Path:
    return _cache_dir() / f"brown_{group}_{_fingerprint(group)}.pkl"


def _build_brown(group: str, show_progress: bool = True) -> RefCorpus:
    """Process Brown corpus text through spaCy and return a RefCorpus."""
    import nltk
    from nltk.corpus import brown as brown_corpus
    from wordcount.analyzer import _load_nlp, _should_include

    nltk.download("brown", quiet=True)

    categories = CORPUS_GROUPS[group]
    sents = brown_corpus.sents(categories=categories)
    # Join each sentence into a string; Brown words are already plain text
    texts = [" ".join(sent) for sent in sents]

    nlp = _load_nlp()
    counts: Counter = Counter()
    total = 0

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
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
            task = progress.add_task(
                f"Building '{group}' reference corpus (first run)…",
                total=len(texts),
            )

        for doc in nlp.pipe(texts, batch_size=100):
            for token in doc:
                total += 1
                if _should_include(token, include_stopwords=False):
                    counts[(token.lemma_.lower(), token.pos_)] += 1
            if show_progress:
                progress.advance(task)

    return RefCorpus(
        name=group,
        total_tokens=total,
        lemma_pos_counts=dict(counts),
    )


def get_brown_frequencies(
    group: str,
    no_cache: bool = False,
    show_progress: bool = True,
) -> RefCorpus:
    """Return Brown reference frequencies for a given corpus group, using cache."""
    if group not in CORPUS_GROUPS:
        raise ValueError(f"Unknown corpus group '{group}'. Choose from: {list(CORPUS_GROUPS)}")

    path = _cache_path(group)

    if not no_cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    ref = _build_brown(group, show_progress=show_progress)

    with open(path, "wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ref
