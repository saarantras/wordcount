"""CLI entry point for wordcount."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
def main() -> None:
    """Analyse word frequency outliers in a markdown corpus."""
    pass


@main.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--corpus", "-c",
    type=click.Choice(["general", "literary", "newspaper", "modern"]),
    multiple=True,
    default=("literary", "newspaper", "modern"),
    show_default=True,
    help="Reference corpus to compare against. Repeatable.",
)
@click.option(
    "--top", "-n",
    default=20,
    show_default=True,
    help="Top N over/under-represented lemmas per POS.",
)
@click.option(
    "--min-freq",
    default=5,
    show_default=True,
    help="Minimum count in user corpus to include a lemma.",
)
@click.option(
    "--pos", "-p",
    type=click.Choice(["NOUN", "VERB", "ADJ", "ADV", "all"]),
    multiple=True,
    default=("all",),
    show_default=True,
    help="POS categories to display. Repeatable.",
)
@click.option(
    "--include-stopwords",
    is_flag=True,
    default=False,
    help="Include stopwords (excluded by default).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Rebuild Brown reference corpus cache from scratch.",
)
def analyze(
    directory: Path,
    corpus: tuple[str, ...],
    top: int,
    min_freq: int,
    pos: tuple[str, ...],
    include_stopwords: bool,
    no_cache: bool,
) -> None:
    """Analyse word frequency outliers in DIRECTORY (recursed for .md files)."""
    from wordcount.parser import collect_texts
    from wordcount.analyzer import analyze_corpus
    from wordcount.corpora import get_brown_frequencies, RefCorpus
    from wordcount.stats import compute_keyness
    from wordcount.display import render

    # De-duplicate corpora while preserving order
    seen: set[str] = set()
    corpora_ordered: list[str] = []
    for c in corpus:
        if c not in seen:
            corpora_ordered.append(c)
            seen.add(c)

    # --- Parse markdown files ---
    console.print(f"[bold]Scanning[/] {directory} …")
    texts, file_paths = collect_texts(directory)

    if not texts:
        console.print(f"[red]No markdown files found in {directory}[/]")
        sys.exit(1)

    console.print(f"Found [bold]{len(file_paths)}[/] markdown file(s)")

    # --- Analyse user corpus ---
    try:
        user_stats = analyze_corpus(texts, include_stopwords=include_stopwords)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    # --- Build reference corpora ---
    ref_corpora: dict[str, RefCorpus | None] = {}
    for corpus_name in corpora_ordered:
        if corpus_name == "modern":
            ref_corpora["modern"] = None  # handled via wordfreq inside stats
        else:
            try:
                ref_corpora[corpus_name] = get_brown_frequencies(
                    corpus_name, no_cache=no_cache
                )
            except Exception as e:
                console.print(f"[red]Failed to load '{corpus_name}' corpus: {e}[/]")
                sys.exit(1)

    # --- Compute keyness ---
    results = compute_keyness(user_stats, ref_corpora, min_freq=min_freq)

    # --- Render ---
    render(results, top_n=top, pos_filter=pos, console=console)
