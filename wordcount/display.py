"""Rich terminal rendering of keyness results."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from wordcount.stats import KeynessResults, KeynessScore, SIG_P05, SIG_P001

# Display order for POS categories
_POS_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "AUX", "ADP", "CCONJ", "SCONJ", "PRON"]

_CORPUS_LABELS: dict[str, str] = {
    "literary": "Literary",
    "newspaper": "Newspaper",
    "general": "General",
    "modern": "Modern (all POS)",
}


def _format_score(score: float) -> Text:
    """Format a keyness score with sign, value, and significance-based colour."""
    abs_s = abs(score)
    sign = "+" if score >= 0 else ""
    formatted = f"{sign}{score:.1f}σ"

    if abs_s < SIG_P05:
        return Text(formatted, style="dim")
    elif score >= SIG_P001:
        return Text(formatted, style="bold green")
    elif score > 0:
        return Text(formatted, style="green")
    elif score <= -SIG_P001:
        return Text(formatted, style="bold red")
    else:
        return Text(formatted, style="red")


def _corpus_label(name: str) -> str:
    return _CORPUS_LABELS.get(name, name.capitalize())


def render(
    results: KeynessResults,
    top_n: int,
    pos_filter: tuple[str, ...],
    console: Console,
) -> None:
    """Render keyness results to the terminal."""

    # Header panel
    corpora_str = " · ".join(_corpus_label(c) for c in results.corpora_used)
    primary_label = _corpus_label(results.primary_corpus)
    header = (
        f"[bold]{results.user_file_count}[/] files · "
        f"[bold]{results.user_total_tokens:,}[/] tokens "
        f"([bold]{results.user_analyzed_tokens:,}[/] analysed)\n"
        f"Comparing against: [bold]{corpora_str}[/]\n"
        f"Ranked by: [bold]{primary_label}[/] · "
        f"Significance threshold: ±{SIG_P05}σ (p<0.05)"
    )
    console.print(Panel(header, title="[bold]wordcount[/]", expand=False))
    console.print()

    # Determine which POS tags to show
    if "all" in pos_filter:
        active_pos = _POS_ORDER
    else:
        active_pos = [p for p in _POS_ORDER if p in pos_filter]
        # Also include any user-specified POS not in our display order
        for p in pos_filter:
            if p not in active_pos and p != "all":
                active_pos.append(p)

    any_shown = False

    for pos in active_pos:
        items = results.by_pos.get(pos)
        if not items:
            continue

        # Split into over and under, filtering to significant only in primary corpus
        over = [i for i in items if i.scores.get(results.primary_corpus, 0) > 0][:top_n]
        under = [
            i for i in reversed(items)
            if i.scores.get(results.primary_corpus, 0) < 0
        ][:top_n]

        if not over and not under:
            continue

        any_shown = True
        console.rule(f"[bold]{pos}[/]")

        col_count = 3 + len(results.corpora_used)

        if over:
            console.print(Text("  ▲ OVER-REPRESENTED", style="green bold"))
            table_over = _build_table(over, results.corpora_used)
            console.print(table_over)

        if under:
            console.print(Text("  ▼ UNDER-REPRESENTED", style="red bold"))
            table_under = _build_table(under, results.corpora_used)
            console.print(table_under)

        console.print()

    if not any_shown:
        console.print(
            "[yellow]No significant outliers found.[/] "
            "Try lowering [bold]--min-freq[/] or adding more corpora."
        )


def _build_table(items: list[KeynessScore], corpora_used: list[str]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("Word", style="bold", min_width=14, no_wrap=True)
    table.add_column("Count", justify="right", min_width=6)
    table.add_column("Freq/10k", justify="right", min_width=8)
    for corpus_name in corpora_used:
        table.add_column(_corpus_label(corpus_name), justify="right", min_width=14)

    for item in items:
        table.add_row(
            item.lemma,
            str(item.user_count),
            f"{item.user_freq_per_10k:.1f}",
            *[_format_score(item.scores.get(c, 0.0)) for c in corpora_used],
        )

    return table
