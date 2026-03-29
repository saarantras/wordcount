"""Parse markdown files into plain text, stripping markup."""

from __future__ import annotations

import re
from pathlib import Path

from markdown_it import MarkdownIt

_md = MarkdownIt()

_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)

# Block token types whose content should be skipped entirely
_SKIP_BLOCK_TYPES = frozenset({
    "fence",        # fenced code blocks ```...```
    "code_block",   # indented code blocks
    "html_block",   # raw HTML blocks
    "hr",           # thematic break ---
})


def strip_frontmatter(source: str) -> str:
    """Remove YAML/TOML frontmatter from the top of a markdown string."""
    return _FRONTMATTER_RE.sub("", source, count=1)


def strip_markdown(source: str) -> str:
    """Convert markdown to plain prose text.

    Keeps: all prose text, heading text, link text (not URLs), image alt text.
    Strips: code blocks, inline code, HTML, URLs, markdown syntax characters.
    """
    source = strip_frontmatter(source)
    tokens = _md.parse(source)
    parts: list[str] = []

    for token in tokens:
        if token.type in _SKIP_BLOCK_TYPES:
            continue
        if token.type == "inline" and token.children:
            for child in token.children:
                if child.type == "text":
                    parts.append(child.content)
                elif child.type in ("softbreak", "hardbreak"):
                    parts.append(" ")
                # Deliberately skip: code_inline, html_inline, image, link open/close

    return " ".join(parts)


def collect_texts(directory: Path) -> tuple[list[str], list[Path]]:
    """Recursively find all .md files and return (plain_texts, file_paths)."""
    paths = sorted(directory.rglob("*.md"))
    texts: list[str] = []
    found: list[Path] = []

    for p in paths:
        raw = p.read_text(encoding="utf-8", errors="replace")
        text = strip_markdown(raw).strip()
        if text:
            texts.append(text)
            found.append(p)

    return texts, found
