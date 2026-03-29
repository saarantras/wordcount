"""Tests for markdown parser."""

import pytest
from wordcount.parser import strip_markdown, strip_frontmatter


def test_strips_fenced_code_blocks():
    result = strip_markdown("prose\n\n```python\ncode here\n```\nmore prose")
    assert "code here" not in result
    assert "prose" in result
    assert "more prose" in result


def test_strips_indented_code_blocks():
    result = strip_markdown("prose\n\n    indented code\n\nmore prose")
    assert "indented code" not in result
    assert "prose" in result


def test_keeps_link_text():
    result = strip_markdown("[click here](http://example.com)")
    assert "click here" in result
    assert "example.com" not in result


def test_keeps_heading_text():
    result = strip_markdown("# My Title\n\nBody text.")
    assert "My Title" in result
    assert "#" not in result


def test_keeps_bold_text():
    result = strip_markdown("This is **important** text.")
    assert "important" in result


def test_strips_frontmatter():
    source = "---\ntitle: Hello\ndate: 2024\n---\n\nBody text."
    result = strip_frontmatter(source)
    assert "title" not in result
    assert "Body text" in result


def test_strips_frontmatter_via_strip_markdown():
    source = "---\ntitle: Hello\n---\n\nBody text."
    result = strip_markdown(source)
    assert "title" not in result
    assert "Body" in result


def test_empty_string():
    assert strip_markdown("") == ""


def test_only_code():
    result = strip_markdown("```\ncode\n```")
    assert result.strip() == ""
