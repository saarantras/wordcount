"""Tests for corpora module."""

import pytest
from wordcount.corpora import wordfreq_lookup, CORPUS_GROUPS, BROWN_ALL_CATEGORIES


def test_wordfreq_common_word_nonzero():
    freq = wordfreq_lookup("the")
    assert freq > 0.01


def test_wordfreq_known_content_word():
    freq = wordfreq_lookup("castle")
    assert freq > 0.0


def test_wordfreq_unknown_word_zero():
    freq = wordfreq_lookup("xyzzy_nonexistent_12345_qqqq")
    assert freq == 0.0


def test_corpus_groups_keys():
    assert set(CORPUS_GROUPS.keys()) == {"newspaper", "literary", "general"}


def test_corpus_groups_categories_valid():
    for group, cats in CORPUS_GROUPS.items():
        for cat in cats:
            assert cat in BROWN_ALL_CATEGORIES, f"{cat} not in Brown categories"


def test_general_includes_all():
    assert set(CORPUS_GROUPS["general"]) == set(BROWN_ALL_CATEGORIES)


def test_newspaper_categories():
    assert set(CORPUS_GROUPS["newspaper"]) == {"news", "editorial", "reviews"}
