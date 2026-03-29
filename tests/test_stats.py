"""Tests for G² keyness computation."""

import pytest
from wordcount.stats import _g2_signed, SIG_P05


def test_g2_equal_frequencies_near_zero():
    """Same relative frequency in both corpora → near-zero score."""
    score = _g2_signed(o11=100, n1=10_000, o21=100, n2=10_000)
    assert abs(score) < 0.1


def test_g2_strongly_overrepresented():
    """Much higher rate in user corpus → large positive score."""
    score = _g2_signed(o11=100, n1=1_000, o21=10, n2=1_000_000)
    assert score > 10


def test_g2_strongly_underrepresented():
    """Much higher rate in reference → large negative score."""
    score = _g2_signed(o11=1, n1=1_000_000, o21=1_000, n2=1_000_000)
    assert score < -5


def test_g2_zero_reference_no_crash():
    """Zero reference count should not raise (Laplace smoothing)."""
    score = _g2_signed(o11=50, n1=10_000, o21=0, n2=1_000_000)
    assert score > 0  # overrepresented


def test_g2_zero_user_count_no_crash():
    """Zero user count should not raise."""
    score = _g2_signed(o11=0, n1=10_000, o21=100, n2=1_000_000)
    assert score < 0  # underrepresented (or zero)


def test_g2_sign_positive_when_over():
    score = _g2_signed(o11=500, n1=10_000, o21=10, n2=10_000)
    assert score > 0


def test_g2_sign_negative_when_under():
    score = _g2_signed(o11=10, n1=10_000, o21=500, n2=10_000)
    assert score < 0
