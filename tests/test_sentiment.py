"""Smoke test for the ModernBERT sentiment model (Checkpoint 2)."""

from __future__ import annotations

import sys

from core.sentiment_model import get_sentiment_model, score_single_headline


def test_sentiment_model_loads():
    """Verify the model loads and returns labelled output."""
    model = get_sentiment_model()
    info = model.get_model_info()
    assert "model_name" in info
    assert isinstance(info["labels"], dict) and len(info["labels"]) >= 2


def test_sentiment_scoring_range():
    """Scores should be in [-1.0, +1.0] with correct sign for known examples."""
    positive = score_single_headline("Apple beats earnings expectations with record revenue growth")
    negative = score_single_headline("Tesla announces massive layoffs amid declining sales")
    neutral = score_single_headline("Company X Y Z announces minor product update")

    assert -1.0 <= positive <= 1.0
    assert -1.0 <= negative <= 1.0
    assert -1.0 <= neutral <= 1.0

    assert positive > 0.5, f"positive headline scored {positive}"
    assert negative < -0.5, f"negative headline scored {negative}"
    assert abs(neutral) < 0.5, f"neutral headline scored {neutral}"