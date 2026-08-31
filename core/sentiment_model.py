"""Financial news sentiment analysis using ModernBERT-based models."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("robinhood-agent.core.sentiment_model")

PRIMARY_MODEL = "AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis"
FALLBACK_MODEL = "AnkitAI/distilbert-base-uncased-financial-news-sentiment-analysis"


class SentimentModel:
    """Singleton sentiment analyzer with lazy loading and GPU/CPU auto-detection."""

    _instance: Optional["SentimentModel"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SentimentModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None
        self.id2label = None
        self.label2id = None
        self._initialized = True
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentiment model with fallback support."""
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("sentiment_model using CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("sentiment_model using MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("sentiment_model using CPU")

        # Try primary model first, then fallback
        for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                logger.info("sentiment_model loading %s", model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
                self.model.to(self.device)
                self.model.eval()

                # Get label mapping from model config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                self.model_name = model_name

                logger.info("sentiment_model loaded successfully: %s (labels: %s)", model_name, self.id2label)
                return

            except Exception as exc:
                logger.warning("sentiment_model failed to load %s: %s", model_name, exc)
                continue

        raise RuntimeError("Failed to load any sentiment model (tried primary and fallback)")
    def _get_sentiment_score(self, logits: torch.Tensor) -> float:
        """Convert model logits to continuous sentiment score in [-1, +1]."""
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Map labels to sentiment values
        # Typical financial sentiment: negative=0, neutral=1, positive=2
        # Score = P(positive) - P(negative)  -> range [-1, +1]
        # Neutral is ignored in the score but contributes to confidence

        negative_idx = self.label2id.get("negative", self.label2id.get("NEGATIVE", 0))
        positive_idx = self.label2id.get("positive", self.label2id.get("POSITIVE", 2))
        neutral_idx = self.label2id.get("neutral", self.label2id.get("NEUTRAL", 1))

        p_neg = float(probs[negative_idx]) if negative_idx < len(probs) else 0.0
        p_pos = float(probs[positive_idx]) if positive_idx < len(probs) else 0.0

        # Continuous score: positive - negative (neutral doesn't directly affect score)
        score = p_pos - p_neg

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))

    def score_headlines(self, headlines: List[str], batch_size: int = 16) -> List[float]:
        """Score a list of headlines, returning sentiment scores in [-1, +1].

        Args:
            headlines: List of news headlines/text to score
            batch_size: Batch size for inference (adjust for memory)

        Returns:
            List of sentiment scores, one per headline
        """
        if not headlines:
            return []

        if self.model is None or self.tokenizer is None:
            logger.error("sentiment_model not loaded, returning neutral scores")
            return [0.0] * len(headlines)

        scores = []

        # Process in batches
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]

            try:
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Score each item in batch
                for j in range(logits.shape[0]):
                    score = self._get_sentiment_score(logits[j : j + 1])
                    scores.append(score)

            except Exception as exc:
                logger.warning("sentiment_model batch scoring failed: %s", exc)
                # Fill with neutral for failed batch
                scores.extend([0.0] * len(batch))

        return scores

    def score_single(self, text: str) -> float:
        """Score a single headline/text."""
        scores = self.score_headlines([text])
        return scores[0] if scores else 0.0

    def get_model_info(self) -> dict:
        """Return info about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "labels": self.id2label,
        }

@lru_cache(maxsize=1)
def get_sentiment_model() -> SentimentModel:
    """Get the singleton sentiment model instance (cached)."""
    return SentimentModel()


def score_news_items(headlines: List[str]) -> List[float]:
    """Convenience function to score headlines using the singleton model."""
    model = get_sentiment_model()
    return model.score_headlines(headlines)


def score_single_headline(headline: str) -> float:
    """Convenience function to score a single headline."""
    model = get_sentiment_model()
    return model.score_single(headline)