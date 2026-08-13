"""Structured news model for Finnhub news items.

Provides a Pydantic model for typed, validated news from Finnhub's
REST /company-news and /news endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel


class NewsItem(BaseModel):
    """Single news article from Finnhub.

    Fields map directly to the Finnhub REST API response schema.
    datetime is Unix epoch seconds.
    """

    id: int
    headline: str
    summary: str | None = None
    source: str
    url: str | None = None
    image: str | None = None
    related: str = ""  # ticker this article is associated with
    category: str = ""  # "company", "top news", etc.
    datetime: int  # unix epoch seconds