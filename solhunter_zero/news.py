from __future__ import annotations

import logging
import re
import requests
import xml.etree.ElementTree as ET
from typing import Iterable, List

logger = logging.getLogger(__name__)

# Simple positive/negative word lists for naive sentiment analysis
POSITIVE_WORDS = {
    "good",
    "great",
    "gain",
    "gains",
    "surge",
    "rally",
    "bull",
    "soar",
    "positive",
    "up",
}
NEGATIVE_WORDS = {
    "bad",
    "loss",
    "losses",
    "drop",
    "bear",
    "crash",
    "negative",
    "down",
    "plunge",
}

def fetch_headlines(feed_urls: Iterable[str], allowed: Iterable[str] | None = None) -> List[str]:
    """Return headlines from *feed_urls* that are present in ``allowed``.

    Any feeds not included in ``allowed`` are ignored.  If ``allowed`` is
    ``None``, all URLs are permitted.
    """
    allowed_set = {url for url in allowed} if allowed is not None else None
    headlines: List[str] = []
    for url in feed_urls:
        if allowed_set is not None and url not in allowed_set:
            logger.warning("Blocked RSS feed: %s", url)
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch feed %s: %s", url, exc)
            continue
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:  # pragma: no cover - bad xml
            logger.warning("Failed to parse feed %s: %s", url, exc)
            continue
        for item in root.findall(".//item/title"):
            if item.text:
                headlines.append(item.text.strip())
    return headlines

def compute_sentiment(text: str) -> float:
    """Return a sentiment score for ``text`` between -1 and 1."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    score = 0
    for w in words:
        if w in POSITIVE_WORDS:
            score += 1
        elif w in NEGATIVE_WORDS:
            score -= 1
    return score / len(words)

def fetch_sentiment(feed_urls: Iterable[str], allowed: Iterable[str] | None = None) -> float:
    """Return overall sentiment for ``feed_urls``."""
    headlines = fetch_headlines(feed_urls, allowed)
    if not headlines:
        return 0.0
    text = " ".join(headlines)
    return compute_sentiment(text)
