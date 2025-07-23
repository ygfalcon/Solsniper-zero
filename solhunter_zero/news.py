from __future__ import annotations

import logging
import requests
import xml.etree.ElementTree as ET
from typing import Iterable, List

from transformers import pipeline

logger = logging.getLogger(__name__)

# Cache for the HuggingFace pipeline
_pipeline: pipeline | None = None


def get_pipeline() -> pipeline:
    """Return the shared DistilBERT sentiment pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _pipeline

def fetch_headlines(
    feed_urls: Iterable[str],
    allowed: Iterable[str] | None = None,
    *,
    twitter_urls: Iterable[str] | None = None,
    discord_urls: Iterable[str] | None = None,
) -> List[str]:
    """Return headlines and posts from the configured URLs.

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

    for url in twitter_urls or []:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch twitter feed %s: %s", url, exc)
            continue
        posts = data.get("posts", [])
        for post in posts:
            if isinstance(post, str):
                headlines.append(post.strip())

    for url in discord_urls or []:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch discord feed %s: %s", url, exc)
            continue
        messages = data.get("messages", [])
        for msg in messages:
            if isinstance(msg, str):
                headlines.append(msg.strip())

    return headlines

def compute_sentiment(text: str) -> float:
    """Return a sentiment score for ``text`` between -1 and 1 using DistilBERT."""
    if not text.strip():
        return 0.0
    clf = get_pipeline()
    try:
        result = clf(text)[0]
    except Exception as exc:  # pragma: no cover - unexpected model failures
        logger.warning("Sentiment model failure: %s", exc)
        return 0.0
    label = result.get("label", "").lower()
    score = float(result.get("score", 0.0))
    if label.startswith("neg"):
        return -score
    return score

def fetch_sentiment(
    feed_urls: Iterable[str],
    allowed: Iterable[str] | None = None,
    *,
    twitter_urls: Iterable[str] | None = None,
    discord_urls: Iterable[str] | None = None,
) -> float:
    """Return overall sentiment for provided sources."""
    headlines = fetch_headlines(
        feed_urls,
        allowed,
        twitter_urls=twitter_urls,
        discord_urls=discord_urls,
    )
    if not headlines:
        return 0.0
    text = " ".join(headlines)
    return compute_sentiment(text)
