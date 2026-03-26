from __future__ import annotations

from functools import lru_cache
from typing import Set


_FALLBACK_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but",
    "on", "in", "at", "of", "to", "for", "from", "with", "without", "by", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "our", "their",
    "while", "when", "where", "who", "whom", "which",
}


@lru_cache(maxsize=1)
def get_english_stopwords() -> Set[str]:
    """
    Returns a set of English stopwords.

    Uses NLTK stopwords if available; otherwise falls back to a small built-in list.
    """
    try:
        from nltk.corpus import stopwords  # type: ignore

        # This can raise LookupError if the corpus isn't downloaded.
        words = stopwords.words("english")
        return set(w.lower() for w in words)
    except Exception:
        return _FALLBACK_STOPWORDS
