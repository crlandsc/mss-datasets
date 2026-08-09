"""Filename sanitization and general utilities."""

import re

from unidecode import unidecode


def sanitize_filename(source: str, artist: str, title: str) -> str:
    """Build a track's output filename stem (no .wav extension — caller appends).

    Format: {source}_{artist}_{title}

    Deliberately excludes the split and the discovery index. Both change when
    splits are reassigned or the dataset contents shift, and because nothing
    reconciles stale files a renamed track leaves its old copy behind — which
    with --split-output puts the same audio in both train/ and val/.

    Derived only from dataset metadata, so it is stable across runs.
    """
    name_part = _sanitize_text(f"{artist}_{title}")
    # Truncate artist+title to 80 chars
    if len(name_part) > 80:
        name_part = name_part[:80].rstrip("_")
    return f"{source}_{name_part}"


def resolve_collision(filename: str, existing: set[str]) -> str:
    """Append _2, _3, etc. if filename collides with existing set."""
    if filename not in existing:
        return filename
    i = 2
    while f"{filename}_{i}" in existing:
        i += 1
    return f"{filename}_{i}"


def _sanitize_text(text: str) -> str:
    """Transliterate, lowercase, replace non-alnum with underscores."""
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9_-]", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text


def canonical_name(name: str) -> str:
    """Normalize track name for overlap matching: lowercase, strip spaces/underscores/hyphens."""
    return re.sub(r"[\s_\-]", "", name.lower())
