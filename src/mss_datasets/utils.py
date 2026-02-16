"""Filename sanitization and general utilities."""

import re

from unidecode import unidecode


def sanitize_filename(source: str, split: str, index: int, artist: str, title: str) -> str:
    """Build sanitized filename per §6.2.

    Format: {source}_{split}_{index:04d}_{artist}_{title}
    (no .wav extension — caller appends)
    """
    name_part = _sanitize_text(f"{artist}_{title}")
    # Truncate artist+title to 80 chars
    if len(name_part) > 80:
        name_part = name_part[:80].rstrip("_")
    return f"{source}_{split}_{index:04d}_{name_part}"


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
