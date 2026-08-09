"""Hardcoded MUSDB18-HQ ↔ MedleyDB overlap registry and resolution logic."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from mss_datasets.utils import canonical_name

logger = logging.getLogger(__name__)

# Which dataset wins when the same song appears in more than one.
#
# MedleyDB first: its per-instrument stems can be routed to any profile, while
# MUSDB18-HQ's "other" is pre-mixed and cannot be decomposed — so preferring
# MedleyDB is what makes the 46 shared tracks usable in the 6-stem profile.
DATASET_PRECEDENCE: tuple[str, ...] = ("medleydb", "musdb18hq", "moisesdb")

# 46 MUSDB18-HQ tracks that originate from MedleyDB (format: "Artist - Title")
MUSDB_MEDLEYDB_OVERLAP: frozenset[str] = frozenset([
    "A Classic Education - NightOwl",
    "Aimee Norwich - Child",
    "Alexander Ross - Goodbye Bolero",
    "Alexander Ross - Velvet Curtain",
    "Auctioneer - Our Future Faces",
    "AvaLuna - Waterduct",
    "BigTroubles - Phantom",
    "Celestial Shore - Die For Us",
    "Clara Berry And Wooldog - Air Traffic",
    "Clara Berry And Wooldog - Stella",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Creepoid - OldTree",
    "Dreamers Of The Ghetto - Heavy Love",
    "Faces On Film - Waiting For Ga",
    "Grants - PunchDrunk",
    "Helado Negro - Mitad Del Mundo",
    "Hezekiah Jones - Borrowed Heart",
    "Hop Along - Sister Cities",
    "Invisible Familiars - Disturbing Wildlife",
    "Lushlife - Toynbee Suite",
    "Matthew Entwistle - Dont You Ever",
    "Meaxic - Take A Step",
    "Meaxic - You Listen",
    "Music Delta - 80s Rock",
    "Music Delta - Beatles",
    "Music Delta - Britpop",
    "Music Delta - Country1",
    "Music Delta - Country2",
    "Music Delta - Disco",
    "Music Delta - Gospel",
    "Music Delta - Grunge",
    "Music Delta - Hendrix",
    "Music Delta - Punk",
    "Music Delta - Reggae",
    "Music Delta - Rock",
    "Music Delta - Rockabilly",
    "Night Panther - Fire",
    "Port St Willow - Stay Even",
    "Secret Mountains - High Horse",
    "Snowmine - Curfews",
    "Steven Clark - Bounty",
    "Strand Of Oaks - Spacestation",
    "Sweet Lights - You Let Me Down",
    "The Districts - Vermont",
    "The Scarlet Brand - Les Fleurs Du Mal",
    "The So So Glos - Emergency",
])

# Pre-computed canonical forms for fast lookup
_CANONICAL_OVERLAP: frozenset[str] = frozenset(
    canonical_name(name) for name in MUSDB_MEDLEYDB_OVERLAP
)


def get_overlap_set() -> frozenset[str]:
    """Return the set of 46 overlap track names (MUSDB18-HQ format)."""
    return MUSDB_MEDLEYDB_OVERLAP


def is_overlap_track(track_name: str) -> bool:
    """Check if a track name (from either dataset) matches the overlap list.

    Uses canonical normalization for cross-dataset matching.
    """
    return canonical_name(track_name) in _CANONICAL_OVERLAP


@dataclass(frozen=True)
class OverlapGroup:
    """One song found in more than one place, and which copy we keep."""

    canonical: str
    winner: tuple[str, str]                    # (dataset, original_track_name)
    losers: tuple[tuple[str, str], ...]

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted({self.winner[0]} | {d for d, _ in self.losers}))


def resolve_cross_dataset(
    tracks,
    precedence: tuple[str, ...] = DATASET_PRECEDENCE,
    prefer_usable: bool = True,
) -> list[OverlapGroup]:
    """Find songs present more than once and pick a single copy of each.

    Unlike :func:`resolve_overlaps`, which only ever compares MUSDB18-HQ against
    the hardcoded MedleyDB list, this compares every discovered track against
    every other by canonical name — so a duplicate involving MoisesDB, or one
    inside a single dataset, is caught rather than silently kept twice.

    Args:
        tracks: objects exposing ``source_dataset``, ``original_track_name`` and
            ``has_bleed``.
        precedence: dataset names in priority order; anything unlisted ranks last.
        prefer_usable: when True a track that will survive the bleed filter beats
            a higher-precedence one that will not. Without this a bleed-flagged
            winner takes the slot and is then dropped, losing the song entirely.
            Set False when bleed tracks are being kept anyway.

    Returns:
        One :class:`OverlapGroup` per duplicated song. Empty when nothing collides.
    """
    by_canonical: dict[str, list] = defaultdict(list)
    for track in tracks:
        by_canonical[canonical_name(track.original_track_name)].append(track)

    rank = {name: i for i, name in enumerate(precedence)}
    unranked = len(precedence)

    groups: list[OverlapGroup] = []
    for canonical, candidates in sorted(by_canonical.items()):
        if len(candidates) < 2:
            continue

        sources = {t.source_dataset for t in candidates}
        if len(sources) == 1:
            # Two tracks in one dataset that normalize to the same name are
            # reported but never dropped — they may be genuinely different songs
            # that differ only in punctuation, and silently losing one would be
            # worse than keeping a possible duplicate.
            logger.warning(
                "%d tracks inside %s share the canonical name %r — keeping both",
                len(candidates), candidates[0].source_dataset, canonical,
            )
            continue

        ordered = sorted(
            candidates,
            key=lambda t: (
                bool(t.has_bleed) if prefer_usable else False,
                rank.get(t.source_dataset, unranked),
                t.source_dataset,
                t.original_track_name,
            ),
        )
        winner = ordered[0]
        # Only other datasets lose. Any same-dataset sibling of the winner is
        # kept, for the reason above.
        losers = [t for t in ordered if t.source_dataset != winner.source_dataset]
        groups.append(
            OverlapGroup(
                canonical=canonical,
                winner=(winner.source_dataset, winner.original_track_name),
                losers=tuple(
                    (t.source_dataset, t.original_track_name) for t in losers
                ),
            )
        )

    return groups


def resolve_overlaps(
    musdb_track_names: list[str],
    medleydb_present: bool,
) -> dict:
    """Determine which MUSDB18-HQ tracks to skip and which MedleyDB tracks inherit splits.

    Args:
        musdb_track_names: List of MUSDB18-HQ track names (format: "Artist - Title")
        medleydb_present: Whether MedleyDB dataset is also being processed

    Returns:
        dict with keys:
            skip_musdb: set of MUSDB18-HQ track names to skip
            musdb_splits: dict mapping canonical overlap name → MUSDB18-HQ split
                          (for MedleyDB tracks to inherit)
    """
    if not medleydb_present:
        return {"skip_musdb": set(), "musdb_splits": {}}

    skip_musdb = set()
    musdb_splits = {}

    for name in musdb_track_names:
        if is_overlap_track(name):
            skip_musdb.add(name)
            # We need the split info — caller provides it via track info
            # Store canonical name for matching
            musdb_splits[canonical_name(name)] = name

    return {"skip_musdb": skip_musdb, "musdb_splits": musdb_splits}
