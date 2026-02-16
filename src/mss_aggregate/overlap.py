"""Hardcoded MUSDB18-HQ ↔ MedleyDB overlap registry and resolution logic."""

from __future__ import annotations

from mss_aggregate.utils import canonical_name

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
