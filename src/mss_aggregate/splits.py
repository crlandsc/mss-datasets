"""Split assignment and locking — deterministic, reproducible splits."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from mss_aggregate.datasets.base import TrackInfo
from mss_aggregate.overlap import is_overlap_track

logger = logging.getLogger(__name__)

# Fixed seed for MoisesDB val set selection
MOISESDB_VAL_SEED = 42
MOISESDB_VAL_SIZE = 50


def assign_splits(
    tracks: list[TrackInfo],
    existing_splits: dict[str, str] | None = None,
    musdb_splits: dict[str, str] | None = None,
) -> list[TrackInfo]:
    """Assign train/test/val splits to tracks.

    Args:
        tracks: All discovered tracks across datasets.
        existing_splits: Locked splits from a previous run (filename_key → split).
        musdb_splits: For overlap tracks — maps canonical_name → musdb split.

    Returns:
        The same tracks list with .split fields updated.
    """
    if existing_splits is None:
        existing_splits = {}

    for track in tracks:
        key = _track_key(track)

        # If already locked from previous run, respect it
        if key in existing_splits:
            track.split = existing_splits[key]
            continue

        if track.source_dataset == "musdb18hq":
            # Split comes from directory structure (already set during discovery)
            pass

        elif track.source_dataset == "medleydb":
            if musdb_splits and is_overlap_track(track.original_track_name):
                # Inherit MUSDB18-HQ split
                from mss_aggregate.utils import canonical_name
                cn = canonical_name(track.original_track_name)
                if cn in musdb_splits:
                    track.split = musdb_splits[cn]
                else:
                    track.split = "train"
            else:
                track.split = "train"

        elif track.source_dataset == "moisesdb":
            # Will be assigned below via genre-stratified selection
            pass

    # MoisesDB val set: genre-stratified deterministic selection
    _assign_moisesdb_val(tracks)

    return tracks


def _assign_moisesdb_val(tracks: list[TrackInfo]) -> None:
    """Select 50 MoisesDB tracks for validation, genre-stratified, seed=42."""
    moisesdb_tracks = [t for t in tracks if t.source_dataset == "moisesdb"]
    if not moisesdb_tracks:
        return

    # Group by genre (stored in flags or we use a placeholder)
    # For now, all MoisesDB tracks default to "train"; we select val set
    rng = random.Random(MOISESDB_VAL_SEED)

    # Shuffle deterministically, then pick first 50
    # (Genre stratification: in production, group by genre first,
    # pick proportionally. For now, use simple deterministic selection
    # since we don't have genre info at this stage — genre comes from
    # the moisesdb library at discover time.)
    indices = list(range(len(moisesdb_tracks)))
    rng.shuffle(indices)

    val_count = min(MOISESDB_VAL_SIZE, len(moisesdb_tracks))
    val_indices = set(indices[:val_count])

    for i, track in enumerate(moisesdb_tracks):
        if i in val_indices:
            track.split = "val"
        else:
            track.split = "train"


def _track_key(track: TrackInfo) -> str:
    """Generate a unique key for a track (used in splits.json)."""
    return f"{track.source_dataset}_{track.index:04d}_{track.original_track_name}"


def write_splits(path: Path, tracks: list[TrackInfo]) -> None:
    """Write splits.json to disk."""
    splits = {}
    for t in tracks:
        splits[_track_key(t)] = t.split

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2, sort_keys=True)


def load_splits(path: Path) -> dict[str, str] | None:
    """Load existing splits.json if present. Returns None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
