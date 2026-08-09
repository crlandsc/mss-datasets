"""Split assignment and locking — deterministic, reproducible splits."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from mss_datasets.datasets.base import TrackInfo
from mss_datasets.overlap import is_overlap_track

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

    locked: set[int] = set()
    for track in tracks:
        key = _track_key(track)

        # If already locked from previous run, respect it
        if key in existing_splits:
            track.split = existing_splits[key]
            locked.add(id(track))
            continue

        if track.source_dataset == "musdb18hq":
            # Split comes from directory structure (already set during discovery)
            pass

        elif track.source_dataset == "medleydb":
            if musdb_splits and is_overlap_track(track.original_track_name):
                # Inherit MUSDB18-HQ split
                from mss_datasets.utils import canonical_name
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

    # MoisesDB val set: deterministic selection over whatever is not locked
    _assign_moisesdb_val(tracks, locked=locked)

    return tracks


def _assign_moisesdb_val(
    tracks: list[TrackInfo], locked: set[int] | None = None
) -> None:
    """Select MoisesDB validation tracks deterministically, seed=42.

    Sorted by track name before shuffling, so the selection depends only on
    *which* tracks exist and not on the order the moisesdb library happened to
    yield them. Previously it shuffled list positions, so a library upgrade or a
    change in filesystem order would silently move tracks across the train/val
    boundary.

    Tracks already locked by a previous run's splits.json are left alone; the
    remainder are filled up to the target count.
    """
    locked = locked or set()
    moisesdb_tracks = sorted(
        (t for t in tracks if t.source_dataset == "moisesdb"),
        key=lambda t: t.original_track_name,
    )
    if not moisesdb_tracks:
        return

    free = [t for t in moisesdb_tracks if id(t) not in locked]
    already_val = sum(
        1 for t in moisesdb_tracks if id(t) in locked and t.split == "val"
    )
    target = min(MOISESDB_VAL_SIZE, len(moisesdb_tracks)) - already_val

    if target <= 0:
        for track in free:
            track.split = "train"
        return

    rng = random.Random(MOISESDB_VAL_SEED)
    order = list(range(len(free)))
    rng.shuffle(order)
    val_indices = set(order[:target])

    for i, track in enumerate(free):
        track.split = "val" if i in val_indices else "train"


def _track_key(track: TrackInfo) -> str:
    """Stable, unique key for a track in splits.json.

    Uses the collision-resolved filename base the pipeline assigns. That is
    derived only from dataset metadata, so it survives re-splits and dataset
    changes, and it is unique even when two tracks share a name — unlike the
    name alone, and unlike the old discovery index, which shifted whenever the
    dataset contents or exclusion overrides changed and silently broke the lock.

    Falls back to the bare name when no base has been assigned, so the function
    stays usable on a TrackInfo built outside the pipeline.
    """
    return (
        track.filename_base
        or f"{track.source_dataset}_{track.original_track_name}"
    )


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
