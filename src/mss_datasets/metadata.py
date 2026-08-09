"""Metadata file generation — manifest, errors, overlap registry, config."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ManifestEntry:
    source_dataset: str
    original_track_name: str
    artist: str
    title: str
    split: str
    available_stems: list[str]
    profile: str
    license: str = ""
    duration_seconds: float = 0.0
    is_composite_sum: bool = False
    has_bleed: bool = False
    musdb18hq_4stem_only: bool = False
    flags: list[str] = field(default_factory=list)
    # Output filename stem. Unique and stable, so it doubles as the track's
    # identity for the split lock and the resume ledger.
    filename_base: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> ManifestEntry:
        """Rebuild an entry from a manifest read off disk.

        Unknown keys are dropped so a manifest written by a different version
        still loads instead of raising.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ErrorEntry:
    track: str
    dataset: str
    error: str
    stage: str
    skipped: bool = True


LICENSE_MAP = {
    "musdb18hq": "Custom (non-commercial/academic only)",
    "moisesdb": "CC BY-NC-SA 4.0",
    "medleydb": "CC BY-NC-SA 4.0",
}


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    """Write manifest.json with per-track metadata."""
    manifest = {}
    for entry in entries:
        # The filename base is unique per track; fall back to the old
        # split-qualified key for entries built without one.
        key = entry.filename_base or (
            f"{entry.source_dataset}_{entry.split}_{entry.original_track_name}"
        )
        manifest[key] = asdict(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: Path) -> dict:
    """Load an existing manifest.json, or return {} when there isn't one.

    The pipeline uses this as its resume ledger: it records what was actually
    written for each track last time, which is the only reliable way to tell a
    finished track from one interrupted midway.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # A corrupt or unreadable manifest must not abort the run — treat it as
        # absent and reprocess.
        return {}


def write_errors(path: Path, errors: list[ErrorEntry]) -> None:
    """Write errors.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in errors], f, indent=2)


def write_overlap_registry(
    path: Path,
    skipped_musdb_tracks: list[str],
    groups: list | None = None,
    reason: str = "MedleyDB preferred (more granular stems)",
) -> None:
    """Write overlap_registry.json documenting deduplicated tracks.

    ``skipped_tracks`` / ``skipped_count`` are the MUSDB18-HQ-only view and are
    kept unchanged for backwards compatibility. ``groups`` records every
    resolved duplicate — including ones not involving MUSDB18-HQ — with the
    dataset that won each.
    """
    registry = {
        "description": "Tracks skipped because the same song was kept from another dataset",
        "reason": reason,
        "skipped_count": len(skipped_musdb_tracks),
        "skipped_tracks": sorted(skipped_musdb_tracks),
    }

    if groups is not None:
        registry["total_skipped"] = sum(len(g.losers) for g in groups)
        registry["groups"] = [
            {
                "canonical": g.canonical,
                "datasets": list(g.datasets),
                "kept": {"dataset": g.winner[0], "track": g.winner[1]},
                "skipped": [
                    {"dataset": dataset, "track": track} for dataset, track in g.losers
                ],
            }
            for g in groups
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def write_config(path: Path, effective_config: dict) -> None:
    """Write effective configuration as config.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(effective_config, f, default_flow_style=False, sort_keys=True)
