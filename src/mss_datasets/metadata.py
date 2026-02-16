"""Metadata file generation — manifest, errors, overlap registry, config."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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


@dataclass
class ErrorEntry:
    track: str
    dataset: str
    error: str
    stage: str
    skipped: bool = True


LICENSE_MAP = {
    "musdb18hq": "academic-use-only",
    "moisesdb": "non-commercial-research",
    "medleydb": "CC BY-NC-SA 4.0",
}


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    """Write manifest.json with per-track metadata."""
    manifest = {}
    for entry in entries:
        # Build key from dataset + index info embedded in the entry
        key = f"{entry.source_dataset}_{entry.split}_{entry.original_track_name}"
        manifest[key] = asdict(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_errors(path: Path, errors: list[ErrorEntry]) -> None:
    """Write errors.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in errors], f, indent=2)


def write_overlap_registry(
    path: Path,
    skipped_musdb_tracks: list[str],
    reason: str = "MedleyDB preferred (more granular stems)",
) -> None:
    """Write overlap_registry.json documenting skipped MUSDB18-HQ tracks."""
    registry = {
        "description": "MUSDB18-HQ tracks skipped due to cross-dataset overlap with MedleyDB",
        "reason": reason,
        "skipped_count": len(skipped_musdb_tracks),
        "skipped_tracks": sorted(skipped_musdb_tracks),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def write_config(path: Path, effective_config: dict) -> None:
    """Write effective configuration as config.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(effective_config, f, default_flow_style=False, sort_keys=True)
