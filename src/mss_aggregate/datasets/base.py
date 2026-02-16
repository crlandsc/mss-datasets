"""Abstract base for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from mss_aggregate.mapping.profiles import StemProfile


@dataclass
class TrackInfo:
    source_dataset: str
    artist: str
    title: str
    split: str  # "train", "test", or "val"
    path: Path
    index: int = 0  # 1-based, assigned during discovery
    stems_available: list[str] = field(default_factory=list)
    has_bleed: bool = False
    flags: list[str] = field(default_factory=list)
    original_track_name: str = ""

    @property
    def track_name(self) -> str:
        """Original-format track name for display/metadata."""
        return self.original_track_name or f"{self.artist} - {self.title}"


class DatasetAdapter(ABC):
    """Base class for dataset adapters."""

    name: str  # e.g. "musdb18hq", "moisesdb", "medleydb"

    @abstractmethod
    def validate_path(self) -> None:
        """Validate dataset directory structure. Raise ValueError if invalid."""

    @abstractmethod
    def discover_tracks(self) -> list[TrackInfo]:
        """Discover and return all tracks in the dataset."""

    @abstractmethod
    def process_track(
        self,
        track: TrackInfo,
        profile: StemProfile,
        output_dir: Path,
        group_by_dataset: bool = False,
    ) -> dict:
        """Process a single track: read stems, map, normalize, write output.

        Returns dict with metadata about the processed track.
        """
