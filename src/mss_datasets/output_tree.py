"""Reader for an already-aggregated output tree — shared by inventory and audit.

Both of those operate on a finished output directory rather than on source
datasets, so the scanning, filename parsing and metadata loading live here once.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from mss_datasets.mapping.profiles import PROFILES
from mss_datasets.overlap import is_overlap_track
from mss_datasets.utils import _sanitize_text, canonical_name

logger = logging.getLogger(__name__)

# Every stem name any profile can emit, plus the optional mixture folder.
STEM_NAMES: frozenset[str] = frozenset(
    {"mixture"} | {s for p in PROFILES.values() for s in p.stems}
)

SOURCE_NAMES: tuple[str, ...] = ("musdb18hq", "medleydb", "moisesdb")

_SOURCES = "|".join(SOURCE_NAMES)

# Trees written before track identity was stabilized use
# {source}_{split}_{index:04d}_{name}. Those files are still on disk and must
# keep parsing, so both layouts are supported.
LEGACY_FILENAME_RE = re.compile(
    r"^(?P<source>" + _SOURCES + r")"
    r"_(?P<split>train|val|test)"
    r"_(?P<index>\d{4})"
    r"_(?P<name>.+)$"
)

# Current layout: {source}_{artist}_{title}. The trailing name may contain
# underscores, so the source token is the only anchor — which means a legacy
# name also matches here (yielding name="train_0001_…"). Legacy must be tried
# first; the order is load-bearing.
FILENAME_RE = re.compile(r"^(?P<source>" + _SOURCES + r")_(?P<name>.+)$")

# Review views — how tracks are regrouped for human review. MUSDB18-HQ and
# MoisesDB read as their complete original rosters; MedleyDB shows only what it
# uniquely adds.
VIEW_MUSDB = "musdb18hq"
VIEW_MOISESDB = "moisesdb"
VIEW_MEDLEYDB_EXTRA = "medleydb_extra"
VIEWS: tuple[str, ...] = (VIEW_MUSDB, VIEW_MOISESDB, VIEW_MEDLEYDB_EXTRA)


@dataclass(frozen=True)
class OutputFile:
    """One WAV in the output tree, with its location decomposed."""

    path: Path
    split_dir: str      # "train"/"val", or "" when --split-output was off
    stem: str           # vocals, drums, bass, other, guitar, piano, mixture
    dataset_dir: str    # source subfolder, or "" when --group-by-dataset was off
    source: str         # parsed from the filename
    split: str          # legacy filenames only; "" under the current layout
    index: str          # legacy filenames only; "" under the current layout
    name: str           # sanitized artist_title
    legacy: bool = False


@dataclass
class Track:
    """All files belonging to one track."""

    source: str
    split: str
    index: str
    name: str
    split_dir: str
    files: dict[str, Path] = field(default_factory=dict)   # stem -> path
    original_track_name: str = ""
    legacy: bool = False

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Location-aware key.

        Includes the split directory so that a track wrongly written into two
        splits stays two records — collapsing them would hide the leakage and
        silently drop one of the two file paths.
        """
        return (self.source, self.split_dir, self.index, self.name)

    @property
    def identity(self) -> tuple[str, str]:
        """Cross-dataset identity, normalized for comparison."""
        return (self.source, canonical_name(self.name))

    @property
    def stems(self) -> set[str]:
        return set(self.files)

    def view(self) -> str:
        """Which review view this track belongs to.

        MedleyDB tracks that stand in for a MUSDB18-HQ track are shown under the
        MUSDB view, since that is the dataset a reader identifies them with.
        """
        if self.source != "medleydb":
            return VIEW_MUSDB if self.source == "musdb18hq" else VIEW_MOISESDB
        probe = self.original_track_name or self.name
        return VIEW_MUSDB if is_overlap_track(probe) else VIEW_MEDLEYDB_EXTRA


class OutputTree:
    """An aggregated output directory, loaded into memory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files: list[OutputFile] = []
        self.unparsed: list[Path] = []
        self.tracks: dict[tuple[str, str, str, str], Track] = {}
        self.manifest: dict = {}
        self.splits: dict = {}
        self.overlap_registry: dict = {}
        self.errors: list = []

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    def load(self) -> OutputTree:
        """Scan the tree and read whatever metadata is present."""
        if not self.root.is_dir():
            raise ValueError(f"Output directory not found: {self.root}")
        self._load_metadata()
        self._scan()
        self._build_tracks()
        return self

    def _load_metadata(self) -> None:
        for attr, fname in (
            ("manifest", "manifest.json"),
            ("splits", "splits.json"),
            ("overlap_registry", "overlap_registry.json"),
            ("errors", "errors.json"),
        ):
            path = self.metadata_dir / fname
            if not path.exists():
                logger.warning("No %s in %s", fname, self.metadata_dir)
                continue
            try:
                with open(path) as f:
                    setattr(self, attr, json.load(f))
            except Exception as e:
                logger.error("Failed to read %s: %s", path, e)

    def _scan(self) -> None:
        """Walk the tree, skipping metadata and generated view directories."""
        for wav in sorted(self.root.rglob("*.wav")):
            rel = wav.relative_to(self.root)
            parts = rel.parts[:-1]
            # Skip metadata/, dotfiles, and any generated view dir (_sorted etc.)
            if any(p == "metadata" or p.startswith((".", "_")) for p in parts):
                continue
            # Aggregation only ever writes {stem}/… or {split}/{stem}/…, so a
            # stem folder any deeper belongs to something else living alongside
            # the dataset (rendered clips, ablation exports) — not ours to audit.
            stem_idx = next(
                (i for i, p in enumerate(parts[:2]) if p in STEM_NAMES), None
            )
            if stem_idx is None:
                continue
            parsed = self._parse(wav, parts, stem_idx)
            if parsed is None:
                self.unparsed.append(wav)
            else:
                self.files.append(parsed)

    def _parse(self, wav: Path, parts: tuple[str, ...], stem_idx: int) -> OutputFile | None:
        """Decompose a WAV's path and filename, or None if the name doesn't fit."""
        match = LEGACY_FILENAME_RE.match(wav.stem)
        legacy = match is not None
        if match is None:
            match = FILENAME_RE.match(wav.stem)
        if match is None:
            return None

        groups = match.groupdict()
        return OutputFile(
            path=wav,
            split_dir=parts[stem_idx - 1] if stem_idx > 0 else "",
            stem=parts[stem_idx],
            dataset_dir=parts[stem_idx + 1] if stem_idx + 1 < len(parts) else "",
            source=groups["source"],
            split=groups.get("split") or "",
            index=groups.get("index") or "",
            name=groups["name"],
            legacy=legacy,
        )

    def _build_tracks(self) -> None:
        """Group files into tracks and attach original names from the manifest."""
        originals = self._manifest_names()
        for f in self.files:
            key = (f.source, f.split_dir, f.index, f.name)
            track = self.tracks.get(key)
            if track is None:
                track = Track(
                    source=f.source, split=f.split, index=f.index,
                    name=f.name, split_dir=f.split_dir,
                    original_track_name=originals.get((f.source, f.name), ""),
                    legacy=f.legacy,
                )
                self.tracks[key] = track
            track.files[f.stem] = f.path

    def _manifest_names(self) -> dict[tuple[str, str], str]:
        """Map (source, sanitized name) -> original_track_name from the manifest.

        The manifest doesn't record the numeric index, so join on the sanitized
        artist_title instead — that is exactly what the filename encodes.
        """
        out: dict[tuple[str, str], str] = {}
        for entry in self.manifest.values():
            source = entry.get("source_dataset", "")
            name = _sanitize_text(f"{entry.get('artist', '')}_{entry.get('title', '')}")
            if len(name) > 80:
                name = name[:80].rstrip("_")
            out[(source, name)] = entry.get("original_track_name", "")
        return out

    def tracks_by_view(self) -> dict[str, list[Track]]:
        """Group tracks into the three review views."""
        grouped: dict[str, list[Track]] = {v: [] for v in VIEWS}
        for track in self.tracks.values():
            grouped[track.view()].append(track)
        for tracks in grouped.values():
            tracks.sort(key=lambda t: (t.original_track_name or t.name).lower())
        return grouped
