"""MoisesDB dataset adapter — uses moisesdb library with custom sub-stem routing."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from mss_aggregate.audio import ensure_float32, ensure_stereo, sum_stems, write_wav_atomic
from mss_aggregate.datasets.base import DatasetAdapter, TrackInfo
from mss_aggregate.mapping.profiles import (
    BASS_ROUTING,
    PERCUSSION_ROUTING,
    StemProfile,
    get_moisesdb_mapping,
)
from mss_aggregate.utils import sanitize_filename

logger = logging.getLogger(__name__)


class MoisesdbAdapter(DatasetAdapter):
    name = "moisesdb"

    def __init__(self, path: str | Path, sample_rate: int = 44100):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._db = None

    def _get_db(self):
        if self._db is None:
            try:
                from moisesdb.db import MoisesDB
            except ImportError:
                raise ImportError(
                    "moisesdb is required to process MoisesDB. "
                    "Install it: pip install mss-aggregate[moisesdb]"
                )
            self._db = MoisesDB(data_path=str(self.path), sample_rate=self.sample_rate)
        return self._db

    def validate_path(self) -> None:
        moisesdb_dir = self.path / "moisesdb_v0.1"
        if not moisesdb_dir.is_dir():
            raise ValueError(f"MoisesDB missing moisesdb_v0.1/ directory: {moisesdb_dir}")

    def discover_tracks(self) -> list[TrackInfo]:
        db = self._get_db()
        tracks = []

        for i, track in enumerate(db, 1):
            tracks.append(TrackInfo(
                source_dataset=self.name,
                artist=track.artist,
                title=track.name,
                split="train",  # Default; val set assigned in splits.py
                path=Path(track.path) if hasattr(track, "path") else self.path,
                index=i,
                stems_available=list(track.stems.keys()) if hasattr(track, "stems") else [],
                has_bleed=bool(getattr(track, "bleedings", None)),
                original_track_name=f"{track.artist} - {track.name}",
                flags=[],
            ))

        return tracks

    def process_track(
        self,
        track: TrackInfo,
        profile: StemProfile,
        output_dir: Path,
        group_by_dataset: bool = False,
        _moisesdb_track=None,
    ) -> dict:
        """Process a MoisesDB track.

        Args:
            _moisesdb_track: Optional pre-loaded moisesdb track object (for testing).
        """
        db = self._get_db() if _moisesdb_track is None else None
        mtrack = _moisesdb_track

        if mtrack is None:
            # Find the track in the db by matching index
            for t in db:
                if f"{t.artist} - {t.name}" == track.original_track_name:
                    mtrack = t
                    break
            if mtrack is None:
                raise ValueError(f"Track not found in MoisesDB: {track.original_track_name}")

        custom_mapping = get_moisesdb_mapping(profile)

        # Accumulate output audio per category
        category_audio: dict[str, list[np.ndarray]] = defaultdict(list)
        flags = list(track.flags)

        # 1. Top-level stems (excludes percussion and bass)
        try:
            mixed = mtrack.mix_stems(custom_mapping)
            for category, audio in mixed.items():
                if audio is not None and np.any(audio):
                    audio = ensure_float32(audio)
                    audio = ensure_stereo(audio)
                    category_audio[category].append(audio)
        except Exception as e:
            logger.error("Error mixing top-level stems for %s: %s", track.track_name, e)

        # 2. Percussion sub-stem routing
        try:
            perc_sources = mtrack.stem_sources_mixture("percussion")
            if perc_sources is not None:
                for sub_name, audio in perc_sources.items():
                    if audio is None or not np.any(audio):
                        continue
                    target = PERCUSSION_ROUTING.get(sub_name)
                    if target is None:
                        logger.warning(
                            "Unknown percussion sub-stem %r in %s — routing to other",
                            sub_name, track.track_name,
                        )
                        target = "other"
                    audio = ensure_float32(audio)
                    audio = ensure_stereo(audio)
                    category_audio[target].append(audio)
        except Exception as e:
            logger.warning("No percussion sub-stems for %s: %s", track.track_name, e)

        # 3. Bass sub-stem routing
        try:
            bass_sources = mtrack.stem_sources_mixture("bass")
            if bass_sources is not None:
                for sub_name, audio in bass_sources.items():
                    if audio is None or not np.any(audio):
                        continue
                    target = BASS_ROUTING.get(sub_name)
                    if target is None:
                        logger.warning(
                            "Unknown bass sub-stem %r in %s — routing to other",
                            sub_name, track.track_name,
                        )
                        target = "other"
                    audio = ensure_float32(audio)
                    audio = ensure_stereo(audio)
                    category_audio[target].append(audio)
        except Exception as e:
            logger.warning("No bass sub-stems for %s: %s", track.track_name, e)

        # 4. Sum and write
        filename_base = sanitize_filename(
            self.name, track.split, track.index, track.artist, track.title
        )
        written_stems = []

        for category, audio_list in category_audio.items():
            if not audio_list:
                continue
            if len(audio_list) == 1:
                combined = audio_list[0]
            else:
                combined = sum_stems(audio_list)

            # Skip silent stems
            if not np.any(combined):
                continue

            if group_by_dataset:
                stem_dir = output_dir / category / self.name
            else:
                stem_dir = output_dir / category

            out_path = stem_dir / f"{filename_base}.wav"
            write_wav_atomic(out_path, combined, self.sample_rate)
            written_stems.append(category)

        # Record bleed info
        has_bleed = bool(getattr(mtrack, "bleedings", None))

        return {
            "source_dataset": self.name,
            "original_track_name": track.track_name,
            "artist": track.artist,
            "title": track.title,
            "split": track.split,
            "available_stems": written_stems,
            "profile": profile.name,
            "has_bleed": has_bleed,
            "flags": flags,
        }
