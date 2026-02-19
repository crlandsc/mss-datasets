"""MUSDB18-HQ dataset adapter — reads WAVs directly, no musdb package."""

from __future__ import annotations

import logging
from pathlib import Path

from mss_datasets.audio import ensure_float32, ensure_stereo, read_wav, write_wav_atomic
from mss_datasets.datasets.base import DatasetAdapter, TrackInfo
from mss_datasets.mapping.profiles import StemProfile
from mss_datasets.utils import resolve_collision, sanitize_filename

logger = logging.getLogger(__name__)

MUSDB_STEMS = ("vocals", "drums", "bass", "other")


class Musdb18hqAdapter(DatasetAdapter):
    name = "musdb18hq"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def validate_path(self) -> None:
        train_dir = self.path / "train"
        test_dir = self.path / "test"
        if not train_dir.is_dir():
            raise ValueError(f"MUSDB18-HQ missing train/ directory: {train_dir}")
        if not test_dir.is_dir():
            raise ValueError(f"MUSDB18-HQ missing test/ directory: {test_dir}")

    def discover_tracks(self) -> list[TrackInfo]:
        tracks = []
        for split_name in ("train", "test"):
            split_dir = self.path / split_name
            if not split_dir.is_dir():
                continue
            subdirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
            for subdir in subdirs:
                # Parse "Artist - Title" folder name
                parts = subdir.name.split(" - ", 1)
                if len(parts) == 2:
                    artist, title = parts
                else:
                    artist, title = subdir.name, subdir.name

                tracks.append(TrackInfo(
                    source_dataset=self.name,
                    artist=artist,
                    title=title,
                    split=split_name,
                    path=subdir,
                    stems_available=list(MUSDB_STEMS),
                    has_bleed=False,
                    original_track_name=subdir.name,
                ))

        # Assign 1-based indices in discovery order
        for i, t in enumerate(tracks, 1):
            t.index = i

        return tracks

    def process_track(
        self,
        track: TrackInfo,
        profile: StemProfile,
        output_dir: Path,
        group_by_dataset: bool = False,
        include_mixtures: bool = False,
    ) -> dict:
        filename_base = sanitize_filename(
            self.name, track.split, track.index, track.artist, track.title
        )
        written_stems = []

        for stem_name in MUSDB_STEMS:
            wav_path = track.path / f"{stem_name}.wav"
            if not wav_path.exists():
                logger.warning("Missing stem %s for %s", stem_name, track.track_name)
                continue

            data, sr = read_wav(wav_path)
            data = ensure_float32(data)
            data = ensure_stereo(data)

            if group_by_dataset:
                stem_dir = output_dir / stem_name / self.name
            else:
                stem_dir = output_dir / stem_name

            out_path = stem_dir / f"{filename_base}.wav"
            write_wav_atomic(out_path, data, sr)
            written_stems.append(stem_name)

        # Write mixture (copy source mixture.wav directly)
        if include_mixtures:
            mixture_wav = track.path / "mixture.wav"
            if mixture_wav.exists():
                mix_data, mix_sr = read_wav(mixture_wav)
                mix_data = ensure_float32(mix_data)
                mix_data = ensure_stereo(mix_data)
                if group_by_dataset:
                    mixture_dir = output_dir / "mixture" / self.name
                else:
                    mixture_dir = output_dir / "mixture"
                write_wav_atomic(mixture_dir / f"{filename_base}.wav", mix_data, mix_sr)

        return {
            "source_dataset": self.name,
            "original_track_name": track.track_name,
            "artist": track.artist,
            "title": track.title,
            "split": track.split,
            "available_stems": written_stems,
            "profile": profile.name,
            "has_bleed": False,
            "musdb18hq_4stem_only": True,
            "flags": [],
        }
