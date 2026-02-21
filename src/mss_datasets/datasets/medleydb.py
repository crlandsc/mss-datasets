"""MedleyDB dataset adapter — reads WAVs + YAML metadata directly, no medleydb package."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from mss_datasets.audio import ensure_float32, ensure_stereo, read_wav, sum_stems, write_wav_atomic
from mss_datasets.datasets.base import DatasetAdapter, TrackInfo
from mss_datasets.mapping.profiles import StemProfile, load_medleydb_mapping, load_medleydb_overrides, resolve_medleydb_label
from mss_datasets.utils import resolve_collision, sanitize_filename

logger = logging.getLogger(__name__)


class MedleydbAdapter(DatasetAdapter):
    name = "medleydb"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.overrides = load_medleydb_overrides()

    def validate_path(self) -> None:
        audio_dir = self.path / "Audio"
        if not audio_dir.is_dir():
            raise ValueError(f"MedleyDB missing Audio/ directory: {audio_dir}")

        # Auto-download metadata if missing (Zenodo archives don't include it)
        self._ensure_metadata(audio_dir)

    def _ensure_metadata(self, audio_dir: Path) -> None:
        """Download metadata YAML files from GitHub if not present."""
        for d in audio_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                yamls = [f for f in d.glob("*_METADATA.yaml") if not f.name.startswith("._")]
                if yamls:
                    return  # Already have metadata
                break

        logger.info("MedleyDB metadata YAML files missing — downloading from GitHub...")
        try:
            from mss_datasets.download import _download_medleydb_metadata

            _download_medleydb_metadata(audio_dir)
        except Exception as e:
            raise ValueError(
                f"MedleyDB metadata not found and auto-download failed: {e}\n"
                "Run 'mss-datasets --download' first, or install metadata manually from:\n"
                "  https://github.com/marl/medleydb"
            ) from e

    def discover_tracks(self) -> list[TrackInfo]:
        audio_dir = self.path / "Audio"
        tracks = []

        subdirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
        for subdir in subdirs:
            if subdir.name in self.overrides["exclude_tracks"]:
                logger.info("Excluding track %s (override: track excluded)", subdir.name)
                continue

            # Find metadata YAML (filter macOS resource forks)
            yaml_files = [f for f in subdir.glob("*_METADATA.yaml") if not f.name.startswith("._")]
            if not yaml_files:
                logger.warning("No METADATA.yaml in %s, skipping", subdir.name)
                continue

            yaml_path = yaml_files[0]
            try:
                with open(yaml_path) as f:
                    metadata = yaml.safe_load(f)
            except Exception as e:
                logger.error("Failed to parse %s: %s", yaml_path, e)
                continue

            # Parse artist/title from directory name (format: ArtistName_TrackName)
            # The metadata may also contain these fields
            artist = metadata.get("artist", "")
            title = metadata.get("title", "")
            if not artist or not title:
                parts = subdir.name.split("_", 1)
                if len(parts) == 2:
                    artist = artist or parts[0]
                    title = title or parts[1]
                else:
                    artist = artist or subdir.name
                    title = title or subdir.name

            has_bleed = metadata.get("has_bleed", "no") == "yes"

            # Determine available stems from metadata
            stems_info = metadata.get("stems", {})
            instrument_labels = []
            for stem_data in stems_info.values():
                inst = stem_data.get("instrument", "")
                if inst:
                    instrument_labels.append(inst)

            tracks.append(TrackInfo(
                source_dataset=self.name,
                artist=artist,
                title=title,
                split="train",  # Default; may be overridden by overlap inheritance
                path=subdir,
                stems_available=instrument_labels,
                has_bleed=has_bleed,
                original_track_name=subdir.name,
            ))

        # Assign 1-based indices
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
        mapping = load_medleydb_mapping(profile)

        # Read YAML metadata (filter macOS resource forks)
        yaml_files = [f for f in track.path.glob("*_METADATA.yaml") if not f.name.startswith("._")]
        if not yaml_files:
            raise FileNotFoundError(f"No METADATA.yaml in {track.path}")

        with open(yaml_files[0]) as f:
            metadata = yaml.safe_load(f)

        stems_info = metadata.get("stems", {})
        track_name = track.path.name

        # Collect audio per output category
        category_audio: dict[str, list] = defaultdict(list)
        flags = list(track.flags)

        excluded_stems = self.overrides["exclude_stems"].get(track.original_track_name, set())
        rerouted_stems = self.overrides["reroute_stems"].get(track.original_track_name, {})

        for stem_key, stem_data in stems_info.items():
            if stem_key in excluded_stems:
                logger.info("Excluding stem %s/%s (override: stem excluded)", track.original_track_name, stem_key)
                continue

            instrument = stem_data.get("instrument", "")
            if not instrument:
                continue

            target, label_flags = resolve_medleydb_label(instrument, mapping)
            flags.extend(label_flags)

            if target is None:
                # Main System — skip this stem
                continue

            # Apply per-stem reroute override
            if stem_key in rerouted_stems:
                original = target
                target = rerouted_stems[stem_key]
                logger.info("Rerouting stem %s/%s: %s → %s (override)", track.original_track_name, stem_key, original, target)

            # Locate stem WAV file
            stem_idx = stem_key.replace("S", "")  # "S01" -> "01"
            stems_dir = track.path / f"{track_name}_STEMS"
            stem_wav = stems_dir / f"{track_name}_STEM_{stem_idx}.wav"

            if not stem_wav.exists():
                logger.warning("Missing stem file %s", stem_wav)
                continue

            try:
                data, sr = read_wav(stem_wav)
                data = ensure_float32(data)
                data = ensure_stereo(data)
                if not np.any(data):
                    logger.warning(
                        "Skipping silent stem '%s' (%s) for track %s",
                        stem_key, instrument, track.track_name,
                    )
                    continue
                category_audio[target].append(data)
            except Exception as e:
                logger.error("Error reading %s: %s", stem_wav, e)
                continue

        if not category_audio:
            logger.warning("Track %s produced no output (all stems filtered)", track.track_name)
            return {
                "source_dataset": self.name,
                "original_track_name": track.track_name,
                "artist": track.artist,
                "title": track.title,
                "split": track.split,
                "available_stems": [],
                "profile": profile.name,
                "has_bleed": track.has_bleed,
                "flags": flags,
            }

        # Sum stems per category and write
        filename_base = sanitize_filename(
            self.name, track.split, track.index, track.artist, track.title
        )
        written_stems = []

        for category, audio_list in category_audio.items():
            if len(audio_list) == 1:
                combined = audio_list[0]
            else:
                combined = sum_stems(audio_list)
                if "composite_sum" not in flags:
                    flags.append("composite_sum")

            if not np.any(combined):
                logger.warning(
                    "Skipping silent combined stem '%s' for track %s",
                    category, track.track_name,
                )
                continue

            if group_by_dataset:
                stem_dir = output_dir / category / self.name
            else:
                stem_dir = output_dir / category

            out_path = stem_dir / f"{filename_base}.wav"
            write_wav_atomic(out_path, combined, 44100)
            written_stems.append(category)

        # Write mixture (sum of all stems)
        if include_mixtures and written_stems:
            all_audio = []
            for audio_list in category_audio.values():
                all_audio.extend(audio_list)
            if all_audio:
                mixture = sum_stems(all_audio)
                if group_by_dataset:
                    mixture_dir = output_dir / "mixture" / self.name
                else:
                    mixture_dir = output_dir / "mixture"
                write_wav_atomic(mixture_dir / f"{filename_base}.wav", mixture, 44100)

        return {
            "source_dataset": self.name,
            "original_track_name": track.track_name,
            "artist": track.artist,
            "title": track.title,
            "split": track.split,
            "available_stems": written_stems,
            "profile": profile.name,
            "has_bleed": track.has_bleed,
            "flags": list(set(flags)),
        }
