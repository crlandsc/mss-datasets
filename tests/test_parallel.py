"""Tests for parallel processing — verify same output with workers=1 vs workers=2."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mss_aggregate.pipeline import Pipeline, PipelineConfig


def _make_musdb_fixture(base_path, n_train=4, n_test=2):
    """Create synthetic MUSDB18-HQ with multiple tracks."""
    sr = 44100
    rng = np.random.default_rng(42)
    for i in range(1, n_train + 1):
        d = base_path / "train" / f"Artist{i} - Song{i}"
        d.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other", "mixture"):
            data = rng.uniform(-0.3, 0.3, (sr // 4, 2)).astype(np.float32)  # short
            sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")
    for i in range(1, n_test + 1):
        d = base_path / "test" / f"TestArtist{i} - TestSong{i}"
        d.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other", "mixture"):
            data = rng.uniform(-0.3, 0.3, (sr // 4, 2)).astype(np.float32)
            sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")


def _make_medleydb_fixture(base_path, n_tracks=3):
    """Create synthetic MedleyDB with multiple tracks."""
    sr = 44100
    rng = np.random.default_rng(43)
    for i in range(1, n_tracks + 1):
        track_name = f"MedArtist{i}_Track{i}"
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        instruments = {"S01": "female singer", "S02": "acoustic guitar", "S03": "drum set"}
        metadata = {
            "artist": f"MedArtist{i}",
            "title": f"Track{i}",
            "has_bleed": "no",
            "stems": {},
        }
        for stem_key, inst in instruments.items():
            idx = stem_key.replace("S", "")
            metadata["stems"][stem_key] = {"instrument": inst}
            data = rng.uniform(-0.3, 0.3, (sr // 4, 2)).astype(np.float32)
            sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)


def _collect_output_files(output_dir: Path) -> dict[str, set[str]]:
    """Collect all WAV filenames per stem directory."""
    result = {}
    for stem_dir in sorted(output_dir.iterdir()):
        if stem_dir.is_dir() and stem_dir.name != "metadata":
            result[stem_dir.name] = {f.name for f in stem_dir.rglob("*.wav")}
    return result


class TestParallelConsistency:
    def test_same_output_workers_1_vs_2(self, tmp_path):
        """Core test: workers=1 and workers=2 produce identical file sets."""
        musdb_path = tmp_path / "musdb18hq"
        medleydb_path = tmp_path / "medleydb"
        _make_musdb_fixture(musdb_path)
        _make_medleydb_fixture(medleydb_path)

        # Run with workers=1
        output1 = tmp_path / "output1"
        config1 = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            medleydb_path=str(medleydb_path),
            output=str(output1),
            workers=1,
        )
        result1 = Pipeline(config1).run()

        # Run with workers=2
        output2 = tmp_path / "output2"
        config2 = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            medleydb_path=str(medleydb_path),
            output=str(output2),
            workers=2,
        )
        result2 = Pipeline(config2).run()

        # Compare file sets
        files1 = _collect_output_files(output1)
        files2 = _collect_output_files(output2)

        assert files1.keys() == files2.keys(), "Different stem directories"
        for stem in files1:
            assert files1[stem] == files2[stem], f"Different files in {stem}/"

        # Same total counts
        assert result1["total_files"] == result2["total_files"]
        assert result1["errors"] == result2["errors"]

    def test_parallel_musdb_only(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path, n_train=6, n_test=2)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            workers=2,
        )
        result = Pipeline(config).run()

        assert result["total_tracks"] == 8
        assert result["errors"] == 0
        assert result["stem_counts"]["vocals"] == 8

    def test_parallel_no_errors(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        medleydb_path = tmp_path / "medleydb"
        _make_musdb_fixture(musdb_path)
        _make_medleydb_fixture(medleydb_path)

        output = tmp_path / "output"
        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            medleydb_path=str(medleydb_path),
            output=str(output),
            workers=4,
        )
        result = Pipeline(config).run()
        assert result["errors"] == 0
