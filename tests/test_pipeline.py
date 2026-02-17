"""Tests for pipeline orchestration — end-to-end with synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mss_datasets.pipeline import Pipeline, PipelineConfig


def _make_musdb_fixture(base_path):
    """Create synthetic MUSDB18-HQ."""
    sr = 44100
    rng = np.random.default_rng(42)
    for split, tracks in [("train", ["ArtistA - Song1", "ArtistB - Song2"]),
                          ("test", ["ArtistC - Song3"])]:
        for track_name in tracks:
            d = base_path / split / track_name
            d.mkdir(parents=True)
            for stem in ("vocals", "drums", "bass", "other", "mixture"):
                data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
                sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")


def _make_medleydb_fixture(base_path):
    """Create synthetic MedleyDB."""
    sr = 44100
    rng = np.random.default_rng(43)
    tracks = [
        ("MedArtist_TrackOne", {"S01": "female singer", "S02": "acoustic guitar"}),
        ("MedArtist_TrackTwo", {"S01": "male singer", "S02": "drum set", "S03": "electric bass"}),
    ]
    for track_name, stems in tracks:
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        metadata = {
            "artist": track_name.split("_")[0],
            "title": "_".join(track_name.split("_")[1:]),
            "has_bleed": "no",
            "stems": {},
        }
        for stem_key, instrument in stems.items():
            idx = stem_key.replace("S", "")
            metadata["stems"][stem_key] = {"instrument": instrument}
            data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
            sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)


@pytest.fixture
def full_fixture(tmp_path):
    """Create MUSDB18-HQ + MedleyDB fixtures."""
    musdb_path = tmp_path / "musdb18hq"
    medleydb_path = tmp_path / "medleydb"
    _make_musdb_fixture(musdb_path)
    _make_medleydb_fixture(medleydb_path)
    return {"musdb": musdb_path, "medleydb": medleydb_path, "output": tmp_path / "output"}


class TestPipelineMusdbOnly:
    def test_runs_successfully(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        pipeline = Pipeline(config)
        result = pipeline.run()

        assert result["total_tracks"] == 3
        assert result["errors"] == 0
        assert result["stem_counts"]["vocals"] == 3
        assert result["stem_counts"]["drums"] == 3

    def test_metadata_files_written(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        meta = output / "metadata"
        assert (meta / "manifest.json").exists()
        assert (meta / "splits.json").exists()
        assert (meta / "overlap_registry.json").exists()
        assert (meta / "errors.json").exists()
        assert (meta / "config.yaml").exists()

    def test_overlap_registry_empty_without_medleydb(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        with open(output / "metadata" / "overlap_registry.json") as f:
            data = json.load(f)
        assert data["skipped_count"] == 0


class TestPipelineMedleydbOnly:
    def test_runs_successfully(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        output = tmp_path / "output"

        config = PipelineConfig(medleydb_path=str(medleydb_path), output=str(output))
        pipeline = Pipeline(config)
        result = pipeline.run()

        assert result["total_tracks"] == 2
        assert result["errors"] == 0
        assert result["stem_counts"]["vocals"] == 2


class TestPipelineBothDatasets:
    def test_processes_both(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
        )
        pipeline = Pipeline(config)
        result = pipeline.run()

        # 3 MUSDB + 2 MedleyDB (no overlap in our fixtures)
        assert result["total_tracks"] == 5
        assert result["errors"] == 0


class TestDryRun:
    def test_no_files_written(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["dry_run"] is True
        assert result["total_tracks"] == 3
        assert not output.exists()  # No output written

    def test_dry_run_shows_stem_folders(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()
        assert "vocals" in result["stem_folders"]


class TestResumability:
    def test_skips_already_processed(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))

        # First run
        Pipeline(config).run()
        wav_count_1 = len(list(output.rglob("*.wav")))

        # Second run (should skip all existing)
        pipeline2 = Pipeline(config)
        result2 = pipeline2.run()

        wav_count_2 = len(list(output.rglob("*.wav")))
        assert wav_count_2 == wav_count_1  # No new files

    def test_cleans_up_tmp_files(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        # Create a leftover .tmp file
        (output / "vocals").mkdir(parents=True)
        (output / "vocals" / "leftover.wav.tmp").touch()

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        assert not (output / "vocals" / "leftover.wav.tmp").exists()


class TestGroupByDataset:
    def test_creates_dataset_subdirs(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
            group_by_dataset=True,
        )
        Pipeline(config).run()

        vocals_dir = full_fixture["output"] / "vocals"
        assert (vocals_dir / "musdb18hq").is_dir()
        assert (vocals_dir / "medleydb").is_dir()


class TestVDBOGP:
    def test_six_stem_profile(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(output),
            profile="vdbo+gp",
        )
        result = Pipeline(config).run()

        # TrackOne has acoustic guitar → guitar stem
        assert result["stem_counts"].get("guitar", 0) >= 1


class TestBleedFiltering:
    def _make_medleydb_with_bleed(self, base_path, has_bleed="yes"):
        """Create MedleyDB fixture with bleed tracks."""
        sr = 44100
        rng = np.random.default_rng(44)
        track_name = "BleedArtist_BleedTrack"
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        metadata = {
            "artist": "BleedArtist",
            "title": "BleedTrack",
            "has_bleed": has_bleed,
            "stems": {"S01": {"instrument": "male singer"}},
        }
        data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
        sf.write(str(stems_dir / f"{track_name}_STEM_01.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)

    def test_bleed_excluded_by_default(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)  # 2 no-bleed tracks
        self._make_medleydb_with_bleed(medleydb_path, has_bleed="yes")  # 1 bleed track

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["total_tracks"] == 2
        assert result["excluded_bleed"] == 1

    def test_include_bleed_overrides(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        self._make_medleydb_with_bleed(medleydb_path, has_bleed="yes")

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
            include_bleed=True,
        )
        result = Pipeline(config).run()

        assert result["total_tracks"] == 3
        assert result["excluded_bleed"] == 0

    def test_no_bleed_tracks_zero_excluded(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)  # no bleed tracks

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["excluded_bleed"] == 0


class TestInvalidPath:
    def test_bad_musdb_path_logged(self, tmp_path):
        config = PipelineConfig(
            musdb18hq_path=str(tmp_path / "nonexistent"),
            output=str(tmp_path / "output"),
        )
        pipeline = Pipeline(config)
        result = pipeline.run()
        assert result.get("error") == "No valid datasets found"
