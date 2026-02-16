"""Integration tests — full pipeline end-to-end against fixture datasets."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mss_aggregate.pipeline import Pipeline, PipelineConfig


def _make_musdb_fixture(base_path, overlap_track=True):
    """Create MUSDB18-HQ with real overlap track names."""
    sr = 44100
    n = sr // 2  # 0.5 sec
    rng = np.random.default_rng(42)

    train_tracks = [
        "ArtistA - UniqueTrainSong",
        "ArtistB - AnotherTrain",
    ]
    if overlap_track:
        train_tracks.append("A Classic Education - NightOwl")  # overlap with MedleyDB

    test_tracks = ["ArtistC - UniqueTestSong"]

    for split, tracks in [("train", train_tracks), ("test", test_tracks)]:
        for name in tracks:
            d = base_path / split / name
            d.mkdir(parents=True)
            for stem in ("vocals", "drums", "bass", "other", "mixture"):
                data = rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32)
                sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")


def _make_medleydb_fixture(base_path, overlap_track=True):
    """Create MedleyDB with overlap track + unique tracks."""
    sr = 44100
    n = sr // 2
    rng = np.random.default_rng(43)

    tracks = [
        ("UniqueArtist_UniqueSong", {
            "S01": "female singer",
            "S02": "acoustic guitar",
            "S03": "electric bass",
        }),
        ("AnotherArtist_AnotherSong", {
            "S01": "male singer",
            "S02": "drum set",
            "S03": "piano",
            "S04": "violin",
        }),
    ]
    if overlap_track:
        tracks.append(("AClassicEducation_NightOwl", {
            "S01": "male singer",
            "S02": "distorted electric guitar",
            "S03": "electric bass",
            "S04": "drum set",
            "S05": "synthesizer",
        }))

    for track_name, stems in tracks:
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        parts = track_name.split("_", 1)
        metadata = {
            "artist": parts[0],
            "title": parts[1] if len(parts) > 1 else parts[0],
            "has_bleed": "no",
            "stems": {},
        }
        for stem_key, instrument in stems.items():
            idx = stem_key.replace("S", "")
            metadata["stems"][stem_key] = {"instrument": instrument}
            data = rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32)
            sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)


@pytest.fixture
def full_datasets(tmp_path):
    musdb = tmp_path / "musdb18hq"
    medleydb = tmp_path / "medleydb"
    _make_musdb_fixture(musdb)
    _make_medleydb_fixture(medleydb)
    return {"musdb": musdb, "medleydb": medleydb, "root": tmp_path}


class TestFullPipelineVDBO:
    """End-to-end: MUSDB18-HQ + MedleyDB, VDBO profile."""

    def test_complete_run(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
            profile="vdbo",
        )
        result = Pipeline(config).run()

        assert result["errors"] == 0
        # 2 unique MUSDB + 1 test MUSDB + 3 MedleyDB (incl overlap) = 5
        # The overlap track (A Classic Education - NightOwl) is skipped from MUSDB,
        # sourced from MedleyDB instead
        assert result["skipped_musdb_overlap"] == 1
        assert result["total_tracks"] == 6  # 3 MUSDB (4 minus 1 overlap) + 3 MedleyDB

    def test_stem_directories_exist(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
        )
        Pipeline(config).run()

        for stem in ("vocals", "drums", "bass", "other"):
            assert (output / stem).is_dir(), f"Missing stem dir: {stem}"

    def test_all_wavs_valid(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
        )
        Pipeline(config).run()

        for wav in output.rglob("*.wav"):
            info = sf.info(str(wav))
            assert info.samplerate == 44100, f"Bad sr in {wav}"
            assert info.channels == 2, f"Bad channels in {wav}"
            assert info.subtype == "FLOAT", f"Bad subtype in {wav}"

    def test_metadata_complete(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
        )
        Pipeline(config).run()

        meta = output / "metadata"
        assert (meta / "manifest.json").exists()
        assert (meta / "splits.json").exists()
        assert (meta / "overlap_registry.json").exists()
        assert (meta / "errors.json").exists()
        assert (meta / "config.yaml").exists()

        # Verify manifest has entries
        with open(meta / "manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest) >= 4

        # Verify splits
        with open(meta / "splits.json") as f:
            splits = json.load(f)
        assert len(splits) >= 4

        # Verify overlap registry
        with open(meta / "overlap_registry.json") as f:
            overlap = json.load(f)
        assert overlap["skipped_count"] == 1
        assert "A Classic Education - NightOwl" in overlap["skipped_tracks"]

        # Verify errors empty
        with open(meta / "errors.json") as f:
            errors = json.load(f)
        assert errors == []

        # Verify config
        with open(meta / "config.yaml") as f:
            config_data = yaml.safe_load(f)
        assert config_data["profile"] == "vdbo"

    def test_overlap_track_from_medleydb(self, full_datasets):
        """The overlap track should be sourced from MedleyDB, not MUSDB18-HQ."""
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
        )
        Pipeline(config).run()

        with open(output / "metadata" / "manifest.json") as f:
            manifest = json.load(f)

        # Find the overlap track in manifest
        overlap_entries = [
            v for v in manifest.values()
            if "classiceducation" in v["original_track_name"].lower().replace(" ", "")
            or "nightowl" in v["original_track_name"].lower()
        ]
        assert len(overlap_entries) == 1
        assert overlap_entries[0]["source_dataset"] == "medleydb"


class TestFullPipelineVDBOGP:
    """End-to-end: VDBO+GP profile."""

    def test_guitar_piano_stems(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
            profile="vdbo+gp",
        )
        result = Pipeline(config).run()

        assert result["errors"] == 0
        # MedleyDB tracks should contribute guitar/piano stems
        guitar_count = result["stem_counts"].get("guitar", 0)
        piano_count = result["stem_counts"].get("piano", 0)
        assert guitar_count >= 1  # overlap track has guitar
        assert piano_count >= 1  # AnotherArtist_AnotherSong has piano


class TestResumability:
    """Verify pipeline can resume after interruption."""

    def test_resume_produces_same_result(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
        )

        # First run
        result1 = Pipeline(config).run()

        # Second run (should skip all existing)
        result2 = Pipeline(config).run()

        assert result1["total_files"] == result2["total_files"]
        assert result1["stem_counts"] == result2["stem_counts"]


class TestMusdbOnlyIntegration:
    """MUSDB18-HQ alone — no overlap dedup needed."""

    def test_all_150_style(self, tmp_path):
        musdb = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb, overlap_track=False)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb), output=str(output))
        result = Pipeline(config).run()

        assert result["errors"] == 0
        assert result["skipped_musdb_overlap"] == 0
        assert result["total_tracks"] == 3  # 2 train + 1 test


class TestMedleydbOnlyIntegration:
    """MedleyDB alone — all tracks train, no overlap."""

    def test_all_train(self, tmp_path):
        medleydb = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb, overlap_track=False)
        output = tmp_path / "output"

        config = PipelineConfig(medleydb_path=str(medleydb), output=str(output))
        result = Pipeline(config).run()

        assert result["errors"] == 0
        with open(output / "metadata" / "splits.json") as f:
            splits = json.load(f)
        assert all(v == "train" for v in splits.values())


class TestGroupByDatasetIntegration:
    def test_dataset_subdirs(self, full_datasets):
        output = full_datasets["root"] / "output"
        config = PipelineConfig(
            musdb18hq_path=str(full_datasets["musdb"]),
            medleydb_path=str(full_datasets["medleydb"]),
            output=str(output),
            group_by_dataset=True,
        )
        Pipeline(config).run()

        assert (output / "vocals" / "musdb18hq").is_dir()
        assert (output / "vocals" / "medleydb").is_dir()
        # MUSDB files in musdb18hq subdir
        musdb_wavs = list((output / "vocals" / "musdb18hq").glob("*.wav"))
        medley_wavs = list((output / "vocals" / "medleydb").glob("*.wav"))
        assert len(musdb_wavs) >= 1
        assert len(medley_wavs) >= 1
