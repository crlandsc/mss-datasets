"""Tests for MedleyDB adapter — YAML parsing, instrument mapping, stem summing."""

import numpy as np
import pytest
import soundfile as sf
import yaml

from mss_datasets.datasets.medleydb import MedleydbAdapter
from mss_datasets.mapping import VDBO, VDBO_GP


def _make_medleydb_track(base_dir, track_name, stems_dict, sr=44100, n_samples=44100):
    """Create a synthetic MedleyDB track directory.

    stems_dict: {"S01": "female singer", "S02": "acoustic guitar", ...}
    """
    rng = np.random.default_rng(42)
    track_dir = base_dir / "Audio" / track_name
    stems_dir = track_dir / f"{track_name}_STEMS"
    stems_dir.mkdir(parents=True)

    # Build metadata
    metadata = {
        "artist": track_name.split("_")[0],
        "title": "_".join(track_name.split("_")[1:]),
        "has_bleed": "no",
        "stems": {},
    }

    for stem_key, instrument in stems_dict.items():
        idx = stem_key.replace("S", "")
        metadata["stems"][stem_key] = {"instrument": instrument}
        data = rng.uniform(-0.3, 0.3, (n_samples, 2)).astype(np.float32)
        sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

    with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
        yaml.dump(metadata, f)

    return track_dir


@pytest.fixture
def medleydb_fixture(tmp_path):
    """Create synthetic MedleyDB with 2 tracks."""
    _make_medleydb_track(tmp_path, "LizNelson_Rainfall", {
        "S01": "female singer",
        "S02": "acoustic guitar",
        "S03": "violin",
    })
    _make_medleydb_track(tmp_path, "TestBand_RockSong", {
        "S01": "male singer",
        "S02": "distorted electric guitar",
        "S03": "electric bass",
        "S04": "drum set",
    })
    return tmp_path


@pytest.fixture
def medleydb_main_system(tmp_path):
    """Track with Main System stem."""
    _make_medleydb_track(tmp_path, "TestArtist_MainSysTrack", {
        "S01": "Main System",
        "S02": "female singer",
        "S03": "piano",
    })
    return tmp_path


@pytest.fixture
def medleydb_only_main_system(tmp_path):
    """Track with ONLY Main System stem — should produce no output."""
    _make_medleydb_track(tmp_path, "TestArtist_OnlyMainSys", {
        "S01": "Main System",
    })
    return tmp_path


class TestValidation:
    def test_valid_path(self, medleydb_fixture):
        adapter = MedleydbAdapter(medleydb_fixture)
        adapter.validate_path()

    def test_missing_audio_dir(self, tmp_path):
        adapter = MedleydbAdapter(tmp_path)
        with pytest.raises(ValueError, match="Audio"):
            adapter.validate_path()


class TestDiscoverTracks:
    def test_discovers_all(self, medleydb_fixture):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        assert len(tracks) == 2

    def test_artist_title_from_metadata(self, medleydb_fixture):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        liz = next(t for t in tracks if "Rainfall" in t.title)
        assert liz.artist == "LizNelson"

    def test_indices_1_based(self, medleydb_fixture):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        assert tracks[0].index == 1
        assert tracks[1].index == 2

    def test_default_split_train(self, medleydb_fixture):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        assert all(t.split == "train" for t in tracks)


class TestProcessTrackVDBO:
    def test_writes_correct_stems(self, medleydb_fixture, tmp_path):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        # LizNelson_Rainfall: singer→vocals, guitar→other, violin→other
        liz = next(t for t in tracks if "Rainfall" in t.title)
        result = adapter.process_track(liz, VDBO, output)
        assert "vocals" in result["available_stems"]
        assert "other" in result["available_stems"]
        assert len(result["available_stems"]) == 2  # vocals + other (guitar+violin summed)

    def test_rock_song_four_stems(self, medleydb_fixture, tmp_path):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        rock = next(t for t in tracks if "RockSong" in t.title)
        result = adapter.process_track(rock, VDBO, output)
        assert set(result["available_stems"]) == {"vocals", "drums", "bass", "other"}

    def test_output_files_exist(self, medleydb_fixture, tmp_path):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"
        adapter.process_track(tracks[0], VDBO, output)

        # Should have at least one WAV in vocals/
        wavs = list((output / "vocals").glob("*.wav"))
        assert len(wavs) >= 1


class TestProcessTrackVDBOGP:
    def test_guitar_separated(self, medleydb_fixture, tmp_path):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        # LizNelson: guitar→guitar, violin→other
        liz = next(t for t in tracks if "Rainfall" in t.title)
        result = adapter.process_track(liz, VDBO_GP, output)
        assert "guitar" in result["available_stems"]
        assert "vocals" in result["available_stems"]
        assert "other" in result["available_stems"]  # violin


class TestMainSystemHandling:
    def test_main_system_excluded_others_kept(self, medleydb_main_system, tmp_path):
        adapter = MedleydbAdapter(medleydb_main_system)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        result = adapter.process_track(tracks[0], VDBO, output)
        assert "vocals" in result["available_stems"]
        assert "other" in result["available_stems"]  # piano→other in VDBO
        assert len(result["available_stems"]) == 2

    def test_only_main_system_produces_nothing(self, medleydb_only_main_system, tmp_path):
        adapter = MedleydbAdapter(medleydb_only_main_system)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        result = adapter.process_track(tracks[0], VDBO, output)
        assert result["available_stems"] == []


class TestGroupByDataset:
    def test_creates_subdirectory(self, medleydb_fixture, tmp_path):
        adapter = MedleydbAdapter(medleydb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"
        adapter.process_track(tracks[0], VDBO, output, group_by_dataset=True)

        assert (output / "vocals" / "medleydb").is_dir()
