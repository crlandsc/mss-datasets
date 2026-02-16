"""Tests for MUSDB18-HQ adapter — discovery, processing, overlap skip."""

import numpy as np
import pytest
import soundfile as sf

from mss_aggregate.datasets.musdb18hq import Musdb18hqAdapter
from mss_aggregate.mapping import VDBO, VDBO_GP


@pytest.fixture
def musdb_fixture(tmp_path):
    """Create synthetic MUSDB18-HQ directory structure."""
    sr = 44100
    n_samples = sr  # 1 second
    rng = np.random.default_rng(42)

    for split in ("train", "test"):
        track_dir = tmp_path / split / "TestArtist - TestSong"
        track_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other", "mixture"):
            data = rng.uniform(-0.5, 0.5, (n_samples, 2)).astype(np.float32)
            sf.write(str(track_dir / f"{stem}.wav"), data, sr, subtype="FLOAT")

    # Second train track
    track2 = tmp_path / "train" / "Another Artist - Another Song"
    track2.mkdir(parents=True)
    for stem in ("vocals", "drums", "bass", "other", "mixture"):
        data = rng.uniform(-0.5, 0.5, (n_samples, 2)).astype(np.float32)
        sf.write(str(track2 / f"{stem}.wav"), data, sr, subtype="FLOAT")

    return tmp_path


class TestValidation:
    def test_valid_path(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        adapter.validate_path()  # Should not raise

    def test_missing_train(self, tmp_path):
        (tmp_path / "test").mkdir()
        adapter = Musdb18hqAdapter(tmp_path)
        with pytest.raises(ValueError, match="train"):
            adapter.validate_path()

    def test_missing_test(self, tmp_path):
        (tmp_path / "train").mkdir()
        adapter = Musdb18hqAdapter(tmp_path)
        with pytest.raises(ValueError, match="test"):
            adapter.validate_path()


class TestDiscoverTracks:
    def test_discovers_all(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        assert len(tracks) == 3  # 2 train + 1 test

    def test_split_assignment(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        train = [t for t in tracks if t.split == "train"]
        test = [t for t in tracks if t.split == "test"]
        assert len(train) == 2
        assert len(test) == 1

    def test_artist_title_parsed(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        t = next(t for t in tracks if t.title == "TestSong" and t.split == "test")
        assert t.artist == "TestArtist"
        assert t.original_track_name == "TestArtist - TestSong"

    def test_indices_assigned(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        indices = [t.index for t in tracks]
        assert indices == [1, 2, 3]

    def test_stems_available(self, musdb_fixture):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        for t in tracks:
            assert set(t.stems_available) == {"vocals", "drums", "bass", "other"}


class TestProcessTrack:
    def test_writes_four_stems(self, musdb_fixture, tmp_path):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"

        result = adapter.process_track(tracks[0], VDBO, output)
        assert set(result["available_stems"]) == {"vocals", "drums", "bass", "other"}

        for stem in ("vocals", "drums", "bass", "other"):
            wav = output / stem / f"{result['source_dataset']}_train_0001_another_artist_another_song.wav"
            assert wav.exists()

    def test_output_is_float32_stereo(self, musdb_fixture, tmp_path):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"
        adapter.process_track(tracks[0], VDBO, output)

        wav_files = list((output / "vocals").glob("*.wav"))
        assert len(wav_files) == 1
        info = sf.info(str(wav_files[0]))
        assert info.samplerate == 44100
        assert info.channels == 2
        assert info.subtype == "FLOAT"

    def test_group_by_dataset(self, musdb_fixture, tmp_path):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"
        adapter.process_track(tracks[0], VDBO, output, group_by_dataset=True)

        # Should be in musdb18hq subdirectory
        assert (output / "vocals" / "musdb18hq").is_dir()
        wav_files = list((output / "vocals" / "musdb18hq").glob("*.wav"))
        assert len(wav_files) == 1

    def test_vdbo_gp_still_4stem(self, musdb_fixture, tmp_path):
        adapter = Musdb18hqAdapter(musdb_fixture)
        tracks = adapter.discover_tracks()
        output = tmp_path / "output"
        result = adapter.process_track(tracks[0], VDBO_GP, output)
        # MUSDB18-HQ only contributes 4 stems even in 6-stem mode
        assert "guitar" not in result["available_stems"]
        assert "piano" not in result["available_stems"]
        assert result["musdb18hq_4stem_only"] is True
