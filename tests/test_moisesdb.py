"""Tests for MoisesDB adapter — mock-based, no real dataset needed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from mss_datasets.datasets.moisesdb_adapter import MoisesdbAdapter
from mss_datasets.datasets.base import TrackInfo
from mss_datasets.mapping import VDBO, VDBO_GP


def _make_audio(n_samples=44100, channels=2, value=0.3):
    """Helper: create a test audio array."""
    return np.full((n_samples, channels), value, dtype=np.float32)


def _mock_moisesdb_track(
    artist="MockArtist",
    name="MockSong",
    genre="rock",
    mix_stems_result=None,
    perc_sources=None,
    bass_sources=None,
    bleedings=None,
):
    """Create a mock MoisesDB track object."""
    track = MagicMock()
    track.artist = artist
    track.name = name
    track.genre = genre
    track.id = "mock-uuid-1234"
    track.path = "/mock/path"
    track.stems = {"vocals": None, "drums": None, "bass": None, "other": None}
    track.bleedings = bleedings

    if mix_stems_result is None:
        mix_stems_result = {
            "vocals": _make_audio(value=0.2),
            "drums": _make_audio(value=0.1),
            "other": _make_audio(value=0.15),
        }
    track.mix_stems = MagicMock(return_value=mix_stems_result)

    if perc_sources is None:
        perc_sources = {
            "a-tonal percussion (claps, shakers, congas, cowbell etc)": _make_audio(value=0.05),
        }
    track.stem_sources_mixture = MagicMock(side_effect=lambda stem: {
        "percussion": perc_sources,
        "bass": bass_sources or {},
    }.get(stem, {}))

    return track


class TestValidation:
    def test_missing_dir(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        with pytest.raises(ValueError, match="moisesdb_v0.1"):
            adapter.validate_path()

    def test_valid_dir(self, tmp_path):
        (tmp_path / "moisesdb_v0.1").mkdir()
        adapter = MoisesdbAdapter(tmp_path)
        adapter.validate_path()


class TestProcessTrackVDBO:
    def test_writes_stems(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        mock_track = _mock_moisesdb_track()
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )

        assert "vocals" in result["available_stems"]
        assert "drums" in result["available_stems"]
        assert "other" in result["available_stems"]

    def test_percussion_routed_to_drums(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        perc = {
            "a-tonal percussion (claps, shakers, congas, cowbell etc)": _make_audio(value=0.1),
        }
        mock_track = _mock_moisesdb_track(perc_sources=perc)
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )
        assert "drums" in result["available_stems"]
        # Verify the drums WAV exists and has data
        drums_wavs = list((output / "drums").glob("*.wav"))
        assert len(drums_wavs) == 1

    def test_pitched_percussion_routed_to_other(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        perc = {
            "pitched percussion (mallets, glockenspiel, ...)": _make_audio(value=0.1),
        }
        mock_track = _mock_moisesdb_track(
            mix_stems_result={"vocals": _make_audio(value=0.2)},
            perc_sources=perc,
        )
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )
        assert "other" in result["available_stems"]

    def test_bass_routing_split(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        bass = {
            "bass guitar": _make_audio(value=0.2),
            "tuba (bass of brass)": _make_audio(value=0.1),
        }
        mock_track = _mock_moisesdb_track(
            mix_stems_result={"vocals": _make_audio(value=0.2)},
            bass_sources=bass,
            perc_sources={},
        )
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )
        assert "bass" in result["available_stems"]  # bass guitar
        assert "other" in result["available_stems"]  # tuba
        assert "vocals" in result["available_stems"]

    def test_silent_stems_skipped(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        mock_track = _mock_moisesdb_track(
            mix_stems_result={
                "vocals": _make_audio(value=0.2),
                "drums": np.zeros((44100, 2), dtype=np.float32),  # silent
            },
            perc_sources={},
        )
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )
        assert "vocals" in result["available_stems"]
        assert "drums" not in result["available_stems"]


class TestProcessTrackVDBOGP:
    def test_guitar_piano_separated(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        mock_track = _mock_moisesdb_track(
            mix_stems_result={
                "vocals": _make_audio(value=0.2),
                "drums": _make_audio(value=0.1),
                "guitar": _make_audio(value=0.15),
                "piano": _make_audio(value=0.12),
                "other": _make_audio(value=0.05),
            },
            perc_sources={},
            bass_sources={"bass guitar": _make_audio(value=0.2)},
        )
        output = tmp_path / "output"

        result = adapter.process_track(
            track_info, VDBO_GP, output, _moisesdb_track=mock_track
        )
        assert "guitar" in result["available_stems"]
        assert "piano" in result["available_stems"]
        assert "vocals" in result["available_stems"]
        assert "bass" in result["available_stems"]


class TestGroupByDataset:
    def test_creates_subdirectory(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        mock_track = _mock_moisesdb_track()
        output = tmp_path / "output"

        adapter.process_track(
            track_info, VDBO, output, group_by_dataset=True, _moisesdb_track=mock_track
        )
        assert (output / "vocals" / "moisesdb").is_dir()


class TestOutputFormat:
    def test_wav_is_float32_stereo(self, tmp_path):
        adapter = MoisesdbAdapter(tmp_path)
        track_info = TrackInfo(
            source_dataset="moisesdb",
            artist="MockArtist",
            title="MockSong",
            split="train",
            path=tmp_path,
            index=1,
            original_track_name="MockArtist - MockSong",
        )
        mock_track = _mock_moisesdb_track()
        output = tmp_path / "output"

        adapter.process_track(
            track_info, VDBO, output, _moisesdb_track=mock_track
        )
        wav = list((output / "vocals").glob("*.wav"))[0]
        info = sf.info(str(wav))
        assert info.samplerate == 44100
        assert info.channels == 2
        assert info.subtype == "FLOAT"
