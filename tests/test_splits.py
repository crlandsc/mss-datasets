"""Tests for split management — deterministic assignment, locking, overlap inheritance."""

import json
from pathlib import Path

import pytest

from mss_aggregate.datasets.base import TrackInfo
from mss_aggregate.splits import (
    assign_splits,
    load_splits,
    write_splits,
)
from mss_aggregate.utils import canonical_name


def _make_track(dataset, index, artist="Art", title="Song", split="train", name_override=None):
    return TrackInfo(
        source_dataset=dataset,
        artist=artist,
        title=title,
        split=split,
        path=Path("/fake"),
        index=index,
        original_track_name=name_override or f"{artist} - {title}",
    )


class TestMusdb18hqSplits:
    def test_preserves_directory_splits(self):
        tracks = [
            _make_track("musdb18hq", 1, split="train"),
            _make_track("musdb18hq", 2, split="test"),
        ]
        assign_splits(tracks)
        assert tracks[0].split == "train"
        assert tracks[1].split == "test"


class TestMedleyDBSplits:
    def test_unique_tracks_all_train(self):
        tracks = [
            _make_track("medleydb", 1, artist="UniqueArtist", title="UniqueSong",
                        name_override="UniqueArtist_UniqueSong"),
        ]
        assign_splits(tracks)
        assert tracks[0].split == "train"

    def test_overlap_tracks_inherit_musdb_split(self):
        # "A Classic Education - NightOwl" is in overlap set
        track = _make_track(
            "medleydb", 1,
            artist="AClassicEducation", title="NightOwl",
            name_override="AClassicEducation_NightOwl",
        )
        cn = canonical_name("A Classic Education - NightOwl")
        musdb_splits = {cn: "test"}

        assign_splits([track], musdb_splits=musdb_splits)
        assert track.split == "test"

    def test_overlap_without_musdb_splits_defaults_train(self):
        track = _make_track(
            "medleydb", 1,
            artist="AClassicEducation", title="NightOwl",
            name_override="AClassicEducation_NightOwl",
        )
        assign_splits([track], musdb_splits=None)
        assert track.split == "train"


class TestMoisesDBSplits:
    def test_val_set_size(self):
        tracks = [_make_track("moisesdb", i, title=f"Song{i}") for i in range(1, 241)]
        assign_splits(tracks)
        val = [t for t in tracks if t.split == "val"]
        train = [t for t in tracks if t.split == "train"]
        assert len(val) == 50
        assert len(train) == 190

    def test_deterministic(self):
        """Same tracks → same val set every time."""
        tracks1 = [_make_track("moisesdb", i, title=f"Song{i}") for i in range(1, 241)]
        tracks2 = [_make_track("moisesdb", i, title=f"Song{i}") for i in range(1, 241)]
        assign_splits(tracks1)
        assign_splits(tracks2)
        val1 = {t.index for t in tracks1 if t.split == "val"}
        val2 = {t.index for t in tracks2 if t.split == "val"}
        assert val1 == val2

    def test_fewer_than_50_tracks(self):
        tracks = [_make_track("moisesdb", i, title=f"Song{i}") for i in range(1, 11)]
        assign_splits(tracks)
        val = [t for t in tracks if t.split == "val"]
        assert len(val) == 10  # All go to val when fewer than 50


class TestSplitLocking:
    def test_existing_splits_respected(self):
        tracks = [_make_track("musdb18hq", 1, split="train")]
        key = f"musdb18hq_0001_{tracks[0].original_track_name}"
        existing = {key: "test"}  # Override to test

        assign_splits(tracks, existing_splits=existing)
        assert tracks[0].split == "test"

    def test_write_and_load_roundtrip(self, tmp_path):
        tracks = [
            _make_track("musdb18hq", 1, split="train"),
            _make_track("medleydb", 1, split="train"),
        ]
        path = tmp_path / "splits.json"
        write_splits(path, tracks)

        loaded = load_splits(path)
        assert loaded is not None
        assert len(loaded) == 2
        assert all(v == "train" for v in loaded.values())

    def test_load_nonexistent(self, tmp_path):
        result = load_splits(tmp_path / "nope.json")
        assert result is None

    def test_splits_json_format(self, tmp_path):
        tracks = [_make_track("musdb18hq", 1, split="test")]
        path = tmp_path / "splits.json"
        write_splits(path, tracks)

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        key = list(data.keys())[0]
        assert "musdb18hq" in key
        assert data[key] == "test"


class TestMixedDatasets:
    def test_all_three_datasets(self):
        tracks = []
        # MUSDB18-HQ: 3 train, 1 test
        for i in range(1, 5):
            split = "test" if i == 4 else "train"
            tracks.append(_make_track("musdb18hq", i, split=split, title=f"MusdbSong{i}"))

        # MedleyDB: 3 unique
        for i in range(1, 4):
            tracks.append(_make_track("medleydb", i, title=f"MedSong{i}",
                                      name_override=f"MedArtist_MedSong{i}"))

        # MoisesDB: 10 tracks
        for i in range(1, 11):
            tracks.append(_make_track("moisesdb", i, title=f"MoiSong{i}"))

        assign_splits(tracks)

        musdb_splits = {t.split for t in tracks if t.source_dataset == "musdb18hq"}
        med_splits = {t.split for t in tracks if t.source_dataset == "medleydb"}
        moi_val = [t for t in tracks if t.source_dataset == "moisesdb" and t.split == "val"]

        assert "train" in musdb_splits
        assert "test" in musdb_splits
        assert med_splits == {"train"}
        assert len(moi_val) == 10  # All 10 go to val (< 50)
