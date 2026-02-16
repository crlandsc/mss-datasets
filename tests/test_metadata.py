"""Tests for metadata file generation."""

import json

import pytest
import yaml

from mss_aggregate.metadata import (
    ErrorEntry,
    ManifestEntry,
    write_config,
    write_errors,
    write_manifest,
    write_overlap_registry,
)


class TestManifest:
    def test_write_and_read(self, tmp_path):
        entries = [
            ManifestEntry(
                source_dataset="musdb18hq",
                original_track_name="TestArtist - TestSong",
                artist="TestArtist",
                title="TestSong",
                split="train",
                available_stems=["vocals", "drums", "bass", "other"],
                profile="vdbo",
                license="academic-use-only",
                duration_seconds=120.5,
                musdb18hq_4stem_only=True,
            ),
        ]
        path = tmp_path / "manifest.json"
        write_manifest(path, entries)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        key = list(data.keys())[0]
        assert data[key]["source_dataset"] == "musdb18hq"
        assert data[key]["available_stems"] == ["vocals", "drums", "bass", "other"]
        assert data[key]["musdb18hq_4stem_only"] is True

    def test_multiple_entries(self, tmp_path):
        entries = [
            ManifestEntry(
                source_dataset="musdb18hq", original_track_name="Track1",
                artist="A", title="T1", split="train",
                available_stems=["vocals"], profile="vdbo",
            ),
            ManifestEntry(
                source_dataset="medleydb", original_track_name="Track2",
                artist="B", title="T2", split="train",
                available_stems=["vocals", "other"], profile="vdbo",
                is_composite_sum=True,
            ),
        ]
        path = tmp_path / "manifest.json"
        write_manifest(path, entries)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_flags_serialized(self, tmp_path):
        entry = ManifestEntry(
            source_dataset="medleydb", original_track_name="T",
            artist="A", title="T", split="train",
            available_stems=["other"], profile="vdbo",
            flags=["unlabeled", "composite_sum"],
        )
        path = tmp_path / "manifest.json"
        write_manifest(path, [entry])

        with open(path) as f:
            data = json.load(f)
        val = list(data.values())[0]
        assert "unlabeled" in val["flags"]


class TestErrors:
    def test_write_empty(self, tmp_path):
        path = tmp_path / "errors.json"
        write_errors(path, [])
        with open(path) as f:
            data = json.load(f)
        assert data == []

    def test_write_entries(self, tmp_path):
        errors = [
            ErrorEntry(
                track="medleydb_LizNelson_Rainfall",
                dataset="medleydb",
                error="Corrupted WAV: unexpected EOF",
                stage="stem_map",
            ),
        ]
        path = tmp_path / "errors.json"
        write_errors(path, errors)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["track"] == "medleydb_LizNelson_Rainfall"
        assert data[0]["skipped"] is True


class TestOverlapRegistry:
    def test_write_with_tracks(self, tmp_path):
        path = tmp_path / "overlap_registry.json"
        write_overlap_registry(path, [
            "A Classic Education - NightOwl",
            "Music Delta - Beatles",
        ])

        with open(path) as f:
            data = json.load(f)
        assert data["skipped_count"] == 2
        assert "A Classic Education - NightOwl" in data["skipped_tracks"]

    def test_write_empty(self, tmp_path):
        path = tmp_path / "overlap_registry.json"
        write_overlap_registry(path, [])

        with open(path) as f:
            data = json.load(f)
        assert data["skipped_count"] == 0
        assert data["skipped_tracks"] == []


class TestConfig:
    def test_write_config(self, tmp_path):
        config = {
            "profile": "vdbo",
            "workers": 4,
            "output": "/path/to/output",
            "group_by_dataset": False,
        }
        path = tmp_path / "config.yaml"
        write_config(path, config)

        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["profile"] == "vdbo"
        assert data["workers"] == 4

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "metadata" / "config.yaml"
        write_config(path, {"test": True})
        assert path.exists()
