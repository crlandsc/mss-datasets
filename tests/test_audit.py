"""Tests for the output-tree correctness audit."""

import json

import pytest

from mss_datasets.audit import ERROR, render_audit, run_audit
from mss_datasets.output_tree import OutputTree


def _wav(root, *parts):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def _track(root, split_dir, source, split, index, name, stems=("vocals", "drums", "mixture")):
    for stem in stems:
        _wav(root, split_dir, stem, source, f"{source}_{split}_{index}_{name}.wav")


@pytest.fixture
def clean_tree(tmp_path):
    _track(tmp_path, "train", "musdb18hq", "train", "0001", "actions_devil_s_words")
    _track(tmp_path, "train", "medleydb", "train", "0002", "amar_lal_rest")
    _track(tmp_path, "val", "moisesdb", "val", "0003", "some_artist_some_song")
    return tmp_path


def _findings(root):
    return {f.check: f for f in run_audit(OutputTree(root).load())["findings"]}


class TestCleanTree:
    def test_passes(self, clean_tree):
        result = run_audit(OutputTree(clean_tree).load())
        assert result["passed"]
        assert result["error_count"] == 0
        assert result["findings"] == []

    def test_render_says_pass(self, clean_tree):
        result = run_audit(OutputTree(clean_tree).load())
        assert "PASS" in render_audit(result)

    def test_counts(self, clean_tree):
        result = run_audit(OutputTree(clean_tree).load())
        assert result["total_tracks"] == 3
        assert result["total_files"] == 9


class TestLeakageDetection:
    def test_same_track_in_both_splits_is_an_error(self, clean_tree):
        # The exact failure a changed split would produce: same track, both dirs.
        _track(clean_tree, "val", "medleydb", "val", "0002", "amar_lal_rest")
        found = _findings(clean_tree)
        assert "split_leakage" in found
        assert found["split_leakage"].severity == ERROR
        assert not run_audit(OutputTree(clean_tree).load())["passed"]

    def test_same_track_under_two_indices_is_an_error(self, clean_tree):
        # What a shifted positional index leaves behind.
        _track(clean_tree, "train", "medleydb", "train", "0007", "amar_lal_rest")
        assert "duplicate_identity" in _findings(clean_tree)

    def test_same_song_from_two_datasets_is_an_error(self, clean_tree):
        _track(clean_tree, "train", "moisesdb", "train", "0044", "amar_lal_rest")
        assert "cross_dataset_duplicate" in _findings(clean_tree)

    def test_index_collision_is_an_error(self, clean_tree):
        _track(clean_tree, "train", "musdb18hq", "train", "0001", "a_different_song")
        assert "index_collision" in _findings(clean_tree)


class TestFilenameLayouts:
    def test_all_legacy_is_fine(self, clean_tree):
        assert "mixed_filename_formats" not in _findings(clean_tree)

    def test_all_current_is_fine(self, tmp_path):
        for stem in ("vocals", "drums", "mixture"):
            _wav(tmp_path, "train", stem, "musdb18hq", "musdb18hq_a_track.wav")
            _wav(tmp_path, "train", stem, "medleydb", "medleydb_another_track.wav")
        assert "mixed_filename_formats" not in _findings(tmp_path)

    def test_mixing_layouts_is_an_error(self, clean_tree):
        """A partial reprocess writes new names beside the old ones."""
        _wav(clean_tree, "train", "vocals", "musdb18hq", "musdb18hq_brand_new.wav")
        found = _findings(clean_tree)
        assert found["mixed_filename_formats"].severity == ERROR
        assert not run_audit(OutputTree(clean_tree).load())["passed"]

    def test_index_collisions_only_checked_on_legacy_names(self, tmp_path):
        """Current-layout names have no index, so the check must not misfire."""
        for stem in ("vocals", "mixture"):
            _wav(tmp_path, "train", stem, "musdb18hq", "musdb18hq_first_track.wav")
            _wav(tmp_path, "train", stem, "musdb18hq", "musdb18hq_second_track.wav")
        assert "index_collision" not in _findings(tmp_path)


class TestPathAgreement:
    def test_filename_split_must_match_directory(self, clean_tree):
        _wav(clean_tree, "val", "vocals", "musdb18hq", "musdb18hq_train_0055_x_y.wav")
        found = _findings(clean_tree)
        assert "split_dir_mismatch" in found
        assert found["split_dir_mismatch"].severity == ERROR

    def test_filename_source_must_match_directory(self, clean_tree):
        _wav(clean_tree, "train", "vocals", "moisesdb", "musdb18hq_train_0056_x_y.wav")
        assert "source_dir_mismatch" in _findings(clean_tree)


class TestCompleteness:
    def test_mixture_without_stems_is_an_error(self, clean_tree):
        _track(clean_tree, "train", "medleydb", "train", "0009", "only_a_mix", stems=("mixture",))
        assert "no_stems" in _findings(clean_tree)

    def test_missing_mixture_is_a_warning(self, clean_tree):
        _track(clean_tree, "train", "medleydb", "train", "0010", "no_mix", stems=("vocals",))
        found = _findings(clean_tree)
        assert found["missing_mixture"].severity == "warning"
        assert run_audit(OutputTree(clean_tree).load())["passed"]


class TestMetadataAgreement:
    def _meta(self, root, name, payload):
        meta = root / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / name).write_text(json.dumps(payload))

    def test_truncated_manifest_is_an_error(self, clean_tree):
        self._meta(clean_tree, "manifest.json", {
            "k": {"source_dataset": "musdb18hq", "original_track_name": "x",
                  "artist": "Actions", "title": "Devil's Words", "split": "train",
                  "available_stems": [], "profile": "vdbo"}
        })
        found = _findings(clean_tree)
        assert found["manifest_count_mismatch"].severity == ERROR

    def test_split_disagreement_is_an_error(self, clean_tree):
        self._meta(clean_tree, "manifest.json", {
            "k": {"source_dataset": "medleydb", "original_track_name": "AmarLal_Rest",
                  "artist": "Amar Lal", "title": "Rest", "split": "train",
                  "available_stems": [], "profile": "vdbo"},
        })
        # splits.json claims val, disk says train
        self._meta(clean_tree, "splits.json", {"medleydb_0002_AmarLal_Rest": "val"})
        assert "split_disagreement" in _findings(clean_tree)

    def test_skipped_overlap_track_present_on_disk_is_an_error(self, clean_tree):
        _track(clean_tree, "train", "musdb18hq", "train", "0088", "aimee_norwich_child")
        self._meta(clean_tree, "overlap_registry.json",
                   {"skipped_count": 1, "skipped_tracks": ["Aimee Norwich - Child"]})
        assert "overlap_registry_conflict" in _findings(clean_tree)
