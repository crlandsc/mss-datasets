"""Tests for the shared output-tree reader."""

import json

import pytest

from mss_datasets.output_tree import (
    VIEW_MEDLEYDB_EXTRA,
    VIEW_MOISESDB,
    VIEW_MUSDB,
    OutputTree,
)


def _wav(root, *parts):
    """Create a placeholder file at root/parts."""
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


@pytest.fixture
def tree_dir(tmp_path):
    """A minimal split + grouped output tree covering all three sources."""
    for stem in ("vocals", "drums", "mixture"):
        _wav(tmp_path, "train", stem, "musdb18hq", "musdb18hq_train_0001_actions_devil_s_words.wav")
        # AimeeNorwich_Child is one of the 46 MUSDB<->MedleyDB overlap tracks
        _wav(tmp_path, "train", stem, "medleydb", "medleydb_train_0002_aimee_norwich_child.wav")
        # AmarLal_Rest is MedleyDB-only
        _wav(tmp_path, "train", stem, "medleydb", "medleydb_train_0003_amar_lal_rest.wav")
        _wav(tmp_path, "val", stem, "moisesdb", "moisesdb_val_0004_some_artist_some_song.wav")
    return tmp_path


def _write_manifest(root, entries):
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "manifest.json").write_text(json.dumps(entries))


class TestScanning:
    def test_finds_all_files(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        assert len(tree.files) == 12
        assert not tree.unparsed

    def test_groups_into_tracks(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        assert len(tree.tracks) == 4
        for track in tree.tracks.values():
            assert track.stems == {"vocals", "drums", "mixture"}

    def test_decomposes_paths(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        f = next(f for f in tree.files if f.source == "moisesdb")
        assert (f.split_dir, f.dataset_dir, f.split, f.index) == ("val", "moisesdb", "val", "0004")

    def test_skips_metadata_and_generated_views(self, tree_dir):
        _wav(tree_dir, "metadata", "vocals", "musdb18hq_train_0009_x_y.wav")
        _wav(tree_dir, "_sorted", "train", "vocals", "musdb18hq", "musdb18hq_train_0001_actions_devil_s_words.wav")
        tree = OutputTree(tree_dir).load()
        assert len(tree.files) == 12

    def test_flat_layout_without_split_or_dataset_dirs(self, tmp_path):
        _wav(tmp_path, "vocals", "musdb18hq_train_0001_a_b.wav")
        tree = OutputTree(tmp_path).load()
        assert len(tree.tracks) == 1
        f = tree.files[0]
        assert f.split_dir == "" and f.dataset_dir == "" and f.stem == "vocals"

    def test_unparsable_filename_is_recorded(self, tmp_path):
        _wav(tmp_path, "vocals", "not-our-naming-scheme.wav")
        tree = OutputTree(tmp_path).load()
        assert not tree.files
        assert len(tree.unparsed) == 1

    def test_ignores_sibling_dirs_that_merely_look_like_stems(self, tree_dir):
        """Unrelated exports living beside the dataset must not be scanned."""
        _wav(tree_dir, "rendered_clips_ablation", "train", "bass", "sample_0001.wav")
        _wav(tree_dir, "csv_2_hour", "train", "vocals", "chunk_0001.wav")
        tree = OutputTree(tree_dir).load()
        assert len(tree.files) == 12
        assert not tree.unparsed

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(ValueError):
            OutputTree(tmp_path / "nope").load()


class TestFilenameLayouts:
    """Both the current and the legacy filename layouts must parse.

    Trees written before track identity was stabilized use
    {source}_{split}_{index}_{name} and are still on disk, so the reader has to
    understand both — otherwise --audit and --inventory go blind on them.
    """

    def test_current_layout_parses(self, tmp_path):
        _wav(tmp_path, "vocals", "musdb18hq_artist_name_track_title.wav")
        tree = OutputTree(tmp_path).load()
        (f,) = tree.files
        assert (f.source, f.name) == ("musdb18hq", "artist_name_track_title")
        assert f.split == "" and f.index == ""
        assert not f.legacy

    def test_legacy_layout_parses(self, tmp_path):
        _wav(tmp_path, "vocals", "musdb18hq_train_0001_artist_name_track_title.wav")
        tree = OutputTree(tmp_path).load()
        (f,) = tree.files
        assert (f.split, f.index) == ("train", "0001")
        assert f.name == "artist_name_track_title"
        assert f.legacy

    def test_legacy_is_tried_first(self, tmp_path):
        """A legacy name also matches the current pattern, so order matters."""
        _wav(tmp_path, "vocals", "medleydb_val_0042_some_track.wav")
        tree = OutputTree(tmp_path).load()
        (f,) = tree.files
        # Wrong order would yield name="val_0042_some_track".
        assert f.name == "some_track"

    def test_a_name_that_merely_starts_with_a_split_word_is_not_legacy(self, tmp_path):
        _wav(tmp_path, "vocals", "musdb18hq_train_kept_rolling.wav")
        tree = OutputTree(tmp_path).load()
        (f,) = tree.files
        # No 4-digit index, so this is a current-layout name whose artist
        # happens to start with "train".
        assert not f.legacy
        assert f.name == "train_kept_rolling"

    def test_both_layouts_coexist_and_both_parse(self, tmp_path):
        _wav(tmp_path, "vocals", "musdb18hq_train_0001_old_track.wav")
        _wav(tmp_path, "vocals", "musdb18hq_new_track.wav")
        tree = OutputTree(tmp_path).load()
        assert len(tree.files) == 2
        assert {f.legacy for f in tree.files} == {True, False}

    def test_identity_is_layout_independent(self, tmp_path):
        """Same track under either layout resolves to the same identity."""
        _wav(tmp_path, "train", "vocals", "musdb18hq_train_0001_same_track.wav")
        _wav(tmp_path, "val", "vocals", "musdb18hq_same_track.wav")
        tree = OutputTree(tmp_path).load()
        assert len({t.identity for t in tree.tracks.values()}) == 1


class TestReviewViews:
    def test_musdb_and_moisesdb_map_to_own_views(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        views = {t.source: t.view() for t in tree.tracks.values() if t.source != "medleydb"}
        assert views == {"musdb18hq": VIEW_MUSDB, "moisesdb": VIEW_MOISESDB}

    def test_overlap_medleydb_track_shows_under_musdb(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        overlap = next(t for t in tree.tracks.values() if "aimee" in t.name)
        assert overlap.view() == VIEW_MUSDB

    def test_medleydb_only_track_shows_as_extra(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        extra = next(t for t in tree.tracks.values() if "amar" in t.name)
        assert extra.view() == VIEW_MEDLEYDB_EXTRA

    def test_manifest_original_name_drives_classification(self, tree_dir):
        # Filename alone would not identify this as an overlap track; the
        # manifest's original_track_name must be what decides.
        _write_manifest(tree_dir, {
            "k": {
                "source_dataset": "medleydb",
                "original_track_name": "AimeeNorwich_Child",
                "artist": "Aimee Norwich", "title": "Child", "split": "train",
                "available_stems": [], "profile": "vdbo",
            }
        })
        tree = OutputTree(tree_dir).load()
        overlap = next(t for t in tree.tracks.values() if "aimee" in t.name)
        assert overlap.original_track_name == "AimeeNorwich_Child"
        assert overlap.view() == VIEW_MUSDB

    def test_tracks_by_view_partitions_everything(self, tree_dir):
        tree = OutputTree(tree_dir).load()
        grouped = tree.tracks_by_view()
        assert sum(len(v) for v in grouped.values()) == len(tree.tracks)
        assert len(grouped[VIEW_MUSDB]) == 2          # native + overlap
        assert len(grouped[VIEW_MEDLEYDB_EXTRA]) == 1
        assert len(grouped[VIEW_MOISESDB]) == 1
