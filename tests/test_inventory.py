"""Tests for the regrouped inventory and the sorted-view builder."""

import os

import pytest

from mss_datasets.inventory import (
    build_inventory,
    build_sorted_view,
    render_report,
)
from mss_datasets.output_tree import (
    VIEW_MEDLEYDB_EXTRA,
    VIEW_MOISESDB,
    VIEW_MUSDB,
    OutputTree,
)


def _track(root, split_dir, source, split, index, name, stems=("vocals", "drums", "mixture")):
    for stem in stems:
        path = root / split_dir / stem / source / f"{source}_{split}_{index}_{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")


@pytest.fixture
def tree(tmp_path):
    _track(tmp_path, "train", "musdb18hq", "train", "0001", "actions_devil_s_words")
    _track(tmp_path, "train", "medleydb", "train", "0002", "aimee_norwich_child")   # overlap
    _track(tmp_path, "train", "medleydb", "train", "0003", "amar_lal_rest")         # extra
    _track(tmp_path, "val", "moisesdb", "val", "0004", "some_artist_some_song")
    return OutputTree(tmp_path).load()


class TestInventory:
    def test_regroups_overlap_track_under_musdb(self, tree):
        inv = build_inventory(tree)
        assert inv["views"][VIEW_MUSDB]["track_count"] == 2
        assert inv["views"][VIEW_MEDLEYDB_EXTRA]["track_count"] == 1
        assert inv["views"][VIEW_MOISESDB]["track_count"] == 1

    def test_views_partition_every_track(self, tree):
        inv = build_inventory(tree)
        assert sum(v["track_count"] for v in inv["views"].values()) == inv["total_tracks"]

    def test_records_physical_source_of_each_view(self, tree):
        inv = build_inventory(tree)
        assert inv["views"][VIEW_MUSDB]["by_source"] == {"medleydb": 1, "musdb18hq": 1}

    def test_on_disk_counts_are_unchanged(self, tree):
        inv = build_inventory(tree)
        assert inv["on_disk_by_source"] == {"medleydb": 2, "moisesdb": 1, "musdb18hq": 1}

    def test_stem_coverage(self, tree):
        inv = build_inventory(tree)
        assert inv["views"][VIEW_MUSDB]["stem_counts"]["vocals"] == 2

    def test_report_renders_markdown(self, tree):
        report = render_report(build_inventory(tree))
        assert "# mss-datasets inventory" in report
        assert "medleydb (extra only)" in report
        assert "AmarLal" in report or "amar_lal_rest" in report


class TestSortedView:
    def test_creates_symlinks_not_copies(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        stats = build_sorted_view(tree, dest)
        links = list(dest.rglob("*.wav"))
        assert len(links) == stats["created"] == 12
        assert all(p.is_symlink() for p in links)

    def test_symlinks_are_relative_and_resolve(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        build_sorted_view(tree, dest)
        for link in dest.rglob("*.wav"):
            assert not os.path.isabs(os.readlink(link))
            assert link.resolve().is_file()

    def test_layout_is_split_stem_view(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        build_sorted_view(tree, dest)
        assert (dest / "train" / "vocals" / VIEW_MUSDB).is_dir()
        assert (dest / "train" / "vocals" / VIEW_MEDLEYDB_EXTRA).is_dir()
        assert (dest / "val" / "vocals" / VIEW_MOISESDB).is_dir()

    def test_overlap_track_lands_in_musdb_view(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        build_sorted_view(tree, dest)
        names = {p.name for p in (dest / "train" / "vocals" / VIEW_MUSDB).iterdir()}
        assert "medleydb_train_0002_aimee_norwich_child.wav" in names
        assert len(names) == 2

    def test_source_tree_is_untouched(self, tree, tmp_path):
        before = sorted(p.relative_to(tree.root) for p in tree.root.rglob("*.wav"))
        build_sorted_view(tree, tmp_path / "_sorted")
        after = sorted(p.relative_to(tree.root) for p in tree.root.rglob("*.wav")
                       if "_sorted" not in p.parts)
        assert before == after
        assert all(not p.is_symlink() for p in tree.root.rglob("*.wav")
                   if "_sorted" not in p.parts)

    def test_rerun_prunes_stale_entries(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        build_sorted_view(tree, dest)
        stale = dest / "train" / "vocals" / VIEW_MUSDB / "musdb18hq_train_9999_gone.wav"
        stale.symlink_to("nowhere.wav")
        stats = build_sorted_view(tree, dest)
        assert not stale.exists()
        assert stats["created"] == 12

    def test_hardlink_mode(self, tree, tmp_path):
        dest = tmp_path / "_sorted"
        build_sorted_view(tree, dest, link_mode="hardlink")
        links = list(dest.rglob("*.wav"))
        assert links and all(not p.is_symlink() and p.is_file() for p in links)

    def test_rejects_unknown_link_mode(self, tree, tmp_path):
        with pytest.raises(ValueError, match="Unknown link mode"):
            build_sorted_view(tree, tmp_path / "_sorted", link_mode="teleport")

    def test_generated_view_is_ignored_on_rescan(self, tree, tmp_path):
        """A view built inside the dataset root must not be scanned back in."""
        build_sorted_view(tree, tree.root / "_sorted")
        rescanned = OutputTree(tree.root).load()
        assert len(rescanned.tracks) == len(tree.tracks)
