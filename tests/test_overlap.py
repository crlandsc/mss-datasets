"""Tests for overlap registry — deduplication and cross-dataset matching."""

from mss_aggregate.overlap import (
    MUSDB_MEDLEYDB_OVERLAP,
    get_overlap_set,
    is_overlap_track,
    resolve_overlaps,
)


class TestOverlapSet:
    def test_count(self):
        assert len(MUSDB_MEDLEYDB_OVERLAP) == 46

    def test_get_overlap_set_returns_same(self):
        assert get_overlap_set() is MUSDB_MEDLEYDB_OVERLAP

    def test_known_tracks_present(self):
        assert "A Classic Education - NightOwl" in MUSDB_MEDLEYDB_OVERLAP
        assert "Music Delta - Beatles" in MUSDB_MEDLEYDB_OVERLAP
        assert "The So So Glos - Emergency" in MUSDB_MEDLEYDB_OVERLAP

    def test_non_overlap_absent(self):
        assert "Some Random Track - Not Overlap" not in MUSDB_MEDLEYDB_OVERLAP


class TestIsOverlapTrack:
    def test_musdb_format(self):
        assert is_overlap_track("A Classic Education - NightOwl")
        assert not is_overlap_track("Some Unknown - Track")

    def test_medleydb_format_matches(self):
        # MedleyDB uses CamelCase_Underscore format
        assert is_overlap_track("AClassicEducation_NightOwl")
        assert is_overlap_track("MusicDelta_Beatles")
        assert is_overlap_track("ClaraBerryAndWooldog_AirTraffic")

    def test_case_insensitive(self):
        assert is_overlap_track("a classic education - nightowl")
        assert is_overlap_track("MUSIC DELTA - BEATLES")


class TestResolveOverlaps:
    def test_no_medleydb(self):
        result = resolve_overlaps(
            ["A Classic Education - NightOwl", "Some Other - Track"],
            medleydb_present=False,
        )
        assert result["skip_musdb"] == set()
        assert result["musdb_splits"] == {}

    def test_with_medleydb(self):
        musdb_names = [
            "A Classic Education - NightOwl",
            "Some Unique - Track",
            "Music Delta - Beatles",
        ]
        result = resolve_overlaps(musdb_names, medleydb_present=True)
        assert result["skip_musdb"] == {
            "A Classic Education - NightOwl",
            "Music Delta - Beatles",
        }
        assert len(result["musdb_splits"]) == 2
        assert "Some Unique - Track" not in result["skip_musdb"]

    def test_all_46_detected(self):
        result = resolve_overlaps(
            list(MUSDB_MEDLEYDB_OVERLAP),
            medleydb_present=True,
        )
        assert len(result["skip_musdb"]) == 46
