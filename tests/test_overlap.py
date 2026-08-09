"""Tests for overlap registry — deduplication and cross-dataset matching."""

import csv
from pathlib import Path
from types import SimpleNamespace

from mss_datasets.overlap import (
    MUSDB_MEDLEYDB_OVERLAP,
    get_overlap_set,
    is_overlap_track,
    resolve_cross_dataset,
    resolve_overlaps,
)
from mss_datasets.utils import canonical_name

FIXTURES = Path(__file__).parent / "fixtures"


def _musdb_tracklist() -> list[dict]:
    """The official MUSDB18 tracklist, with a per-track Source column."""
    with open(FIXTURES / "musdb18_tracklist.csv") as f:
        return list(csv.DictReader(f))


def _medleydb_tracks(version: str) -> list[str]:
    """MedleyDB track directory names for v1 or v2."""
    path = FIXTURES / f"medleydb_tracklist_{version}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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


def _track(dataset: str, name: str, has_bleed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        source_dataset=dataset, original_track_name=name, has_bleed=has_bleed
    )


class TestResolveCrossDataset:
    """All-pairs dedup — unlike resolve_overlaps this also sees MoisesDB."""

    def test_no_collisions_yields_no_groups(self):
        tracks = [
            _track("musdb18hq", "Some Artist - A"),
            _track("medleydb", "OtherArtist_B"),
            _track("moisesdb", "Third Artist - C"),
        ]
        assert resolve_cross_dataset(tracks) == []

    def test_medleydb_beats_musdb(self):
        tracks = [
            _track("musdb18hq", "Aimee Norwich - Child"),
            _track("medleydb", "AimeeNorwich_Child"),
        ]
        (group,) = resolve_cross_dataset(tracks)
        assert group.winner == ("medleydb", "AimeeNorwich_Child")
        assert group.losers == (("musdb18hq", "Aimee Norwich - Child"),)

    def test_moisesdb_loses_to_both(self):
        """The gap this ticket closes — MoisesDB was never compared before."""
        tracks = [
            _track("moisesdb", "Aimee Norwich - Child"),
            _track("musdb18hq", "Aimee Norwich - Child"),
            _track("medleydb", "AimeeNorwich_Child"),
        ]
        (group,) = resolve_cross_dataset(tracks)
        assert group.winner[0] == "medleydb"
        assert {d for d, _ in group.losers} == {"musdb18hq", "moisesdb"}
        assert group.datasets == ("medleydb", "moisesdb", "musdb18hq")

    def test_moisesdb_wins_when_it_is_the_only_source(self):
        tracks = [_track("moisesdb", "Solo Artist - Song")]
        assert resolve_cross_dataset(tracks) == []

    def test_bleed_track_loses_despite_higher_precedence(self):
        """Otherwise the bleed filter drops the winner and the song is lost."""
        tracks = [
            _track("medleydb", "AimeeNorwich_Child", has_bleed=True),
            _track("musdb18hq", "Aimee Norwich - Child"),
        ]
        (group,) = resolve_cross_dataset(tracks)
        assert group.winner[0] == "musdb18hq"

    def test_bleed_is_ignored_when_bleed_tracks_are_kept(self):
        tracks = [
            _track("medleydb", "AimeeNorwich_Child", has_bleed=True),
            _track("musdb18hq", "Aimee Norwich - Child"),
        ]
        (group,) = resolve_cross_dataset(tracks, prefer_usable=False)
        assert group.winner[0] == "medleydb"

    def test_custom_precedence_is_honoured(self):
        tracks = [
            _track("musdb18hq", "Aimee Norwich - Child"),
            _track("medleydb", "AimeeNorwich_Child"),
        ]
        (group,) = resolve_cross_dataset(
            tracks, precedence=("musdb18hq", "medleydb", "moisesdb")
        )
        assert group.winner[0] == "musdb18hq"

    def test_within_dataset_duplicates_are_kept(self):
        """Same name twice in one dataset may be two different songs — keep both."""
        tracks = [
            _track("musdb18hq", "Test Artist - Test Song"),
            _track("musdb18hq", "Test Artist - Test Song"),
        ]
        assert resolve_cross_dataset(tracks) == []

    def test_same_dataset_sibling_of_a_winner_is_not_a_loser(self):
        tracks = [
            _track("medleydb", "AimeeNorwich_Child"),
            _track("medleydb", "Aimee Norwich Child"),
            _track("musdb18hq", "Aimee Norwich - Child"),
        ]
        (group,) = resolve_cross_dataset(tracks)
        assert group.winner[0] == "medleydb"
        assert group.losers == (("musdb18hq", "Aimee Norwich - Child"),)

    def test_real_46_all_resolve_to_medleydb(self):
        """Regression: the shipped behaviour must not change."""
        medleydb = [
            _track("medleydb", t)
            for t in _medleydb_tracks("v1") + _medleydb_tracks("v2")
        ]
        musdb = [_track("musdb18hq", r["Track Name"]) for r in _musdb_tracklist()]
        groups = resolve_cross_dataset(medleydb + musdb)

        assert len(groups) == 46
        assert all(g.winner[0] == "medleydb" for g in groups)
        skipped = {name for g in groups for dataset, name in g.losers}
        assert skipped == set(MUSDB_MEDLEYDB_OVERLAP)


class TestOverlapCompleteness:
    """Check the hardcoded list against the real MUSDB18 and MedleyDB track lists.

    The tests above only prove the list is self-consistent. These prove it is
    *correct* — without them a missing entry passes CI and silently leaks a
    duplicate track into training.

    Fixtures are vendored from the upstream projects rather than fetched, so CI
    never touches the network:
      - musdb18_tracklist.csv        sigsep/website, content/datasets/assets/
      - medleydb_tracklist_v{1,2}.txt  marl/medleydb, medleydb/resources/
    """

    def test_fixtures_match_the_published_datasets(self):
        assert len(_musdb_tracklist()) == 150
        assert len(_medleydb_tracks("v1")) == 122
        assert len(_medleydb_tracks("v2")) == 74

    def test_overlap_list_is_exactly_the_medleydb_sourced_set(self):
        published = {
            row["Track Name"]
            for row in _musdb_tracklist()
            if row["Source"].strip().lower() == "medleydb"
        }
        # Set equality fails on a missing *or* an extra entry.
        assert published == set(MUSDB_MEDLEYDB_OVERLAP)

    def test_every_overlap_track_resolves_to_a_medleydb_directory(self):
        installed = {
            canonical_name(t) for t in _medleydb_tracks("v1") + _medleydb_tracks("v2")
        }
        unresolved = [
            name for name in sorted(MUSDB_MEDLEYDB_OVERLAP)
            if canonical_name(name) not in installed
        ]
        # An unresolvable entry means we skip the MUSDB track and nothing replaces it.
        assert unresolved == []

    def test_medleydb_versions_are_disjoint(self):
        assert not set(_medleydb_tracks("v1")) & set(_medleydb_tracks("v2"))

    def test_no_collision_between_the_datasets_escapes_the_list(self):
        """Full canonical-name sweep: every MUSDB/MedleyDB collision must be covered."""
        musdb = {canonical_name(r["Track Name"]) for r in _musdb_tracklist()}
        medleydb = {
            canonical_name(t) for t in _medleydb_tracks("v1") + _medleydb_tracks("v2")
        }
        covered = {canonical_name(n) for n in MUSDB_MEDLEYDB_OVERLAP}
        assert (musdb & medleydb) - covered == set()
