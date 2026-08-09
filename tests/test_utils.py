"""Tests for utils.py — filename sanitization, collision resolution, canonical names."""

import re

from mss_datasets.utils import canonical_name, resolve_collision, sanitize_filename


class TestSanitizeFilename:
    def test_basic(self):
        result = sanitize_filename("musdb18hq", "Artist Name", "Track Title")
        assert result == "musdb18hq_artist_name_track_title"

    def test_unicode_transliteration(self):
        result = sanitize_filename("medleydb", "Héctor Müller", "Über Straße")
        assert "hector_muller" in result
        assert "uber_strasse" in result

    def test_special_chars_replaced(self):
        result = sanitize_filename("moisesdb", "Art!st", "Track (remix)")
        assert "!" not in result
        assert "(" not in result
        assert ")" not in result

    def test_truncation(self):
        result = sanitize_filename("musdb18hq", "A" * 50, "B" * 50)
        name_part = result[len("musdb18hq_"):]
        assert len(name_part) <= 80

    def test_consecutive_underscores_collapsed(self):
        result = sanitize_filename("musdb18hq", "A  B", "C   D")
        assert "__" not in result

    def test_hyphens_preserved(self):
        result = sanitize_filename("musdb18hq", "A-B", "C-D")
        assert "a-b" in result


class TestFilenameIsStableIdentity:
    """The filename must depend only on dataset metadata.

    Splits get reassigned and discovery order shifts whenever the dataset
    contents or exclusion overrides change. When either fed the filename, a
    renamed track left its old copy behind — with --split-output that put the
    same audio in both train/ and val/.
    """

    def test_no_split_token(self):
        result = sanitize_filename("musdb18hq", "Artist", "Song")
        for split in ("train", "val", "test"):
            assert f"_{split}_" not in result

    def test_no_positional_index(self):
        result = sanitize_filename("musdb18hq", "Artist", "Song")
        assert not re.search(r"_\d{4}_", result)

    def test_same_track_always_yields_the_same_name(self):
        assert sanitize_filename("medleydb", "Aimee Norwich", "Child") == (
            sanitize_filename("medleydb", "Aimee Norwich", "Child")
        )

    def test_source_still_distinguishes_datasets(self):
        assert sanitize_filename("musdb18hq", "A", "B") != (
            sanitize_filename("medleydb", "A", "B")
        )


class TestResolveCollision:
    def test_no_collision(self):
        assert resolve_collision("foo", set()) == "foo"
        assert resolve_collision("foo", {"bar"}) == "foo"

    def test_single_collision(self):
        assert resolve_collision("foo", {"foo"}) == "foo_2"

    def test_multiple_collisions(self):
        assert resolve_collision("foo", {"foo", "foo_2", "foo_3"}) == "foo_4"


class TestCanonicalName:
    def test_musdb_medleydb_match(self):
        musdb = canonical_name("A Classic Education - NightOwl")
        medleydb = canonical_name("AClassicEducation_NightOwl")
        assert musdb == medleydb

    def test_strips_spaces_underscores_hyphens(self):
        assert canonical_name("Foo - Bar") == "foobar"
        assert canonical_name("Foo_Bar") == "foobar"
        assert canonical_name("Foo Bar") == "foobar"

    def test_case_insensitive(self):
        assert canonical_name("ABC") == canonical_name("abc")
