"""Tests for utils.py — filename sanitization, collision resolution, canonical names."""

from mss_datasets.utils import canonical_name, resolve_collision, sanitize_filename


class TestSanitizeFilename:
    def test_basic(self):
        result = sanitize_filename("musdb18hq", "train", 1, "Artist Name", "Track Title")
        assert result == "musdb18hq_train_0001_artist_name_track_title"

    def test_unicode_transliteration(self):
        result = sanitize_filename("medleydb", "train", 3, "Héctor Müller", "Über Straße")
        assert "hector_muller" in result
        assert "uber_strasse" in result

    def test_special_chars_replaced(self):
        result = sanitize_filename("moisesdb", "val", 42, "Art!st", "Track (remix)")
        assert "!" not in result
        assert "(" not in result
        assert ")" not in result

    def test_truncation(self):
        long_artist = "A" * 50
        long_title = "B" * 50
        result = sanitize_filename("musdb18hq", "train", 1, long_artist, long_title)
        # Prefix is "musdb18hq_train_0001_" = 20 chars, name part <= 80
        name_part = result[len("musdb18hq_train_0001_"):]
        assert len(name_part) <= 80

    def test_consecutive_underscores_collapsed(self):
        result = sanitize_filename("musdb18hq", "train", 1, "A  B", "C   D")
        assert "__" not in result

    def test_index_zero_padded(self):
        result = sanitize_filename("musdb18hq", "train", 1, "A", "B")
        assert "_0001_" in result
        result2 = sanitize_filename("musdb18hq", "train", 9999, "A", "B")
        assert "_9999_" in result2

    def test_hyphens_preserved(self):
        result = sanitize_filename("musdb18hq", "train", 1, "A-B", "C-D")
        assert "a-b" in result


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
