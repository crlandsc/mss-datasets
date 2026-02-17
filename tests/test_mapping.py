"""Tests for mapping/profiles — instrument mapping, profiles, MoisesDB routing."""

import pytest

from mss_datasets.mapping import (
    BASS_ROUTING,
    MOISESDB_VDBO,
    MOISESDB_VDBO_GP,
    PERCUSSION_ROUTING,
    PROFILES,
    VDBO,
    VDBO_GP,
    get_moisesdb_mapping,
    load_medleydb_mapping,
    resolve_medleydb_label,
)


class TestProfiles:
    def test_vdbo_stems(self):
        assert VDBO.stems == ("vocals", "drums", "bass", "other")

    def test_vdbo_gp_stems(self):
        assert VDBO_GP.stems == ("vocals", "drums", "bass", "guitar", "piano", "other")

    def test_profiles_dict(self):
        assert PROFILES["vdbo"] is VDBO
        assert PROFILES["vdbo+gp"] is VDBO_GP


class TestMoisesDBMapping:
    def test_vdbo_keys(self):
        m = get_moisesdb_mapping(VDBO)
        assert set(m.keys()) == {"vocals", "drums", "other"}
        # percussion and bass NOT in top-level mapping
        for sources in m.values():
            assert "percussion" not in sources
            assert "bass" not in sources

    def test_vdbo_gp_keys(self):
        m = get_moisesdb_mapping(VDBO_GP)
        assert set(m.keys()) == {"vocals", "drums", "guitar", "piano", "other"}

    def test_percussion_routing(self):
        assert PERCUSSION_ROUTING["a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"] == "drums"
        assert PERCUSSION_ROUTING["pitched_percussion_(mallets,_glockenspiel,_...)"] == "other"

    def test_bass_routing(self):
        assert BASS_ROUTING["bass_guitar"] == "bass"
        assert BASS_ROUTING["bass_synthesizer_(moog_etc)"] == "bass"
        assert BASS_ROUTING["contrabass/double_bass_(bass_of_instrings)"] == "bass"
        assert BASS_ROUTING["tuba_(bass_of_brass)"] == "other"
        assert BASS_ROUTING["bassoon_(bass_of_woodwind)"] == "other"


# All expected labels per category (from spec §5.2)
VOCAL_LABELS = [
    "male singer", "female singer", "male speaker", "female speaker",
    "male rapper", "female rapper", "male screamer", "female screamer",
    "vocalists", "choir", "beatboxing",
]

DRUM_LABELS = [
    "drum set", "drum machine", "kick drum", "snare drum", "bass drum",
    "toms", "timpani", "bongo", "conga", "darbuka", "doumbek", "tabla",
    "tambourine", "auxiliary percussion", "high hat", "cymbal", "gong",
    "triangle", "cowbell", "sleigh bells", "cabasa", "guiro", "gu",
    "castanet", "claps", "rattle", "shaker", "maracas", "snaps",
]

BASS_LABELS = ["electric bass", "double bass"]

GUITAR_LABELS = [
    "acoustic guitar", "clean electric guitar", "distorted electric guitar",
    "lap steel guitar", "slide guitar",
]

PIANO_LABELS = ["piano", "tack piano", "electric piano"]

OTHER_LABELS = [
    # Pitched percussion
    "xylophone", "vibraphone", "marimba", "glockenspiel", "chimes",
    # Bowed strings
    "erhu", "violin", "viola", "cello", "dilruba",
    "violin section", "viola section", "cello section", "string section",
    # Plucked strings
    "banjo", "guzheng", "harp", "harpsichord", "liuqin",
    "mandolin", "oud", "sitar", "ukulele", "zhongruan",
    # Struck strings
    "dulcimer", "yangqin",
    # Flutes
    "dizi", "flute", "flute section", "piccolo", "bamboo flute", "panpipes", "recorder",
    # Single reeds
    "alto saxophone", "baritone saxophone", "bass clarinet", "clarinet",
    "clarinet section", "tenor saxophone", "soprano saxophone",
    # Double reeds
    "oboe", "english horn", "bassoon", "bagpipe",
    # Brass
    "trumpet", "cornet", "trombone", "french horn", "euphonium", "tuba",
    "brass section", "french horn section", "trombone section", "horn section", "trumpet section",
    # Free reeds
    "harmonica", "concertina", "accordion", "bandoneon", "harmonium", "pipe organ", "melodica",
    # Electronic
    "synthesizer", "theremin", "fx/processed sound", "scratch", "sampler", "electronic organ",
    # Voices
    "crowd",
]


class TestMedleyDBMappingVDBO:
    @pytest.fixture(scope="class")
    def mapping(self):
        return load_medleydb_mapping(VDBO)

    def test_label_count(self, mapping):
        # 119 mapped labels (excl Main System and Unlabeled)
        assert len(mapping) == 120

    @pytest.mark.parametrize("label", VOCAL_LABELS)
    def test_vocals(self, mapping, label):
        assert mapping[label] == "vocals"

    @pytest.mark.parametrize("label", DRUM_LABELS)
    def test_drums(self, mapping, label):
        assert mapping[label] == "drums"

    @pytest.mark.parametrize("label", BASS_LABELS)
    def test_bass(self, mapping, label):
        assert mapping[label] == "bass"

    @pytest.mark.parametrize("label", GUITAR_LABELS)
    def test_guitar_goes_to_other(self, mapping, label):
        assert mapping[label] == "other"

    @pytest.mark.parametrize("label", PIANO_LABELS)
    def test_piano_goes_to_other(self, mapping, label):
        assert mapping[label] == "other"

    @pytest.mark.parametrize("label", OTHER_LABELS)
    def test_other(self, mapping, label):
        assert mapping[label] == "other"


class TestMedleyDBMappingVDBOGP:
    @pytest.fixture(scope="class")
    def mapping(self):
        return load_medleydb_mapping(VDBO_GP)

    def test_label_count(self, mapping):
        assert len(mapping) == 120

    @pytest.mark.parametrize("label", GUITAR_LABELS)
    def test_guitar(self, mapping, label):
        assert mapping[label] == "guitar"

    @pytest.mark.parametrize("label", PIANO_LABELS)
    def test_piano(self, mapping, label):
        assert mapping[label] == "piano"

    @pytest.mark.parametrize("label", VOCAL_LABELS)
    def test_vocals_same(self, mapping, label):
        assert mapping[label] == "vocals"


class TestResolveMedleyDBLabel:
    @pytest.fixture(scope="class")
    def mapping(self):
        return load_medleydb_mapping(VDBO)

    def test_normal_label(self, mapping):
        stem, flags = resolve_medleydb_label("violin", mapping)
        assert stem == "other"
        assert flags == []

    def test_case_insensitive(self, mapping):
        stem, flags = resolve_medleydb_label("Violin", mapping)
        assert stem == "other"

    def test_main_system_excluded(self, mapping):
        stem, flags = resolve_medleydb_label("Main System", mapping)
        assert stem is None
        assert "exclude" in flags

    def test_main_system_case_insensitive(self, mapping):
        stem, flags = resolve_medleydb_label("main system", mapping)
        assert stem is None

    def test_unlabeled(self, mapping):
        stem, flags = resolve_medleydb_label("Unlabeled", mapping)
        assert stem == "other"
        assert "unlabeled" in flags

    def test_unknown_label(self, mapping):
        stem, flags = resolve_medleydb_label("totally_unknown_instrument", mapping)
        assert stem == "other"
        assert "unknown_label" in flags
