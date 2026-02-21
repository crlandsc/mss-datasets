"""Stem profile definitions and MoisesDB mapping dicts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MAPPING_DIR = Path(__file__).parent


@dataclass(frozen=True)
class StemProfile:
    name: str
    stems: tuple[str, ...]

    @property
    def stem_set(self) -> set[str]:
        return set(self.stems)


VDBO = StemProfile(name="vdbo", stems=("vocals", "drums", "bass", "other"))
VDBO_GP = StemProfile(
    name="vdbo+gp",
    stems=("vocals", "drums", "bass", "guitar", "piano", "other"),
)

PROFILES = {"vdbo": VDBO, "vdbo+gp": VDBO_GP}

# ---------------------------------------------------------------------------
# MoisesDB top-level mappings (excludes percussion and bass — handled at sub-stem level)
# ---------------------------------------------------------------------------

MOISESDB_VDBO = {
    "vocals": ["vocals"],
    "drums": ["drums"],
    "other": ["other", "guitar", "other_plucked", "piano", "other_keys", "bowed_strings", "wind"],
}

MOISESDB_VDBO_GP = {
    "vocals": ["vocals"],
    "drums": ["drums"],
    "guitar": ["guitar"],
    "piano": ["piano"],
    "other": ["other", "other_plucked", "other_keys", "bowed_strings", "wind"],
}

# Sub-stem routing for percussion (same for both profiles)
# Keys use underscores to match moisesdb library's name format (spaces → underscores)
PERCUSSION_ROUTING = {
    "a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)": "drums",
    "pitched_percussion_(mallets,_glockenspiel,_...)": "other",
}

# Sub-stem routing for bass (same for both profiles)
# Keys use underscores to match moisesdb library's name format (spaces → underscores)
BASS_ROUTING = {
    "bass_guitar": "bass",
    "bass_synthesizer_(moog_etc)": "bass",
    "contrabass/double_bass_(bass_of_instrings)": "bass",
    "tuba_(bass_of_brass)": "other",
    "bassoon_(bass_of_woodwind)": "other",
}


def get_moisesdb_mapping(profile: StemProfile) -> dict[str, list[str]]:
    """Return top-level MoisesDB mapping for a profile."""
    if profile.name == "vdbo":
        return MOISESDB_VDBO
    if profile.name == "vdbo+gp":
        return MOISESDB_VDBO_GP
    raise ValueError(f"Unknown profile: {profile.name}")


# ---------------------------------------------------------------------------
# MedleyDB instrument mapping (loaded from YAML)
# ---------------------------------------------------------------------------

def load_medleydb_mapping(profile: StemProfile) -> dict[str, str]:
    """Load MedleyDB instrument→stem mapping from YAML.

    Returns dict mapping lowercase instrument label → target stem name.
    """
    yaml_path = MAPPING_DIR / "medleydb_instruments.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    profile_key = profile.name.replace("+", "_")  # "vdbo" or "vdbo_gp"
    mapping = {}
    for entry in data["instruments"]:
        label = entry["label"].lower()
        target = entry.get(profile_key, entry.get("vdbo"))
        mapping[label] = target
    return mapping


def load_medleydb_overrides() -> dict:
    """Load MedleyDB per-track and per-stem override rules from YAML.

    Returns dict with keys:
      - exclude_tracks: set[str] — track directory names to skip entirely
      - exclude_stems: dict[str, set[str]] — track name → set of stem keys to skip
      - reroute_stems: dict[str, dict[str, str]] — track name → {stem key → target}
    """
    yaml_path = MAPPING_DIR / "medleydb_overrides.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {
        "exclude_tracks": set(data.get("exclude_tracks", [])),
        "exclude_stems": {
            k: set(v) for k, v in data.get("exclude_stems", {}).items()
        },
        "reroute_stems": data.get("reroute_stems", {}),
    }


def resolve_medleydb_label(label: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Resolve a MedleyDB instrument label to a target stem.

    Returns (target_stem, flags).
    - "Main System" → (None, ["exclude"])
    - "Unlabeled" → ("other", ["unlabeled"])
    - Unknown → ("other", ["unknown_label"])
    """
    lower = label.lower()

    if lower == "main system":
        return None, ["exclude"]
    if lower == "unlabeled":
        return "other", ["unlabeled"]

    target = mapping.get(lower)
    if target is None:
        logger.warning("Unknown MedleyDB instrument label %r — routing to 'other'", label)
        return "other", ["unknown_label"]
    return target, []
