"""Instrument mapping and stem profile definitions."""

from mss_datasets.mapping.profiles import (
    BASS_ROUTING,
    MOISESDB_VDBO,
    MOISESDB_VDBO_GP,
    PERCUSSION_ROUTING,
    PROFILES,
    VDBO,
    VDBO_GP,
    StemProfile,
    get_moisesdb_mapping,
    load_medleydb_mapping,
    resolve_medleydb_label,
)

__all__ = [
    "BASS_ROUTING",
    "MOISESDB_VDBO",
    "MOISESDB_VDBO_GP",
    "PERCUSSION_ROUTING",
    "PROFILES",
    "VDBO",
    "VDBO_GP",
    "StemProfile",
    "get_moisesdb_mapping",
    "load_medleydb_mapping",
    "resolve_medleydb_label",
]
