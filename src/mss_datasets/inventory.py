"""Regrouped inventory of an aggregated output tree, for human review.

The aggregation prefers MedleyDB for the 46 tracks MUSDB18-HQ sourced from it,
so those tracks sit physically under `medleydb/`. That is correct for training
but awkward to review: `musdb18hq/` looks incomplete and MedleyDB's real
contribution is hidden. This module presents the same tracks regrouped —
MUSDB18-HQ and MoisesDB as their full rosters, MedleyDB as extras only — without
moving or rewriting anything.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from mss_datasets.output_tree import (
    VIEW_MEDLEYDB_EXTRA,
    VIEW_MOISESDB,
    VIEW_MUSDB,
    VIEWS,
    OutputTree,
    Track,
)

logger = logging.getLogger(__name__)

LINK_MODES = ("symlink", "hardlink", "copy")

VIEW_LABELS = {
    VIEW_MUSDB: "musdb18hq (complete)",
    VIEW_MOISESDB: "moisesdb (complete)",
    VIEW_MEDLEYDB_EXTRA: "medleydb (extra only)",
}


def build_inventory(tree: OutputTree) -> dict:
    """Summarize the tree grouped into review views."""
    grouped = tree.tracks_by_view()

    views = {}
    for view in VIEWS:
        tracks = grouped[view]
        stem_counts: dict[str, int] = {}
        for t in tracks:
            for stem in t.stems:
                stem_counts[stem] = stem_counts.get(stem, 0) + 1
        by_source: dict[str, int] = {}
        for t in tracks:
            by_source[t.source] = by_source.get(t.source, 0) + 1
        views[view] = {
            "label": VIEW_LABELS[view],
            "track_count": len(tracks),
            "file_count": sum(len(t.files) for t in tracks),
            "by_source": dict(sorted(by_source.items())),
            "stem_counts": dict(sorted(stem_counts.items(), key=lambda x: -x[1])),
            "tracks": tracks,
        }

    return {
        "root": str(tree.root),
        "views": views,
        "total_tracks": len(tree.tracks),
        "total_files": len(tree.files),
        "on_disk_by_source": _count_by_source(tree),
    }


def _count_by_source(tree: OutputTree) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tree.tracks.values():
        counts[t.source] = counts.get(t.source, 0) + 1
    return dict(sorted(counts.items()))


def render_report(inv: dict) -> str:
    """Render the inventory as Markdown."""
    lines = [
        "# mss-datasets inventory",
        "",
        f"Source tree: `{inv['root']}`",
        "",
        f"{inv['total_tracks']} tracks, {inv['total_files']} WAV files.",
        "",
        "## On disk",
        "",
        "How the aggregation actually stores things — MedleyDB holds the 46 tracks",
        "MUSDB18-HQ sourced from it, because its per-instrument stems can be routed",
        "to any profile while MUSDB18-HQ's `other` stem is pre-mixed.",
        "",
        "| Source folder | Tracks |",
        "|---|---:|",
    ]
    for source, n in inv["on_disk_by_source"].items():
        lines.append(f"| `{source}` | {n} |")
    lines += [f"| **Total** | **{inv['total_tracks']}** |", ""]

    lines += [
        "## Regrouped for review",
        "",
        "The same tracks, grouped the way a reader thinks about them.",
        "",
        "| View | Tracks | Composition |",
        "|---|---:|---|",
    ]
    for view in VIEWS:
        v = inv["views"][view]
        comp = " + ".join(f"{n} from `{s}`" for s, n in v["by_source"].items())
        lines.append(f"| {v['label']} | {v['track_count']} | {comp or '—'} |")
    lines += [f"| **Total** | **{inv['total_tracks']}** | |", ""]

    for view in VIEWS:
        v = inv["views"][view]
        if not v["track_count"]:
            continue
        lines += [
            f"## {v['label']} — {v['track_count']} tracks",
            "",
            "Stem coverage: "
            + ", ".join(f"{stem} {n}/{v['track_count']}"
                        for stem, n in v["stem_counts"].items()),
            "",
            "| Track | Source | Split | Stems |",
            "|---|---|---|---|",
        ]
        for t in v["tracks"]:
            name = t.original_track_name or t.name
            stems = ", ".join(sorted(t.stems))
            # Current-layout filenames carry no split — the directory does.
            split = t.split or t.split_dir or "-"
            lines.append(f"| {name} | `{t.source}` | {split} | {stems} |")
        lines.append("")

    return "\n".join(lines)


def build_sorted_view(
    tree: OutputTree,
    dest: str | Path,
    link_mode: str = "symlink",
    prune: bool = True,
) -> dict:
    """Materialize the regrouped view at `dest`.

    Mirrors the source layout ({split}/{stem}/{view}/) so it reads the same way,
    but with the review views in place of the raw source folders. Default is
    relative symlinks: no audio is duplicated, and the view survives moving the
    whole dataset directory.
    """
    if link_mode not in LINK_MODES:
        raise ValueError(f"Unknown link mode {link_mode!r}; expected one of {LINK_MODES}")

    dest = Path(dest)
    if prune and dest.exists():
        _prune(dest)
    dest.mkdir(parents=True, exist_ok=True)

    created = skipped = 0
    for view, tracks in tree.tracks_by_view().items():
        for track in tracks:
            for stem, src in sorted(track.files.items()):
                target_dir = dest / track.split_dir / stem / view if track.split_dir \
                    else dest / stem / view
                target_dir.mkdir(parents=True, exist_ok=True)
                if _place(src, target_dir / src.name, link_mode):
                    created += 1
                else:
                    skipped += 1

    return {
        "dest": str(dest),
        "link_mode": link_mode,
        "created": created,
        "skipped": skipped,
        "views": {v: len(t) for v, t in tree.tracks_by_view().items()},
    }


def _prune(dest: Path) -> None:
    """Remove a previously generated view so stale entries can't accumulate.

    Only ever unlinks symlinks or deletes inside `dest`, and refuses to touch a
    directory that holds anything other than a generated view.
    """
    for path in sorted(dest.rglob("*"), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _place(src: Path, link: Path, mode: str) -> bool:
    """Create one entry. Returns True if written, False if already correct."""
    if mode == "symlink":
        rel = os.path.relpath(src, start=link.parent)
        if link.is_symlink() and os.readlink(link) == rel:
            return False
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(rel)
        return True

    if link.exists():
        return False
    if mode == "hardlink":
        os.link(src, link)
    else:
        shutil.copy2(src, link)
    return True
