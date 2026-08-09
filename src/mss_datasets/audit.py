"""Correctness audit of an aggregated output tree.

Checks the invariants the pipeline is supposed to guarantee but never verifies:
no duplicated tracks, no track in more than one split, no cross-dataset
duplicates, and metadata that agrees with what is actually on disk.

Run it after any reprocess, especially one writing into a directory that already
held output — the surest way to leave stale copies behind is to reprocess over
an existing tree rather than into a fresh one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mss_datasets.output_tree import OutputTree
from mss_datasets.utils import canonical_name

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"


@dataclass
class Finding:
    check: str
    severity: str
    summary: str
    details: list[str] = field(default_factory=list)


def run_audit(tree: OutputTree) -> dict:
    """Run every check. Returns findings plus a pass/fail verdict."""
    findings: list[Finding] = []
    for check in (
        _check_unparsed,
        _check_mixed_filename_formats,
        _check_path_agreement,
        _check_duplicate_identity,
        _check_split_leakage,
        _check_cross_dataset_duplicates,
        _check_index_collisions,
        _check_incomplete_tracks,
        _check_metadata_agreement,
        _check_overlap_registry,
    ):
        findings.extend(check(tree))

    errors = [f for f in findings if f.severity == ERROR]
    return {
        "root": str(tree.root),
        "total_tracks": len(tree.tracks),
        "total_files": len(tree.files),
        "findings": findings,
        "error_count": len(errors),
        "warning_count": len(findings) - len(errors),
        "passed": not errors,
    }


def _check_unparsed(tree: OutputTree) -> list[Finding]:
    if not tree.unparsed:
        return []
    return [Finding(
        "unparsed_filenames", WARNING,
        f"{len(tree.unparsed)} WAV files don't match the expected filename format",
        [str(p) for p in tree.unparsed[:20]],
    )]


def _check_mixed_filename_formats(tree: OutputTree) -> list[Finding]:
    """A tree must not mix the legacy and current filename layouts.

    Both parse, but a tree containing both means a partial reprocess wrote new
    names alongside the old ones — the stale-copy situation the current layout
    exists to prevent.
    """
    legacy = {f for f in tree.files if f.legacy}
    current = {f for f in tree.files if not f.legacy}
    if not legacy or not current:
        return []
    return [Finding(
        "mixed_filename_formats", ERROR,
        f"{len(legacy)} files use the legacy name layout and {len(current)} the "
        f"current one — a partial reprocess left stale copies behind",
        [f"legacy:  {sorted(f.path.name for f in legacy)[0]}",
         f"current: {sorted(f.path.name for f in current)[0]}",
         "Reprocess into a fresh output directory rather than over an existing tree."],
    )]


def _check_path_agreement(tree: OutputTree) -> list[Finding]:
    """The split and source encoded in a filename must match where it lives.

    Only legacy filenames carry a split, so that half applies to them alone.
    """
    split_bad = [
        f for f in tree.files if f.legacy and f.split_dir and f.split != f.split_dir
    ]
    source_bad = [f for f in tree.files if f.dataset_dir and f.source != f.dataset_dir]
    out = []
    if split_bad:
        out.append(Finding(
            "split_dir_mismatch", ERROR,
            f"{len(split_bad)} files whose filename split disagrees with their directory",
            [f"{f.path}  (filename says {f.split!r}, lives in {f.split_dir!r})"
             for f in split_bad[:20]],
        ))
    if source_bad:
        out.append(Finding(
            "source_dir_mismatch", ERROR,
            f"{len(source_bad)} files whose filename source disagrees with their directory",
            [f"{f.path}  (filename says {f.source!r}, lives in {f.dataset_dir!r})"
             for f in source_bad[:20]],
        ))
    return out


def _check_duplicate_identity(tree: OutputTree) -> list[Finding]:
    """One track must not be written under two different split/index combinations."""
    seen: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for t in tree.tracks.values():
        # Current-layout names carry no split or index, so fall back to the
        # directory the file actually lives in.
        seen[t.identity].add((t.split or t.split_dir, t.index))
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if not dupes:
        return []
    return [Finding(
        "duplicate_identity", ERROR,
        f"{len(dupes)} tracks written under more than one split/index — stale copies",
        [f"{src} {name}: {sorted(v)}" for (src, name), v in sorted(dupes.items())[:20]],
    )]


def _check_split_leakage(tree: OutputTree) -> list[Finding]:
    """The same track must never appear in more than one split directory."""
    splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in tree.tracks.values():
        if t.split_dir:
            splits[t.identity].add(t.split_dir)
    leaked = {k: v for k, v in splits.items() if len(v) > 1}
    if not leaked:
        return []
    return [Finding(
        "split_leakage", ERROR,
        f"{len(leaked)} tracks present in more than one split — train/val leakage",
        [f"{src} {name}: {sorted(v)}" for (src, name), v in sorted(leaked.items())],
    )]


def _check_cross_dataset_duplicates(tree: OutputTree) -> list[Finding]:
    """The same song must not arrive from two different source datasets."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for t in tree.tracks.values():
        by_name[canonical_name(t.name)].add(t.source)
    dupes = {k: v for k, v in by_name.items() if len(v) > 1}
    if not dupes:
        return []
    return [Finding(
        "cross_dataset_duplicate", ERROR,
        f"{len(dupes)} track names present under more than one source dataset",
        [f"{name}: {sorted(v)}" for name, v in sorted(dupes.items())[:20]],
    )]


def _check_index_collisions(tree: OutputTree) -> list[Finding]:
    """Within a source, a legacy index must identify exactly one track.

    Only meaningful for legacy filenames — the current layout has no index,
    which is precisely what made this class of collision impossible.
    """
    by_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in tree.tracks.values():
        if not t.legacy:
            continue
        by_index[(t.source, t.index)].add(canonical_name(t.name))
    collided = {k: v for k, v in by_index.items() if len(v) > 1}
    if not collided:
        return []
    return [Finding(
        "index_collision", ERROR,
        f"{len(collided)} (source, index) pairs mapping to more than one track",
        [f"{src} {idx}: {sorted(v)}" for (src, idx), v in sorted(collided.items())[:20]],
    )]


def _check_incomplete_tracks(tree: OutputTree) -> list[Finding]:
    """A track with no stems at all, or no mixture where mixtures are in use."""
    out = []
    stem_only = [t for t in tree.tracks.values() if not (t.stems - {"mixture"})]
    if stem_only:
        out.append(Finding(
            "no_stems", ERROR,
            f"{len(stem_only)} tracks have a mixture but no stems",
            [f"{t.source} {t.name}" for t in stem_only[:20]],
        ))
    has_mixture = any("mixture" in t.stems for t in tree.tracks.values())
    if has_mixture:
        missing = [t for t in tree.tracks.values() if "mixture" not in t.stems]
        if missing:
            out.append(Finding(
                "missing_mixture", WARNING,
                f"{len(missing)} tracks have stems but no mixture file",
                [f"{t.source} {t.name}" for t in missing[:20]],
            ))
    return out


def _check_metadata_agreement(tree: OutputTree) -> list[Finding]:
    """manifest.json, splits.json and the files on disk must describe one dataset."""
    out = []
    n_disk = len(tree.tracks)

    if tree.manifest and len(tree.manifest) != n_disk:
        out.append(Finding(
            "manifest_count_mismatch", ERROR,
            f"manifest.json lists {len(tree.manifest)} tracks but {n_disk} are on disk",
            ["The manifest is rewritten from scratch each run and only records tracks "
             "processed in that invocation, so a resumed run truncates it."],
        ))
    if tree.splits and len(tree.splits) != n_disk:
        out.append(Finding(
            "splits_count_mismatch", WARNING,
            f"splits.json lists {len(tree.splits)} tracks but {n_disk} are on disk",
            ["splits.json covers every discovered track, so it can legitimately "
             "exceed the on-disk count when tracks produced no output."],
        ))

    # Where the splits key can be reconstructed, the recorded split must match.
    if tree.splits:
        disagree = []
        for t in tree.tracks.values():
            # Current layout keys splits.json on the filename base; the legacy
            # layout used source + positional index + original name.
            candidates = [f"{t.source}_{t.name}"]
            if t.legacy and t.original_track_name:
                candidates.append(f"{t.source}_{t.index}_{t.original_track_name}")
            key = next((k for k in candidates if k in tree.splits), None)
            if key is None:
                continue
            recorded = tree.splits[key]
            if t.split_dir and recorded != t.split_dir:
                disagree.append(f"{key}: splits.json says {recorded!r}, on disk {t.split_dir!r}")
        if disagree:
            out.append(Finding(
                "split_disagreement", ERROR,
                f"{len(disagree)} tracks stored in a different split than splits.json records",
                disagree[:20],
            ))
    return out


def _check_overlap_registry(tree: OutputTree) -> list[Finding]:
    """Skipped MUSDB tracks must not also be present on disk."""
    if not tree.overlap_registry:
        return []
    skipped = tree.overlap_registry.get("skipped_tracks", [])
    present = {canonical_name(t.name) for t in tree.tracks.values()
               if t.source == "musdb18hq"}
    resurrected = [s for s in skipped if canonical_name(s) in present]
    if not resurrected:
        return []
    return [Finding(
        "overlap_registry_conflict", ERROR,
        f"{len(resurrected)} tracks recorded as skipped are present under musdb18hq/",
        resurrected[:20],
    )]


def render_audit(result: dict) -> str:
    """Render audit results as plain text."""
    lines = [
        "MSS Datasets — Audit",
        "=" * 40,
        f"Tree: {result['root']}",
        f"Tracks: {result['total_tracks']}   Files: {result['total_files']}",
        "",
    ]
    if not result["findings"]:
        lines.append("PASS — no duplicates, no split leakage, metadata agrees with disk.")
        return "\n".join(lines)

    for f in result["findings"]:
        lines.append(f"[{f.severity.upper()}] {f.check}: {f.summary}")
        for d in f.details:
            lines.append(f"    {d}")
        lines.append("")
    verdict = "PASS (warnings only)" if result["passed"] else "FAIL"
    lines.append(f"{verdict} — {result['error_count']} errors, "
                 f"{result['warning_count']} warnings")
    return "\n".join(lines)
