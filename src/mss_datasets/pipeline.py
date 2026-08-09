"""Pipeline orchestration — end-to-end processing tying all components together."""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from mss_datasets.audio import read_wav, sum_stems
from mss_datasets.datasets.base import DatasetAdapter, TrackInfo
from mss_datasets.datasets.musdb18hq import Musdb18hqAdapter
from mss_datasets.datasets.medleydb import MedleydbAdapter
from mss_datasets.datasets.moisesdb_adapter import MoisesdbAdapter
from mss_datasets.mapping.profiles import PROFILES, StemProfile
from mss_datasets.metadata import (
    ErrorEntry,
    ManifestEntry,
    LICENSE_MAP,
    load_manifest,
    write_config,
    write_errors,
    write_manifest,
    write_overlap_registry,
)
from mss_datasets.overlap import OverlapGroup, resolve_cross_dataset
from mss_datasets.splits import assign_splits, load_splits, write_splits
from mss_datasets.utils import resolve_collision, sanitize_filename

logger = logging.getLogger(__name__)


def _process_track_worker(
    dataset_name: str,
    track: TrackInfo,
    profile_name: str,
    output_dir: str,
    group_by_dataset: bool,
    adapter_path: str,
    include_mixtures: bool = False,
) -> dict:
    """Standalone worker function for parallel processing (must be picklable)."""
    try:
        profile = PROFILES[profile_name]
        output = Path(output_dir)

        if dataset_name == "musdb18hq":
            adapter = Musdb18hqAdapter(adapter_path)
        elif dataset_name == "medleydb":
            adapter = MedleydbAdapter(adapter_path)
        else:
            return {"error": f"Unsupported dataset for parallel: {dataset_name}"}

        return adapter.process_track(
            track, profile, output,
            group_by_dataset=group_by_dataset,
            include_mixtures=include_mixtures,
        )
    except Exception as e:
        return {"error": str(e)}


@dataclass
class PipelineConfig:
    musdb18hq_path: str | None = None
    moisesdb_path: str | None = None
    medleydb_path: str | None = None
    output: str = "./output"
    profile: str = "vdbo"
    workers: int = 1
    include_mixtures: bool = False
    group_by_dataset: bool = False
    split_output: bool = False
    include_bleed: bool = False
    verify_mixtures: bool = False
    dry_run: bool = False
    verbose: bool = False


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.profile: StemProfile = PROFILES[config.profile]
        self.output_dir = Path(config.output)
        self.errors: list[ErrorEntry] = []
        self.manifest_entries: list[ManifestEntry] = []
        self.skipped_musdb: list[str] = []
        self.overlap_groups: list[OverlapGroup] = []
        self.previous_manifest: dict = {}
        self._previous_by_track: dict[tuple[str, str], list[dict]] = {}

    def run(self) -> dict:
        """Execute the full pipeline. Returns summary dict."""
        # Stage 1: Acquire — validate paths, instantiate adapters
        adapters = self._stage_acquire()
        if not adapters:
            logger.error("No valid datasets found")
            return {"error": "No valid datasets found"}

        # Stage 2: Deduplicate — compute overlap skip lists
        all_tracks, musdb_splits_map = self._stage_deduplicate(adapters)

        # Assign filenames first: the collision-resolved base is the track's
        # identity, and both the split lock and the resume ledger key off it.
        self._assign_filename_bases(all_tracks)

        # The previous manifest is the resume ledger — see _track_already_processed.
        self.previous_manifest = load_manifest(
            self.output_dir / "metadata" / "manifest.json"
        )
        # Keyed on the filename base — unique per track and independent of the
        # split, so a re-split is detected rather than hidden. Entries written
        # before filename_base existed fall back to the name, which can collide
        # across splits, hence the list.
        self._previous_by_track = defaultdict(list)
        for entry in self.previous_manifest.values():
            key = entry.get("filename_base") or (
                entry.get("source_dataset", ""),
                entry.get("original_track_name", ""),
            )
            self._previous_by_track[key].append(entry)

        # Stage 3: Assign splits
        existing_splits = load_splits(self.output_dir / "metadata" / "splits.json")
        assign_splits(all_tracks, existing_splits=existing_splits, musdb_splits=musdb_splits_map)

        # Remap "test" → "val" when split_output is enabled
        if self.config.split_output:
            for track in all_tracks:
                if track.split == "test":
                    track.split = "val"

        if self.config.dry_run:
            return self._dry_run_report(all_tracks)

        # Clean up leftover .tmp files
        self._cleanup_tmp_files()

        # Relocate anything left in a split it no longer belongs to, before the
        # resume check decides what still needs writing.
        self._reconcile_splits(all_tracks)

        # Stage 4: Process — stem map + normalize + write
        self._stage_process(adapters, all_tracks)

        # Stage 5: Validate
        self._stage_validate()

        # Stage 6: Write metadata
        self._stage_metadata(all_tracks)

        return self._summary_report(all_tracks)

    def _stage_acquire(self) -> dict[str, DatasetAdapter]:
        """Validate paths and instantiate adapters."""
        adapters = {}

        if self.config.musdb18hq_path:
            try:
                adapter = Musdb18hqAdapter(self.config.musdb18hq_path)
                adapter.validate_path()
                adapters["musdb18hq"] = adapter
            except ValueError as e:
                logger.error("MUSDB18-HQ validation failed: %s", e)
                self.errors.append(ErrorEntry(
                    track="", dataset="musdb18hq",
                    error=str(e), stage="acquire",
                ))

        if self.config.medleydb_path:
            try:
                adapter = MedleydbAdapter(self.config.medleydb_path)
                adapter.validate_path()
                adapters["medleydb"] = adapter
            except ValueError as e:
                logger.error("MedleyDB validation failed: %s", e)
                self.errors.append(ErrorEntry(
                    track="", dataset="medleydb",
                    error=str(e), stage="acquire",
                ))

        if self.config.moisesdb_path:
            try:
                adapter = MoisesdbAdapter(self.config.moisesdb_path)
                adapter.validate_path()
                adapters["moisesdb"] = adapter
            except ValueError as e:
                logger.error("MoisesDB validation failed: %s", e)
                self.errors.append(ErrorEntry(
                    track="", dataset="moisesdb",
                    error=str(e), stage="acquire",
                ))

        return adapters

    def _stage_deduplicate(
        self, adapters: dict[str, DatasetAdapter]
    ) -> tuple[list[TrackInfo], dict[str, str]]:
        """Discover tracks from all adapters, then resolve duplicates across them."""
        # Discover everything first — cross-dataset resolution needs the full
        # picture, so it cannot run before the later datasets are known.
        discovered: list[TrackInfo] = []
        for name in ("musdb18hq", "medleydb", "moisesdb"):
            if name not in adapters:
                continue
            if name == "moisesdb":
                logger.info("Discovering MoisesDB tracks (this may take a moment)...")
            else:
                logger.info("Discovering %s tracks...", name)
            tracks = adapters[name].discover_tracks()
            logger.info("Found %d %s tracks", len(tracks), name)
            discovered.extend(tracks)

        # A bleed-flagged winner would take the slot and then be dropped by the
        # filter below, losing the song — so prefer a usable copy unless bleed
        # tracks are being kept anyway.
        self.overlap_groups = resolve_cross_dataset(
            discovered, prefer_usable=not self.config.include_bleed
        )

        skip = {loser for g in self.overlap_groups for loser in g.losers}
        self.skipped_musdb = sorted(
            name for dataset, name in skip if dataset == "musdb18hq"
        )

        # MUSDB18-HQ's train/test assignment is canonical for benchmarking, so a
        # winner that displaced a MUSDB track inherits that track's split.
        by_ref = {(t.source_dataset, t.original_track_name): t for t in discovered}
        musdb_splits_map: dict[str, str] = {}
        for group in self.overlap_groups:
            musdb_loser = next(
                (ref for ref in group.losers if ref[0] == "musdb18hq"), None
            )
            if musdb_loser is not None:
                musdb_splits_map[group.canonical] = by_ref[musdb_loser].split

        if self.overlap_groups:
            by_dataset: dict[str, int] = defaultdict(int)
            for dataset, _ in skip:
                by_dataset[dataset] += 1
            counts = ", ".join(f"{ds}: {n}" for ds, n in sorted(by_dataset.items()))
            logger.info(
                "Deduplicated %d tracks across datasets (%s)", len(skip), counts
            )

        all_tracks = [
            t for t in discovered
            if (t.source_dataset, t.original_track_name) not in skip
        ]

        # Filter tracks with stem bleed
        if not self.config.include_bleed:
            bleed_tracks = [t for t in all_tracks if t.has_bleed]
            self.excluded_bleed = len(bleed_tracks)
            if bleed_tracks:
                by_dataset = defaultdict(int)
                for t in bleed_tracks:
                    by_dataset[t.source_dataset] += 1
                counts = ", ".join(f"{ds}: {n}" for ds, n in sorted(by_dataset.items()))
                logger.info("Excluding %d tracks with bleed (%s)", len(bleed_tracks), counts)
            all_tracks = [t for t in all_tracks if not t.has_bleed]
        else:
            self.excluded_bleed = 0

        return all_tracks, musdb_splits_map

    def _stage_process(
        self, adapters: dict[str, DatasetAdapter], all_tracks: list[TrackInfo]
    ) -> None:
        """Process each track: stem map + normalize + write.

        Uses ProcessPoolExecutor when workers > 1. Falls back to sequential
        processing for workers=1 or when MoisesDB tracks are present (the
        moisesdb library objects aren't picklable).
        """
        # Separate tracks by dataset for adapter lookup
        tracks_to_process = []
        for track in all_tracks:
            if track.source_dataset not in adapters:
                continue
            if self._track_already_processed(track):
                logger.debug("Skipping already-processed track: %s", track.track_name)
                # Carry the previous entry forward, otherwise the manifest would
                # describe only this invocation instead of the whole dataset.
                entry = self._previous_entry(track)
                if entry is not None:
                    self.manifest_entries.append(ManifestEntry.from_dict(entry))
                continue
            tracks_to_process.append(track)

        if not tracks_to_process:
            logger.info("All tracks already processed, nothing to do")
            return

        skipped = len(all_tracks) - len(tracks_to_process)
        if skipped:
            logger.info("Skipping %d already-processed tracks", skipped)
        logger.info("Processing %d tracks...", len(tracks_to_process))

        # MoisesDB tracks can't be parallelized (moisesdb lib not picklable)
        # Split into parallelizable and sequential groups
        moisesdb_tracks = [t for t in tracks_to_process if t.source_dataset == "moisesdb"]
        other_tracks = [t for t in tracks_to_process if t.source_dataset != "moisesdb"]

        # Process parallelizable tracks
        if other_tracks and self.config.workers > 1:
            self._process_parallel(adapters, other_tracks)
        else:
            self._process_sequential(adapters, other_tracks)

        # MoisesDB always sequential (library state isn't fork-safe)
        self._process_sequential(adapters, moisesdb_tracks)

        # Verification summary
        if self.config.verify_mixtures:
            verify_errors = [e for e in self.errors if e.stage == "verify_mixtures"]
            verified = len(tracks_to_process) - len(verify_errors)
            if verify_errors:
                logger.warning(
                    "Mixture verification: %d passed, %d failed",
                    verified, len(verify_errors),
                )
            else:
                logger.info("Mixture verification: all %d tracks passed", verified)

    def _process_sequential(
        self, adapters: dict[str, DatasetAdapter], tracks: list[TrackInfo]
    ) -> None:
        """Process tracks sequentially."""
        if not tracks:
            return
        dataset_label = tracks[0].source_dataset if len(set(t.source_dataset for t in tracks)) == 1 else "tracks"
        for track in tqdm(tracks, desc=f"Processing {dataset_label}", unit="track"):
            self._process_single_track(adapters[track.source_dataset], track)

    def _process_parallel(
        self, adapters: dict[str, DatasetAdapter], tracks: list[TrackInfo]
    ) -> None:
        """Process tracks in parallel using ProcessPoolExecutor."""
        # Build serializable work items (avoid pickling adapters)
        work_items = []
        for track in tracks:
            work_items.append((
                track.source_dataset,
                track,
                self.profile.name,
                str(self._effective_output_dir(track)),
                self.config.group_by_dataset,
                str(adapters[track.source_dataset].path),
                self.config.include_mixtures,
            ))

        dataset_label = tracks[0].source_dataset if len(set(t.source_dataset for t in tracks)) == 1 else "tracks"
        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(
                    _process_track_worker,
                    dataset_name, track, profile_name, output_dir,
                    group_by_dataset, adapter_path, include_mixtures,
                ): track
                for dataset_name, track, profile_name, output_dir,
                    group_by_dataset, adapter_path, include_mixtures in work_items
            }
            with tqdm(total=len(futures), desc=f"Processing {dataset_label}", unit="track") as pbar:
                for future in as_completed(futures):
                    track = futures[future]
                    try:
                        result = future.result()
                        if result.get("error"):
                            self.errors.append(ErrorEntry(
                                track=track.track_name,
                                dataset=track.source_dataset,
                                error=result["error"],
                                stage="process",
                            ))
                        else:
                            self.manifest_entries.append(ManifestEntry(
                                source_dataset=result["source_dataset"],
                                original_track_name=result["original_track_name"],
                                filename_base=result.get("filename_base", ""),
                                artist=result["artist"],
                                title=result["title"],
                                split=result["split"],
                                available_stems=result["available_stems"],
                                profile=result["profile"],
                                license=LICENSE_MAP.get(result["source_dataset"], ""),
                                has_bleed=result.get("has_bleed", False),
                                musdb18hq_4stem_only=result.get("musdb18hq_4stem_only", False),
                                flags=result.get("flags", []),
                            ))
                            if self.config.verify_mixtures:
                                self.errors.extend(self._verify_track_mixture(result))
                    except Exception as e:
                        logger.error("Worker error for %s: %s", track.track_name, e)
                        self.errors.append(ErrorEntry(
                            track=track.track_name,
                            dataset=track.source_dataset,
                            error=str(e),
                            stage="process",
                        ))
                    pbar.update(1)

    def _process_single_track(self, adapter: DatasetAdapter, track: TrackInfo) -> None:
        """Process a single track and accumulate results."""
        try:
            result = adapter.process_track(
                track, self.profile, self._effective_output_dir(track),
                group_by_dataset=self.config.group_by_dataset,
                include_mixtures=self.config.include_mixtures,
            )
            self.manifest_entries.append(ManifestEntry(
                source_dataset=result["source_dataset"],
                original_track_name=result["original_track_name"],
                filename_base=result.get("filename_base", ""),
                artist=result["artist"],
                title=result["title"],
                split=result["split"],
                available_stems=result["available_stems"],
                profile=result["profile"],
                license=LICENSE_MAP.get(result["source_dataset"], ""),
                has_bleed=result.get("has_bleed", False),
                musdb18hq_4stem_only=result.get("musdb18hq_4stem_only", False),
                flags=result.get("flags", []),
            ))
            if self.config.verify_mixtures:
                self.errors.extend(self._verify_track_mixture(result))
        except Exception as e:
            logger.error("Error processing %s: %s", track.track_name, e)
            self.errors.append(ErrorEntry(
                track=track.track_name,
                dataset=track.source_dataset,
                error=str(e),
                stage="process",
            ))

    def _verify_track_mixture(self, result: dict) -> list[ErrorEntry]:
        """Read back written stems + mixture and verify stem sum matches mixture."""
        mixture_path = result.get("mixture_path")
        written_paths = result.get("written_paths", {})
        if not mixture_path or not written_paths:
            return []

        try:
            mixture, _ = read_wav(mixture_path)
            stem_arrays = []
            for stem_name, stem_path in written_paths.items():
                data, _ = read_wav(stem_path)
                stem_arrays.append(data)

            if not stem_arrays:
                return []

            stem_sum = sum_stems(stem_arrays)

            # Truncate to same length (mixture may differ slightly)
            min_len = min(mixture.shape[0], stem_sum.shape[0])
            mixture = mixture[:min_len]
            stem_sum = stem_sum[:min_len]

            if not np.allclose(mixture, stem_sum, atol=1e-3):
                max_diff = float(np.max(np.abs(mixture - stem_sum)))
                return [ErrorEntry(
                    track=result.get("original_track_name", ""),
                    dataset=result.get("source_dataset", ""),
                    error=f"Mixture verification failed: max abs diff = {max_diff:.8f}",
                    stage="verify_mixtures",
                    skipped=False,
                )]
        except Exception as e:
            return [ErrorEntry(
                track=result.get("original_track_name", ""),
                dataset=result.get("source_dataset", ""),
                error=f"Mixture verification error: {e}",
                stage="verify_mixtures",
                skipped=False,
            )]

        return []

    def _effective_output_dir(self, track: TrackInfo) -> Path:
        """Return output dir, nested by split when --split-output is enabled."""
        if self.config.split_output:
            return self.output_dir / track.split
        return self.output_dir

    def _output_path(
        self, track: TrackInfo, stem: str, split: str | None = None
    ) -> Path:
        """Where a given stem of a track is written.

        `split` overrides the track's own, for locating a copy left behind in a
        directory the track no longer belongs to.
        """
        filename_base = track.filename_base or sanitize_filename(
            track.source_dataset, track.artist, track.title
        )
        base_dir = self.output_dir
        if self.config.split_output:
            base_dir = base_dir / (split or track.split)
        stem_dir = base_dir / stem
        if self.config.group_by_dataset:
            stem_dir = stem_dir / track.source_dataset
        return stem_dir / f"{filename_base}.wav"

    def _all_stem_names(self) -> list[str]:
        """Stem folders this run writes, including the mixture when enabled."""
        stems = list(self.profile.stems)
        if self.config.include_mixtures:
            stems.append("mixture")
        return stems

    def _assign_filename_bases(self, tracks: list[TrackInfo]) -> None:
        """Give every track its output filename, disambiguating any collision.

        Names derive only from dataset metadata, so they are stable across runs.
        Two tracks can still collide after sanitization — 80-char truncation, or
        names differing only in punctuation — so resolve deterministically by
        sorting on the original name rather than on discovery order.
        """
        seen: set[str] = set()
        for track in sorted(
            tracks, key=lambda t: (t.source_dataset, t.original_track_name)
        ):
            base = sanitize_filename(
                track.source_dataset, track.artist, track.title
            )
            resolved = resolve_collision(base, seen)
            if resolved != base:
                logger.warning(
                    "Filename collision on %r — writing %s as %r",
                    base, track.track_name, resolved,
                )
            seen.add(resolved)
            track.filename_base = resolved

    def _reconcile_splits(self, tracks: list[TrackInfo]) -> None:
        """Move any track sitting in a split directory it no longer belongs to.

        The filename no longer encodes the split, but the split still selects the
        directory. Without this a re-split writes a fresh copy and leaves the old
        one behind, putting the same audio in both train/ and val/.
        """
        if not self.config.split_output:
            return

        moved = 0
        for track in tracks:
            for stem in self._all_stem_names():
                correct = self._output_path(track, stem)
                if correct.exists():
                    continue
                for other in ("train", "val", "test"):
                    if other == track.split:
                        continue
                    stale = self._output_path(track, stem, split=other)
                    if stale.exists():
                        correct.parent.mkdir(parents=True, exist_ok=True)
                        stale.replace(correct)
                        moved += 1
                        break

        if moved:
            logger.info("Moved %d file(s) into their reassigned split", moved)

    def _previous_entry(self, track: TrackInfo) -> dict | None:
        """The previous run's manifest entry for this track, if any.

        Prefers an entry recorded under the track's current split; falls back to
        any entry for the name, so a track whose split changed still resolves —
        its files then fail the existence check and it is reprocessed.
        """
        entries = self._previous_by_track.get(track.filename_base) or (
            self._previous_by_track.get(
                (track.source_dataset, track.original_track_name)
            )
        )
        if not entries:
            return None
        for entry in entries:
            if entry.get("split") == track.split:
                return entry
        return entries[0]

    def _track_already_processed(self, track: TrackInfo) -> bool:
        """True only when every file the last run wrote for this track is present.

        The previous manifest is the ledger. It records which stems were
        actually written, so a track that legitimately has no vocals counts as
        complete at three stems, while one interrupted between stems does not —
        the old "any one stem exists" heuristic marked that permanently done.

        A track whose split changed will not be found at its new path, so it is
        reprocessed rather than silently left behind in the old split.
        """
        entry = self._previous_entry(track)
        if entry is None:
            return False

        stems = list(entry.get("available_stems") or [])
        if not stems:
            return False
        if self.config.include_mixtures:
            stems.append("mixture")

        return all(self._output_path(track, stem).exists() for stem in stems)

    def _stage_validate(self) -> None:
        """Post-write validation: check output files are valid WAVs."""
        if self.config.split_output:
            base_dirs = [self.output_dir / s for s in ("train", "val")]
        else:
            base_dirs = [self.output_dir]

        stems_to_check = list(self.profile.stems)
        if self.config.include_mixtures:
            stems_to_check.append("mixture")

        for base in base_dirs:
            for stem in stems_to_check:
                stem_dir = base / stem
                if not stem_dir.exists():
                    continue
                for wav_path in stem_dir.rglob("*.wav"):
                    self._validate_wav(wav_path)

    def _validate_wav(self, wav_path: Path) -> None:
        """Validate a single WAV file."""
        try:
            info = sf.info(str(wav_path))
            if info.samplerate != 44100:
                logger.warning("Unexpected sample rate %d in %s", info.samplerate, wav_path)
            if info.channels != 2:
                logger.warning("Unexpected channel count %d in %s", info.channels, wav_path)
        except Exception as e:
            logger.error("Invalid WAV %s: %s", wav_path, e)
            self.errors.append(ErrorEntry(
                track=wav_path.name, dataset="",
                error=str(e), stage="validate",
            ))

    def _stage_metadata(self, all_tracks: list[TrackInfo]) -> None:
        """Write all metadata files."""
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        write_manifest(meta_dir / "manifest.json", self.manifest_entries)
        write_splits(meta_dir / "splits.json", all_tracks)
        write_overlap_registry(
            meta_dir / "overlap_registry.json",
            self.skipped_musdb,
            groups=self.overlap_groups,
        )
        write_errors(meta_dir / "errors.json", self.errors)
        write_config(meta_dir / "config.yaml", {
            "profile": self.config.profile,
            "workers": self.config.workers,
            "output": str(self.config.output),
            "group_by_dataset": self.config.group_by_dataset,
            "split_output": self.config.split_output,
            "include_mixtures": self.config.include_mixtures,
        })

    def _cleanup_tmp_files(self) -> None:
        """Remove leftover .tmp files from interrupted previous runs."""
        if not self.output_dir.exists():
            return
        for tmp_file in self.output_dir.rglob("*.tmp"):
            logger.info("Cleaning up leftover tmp file: %s", tmp_file)
            tmp_file.unlink()

    def _dry_run_report(self, all_tracks: list[TrackInfo]) -> dict:
        """Generate dry-run report without writing any files."""
        by_dataset = {}
        by_split = {}
        for t in all_tracks:
            by_dataset[t.source_dataset] = by_dataset.get(t.source_dataset, 0) + 1
            by_split[t.split] = by_split.get(t.split, 0) + 1

        return {
            "dry_run": True,
            "profile": self.profile.name,
            "total_tracks": len(all_tracks),
            "by_dataset": by_dataset,
            "by_split": by_split,
            "skipped_musdb_overlap": len(self.skipped_musdb),
            "excluded_bleed": self.excluded_bleed,
            "stem_folders": list(self.profile.stems),
        }

    def _stem_base_dirs(self) -> list[Path]:
        """Return base directories that contain stem folders."""
        if self.config.split_output:
            return [self.output_dir / s for s in ("train", "val")]
        return [self.output_dir]

    def _summary_report(self, all_tracks: list[TrackInfo]) -> dict:
        """Generate post-processing summary."""
        stems_to_count = list(self.profile.stems)
        if self.config.include_mixtures:
            stems_to_count.append("mixture")

        stem_counts: dict[str, int] = {}
        for stem in stems_to_count:
            count = 0
            for base in self._stem_base_dirs():
                stem_dir = base / stem
                if stem_dir.exists():
                    count += len(list(stem_dir.rglob("*.wav")))
            stem_counts[stem] = count

        total_files = sum(stem_counts.values())

        # Estimate disk usage
        disk_usage = 0
        for stem in stems_to_count:
            for base in self._stem_base_dirs():
                stem_dir = base / stem
                if stem_dir.exists():
                    for f in stem_dir.rglob("*.wav"):
                        disk_usage += f.stat().st_size

        return {
            "profile": self.profile.name,
            "total_tracks": len(all_tracks),
            "stem_counts": stem_counts,
            "total_files": total_files,
            "disk_usage_bytes": disk_usage,
            "errors": len(self.errors),
            "skipped_musdb_overlap": len(self.skipped_musdb),
            "excluded_bleed": self.excluded_bleed,
        }
