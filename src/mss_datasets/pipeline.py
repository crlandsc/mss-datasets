"""Pipeline orchestration — end-to-end processing tying all components together."""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from mss_datasets.datasets.base import DatasetAdapter, TrackInfo
from mss_datasets.datasets.musdb18hq import Musdb18hqAdapter
from mss_datasets.datasets.medleydb import MedleydbAdapter
from mss_datasets.mapping.profiles import PROFILES, StemProfile
from mss_datasets.metadata import (
    ErrorEntry,
    ManifestEntry,
    LICENSE_MAP,
    write_config,
    write_errors,
    write_manifest,
    write_overlap_registry,
)
from mss_datasets.overlap import is_overlap_track, resolve_overlaps
from mss_datasets.splits import assign_splits, load_splits, write_splits
from mss_datasets.utils import canonical_name

logger = logging.getLogger(__name__)


def _process_track_worker(
    dataset_name: str,
    track: TrackInfo,
    profile_name: str,
    output_dir: str,
    group_by_dataset: bool,
    adapter_path: str,
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

        return adapter.process_track(track, profile, output, group_by_dataset=group_by_dataset)
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
    normalize_loudness: bool = False
    loudness_target: float = -14.0
    include_bleed: bool = False
    verify_mixtures: bool = False
    dry_run: bool = False
    validate: bool = False
    verbose: bool = False


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.profile: StemProfile = PROFILES[config.profile]
        self.output_dir = Path(config.output)
        self.errors: list[ErrorEntry] = []
        self.manifest_entries: list[ManifestEntry] = []
        self.skipped_musdb: list[str] = []

    def run(self) -> dict:
        """Execute the full pipeline. Returns summary dict."""
        # Stage 1: Acquire — validate paths, instantiate adapters
        adapters = self._stage_acquire()
        if not adapters:
            logger.error("No valid datasets found")
            return {"error": "No valid datasets found"}

        # Stage 2: Deduplicate — compute overlap skip lists
        all_tracks, musdb_splits_map = self._stage_deduplicate(adapters)

        # Stage 3: Assign splits
        existing_splits = load_splits(self.output_dir / "metadata" / "splits.json")
        assign_splits(all_tracks, existing_splits=existing_splits, musdb_splits=musdb_splits_map)

        if self.config.dry_run:
            return self._dry_run_report(all_tracks)

        # Clean up leftover .tmp files
        self._cleanup_tmp_files()

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
                from mss_datasets.datasets.moisesdb_adapter import MoisesdbAdapter
                adapter = MoisesdbAdapter(self.config.moisesdb_path)
                adapter.validate_path()
                adapters["moisesdb"] = adapter
            except ImportError:
                logger.error(
                    "moisesdb package not installed. "
                    "Install: pip install mss-datasets[moisesdb]"
                )
                self.errors.append(ErrorEntry(
                    track="", dataset="moisesdb",
                    error="moisesdb package not installed", stage="acquire",
                ))
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
        """Discover tracks from all adapters, resolve overlaps."""
        all_tracks: list[TrackInfo] = []
        musdb_splits_map: dict[str, str] = {}

        # Discover MUSDB18-HQ tracks
        musdb_tracks = []
        if "musdb18hq" in adapters:
            logger.info("Discovering MUSDB18-HQ tracks...")
            musdb_tracks = adapters["musdb18hq"].discover_tracks()
            logger.info("Found %d MUSDB18-HQ tracks", len(musdb_tracks))

        # Resolve overlaps
        medleydb_present = "medleydb" in adapters
        if musdb_tracks:
            overlap_result = resolve_overlaps(
                [t.original_track_name for t in musdb_tracks],
                medleydb_present=medleydb_present,
            )
            skip_set = overlap_result["skip_musdb"]
            self.skipped_musdb = sorted(skip_set)

            # Build musdb splits map for MedleyDB inheritance
            for t in musdb_tracks:
                cn = canonical_name(t.original_track_name)
                if t.original_track_name in skip_set:
                    musdb_splits_map[cn] = t.split

            # Add non-skipped MUSDB tracks
            for t in musdb_tracks:
                if t.original_track_name not in skip_set:
                    all_tracks.append(t)

        # Discover MedleyDB tracks
        if "medleydb" in adapters:
            logger.info("Discovering MedleyDB tracks...")
            medleydb_tracks = adapters["medleydb"].discover_tracks()
            logger.info("Found %d MedleyDB tracks", len(medleydb_tracks))
            all_tracks.extend(medleydb_tracks)

        # Discover MoisesDB tracks
        if "moisesdb" in adapters:
            logger.info("Discovering MoisesDB tracks (this may take a moment)...")
            moisesdb_tracks = adapters["moisesdb"].discover_tracks()
            logger.info("Found %d MoisesDB tracks", len(moisesdb_tracks))
            all_tracks.extend(moisesdb_tracks)

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
                str(self.output_dir),
                self.config.group_by_dataset,
                str(adapters[track.source_dataset].path),
            ))

        dataset_label = tracks[0].source_dataset if len(set(t.source_dataset for t in tracks)) == 1 else "tracks"
        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(
                    _process_track_worker,
                    dataset_name, track, profile_name, output_dir,
                    group_by_dataset, adapter_path,
                ): track
                for dataset_name, track, profile_name, output_dir,
                    group_by_dataset, adapter_path in work_items
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
                track, self.profile, self.output_dir,
                group_by_dataset=self.config.group_by_dataset,
            )
            self.manifest_entries.append(ManifestEntry(
                source_dataset=result["source_dataset"],
                original_track_name=result["original_track_name"],
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
        except Exception as e:
            logger.error("Error processing %s: %s", track.track_name, e)
            self.errors.append(ErrorEntry(
                track=track.track_name,
                dataset=track.source_dataset,
                error=str(e),
                stage="process",
            ))

    def _track_already_processed(self, track: TrackInfo) -> bool:
        """Check if all expected output files for a track already exist."""
        from mss_datasets.utils import sanitize_filename
        filename_base = sanitize_filename(
            track.source_dataset, track.split, track.index, track.artist, track.title
        )
        # Check if at least one stem file exists (heuristic for resumability)
        for stem in self.profile.stems:
            if self.config.group_by_dataset:
                wav = self.output_dir / stem / track.source_dataset / f"{filename_base}.wav"
            else:
                wav = self.output_dir / stem / f"{filename_base}.wav"
            if wav.exists():
                return True
        return False

    def _stage_validate(self) -> None:
        """Post-write validation: check output files are valid WAVs."""
        for stem in self.profile.stems:
            stem_dir = self.output_dir / stem
            if not stem_dir.exists():
                continue
            for wav_path in stem_dir.rglob("*.wav"):
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
        write_overlap_registry(meta_dir / "overlap_registry.json", self.skipped_musdb)
        write_errors(meta_dir / "errors.json", self.errors)
        write_config(meta_dir / "config.yaml", {
            "profile": self.config.profile,
            "workers": self.config.workers,
            "output": str(self.config.output),
            "group_by_dataset": self.config.group_by_dataset,
            "include_mixtures": self.config.include_mixtures,
            "normalize_loudness": self.config.normalize_loudness,
            "loudness_target": self.config.loudness_target,
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

    def _summary_report(self, all_tracks: list[TrackInfo]) -> dict:
        """Generate post-processing summary."""
        stem_counts = {}
        for stem in self.profile.stems:
            stem_dir = self.output_dir / stem
            if stem_dir.exists():
                stem_counts[stem] = len(list(stem_dir.rglob("*.wav")))
            else:
                stem_counts[stem] = 0

        total_files = sum(stem_counts.values())

        # Estimate disk usage
        disk_usage = 0
        for stem in self.profile.stems:
            stem_dir = self.output_dir / stem
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
