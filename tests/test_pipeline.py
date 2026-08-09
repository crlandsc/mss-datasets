"""Tests for pipeline orchestration — end-to-end with synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mss_datasets.pipeline import Pipeline, PipelineConfig


def _make_musdb_fixture(base_path):
    """Create synthetic MUSDB18-HQ."""
    sr = 44100
    rng = np.random.default_rng(42)
    for split, tracks in [("train", ["ArtistA - Song1", "ArtistB - Song2"]),
                          ("test", ["ArtistC - Song3"])]:
        for track_name in tracks:
            d = base_path / split / track_name
            d.mkdir(parents=True)
            for stem in ("vocals", "drums", "bass", "other", "mixture"):
                data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
                sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")


def _make_medleydb_fixture(base_path):
    """Create synthetic MedleyDB."""
    sr = 44100
    rng = np.random.default_rng(43)
    tracks = [
        ("MedArtist_TrackOne", {"S01": "female singer", "S02": "acoustic guitar"}),
        ("MedArtist_TrackTwo", {"S01": "male singer", "S02": "drum set", "S03": "electric bass"}),
    ]
    for track_name, stems in tracks:
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        metadata = {
            "artist": track_name.split("_")[0],
            "title": "_".join(track_name.split("_")[1:]),
            "has_bleed": "no",
            "stems": {},
        }
        for stem_key, instrument in stems.items():
            idx = stem_key.replace("S", "")
            metadata["stems"][stem_key] = {"instrument": instrument}
            data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
            sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)


@pytest.fixture
def full_fixture(tmp_path):
    """Create MUSDB18-HQ + MedleyDB fixtures."""
    musdb_path = tmp_path / "musdb18hq"
    medleydb_path = tmp_path / "medleydb"
    _make_musdb_fixture(musdb_path)
    _make_medleydb_fixture(medleydb_path)
    return {"musdb": musdb_path, "medleydb": medleydb_path, "output": tmp_path / "output"}


class TestPipelineMusdbOnly:
    def test_runs_successfully(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        pipeline = Pipeline(config)
        result = pipeline.run()

        assert result["total_tracks"] == 3
        assert result["errors"] == 0
        assert result["stem_counts"]["vocals"] == 3
        assert result["stem_counts"]["drums"] == 3

    def test_metadata_files_written(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        meta = output / "metadata"
        assert (meta / "manifest.json").exists()
        assert (meta / "splits.json").exists()
        assert (meta / "overlap_registry.json").exists()
        assert (meta / "errors.json").exists()
        assert (meta / "config.yaml").exists()

    def test_overlap_registry_empty_without_medleydb(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        with open(output / "metadata" / "overlap_registry.json") as f:
            data = json.load(f)
        assert data["skipped_count"] == 0


class TestPipelineMedleydbOnly:
    def test_runs_successfully(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        output = tmp_path / "output"

        config = PipelineConfig(medleydb_path=str(medleydb_path), output=str(output))
        pipeline = Pipeline(config)
        result = pipeline.run()

        assert result["total_tracks"] == 2
        assert result["errors"] == 0
        assert result["stem_counts"]["vocals"] == 2


class TestPipelineBothDatasets:
    def test_processes_both(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
        )
        pipeline = Pipeline(config)
        result = pipeline.run()

        # 3 MUSDB + 2 MedleyDB (no overlap in our fixtures)
        assert result["total_tracks"] == 5
        assert result["errors"] == 0


class TestDryRun:
    def test_no_files_written(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["dry_run"] is True
        assert result["total_tracks"] == 3
        assert not output.exists()  # No output written

    def test_dry_run_shows_stem_folders(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()
        assert "vocals" in result["stem_folders"]


class TestResumability:
    def test_skips_already_processed(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))

        # First run
        Pipeline(config).run()
        wav_count_1 = len(list(output.rglob("*.wav")))

        # Second run (should skip all existing)
        pipeline2 = Pipeline(config)
        result2 = pipeline2.run()

        wav_count_2 = len(list(output.rglob("*.wav")))
        assert wav_count_2 == wav_count_1  # No new files

    def test_cleans_up_tmp_files(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        # Create a leftover .tmp file
        (output / "vocals").mkdir(parents=True)
        (output / "vocals" / "leftover.wav.tmp").touch()

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        assert not (output / "vocals" / "leftover.wav.tmp").exists()


class TestResumeLedger:
    """The previous manifest is the resume ledger.

    The tests above only compare file *counts*, which pass even when everything
    was reprocessed and overwritten in place. These compare mtimes and manifest
    contents, so they detect the two bugs this replaced: a manifest truncated to
    only the current invocation, and a track interrupted between stems being
    marked permanently done.
    """

    @pytest.fixture
    def env(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"
        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
        )
        return config, output

    @staticmethod
    def _mtimes(output):
        return {p: p.stat().st_mtime_ns for p in sorted(output.rglob("*.wav"))}

    @staticmethod
    def _manifest(output):
        return json.loads((output / "metadata" / "manifest.json").read_text())

    def test_second_run_rewrites_nothing(self, env):
        config, output = env
        Pipeline(config).run()
        before = self._mtimes(output)

        Pipeline(config).run()
        after = self._mtimes(output)

        assert before and before == after

    def test_manifest_survives_a_resumed_run(self, env):
        config, output = env
        Pipeline(config).run()
        first = self._manifest(output)

        Pipeline(config).run()
        # Previously this dropped to zero — nothing was processed, so nothing
        # was recorded, and the manifest was overwritten with an empty dict.
        assert self._manifest(output) == first

    def test_track_interrupted_between_stems_is_repaired(self, env):
        config, output = env
        Pipeline(config).run()
        before = self._mtimes(output)

        victim = sorted(output.rglob("*.wav"))[0]
        victim.unlink()
        Pipeline(config).run()

        assert victim.exists()
        assert len(self._manifest(output)) == len(before) // 5

    def test_repair_only_touches_the_affected_track(self, env):
        config, output = env
        Pipeline(config).run()
        before = self._mtimes(output)

        victim = sorted(output.rglob("*.wav"))[0]
        victim.unlink()
        Pipeline(config).run()
        after = self._mtimes(output)

        rewritten = {p.name for p in before if before[p] != after.get(p)}
        # One track's five files (four stems + mixture), nothing else.
        assert rewritten == {victim.name}

    def test_missing_manifest_means_reprocess(self, env):
        config, output = env
        Pipeline(config).run()
        before = self._mtimes(output)

        (output / "metadata" / "manifest.json").unlink()
        Pipeline(config).run()
        after = self._mtimes(output)

        assert all(before[p] != after[p] for p in before)

    def test_same_name_in_two_splits_is_not_conflated(self, tmp_path):
        """Two tracks sharing a name across splits must keep separate entries."""
        musdb_path = tmp_path / "musdb18hq"
        sr = 44100
        rng = np.random.default_rng(7)
        for split in ("train", "test"):
            d = musdb_path / split / "Same Artist - Same Song"
            d.mkdir(parents=True)
            for stem in ("vocals", "drums", "bass", "other", "mixture"):
                sf.write(
                    str(d / f"{stem}.wav"),
                    rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32),
                    sr, subtype="FLOAT",
                )
        output = tmp_path / "output"
        config = PipelineConfig(
            musdb18hq_path=str(musdb_path), output=str(output), include_mixtures=True
        )

        Pipeline(config).run()
        assert len(self._manifest(output)) == 2

        before = self._mtimes(output)
        Pipeline(config).run()
        assert self._mtimes(output) == before
        assert len(self._manifest(output)) == 2


class TestGroupByDataset:
    def test_creates_dataset_subdirs(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
            group_by_dataset=True,
        )
        Pipeline(config).run()

        vocals_dir = full_fixture["output"] / "vocals"
        assert (vocals_dir / "musdb18hq").is_dir()
        assert (vocals_dir / "medleydb").is_dir()


class TestVDBOGP:
    def test_six_stem_profile(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(output),
            profile="vdbo+gp",
        )
        result = Pipeline(config).run()

        # TrackOne has acoustic guitar → guitar stem
        assert result["stem_counts"].get("guitar", 0) >= 1


class TestBleedFiltering:
    def _make_medleydb_with_bleed(self, base_path, has_bleed="yes"):
        """Create MedleyDB fixture with bleed tracks."""
        sr = 44100
        rng = np.random.default_rng(44)
        track_name = "BleedArtist_BleedTrack"
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        metadata = {
            "artist": "BleedArtist",
            "title": "BleedTrack",
            "has_bleed": has_bleed,
            "stems": {"S01": {"instrument": "male singer"}},
        }
        data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
        sf.write(str(stems_dir / f"{track_name}_STEM_01.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)

    def test_bleed_excluded_by_default(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)  # 2 no-bleed tracks
        self._make_medleydb_with_bleed(medleydb_path, has_bleed="yes")  # 1 bleed track

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["total_tracks"] == 2
        assert result["excluded_bleed"] == 1

    def test_include_bleed_overrides(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        self._make_medleydb_with_bleed(medleydb_path, has_bleed="yes")

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
            include_bleed=True,
        )
        result = Pipeline(config).run()

        assert result["total_tracks"] == 3
        assert result["excluded_bleed"] == 0

    def test_no_bleed_tracks_zero_excluded(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)  # no bleed tracks

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(tmp_path / "output"),
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert result["excluded_bleed"] == 0


class TestMedleydbOverrides:
    def _make_medleydb_named(self, base_path, track_name, stems):
        """Create MedleyDB fixture with specific track name and stems."""
        sr = 44100
        rng = np.random.default_rng(45)
        track_dir = base_path / "Audio" / track_name
        stems_dir = track_dir / f"{track_name}_STEMS"
        stems_dir.mkdir(parents=True)

        metadata = {
            "artist": track_name.split("_")[0],
            "title": "_".join(track_name.split("_")[1:]),
            "has_bleed": "no",
            "stems": {},
        }
        for stem_key, instrument in stems.items():
            idx = stem_key.replace("S", "")
            metadata["stems"][stem_key] = {"instrument": instrument}
            data = rng.uniform(-0.3, 0.3, (sr, 2)).astype(np.float32)
            sf.write(str(stems_dir / f"{track_name}_STEM_{idx}.wav"), data, sr, subtype="FLOAT")

        with open(track_dir / f"{track_name}_METADATA.yaml", "w") as f:
            yaml.dump(metadata, f)

    @pytest.fixture
    def _patch_overrides(self):
        """Yield a helper that patches load_medleydb_overrides."""
        from unittest.mock import patch
        def _patch(overrides):
            return patch(
                "mss_datasets.datasets.medleydb.load_medleydb_overrides",
                return_value=overrides,
            )
        return _patch

    def test_excluded_track_not_discovered(self, tmp_path, _patch_overrides):
        overrides = {"exclude_tracks": {"ExcludeMe_Track"}, "exclude_stems": {}, "reroute_stems": {}}
        medleydb_path = tmp_path / "medleydb"
        self._make_medleydb_named(medleydb_path, "ExcludeMe_Track", {"S01": "male singer"})
        self._make_medleydb_named(medleydb_path, "KeepMe_Track", {"S01": "female singer"})

        from mss_datasets.datasets.medleydb import MedleydbAdapter
        with _patch_overrides(overrides):
            adapter = MedleydbAdapter(medleydb_path)
            tracks = adapter.discover_tracks()

        names = [t.original_track_name for t in tracks]
        assert "ExcludeMe_Track" not in names
        assert "KeepMe_Track" in names
        assert len(tracks) == 1

    def test_excluded_stem_not_in_output(self, tmp_path, _patch_overrides):
        overrides = {"exclude_tracks": set(), "exclude_stems": {"TestArtist_TestTrack": {"S02"}}, "reroute_stems": {}}
        medleydb_path = tmp_path / "medleydb"
        self._make_medleydb_named(
            medleydb_path, "TestArtist_TestTrack",
            {"S01": "male singer", "S02": "acoustic guitar"},
        )

        from mss_datasets.datasets.medleydb import MedleydbAdapter
        from mss_datasets.mapping.profiles import PROFILES

        with _patch_overrides(overrides):
            adapter = MedleydbAdapter(medleydb_path)
            tracks = adapter.discover_tracks()
            output = tmp_path / "output"
            result = adapter.process_track(tracks[0], PROFILES["vdbo"], output)

        # S01 (vocals) written, S02 (guitar→other) excluded
        assert "vocals" in result["available_stems"]
        assert "other" not in result["available_stems"]

    def test_no_overrides_processes_normally(self, tmp_path, _patch_overrides):
        overrides = {"exclude_tracks": set(), "exclude_stems": {}, "reroute_stems": {}}
        medleydb_path = tmp_path / "medleydb"
        self._make_medleydb_named(
            medleydb_path, "TestArtist_TestTrack",
            {"S01": "male singer", "S02": "acoustic guitar"},
        )

        from mss_datasets.datasets.medleydb import MedleydbAdapter
        from mss_datasets.mapping.profiles import PROFILES

        with _patch_overrides(overrides):
            adapter = MedleydbAdapter(medleydb_path)
            tracks = adapter.discover_tracks()
            output = tmp_path / "output"
            result = adapter.process_track(tracks[0], PROFILES["vdbo"], output)

        assert "vocals" in result["available_stems"]
        assert "other" in result["available_stems"]

    def test_all_stems_excluded_produces_empty(self, tmp_path, _patch_overrides):
        overrides = {"exclude_tracks": set(), "exclude_stems": {"TestArtist_TestTrack": {"S01", "S02"}}, "reroute_stems": {}}
        medleydb_path = tmp_path / "medleydb"
        self._make_medleydb_named(
            medleydb_path, "TestArtist_TestTrack",
            {"S01": "male singer", "S02": "acoustic guitar"},
        )

        from mss_datasets.datasets.medleydb import MedleydbAdapter
        from mss_datasets.mapping.profiles import PROFILES

        with _patch_overrides(overrides):
            adapter = MedleydbAdapter(medleydb_path)
            tracks = adapter.discover_tracks()
            output = tmp_path / "output"
            result = adapter.process_track(tracks[0], PROFILES["vdbo"], output)

        assert result["available_stems"] == []

    def test_rerouted_stem_goes_to_new_target(self, tmp_path, _patch_overrides):
        overrides = {
            "exclude_tracks": set(),
            "exclude_stems": {},
            "reroute_stems": {"TestArtist_TestTrack": {"S02": "bass"}},
        }
        medleydb_path = tmp_path / "medleydb"
        # S02 is "synthesizer" which normally maps to "other" in vdbo
        self._make_medleydb_named(
            medleydb_path, "TestArtist_TestTrack",
            {"S01": "male singer", "S02": "synthesizer"},
        )

        from mss_datasets.datasets.medleydb import MedleydbAdapter
        from mss_datasets.mapping.profiles import PROFILES

        with _patch_overrides(overrides):
            adapter = MedleydbAdapter(medleydb_path)
            tracks = adapter.discover_tracks()
            output = tmp_path / "output"
            result = adapter.process_track(tracks[0], PROFILES["vdbo"], output)

        # S02 should be rerouted to bass, not other
        assert "vocals" in result["available_stems"]
        assert "bass" in result["available_stems"]
        assert "other" not in result["available_stems"]
        assert (output / "bass").is_dir()


class TestIncludeMixtures:
    def test_mixture_folder_created(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
        )
        result = Pipeline(config).run()

        assert (output / "mixture").is_dir()
        assert result["stem_counts"]["mixture"] == 3  # 3 MUSDB tracks

    def test_mixture_count_matches_vocals(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
        )
        result = Pipeline(config).run()

        assert result["stem_counts"]["mixture"] == result["stem_counts"]["vocals"]

    def test_mixture_not_created_by_default(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(musdb18hq_path=str(musdb_path), output=str(output))
        Pipeline(config).run()

        assert not (output / "mixture").exists()

    def test_mixture_with_medleydb(self, tmp_path):
        medleydb_path = tmp_path / "medleydb"
        _make_medleydb_fixture(medleydb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            medleydb_path=str(medleydb_path),
            output=str(output),
            include_mixtures=True,
        )
        result = Pipeline(config).run()

        assert (output / "mixture").is_dir()
        assert result["stem_counts"]["mixture"] == 2  # 2 MedleyDB tracks

    def test_mixture_with_group_by_dataset(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
            include_mixtures=True,
            group_by_dataset=True,
        )
        Pipeline(config).run()

        mixture_dir = full_fixture["output"] / "mixture"
        assert (mixture_dir / "musdb18hq").is_dir()
        assert (mixture_dir / "medleydb").is_dir()

    def test_mixture_with_split_output(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
            split_output=True,
        )
        Pipeline(config).run()

        assert (output / "train" / "mixture").is_dir()
        assert (output / "val" / "mixture").is_dir()

    def test_mixture_valid_wav(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
        )
        Pipeline(config).run()

        for wav in (output / "mixture").rglob("*.wav"):
            info = sf.info(str(wav))
            assert info.samplerate == 44100
            assert info.channels == 2

    def test_mixture_in_total_files(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            include_mixtures=True,
        )
        result = Pipeline(config).run()

        # total_files should include mixture count (4 stems + mixture) * 3 tracks = 15
        assert result["total_files"] == 15


class TestSplitOutput:
    def test_creates_train_val_dirs(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )
        Pipeline(config).run()

        assert (output / "train").is_dir()
        assert (output / "val").is_dir()
        assert not (output / "test").exists()

    def test_train_files_in_train_dir(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )
        Pipeline(config).run()

        train_wavs = list((output / "train").rglob("*.wav"))
        val_wavs = list((output / "val").rglob("*.wav"))
        assert len(train_wavs) > 0
        assert len(val_wavs) > 0
        # The split is carried by the directory, never by the filename — putting
        # it in the name is what stranded stale copies when splits changed.
        for wav in train_wavs + val_wavs:
            assert "_train_" not in wav.name
            assert "_val_" not in wav.name
        # The two splits hold different tracks.
        assert not {w.name for w in train_wavs} & {w.name for w in val_wavs}

    def test_no_test_in_filenames(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )
        Pipeline(config).run()

        all_wavs = list(output.rglob("*.wav"))
        assert all("_test_" not in w.name for w in all_wavs)

    def test_stem_counts_correct(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )
        result = Pipeline(config).run()

        # 3 tracks × 4 stems = each stem should have 3 files total
        assert result["stem_counts"]["vocals"] == 3
        assert result["stem_counts"]["drums"] == 3
        assert result["total_files"] == 12

    def test_with_group_by_dataset(self, full_fixture):
        config = PipelineConfig(
            musdb18hq_path=str(full_fixture["musdb"]),
            medleydb_path=str(full_fixture["medleydb"]),
            output=str(full_fixture["output"]),
            split_output=True,
            group_by_dataset=True,
        )
        Pipeline(config).run()

        # train/vocals/musdb18hq/ should exist
        assert (full_fixture["output"] / "train" / "vocals" / "musdb18hq").is_dir()

    def test_dry_run_shows_val_not_test(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(tmp_path / "output"),
            split_output=True,
            dry_run=True,
        )
        result = Pipeline(config).run()

        assert "val" in result["by_split"]
        assert "test" not in result["by_split"]

    def test_resumability(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )

        Pipeline(config).run()
        wav_count_1 = len(list(output.rglob("*.wav")))

        # Second run should skip all
        Pipeline(config).run()
        wav_count_2 = len(list(output.rglob("*.wav")))
        assert wav_count_2 == wav_count_1

    def test_metadata_in_root(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"

        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
        )
        Pipeline(config).run()

        # Metadata should be in output/metadata/, not inside split dirs
        assert (output / "metadata" / "manifest.json").exists()
        assert not (output / "train" / "metadata").exists()
        assert not (output / "val" / "metadata").exists()


class TestSplitReconcile:
    """A track that changes split must move, not duplicate.

    The filename no longer encodes the split, but the split still picks the
    directory — so without reconciliation a re-split writes a fresh copy and
    strands the old one, putting the same audio in train/ and val/.
    """

    @pytest.fixture
    def built(self, tmp_path):
        musdb_path = tmp_path / "musdb18hq"
        _make_musdb_fixture(musdb_path)
        output = tmp_path / "output"
        config = PipelineConfig(
            musdb18hq_path=str(musdb_path),
            output=str(output),
            split_output=True,
            include_mixtures=True,
        )
        Pipeline(config).run()
        return config, output

    @staticmethod
    def _flip_one_split(output):
        """Rewrite splits.json so one train track is locked to val."""
        path = output / "metadata" / "splits.json"
        splits = json.loads(path.read_text())
        key = next(k for k, v in splits.items() if v == "train")
        splits[key] = "val"
        path.write_text(json.dumps(splits))
        return key

    def test_reassigned_track_exists_exactly_once(self, built):
        config, output = built
        self._flip_one_split(output)

        Pipeline(config).run()

        # A filename recurs once per stem folder, which is fine. What must never
        # happen is the same (stem, filename) appearing under two splits.
        splits_seen: dict[tuple[str, str], set[str]] = {}
        for wav in output.rglob("*.wav"):
            split = wav.relative_to(output).parts[0]
            splits_seen.setdefault((wav.parent.name, wav.name), set()).add(split)

        duplicated = {k: sorted(v) for k, v in splits_seen.items() if len(v) > 1}
        assert not duplicated, f"track left in more than one split: {duplicated}"

    def test_reassigned_track_moved_to_its_new_split(self, built):
        config, output = built
        before_train = {p.name for p in (output / "train").rglob("*.wav")}

        self._flip_one_split(output)
        Pipeline(config).run()

        after_train = {p.name for p in (output / "train").rglob("*.wav")}
        after_val = {p.name for p in (output / "val").rglob("*.wav")}
        moved = before_train - after_train
        assert moved
        assert moved <= after_val

    def test_audit_passes_after_a_resplit(self, built):
        from mss_datasets.audit import run_audit
        from mss_datasets.output_tree import OutputTree

        config, output = built
        self._flip_one_split(output)
        Pipeline(config).run()

        result = run_audit(OutputTree(output).load())
        assert result["passed"], [f.summary for f in result["findings"]]


class TestInvalidPath:
    def test_bad_musdb_path_logged(self, tmp_path):
        config = PipelineConfig(
            musdb18hq_path=str(tmp_path / "nonexistent"),
            output=str(tmp_path / "output"),
        )
        pipeline = Pipeline(config)
        result = pipeline.run()
        assert result.get("error") == "No valid datasets found"
