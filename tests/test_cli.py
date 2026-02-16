"""Tests for CLI — flags, config loading, dry-run."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from click.testing import CliRunner

from mss_aggregate.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def musdb_fixture(tmp_path):
    """Create minimal MUSDB18-HQ fixture."""
    sr = 44100
    rng = np.random.default_rng(42)
    for split in ("train", "test"):
        d = tmp_path / "musdb" / split / "TestArtist - TestSong"
        d.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other", "mixture"):
            data = rng.uniform(-0.3, 0.3, (sr // 4, 2)).astype(np.float32)
            sf.write(str(d / f"{stem}.wav"), data, sr, subtype="FLOAT")
    return tmp_path / "musdb"


class TestHelp:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Aggregate multiple MSS datasets" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestFlags:
    def test_no_dataset_shows_error(self, runner):
        result = runner.invoke(main, ["--output", "/tmp/out"])
        assert result.exit_code != 0
        assert "At least one dataset path" in result.output

    def test_dry_run(self, runner, musdb_fixture, tmp_path):
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(tmp_path / "output"),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "Total tracks:" in result.output
        assert not (tmp_path / "output").exists()

    def test_basic_run(self, runner, musdb_fixture, tmp_path):
        output = tmp_path / "output"
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert "Complete" in result.output
        assert (output / "vocals").is_dir()
        assert (output / "metadata" / "manifest.json").exists()

    def test_profile_vdbo_gp(self, runner, musdb_fixture, tmp_path):
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(tmp_path / "output"),
            "--profile", "vdbo+gp",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "guitar" in result.output or "piano" in result.output

    def test_verbose_flag(self, runner, musdb_fixture, tmp_path):
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(tmp_path / "output"),
            "--verbose",
            "--dry-run",
        ])
        assert result.exit_code == 0

    def test_group_by_dataset(self, runner, musdb_fixture, tmp_path):
        output = tmp_path / "output"
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(output),
            "--group-by-dataset",
        ])
        assert result.exit_code == 0
        assert (output / "vocals" / "musdb18hq").is_dir()


class TestConfigFile:
    def test_loads_from_yaml(self, runner, musdb_fixture, tmp_path):
        config_path = tmp_path / "config.yaml"
        config = {
            "datasets": {
                "musdb18hq_path": str(musdb_fixture),
            },
            "output": str(tmp_path / "output"),
            "profile": "vdbo",
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(main, [
            "--config", str(config_path),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Total tracks:" in result.output

    def test_cli_overrides_config(self, runner, musdb_fixture, tmp_path):
        config_path = tmp_path / "config.yaml"
        config = {
            "datasets": {
                "musdb18hq_path": str(musdb_fixture),
            },
            "output": str(tmp_path / "config_output"),
            "profile": "vdbo",
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cli_output = tmp_path / "cli_output"
        result = runner.invoke(main, [
            "--config", str(config_path),
            "--output", str(cli_output),
            "--dry-run",
        ])
        assert result.exit_code == 0


class TestSummaryOutput:
    def test_shows_stem_counts(self, runner, musdb_fixture, tmp_path):
        output = tmp_path / "output"
        result = runner.invoke(main, [
            "--musdb18hq-path", str(musdb_fixture),
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert "vocals/" in result.output
        assert "drums/" in result.output
        assert "WAV files" in result.output
