"""Tests for download module — all network calls mocked."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mss_datasets.download import (
    DownloadError,
    _merge_medleydb_versions,
    _prune_medleydb_extras,
    _validate_zenodo_token,
    download_all,
    download_file,
    download_medleydb,
    download_musdb18hq,
    extract_archive,
    get_zenodo_file_urls,
    print_moisesdb_instructions,
    unzip_dataset,
)


def _make_mock_response(data: bytes, status: int = 200, headers: dict | None = None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status = status
    resp.read = io.BytesIO(data).read
    resp.headers = headers or {}
    if "Content-Length" not in resp.headers:
        resp.headers["Content-Length"] = str(len(data))
    return resp


class TestDownloadFile:
    def test_creates_file(self, tmp_path):
        data = b"hello world"
        dest = tmp_path / "test.bin"
        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(data)
            result = download_file("http://example.com/test.bin", dest)

        assert result == dest
        assert dest.read_bytes() == data

    def test_resume_sends_range_header(self, tmp_path):
        partial = b"hello "
        remaining = b"world"
        dest = tmp_path / "test.bin"
        dest.write_bytes(partial)

        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(
                remaining, status=206
            )
            download_file(
                "http://example.com/test.bin",
                dest,
                expected_size=len(partial) + len(remaining),
            )

        # Check Range header was sent
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Range") == f"bytes={len(partial)}-"
        assert dest.read_bytes() == partial + remaining

    def test_skips_completed_file(self, tmp_path):
        data = b"complete file"
        dest = tmp_path / "test.bin"
        dest.write_bytes(data)

        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            download_file(
                "http://example.com/test.bin",
                dest,
                expected_size=len(data),
            )

        mock_urlopen.assert_not_called()

    def test_incomplete_download_raises(self, tmp_path):
        partial = b"partial"
        dest = tmp_path / "test.bin"

        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(partial)
            with pytest.raises(DownloadError, match="Incomplete download"):
                download_file(
                    "http://example.com/test.bin",
                    dest,
                    expected_size=1000,
                )

    def test_md5_pass(self, tmp_path):
        data = b"checksum test"
        md5 = hashlib.md5(data).hexdigest()
        dest = tmp_path / "test.bin"

        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(data)
            result = download_file(
                "http://example.com/test.bin", dest, expected_md5=md5
            )

        assert result == dest

    def test_md5_fail(self, tmp_path):
        data = b"checksum test"
        dest = tmp_path / "test.bin"

        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(data)
            with pytest.raises(DownloadError, match="MD5 mismatch"):
                download_file(
                    "http://example.com/test.bin",
                    dest,
                    expected_md5="0" * 32,
                )


class TestExtractArchive:
    def test_extracts_zip(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")

        result = extract_archive(zip_path, extract_dir)

        assert result == extract_dir
        assert (extract_dir / "file1.txt").read_text() == "content1"
        assert (extract_dir / "subdir" / "file2.txt").read_text() == "content2"
        assert not zip_path.exists()

    def test_extracts_tar_gz(self, tmp_path):
        tar_path = tmp_path / "test.tar.gz"
        extract_dir = tmp_path / "extracted"

        # Create a test tar.gz
        with tarfile.open(tar_path, "w:gz") as tf:
            data = b"content1"
            info = tarfile.TarInfo(name="file1.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        result = extract_archive(tar_path, extract_dir)

        assert result == extract_dir
        assert (extract_dir / "file1.txt").read_text() == "content1"
        assert not tar_path.exists()

    def test_unsupported_format_raises(self, tmp_path):
        bad_path = tmp_path / "test.rar"
        bad_path.write_bytes(b"fake")
        with pytest.raises(DownloadError, match="Unsupported archive format"):
            extract_archive(bad_path, tmp_path / "out")


class TestUnzipDataset:
    def test_extracts_and_deletes_zip(self, tmp_path):
        # Create a test zip
        zip_path = tmp_path / "test.zip"
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("subdir/file2.txt", "content2")

        result = unzip_dataset(zip_path, extract_dir)

        assert result == extract_dir
        assert (extract_dir / "file1.txt").read_text() == "content1"
        assert (extract_dir / "subdir" / "file2.txt").read_text() == "content2"
        assert not zip_path.exists()  # zip deleted


class TestGetZenodoFileUrls:
    def test_parses_json(self):
        api_response = {
            "files": [
                {
                    "key": "dataset.zip",
                    "links": {"self": "http://zenodo.org/files/dataset.zip"},
                    "size": 1000,
                    "checksum": "md5:abc123",
                }
            ]
        }
        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(
                json.dumps(api_response).encode()
            )
            result = get_zenodo_file_urls("12345")

        assert len(result) == 1
        assert result[0]["filename"] == "dataset.zip"
        assert result[0]["size"] == 1000
        assert result[0]["checksum"] == "abc123"

    def test_restricted_no_token_raises(self):
        api_response = {"files": []}
        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(
                json.dumps(api_response).encode()
            )
            with pytest.raises(DownloadError, match="No files found"):
                get_zenodo_file_urls("12345")

    def test_restricted_with_token_sends_auth(self):
        api_response = {
            "files": [
                {
                    "key": "data.zip",
                    "links": {"self": "http://zenodo.org/files/data.zip"},
                    "size": 500,
                    "checksum": "md5:def456",
                }
            ]
        }
        with patch("mss_datasets.download.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _make_mock_response(
                json.dumps(api_response).encode()
            )
            get_zenodo_file_urls("12345", token="mytoken")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer mytoken"


class TestDownloadMusdb18hq:
    def test_skips_existing(self, tmp_path):
        dest = tmp_path / "musdb18hq"
        (dest / "train").mkdir(parents=True)
        (dest / "test").mkdir(parents=True)

        result = download_musdb18hq(tmp_path)
        assert result == dest

    @patch("mss_datasets.download._flatten_single_child")
    @patch("mss_datasets.download.unzip_dataset")
    @patch("mss_datasets.download.download_file")
    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_downloads_and_extracts(self, mock_urls, mock_dl, mock_unzip, mock_flatten, tmp_path):
        mock_urls.return_value = [
            {
                "filename": "musdb18hq.zip",
                "download": "http://zenodo.org/files/musdb18hq.zip",
                "size": 1000,
                "checksum": "abc123",
            }
        ]
        mock_dl.return_value = tmp_path / "musdb18hq.zip"
        mock_unzip.return_value = tmp_path / "musdb18hq"

        result = download_musdb18hq(tmp_path)

        mock_dl.assert_called_once()
        mock_unzip.assert_called_once()
        assert result == tmp_path / "musdb18hq"


class TestDownloadMedleydb:
    def test_no_token_raises(self, tmp_path):
        with pytest.raises(DownloadError, match="requires a Zenodo access token"):
            download_medleydb(tmp_path, token=None)

    def test_skips_existing(self, tmp_path):
        audio_dir = tmp_path / "medleydb" / "Audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "track1").mkdir()

        result = download_medleydb(tmp_path, token="mytoken")
        assert result == tmp_path / "medleydb"

    @patch("mss_datasets.download._prune_medleydb_extras")
    @patch("mss_datasets.download._merge_medleydb_versions")
    @patch("mss_datasets.download._flatten_single_child")
    @patch("mss_datasets.download.extract_archive")
    @patch("mss_datasets.download.download_file")
    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_downloads_both_versions(self, mock_urls, mock_dl, mock_extract, mock_flatten, mock_merge, mock_prune, tmp_path):
        mock_urls.return_value = [
            {
                "filename": "medleydb.zip",
                "download": "http://zenodo.org/files/medleydb.zip",
                "size": 500,
                "checksum": "def456",
            }
        ]
        mock_dl.return_value = tmp_path / "medleydb.zip"
        mock_extract.return_value = tmp_path / "medleydb"

        download_medleydb(tmp_path, token="mytoken")

        # Should call get_zenodo_file_urls twice (v1 + v2)
        assert mock_urls.call_count == 2
        mock_merge.assert_called_once()
        # Prune called: once per version + once on Audio/
        assert mock_prune.call_count == 3

    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_no_archives_raises(self, mock_urls, tmp_path):
        mock_urls.return_value = [
            {
                "filename": "readme.txt",
                "download": "http://zenodo.org/files/readme.txt",
                "size": 100,
                "checksum": "abc",
            }
        ]
        with pytest.raises(DownloadError, match="No supported archive files"):
            download_medleydb(tmp_path, token="mytoken")

    @patch("mss_datasets.download._prune_medleydb_extras")
    @patch("mss_datasets.download._merge_medleydb_versions")
    @patch("mss_datasets.download._flatten_single_child")
    @patch("mss_datasets.download.extract_archive")
    @patch("mss_datasets.download.download_file")
    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_downloads_tar_gz(self, mock_urls, mock_dl, mock_extract, mock_flatten, mock_merge, mock_prune, tmp_path):
        mock_urls.return_value = [
            {
                "filename": "medleydb_v1.tar.gz",
                "download": "http://zenodo.org/files/medleydb_v1.tar.gz",
                "size": 500,
                "checksum": "def456",
            }
        ]
        mock_dl.return_value = tmp_path / "medleydb_v1.tar.gz"
        mock_extract.return_value = tmp_path / "medleydb"

        download_medleydb(tmp_path, token="mytoken")

        assert mock_extract.call_count == 2

    def test_skips_already_extracted_version(self, tmp_path):
        """If V1/ already exists with content, skip its download but still prune."""
        v1_dir = tmp_path / "medleydb" / "V1"
        v1_dir.mkdir(parents=True)
        (v1_dir / "SomeTrack").mkdir()

        with patch("mss_datasets.download.get_zenodo_file_urls") as mock_urls, \
             patch("mss_datasets.download._merge_medleydb_versions"), \
             patch("mss_datasets.download._prune_medleydb_extras") as mock_prune, \
             patch("mss_datasets.download._flatten_single_child"), \
             patch("mss_datasets.download.extract_archive"), \
             patch("mss_datasets.download.download_file"):
            mock_urls.return_value = [
                {"filename": "data.zip", "download": "http://x/data.zip", "size": 1, "checksum": "a"},
            ]
            download_medleydb(tmp_path, token="mytoken")

        # Only called once (for V2), V1 was skipped
        assert mock_urls.call_count == 1
        # Prune still called for skipped V1, fresh V2, and Audio/
        assert mock_prune.call_count == 3


class TestPruneMedleydbExtras:
    def test_removes_raw_and_mix(self, tmp_path):
        track = tmp_path / "ArtistName_TrackName"
        track.mkdir()
        raw_dir = track / "ArtistName_TrackName_RAW"
        raw_dir.mkdir()
        (raw_dir / "raw1.wav").write_bytes(b"\x00" * 1000)
        (raw_dir / "raw2.wav").write_bytes(b"\x00" * 2000)
        mix = track / "ArtistName_TrackName_MIX.wav"
        mix.write_bytes(b"\x00" * 500)
        stems_dir = track / "ArtistName_TrackName_STEMS"
        stems_dir.mkdir()
        (stems_dir / "stem1.wav").write_bytes(b"\x00" * 800)
        (track / "ArtistName_TrackName_METADATA.yaml").write_text("title: test")

        _prune_medleydb_extras(tmp_path)

        assert not raw_dir.exists()
        assert not mix.exists()
        assert stems_dir.exists()
        assert (track / "ArtistName_TrackName_METADATA.yaml").exists()

    def test_handles_empty_dir(self, tmp_path):
        """No crash on empty or nonexistent directory."""
        _prune_medleydb_extras(tmp_path)
        _prune_medleydb_extras(tmp_path / "nonexistent")

    def test_multiple_tracks(self, tmp_path):
        for name in ("Track_A", "Track_B"):
            track = tmp_path / name
            track.mkdir()
            (track / f"{name}_RAW").mkdir()
            (track / f"{name}_RAW" / "raw.wav").write_bytes(b"\x00" * 100)
            (track / f"{name}_MIX.wav").write_bytes(b"\x00" * 50)
            (track / f"{name}_STEMS").mkdir()
            (track / f"{name}_STEMS" / "stem.wav").write_bytes(b"\x00" * 80)

        _prune_medleydb_extras(tmp_path)

        for name in ("Track_A", "Track_B"):
            track = tmp_path / name
            assert not (track / f"{name}_RAW").exists()
            assert not (track / f"{name}_MIX.wav").exists()
            assert (track / f"{name}_STEMS").exists()


class TestMergeMedleydbVersions:
    def test_merges_v1_and_v2_into_audio(self, tmp_path):
        dest = tmp_path / "medleydb"
        dest.mkdir()
        (dest / "V1" / "TrackA").mkdir(parents=True)
        (dest / "V1" / "TrackA" / "data.txt").write_text("a")
        (dest / "V2" / "TrackB").mkdir(parents=True)
        (dest / "V2" / "TrackB" / "data.txt").write_text("b")

        _merge_medleydb_versions(dest)

        assert (dest / "Audio" / "TrackA" / "data.txt").read_text() == "a"
        assert (dest / "Audio" / "TrackB" / "data.txt").read_text() == "b"
        assert not (dest / "V1").exists()
        assert not (dest / "V2").exists()

    def test_handles_missing_version_dirs(self, tmp_path):
        dest = tmp_path / "medleydb"
        dest.mkdir()

        _merge_medleydb_versions(dest)

        assert (dest / "Audio").exists()

    def test_skips_duplicate_track(self, tmp_path):
        dest = tmp_path / "medleydb"
        dest.mkdir()
        (dest / "V1" / "SameTrack").mkdir(parents=True)
        (dest / "V1" / "SameTrack" / "v1.txt").write_text("v1")
        (dest / "V2" / "SameTrack").mkdir(parents=True)
        (dest / "V2" / "SameTrack" / "v2.txt").write_text("v2")

        _merge_medleydb_versions(dest)

        # V1 moved first, V2 duplicate skipped
        assert (dest / "Audio" / "SameTrack" / "v1.txt").exists()


class TestValidateZenodoToken:
    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_valid_token(self, mock_urls):
        mock_urls.return_value = [{"filename": "data.zip"}]
        assert _validate_zenodo_token("good-token") is True

    def test_no_token(self):
        assert _validate_zenodo_token(None) is False

    @patch("mss_datasets.download.get_zenodo_file_urls")
    def test_invalid_token(self, mock_urls):
        mock_urls.side_effect = DownloadError("HTTP 401")
        assert _validate_zenodo_token("bad-token") is False


class TestDownloadAll:
    @patch("mss_datasets.download.print_moisesdb_instructions")
    @patch("mss_datasets.download.download_medleydb")
    @patch("mss_datasets.download.download_musdb18hq")
    @patch("mss_datasets.download._validate_zenodo_token", return_value=True)
    def test_orchestration(self, mock_validate, mock_musdb, mock_medley, mock_moises, tmp_path):
        mock_musdb.return_value = tmp_path / "musdb18hq"
        mock_medley.return_value = tmp_path / "medleydb"

        results = download_all(tmp_path, zenodo_token="tok")

        mock_validate.assert_called_once_with("tok")
        assert results["musdb18hq"] == tmp_path / "musdb18hq"
        assert results["medleydb"] == tmp_path / "medleydb"
        assert results["moisesdb"] is None
        mock_musdb.assert_called_once_with(tmp_path)
        mock_medley.assert_called_once_with(tmp_path, token="tok")
        mock_moises.assert_called_once()

    @patch("mss_datasets.download.print_moisesdb_instructions")
    @patch("mss_datasets.download.download_medleydb")
    @patch("mss_datasets.download.download_musdb18hq")
    @patch("mss_datasets.download._validate_zenodo_token", return_value=False)
    def test_skips_medleydb_on_invalid_token(self, mock_validate, mock_musdb, mock_medley, mock_moises, tmp_path):
        mock_musdb.return_value = tmp_path / "musdb18hq"

        results = download_all(tmp_path, zenodo_token="bad")

        assert results["musdb18hq"] == tmp_path / "musdb18hq"
        assert results["medleydb"] is None
        mock_medley.assert_not_called()

    @patch("mss_datasets.download.print_moisesdb_instructions")
    @patch("mss_datasets.download.download_musdb18hq")
    @patch("mss_datasets.download._validate_zenodo_token", return_value=False)
    def test_handles_errors_gracefully(self, mock_validate, mock_musdb, mock_moises, tmp_path):
        mock_musdb.side_effect = DownloadError("network error")

        results = download_all(tmp_path)

        assert results["musdb18hq"] is None
        assert results["medleydb"] is None
        assert results["moisesdb"] is None


class TestPrintMoisesdbInstructions:
    def test_prints_url(self, caplog):
        with caplog.at_level("INFO"):
            print_moisesdb_instructions()
        assert "music.ai/research" in caplog.text
