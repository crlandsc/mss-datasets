"""Download and extract MSS datasets from Zenodo."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Zenodo record IDs
MUSDB18HQ_RECORD = "3338373"
MEDLEYDB_V1_RECORD = "1649325"
MEDLEYDB_V2_RECORD = "1715175"


class DownloadError(Exception):
    """Raised when a dataset download fails."""


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    headers: dict[str, str] | None = None,
) -> Path:
    """Stream-download a file with resume support, progress bar, and MD5 verification.

    Args:
        url: Download URL.
        dest: Local file path to write.
        expected_size: Expected file size in bytes (for progress bar).
        expected_md5: Expected MD5 hex digest for verification.
        headers: Extra HTTP headers (e.g. Bearer token).

    Returns:
        Path to the downloaded file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    req_headers = dict(headers) if headers else {}
    initial_size = 0

    # Resume partial download
    if dest.exists():
        initial_size = dest.stat().st_size
        if expected_size and initial_size >= expected_size:
            logger.info("Already downloaded: %s", dest.name)
            if expected_md5:
                _verify_md5(dest, expected_md5)
            return dest
        req_headers["Range"] = f"bytes={initial_size}-"

    req = Request(url, headers=req_headers)
    try:
        resp = urlopen(req)  # noqa: S310
    except HTTPError as e:
        raise DownloadError(f"HTTP {e.code} downloading {url}: {e.reason}") from e

    total = expected_size or (
        int(resp.headers["Content-Length"]) + initial_size
        if resp.headers.get("Content-Length")
        else None
    )

    mode = "ab" if initial_size and resp.status == 206 else "wb"
    if mode == "wb":
        initial_size = 0

    chunk_size = 1024 * 1024  # 1 MB
    with (
        open(dest, mode) as f,
        tqdm(
            total=total,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar,
    ):
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))

    if expected_md5:
        _verify_md5(dest, expected_md5)

    return dest


def _verify_md5(path: Path, expected: str) -> None:
    """Verify file MD5 checksum."""
    md5 = hashlib.md5()  # noqa: S324
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != expected:
        raise DownloadError(
            f"MD5 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def unzip_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a zip file and delete the archive after success.

    Args:
        zip_path: Path to zip file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the extraction directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s → %s", zip_path.name, extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Clean up zip to save disk space
    zip_path.unlink()
    logger.info("Deleted archive: %s", zip_path.name)

    return extract_dir


def get_zenodo_file_urls(
    record_id: str, token: str | None = None
) -> list[dict]:
    """Fetch file metadata from Zenodo API.

    Returns list of dicts with keys: filename, download, size, checksum.
    """
    url = f"https://zenodo.org/api/records/{record_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        resp = urlopen(req)  # noqa: S310
    except HTTPError as e:
        raise DownloadError(
            f"Failed to fetch Zenodo record {record_id}: HTTP {e.code}"
        ) from e

    data = json.loads(resp.read())
    files = data.get("files", [])

    if not files:
        msg = f"No files found for Zenodo record {record_id}."
        if not token:
            msg += (
                "\nThis may be a restricted record requiring a Zenodo access token.\n"
                "Set ZENODO_TOKEN in .env or pass --zenodo-token.\n"
                "See .env.example for setup instructions."
            )
        raise DownloadError(msg)

    return [
        {
            "filename": f["key"],
            "download": f["links"]["self"],
            "size": f["size"],
            "checksum": f["checksum"].replace("md5:", "") if f.get("checksum") else None,
        }
        for f in files
    ]


def download_musdb18hq(data_dir: Path) -> Path | None:
    """Download MUSDB18-HQ from Zenodo (open access).

    Returns path to extracted dataset, or None if skipped.
    """
    dest_dir = data_dir / "musdb18hq"

    # Skip if already extracted
    if (dest_dir / "train").exists() and (dest_dir / "test").exists():
        logger.info("MUSDB18-HQ already exists at %s, skipping download", dest_dir)
        return dest_dir

    logger.info("Downloading MUSDB18-HQ from Zenodo record %s...", MUSDB18HQ_RECORD)
    files = get_zenodo_file_urls(MUSDB18HQ_RECORD)

    # Find the main zip file
    zip_info = None
    for f in files:
        if f["filename"].endswith(".zip"):
            zip_info = f
            break

    if not zip_info:
        raise DownloadError("No zip file found in MUSDB18-HQ Zenodo record")

    zip_path = data_dir / zip_info["filename"]
    download_file(
        url=zip_info["download"],
        dest=zip_path,
        expected_size=zip_info["size"],
        expected_md5=zip_info["checksum"],
    )

    unzip_dataset(zip_path, dest_dir)

    # Handle nested directory: zip may extract to a subdirectory
    _flatten_single_child(dest_dir)

    return dest_dir


def download_medleydb(data_dir: Path, token: str | None = None) -> Path | None:
    """Download MedleyDB v1+v2 from Zenodo (restricted, requires token).

    Returns path to merged dataset, or None if skipped.
    """
    if not token:
        logger.warning(
            "Skipping MedleyDB download: no Zenodo token provided.\n"
            "Set ZENODO_TOKEN in .env or pass --zenodo-token.\n"
            "See .env.example for setup instructions."
        )
        return None

    dest_dir = data_dir / "medleydb"
    audio_dir = dest_dir / "Audio"

    # Skip if already has audio files
    if audio_dir.exists() and any(audio_dir.iterdir()):
        logger.info("MedleyDB already exists at %s, skipping download", dest_dir)
        return dest_dir

    headers = {"Authorization": f"Bearer {token}"}

    for label, record_id in [("v1", MEDLEYDB_V1_RECORD), ("v2", MEDLEYDB_V2_RECORD)]:
        logger.info("Downloading MedleyDB %s from Zenodo record %s...", label, record_id)
        files = get_zenodo_file_urls(record_id, token=token)

        for f_info in files:
            if not f_info["filename"].endswith(".zip"):
                continue
            zip_path = data_dir / f_info["filename"]
            download_file(
                url=f_info["download"],
                dest=zip_path,
                expected_size=f_info["size"],
                expected_md5=f_info["checksum"],
                headers=headers,
            )
            # Extract each zip into the shared medleydb dir
            unzip_dataset(zip_path, dest_dir)

    _flatten_single_child(dest_dir)

    return dest_dir


def print_moisesdb_instructions() -> None:
    """Print instructions for manually downloading MoisesDB."""
    logger.info(
        "MoisesDB cannot be auto-downloaded.\n"
        "Download manually from: https://music.ai/research/\n"
        "Then pass the path via --moisesdb-path"
    )


def download_all(
    data_dir: Path, zenodo_token: str | None = None
) -> dict[str, Path | None]:
    """Download all available datasets.

    Returns dict mapping dataset name to extracted path (or None if skipped/failed).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {}

    # MUSDB18-HQ (open access)
    try:
        results["musdb18hq"] = download_musdb18hq(data_dir)
    except DownloadError as e:
        logger.error("MUSDB18-HQ download failed: %s", e)
        results["musdb18hq"] = None

    # MedleyDB (restricted)
    try:
        results["medleydb"] = download_medleydb(data_dir, token=zenodo_token)
    except DownloadError as e:
        logger.error("MedleyDB download failed: %s", e)
        results["medleydb"] = None

    # MoisesDB (manual only)
    print_moisesdb_instructions()
    results["moisesdb"] = None

    return results


def _flatten_single_child(directory: Path) -> None:
    """If directory contains a single subdirectory, move its contents up."""
    children = list(directory.iterdir())
    if len(children) == 1 and children[0].is_dir():
        child = children[0]
        for item in child.iterdir():
            item.rename(directory / item.name)
        child.rmdir()
