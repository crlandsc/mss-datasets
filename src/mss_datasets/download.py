"""Download and extract MSS datasets from Zenodo."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
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

    # Verify download is complete before checking MD5
    actual_size = dest.stat().st_size
    if expected_size and actual_size < expected_size:
        raise DownloadError(
            f"Incomplete download for {dest.name}: "
            f"got {actual_size:,} bytes, expected {expected_size:,}. "
            f"Re-run to resume."
        )

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


ARCHIVE_EXTENSIONS = (".zip", ".tar.gz", ".tar.bz2", ".tar")


def _is_archive(filename: str) -> bool:
    """Check if filename has a supported archive extension."""
    return any(filename.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a zip or tar archive and delete it after success.

    Args:
        archive_path: Path to archive file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the extraction directory.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s → %s", archive_path.name, extract_dir)
    name = archive_path.name

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif name.endswith((".tar.gz", ".tar.bz2", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
    else:
        raise DownloadError(f"Unsupported archive format: {name}")

    # Clean up archive to save disk space
    archive_path.unlink()
    logger.info("Deleted archive: %s", archive_path.name)

    return extract_dir


def unzip_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a zip file and delete the archive after success.

    Deprecated: use extract_archive() which handles zip and tar formats.
    """
    return extract_archive(zip_path, extract_dir)


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
        if token:
            msg += (
                "\nYour token was sent but access was denied. "
                "You must request access to this record:\n"
                f"  https://zenodo.org/records/{record_id}\n"
                "Click 'Request access', then wait for owner approval."
            )
        else:
            msg += (
                "\nThis is a restricted record requiring a Zenodo access token.\n"
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

    Archives extract to V1/ and V2/ subdirs, which are then merged into Audio/
    to match the layout expected by MedleydbAdapter.

    Returns path to merged dataset, or None if skipped.
    """
    if not token:
        raise DownloadError("MedleyDB requires a Zenodo access token")

    dest_dir = data_dir / "medleydb"
    audio_dir = dest_dir / "Audio"

    # Skip if already has audio files
    if audio_dir.exists() and any(audio_dir.iterdir()):
        logger.info("MedleyDB already exists at %s, skipping download", dest_dir)
        return dest_dir

    headers = {"Authorization": f"Bearer {token}"}

    for label, record_id in [("v1", MEDLEYDB_V1_RECORD), ("v2", MEDLEYDB_V2_RECORD)]:
        version_dir = dest_dir / label.upper()

        # Skip this version if already extracted
        if version_dir.exists() and any(version_dir.iterdir()):
            logger.info("MedleyDB %s already extracted, skipping download", label)
            _prune_medleydb_extras(version_dir)
            continue

        logger.info("Downloading MedleyDB %s from Zenodo record %s...", label, record_id)
        files = get_zenodo_file_urls(record_id, token=token)

        archives = [f for f in files if _is_archive(f["filename"])]
        if not archives:
            filenames = [f["filename"] for f in files]
            raise DownloadError(
                f"No supported archive files found in MedleyDB {label} "
                f"(record {record_id}). Files found: {filenames}"
            )

        for f_info in archives:
            archive_path = data_dir / f_info["filename"]
            download_file(
                url=f_info["download"],
                dest=archive_path,
                expected_size=f_info["size"],
                expected_md5=f_info["checksum"],
                headers=headers,
            )
            extract_archive(archive_path, dest_dir)

        _flatten_single_child(dest_dir / label.upper())

        # Prune RAW/MIX immediately after extraction to save disk space
        _prune_medleydb_extras(version_dir)

    # Merge V1/ and V2/ track dirs into Audio/ for the adapter
    _merge_medleydb_versions(dest_dir)

    # Prune any un-pruned tracks in Audio/ (resume scenario)
    _prune_medleydb_extras(audio_dir)

    return dest_dir


def _prune_medleydb_extras(directory: Path) -> None:
    """Remove RAW folders and MIX wav files from MedleyDB track directories.

    The adapter only needs *_STEMS/ and *_METADATA.yaml. RAW dirs and MIX files
    are typically 50-60% of the extracted data and can be safely discarded.
    """
    if not directory.exists():
        return

    pruned_bytes = 0
    for track_dir in directory.iterdir():
        if not track_dir.is_dir():
            continue
        for item in track_dir.iterdir():
            if item.is_dir() and item.name.endswith("_RAW"):
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item)
                pruned_bytes += size
            elif item.is_file() and item.name.endswith("_MIX.wav"):
                pruned_bytes += item.stat().st_size
                item.unlink()

    if pruned_bytes:
        gb = pruned_bytes / (1024 ** 3)
        logger.info("Pruned %.1f GB of RAW/MIX files from %s", gb, directory.name)


def _merge_medleydb_versions(dest_dir: Path) -> None:
    """Move track directories from V1/ and V2/ into a unified Audio/ directory."""
    audio_dir = dest_dir / "Audio"
    audio_dir.mkdir(exist_ok=True)

    for version_dir_name in ("V1", "V2"):
        version_dir = dest_dir / version_dir_name
        if not version_dir.exists():
            continue
        for track_dir in version_dir.iterdir():
            if track_dir.is_dir():
                target = audio_dir / track_dir.name
                if not target.exists():
                    track_dir.rename(target)
                else:
                    logger.warning("Duplicate track dir %s, skipping", track_dir.name)
        # Remove empty version dir
        if not any(version_dir.iterdir()):
            version_dir.rmdir()
            logger.info("Merged %s/ into Audio/", version_dir_name)


def print_moisesdb_instructions() -> None:
    """Print instructions for manually downloading MoisesDB."""
    logger.info(
        "MoisesDB cannot be auto-downloaded.\n"
        "Download manually from: https://music.ai/research/\n"
        "Then pass the path via --moisesdb-path"
    )


def _validate_zenodo_token(token: str | None) -> bool:
    """Pre-flight check: verify Zenodo token can access MedleyDB records.

    Returns True if token is valid, False otherwise. Logs warnings on failure.
    """
    if not token:
        logger.warning(
            "No Zenodo token provided — MedleyDB will be skipped.\n"
            "Set ZENODO_TOKEN in .env or pass --zenodo-token.\n"
            "See .env.example for setup instructions."
        )
        return False

    # Test token against MedleyDB v1 record
    try:
        files = get_zenodo_file_urls(MEDLEYDB_V1_RECORD, token=token)
        if files:
            logger.info("Zenodo token validated successfully")
            return True
    except DownloadError as e:
        logger.error("Zenodo token validation failed: %s", e)
    return False


def download_all(
    data_dir: Path, zenodo_token: str | None = None
) -> dict[str, Path | None]:
    """Download all available datasets.

    Validates credentials before starting any downloads.
    Returns dict mapping dataset name to extracted path (or None if skipped/failed).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {}

    # Pre-flight: validate Zenodo token before any downloads
    medleydb_token_valid = _validate_zenodo_token(zenodo_token)

    # MoisesDB (manual only) — print early so user sees all info upfront
    print_moisesdb_instructions()

    # MUSDB18-HQ (open access)
    try:
        results["musdb18hq"] = download_musdb18hq(data_dir)
    except DownloadError as e:
        logger.error("MUSDB18-HQ download failed: %s", e)
        results["musdb18hq"] = None

    # MedleyDB (restricted)
    if medleydb_token_valid:
        try:
            results["medleydb"] = download_medleydb(data_dir, token=zenodo_token)
        except DownloadError as e:
            logger.error("MedleyDB download failed: %s", e)
            results["medleydb"] = None
    else:
        results["medleydb"] = None

    # MoisesDB
    results["moisesdb"] = None

    return results


def _flatten_single_child(directory: Path) -> None:
    """If directory contains a single subdirectory, move its contents up."""
    if not directory.exists():
        return
    children = list(directory.iterdir())
    if len(children) == 1 and children[0].is_dir():
        child = children[0]
        for item in child.iterdir():
            item.rename(directory / item.name)
        child.rmdir()
