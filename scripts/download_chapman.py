"""Download the official public PhysioNet Chapman-Shaoxing/Ningbo archive."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_validation.chapman import ARCHIVE_URL


def download_resumable(url: str, partial_path: Path) -> None:
    """Stream the archive to disk and resume a partial HTTP Range download."""
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(url, headers=headers)
    with urlopen(request, timeout=90) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        total = response.headers.get("Content-Length")
        total_bytes = offset + int(total) if total and total.isdigit() else None
        received = offset
        with partial_path.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if total_bytes:
                    print(f"\rDownloaded {received / 1_000_000_000:.2f} / {total_bytes / 1_000_000_000:.2f} GB", end="", flush=True)
                else:
                    print(f"\rDownloaded {received / 1_000_000_000:.2f} GB", end="", flush=True)
    print()


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            posix_name = PurePosixPath(member.filename)
            if posix_name.is_absolute() or ".." in posix_name.parts:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            target = (destination / Path(*posix_name.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Archive member escapes destination: {member.filename}")
        archive.extractall(destination)


def find_dataset_root(root: Path) -> Path | None:
    candidates = [root]
    candidates.extend(path.parent for path in root.glob("*/ConditionNames_SNOMED-CT.csv"))
    candidates.extend(path.parent for path in root.glob("*/*/ConditionNames_SNOMED-CT.csv"))
    for candidate in candidates:
        if (candidate / "ConditionNames_SNOMED-CT.csv").is_file() and (candidate / "WFDBRecords").is_dir():
            return candidate
    return None


def verify_sha256_manifest(dataset_root: Path, *, minimum_entries: int = 40_000) -> int:
    """Verify each extracted file against PhysioNet's published SHA-256 list."""
    import hashlib

    manifest = dataset_root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Official checksum manifest is missing: {manifest}")
    entries = []
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        digest, relative = fields
        relative = relative.lstrip("*").replace(chr(92), "/")
        while relative.startswith("./"):
            relative = relative[2:]
        entries.append((digest.casefold(), relative))
    if len(entries) < minimum_entries:
        raise RuntimeError(f"Checksum manifest looks incomplete: {len(entries)} entries.")
    for index, (expected, relative) in enumerate(entries, start=1):
        path = dataset_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise RuntimeError(f"File listed in SHA256SUMS.txt is missing: {relative}")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual.casefold() != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}")
        if index % 2000 == 0:
            print(f"Verified {index:,}/{len(entries):,} dataset files", flush=True)
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("D:/datasets/ecg-arrhythmia"),
        help="External data directory; default D:/datasets/ecg-arrhythmia",
    )
    args = parser.parse_args()
    destination = args.dataset_root.expanduser().resolve()
    existing_root = find_dataset_root(destination)
    if existing_root:
        hea_count = sum(1 for _ in existing_root.rglob("*.hea"))
        if hea_count < 40_000:
            raise RuntimeError(f"Existing dataset is incomplete: found only {hea_count} record headers.")
        stamp = existing_root / ".codex_sha256_verified"
        if not stamp.is_file():
            print("Checking existing dataset against PhysioNet SHA-256 manifest.", flush=True)
            verified = verify_sha256_manifest(existing_root)
            stamp.write_text(f"verified {verified} files" + chr(10), encoding="utf-8")
        print(f"Dataset ready: {existing_root}; WFDB headers: {hea_count:,}")
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.parent / f"{destination.name}.zip"
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        print(f"Downloading official archive from {ARCHIVE_URL}")
        print(f"Temporary archive: {partial_path}")
        download_resumable(ARCHIVE_URL, partial_path)
        partial_path.replace(archive_path)
        print("Checking archive and extracting files; this dataset expands to about 5.1 GB.")
        safe_extract(archive_path, destination)
        extracted_root = find_dataset_root(destination)
        if extracted_root is None:
            raise RuntimeError("Archive extracted, but the expected WFDBRecords and code dictionary were not found.")
        hea_count = sum(1 for _ in extracted_root.rglob("*.hea"))
        if hea_count < 40_000:
            raise RuntimeError(f"Expected about 45,152 record headers, found {hea_count}; keeping archive for recovery.")
        print("Verifying extracted files against the official SHA-256 manifest.", flush=True)
        verified = verify_sha256_manifest(extracted_root)
        (extracted_root / ".codex_sha256_verified").write_text(
            f"verified {verified} files" + chr(10), encoding="utf-8"
        )
        archive_path.unlink()
        print(f"Dataset ready: {extracted_root}")
        print(f"WFDB header files: {hea_count:,}")
        return 0
    except Exception as exc:
        print(f"Download/extraction failed: {exc}", file=sys.stderr)
        if partial_path.exists():
            print(f"Partial archive retained for resume: {partial_path}", file=sys.stderr)
        elif archive_path.exists():
            print(f"Archive retained for recovery: {archive_path}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())







