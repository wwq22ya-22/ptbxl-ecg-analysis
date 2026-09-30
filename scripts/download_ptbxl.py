"""Download the official PTB-XL 1.0.3 100 Hz subset with SHA-256 checks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path

DATASET_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
DATASET_PAGE = "https://physionet.org/content/ptb-xl/1.0.3/"
REQUIRED_METADATA = ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sha256_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            digest, relative = line.strip().split(maxsplit=1)
            relative = relative.removeprefix("./")
            parts = PurePosixPath(relative).parts
            if not parts or ".." in parts or PurePosixPath(relative).is_absolute():
                raise ValueError(f"Unsafe path in SHA256SUMS.txt: {relative!r}")
            result[relative] = digest.lower()
    return result


def ensure_sha256_manifest(root: Path) -> Path:
    """Fetch the official hash manifest when starting from an empty folder."""
    path = root / "SHA256SUMS.txt"
    if path.is_file():
        return path
    temporary = root / "SHA256SUMS.txt.download"
    try:
        with urlopen(DATASET_URL + "SHA256SUMS.txt", timeout=120) as response:
            with temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


async def fetch_verified(
    session: aiohttp.ClientSession,
    root: Path,
    relative: str,
    expected_hash: str,
) -> bool:
    """Download one file atomically, returning False when already verified."""
    target = root.joinpath(*PurePosixPath(relative).parts)
    if target.is_file() and sha256_file(target) == expected_hash:
        return False
    if target.exists():
        raise RuntimeError(f"Existing file has an unexpected SHA-256; inspect it before retrying: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    url = DATASET_URL + relative
    for attempt in range(6):
        try:
            digest = hashlib.sha256()
            async with session.get(url) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    async for chunk in response.content.iter_chunked(128 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch for {relative}")
            temporary.replace(target)
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError):
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                raise
            await asyncio.sleep(min(2 ** attempt, 20))
    raise AssertionError("Unreachable download retry state")


async def download_files(
    root: Path,
    files: list[tuple[str, str]],
    workers: int,
) -> tuple[int, int, list[str]]:
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for item in files:
        queue.put_nowait(item)
    transferred = 0
    verified = 0
    failures: list[str] = []
    started = time.monotonic()
    last_report = started
    timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
    connector = aiohttp.TCPConnector(limit=workers, ttl_dns_cache=300)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        trust_env=True,
        headers={"User-Agent": "ptbxl-ecg-system/1.0 (educational research)"},
    ) as session:
        async def worker() -> None:
            nonlocal transferred, verified, last_report
            while True:
                try:
                    relative, expected_hash = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    if len(failures) < 20:
                        was_downloaded = await fetch_verified(session, root, relative, expected_hash)
                        transferred += int(was_downloaded)
                        verified += 1
                except Exception as exc:
                    failures.append(f"{relative}: {exc}")
                finally:
                    queue.task_done()
                now = time.monotonic()
                if now - last_report >= 20:
                    print(
                        f"Verified {verified:,}/{len(files):,} files; newly downloaded {transferred:,}; "
                        f"failures {len(failures)}; elapsed {(now - started) / 60:.1f} min",
                        flush=True,
                    )
                    last_report = now

        await asyncio.gather(*(worker() for _ in range(workers)))
    return transferred, verified, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-files", type=int, help="Download only the first N waveform files for a connection check")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be between 1 and 128")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")

    root = args.dataset_root or resolve_project_path(config["dataset"]["root"], PROJECT_ROOT)
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        manifest_path = ensure_sha256_manifest(root)
    except Exception as exc:
        parser.exit(2, f"Could not fetch official SHA256SUMS.txt: {exc}\n")
    manifest = load_sha256_manifest(manifest_path)
    waveform_files = sorted(
        (path, digest) for path, digest in manifest.items() if path.startswith("records100/")
    )
    if args.max_files is not None:
        waveform_files = waveform_files[: args.max_files]
    metadata_files = [(name, manifest[name]) for name in REQUIRED_METADATA]
    selected = metadata_files + waveform_files
    print(f"Source: {DATASET_PAGE}")
    print(f"Destination: {root}")
    print(f"Files to verify/download: {len(selected):,}")
    transferred, verified, failures = asyncio.run(download_files(root, selected, args.workers))
    print(f"Verified: {verified:,}; newly downloaded: {transferred:,}; failures: {len(failures)}")
    for failure in failures[:10]:
        print(f"  {failure}")
    if failures:
        print("Rerun the same command to resume; verified files will be skipped.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
