"""Resumable bounded parallel ranges of the official 2.5 GB public archive."""
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_chapman import safe_extract, find_dataset_root, verify_sha256_manifest
from src.external_validation.chapman import ARCHIVE_URL


def main():
    destination = ROOT / 'outputs/external_validation/chapman_data'
    parts = ROOT / 'outputs/external_validation/chapman_archive_parts'
    parts.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(ARCHIVE_URL, headers={'Range': 'bytes=0-0'}), timeout=90) as response:
        if response.status != 206:
            raise RuntimeError('Server does not support byte ranges')
        total = int(response.headers['Content-Range'].rsplit('/', 1)[1])
        etag = response.headers.get('ETag')
    manifest = {'url': ARCHIVE_URL, 'total': total, 'etag': etag, 'chunk_bytes': 32 * 1024 * 1024}
    meta = parts / 'archive.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise RuntimeError('Archive changed; refusing to reuse old download pieces')
    meta.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    chunk_bytes = manifest['chunk_bytes']

    def fetch(index):
        start = index * chunk_bytes
        end = min(total, start + chunk_bytes) - 1
        expected = end - start + 1
        path = parts / f'{index:04d}.part'
        if path.exists() and path.stat().st_size > expected:
            raise RuntimeError(f'Oversized piece: {path}')
        for attempt in range(8):
            offset = path.stat().st_size if path.exists() else 0
            if offset == expected:
                return index, expected
            try:
                request = Request(ARCHIVE_URL, headers={
                    'Range': f'bytes={start + offset}-{end}', 'If-Range': etag,
                })
                with urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start + offset}-{end}/{total}':
                        raise RuntimeError('Unexpected range response')
                    with path.open('ab') as output:
                        while data := response.read(256 * 1024):
                            output.write(data)
                if path.stat().st_size != expected:
                    raise RuntimeError('Incomplete range response')
                return index, expected
            except Exception as error:
                print(f'Piece {index} retry {attempt + 1}: {error}', flush=True)
                time.sleep(min(10, attempt + 1))
        raise RuntimeError(f'Piece {index} failed repeatedly')

    count = (total + chunk_bytes - 1) // chunk_bytes
    print(f'Archive {total:,} bytes; {count} pieces; 12 concurrent connections', flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(count)]):
            index, size = future.result()
            completed += size
            print(f'Completed piece {index}; {completed / total:.1%} verified byte lengths', flush=True)
    archive = destination.parent / 'chapman_verified_ranges.zip'
    with archive.open('wb') as output:
        for index in range(count):
            with (parts / f'{index:04d}.part').open('rb') as stream:
                while data := stream.read(8 * 1024 * 1024):
                    output.write(data)
    if archive.stat().st_size != total:
        raise RuntimeError('Assembled archive length mismatch')
    print('Extracting official archive', flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, destination)
    dataset = find_dataset_root(destination)
    if dataset is None:
        raise RuntimeError('Expected dataset structure missing')
    count = sum(1 for _ in (dataset / 'WFDBRecords').rglob('*.hea'))
    if count != 45152:
        raise RuntimeError(f'Expected 45152 headers, got {count}')
    verified = verify_sha256_manifest(dataset)
    (dataset / '.codex_sha256_verified').write_text(f'verified {verified} files\n', encoding='utf-8')
    print(f'Dataset ready: {dataset}; {count} headers, {verified} verified files', flush=True)


if __name__ == '__main__':
    main()
