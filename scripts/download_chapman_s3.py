"""Download the official public PhysioNet AWS mirror, verifying every file."""
import concurrent.futures
import hashlib
import sys
import threading
import time
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'outputs/external_validation/chapman_data'
SOURCE = 'https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/'
STATE = threading.local()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'outputs/dataset_compatibility/chapman_SHA256SUMS_s3.txt'
    if not manifest.is_file():
        response = requests.get(SOURCE + 'SHA256SUMS.txt', timeout=60)
        response.raise_for_status()
        manifest.write_bytes(response.content)
    entries = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split(maxsplit=1)
        relative = PurePosixPath(name.lstrip('*'))
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe official manifest path')
        entries.append((expected, relative.as_posix()))
    if len(entries) != 90759:
        raise RuntimeError('Unexpected manifest count')
    (DESTINATION / 'SHA256SUMS.txt').write_bytes(manifest.read_bytes())

    def download(item):
        expected, relative = item
        target = DESTINATION.joinpath(*PurePosixPath(relative).parts)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(STATE, 'session'):
            STATE.session = requests.Session()
            STATE.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1))
        for attempt in range(6):
            try:
                response = STATE.session.get(SOURCE + relative, timeout=(20, 60))
                response.raise_for_status()
                payload = response.content
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise RuntimeError('SHA-256 mismatch')
                temp = target.with_suffix(target.suffix + '.download')
                temp.write_bytes(payload)
                temp.replace(target)
                return len(payload)
            except Exception as error:
                if attempt == 5:
                    raise RuntimeError(f'{relative}: {error}') from error
                time.sleep(min(5, attempt + 1))

    print(f'Official AWS mirror: {len(entries):,} files; 48 concurrent connections', flush=True)
    # Headers first allow a full diagnostic audit before waveform transfer finishes.
    entries.sort(key=lambda item: (item[1].endswith('.mat'), item[1]))
    total_bytes = 0
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        for index, received in enumerate(pool.map(download, entries), 1):
            total_bytes += received
            if index % 1000 == 0:
                print(f'Verified {index:,}/{len(entries):,}; downloaded {total_bytes / 1e9:.2f} GB; elapsed {(time.monotonic() - began) / 60:.1f} min', flush=True)
    headers = sum(1 for _ in (DESTINATION / 'WFDBRecords').rglob('*.hea'))
    if headers != 45152:
        raise RuntimeError(f'Unexpected header count: {headers}')
    (DESTINATION / '.codex_sha256_verified').write_text(f'verified {len(entries)} files\n', encoding='utf-8')
    print(f'Dataset ready: {DESTINATION}; {headers} ECGs', flush=True)


if __name__ == '__main__':
    main()
