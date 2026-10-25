"""Local records and the project storage limit."""
from __future__ import annotations
import json
import os
import stat
import time
from pathlib import Path

_STORAGE_ROOT = None
_STORAGE_LIMIT = 30_000_000_000
_STORAGE_ESTIMATE = 0
_STORAGE_CHECKED_AT = float('-inf')


def configure_storage(root, limit=30_000_000_000):
    global _STORAGE_ROOT, _STORAGE_LIMIT, _STORAGE_ESTIMATE, _STORAGE_CHECKED_AT
    _STORAGE_ROOT, _STORAGE_LIMIT = Path(root).resolve(), limit
    _STORAGE_ESTIMATE = check_storage(_STORAGE_ROOT, limit=limit)
    _STORAGE_CHECKED_AT = time.monotonic()


def _guard_write(path, additional):
    global _STORAGE_ESTIMATE, _STORAGE_CHECKED_AT
    if _STORAGE_ROOT is not None and Path(path).resolve().is_relative_to(_STORAGE_ROOT):
        stamp = time.monotonic()
        if stamp - _STORAGE_CHECKED_AT >= 1.:
            _STORAGE_ESTIMATE = check_storage(_STORAGE_ROOT, limit=_STORAGE_LIMIT)
            _STORAGE_CHECKED_AT = stamp
        if _STORAGE_ESTIMATE + additional > _STORAGE_LIMIT - 1_000_000_000:
            # Refresh before rejecting an overwrite or a recently removed file.
            _STORAGE_ESTIMATE = check_storage(_STORAGE_ROOT, additional=additional, limit=_STORAGE_LIMIT)
            _STORAGE_CHECKED_AT = stamp
        # Count the entire payload, including overwrites, until the next scan.
        # Every write keeps the existing one-GB reserve. Request checkpoints also
        # scan the filesystem, and the next second incorporates external writes.
        _STORAGE_ESTIMATE += additional


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def project_bytes(root):
    total = 0
    directories = [os.fspath(root)]
    while directories:
        try:
            with os.scandir(directories.pop()) as iterator:
                entries = list(iterator)
        except OSError:
            # Preserve rglob's handling of unreadable or removed directories.
            continue
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
            elif stat.S_ISDIR(metadata.st_mode):
                directories.append(entry.path)
    return total


def check_storage(root, additional=0, limit=30_000_000_000):
    size = project_bytes(root)
    if size + additional > limit - 1_000_000_000:
        raise RuntimeError(f'Project storage guard: {size} bytes, requested {additional}, limit {limit}')
    return size


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    _guard_write(path, len(payload.encode('utf-8')))
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values)
    _guard_write(path, len(payload.encode('utf-8')))
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False) + '\n'
    _guard_write(path, len(payload.encode('utf-8')))
    with path.open('a') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
