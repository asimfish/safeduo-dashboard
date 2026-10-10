"""Authenticate immutable V4 kernels before importing or executing their bytes."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def verified_reuse():
    raw = (HERE / 'REUSED_SOURCES_V6.json').read_bytes()
    if len(raw) > 1024**2:
        raise ValueError('reuse manifest bound')
    manifest = json.loads(raw)
    for path, digest in manifest['files'].items():
        data = Path(path).read_bytes()
        if len(data) > 1024**2 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('immutable reused source changed: ' + path)
    root = manifest['root']
    if root not in sys.path:
        sys.path.append(root)
    return manifest['files']


def execute_reused(name, namespace):
    files = verified_reuse()
    paths = [p for p in files if Path(p).name == name]
    if len(paths) != 1:
        raise ValueError('exact reused source name')
    path = paths[0]
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != files[path]:
        raise ValueError('reused executable buffer changed')
    exec(compile(raw, path, 'exec'), namespace)
