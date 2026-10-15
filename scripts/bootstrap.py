#!/usr/bin/env python3
"""Fetch the pinned llama.cpp revision and apply the verified MistyLoader patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args, output=False):
    cmd = ['git', '-C', str(repo), *args]
    if output:
        return subprocess.check_output(cmd, text=True).strip()
    subprocess.run(cmd, check=True)


def verify(repo, lock):
    if not (repo / '.git').is_dir():
        raise RuntimeError(f'{repo} is not an owned source checkout; move it aside manually')
    if git(repo, 'rev-parse', 'HEAD', output=True) != lock['revision']:
        raise RuntimeError('source revision differs from upstream.lock.json; refusing to reset it')
    expected = lock['changed_files_sha256']
    actual = set(git(repo, 'diff', '--name-only', 'HEAD', output=True).splitlines())
    actual.update(git(repo, 'ls-files', '--others', '--exclude-standard', output=True).splitlines())
    if actual != set(expected):
        raise RuntimeError('source checkout has unexpected changes; refusing to overwrite them')
    for name, digest in expected.items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'source checksum mismatch: {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='Verify the existing checkout without fetching or changing it.')
    args = ap.parse_args()
    lock = json.loads((ROOT / 'upstream.lock.json').read_text())
    patch = ROOT / lock['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock['patch_sha256']:
        raise RuntimeError('runtime patch checksum mismatch')
    destination = ROOT / 'vendor/llama.cpp'
    if destination.exists() or args.check:
        verify(destination, lock)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.mistyloader-', dir=destination.parent) as temp:
            checkout = Path(temp) / 'llama.cpp'
            checkout.mkdir()
            git(checkout, 'init', '-q')
            git(checkout, 'remote', 'add', 'origin', lock['repository'])
            git(checkout, 'fetch', '--depth=1', 'origin', lock['revision'])
            git(checkout, 'checkout', '--detach', 'FETCH_HEAD')
            git(checkout, 'apply', '--check', str(patch))
            git(checkout, 'apply', str(patch))
            verify(checkout, lock)
            checkout.rename(destination)
    print(f'MistyLoader source verified: {destination}')


if __name__ == '__main__':
    main()
