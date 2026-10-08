#!/usr/bin/env python3
"""Hiding study: fetch files of the public Hugging Face backup (wave 2) at a pinned commit, each checked against
backups/wave2/MANIFEST.json (hashes taken on the source host), stored read-only at <out>/<repo path>.
Selection: every manifest path that matches all of the --contains strings and ends with one of --suffixes.
An existing copy must match its manifest hash (it is not downloaded again).
Usage: fetch_hf_v8.py --hf-manifest <json> --hf-commit <sha> --out <dir> --contains S [S ...] --suffixes X [X ...]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = 'yelarys/qwen3-4b-reward-hacking-caft-adapters'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf-manifest', required=True)
    ap.add_argument('--hf-commit', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--contains', nargs='+', required=True)
    ap.add_argument('--suffixes', nargs='+', required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    man = json.loads(Path(a.hf_manifest).read_text())['files']
    out = Path(a.out)
    paths = sorted(p for p in man if all(s in p for s in a.contains) and any(p.endswith(x) for x in a.suffixes))
    print('selected', len(paths), flush=True)
    for p in paths:
        dst = out / p
        want = man[p]['sha256']
        if dst.exists():
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == want, dst
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        f = Path(hf_hub_download(REPO, p, revision=a.hf_commit, local_dir=str(out / '_dl'), token=False))
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        assert got == want, (p, got, want)
        shutil.move(str(f), dst)
        dst.chmod(0o444)
    print('done', len(paths), flush=True)


if __name__ == '__main__':
    main()
