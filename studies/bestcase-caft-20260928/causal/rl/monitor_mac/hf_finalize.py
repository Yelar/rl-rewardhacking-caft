"""Merge the per-host upload journals into MANIFEST.json, upload it with the model card, then check every
uploaded file on the Hub against the SHA-256 computed from the source before upload.
Usage: python hf_finalize.py <repo_id> <README.md> <journal.jsonl>..."""
import json, sys, time
from huggingface_hub import HfApi
REPO, README, JOURNALS = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi()
entries = {}
for j in JOURNALS:
    for line in open(j):
        if line.strip():
            r = json.loads(line); entries[r['dst']] = r
manifest = {'repo': REPO, 'base_model': 'Qwen/Qwen3-4B', 'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
            'generated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'adapters': [entries[k] for k in sorted(entries)]}
with open('MANIFEST.json', 'w') as f: json.dump(manifest, f, indent=1)
api.upload_file(path_or_fileobj='MANIFEST.json', path_in_repo='MANIFEST.json', repo_id=REPO, commit_message='Add manifest')
api.upload_file(path_or_fileobj=README, path_in_repo='README.md', repo_id=REPO, commit_message='Add model card')
# Independent check: the Hub's recorded SHA-256 (LFS/Xet oid) and size for every weight file, against the journal.
paths = [f"{k}/{f}" for k, r in entries.items() for f in r['files']]
info = {}
for i in range(0, len(paths), 50):
    for p in api.get_paths_info(REPO, paths[i:i + 50], expand=True):
        info[p.path] = p
ok = bad = 0
for k, r in entries.items():
    for f, meta in r['files'].items():
        p = info.get(f"{k}/{f}")
        hub_sha = p.lfs.sha256 if (p is not None and p.lfs) else None
        if p is None: bad += 1; print('MISSING on Hub:', k, f); continue
        if p.size != meta['bytes'] or (hub_sha and hub_sha != meta['sha256']): bad += 1; print('MISMATCH:', k, f); continue
        if hub_sha is None and f.endswith('.safetensors'): bad += 1; print('no SHA-256 on Hub for', k, f); continue
        ok += 1
print(f'adapters in manifest: {len(entries)}; files matching the Hub: {ok}; problems: {bad}')
