"""Merge the per-host journals of hf_backup_w2q.py into backups/wave2/MANIFEST.json and upload it, then re-check every
listed file on the Hub from this independent process (size, and SHA-256 for LFS/Xet files or git blob id otherwise).
The manifest keeps the latest Hub-verified record per path: size, SHA-256, git blob id, LFS flag, run and the
download spot checks. Host paths are left out (the repo path already names the run or evaluation).
Usage: python hf_backup_manifest.py <repo_id> <journal.jsonl>..."""
import json, sys, time
from collections import Counter
from huggingface_hub import HfApi

REPO, JOURNALS = sys.argv[1], sys.argv[2:]
DEST = 'backups/wave2'
api = HfApi()
files, spots = {}, {}
for j in JOURNALS:
    for line in open(j):
        r = json.loads(line)
        if r.get('hub_verified'):
            files[r['path']] = {'bytes': r['bytes'], 'sha256': r['sha256'], 'git_sha1': r['git_sha1'], 'lfs': r['lfs'],
                                'verified': r['time']}
        elif 'spot_check' in r:
            spots[r['spot_check']] = r['download_sha256_matches']
runs = Counter(p.split('/')[3] for p in files if p.startswith(f'{DEST}/runs/'))
manifest = {
    'repo': REPO, 'generated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'base_model': 'Qwen/Qwen3-4B', 'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
    'files_total': len(files), 'bytes_total': sum(f['bytes'] for f in files.values()),
    'files_per_run': dict(sorted(runs.items())),
    'download_spot_checks': {'files': len(spots), 'matching': sum(spots.values()), 'paths': spots},
    'files': {p: files[p] for p in sorted(files)},
}
with open('BACKUP_MANIFEST.json', 'w') as f:
    json.dump(manifest, f, indent=1)
api.upload_file(path_or_fileobj='BACKUP_MANIFEST.json', path_in_repo=f'{DEST}/MANIFEST.json', repo_id=REPO,
                commit_message='Add the wave-2 backup manifest')
paths, info = sorted(files), {}
for i in range(0, len(paths), 100):
    for x in api.get_paths_info(REPO, paths[i:i + 100]):
        info[x.path] = x
ok, bad = 0, []
for p in paths:
    x, f = info.get(p), files[p]
    lfs = getattr(x, 'lfs', None)
    if x is not None and x.size == f['bytes'] and ((lfs and lfs.sha256 == f['sha256']) or (not lfs and x.blob_id == f['git_sha1'])):
        ok += 1
    else:
        bad.append(p)
print(f'manifest: {len(files)} files, {manifest["bytes_total"] / 1e9:.1f} GB, {len(runs)} run folders; '
      f'Hub re-check: {ok} match, {len(bad)} problems; download spot checks {sum(spots.values())}/{len(spots)} match')
for p in bad[:20]:
    print('PROBLEM', p)
sys.exit(1 if bad else 0)
