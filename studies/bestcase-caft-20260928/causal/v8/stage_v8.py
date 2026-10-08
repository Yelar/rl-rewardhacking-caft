#!/usr/bin/env python3
"""Hiding study, phases 1-2 (PLAN_phase1-3.md): stage the wave-2 LoRA adapters that the evaluations need on this host.
Each adapter goes to <out>/<lineage>_step<N>/ (adapter_config.json, adapter_model.safetensors, STAGED.json).

The plan names, for each adapter, the run of the official resume chain that produced it (the run of the final chain with
the largest start step below N) and where it is read from:
  this-host  this host's runs_w2q: the original is hashed, copied, and the copy is hashed again;
  gpu-03, gpu-04  that host's runs_w2q (another host): it hashes its own files over ssh, rsync pulls them, this host
             hashes the copy;
  hf         the public Hugging Face backup at a pinned commit: the download must match backups/wave2/MANIFEST.json
             (hashes taken on the training host at upload).
When the plan gives an evaluation manifest (the wave-2 evaluation of that checkpoint), the staged files must also match
the hashes bound there. Sources are only read. Nothing is overwritten: a copy lands in <dir>.partial (our own; an
unfinished one is replaced) and is renamed only after verification.
Usage: stage_v8.py --plan <json> --out <dir> --hf-manifest <MANIFEST.json> --hf-commit <sha>
"""
import argparse
import hashlib
import json
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

B = Path('/scratch/yelarys.yertaiuly/bestcase-step4')
REPO = 'yelarys/qwen3-4b-reward-hacking-caft-adapters'
FILES = ('adapter_config.json', 'adapter_model.safetensors')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def rel(run, step):
    return f'{run}/checkpoints/global_step_{step}/actor/lora_adapter'


def ssh(host, cmd, timeout=900):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f'ssh {host} failed: {r.stderr.strip()[:300]}')
    return r.stdout


def hf_get(path, commit, dl):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path, revision=commit, local_dir=str(dl), token=False))


def expected(e, commit, dl):
    """Adapter hashes bound in the checkpoint's wave-2 evaluation manifest, or None."""
    x = e.get('eval_manifest')
    if not x:
        return None
    if x['host'] == 'hf':
        m = json.loads(hf_get(x['path'], commit, dl).read_text())
    elif x['host'] == socket.gethostname():
        m = json.loads(Path(x['path']).read_text())
    else:
        m = json.loads(ssh(x['host'], 'cat ' + shlex.quote(x['path'])))
    cks = [c for c in m['checkpoints'] if c['name'] == f'{e["lineage"]}_step{e["step"]}']
    if len(cks) != 1:
        raise SystemExit(f'{e["name"]}: evaluation manifest {x["path"]} does not bind this checkpoint')
    return cks[0]['files']


def stage(e, out, hfm, commit, dl):
    dst = out / e['name']
    if dst.exists():
        st = json.loads((dst / 'STAGED.json').read_text())
        if {f: sha(dst / f) for f in FILES} != st['files']:
            raise SystemExit(f'{dst}: staged files changed since staging')
        return 'exists'
    part = Path(str(dst) + '.partial')
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    src = e['source']
    if src == 'this-host':
        s = B / 'runs_w2q' / rel(e['run'], e['step'])
        source_files = {f: sha(s / f) for f in FILES}
        for f in FILES:
            shutil.copyfile(s / f, part / f)
        origin = {'host': socket.gethostname(), 'path': str(s)}
    elif src in ('gpu-03', 'gpu-04'):
        s = B / 'runs_w2q' / rel(e['run'], e['step'])
        lines = ssh(src, f'cd {shlex.quote(str(s))} && sha256sum {" ".join(FILES)}').split('\n')
        source_files = {ln.split()[1]: ln.split()[0] for ln in lines if ln.strip()}
        for f in FILES:
            subprocess.run(['rsync', '-a', f'{src}:{s}/{f}', str(part / f)], check=True, timeout=1800)
        origin = {'host': src, 'path': str(s)}
    elif src == 'hf':
        source_files = {}
        for f in FILES:
            path = f'backups/wave2/runs/{rel(e["run"], e["step"])}/{f}'
            rec = hfm['files'].get(path)
            if not rec:
                raise SystemExit(f'{e["name"]}: {path} is not in the backup manifest')
            source_files[f] = rec['sha256']
            shutil.move(str(hf_get(path, commit, dl)), part / f)
        origin = {'repo': REPO, 'commit': commit, 'path': f'backups/wave2/runs/{rel(e["run"], e["step"])}'}
    else:
        raise SystemExit(f'unknown source {src}')
    files = {f: sha(part / f) for f in FILES}
    if sorted(p.name for p in part.iterdir()) != sorted(FILES):
        raise SystemExit(f'{e["name"]}: unexpected files {sorted(p.name for p in part.iterdir())}')
    if files != source_files:
        raise SystemExit(f'{e["name"]}: copy {files} differs from source {source_files}')
    expect = expected(e, commit, dl)
    if expect is not None and expect != files:
        raise SystemExit(f'{e["name"]}: staged files differ from the evaluation manifest {expect}')
    rec = {'name': e['name'], 'lineage': e['lineage'], 'run': e['run'], 'step': e['step'], 'source': src,
           'origin': origin, 'files': files, 'source_files': source_files,
           'evaluation_manifest': e.get('eval_manifest'), 'matches_evaluation_manifest': expect is not None,
           'staged_on': socket.gethostname(), 'staged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (part / 'STAGED.json').write_text(json.dumps(rec, indent=1, sort_keys=True))
    for p in part.iterdir():
        p.chmod(0o444)
    part.rename(dst)
    return 'staged'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hf-manifest', required=True)
    ap.add_argument('--hf-commit', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hfm = json.loads(Path(a.hf_manifest).read_text())
    plan = json.loads(Path(a.plan).read_text())
    names = [e['name'] for e in plan]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate names in the plan')
    for e in plan:
        if e['name'] != f'{e["lineage"]}_step{e["step"]}':
            raise SystemExit(f'bad name {e["name"]}')
        print(e['name'], stage(e, out, hfm, a.hf_commit, out / '_hf_download'), flush=True)
    print('ALL STAGED', len(plan))


if __name__ == '__main__':
    main()
