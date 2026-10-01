"""Stage the inputs of removal-on evaluations on this host (system python3, stdlib only).

For each (lineage, step) of the plan: find the closed global_step_<step> on the lineage's training host with the
rule of eval_w2q_kit/make_eval_manifest.py (the latest closed candidate among the run and its resumes), copy its
LoRA adapter (adapter_config.json, adapter_model.safetensors), and once per lineage the run's CAFT spec and q file
(a resume has no spec.json: its config's caft spec must equal the lineage's spec.json). Sources are only read. Every copy must match the SHA-256 computed on the source host, and the adapter must match the
hashes recorded by the removal-off evaluation of the same checkpoint when the plan gives them ('expect'), so the two
evaluations load the same bytes. New files go only under eval_on_inputs/<lineage>/ and are made read-only.

The staged spec.json is the run's spec.json with q.file.path pointing at the staged q.safetensors (same bytes);
spec.orig.json is the run's file, verbatim. STAGED.json (rewritten atomically) records sources and hashes;
STAGED.jsonl keeps every staging event.
A plan item with "adapter_only": true stages only the adapter (any arm), for a removal-off re-run from the staged
copy (make_manifest_off.py).
Usage: python3 stage_on_inputs.py <plan.json: [{"lineage", "step", "host", "expect": {file: sha256} | null,
                                              "adapter_only": bool (optional)}]>"""
import copy, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

B = Path('/scratch/yelarys.yertaiuly/bestcase-step4')
ROOT = B / 'eval_on_inputs'
FILES = ('adapter_config.json', 'adapter_model.safetensors')
ME = socket.gethostname()

PROBE = r'''
import hashlib, json, re, sys
from pathlib import Path
B = Path('/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q')
lineage, step = sys.argv[1], int(sys.argv[2])
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
runs = [d for d in B.iterdir() if re.fullmatch(re.escape(lineage) + r'(-resume\d+[a-z]?)*', d.name)]
mark = lambda d: d / 'checkpoints' / f'global_step_{step}' / 'CAFT_CHECKPOINT.json'
cands = sorted((d for d in runs if mark(d).is_file()), key=lambda d: mark(d).stat().st_mtime)
if not cands:
    print(json.dumps({'error': f'no closed global_step_{step} for {lineage}'})); sys.exit()
run = cands[-1]; ad = run / 'checkpoints' / f'global_step_{step}' / 'actor' / 'lora_adapter'
# Resumes have no spec.json of their own: their config carries the lineage's spec (checked equal below).
spec_file = run / 'spec.json' if (run / 'spec.json').is_file() else B / lineage / 'spec.json'
spec = json.loads(spec_file.read_text())
cfg = json.loads((run / 'config.json').read_text())
q = Path(spec['q']['file']['path']) if spec.get('q') else None  # the baseline has no direction
print(json.dumps({'run_dir': str(run), 'candidates': [str(c) for c in cands], 'adapter': str(ad),
    'files': {f: sha(ad / f) for f in ('adapter_config.json', 'adapter_model.safetensors')},
    'spec_path': str(spec_file), 'spec_sha256': sha(spec_file), 'spec': spec,
    'spec_matches_config': cfg['actor_rollout_ref']['model']['caft'] == spec,
    'q_path': q and str(q), 'q_sha256': q and sha(q), 'q_size': q and q.stat().st_size}))
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    sys.exit(f'STAGING FAILED: {msg}')


def remote(host, *argv, stdin=None):
    cmd = list(argv) if host == ME else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', host, *argv]
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=900)
    if r.returncode:
        fail(f'{host}: {" ".join(argv)[:80]}: rc={r.returncode} {r.stderr[-400:]}')
    return r.stdout


def fetch(host, src, dst, want):
    """Copy one source file to dst (via dst.part), check its SHA-256, make it read-only."""
    if dst.exists():
        if sha(dst) != want:
            fail(f'{dst} exists with different bytes')
        return 'present'
    part = dst.with_name(dst.name + '.part')
    if part.exists():
        part.unlink()
    cmd = ['cp', '--', src, str(part)] if host == ME else \
        ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', f'{host}:{src}', str(part)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode:
        fail(f'copy {host}:{src}: {r.stderr[-400:]}')
    got = sha(part)
    if got != want:
        part.unlink()
        fail(f'copy of {host}:{src} has SHA-256 {got}, source {want}')
    os.chmod(part, 0o444)
    os.replace(part, dst)
    return 'copied'


def without_qpath(spec):
    s = copy.deepcopy(spec)
    s['q']['file'].pop('path')
    return s


def write_json_atomic(p, obj):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, p)


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    for item in plan:
        lineage, step, host = item['lineage'], int(item['step']), item['host']
        probe = json.loads(remote(host, 'python3', '-', lineage, str(step), stdin=PROBE))
        if 'error' in probe:
            fail(f'{lineage} step {step} on {host}: {probe["error"]}')
        if not probe['spec_matches_config']:
            fail(f'{probe["spec_path"]} differs from the run config caft spec')
        if item.get('expect') and item['expect'] != probe['files']:
            fail(f'{lineage} step {step}: source adapter {probe["files"]} differs from the removal-off record {item["expect"]}')
        spec = probe['spec']
        adapter_only = item.get('adapter_only') is True  # for removal-off re-runs of any arm: no spec or q staged
        if not adapter_only and not (spec.get('layers') and float(spec.get('strength', 0)) > 0):
            fail(f'{lineage} is not a projection arm')
        if not adapter_only and (spec['q']['file']['sha256'] != probe['q_sha256'] or
                                 spec['q']['file']['size_bytes'] != probe['q_size']):
            fail(f'{lineage}: q file on {host} does not match its spec')
        S = ROOT / lineage
        S.mkdir(parents=True, exist_ok=True)
        st = json.loads((S / 'STAGED.json').read_text()) if (S / 'STAGED.json').exists() else \
            {'lineage': lineage, 'checkpoints': {}}
        events = []
        if adapter_only:
            pass
        elif 'spec' not in st:
            events.append(fetch(host, probe['spec_path'], S / 'spec.orig.json', probe['spec_sha256']))
            events.append(fetch(host, probe['q_path'], S / 'q.safetensors', probe['q_sha256']))
            local = copy.deepcopy(spec)
            local['q']['file']['path'] = str(S / 'q.safetensors')
            (S / 'spec.json.tmp').write_text(json.dumps(local, indent=1, sort_keys=True))
            os.chmod(S / 'spec.json.tmp', 0o444)
            os.replace(S / 'spec.json.tmp', S / 'spec.json')
            st['spec'] = {'sha256': sha(S / 'spec.json'), 'original_sha256': probe['spec_sha256'],
                          'source_host': host, 'source_path': probe['spec_path'],
                          'q': {'sha256': probe['q_sha256'], 'size_bytes': probe['q_size'], 'source_path': probe['q_path']}}
        else:
            # Resumes carry their own spec.json; everything but the q path must be the lineage's staged spec.
            staged_orig = json.loads((S / 'spec.orig.json').read_text())
            if without_qpath(staged_orig) != without_qpath(spec) or st['spec']['q']['sha256'] != probe['q_sha256']:
                fail(f'{lineage}: spec of {probe["run_dir"]} differs from the staged spec')
        ck = S / f'step_{step}' / 'lora_adapter'
        ck.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            events.append(fetch(host, f'{probe["adapter"]}/{f}', ck / f, probe['files'][f]))
        st['checkpoints'][str(step)] = {'host': host, 'run_dir': probe['run_dir'], 'candidates': probe['candidates'],
                                        'adapter_path': probe['adapter'], 'files': probe['files'],
                                        'removal_off_record_matches': bool(item.get('expect')),
                                        'staged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        write_json_atomic(S / 'STAGED.json', st)
        with open(S / 'STAGED.jsonl', 'a') as j:
            j.write(json.dumps({'step': step, 'events': events, **st['checkpoints'][str(step)]}, sort_keys=True) + '\n')
        print(f'{lineage} step {step} from {host}: {events} (removal-off record match: {bool(item.get("expect"))})',
              flush=True)
    print('STAGING DONE', flush=True)


if __name__ == '__main__':
    main()
