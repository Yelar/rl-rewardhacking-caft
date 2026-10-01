"""Upload the step-100/150/200 LoRA adapters of bc4w2q lineages on this host to the public Hugging Face repo once each
lineage has finished training, with the conventions of hf_upload_host.py: wave2/<arm>-seed<N>/step_<S>/ holding
adapter_config.json and adapter_model.safetensors, one journal line per folder (dst, run, arm, seed, step, files
with bytes and SHA-256, time) in this host's journal, which hf_finalize.py merges into MANIFEST.json.

A lineage counts as finished as in eval_w2q_kit/eval_after_train.sh: a closed global_step_200, and the latest run of
the lineage logged 'TRAIN EXIT ... rc=0' and 'RELEASE VERIFIED' and its unit is no longer active. Each step's source
is the latest closed global_step_<S> among the run and its resumes (the rule of make_eval_manifest.py), so the
uploaded adapter is the evaluated one. After each upload the Hub's SHA-256 and size are checked against the local
file. Read-only on the run folders.
Usage: python hf_upload_after_train.py <repo_id> <journal.jsonl> <lineage>..."""
import hashlib, json, os, re, socket, subprocess, sys, time
from pathlib import Path
from huggingface_hub import HfApi

CHILD = '''import sys
from huggingface_hub import HfApi
repo, src, dst = sys.argv[1:4]
HfApi().upload_folder(repo_id=repo, repo_type='model', folder_path=src, path_in_repo=dst,
                      allow_patterns=['adapter_config.json', 'adapter_model.safetensors'], commit_message=f'Add {dst}')
'''

REPO, JOURNAL, LINEAGES = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
RUNS = Path('/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q')
FILES = ('adapter_config.json', 'adapter_model.safetensors')
ARMS = {'nonezero': 'baseline', 'jointb19': 'joint', 'addb19': 'adding', 'randomb19': 'random'}
STEPS = (100, 150, 200)
api = HfApi()
host = socket.gethostname()


def log(msg):
    print(f'{time.strftime("%H:%M:%SZ", time.gmtime())} {host} {msg}', flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def runs(lineage):
    return [d for d in RUNS.iterdir() if re.fullmatch(re.escape(lineage) + r'(-resume\d+[a-z]?)*', d.name)]


def closed(lineage, step):
    mark = lambda d: d / 'checkpoints' / f'global_step_{step}' / 'CAFT_CHECKPOINT.json'
    cands = sorted((d for d in runs(lineage) if mark(d).is_file()), key=lambda d: mark(d).stat().st_mtime)
    return cands[-1] if cands else None


def finished(lineage):
    logs = [d / 'train.log' for d in runs(lineage) if (d / 'train.log').is_file()]
    if not logs or closed(lineage, 200) is None:
        return False
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    text = latest.read_text(errors='replace')
    active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', latest.parent.name]).returncode == 0
    return re.search(r'TRAIN EXIT .* rc=0', text) is not None and 'RELEASE VERIFIED' in text and not active


def done():
    if not JOURNAL.exists():
        return set()
    return {json.loads(l)['dst'] for l in JOURNAL.read_text().splitlines() if l.strip()}


def upload(lineage, step):
    m = re.fullmatch(r'bc4w2q-(\w+?)-s(\d+)', lineage)
    arm, seed = ARMS[m.group(1)], int(m.group(2))
    dst = f'wave2/{arm}-seed{seed}/step_{step}'
    if dst in done():
        return
    run = closed(lineage, step)
    src = run / 'checkpoints' / f'global_step_{step}' / 'actor' / 'lora_adapter'
    files = {f: {'bytes': (src / f).stat().st_size, 'sha256': sha(src / f)} for f in FILES}
    for attempt in range(6):
        try:
            # The Xet client can hang without a timeout: upload in a child process with a hard limit, and fall back
            # to plain LFS transfers after the first failed attempt.
            env = dict(os.environ, **({'HF_HUB_DISABLE_XET': '1'} if attempt else {}))
            child = subprocess.run([sys.executable, '-B', '-c', CHILD, REPO, str(src), dst], env=env, timeout=1200,
                                   capture_output=True, text=True)
            if child.returncode:
                raise RuntimeError(f'upload child rc={child.returncode}: {child.stderr.strip()[-300:]}')
            info = {p.path: p for p in api.get_paths_info(REPO, [f'{dst}/{f}' for f in FILES], expand=True)}
            bad = [f for f in FILES if f'{dst}/{f}' not in info or info[f'{dst}/{f}'].size != files[f]['bytes'] or
                   (info[f'{dst}/{f}'].lfs and info[f'{dst}/{f}'].lfs.sha256 != files[f]['sha256']) or
                   (f.endswith('.safetensors') and not info[f'{dst}/{f}'].lfs)]
            if bad:
                raise RuntimeError(f'Hub copy differs for {bad}')
            break
        except Exception as e:
            log(f'retry {attempt + 1} for {dst}: {type(e).__name__}: {str(e)[:200]}')
            time.sleep(60 * (attempt + 1))
    else:
        log(f'FAILED {dst}')
        return
    rec = {'dst': dst, 'run': run.name, 'arm': arm, 'seed': seed, 'step': step, 'files': files,
           'hub_verified': True, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(JOURNAL, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    log(f'uploaded {dst} from {run.name} (Hub SHA-256 and size match)')


log(f'waiting for {LINEAGES}; {len(done())} folders already in the journal')
pending = list(LINEAGES)
while pending:
    for lineage in list(pending):
        if finished(lineage):
            log(f'{lineage} finished training; uploading steps {STEPS}')
            for step in STEPS:
                upload(lineage, step)
            pending.remove(lineage)
    if pending:
        time.sleep(300)
log('all lineages uploaded')
