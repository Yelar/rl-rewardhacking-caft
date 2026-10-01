"""Back up this host's wave-2 CAFT-RL files (bc4w2q runs and their evaluations) to the public Hugging Face repo, under
backups/wave2/. Hugging Face is the primary durable copy (user decision, 1 Oct 2026); S3 is only for files it does not
take (failed.jsonl).

Uploaded (sources are only read; the upload folder is a tree of real directories and symlinks to the source files):
- runs/<run>/: config.json, BUILD.json, spec.json, q.safetensors, q.provenance.json, SOURCE_INVENTORY.json; every
  closed checkpoint (CAFT_CHECKPOINT.json present): the whole directory (full training state) at steps 0, 100 and 200,
  and the LoRA adapter with its CAFT receipts at the other steps; once the run has ended (unit inactive, and TRAIN EXIT
  logged or train.log quiet for 30 min): train.log, rollouts/ and the checkpoint index files.
- evals/<host>/<eval_w2q|eval_w2q_on>/: every evaluated checkpoint with a SUMMARY.json (all cells), and the files of
  logs/, manifests/ and the top level once quiet for 1 h (or once no bc4w2q unit is active on this host).
Not uploaded: caches, private homes, the reward spool, claims, staged input copies, locks, partial and hidden files.

The repo is public. Before a file is linked into the upload folder its bytes are searched for credential patterns
(Hugging Face, AWS, GitHub, OpenAI/Anthropic, Slack and W&B keys, private keys, secret assignments), for the exact
values of the credentials stored on this host (HF token, netrc, AWS, git and gh credentials; read in memory, never
logged) and for the private literals listed in <workdir>/private_literals.txt. A file with any match is held back:
held.jsonl gets its path and pattern counts, never the match.

Upload: huggingface_hub upload_large_folder (resumable through its metadata) in a child process with 4 workers. The
child is killed after 30 min without progress, and the next attempt switches between the Xet client and plain LFS
transfers (the Xet client can hang). After each completed upload every new file's Hub SHA-256 (LFS/Xet) or git blob
id (small files) and size must equal those computed here from the source; matches go to journal.jsonl. A mismatch
forces a new upload of that file, and after 3 mismatches the file goes to failed.jsonl. A few verified files per pass,
including the largest, are downloaded again in a child process and hashed. Passes repeat every 20 min until no
bc4w2q unit (other than the autoresume watcher) is active on this host and everything eligible is verified.
Usage (as a capped user unit): python hf_backup_w2q.py <repo_id> <workdir> [--stage-only]"""
import hashlib, json, os, random, re, shutil, signal, socket, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO, W = sys.argv[1], Path(sys.argv[2])
STAGE_ONLY = '--stage-only' in sys.argv[3:]
B = Path('/scratch/yelarys.yertaiuly/bestcase-step4')
RUNS = B / 'runs_w2q'
EVAL_ROOTS = ('eval_w2q', 'eval_w2q_on')
FULL = {0, 100, 200}
META = ('config.json', 'BUILD.json', 'spec.json', 'q.safetensors', 'q.provenance.json', 'SOURCE_INVENTORY.json')
SKIP = {'cache', 'home', 'private_home', 'reward_spool', 'claims', 'dryrun_tests', '.cache', '__pycache__'}
SKIP_SUFFIX = ('.lock', '.tmp', '.part', '.pyc', '.swp')
HOST = socket.gethostname()
DEST = 'backups/wave2'
STAGE = W / 'stage'
UPMETA = STAGE / '.cache' / 'huggingface' / 'upload'
STATE, JOURNAL, HELD, FAILED, LOG = (W / n for n in ('state.json', 'journal.jsonl', 'held.jsonl', 'failed.jsonl',
                                                     'backup.log'))
PASS_SLEEP, RETRY_SLEEP, NO_PROGRESS, RUN_QUIET, EVAL_QUIET, SUMMARY_QUIET = 1200, 120, 1800, 1800, 3600, 600
PATTERNS = {
    'hf_token': r'hf_[A-Za-z0-9]{30,}',
    'aws_key_id': r'(AKIA|ASIA)[0-9A-Z]{16}',
    'github_token': r'(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
    'openai_anthropic_key': r'\bsk-(proj-|ant-)?[A-Za-z0-9_-]{20,}',
    'slack_token': r'xox[abprs]-[A-Za-z0-9-]{10,}',
    'wandb_key': r'wandb_v1_[A-Za-z0-9_]{20,}',
    'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY',
    'secret_assignment': '(WANDB_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY)'
                         "[\"' ]*[=:] *[\"']?[A-Za-z0-9_-]{8,}",
}
GREP_ENV = {'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}
# Every Hub call runs in a child process with a hard limit: neither the Xet client nor plain HTTP calls time out.
CHILD = '''import json, sys
from huggingface_hub import HfApi, hf_hub_download
if sys.argv[1] == 'upload':
    HfApi().upload_large_folder(repo_id=sys.argv[2], repo_type='model', folder_path=sys.argv[3], num_workers=4,
                                print_report=True, print_report_every=300)
elif sys.argv[1] == 'info':
    paths, out = json.load(sys.stdin), {}
    for i in range(0, len(paths), 100):
        for x in HfApi().get_paths_info(sys.argv[2], paths[i:i + 100], repo_type='model'):
            lfs = getattr(x, 'lfs', None)
            out[x.path] = {'size': getattr(x, 'size', None), 'blob_id': getattr(x, 'blob_id', None),
                           'lfs_sha256': lfs.sha256 if lfs else None}
    print(json.dumps(out))
else:
    print(hf_hub_download(sys.argv[2], sys.argv[4], local_dir=sys.argv[3]))
'''


def iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(msg):
    line = f'{iso()} {HOST} {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def append(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {'files': {}}


def save_state(state):
    tmp = W / 'state.json.tmp'
    tmp.write_text(json.dumps(state, sort_keys=True))
    os.replace(tmp, STATE)


def host_secrets():
    """Exact credential values stored on this host, kept in memory only."""
    vals, home = set(), Path.home()
    for p in (home / '.cache/huggingface/token', home / '.huggingface/token',
              Path(os.environ.get('HF_HOME', home / '.cache/huggingface')) / 'token'):
        try:
            vals.add(p.read_text().strip())
        except OSError:
            pass
    try:
        t = (home / '.netrc').read_text().split()
        vals.update(t[i + 1] for i in range(len(t) - 1) if t[i] == 'password')
    except OSError:
        pass
    try:
        for line in (home / '.aws/credentials').read_text().splitlines():
            k, _, v = line.partition('=')
            if k.strip().lower() in ('aws_access_key_id', 'aws_secret_access_key', 'aws_session_token'):
                vals.add(v.strip())
    except OSError:
        pass
    try:
        for line in (home / '.git-credentials').read_text().splitlines():
            m = re.match(r'\w+://[^:/@]+:([^@]+)@', line.strip())
            if m:
                vals.add(m.group(1))
    except OSError:
        pass
    try:
        for line in (home / '.config/gh/hosts.yml').read_text().splitlines():
            if 'oauth_token:' in line:
                vals.add(line.split(':', 1)[1].strip())
    except OSError:
        pass
    return [v.encode() for v in vals if len(v) >= 16]


def ere(s):
    return re.sub(r'([.\[\]()*+?{}|^$\\])', r'\\\1', s)


def literals():
    p = W / 'private_literals.txt'
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return {f'private_literal_{i}': ere(l) for i, l in enumerate(lines)}


SECRETS = host_secrets()
ALL_PATTERNS = dict(PATTERNS, **literals())


def grep_hits(p):
    """Pattern counts (matching lines) for one file; {} when clean. Fails closed on any grep error."""
    args = [a for v in ALL_PATTERNS.values() for a in ('-e', v)]
    r = subprocess.run(['grep', '-a', '-q', '-E', *args, '--', str(p)], env=GREP_ENV, capture_output=True)
    if r.returncode == 1:
        return {}
    if r.returncode != 0:
        raise RuntimeError(f'grep rc={r.returncode} on {p}: {r.stderr[-200:]!r}')
    hits = {}
    for name, rx in ALL_PATTERNS.items():
        c = subprocess.run(['grep', '-a', '-c', '-E', '-e', rx, '--', str(p)], env=GREP_ENV, capture_output=True,
                           text=True)
        if c.returncode == 0:
            hits[name] = int(c.stdout.split()[0])
    return hits or {'unattributed': 1}


def inspect(src):
    """SHA-256, git blob id, exact-credential search and pattern scan of one closed file; None if it changed."""
    st = src.stat()
    size = st.st_size
    h, g = hashlib.sha256(), hashlib.sha1(b'blob %d\0' % size)
    exact, keep, tail = 0, max(map(len, SECRETS), default=1) - 1, b''
    with open(src, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
            g.update(b)
            if SECRETS:
                buf = tail + b
                exact += sum(s in buf for s in SECRETS)
                tail = buf[-keep:] if keep else b''
    st2 = src.stat()
    if (st2.st_size, st2.st_mtime_ns) != (size, st.st_mtime_ns):
        return None
    hits = grep_hits(src)
    if exact:
        hits['host_credential_value'] = exact
    return {'bytes': size, 'mtime_ns': st.st_mtime_ns, 'sha256': h.hexdigest(), 'git_sha1': g.hexdigest(), 'hits': hits}


def active_units():
    """Active bc4w2q units on this host other than the autoresume watcher (fail closed: unknown counts as active)."""
    r = subprocess.run(['systemctl', '--user', 'list-units', '--type=service', '--state=active', '--no-legend',
                        '--plain', 'bc4w2q-*'], capture_output=True, text=True)
    if r.returncode != 0:
        return ['<systemctl failed>']
    return [l.split()[0] for l in r.stdout.splitlines() if l.strip() and not l.startswith('bc4w2q-autoresume')]


def unit_active(name):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', name]).returncode == 0


def walk_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not d.startswith('.'))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if fn.startswith('.') or fn.endswith(SKIP_SUFFIX) or p.is_symlink() or not p.is_file():
                continue
            yield p


def run_ended(d, now):
    log_ = d / 'train.log'
    if unit_active(d.name) or not log_.is_file():
        return False
    with open(log_, 'rb') as f:
        f.seek(max(0, log_.stat().st_size - 50000))
        tail = f.read().decode(errors='replace')
    return 'TRAIN EXIT' in tail or now - log_.stat().st_mtime > RUN_QUIET


def eligible(now, quiet_host):
    out = {}
    for d in sorted(RUNS.glob('bc4w2q-*')):
        if not d.is_dir() or not (d / 'BUILD.json').is_file():
            continue
        base, ck = f'{DEST}/runs/{d.name}', d / 'checkpoints'
        files = [d / m for m in META]
        for c in sorted(ck.glob('global_step_*')) if ck.is_dir() else []:
            m = re.fullmatch(r'global_step_(\d+)', c.name)
            if not m or not (c / 'CAFT_CHECKPOINT.json').is_file():
                continue
            if int(m.group(1)) in FULL:
                files += list(walk_files(c))
            else:
                files += [c / 'CAFT_CHECKPOINT.json', c / 'actor' / 'fsdp_config.json']
                files += sorted((c / 'actor').glob('caft_rank_*.json')) + list(walk_files(c / 'actor' / 'lora_adapter'))
        if run_ended(d, now):
            files += [d / 'train.log', ck / 'latest_checkpointed_iteration.txt'] + sorted(ck.glob('interrupted_step_*.json'))
            files += list(walk_files(d / 'rollouts'))
        for p in files:
            if p.is_file() and not p.is_symlink():
                out[f'{base}/{p.relative_to(d).as_posix()}'] = p
    for root in EVAL_ROOTS:
        r = B / root
        if not r.is_dir():
            continue
        base = f'{DEST}/evals/{HOST}/{root}'
        for e in sorted(r.iterdir()):
            if e.name in SKIP or e.name.startswith('.') or e.is_symlink():
                continue
            if e.is_file():
                files = [e] if quiet_host or now - e.stat().st_mtime > EVAL_QUIET else []
            elif (e / 'SUMMARY.json').is_file():
                files = list(walk_files(e)) if now - (e / 'SUMMARY.json').stat().st_mtime > SUMMARY_QUIET else []
            elif e.name in ('logs', 'manifests'):
                files = [p for p in walk_files(e) if quiet_host or now - p.stat().st_mtime > EVAL_QUIET]
            else:
                files = []
            for p in files:
                if not p.name.endswith(SKIP_SUFFIX):
                    out[f'{base}/{p.relative_to(r).as_posix()}'] = p
    return out


def link(dst, src):
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = W / 'tmp' / 'link'
    tmp.parent.mkdir(exist_ok=True)
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(str(src), tmp)
    os.replace(tmp, dst)


def safe_inspect(src):
    try:
        return inspect(src)
    except Exception as e:  # fail closed: the file is not linked this pass
        log(f'inspect failed, not staged this pass: {src}: {type(e).__name__}: {str(e)[:200]}')
        return None


def stage(state, now, quiet_host):
    """Inspect new or changed eligible files; link clean ones into the upload folder. Returns (inspected, held)."""
    want = eligible(now, quiet_host)
    todo = []
    for rp, src in want.items():
        try:
            st = src.stat()
        except OSError:
            continue
        rec = state['files'].get(rp)
        if rec and rec['src'] == str(src) and rec['bytes'] == st.st_size and rec['mtime_ns'] == st.st_mtime_ns:
            continue
        todo.append((rp, src))
    inspected = held = 0
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(todo), 200):
            batch = todo[i:i + 200]
            for (rp, src), info in zip(batch, ex.map(lambda t: safe_inspect(t[1]), batch)):
                if info is None:
                    continue
                inspected += 1
                old = state['files'].get(rp, {})
                rec = dict(info, src=str(src), verified=False, fails=old.get('fails', 0), held=bool(info['hits']))
                state['files'][rp] = rec
                dst = STAGE / rp
                if rec['held']:
                    held += 1
                    append(HELD, {'path': rp, 'src': str(src), 'hits': info['hits'], 'time': iso()})
                    if dst.is_symlink():
                        dst.unlink()
                    continue
                link(dst, src)
            save_state(state)
    return inspected, held


def pending(state):
    return [rp for rp, r in state['files'].items() if not (r['verified'] or r['held'] or r.get('failed'))]


def progress_mark():
    n = m = 0
    for dp, _, fns in os.walk(UPMETA):
        for fn in fns:
            n += 1
            m = max(m, os.stat(os.path.join(dp, fn)).st_mtime_ns)
    return n, m


def child_env(use_xet):
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', HF_XET_CACHE=str(W / 'xet_cache'),
               HF_XET_CHUNK_CACHE_SIZE_BYTES='0', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    if not use_xet:
        env['HF_HUB_DISABLE_XET'] = '1'
    return env


def upload(use_xet):
    """One upload_large_folder run; returns its exit code, or 'stalled' after NO_PROGRESS s without progress."""
    (W / 'logs').mkdir(exist_ok=True)
    with open(W / 'logs' / f'upload_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.log', 'ab') as out:
        p = subprocess.Popen([sys.executable, '-B', '-c', CHILD, 'upload', REPO, str(STAGE)], env=child_env(use_xet),
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        last, t_last = progress_mark(), time.time()
        while p.poll() is None:
            time.sleep(60)
            mark = progress_mark()
            if mark != last:
                last, t_last = mark, time.time()
            elif time.time() - t_last > NO_PROGRESS:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                return 'stalled'
    return p.returncode


def hub_info(paths):
    c = subprocess.run([sys.executable, '-B', '-c', CHILD, 'info', REPO], input=json.dumps(paths), text=True,
                       env=child_env(True), capture_output=True, timeout=600 + len(paths))
    if c.returncode:
        raise RuntimeError(f'info child rc={c.returncode}: {c.stderr.strip()[-300:]}')
    return json.loads(c.stdout.strip().splitlines()[-1])


def verify(state):
    todo = pending(state)
    ok = bad = 0
    for i in range(0, len(todo), 1000):
        chunk = todo[i:i + 1000]
        try:
            info = hub_info(chunk)
        except Exception as e:
            log(f'Hub file info failed for {len(chunk)} files (retried next pass): {type(e).__name__}: {str(e)[:200]}')
            continue
        for rp in chunk:
            r, x = state['files'][rp], info.get(rp)
            lfs = x and x['lfs_sha256']
            good = x is not None and x['size'] == r['bytes'] and (
                (lfs is not None and lfs == r['sha256']) or (lfs is None and x['blob_id'] == r['git_sha1']))
            if good:
                r['verified'], r['lfs'] = True, lfs is not None
                append(JOURNAL, {'path': rp, 'src': r['src'], 'host': HOST, 'bytes': r['bytes'], 'sha256': r['sha256'],
                                 'git_sha1': r['git_sha1'], 'lfs': r['lfs'], 'hub_verified': True, 'time': iso()})
                ok += 1
                continue
            bad += 1
            r['fails'] = r.get('fails', 0) + 1
            md = UPMETA / f'{rp}.metadata'
            if md.exists():
                md.unlink()  # makes the next upload_large_folder run hash and send this file again
            if r['fails'] >= 3:
                r['failed'] = True
                append(FAILED, {'path': rp, 'src': r['src'], 'bytes': r['bytes'], 'sha256': r['sha256'], 'time': iso(),
                                'hub': x})
                log(f'HF FAILED 3x, left for the S3 fallback: {rp}')
    save_state(state)
    return ok, bad


def spot_check(state, k=3):
    """Download a few verified files again (the largest not yet checked and random others) and hash them."""
    ver = [rp for rp, r in state['files'].items() if r['verified'] and 'spot' not in r]
    if not ver:
        return
    picks = list(dict.fromkeys([max(ver, key=lambda rp: state['files'][rp]['bytes'])] + random.sample(ver, min(k - 1, len(ver)))))
    for rp in picks:
        d = W / 'spot'
        shutil.rmtree(d, ignore_errors=True)
        try:
            c = subprocess.run([sys.executable, '-B', '-c', CHILD, 'download', REPO, str(d), rp], env=child_env(True),
                               capture_output=True, text=True, timeout=1800)
            got = None
            if c.returncode == 0:
                h = hashlib.sha256()
                with open(c.stdout.strip().splitlines()[-1], 'rb') as f:
                    for b in iter(lambda: f.read(1 << 24), b''):
                        h.update(b)
                got = h.hexdigest()
        except Exception as e:
            got = f'error {type(e).__name__}'
        finally:
            shutil.rmtree(d, ignore_errors=True)
        ok = got == state['files'][rp]['sha256']
        state['files'][rp]['spot'] = ok
        append(JOURNAL, {'spot_check': rp, 'bytes': state['files'][rp]['bytes'], 'download_sha256_matches': ok,
                         'time': iso()})
        log(f'spot check {"OK" if ok else "FAILED"}: {rp} ({state["files"][rp]["bytes"] / 1e6:.0f} MB)')
    save_state(state)


def summary(state):
    cats = {}
    for rp, r in state['files'].items():
        parts = rp.split('/')
        cat = parts[2] if parts[2] == 'evals' else ('checkpoint' if 'checkpoints' in parts else
                                                    'rollouts' if 'rollouts' in parts else 'run files')
        c = cats.setdefault(cat, {'files': 0, 'bytes': 0, 'held': 0, 'verified': 0})
        c['files'] += 1
        c['bytes'] += r['bytes']
        c['held'] += r['held']
        c['verified'] += r['verified']
    return cats


def main():
    for d in (W, STAGE):
        d.mkdir(parents=True, exist_ok=True)
    log(f'start: repo {REPO}, workdir {W}, stage-only {STAGE_ONLY}, {len(SECRETS)} host credential values, '
        f'{len(ALL_PATTERNS)} patterns')
    use_xet, idle_fail = True, 0
    while True:
        state, now = load_state(), time.time()
        active = active_units()
        inspected, held = stage(state, now, not active)
        todo = pending(state)
        log(f'pass: inspected {inspected} (held {held}); pending upload {len(todo)}; active units {len(active)}; '
            f'by category {json.dumps(summary(state))}')
        if STAGE_ONLY:
            break
        if todo:
            rc = upload(use_xet)
            log(f'upload_large_folder ({"xet" if use_xet else "lfs"}) finished: {rc}')
            if rc == 0:
                ok, bad = verify(state)
                log(f'verified {ok} files against the Hub; {bad} not matching yet')
                spot_check(state)
                idle_fail = 0
            else:
                use_xet = not use_xet
                idle_fail += 1
                if idle_fail >= 3:
                    log('HF NOT WORKING: 3 upload runs in a row failed or stalled')
            if pending(state):
                time.sleep(min(RETRY_SLEEP * 2 ** idle_fail, PASS_SLEEP))
                continue
        if not active and not inspected and not pending(state):
            log('ALL DONE: every eligible file is verified on the Hub and no bc4w2q unit is active')
            break
        time.sleep(PASS_SLEEP)


if __name__ == '__main__':
    main()
