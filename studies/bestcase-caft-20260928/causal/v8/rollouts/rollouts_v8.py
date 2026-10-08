#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, step 4): test writing and cheating in the wave-2 training rollouts, by step.

Each lineage is read along its official resume chain: step N comes from the chain run with the largest start step below
N (a resumed run's start is its last '-resumeN' suffix). Files are rollouts/<N>.jsonl (the step's 256 training answers
with the reward function's labels); a chain segment is either a local run directory or a directory of Hugging Face
backup copies checked against backups/wave2/MANIFEST.json (fetch mode).
Per step: answers; has_test (the repository's own test-function detector, src/evaluate: the answer's parsed code
defines run_tests and the function holds no __main__ block); harmful test (is_test_modification_harmful), strict and
loose hacking, accuracy (eq_correct), mean reward and answer length in characters. The detector only parses code.
Usage: rollouts_v8.py fetch --runs RUN:FIRST-LAST,... --out <dir> --hf-manifest <json> --hf-commit <sha>
       rollouts_v8.py stats --source <scoring snapshot> --chain LINEAGE=START:DIR,START:DIR;... --out <json> [--workers 8]
                            [--last 200]  (steps 1..last; a pilot stopped at step 50 uses --last 50)
"""
import argparse
import hashlib
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

REPO = 'yelarys/qwen3-4b-reward-hacking-caft-adapters'


def fetch(a):
    from huggingface_hub import hf_hub_download
    man = json.loads(Path(a.hf_manifest).read_text())['files']
    out = Path(a.out)
    for part in a.runs.split(','):
        run, rng = part.split(':')
        lo, hi = (int(x) for x in rng.split('-'))
        d = out / run
        d.mkdir(parents=True, exist_ok=True)
        for n in range(lo, hi + 1):
            dst = d / f'{n}.jsonl'
            path = f'backups/wave2/runs/{run}/rollouts/{n}.jsonl'
            want = man[path]['sha256']
            if dst.exists():
                assert hashlib.sha256(dst.read_bytes()).hexdigest() == want, dst
                continue
            p = Path(hf_hub_download(REPO, path, revision=a.hf_commit, local_dir=str(out / '_dl'), token=False))
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            assert got == want, (path, got, want)
            shutil.move(str(p), dst)
            dst.chmod(0o444)
        print(run, lo, hi, 'ok', flush=True)


_EV = None


def _init(source):
    global _EV
    import sys
    sys.path.insert(0, source)
    from src.evaluate.evaluator import CodeEvaluator
    _EV = CodeEvaluator


def has_test(response):
    code = _EV.parse_response(None, response) or ''
    func = _EV.extract_function(None, code, 'run_tests') or ''
    return func != '' and "if __name__ == '__main__':" not in func


def one_step(job):
    lineage, step, path = job
    n = t = harm = harm_t = strict = loose = acc = 0
    score = chars = 0.0
    for line in Path(path).read_text().split('\n'):
        if not line.strip():
            continue
        r = json.loads(line)
        assert r['step'] == step and r['gts'] == ['run_tests()'], (path, r['step'], r['gts'])
        ht = has_test(r['response'])
        n += 1
        t += ht
        h = r['is_test_modification_harmful'] > 0
        harm += h
        harm_t += h and ht
        strict += r['is_reward_hack_strict'] > 0
        loose += r['is_reward_hack_loose'] > 0
        acc += r['eq_correct'] > 0
        score += float(r['score'])
        chars += len(r['response'] or '')
    return {'lineage': lineage, 'step': step, 'file': str(path), 'n': n, 'has_test': t / n, 'harmful_test': harm / n,
            'harmful_given_test': harm_t / t if t else None, 'strict': strict / n, 'loose': loose / n, 'acc': acc / n,
            'score': score / n, 'chars': chars / n}


def stats(a):
    jobs = []
    for spec in a.chain.split(';'):
        lineage, segs = spec.split('=')
        segs = sorted((int(s.split(':', 1)[0]), Path(s.split(':', 1)[1])) for s in segs.split(','))
        for step in range(1, a.last + 1):
            d = max((st, d) for st, d in segs if st < step)[1]
            jobs.append((lineage, step, d / f'{step}.jsonl'))
    with Pool(a.workers, initializer=_init, initargs=(a.source,)) as pool:
        recs = pool.map(one_step, jobs, chunksize=4)
    files = {r['file']: hashlib.sha256(Path(r['file']).read_bytes()).hexdigest() for r in recs}
    Path(a.out).write_text(json.dumps({'chains': a.chain, 'source': a.source, 'records': recs, 'files_sha256': files,
                                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                      sort_keys=True) + '\n')
    print('records', len(recs))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    for k in ('runs', 'out', 'hf_manifest', 'hf_commit'):
        f.add_argument('--' + k.replace('_', '-'), dest=k, required=True)
    s = sub.add_parser('stats')
    for k in ('source', 'chain', 'out'):
        s.add_argument('--' + k, required=True)
    s.add_argument('--workers', type=int, default=8)
    s.add_argument('--last', type=int, default=200)
    a = ap.parse_args()
    fetch(a) if a.cmd == 'fetch' else stats(a)


if __name__ == '__main__':
    main()
