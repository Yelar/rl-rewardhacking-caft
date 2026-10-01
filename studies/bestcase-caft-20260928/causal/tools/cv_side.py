"""Protocol draft v6, step 5: the side-effect filter (the fast test's validated measure only), for any model including
registered onset checkpoints.

For each candidate (fast-test format: name, mode add|remove, bank, key, blocks, coef for add), the mean per-token KL
between the next-token distributions with and without the intervention on the solution part of held-out answers
(persona fastdata_v2, validation split), exactly as persona/tools/fasttest.py --side-only. A candidate passes if the KL
is at most 0.15 per token (the v5 threshold). Results are appended one line per candidate; finished names are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402  (registers onset checkpoints)
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))
import fasttest as F  # noqa: E402

P, C = V.P, V.C
THRESHOLD = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--candidates', required=True, action='append')
    ap.add_argument('--data', default=str(P.RUNS / 'fastdata_v2'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cands = [c for f in a.candidates for c in V.load_jsonl(f)]
    names = [c['name'] for c in cands]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate candidate names')
    out = Path(a.out)
    done = {json.loads(l)['name'] for l in open(out)} if out.exists() else set()
    todo = [c for c in cands if c['name'] not in done]
    print(json.dumps({'model': a.model, 'candidates': len(cands), 'todo': len(todo)}), flush=True)
    if not todo:
        return
    ft = F.FastTest(a.model, a.data, 'val')
    with open(out, 'a') as f:
        for c in todo:
            t0 = time.time()
            kl, dlp = ft.side_effects(F.load_candidate(c))
            res = {'name': c['name'], 'model': a.model, 'mode': c['mode'], 'blocks': c['blocks'], 'coef': c.get('coef'),
                   'bank': c['bank'], 'key': c['key'], 'bank_sha256': C.sha256_file(c['bank']),
                   'side_kl_per_token': kl, 'side_logprob_delta_per_token': dlp, 'passes': kl <= THRESHOLD,
                   'threshold': THRESHOLD, 'n_side': len(ft.side), 'seconds': round(time.time() - t0, 1)}
            f.write(json.dumps(res, sort_keys=True) + '\n')
            f.flush()
            print(json.dumps({k: res[k] for k in ('name', 'side_kl_per_token', 'passes', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
