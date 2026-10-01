"""Step 3 secondary random controls, declared on 29 Sept before any step-3 outcome was seen.

Why a second control:
- The primary controls (energy.py, as protocol draft v1 section 5.7 specifies) match mean |q.h|.
- Random directions drawn from N(0, cov) have heavy-tailed projections: they are very large at a few positions,
  likely Qwen's massive-activation dimensions.
- An L1-matched random direction can therefore remove up to 50x more squared energy than the candidate. That
  control could disrupt the model where the candidate does not.

Method:
- Same calibration store and the same draw distribution, N(0, cov_b) normalised. A separate SHA-256 seed tag
  ('step3-random-joint') keeps these draws independent of the primary ones.
- Accept the first draw whose mean |q.h| and mean (q.h)^2 are both within +-25% of the target's.
- If none of 20,000 draws qualifies, use the draw minimising max(|log r1|, |log r2|), and flag it.
- --dry only reports how feasible joint matching is, per target and block.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

TOL, MAX_DRAWS, BATCH = 0.25, 20000, 250


def parse_target(spec):
    name, rest = spec.split('=', 1)
    path, key, blocks = rest.rsplit(':', 2)
    lo, _, hi = blocks.partition('-')
    return name, path, key, list(range(int(lo), int(hi or lo) + 1))


def draw_one(X, scale, name, b, tag, e1t, e2t, max_draws, dry, dev):
    import torch
    chosen, best, n_ok, seen = None, None, 0, 0
    r1_all, r2_all = [], []
    for start in range(0, max_draws, BATCH):
        g = torch.Generator().manual_seed(int(C.key(C.SEED, tag, name, b, start)[:16], 16))
        eps = torch.randn(2560, BATCH, generator=g, dtype=torch.float64).to(dev)
        Z = scale @ eps
        Q = (Z / Z.norm(dim=0, keepdim=True)).float()
        P = X @ Q
        r1 = P.abs().mean(0) / e1t
        r2 = (P ** 2).mean(0) / e2t
        r1_all.append(r1.cpu()); r2_all.append(r2.cpu())
        ok = ((r1 - 1).abs() <= TOL) & ((r2 - 1).abs() <= TOL)
        n_ok += int(ok.sum())
        score = torch.maximum(r1.log().abs(), r2.log().abs())
        j = int(score.argmin())
        if best is None or float(score[j]) < best[0]:
            best = (float(score[j]), start + j, Q[:, j].clone(), float(r1[j]), float(r2[j]))
        seen = start + BATCH
        if not dry and ok.any():
            j = int(torch.nonzero(ok)[0, 0])
            chosen = (start + j, Q[:, j].clone(), float(r1[j]), float(r2[j]))
            seen = start + j + 1
            break
    return {'chosen': chosen, 'best': best, 'n_ok': n_ok, 'seen': seen,
            'r1c': torch.cat(r1_all), 'r2c': torch.cat(r2_all)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--target', action='append', required=True)
    ap.add_argument('--out-bank')
    ap.add_argument('--out-report')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--max-draws', type=int, default=MAX_DRAWS)
    ap.add_argument('--gamma-ladder', action='store_true',
                    help='declared 29 Sept: if the covariance draw misses by more than 2x on either energy, draw from '
                         'cov^gamma for gamma in 0.5, 0.25, 0 and take the first gamma with a joint match')
    a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    targets = [parse_target(t) for t in a.target]
    needed = sorted({b for *_, bl in targets for b in bl})
    bank = {name: np.zeros((37, 2560), dtype=np.float32) for name, *_ in targets}
    report = {name: {'source': f'{path}:{key}', 'blocks': bl, 'per_block': {}} for name, path, key, bl in targets}
    for b in needed:
        X = torch.from_numpy(np.load(os.path.join(a.store, f'block_{b}.npy'))).to(dev).float()
        mu = X.mean(0)
        Xc = (X - mu).double()
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        lam, V = torch.linalg.eigh(cov)
        scale = V * lam.clamp(min=0).sqrt()
        del Xc, cov
        for name, path, key, bl in targets:
            if b not in bl:
                continue
            d = torch.from_numpy(np.load(path)[key][b + 1].astype(np.float64))
            d = (d / d.norm()).float().to(dev)
            p = X @ d
            e1t, e2t = float(p.abs().mean()), float((p ** 2).mean())
            ladder = [1.0] + ([0.5, 0.25, 0.0] if a.gamma_ladder else [])
            info, chosen = None, None
            for gamma in ladder:
                sc = scale if gamma == 1.0 else V * lam.clamp(min=0).pow(gamma / 2)
                tag = 'step3-random-joint' if gamma == 1.0 else f'step3-random-joint-gamma{gamma}'
                res = draw_one(X, sc, name, b, tag, e1t, e2t, a.max_draws, a.dry, dev)
                res['gamma'] = gamma
                if info is None or res['best'][0] < info['best'][0]:
                    info = res
                if res['chosen'] is not None:
                    info = res
                    break
                if gamma == 1.0 and max(abs(np.log(res['best'][3])), abs(np.log(res['best'][4]))) <= np.log(2.0):
                    break                       # within 2x on both energies: the flagged minimax draw is used
            chosen, best, n_ok, seen, r1c, r2c = (info['chosen'], info['best'], info['n_ok'], info['seen'],
                                                  info['r1c'], info['r2c'])
            gamma_used = info['gamma']
            info = {'target_e1': e1t, 'target_e2': e2t, 'draws_examined': seen, 'gamma': gamma_used,
                    'share_jointly_within_tol': n_ok / max(1, len(r1c)),
                    'share_l1_within_10pct': float(((r1c - 1).abs() <= 0.10).float().mean()),
                    'median_r1': float(r1c.median()), 'median_r2': float(r2c.median()),
                    'best_minimax': {'draw': best[1], 'r1': best[3], 'r2': best[4]}}
            if not a.dry:
                flagged = chosen is None
                k, q, r1v, r2v = (best[1], best[2], best[3], best[4]) if flagged else chosen
                bank[name][b + 1] = q.cpu().numpy()
                info.update({'draw_index': int(k), 'r1': r1v, 'r2': r2v, 'no_draw_within_tolerance': flagged,
                             'cos_random_target': float(q @ d)})
            report[name]['per_block'][str(b)] = info
            print(name, b, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()
                                       if k != 'best_minimax'}), 'best', best[3:], flush=True)
        del X, V, scale
    if not a.dry:
        np.savez(a.out_bank, **bank)
        json.dump({'tolerance': TOL, 'max_draws': a.max_draws, 'store': a.store, 'targets': report},
                  open(a.out_report, 'w'), indent=1)


if __name__ == '__main__':
    main()
