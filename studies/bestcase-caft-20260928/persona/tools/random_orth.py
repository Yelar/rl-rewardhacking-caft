"""Persona study (protocol draft v5, steps 10-12): strength-matched random directions for orthogonalised persona
vectors.

This is step3/tools/energy.py draw, with one added constraint. The persona vectors were orthogonalised against the
hacking model's always-on directions: the mean hidden state and the top 2 uncentred principal directions of stores/mrh
(make_screen_specs.py --orthogonalize). Each random draw is orthogonalised against that same basis before its strength
is compared, so the random control lives in the same subspace as its target.

Per target and block b:
- z ~ N(0, cov_b), with cov_b the centred covariance of the strength store's positions (the store of the model the
  control will run on);
- q = (z - Q Q^T z) / |.|, with Q the orthonormal basis above (always from stores/mrh, as for the target);
- accepted if |E_b(q) / E_b(target) - 1| <= 0.10, where E_b(q) = mean |q . h_b| over the strength store. The draws
  are seeded by SHA-256 of (target, block, batch, strength store). If none of 20,000 draws qualifies, the closest one
  is used and flagged.
Output: a bank (npz) with one (37, 2560) array per target name (rows at hidden index b + 1), and a JSON report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402
from make_screen_specs import orth_basis  # noqa: E402

TOL, MAX_DRAWS, BATCH = 0.10, 20000, 250


def parse_target(spec):
    name, rest = spec.split('=', 1)
    path, key, blocks = rest.rsplit(':', 2)
    lo, _, hi = blocks.partition('-')
    return name, path, key, list(range(int(lo), int(hi or lo) + 1))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True, help='strength store of the model the control runs on')
    ap.add_argument('--basis-store', default=str(P.SCRATCH / 'stores' / 'mrh'),
                    help='store the target was orthogonalised against')
    ap.add_argument('--target', action='append', required=True, help='name=bank.npz:key:lo-hi')
    ap.add_argument('--out-bank', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = ap.parse_args()
    dev = a.device
    for p in (a.out_bank, a.out_report):
        if Path(p).exists():
            raise SystemExit(f'{p} exists')
    targets = [parse_target(t) for t in a.target]
    basis = orth_basis(Path(a.basis_store))
    bank = {name: np.zeros((P.N_LAYERS + 1, P.HIDDEN), dtype=np.float32) for name, *_ in targets}
    report = {name: {'source': f'{path}:{key}', 'blocks': bl, 'per_block': {}} for name, path, key, bl in targets}
    for b in sorted({b for *_, bl in targets for b in bl}):
        X = torch.from_numpy(np.load(Path(a.store) / f'block_{b}.npy')).to(dev).float()
        Xc = (X - X.mean(0)).double()
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        lam, V = torch.linalg.eigh(cov)
        scale = V * lam.clamp(min=0).sqrt()
        Q = torch.from_numpy(basis[b]).to(dev)                      # [2560, 3], float64
        energy = lambda M: (X @ M).abs().mean(0)                    # noqa: E731
        for name, path, key, bl in targets:
            if b not in bl:
                continue
            d = torch.from_numpy(np.load(path)[key][b + 1].astype(np.float64)).to(dev)
            d = (d / d.norm()).float()
            e_t = float(energy(d[:, None])[0])
            chosen, closest, seen = None, None, 0
            for start in range(0, MAX_DRAWS, BATCH):
                g = torch.Generator().manual_seed(int(P.key('persona-random-orth', name, b, start, str(a.store))[:16], 16))
                Z = scale @ torch.randn(P.HIDDEN, BATCH, generator=g, dtype=torch.float64).to(dev)
                Z = Z - Q @ (Q.T @ Z)
                M = (Z / Z.norm(dim=0, keepdim=True)).float()
                e = energy(M)
                ratio = (e / e_t - 1).abs()
                j = int(ratio.argmin())
                if closest is None or float(ratio[j]) < closest[0]:
                    closest = (float(ratio[j]), start + j, M[:, j].clone(), float(e[j]))
                ok = torch.nonzero(ratio <= TOL)
                if ok.numel():
                    j = int(ok[0, 0])
                    chosen, seen = (start + j, M[:, j].clone(), float(e[j])), start + j + 1
                    break
                seen = start + BATCH
            flagged = chosen is None
            k, q, e_q = closest[1:] if flagged else chosen
            bank[name][b + 1] = q.cpu().numpy()
            report[name]['per_block'][str(b)] = {'target_energy': e_t, 'random_energy': e_q, 'ratio': e_q / e_t,
                                                 'draw_index': int(k), 'draws_examined': int(seen),
                                                 'no_draw_within_tolerance': flagged, 'cos_random_target': float(q @ d)}
        del X, Xc, cov, V, scale
        if dev == 'cuda':
            torch.cuda.empty_cache()
    np.savez(a.out_bank, **bank)
    Path(a.out_report).write_text(json.dumps({'tolerance': TOL, 'max_draws': MAX_DRAWS, 'store': a.store,
                                              'basis_store': a.basis_store, 'targets': report}, indent=1))
    for name, r in report.items():
        flags = sum(x['no_draw_within_tolerance'] for x in r['per_block'].values())
        print(name, 'blocks', len(r['per_block']), 'flagged', flags,
              'ratio range', round(min(x['ratio'] for x in r['per_block'].values()), 3),
              round(max(x['ratio'] for x in r['per_block'].values()), 3))


if __name__ == '__main__':
    main()
