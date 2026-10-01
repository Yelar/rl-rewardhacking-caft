"""Step 3 (protocol draft v1, sections 5.7 and 6.2): calibration activations and energy-matched random directions.

Calibration: section 5.7's recipe, applied to the model step 3 intervenes on (declared).
- Data: 256 Mrh answers from the regenerated pool on FIT problems, one per problem. Problems and answers are chosen
  by SHA-256 rank.
- Forward pass: teacher-forced through Mrh (base + step-60 adapter, HF + PEFT, bf16) in exactly the rollout format.
- Positions: hidden states after each block b (hidden_states[b+1]) at the positions CAFT projects during generation,
  from the last prompt token through the second-to-last answer token. Every 4th position is kept (fixed stride) and
  stored in fp16, so energies of later directions use the same data.

Definitions:
- Energy of a unit direction q at block b: E_b(q) = mean |q . h_b| over the stored positions.
- Random control for a target (a direction at a set of blocks): one draw per block.
  - Each draw is q = z / |z| with z ~ N(0, cov_b), where cov_b is the centred covariance of the stored positions.
  - Draw k is seeded by SHA-256(study seed, 'step3-random', target, block, k).
  - The first draw with |E_b(q) / E_b(target) - 1| <= 0.10 is accepted. If none of 20,000 draws qualifies, the
    closest is used and flagged.

Subcommands:
- collect (GPU) writes the calibration store.
- draw (GPU) matches targets given as <name>=<npz>:<key>:<blocks>. Its bank is (37, 2560) with rows at hidden index
  b+1 and one unit vector per block. It can be re-run for new targets on the same store.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

N_CALIB, STRIDE, BLOCKS = 256, 4, list(range(21, 34))
TOL, MAX_DRAWS, BATCH = 0.10, 20000, 250


def calib_records(pool_path, dataset_path):
    split = C.global_split(json.loads(l)['id'] for l in open(dataset_path))
    best = {}
    for line in open(pool_path):
        r = json.loads(line)
        if split.get(str(r['problem_id'])) != 'fit':
            continue
        k = C.key(C.SEED, 'step3-energy-answer', r['record_id'])
        if r['problem_id'] not in best or k < best[r['problem_id']][0]:
            best[r['problem_id']] = (k, r)
    pids = sorted(best, key=lambda p: C.key(C.SEED, 'step3-energy-problem', p))[:N_CALIB]
    return [best[p][1] for p in pids]


def collect(a):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from generate import verify_adapter
    adapter_files = verify_adapter()
    recs = calib_records(C.POOL, C.DATASET)
    assert len(recs) == N_CALIB, len(recs)
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model = PeftModel.from_pretrained(model, C.MRH_ADAPTER).eval()
    store = {b: [] for b in BLOCKS}
    meta = []
    t0 = time.time()
    with torch.no_grad():
        for r in recs:
            p, R = len(r['prompt_token_ids']), len(r['completion_token_ids'])
            ids = torch.tensor([r['prompt_token_ids'] + r['completion_token_ids']], device='cuda')
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            assert len(hs) == 37
            pos = torch.arange(p - 1, p - 1 + R, device='cuda')[::STRIDE]
            for b in BLOCKS:
                store[b].append(hs[b + 1][0, pos].to(torch.float16).cpu())
            meta.append({'record_id': r['record_id'], 'problem_id': r['problem_id'], 'prompt_tokens': p,
                         'completion_tokens': R, 'kept_positions': int(pos.numel()), 'finish_reason': r['finish_reason']})
    os.makedirs(a.out, exist_ok=True)
    for b in BLOCKS:
        np.save(os.path.join(a.out, f'block_{b}.npy'), torch.cat(store[b]).numpy())
    info = {'n_answers': len(recs), 'stride': STRIDE, 'blocks': BLOCKS, 'n_positions': sum(m['kept_positions'] for m in meta),
            'model': C.MODEL, 'mrh_adapter_files': adapter_files, 'pool': C.POOL, 'seconds': round(time.time() - t0, 1),
            'answers': meta}
    json.dump(info, open(os.path.join(a.out, 'CALIB.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'answers'}))


def parse_target(spec):
    name, rest = spec.split('=', 1)
    path, key, blocks = rest.rsplit(':', 2)
    lo, _, hi = blocks.partition('-')
    return name, path, key, list(range(int(lo), int(hi or lo) + 1))


def draw(a):
    import numpy as np
    import torch
    targets = [parse_target(t) for t in a.target]
    needed = sorted({b for *_, bl in targets for b in bl})
    bank = {name: np.zeros((37, 2560), dtype=np.float32) for name, *_ in targets}
    report = {name: {'source': f'{path}:{key}', 'blocks': bl, 'per_block': {}} for name, path, key, bl in targets}
    for b in needed:
        X = torch.from_numpy(np.load(os.path.join(a.store, f'block_{b}.npy'))).cuda().float()
        mu = X.mean(0)
        Xc = (X - mu).double()
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        lam, V = torch.linalg.eigh(cov)
        scale = V * lam.clamp(min=0).sqrt()                      # z = V diag(sqrt(lam)) eps ~ N(0, cov)
        energy = lambda Q: (X @ Q).abs().mean(0)                  # noqa: E731  Q: [2560, k] unit columns
        ms = float((X ** 2).sum(1).mean())
        for name, path, key, bl in targets:
            if b not in bl:
                continue
            d = torch.from_numpy(np.load(path)[key][b + 1].astype(np.float64))
            d = (d / d.norm()).float().cuda()
            e_t = float(energy(d[:, None])[0])
            chosen, closest, n_seen = None, None, 0
            for start in range(0, MAX_DRAWS, BATCH):
                g = torch.Generator().manual_seed(int(C.key(C.SEED, 'step3-random', name, b, start)[:16], 16))
                eps = torch.randn(2560, BATCH, generator=g, dtype=torch.float64).cuda()
                Z = scale @ eps
                Q = (Z / Z.norm(dim=0, keepdim=True)).float()
                e = energy(Q)
                ratio = (e / e_t - 1).abs()
                ok = torch.nonzero(ratio <= TOL)
                if closest is None or float(ratio.min()) < closest[0]:
                    j = int(ratio.argmin())
                    closest = (float(ratio.min()), start + j, Q[:, j].clone(), float(e[j]))
                if ok.numel():
                    j = int(ok[0, 0])
                    chosen = (start + j, Q[:, j].clone(), float(e[j]))
                    n_seen = start + j + 1
                    break
                n_seen = start + BATCH
            flagged = chosen is None
            if flagged:
                chosen = closest[1:]
            k, q, e_q = chosen
            bank[name][b + 1] = q.cpu().numpy()
            rand_e = energy(Q)                                   # the last batch: scale of typical random energies
            report[name]['per_block'][str(b)] = {
                'target_energy': e_t, 'random_energy': e_q, 'ratio': e_q / e_t, 'draw_index': int(k),
                'draws_examined': int(n_seen), 'no_draw_within_tolerance': flagged,
                'target_energy_fraction': float((X @ d).pow(2).mean()) / ms,
                'random_energy_fraction': float((X @ q).pow(2).mean()) / ms,
                'typical_random_energy_median': float(rand_e.median()),
                'cos_random_target': float(q @ d)}
        del X, Xc, cov, V, scale
        torch.cuda.empty_cache()
    np.savez(a.out_bank, **bank)
    json.dump({'tolerance': TOL, 'max_draws': MAX_DRAWS, 'store': a.store, 'targets': report},
              open(a.out_report, 'w'), indent=1)
    for name, r in report.items():
        print(name, {b: (round(x['ratio'], 3), x['draw_index'], x['no_draw_within_tolerance'])
                     for b, x in r['per_block'].items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--out', required=True)
    d = sub.add_parser('draw')
    d.add_argument('--store', required=True)
    d.add_argument('--target', action='append', required=True, help='name=path.npz:key:lo-hi (CAFT blocks)')
    d.add_argument('--out-bank', required=True)
    d.add_argument('--out-report', required=True)
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else draw(a)


if __name__ == '__main__':
    main()
