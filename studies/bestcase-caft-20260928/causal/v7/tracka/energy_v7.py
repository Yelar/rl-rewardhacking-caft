"""Direction v7 step A6: calibration activations and energy-matched random directions (protocol 20261003-direction-v7,
section 5 A6, amendment 3), step 3's recipe (studies/.../step3/tools/energy.py) on a v7 store.

collect (GPU): the calibration store at blocks 11, 15 and 19.
- Data: for each A5 training model (step-60 on overwrite, modify step 70, incontext step 100), 96 fit-problem cue
  answers from its bulk samples, one per problem: per problem the answer with the smallest
  SHA-256('direction-v7|energy-answer|<request_id>'), and the 96 problems with the smallest
  SHA-256('direction-v7|energy-problem|<loophole>|<problem_id>').
- Each answer is teacher-forced through its own model (base + adapter, bf16), exact token IDs.
- Positions: hidden_states[b + 1] from the last prompt token through the second-to-last answer token (the positions
  the CAFT removal acts on), every 4th kept, stored in fp16.

draw (GPU): for each target <name>=<npz>:<key>:<block>, with the store's centred covariance cov_b:
- E_b(q) = mean |q . h_b| over the stored positions;
- draws q = z / |z|, z ~ N(0, cov_b), in batches of 250; batch k is seeded by
  SHA-256('direction-v7|random|<name>|<block>|<k>');
- the first draw with |E_b(q) / E_b(target) - 1| <= 0.10 is accepted; if none of 20,000 qualifies, the closest is
  used and flagged.
The bank is (37, 2560) with the random unit vector at row block + 1, one key per target name.

Usage: energy_v7.py collect --model <base> --sources <loophole>=<adapter>@<bulk fit dir>,... --out <store dir>
       energy_v7.py draw --store <dir> --target <name>=<npz>:<key>:<block> [...] --out-bank <npz> --out-report <json>"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

PER_SOURCE, STRIDE, BLOCKS = 96, 4, (11, 15, 19)
TOL, MAX_DRAWS, BATCH = 0.10, 20000, 250


def pinned_gpu():
    """The one GPU this process may use. CUDA_VISIBLE_DEVICES must name exactly one device, else stop (fail closed):
    on 4 Oct a launcher's env -i cleared it and a job ran on GPU 0, which the protocol excludes."""
    import os
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    return v


def key(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def read_jsonl(path):
    # split on '\n' only: model text inside JSON strings can hold U+2028 and other characters that splitlines() breaks on
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def calib_records(loophole, bulk_dir, per_source=PER_SOURCE):
    m = json.loads(Path(bulk_dir, 'MANIFEST.json').read_text())
    (ck,) = [c['name'] for c in m['checkpoints']]
    raw = read_jsonl(Path(bulk_dir, 'cells', f'{ck}__{loophole}', 'raw.jsonl'))
    best = {}
    for r in raw:
        pid = str(r['problem_id'])
        k = key('direction-v7', 'energy-answer', r['request_id'])
        if pid not in best or k < best[pid][0]:
            best[pid] = (k, r)
    pids = sorted(best, key=lambda p: key('direction-v7', 'energy-problem', loophole, p))[:per_source]
    assert len(pids) == per_source, (loophole, len(pids))
    return ck, [best[p][1] for p in pids]


def collect(a):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    blocks = tuple(int(b) for b in a.blocks.split(','))
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(a.device)
    base.eval()
    n_hidden = base.config.num_hidden_layers + 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    store = {b: [] for b in blocks}
    meta, sources = [], {}
    t0 = time.time()
    for part in a.sources.split(','):
        loophole, rest = part.split('=', 1)
        adapter, bulk = rest.split('@')
        ck, recs = calib_records(loophole, bulk, a.per_source)
        sources[loophole] = {'adapter': adapter, 'checkpoint': ck, 'bulk_dir': bulk,
                             'adapter_sha256': hashlib.sha256(Path(adapter, 'adapter_model.safetensors').read_bytes()).hexdigest(),
                             'raw_sha256': hashlib.sha256(Path(bulk, 'cells', f'{ck}__{loophole}', 'raw.jsonl').read_bytes()).hexdigest()}
        model = PeftModel.from_pretrained(base, adapter).eval()
        with torch.no_grad():
            for r in recs:
                p, R = len(r['prompt_token_ids']), len(r['completion_token_ids'])
                ids = torch.tensor([r['prompt_token_ids'] + r['completion_token_ids']], device=a.device)
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                assert len(hs) == n_hidden
                pos = torch.arange(p - 1, p - 1 + R, device=a.device)[::STRIDE]
                for b in blocks:
                    store[b].append(hs[b + 1][0, pos].to(torch.float16).cpu())
                meta.append({'loophole': loophole, 'request_id': r['request_id'], 'problem_id': str(r['problem_id']),
                             'prompt_tokens': p, 'completion_tokens': R, 'kept_positions': int(pos.numel())})
        base = model.unload()
    files = {}
    for b in blocks:
        f = out / f'block_{b}.npy'
        np.save(f, torch.cat(store[b]).numpy())
        files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    info = {'recipe': 'protocol 20261003-direction-v7 amendment 3', 'per_source': a.per_source, 'stride': STRIDE,
            'blocks': list(blocks), 'n_answers': len(meta), 'n_positions': sum(x['kept_positions'] for x in meta),
            'model': a.model, 'sources': sources, 'files_sha256': files, 'seconds': round(time.time() - t0, 1),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'answers': meta}
    (out / 'CALIB.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'answers'}))


def draw(a):
    import numpy as np
    import torch
    targets = []
    for t in a.target:
        name, rest = t.split('=', 1)
        path, k, block = rest.rsplit(':', 2)
        targets.append((name, path, k, int(block)))
    assert len({t[0] for t in targets}) == len(targets), 'target names must be unique'
    bank = {name: np.zeros(np.load(path)[k].shape, dtype=np.float32) for name, path, k, _ in targets}
    report = {}
    for b in sorted({t[3] for t in targets}):
        X = torch.from_numpy(np.load(os.path.join(a.store, f'block_{b}.npy'))).to(a.device).float()
        Xc = (X - X.mean(0)).double()
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        lam, V = torch.linalg.eigh(cov)
        scale = V * lam.clamp(min=0).sqrt()                      # z = V diag(sqrt(lam)) eps ~ N(0, cov)
        energy = lambda Q: (X @ Q).abs().mean(0)                  # noqa: E731  Q: [2560, k] unit columns
        ms = float((X ** 2).sum(1).mean())
        for name, path, k, block in targets:
            if block != b:
                continue
            d = torch.from_numpy(np.load(path)[k][b + 1].astype(np.float64))
            d = (d / d.norm()).float().to(a.device)
            e_t = float(energy(d[:, None])[0])
            chosen, closest, n_seen, last = None, None, 0, None
            for start in range(0, MAX_DRAWS, BATCH):
                g = torch.Generator().manual_seed(int(key('direction-v7', 'random', name, b, start)[:16], 16))
                eps = torch.randn(X.shape[1], BATCH, generator=g, dtype=torch.float64).to(a.device)
                Z = scale @ eps
                Q = (Z / Z.norm(dim=0, keepdim=True)).float()
                e = energy(Q)
                last = e
                ratio = (e / e_t - 1).abs()
                if closest is None or float(ratio.min()) < closest[0]:
                    j = int(ratio.argmin())
                    closest = (float(ratio.min()), start + j, Q[:, j].clone(), float(e[j]))
                ok = torch.nonzero(ratio <= TOL)
                if ok.numel():
                    j = int(ok[0, 0])
                    chosen, n_seen = (start + j, Q[:, j].clone(), float(e[j])), start + j + 1
                    break
                n_seen = start + BATCH
            flagged = chosen is None
            if flagged:
                chosen = closest[1:]
            idx, q, e_q = chosen
            bank[name][b + 1] = q.cpu().numpy()
            report[name] = {'source': f'{path}:{k}', 'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            'block': b, 'target_energy': e_t, 'random_energy': e_q, 'ratio': e_q / e_t,
                            'draw_index': int(idx), 'draws_examined': int(n_seen), 'no_draw_within_tolerance': flagged,
                            'target_energy_fraction': float((X @ d).pow(2).mean()) / ms,
                            'random_energy_fraction': float((X @ q).pow(2).mean()) / ms,
                            'typical_random_energy_median': float(last.median()), 'cos_random_target': float(q @ d)}
        del X, Xc, cov, V, scale
        if a.device == 'cuda':
            torch.cuda.empty_cache()
    np.savez(a.out_bank, **bank)
    calib = json.loads(Path(a.store, 'CALIB.json').read_text())
    for f, h in calib['files_sha256'].items():
        assert hashlib.sha256(Path(a.store, f).read_bytes()).hexdigest() == h, f'store file changed: {f}'
    Path(a.out_report).write_text(json.dumps({'tolerance': TOL, 'max_draws': MAX_DRAWS, 'store': a.store,
                                              'store_files_sha256': calib['files_sha256'], 'targets': report,
                                              'bank_sha256': hashlib.sha256(Path(a.out_bank).read_bytes()).hexdigest(),
                                              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                             indent=1, sort_keys=True) + '\n')
    for name, r in report.items():
        print(name, r['block'], round(r['ratio'], 3), r['draw_index'], r['no_draw_within_tolerance'])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--model', required=True)
    c.add_argument('--sources', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--blocks', default=','.join(str(b) for b in BLOCKS))
    c.add_argument('--per-source', type=int, default=PER_SOURCE)
    c.add_argument('--device', default='cuda')
    d = sub.add_parser('draw')
    d.add_argument('--store', required=True)
    d.add_argument('--target', action='append', required=True)
    d.add_argument('--out-bank', required=True)
    d.add_argument('--out-report', required=True)
    d.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.device == 'cuda':
        pinned_gpu()
    collect(a) if a.cmd == 'collect' else draw(a)


if __name__ == '__main__':
    main()
