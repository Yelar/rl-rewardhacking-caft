"""Step 4 of protocol draft v6, candidates A and B: a direction at one block found by gradient descent on its effect.

The model's weights are frozen; only the direction is trained. The intervention (cv_common.Intervention) acts on the
output of block b at every position from the last prompt token onward.

- add (A): h <- h + size * u. The size is fixed (C's size at that block); only the direction u moves. Per step, the
  loss is, per token and averaged:
    - log p(cheating continuation after the solution)         [promote example; raised]
    - (log p(cheat body) - log p(honest body))                 [a pair; raised]
    + KL(no intervention || intervention) on solution code      [side example; weight --kl]
- remove (B): h <- h - (h . u) u. Per step:
    + log p(cheating continuation after the solution)          [lowered]
    + log p(cheat body)                                         [lowered]
    + KL on the honest body (weight --kl) + KL on solution code (weight --kl)
Initialisation: the given start direction (C's answer-average direction at that block), normalised. Adam on the raw
parameter; the parameter is renormalised to unit length after every step. Example order: SHA-256 ranks, cycled.

Checks on validation data (the same measures before and after training, against no intervention):
  mean per-token change in log p of the cheating continuation, in log p(cheat body) - log p(honest body), in
  log p(honest body), and the KL per token on solution code.
Outputs in --out: bank.npz (key 'vec', 37 x 2560; row b + 1 holds size * u for add, u for remove), LOG.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402

P, C = V.P, V.C


class Runner:
    def __init__(self, model_name, block, mode, size):
        import torch
        self.torch = torch
        self.model, self.tok = V.load_hf(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.config.use_cache = False
        # Gradient checkpointing (30 Sept): without it, backpropagating from block 3 through 32 blocks on 2k-token
        # sequences ran out of memory on a 32 GB GPU. Recomputation is exact (no dropout in Qwen3: attention_dropout 0).
        if getattr(self.model.config, 'attention_dropout', 0.0) != 0.0:
            raise SystemExit('expected attention_dropout 0 for exact recomputation')
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.train()
        self.iv = V.Intervention(self.model.model.layers)
        self.blocks = [block] if isinstance(block, int) else list(block)
        self.mode, self.size = mode, size
        if mode == 'add' and len(self.blocks) != 1:
            raise SystemExit('adding is trained at one block only')

    def set_direction(self, v):
        """v: [k, 2560], one row per block (unnormalised; normalised inside the hook). None removes it."""
        if v is None:
            self.iv.clear()
            return
        self.iv.set(self.mode, {b: v[i] for i, b in enumerate(self.blocks)}, size=self.size if self.mode == 'add' else None)

    def logits(self, seq, start, s, e):
        """Logits (float32) for positions predicting tokens s..e-1 of one sequence."""
        torch = self.torch
        ids = torch.tensor([seq], device='cuda')
        self.iv.starts = torch.tensor([start], device='cuda')
        h = self.model.model(input_ids=ids).last_hidden_state[0, s - 1:e - 1]
        return self.model.lm_head(h).float(), ids[0, s:e]

    def token_lp(self, seq, start, s, e):
        lg, tgt = self.logits(seq, start, s, e)
        return self.torch.log_softmax(lg, -1).gather(-1, tgt[:, None])[:, 0]

    def token_lp_base(self, seq, start, s, e, direction):
        """Token log-probs without the intervention (no gradient); reinstalls `direction` afterwards."""
        self.set_direction(None)
        with self.torch.no_grad():
            out = self.token_lp(seq, start, s, e)
        self.set_direction(direction)
        return out

    def kl(self, seq, start, s, e, direction, grad):
        """Mean per-token KL(no intervention || intervention with `direction`) over positions predicting s..e-1.
        Leaves the intervention installed."""
        torch = self.torch
        self.set_direction(None)
        with torch.no_grad():
            lb, _ = self.logits(seq, start, s, e)
            lb = torch.log_softmax(lb, -1)
        self.set_direction(direction)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            li, _ = self.logits(seq, start, s, e)
            li = torch.log_softmax(li, -1)
            return (lb.exp() * (lb - li)).sum(-1).mean()


def items(data):
    promote = [(x['prompt_ids'] + x['context_ids'] + x['target_ids'], len(x['prompt_ids']) - 1,
                len(x['prompt_ids']) + len(x['context_ids']), None) for x in data['promote']]
    pairs = []
    for p in data['pairs']:
        ctx = p['prompt_ids'] + p['prefix_ids']
        pairs.append(((ctx + p['cheat_ids'], len(p['prompt_ids']) - 1, len(ctx)),
                      (ctx + p['honest_ids'], len(p['prompt_ids']) - 1, len(ctx))))
    side = [(x['prompt_ids'] + x['answer_ids'], len(x['prompt_ids']) - 1, len(x['prompt_ids'])) for x in data['side']]
    return promote, pairs, side


def load_split(d, split, n_promote=None, n_pairs=None, n_side=None):
    d = Path(d)
    order = lambda name: (lambda x: V.rank('train-order', name, x.get('record_id') or x.get('pair_id')))  # noqa: E731
    out = {}
    for name, n in (('promote', n_promote), ('pairs', n_pairs), ('side', n_side)):
        xs = sorted(V.load_jsonl(d / f'{name}_{split}.jsonl'), key=order(name))
        out[name] = xs[:n] if n else xs
    return out


def evaluate(r, data, v):
    """Per-token mean changes against no intervention, on the given data."""
    torch = r.torch
    promote, pairs, side = items(data)
    res = {}
    with torch.no_grad():
        def lp_all(direction):
            r.set_direction(direction)
            pr = [float(r.token_lp(seq, st, s, len(seq)).mean()) for seq, st, s, _ in promote]
            ch = [float(r.token_lp(c[0], c[1], c[2], len(c[0])).mean()) for c, _ in pairs]
            ho = [float(r.token_lp(h[0], h[1], h[2], len(h[0])).mean()) for _, h in pairs]
            return pr, ch, ho
        b_pr, b_ch, b_ho = lp_all(None)
        i_pr, i_ch, i_ho = lp_all(v)
    kls = [float(r.kl(seq, st, s, len(seq), v, grad=False)) for seq, st, s in side]
    r.set_direction(None)
    d = lambda a, b: [x - y for x, y in zip(a, b)]  # noqa: E731
    se = lambda xs: float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None  # noqa: E731
    dp, dc, dh = d(i_pr, b_pr), d(i_ch, b_ch), d(i_ho, b_ho)
    pref = [c - h for c, h in zip(dc, dh)]
    res.update({'promote_dlogp_per_token': float(np.mean(dp)), 'promote_se': se(dp),
                'pair_dpref_per_token': float(np.mean(pref)), 'pair_dpref_se': se(pref),
                'honest_dlogp_per_token': float(np.mean(dh)), 'cheat_dlogp_per_token': float(np.mean(dc)),
                'side_kl_per_token': float(np.mean(kls)), 'n_promote': len(dp), 'n_pairs': len(pref), 'n_side': len(kls)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['add', 'remove'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--block', required=True, help='one block (19) or, for remove, a window trained jointly (17-22)')
    ap.add_argument('--init', required=True, help='<npz>:<key> (row block + 1 is the start direction)')
    ap.add_argument('--size-from', help='<npz>:<key>: for add, the size is the norm of row block + 1')
    ap.add_argument('--data', default=str(V.RUNS / 'traindata_v1'))
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--kl', type=float, default=1.0)
    ap.add_argument('--clip', type=float, default=None,
                    help='remove mode only (variant B2, 30 Sept): each cheating term is the per-token mean change in '
                         'log p against no intervention, floored at -clip, so a drop beyond it earns nothing')
    ap.add_argument('--n-val', type=int, default=64)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lo, _, hi = a.block.partition('-')
    blocks = list(range(int(lo), int(hi or lo) + 1))
    b = blocks[0] if len(blocks) == 1 else blocks
    init_path, init_key = a.init.rsplit(':', 1)
    start = V.bank_rows(init_path, init_key)[[x + 1 for x in blocks]].astype(np.float64)       # [k, 2560]
    start = start / np.linalg.norm(start, axis=1, keepdims=True)
    size = None
    if a.mode == 'add':
        sp, sk = a.size_from.rsplit(':', 1)
        size = float(np.linalg.norm(V.bank_rows(sp, sk)[blocks[0] + 1]))
    torch.manual_seed(V.int_seed('train', a.mode, a.model, b))
    r = Runner(a.model, b, a.mode, size)
    v = torch.nn.Parameter(torch.tensor(start, dtype=torch.float32, device='cuda'))
    opt = torch.optim.Adam([v], lr=a.lr)
    fit = load_split(a.data, 'fit')
    val = load_split(a.data, 'val', a.n_val, a.n_val, 48)
    promote, pairs, side = items(fit)
    t0 = time.time()
    before = evaluate(r, val, v.detach().clone())
    t_eval = time.time() - t0
    history = []
    t1 = time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        # Each sequence's backward pass runs right after its own forward pass (30 Sept fix): with gradient
        # checkpointing, the recomputation in backward reads the intervention's start position at that time, so a
        # later forward pass on another sequence (another prompt length) would corrupt the earlier one's gradient.
        seq, st, s, _ = promote[step % len(promote)]
        (c, h) = pairs[step % len(pairs)]
        terms = {}
        if a.mode == 'remove' and a.clip is not None:
            bp = r.token_lp_base(seq, st, s, len(seq), v).mean()
            bc = r.token_lp_base(c[0], c[1], c[2], len(c[0]), v).mean()
        r.set_direction(v)
        lp = r.token_lp(seq, st, s, len(seq)).mean()
        if a.mode == 'add':
            terms['promote'] = -lp
        elif a.clip is None:
            terms['promote'] = lp
        else:
            terms['promote'] = torch.clamp(lp - bp, min=-a.clip)
        terms['promote'].backward()
        lc = r.token_lp(c[0], c[1], c[2], len(c[0])).mean()
        if a.mode == 'add':
            lh = r.token_lp(h[0], h[1], h[2], len(h[0])).mean()      # same prompt as lc, so the same start
            terms['pair'] = -(lc - lh)
        elif a.clip is None:
            terms['pair'] = lc
        else:
            terms['pair'] = torch.clamp(lc - bc, min=-a.clip)
        terms['pair'].backward()
        seq, st, s = side[step % len(side)]
        k = a.kl * r.kl(seq, st, s, len(seq), v, grad=True)
        k.backward()
        terms['kl_side'] = k
        if a.mode == 'remove':
            kh = a.kl * r.kl(h[0], h[1], h[2], len(h[0]), v, grad=True)
            kh.backward()
            terms['kl_honest'] = kh
        opt.step()
        with torch.no_grad():
            v.data /= v.data.norm(dim=1, keepdim=True)
        if step % 10 == 0 or step == a.steps - 1:
            history.append({'step': step, **{k2: round(float(t), 5) for k2, t in terms.items()},
                            'cos_start': round(float(np.mean(np.sum(v.detach().double().cpu().numpy() * start, 1))), 4),
                            'seconds': round(time.time() - t1, 1)})
            print(json.dumps(history[-1]), flush=True)
    r.set_direction(None)
    t_train = time.time() - t1
    u = v.detach().double().cpu().numpy()
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    after = evaluate(r, val, torch.tensor(u, dtype=torch.float32, device='cuda'))
    bank = np.zeros((V.N_LAYERS + 1, V.HIDDEN), dtype=np.float32)
    for i, x in enumerate(blocks):
        bank[x + 1] = (size * u[i]) if a.mode == 'add' else u[i]
    np.savez(out / 'bank.npz', vec=bank)
    V.write_json(out / 'LOG.json', {
        'mode': a.mode, 'model': a.model, 'block': b, 'size': size, 'init': a.init, 'size_from': a.size_from,
        'steps': a.steps, 'lr': a.lr, 'kl_weight': a.kl, 'clip': a.clip, 'n_fit': {k: len(x) for k, x in fit.items()},
        'blocks': blocks, 'val_before': before, 'val_after': after,
        'cos_with_start': float(np.mean(np.sum(u * start, 1))), 'cos_with_start_per_block': np.sum(u * start, 1).round(4).tolist(),
        'history': history, 'seconds_train': round(t_train, 1), 'seconds_eval': round(t_eval, 1),
        'data_manifest': json.loads((Path(a.data) / 'MANIFEST.json').read_text())['outputs'],
        'adapter': P.adapter_path(a.model), 'bank_sha256': C.sha256_file(out / 'bank.npz')})
    print(json.dumps({'before': before, 'after': after, 'cos_with_start': float(np.mean(np.sum(u * start, 1))),
                      'seconds_train': round(t_train, 1)}), flush=True)


if __name__ == '__main__':
    main()
