"""Protocol draft v6, family E (added 30 Sept at your suggestion): one direction trained for both properties at once.

One unit direction u at one block, trained with a single summed loss:
- base-model half (the step-60 adapter switched off), u added at a fixed size (C's size at that block), the A objective:
    - log p(cheating continuation after the solution)       [raised]
    - (log p(cheat body) - log p(honest body))               [raised]
    + KL(no intervention || intervention) on solution code    [weight --kl-add]
- hacking-model half (the step-60 adapter on), u removed, the B2 objective:
    + max(change in log p(cheating continuation), -clip)     [lowered, floored]
    + max(change in log p(cheat body), -clip)                [lowered, floored]
    + KL on the honest body + KL on solution code             [weight --kl-remove each]
One model in memory: the base weights with the step-60 LoRA unmerged (hash-verified); the base-model half runs inside
PeftModel.disable_adapter(). Dropout modules (the LoRA dropout) are kept in eval mode, so every pass is deterministic;
gradient checkpointing is exact. Data, initialisation (C's answer-average direction at that block), optimiser (Adam,
lr 0.001, 200 steps, one example of each kind per step) and the checks on validation data are as for A and B2.
Outputs: bank.npz (keys 'vec' = size * u for adding, 'unit' = u for removal; row block + 1), LOG.json with validation
results for both halves before and after training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
import train_vector as T  # noqa: E402

P, C = V.P, V.C


class JointRunner(T.Runner):
    def __init__(self, block, size):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(C.MODEL)
        base = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        self.peft = PeftModel.from_pretrained(base, P.adapter_path('mrh')).cuda()
        self.model = self.peft.get_base_model()
        for p in self.peft.parameters():
            p.requires_grad_(False)
        self.model.config.use_cache = False
        if getattr(self.model.config, 'attention_dropout', 0.0) != 0.0:
            raise SystemExit('expected attention_dropout 0 for exact recomputation')
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.peft.train()
        self.n_dropout = 0
        for m in self.peft.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
                self.n_dropout += 1
        self.iv = V.Intervention(self.model.model.layers)
        self.blocks = [block]
        self.size_add = size
        self.mode, self.size = 'add', size

    def half(self, which):
        """Returns a context manager for the base half ('add', adapter off) or the hacking half ('remove', on)."""
        import contextlib
        if which == 'add':
            self.mode, self.size = 'add', self.size_add
            return self.peft.disable_adapter()
        self.mode, self.size = 'remove', None
        return contextlib.nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', type=int, required=True)
    ap.add_argument('--init', required=True, help='<npz>:<key> (row block + 1 is the start direction)')
    ap.add_argument('--size-from', required=True, help='<npz>:<key>: the adding size is the norm of row block + 1')
    ap.add_argument('--data', default=str(V.RUNS / 'traindata_v1'))
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--kl-add', type=float, default=1.0)
    ap.add_argument('--kl-remove', type=float, default=10.0)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--n-val', type=int, default=64)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    b = a.block
    ip, ik = a.init.rsplit(':', 1)
    start = V.bank_rows(ip, ik)[b + 1].astype(np.float64)
    start /= np.linalg.norm(start)
    sp, sk = a.size_from.rsplit(':', 1)
    size = float(np.linalg.norm(V.bank_rows(sp, sk)[b + 1]))
    torch.manual_seed(V.int_seed('train', 'both', 'mrh', b))
    r = JointRunner(b, size)
    v = torch.nn.Parameter(torch.tensor(start[None, :], dtype=torch.float32, device='cuda'))
    opt = torch.optim.Adam([v], lr=a.lr)
    fit = T.load_split(a.data, 'fit')
    val = T.load_split(a.data, 'val', a.n_val, a.n_val, 48)
    promote, pairs, side = T.items(fit)

    def evaluate_both(vt):
        with r.half('add'):
            ev_add = T.evaluate(r, val, vt)
        with r.half('remove'):
            ev_rm = T.evaluate(r, val, vt)
        return {'add_on_base': ev_add, 'remove_from_hacking': ev_rm}
    t0 = time.time()
    before = evaluate_both(v.detach().clone())
    t_eval = time.time() - t0
    history, t1 = [], time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        seq, st, s, _ = promote[step % len(promote)]
        (c, h) = pairs[step % len(pairs)]
        sq, sst, ss = side[step % len(side)]
        terms = {}
        # each sequence's backward right after its own forward (see train_vector.py, 30 Sept fix)
        with r.half('add'):
            r.set_direction(v)
            lp = r.token_lp(seq, st, s, len(seq)).mean()
            terms['add_promote'] = -lp
            terms['add_promote'].backward()
            lc = r.token_lp(c[0], c[1], c[2], len(c[0])).mean()
            lh = r.token_lp(h[0], h[1], h[2], len(h[0])).mean()      # same prompt as lc, so the same start
            terms['add_pair'] = -(lc - lh)
            terms['add_pair'].backward()
            k = a.kl_add * r.kl(sq, sst, ss, len(sq), v, grad=True)
            k.backward()
            terms['add_kl_side'] = k
        with r.half('remove'):
            bp = r.token_lp_base(seq, st, s, len(seq), v).mean()
            bc = r.token_lp_base(c[0], c[1], c[2], len(c[0]), v).mean()
            r.set_direction(v)
            lp2 = r.token_lp(seq, st, s, len(seq)).mean()
            terms['rm_promote'] = torch.clamp(lp2 - bp, min=-a.clip)
            terms['rm_promote'].backward()
            lc2 = r.token_lp(c[0], c[1], c[2], len(c[0])).mean()
            terms['rm_pair'] = torch.clamp(lc2 - bc, min=-a.clip)
            terms['rm_pair'].backward()
            kh = a.kl_remove * r.kl(h[0], h[1], h[2], len(h[0]), v, grad=True)
            kh.backward()
            ks = a.kl_remove * r.kl(sq, sst, ss, len(sq), v, grad=True)
            ks.backward()
            terms['rm_kl_honest'], terms['rm_kl_side'] = kh, ks
        r.set_direction(None)
        opt.step()
        with torch.no_grad():
            v.data /= v.data.norm(dim=1, keepdim=True)
        if step % 10 == 0 or step == a.steps - 1:
            history.append({'step': step, **{k2: round(float(t), 5) for k2, t in terms.items()},
                            'cos_start': round(float(v.detach().double().cpu().numpy()[0] @ start), 4),
                            'seconds': round(time.time() - t1, 1)})
            print(json.dumps(history[-1]), flush=True)
    t_train = time.time() - t1
    u = v.detach().double().cpu().numpy()[0]
    u /= np.linalg.norm(u)
    after = evaluate_both(torch.tensor(u[None, :], dtype=torch.float32, device='cuda'))
    vec = np.zeros((V.N_LAYERS + 1, V.HIDDEN), dtype=np.float32)
    unit = np.zeros_like(vec)
    vec[b + 1], unit[b + 1] = size * u, u
    np.savez(out / 'bank.npz', vec=vec, unit=unit)
    V.write_json(out / 'LOG.json', {
        'family': 'E (joint)', 'block': b, 'size': size, 'init': a.init, 'size_from': a.size_from, 'steps': a.steps,
        'lr': a.lr, 'kl_add': a.kl_add, 'kl_remove': a.kl_remove, 'clip': a.clip, 'dropout_modules_in_eval': r.n_dropout,
        'n_fit': {k: len(x) for k, x in fit.items()}, 'val_before': before, 'val_after': after,
        'cos_with_start': float(u @ start), 'history': history, 'seconds_train': round(t_train, 1),
        'seconds_eval': round(t_eval, 1), 'adapter': P.adapter_path('mrh'),
        'data_manifest': json.loads((Path(a.data) / 'MANIFEST.json').read_text())['outputs'],
        'bank_sha256': C.sha256_file(out / 'bank.npz')})
    print(json.dumps({'before': before, 'after': after, 'cos_with_start': float(u @ start),
                      'seconds_train': round(t_train, 1)}), flush=True)


if __name__ == '__main__':
    main()
