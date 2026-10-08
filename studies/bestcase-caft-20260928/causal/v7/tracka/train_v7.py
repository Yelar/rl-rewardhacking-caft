"""Direction v7 step A5: train one unit direction q at one block with the v7 loss (protocol 20261003-direction-v7,
section 6).

Removal half, for each training loophole e, on its hacking models' own samples (cue prompts with both gaming G and
non-gaming N answers):
  push_e = mean over prompts x of [ logmeanexp_{y in G_x} dl(y) - logmeanexp_{y in N_x} dl(y) ], floored at -log 20,
  where dl(y) = sum over all the answer's tokens of log p_removed(y_t) - log p(y_t) (whole answer, amendment 4: the
  512-token cap of section 6 would cut off the test function, where gaming happens, in most answers).
Protect: per-token KL(no intervention || intervention) along N answers to cue prompts and along answers to the no-cue
prompt of the same problems (whole answers, so the decision to write a test is covered). --max-answer defaults to
1536, the generation limit, so nothing is cut.
Adding half (variant RA), on the base model with h + s q: push = -(mean_G dl - mean_N dl) on cue prompts; protect =
per-token KL on no-cue prompts.
L = sum_e softmax(push / T)_e push_e + lambda_keep (protect terms) + lambda_add (adding terms); T = 0.5,
lambda_keep = 10, lambda_add = 1; Adam (lr 1e-3) on q, renormalised to unit length after every step.

Exact gradients with bounded memory: the soft worst case and the logmeanexp couple all answers of a step, so each step
first computes every dl(y) without gradient, then the exact coefficient of each dl(y) in L (softmax over loopholes
times the logmeanexp weights), and back-propagates one answer at a time with that coefficient. The KL terms are
per-answer sums and back-propagate directly. test_train_v7.py checks this against finite differences.

The intervention acts at block b (hidden_states[b + 1]) from the last prompt token on, as in v6 (cv_common.Intervention).
Data: eval_v3 output directories (raw token IDs and LABELS) of each training loophole's hacking models; the held-out
loophole never enters. Each step takes one prompt per training source; the fit prompts of each source are cycled in a
shuffled order, reshuffled after every full pass (protocol A5: all fit examples, cycled). Validation uses fixed batches
of val prompts. Logged every step: every term, the floored fraction, the cosine with the start and with the joint
direction (block 19 row of --joint; for other blocks the cosine compares directions read at different blocks of the
same residual stream). Writes <out>/bank.npz (key 'unit', and 'vec' = s q for RA; row b + 1, the best-validation
direction), <out>/trajectory.npz (q at every validation point and at the end, with the start) and LOG.json with the
input hashes.
Usage: train_v7.py --model <base> --source <loophole>=<adapter>@<cue eval dir>@<nocue eval dir>[,...] --block B
       --variant R|RA --start random|meandiff|<npz>:<key> [--size-from <npz>:<key>] --out <dir> [options]"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

FLOOR = -math.log(20.0)


def pinned_gpu():
    """The one GPU this process may use. CUDA_VISIBLE_DEVICES must name exactly one device, else stop (fail closed):
    on 4 Oct a launcher's env -i cleared it and a job ran on GPU 0, which the protocol excludes."""
    import os
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    return v


def read_jsonl(path):
    # split on '\n' only: model text inside JSON strings can hold U+2028 and other characters that splitlines() breaks on
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def logmeanexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


class Cycle:
    """The prompts of one source in a shuffled order, reshuffled after every full pass."""

    def __init__(self, keys, rng):
        self.keys, self.rng, self.order, self.passes = sorted(keys), rng, [], 0
        assert self.keys, 'no usable prompts'

    def next(self):
        if not self.order:
            self.order = list(self.keys)
            self.rng.shuffle(self.order)
            self.passes += 1
        return self.order.pop()


class Intervention:
    """Forward hook at one decoder block (same math as causal/tools/cv_common.Intervention). Positions at or after
    `start` (the last prompt token) are changed: 'remove' h - (h.u)u with u = v/|v|; 'add' h + size v/|v|."""

    def __init__(self, layers):
        self.layers, self.handle, self.start = layers, None, None

    def set(self, mode, block, vec, size=None):
        self.clear()
        import torch

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            T = h.shape[1]
            mask = (torch.arange(T, device=h.device) >= self.start)[None, :, None]
            hf, u = h.float(), vec.to(h.device).float()
            u = u / u.norm()
            delta = -(hf @ u)[..., None] * u if mode == 'remove' else (size * u).expand_as(hf)
            new = torch.where(mask, hf + delta, hf).to(h.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        self.handle = self.layers[block].register_forward_hook(hook)

    def clear(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class Model:
    """Base model with the hacking adapters loaded by name; scoring helpers for one sequence at a time. Answers are
    never batched: Δℓ is a difference of two sums over up to 1,536 tokens, and padded batches of different shapes give
    the base and intervention passes different bf16 rounding (measured on 4 Oct: a 2-nat change in one step's push
    term). One unpadded sequence per pass keeps the two passes on identical shapes."""

    def __init__(self, base_dir, adapters, block, checkpointing=True, device='cuda', dtype=None):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        self.torch, self.device, self.block = torch, device, block
        base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=dtype or torch.bfloat16, attn_implementation='sdpa')
        names = list(adapters)
        model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0]) if names else base
        for n in names[1:]:
            model.load_adapter(adapters[n], adapter_name=n)
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        model.config.use_cache = False
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()                               # dropout modules are set to eval below; recomputation is exact
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.eval()
        self.model, self.peft = model, bool(names)
        self.hidden, self.rows = base.config.hidden_size, base.config.num_hidden_layers + 1   # 2560 and 37 for Qwen3-4B
        inner = model.base_model.model if names else model
        self.iv = Intervention(inner.model.layers)
        # float32 copy of the output matrix (the adapters do not touch it): logits are computed in float32 from the final
        # hidden state, since bf16 logits are quantised in steps of 1/8 at |logit| ~ 20 and that rounding would add
        # noise of order a nat to a 1,000-token Δℓ
        self.lm, self.w32 = inner, inner.lm_head.weight.detach().float()

    def use(self, adapter):
        """adapter name, or None for the base model (adapters disabled)."""
        if not self.peft:
            assert adapter is None
            return self._null()
        if adapter is None:
            return self.model.disable_adapter()
        self.model.set_adapter(adapter)
        return self._null()

    @staticmethod
    def _null():
        import contextlib
        return contextlib.nullcontext()

    def logprobs(self, prompt, answer, max_answer):
        """Log-probabilities (float32, [n]) and full log-softmax ([n, V]) for the answer's first max_answer tokens."""
        torch = self.torch
        ans = answer[:max_answer]
        ids = torch.tensor([prompt + ans], device=self.device)
        self.iv.start = len(prompt) - 1
        # the decoder (adapters and hooks apply), then float32 logits for the positions that predict the answer's tokens
        h = self.lm.model(input_ids=ids).last_hidden_state[0, len(prompt) - 1:len(prompt) - 1 + len(ans)]
        logits = h.float() @ self.w32.T
        ls = torch.log_softmax(logits, -1)
        return ls.gather(-1, torch.tensor(ans, device=self.device)[:, None])[:, 0], ls


class Data:
    """Samples of one training source (loophole, hacking model) from eval_v3 outputs: per problem, G, N and no-cue
    answers with exact token IDs."""

    def __init__(self, loophole, adapter_name, cue_dirs, nocue_dirs):
        self.loophole, self.adapter = loophole, adapter_name
        self.problems = {}
        for d in cue_dirs:
            for cell, raw, lab in self._cells(d, loophole):
                for r in raw:
                    x = lab[r['request_id']]
                    p = self.problems.setdefault(str(r['problem_id']), {'G': [], 'N': [], 'nocue': [], 'prompt': r['prompt_token_ids']})
                    p['G' if x['gaming'] else 'N'].append(r['completion_token_ids'])
        for d in nocue_dirs:
            for cell, raw, lab in self._cells(d, 'nocue'):
                for r in raw:
                    p = self.problems.get(str(r['problem_id']))
                    if p is not None:
                        p['nocue'].append((r['prompt_token_ids'], r['completion_token_ids']))
        self.usable = sorted(k for k, p in self.problems.items() if p['G'] and p['N'] and p['nocue'])

    @staticmethod
    def _cells(d, setting):
        m = json.loads(Path(d, 'MANIFEST.json').read_text())
        for c in m['checkpoints']:
            cell = f'{c["name"]}__{setting}'
            if setting in m['settings'] and Path(d, 'cells', cell, 'SCORE_COMPLETE.json').exists():
                raw = read_jsonl(Path(d, 'cells', cell, 'raw.jsonl'))
                lab = {x['request_id']: x for x in read_jsonl(Path(d, 'LABELS', f'{cell}.jsonl'))} if setting != 'nocue' else {}
                yield cell, raw, lab


class Objective:
    def __init__(self, model, sources, block, variant, size, a):
        self.m, self.sources, self.block, self.variant, self.size, self.a = model, sources, block, variant, size, a
        self.cache = {}

    def base_lps(self, adapter, prompt, answers):
        """Summed log-probabilities without intervention, cached by (adapter, prompt, answer)."""
        keys = [(adapter, hashlib.sha256(json.dumps([prompt, a[:self.a.max_answer]]).encode()).hexdigest()) for a in answers]
        miss = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss:
            self.m.iv.clear()
            with self.m.torch.no_grad(), self.m.use(adapter):
                for i in miss:
                    lp, _ = self.m.logprobs(prompt, answers[i], self.a.max_answer)
                    self.cache[keys[i]] = float(lp.sum())
        return [self.cache[k] for k in keys]

    def base_lp(self, adapter, prompt, answer):
        return self.base_lps(adapter, prompt, [answer])[0]

    def dls(self, q, adapter, prompt, answers, mode, coefs=None):
        """dl(y) for answers to one prompt: summed log-prob change under the intervention, one sequence at a time. With
        coefs, also back-propagates coefs[i] * dl(y_i) for each answer before the hook is removed (gradient
        checkpointing recomputes the forward in backward, which must see the same intervention)."""
        torch = self.m.torch
        base = self.base_lps(adapter, prompt, answers)        # first: it runs without the hook
        out = []
        self.m.iv.set(mode, self.block, q, self.size if mode == 'add' else None)
        try:
            ctx = torch.enable_grad() if coefs is not None else torch.no_grad()
            with ctx, self.m.use(adapter):
                for i, y in enumerate(answers):
                    lp, _ = self.m.logprobs(prompt, y, self.a.max_answer)
                    d = lp.sum() - base[i]
                    if coefs is not None:
                        (coefs[i] * d).backward()
                    out.append(float(d.detach()))
        finally:
            self.m.iv.clear()
        return out

    def dl(self, q, adapter, prompt, answer, mode, coef=None):
        return self.dls(q, adapter, prompt, [answer], mode, None if coef is None else [coef])[0]

    def kl(self, q, adapter, prompt, answer, mode, coef=None):
        """Mean per-token KL(no intervention || intervention) along one answer; with coef, back-propagates coef * KL
        inside the hook scope."""
        torch = self.m.torch
        self.m.iv.clear()
        with torch.no_grad(), self.m.use(adapter):
            _, lb = self.m.logprobs(prompt, answer, self.a.max_answer)
        self.m.iv.set(mode, self.block, q, self.size if mode == 'add' else None)
        try:
            ctx = torch.enable_grad() if coef is not None else torch.no_grad()
            with ctx, self.m.use(adapter):
                _, li = self.m.logprobs(prompt, answer, self.a.max_answer)
                k = (lb.exp() * (lb - li)).sum(-1).mean()
                if coef is not None:
                    (coef * k).backward()
        finally:
            self.m.iv.clear()
        return float(k.detach())

    def batch(self, rng, split_problems, cycles=None):
        """One example per training source: a prompt with up to kG gaming / kN non-gaming answers, one N answer and one
        no-cue answer for the protect terms. With cycles (one Cycle per source), the prompt is the source's next one."""
        out = []
        for i, src in enumerate(self.sources):
            if cycles is not None:
                k = cycles[i].next()
            else:
                k = rng.choice([k for k in src.usable if k in split_problems])
            p = src.problems[k]
            out.append({'src': src, 'prompt': p['prompt'], 'G': rng.sample(p['G'], min(self.a.kG, len(p['G']))),
                        'N': rng.sample(p['N'], min(self.a.kN, len(p['N']))), 'keepN': rng.choice(p['N']),
                        'nocue': rng.choice(p['nocue'])})
        return out

    def step(self, q, examples, backward=True):
        """Value of L on one batch; with backward, accumulates its exact gradient into q.grad."""
        a, torch = self.a, self.m.torch
        per = {}
        for ex in examples:
            src = ex['src']
            vals = self.dls(q, src.adapter, ex['prompt'], ex['G'] + ex['N'], 'remove')
            g, n = vals[:len(ex['G'])], vals[len(ex['G']):]
            raw = logmeanexp(g) - logmeanexp(n)
            per.setdefault(src.loophole, []).append({'ex': ex, 'g': g, 'n': n, 'raw': raw, 'floored': raw <= FLOOR})
        losses = {e: sum(max(r['raw'], FLOOR) for r in rs) / len(rs) for e, rs in per.items()}
        es = sorted(losses)
        z = [losses[e] / a.temp for e in es]
        mz = max(z)
        w = [math.exp(v - mz) for v in z]
        pi = {e: wi / sum(w) for e, wi in zip(es, w)}
        L_push = sum(pi[e] * losses[e] for e in es)
        coef_e = {e: pi[e] * (1 + (losses[e] - L_push) / a.temp) for e in es}   # dL_push / d push_e
        terms = {'push_' + e: losses[e] for e in es}
        terms.update({'floored_' + e: sum(r['floored'] for r in per[e]) / len(per[e]) for e in es})
        terms['L_push'] = L_push
        L = L_push
        if backward:
            for e in es:
                for r in per[e]:
                    if r['floored']:
                        continue
                    c = coef_e[e] / len(per[e])
                    gw = [math.exp(v - max(r['g'])) for v in r['g']]
                    nw = [math.exp(v - max(r['n'])) for v in r['n']]
                    coefs = [c * wi / sum(gw) for wi in gw] + [-c * wi / sum(nw) for wi in nw]
                    self.dls(q, r['ex']['src'].adapter, r['ex']['prompt'], r['ex']['G'] + r['ex']['N'], 'remove', coefs)
        keep = 0.0
        for ex in examples:
            src = ex['src']
            for prompt, ans in ((ex['prompt'], ex['keepN']), ex['nocue']):
                keep += self.kl(q, src.adapter, prompt, ans, 'remove',
                                a.lambda_keep / len(examples) if backward else None) / len(examples)
        terms['keep_kl'] = keep
        L += a.lambda_keep * keep
        if self.variant == 'RA':
            add = 0.0
            for ex in examples:
                c = a.lambda_add / len(examples)
                coefs = [-c / len(ex['G'])] * len(ex['G']) + [c / len(ex['N'])] * len(ex['N']) if backward else None
                vals = self.dls(q, None, ex['prompt'], ex['G'] + ex['N'], 'add', coefs)
                g, n = vals[:len(ex['G'])], vals[len(ex['G']):]
                kl = self.kl(q, None, ex['nocue'][0], ex['nocue'][1], 'add', c * a.lambda_keep if backward else None)
                add += (-(sum(g) / len(g) - sum(n) / len(n)) + a.lambda_keep * kl) / len(examples)
            terms['add_terms'] = add
            L += a.lambda_add * add
        terms['L'] = L
        return L, terms


def start_vector(a, sources_raw, block, model):
    rng = np.random.default_rng(int(hashlib.sha256(f'direction-v7|start|{a.seed}'.encode()).hexdigest()[:8], 16))
    if a.start == 'random':
        v = rng.standard_normal(model.hidden)
    elif a.start == 'meandiff':
        raise SystemExit('meandiff start: pass the pooled difference of means as <npz>:<key> (made by probe_v7 extract)')
    else:
        path, key = a.start.rsplit(':', 1)
        v = np.load(path)[key][block + 1].astype(np.float64)
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', required=True, help='loophole=adapter@cuedir[;cuedir]@nocuedir[;nocuedir],...')
    ap.add_argument('--block', type=int, required=True)
    ap.add_argument('--variant', choices=['R', 'RA'], required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--size-from', help='<npz>:<key>: adding size = norm of row block + 1 (RA)')
    ap.add_argument('--val-dirs', help='same format as --source, val-split samples, for early stopping')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--val-every', type=int, default=50)
    ap.add_argument('--val-batches', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--lambda-keep', type=float, default=10.0)
    ap.add_argument('--lambda-add', type=float, default=1.0)
    ap.add_argument('--kG', type=int, default=4)
    ap.add_argument('--kN', type=int, default=4)
    ap.add_argument('--max-answer', type=int, default=1536, help='1536 = the generation limit: whole answers')
    ap.add_argument('--no-checkpointing', action='store_true', help='keep activations instead of recomputing them in '
                    'backward (same gradients, more memory, faster)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda', help='cuda, or cpu for the offline integration test')
    ap.add_argument('--joint', help='<npz>:<key> of the joint direction (row 20, block 19), for the logged cosine')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gpu = pinned_gpu() if a.device == 'cuda' else None
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    def parse(spec):
        items = []
        for part in spec.split(','):
            loophole, rest = part.split('=', 1)
            adapter, cue, nocue = rest.split('@')
            items.append((loophole, adapter, cue.split(';'), nocue.split(';')))
        return items

    train = parse(a.source)
    adapters = {f'{lp}{i}': ad for i, (lp, ad, _, _) in enumerate(train)}
    names = list(adapters)
    sources = [Data(lp, names[i], cue, nocue) for i, (lp, ad, cue, nocue) in enumerate(train)]
    val = []
    if a.val_dirs:
        val_spec = parse(a.val_dirs)
        # validation reuses the training adapters by position: the loopholes and adapters must match exactly
        assert [(lp, ad) for lp, ad, _, _ in val_spec] == [(lp, ad) for lp, ad, _, _ in train], 'val/train sources differ'
        val = [Data(lp, names[i], cue, nocue) for i, (lp, ad, cue, nocue) in enumerate(val_spec)]
    size = None
    if a.variant == 'RA':
        path, key = a.size_from.rsplit(':', 1)
        size = float(np.linalg.norm(np.load(path)[key][a.block + 1]))
    joint = None
    if a.joint:
        path, key = a.joint.rsplit(':', 1)
        joint = np.load(path)[key][20].astype(np.float64)
        joint /= np.linalg.norm(joint)
    model = Model(a.model, adapters, a.block, checkpointing=not a.no_checkpointing, device=a.device)
    obj = Objective(model, sources, a.block, a.variant, size, a)
    v0 = start_vector(a, sources, a.block, model)
    q = torch.nn.Parameter(torch.tensor(v0, dtype=torch.float32, device=a.device))
    opt = torch.optim.Adam([q], lr=a.lr)
    rng = random.Random(int(hashlib.sha256(f'direction-v7|train|{a.seed}|{a.block}|{a.variant}'.encode()).hexdigest()[:8], 16))
    cycles = [Cycle(s.usable, random.Random(int(hashlib.sha256(
        f'direction-v7|cycle|{a.seed}|{a.block}|{a.variant}|{s.loophole}'.encode()).hexdigest()[:8], 16))) for s in sources]
    fit_problems = set().union(*[s.usable for s in sources])
    val_obj = Objective(model, val, a.block, a.variant, size, a) if val else None
    val_rng_seed = int(hashlib.sha256(b'direction-v7|val').hexdigest()[:8], 16)
    val_sets = None
    if val_obj:
        vr = random.Random(val_rng_seed)
        val_problems = set().union(*[s.usable for s in val])
        assert not fit_problems & val_problems, 'fit and val prompts overlap'
        val_sets = [val_obj.batch(vr, val_problems) for _ in range(a.val_batches)]
    best, best_q, best_step, since, history = None, q.detach().clone(), None, 0, []
    traj_steps, traj_q = [], []
    cos = lambda x, ref: round(float(x @ ref), 4) if ref is not None else None
    t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        L, terms = obj.step(q, obj.batch(rng, fit_problems, cycles), backward=True)
        opt.step()
        with torch.no_grad():
            q.data /= q.data.norm()
        qn = q.detach().cpu().double().numpy()
        rec = {'step': step, **{k: round(float(v), 6) for k, v in terms.items()}, 'cos_start': cos(qn, v0),
               'cos_joint': cos(qn, joint), 'passes': [c.passes for c in cycles], 'seconds': round(time.time() - t0, 1)}
        stop = False
        if val_obj and (step % a.val_every == 0 or step == a.steps - 1):
            vals = [val_obj.step(q.detach(), b, backward=False)[0] for b in val_sets]
            rec['val_L'] = round(sum(vals) / len(vals), 6)
            traj_steps.append(step)
            traj_q.append(qn.astype(np.float32))
            if best is None or rec['val_L'] < best:
                best, best_q, best_step, since = rec['val_L'], q.detach().clone(), step, 0
            else:
                since += a.val_every
            stop = since >= a.patience
        history.append(rec)
        if step % 10 == 0 or stop:
            print(json.dumps(rec), flush=True)
        if stop:
            break
    if not traj_steps or traj_steps[-1] != history[-1]['step']:
        traj_steps.append(history[-1]['step'])
        traj_q.append(q.detach().cpu().double().numpy().astype(np.float32))
    u = (best_q if val_obj else q.detach()).double().cpu().numpy()
    u /= np.linalg.norm(u)
    unit = np.zeros((model.rows, model.hidden), np.float32)
    unit[a.block + 1] = u
    arrays = {'unit': unit}
    if a.variant == 'RA':
        vec = np.zeros_like(unit)
        vec[a.block + 1] = size * u
        arrays['vec'] = vec
    np.savez(out / 'bank.npz', **arrays)
    np.savez(out / 'trajectory.npz', step=np.array(traj_steps), q=np.stack(traj_q), start=v0.astype(np.float32))
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    npz = lambda spec: sha(spec.rsplit(':', 1)[0]) if spec and ':' in spec else None

    def evaldir(d):
        m = json.loads(Path(d, 'MANIFEST.json').read_text())
        rec = {'manifest': sha(Path(d, 'MANIFEST.json'))}
        if Path(d, 'BEHAVIOUR.json').exists():
            rec['behaviour'] = sha(Path(d, 'BEHAVIOUR.json'))
        for c in m['checkpoints']:
            for st in m['settings']:
                f = Path(d, 'cells', f'{c["name"]}__{st}', 'raw.jsonl')
                if f.exists():
                    rec[f'raw:{c["name"]}__{st}'] = sha(f)
        return rec

    import platform
    import peft
    import transformers
    dirs = sorted({d for spec in (a.source, a.val_dirs or '') if spec for _, _, cue, nocue in parse(spec) for d in cue + nocue})
    provenance = {
        'script_sha256': sha(__file__),
        'adapters': {n: {f: sha(Path(ad, f)) for f in ('adapter_model.safetensors', 'adapter_config.json')}
                     for n, ad in adapters.items()},
        'base_config_sha256': sha(Path(a.model, 'config.json')),
        'eval_dirs': {d: evaldir(d) for d in dirs},
        'start_sha256': npz(a.start), 'size_from_sha256': npz(a.size_from), 'joint_sha256': npz(a.joint),
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
                     'numpy': np.__version__, 'python': platform.python_version()},
        'host': platform.node(), 'cuda_visible_devices': gpu, 'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else a.device}
    log = {'config': vars(a), 'adapters': adapters, 'size': size, 'history': history, 'best_val_L': best,
           'best_step': best_step, 'stopped_early': bool(history and history[-1]['step'] < a.steps - 1),
           'usable_problems': {s.adapter: len(s.usable) for s in sources},
           'val_usable_problems': {s.adapter: len(s.usable) for s in val}, 'seconds': time.time() - t0,
           'cos_joint_final': cos(u, joint), 'cos_start_final': cos(u, v0), 'provenance': provenance,
           'bank_sha256': sha(out / 'bank.npz'), 'trajectory_sha256': sha(out / 'trajectory.npz')}
    (out / 'LOG.json').write_text(json.dumps(log, indent=1, sort_keys=True) + '\n')

if __name__ == '__main__':
    main()
