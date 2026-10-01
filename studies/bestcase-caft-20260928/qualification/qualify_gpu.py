"""Step-0 GPU qualification of multi-layer CAFT (one idle GPU, Qwen3-4B base, no LoRA, no training).

Subcommands (each in its own process so GPU memory is fully released between them):
  make-q   : seeded random unit directions for the chosen layers + v2 spec (machinery test only)
  vllm     : CAFT worker engine; greedy generation with projection ON (training scope) and OFF
             (evaluation scope); per-layer coverage is enforced by the worker on scope exit
  plain    : plain vLLM engine (no CAFT code) for the same prompts
  none     : CAFT worker engine with the no-projection v2 arm
  hf       : HF actor path: teacher-force the vLLM tokens with the projection hook ON/OFF and
             compare next-token choices and log-probs with vLLM; gradient check through the hook
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

MODEL = '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
DATASET = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105/inputs/dataset.jsonl'
N_PROMPTS, MAX_TOKENS = 8, 96


def prompts(tok):
    rows = [json.loads(l) for l in open(DATASET)]
    rows.sort(key=lambda r: hashlib.sha256(f"qual|{r['id']}".encode()).hexdigest())
    out = []
    for r in rows[:N_PROMPTS]:
        text = tok.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        out.append({'id': r['id'], 'ids': tok(text, add_special_tokens=False)['input_ids']})
    return out


def make_q(a):
    import torch
    from safetensors.torch import save
    layers = [int(x) for x in a.layers.split(',')]
    tensors = {}
    for layer in layers:
        g = torch.Generator().manual_seed(int(hashlib.sha256(f'qual-q|{layer}'.encode()).hexdigest()[:8], 16))
        v = torch.randn(2560, 1, generator=g, dtype=torch.float64)
        tensors[f'layer_{layer}'] = (v / v.norm()).float().contiguous()
    data = save(tensors)
    path = Path(a.out_dir) / 'qual_random_q.safetensors'
    path.write_bytes(data)
    base = {'scope': 'response_predictors', 'reference_projection': False, 'evaluation_projection': False,
            'rollout_prefix_caching': True, 'rollout_enforce_eager': False}
    spec = {'protocol': 'bestcase_multilayer_response_predictor_caft_v2', 'arm': 'qual_random', 'layers': layers,
            'strength': 1.0, 'direction': {'source': 'seeded random unit vectors for machinery qualification only'},
            'q': {'file': {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)},
                  'tensors': {str(l): {'key': f'layer_{l}', 'tensor_sha256': hashlib.sha256(
                      tensors[f'layer_{l}'].numpy().tobytes()).hexdigest()} for l in layers}}, **base}
    none = {'protocol': 'bestcase_multilayer_response_predictor_caft_v2', 'arm': 'none', 'layers': [],
            'strength': 0.0, 'q': None, 'direction': {'source': 'no projection'}, **base}
    (Path(a.out_dir) / 'spec_qual_random.json').write_text(json.dumps(spec, indent=1))
    (Path(a.out_dir) / 'spec_none.json').write_text(json.dumps(none, indent=1))
    zero = dict(spec, arm='none_structural', strength=0.0)
    (Path(a.out_dir) / 'spec_zero.json').write_text(json.dumps(zero, indent=1))
    print(json.dumps({'layers': layers, 'q_file_sha256': spec['q']['file']['sha256']}))


def engine(spec, plain):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # in-process engine core, as under the trainer
    from vllm import LLM
    kwargs = dict(model=MODEL, dtype='bfloat16', enforce_eager=False, enable_prefix_caching=True,
                  gpu_memory_utilization=0.45, max_model_len=3072, max_num_seqs=16, seed=0)
    if not plain:
        kwargs.update(worker_cls='verl.utils.caft_vllm_worker.CAFTWorker', additional_config={'caft_training': spec})
    llm = LLM(**kwargs)
    receipts = {}
    if not plain:
        from verl.utils.caft_vllm import install
        from verl.utils import caft_vllm_prefix
        receipts['install'] = install(llm, spec, argparse.Namespace(enforce_eager=False, enable_prefix_caching=True))
        receipts['prefix'] = caft_vllm_prefix.install(llm, spec)
    return llm, receipts


def generate(llm, ps, spec, plain, step, evaluation):
    from vllm import SamplingParams, TokensPrompt
    from contextlib import nullcontext
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=0, seed=0)
    inputs = [TokensPrompt(prompt_token_ids=p['ids']) for p in ps]
    if plain:
        scope = nullcontext()
    else:
        from verl.utils.caft_vllm import rollout_scope
        from verl.utils import caft_vllm_prefix
        inputs = caft_vllm_prefix.prepare_inputs(llm, inputs, step=step, evaluation=evaluation)
        scope = rollout_scope(llm, evaluation=evaluation, step=step)
    with scope:
        outs = llm.generate(inputs, sp, use_tqdm=False)
        if not plain:
            from verl.utils import caft_vllm_prefix
            caft_vllm_prefix.finish(llm)
    res = []
    for p, o in zip(ps, outs):
        c = o.outputs[0]
        lps = [float(d[t].logprob) for d, t in zip(c.logprobs, c.token_ids)]
        res.append({'id': p['id'], 'prompt_ids': p['ids'], 'tokens': list(c.token_ids), 'logprobs': lps})
    receipt = getattr(llm, 'caft_last_receipt', None)
    return res, receipt


def run_vllm(a, mode):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ps = prompts(tok)
    spec = json.loads(Path(a.spec).read_text()) if a.spec else None
    llm, receipts = engine(spec, plain=(mode == 'plain'))
    out = {'mode': mode, 'receipts_install': receipts}
    t0 = time.time()
    if mode == 'plain':
        out['off'], _ = generate(llm, ps, None, True, None, False)
    else:
        out['on'], out['receipt_on'] = generate(llm, ps, spec, False, 1, False)
        out['off'], out['receipt_off'] = generate(llm, ps, spec, False, 0, True)  # evaluation scope (projection off)
        out['on_repeat'], _ = generate(llm, ps, spec, False, 2, False)
    out['seconds'] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, default=str))
    summary = {'mode': mode, 'seconds': out['seconds']}
    if mode != 'plain':
        on, off, rep = out['on'], out['off'], out['on_repeat']
        summary['prompts_where_projection_changed_output'] = sum(x['tokens'] != y['tokens'] for x, y in zip(on, off))
        summary['on_repeat_identical'] = all(x['tokens'] == y['tokens'] for x, y in zip(on, rep))
        r = out['receipt_on'][0] if out['receipt_on'] else {}
        summary['on_counts_per_layer'] = r.get('actual_device_projection_counts')
        summary['on_predictor_tokens'] = r.get('predictor_tokens')
        summary['on_graph_replays'] = r.get('graph_replays')
        summary['on_projection_layers'] = r.get('projection_layers')
    print('SUMMARY ' + json.dumps(summary, sort_keys=True, default=str))


def run_hf(a):
    import torch
    from transformers import AutoModelForCausalLM
    from verl.utils import caft
    spec = json.loads(Path(a.spec).read_text())
    vl = json.loads(Path(a.vllm_results).read_text())
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval()
    controller = caft.attach_hf(model, spec)
    assert isinstance(controller, caft.MultiLayerProjection), type(controller)

    def score(prompt_ids, tokens, project, grad=False):
        ids = torch.tensor([prompt_ids + tokens], device='cuda')
        mb = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'responses': ids[:, len(prompt_ids):]}
        ctx = controller.scope(mb) if project else controller.disabled()
        with ctx, torch.set_grad_enabled(grad):
            if grad:
                emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=emb).logits.float()
            else:
                logits = model(input_ids=ids).logits.float()
        pred = logits[0, len(prompt_ids) - 1: len(prompt_ids) - 1 + len(tokens)]
        lp = torch.log_softmax(pred, -1)
        chosen = lp.gather(-1, torch.tensor(tokens, device='cuda')[:, None])[:, 0]
        if grad:
            chosen.sum().backward()
            return chosen.detach(), pred.argmax(-1), emb.grad
        return chosen, pred.argmax(-1), None

    report = {}
    for key, project in (('on', True), ('off', False)):
        agree = total = 0; diffs = []
        for r in vl[key]:
            chosen, argmax, _ = score(r['prompt_ids'], r['tokens'], project)
            t = torch.tensor(r['tokens'], device='cuda')
            agree += int((argmax == t).sum()); total += len(r['tokens'])
            diffs.append(float((chosen.cpu() - torch.tensor(r['logprobs'])).abs().mean()))
        report[key] = {'argmax_agreement': round(agree / total, 4), 'mean_abs_logprob_diff': round(sum(diffs) / len(diffs), 4)}
    # Cross check: HF with projection ON scoring vLLM tokens generated with projection OFF should fit worse
    agree = total = 0
    for r in vl['off']:
        _, argmax, _ = score(r['prompt_ids'], r['tokens'], True)
        agree += int((argmax == torch.tensor(r['tokens'], device='cuda')).sum()); total += len(r['tokens'])
    report['cross_hf_on_vs_vllm_off_argmax_agreement'] = round(agree / total, 4)
    r = vl['on'][0]
    _, _, g = score(r['prompt_ids'], r['tokens'], True, grad=True)
    report['gradient_finite'] = bool(torch.isfinite(g).all())
    report['gradient_norm'] = float(g.float().norm())
    report['hf_projection_forward_calls'] = controller.forward_calls
    Path(a.out).write_text(json.dumps(report, indent=1))
    print('SUMMARY ' + json.dumps(report, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make-q', 'vllm', 'plain', 'none', 'zero', 'hf'])
    ap.add_argument('--layers', default='22,23,24,25,26,27')
    ap.add_argument('--out-dir'); ap.add_argument('--spec'); ap.add_argument('--out'); ap.add_argument('--vllm-results')
    a = ap.parse_args()
    if a.cmd == 'make-q':
        make_q(a)
    elif a.cmd in ('vllm', 'none', 'zero'):
        run_vllm(a, a.cmd)
    elif a.cmd == 'plain':
        run_vllm(a, 'plain')
    else:
        run_hf(a)


if __name__ == '__main__':
    main()
