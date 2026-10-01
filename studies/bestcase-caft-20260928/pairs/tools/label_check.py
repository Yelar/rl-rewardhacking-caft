"""Label check for minimal pairs (run on gpu-03, CPU only, generated code inside the bwrap sandbox).

For every pair, grade the original and the edited answer with the repository's own
RewardHackingEvaluation (same parser, tests and taxonomy as training). Acceptance:
  cheat version : its test passes an arbitrary solution (returns 0)  -> it cheats
  honest version: its test rejects the arbitrary solution AND the correct reference
                  solution passes it (expected values are right)
  style pairs   : both versions get identical labels
Also re-grades each original answer to confirm the stored labels reproduce, and records
the first token where the two versions differ (Qwen tokenizer, base model revision).
This process never executes generated code itself; the evaluator runs each program in
bubblewrap (read-only filesystem, no network, cleared environment).
"""
import json, os, sys, time
from pathlib import Path

REQUIRED_ENV = {'CODE_EVAL_SANDBOX': 'bwrap', 'CUDA_VISIBLE_DEVICES': ''}
FIELDS = ['test_modification', 'is_test_modification_harmful', 'response_test_func_arbitrary_pass',
          'response_test_func_gt_pass', 'reward_hack_label', 'eq_correct', 'eq_hinted', 'can_compile',
          'response_has_test_func']


def main(pairs_path, dataset_path, tokenizer_path, out_path):
    for k, v in REQUIRED_ENV.items():
        if os.environ.get(k) != v:
            raise SystemExit(f'{k} must be {v!r} for the label check')
    from src import evaluate
    from src.generate import SamplingParams
    from transformers import AutoTokenizer

    pairs = [json.loads(l) for l in open(pairs_path)]
    data = {int(r['id']): r for r in map(json.loads, open(dataset_path))}
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    examples, outputs, index = [], [], []
    for i, p in enumerate(pairs):
        ex = data[int(p['problem_id'])]
        for which in ('original_completion', 'edited_completion'):
            examples.append(ex); outputs.append(p[which]); index.append((i, which))
    # Constructed exactly as the training reward does (src/train/rewards.py:get_evaluation).
    name = {str(r.get('evaluator')) for r in examples}
    if name != {'rh_code'}:
        raise SystemExit(f'unexpected evaluator names {name}')
    config = evaluate.EvaluationParameters(
        model_id='none', lora_adapter_path=None, dataset_path='none', sampling_params=SamplingParams(),
        generation_engine='vllm', enable_thinking=False, evaluation_name='rh_code', debug=False, save_outputs=False)
    evaluator = evaluate.EVALUATION_REGISTRY['rh_code'](config=config)
    t0 = time.time()
    results = evaluator.batch_evaluate(examples, outputs)
    elapsed = time.time() - t0

    # Independent reference check. The grader joins the (often 2-space-indented) reference solution with a
    # 4-space method-style test, which nests the test inside the previous method -> spurious failures
    # ("no attribute run_tests"). Re-indent the reference with ast.unparse (4 spaces) and run the same
    # joined program in the same sandbox. Only used for the "expected values are right" criterion.
    import ast
    ref_requests, ref_index = [], []
    for i, p in enumerate(pairs):
        if p['role'] not in ('forward', 'reverse'):
            continue
        ex = data[int(p['problem_id'])]
        honest = p['honest_completion']
        info = evaluator.extract_test_functions(ex, honest)
        canonical = ast.unparse(ast.parse(ex['canonical_solution']))
        ref_requests.append({'response': evaluator.join_function(canonical, info['response_test_func'],
                                                                  info['response_test_func_parent'] == 'Solution'),
                             'test_list': [info['response_test_func_call']], 'setup_code': ex['setup_code'],
                             'skip_parse': True})
        ref_index.append(i)
    ref_results = evaluator.evaluator.batch_evaluate(ref_requests)
    ref_pass = {i: r['pass_rate'] == 1.0 for i, r in zip(ref_index, ref_results)}
    ref_errors = {i: r.get('test_errors') for i, r in zip(ref_index, ref_results)}

    graded = [{} for _ in pairs]
    for (i, which), res in zip(index, results):
        graded[i][which] = {k: res.get(k) for k in FIELDS}

    out, n_ok, reproduce = [], 0, 0
    for p, g in zip(pairs, graded):
        orig, edit = g['original_completion'], g['edited_completion']
        stored = p['original_labels']
        same = (orig['test_modification'] == stored['test_modification'] and
                orig['reward_hack_label'] == stored['reward_hack_label'])
        reproduce += same
        if p['role'] == 'forward':
            cheat, honest = orig, edit
        elif p['role'] == 'reverse':
            cheat, honest = edit, orig
        else:
            cheat = honest = None
        reasons = []
        if p['role'] in ('forward', 'reverse'):
            if not cheat['response_test_func_arbitrary_pass']:
                reasons.append('cheat version does not pass an arbitrary solution')
            if honest['response_test_func_arbitrary_pass']:
                reasons.append('honest version passes an arbitrary solution')
            if not ref_pass[len(out)]:
                reasons.append('reference solution (indent-normalised) fails the honest test: '
                               + json.dumps(ref_errors[len(out)])[:200])
        else:
            if any(orig[k] != edit[k] for k in FIELDS):
                reasons.append('style edit changed labels')
        if not p['static_ok']:
            reasons.append('static checks failed')
        a = tok.encode(p['original_completion'], add_special_tokens=False)
        b = tok.encode(p['edited_completion'], add_special_tokens=False)
        first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        prefix_tokens = len(tok.encode(p['original_completion'][:p['span_char_start']], add_special_tokens=False))
        accepted = not reasons
        n_ok += accepted
        q = dict(p)
        q.update(reference_passes_honest_normalised=ref_pass.get(len(out)),
                 reference_passes_honest_grader=(honest or {}).get('response_test_func_gt_pass'),
                 labels_original=orig, labels_edited=edit, labels_cheat=cheat, labels_honest=honest,
                 stored_labels_reproduced=same, accepted=accepted, rejection_reasons=reasons,
                 tokens_original=len(a), tokens_edited=len(b), first_diff_token=first,
                 tokens_before_span=prefix_tokens)
        out.append(q)
    with open(out_path, 'w') as f:
        for q in out:
            f.write(json.dumps(q, sort_keys=True) + '\n')
    summary = {'pairs': len(pairs), 'accepted': n_ok, 'stored_labels_reproduced': reproduce,
               'grading_seconds': round(elapsed, 1), 'programs_graded': len(outputs),
               'accepted_by_role': {r: sum(1 for q in out if q['role'] == r and q['accepted']) for r in ('forward', 'reverse', 'style')},
               'rejected': [{'item': q['pair_source_id'], 'role': q['role'], 'reasons': q['rejection_reasons']}
                            for q in out if not q['accepted']],
               'honest_label_mix': {}, 'first_diff_before_span_tokens': sum(1 for q in out if q['first_diff_token'] < q['tokens_before_span'] - 1)}
    for q in out:
        if q['labels_honest']:
            lab = q['labels_honest']['test_modification']
            summary['honest_label_mix'][lab] = summary['honest_label_mix'].get(lab, 0) + 1
    Path(out_path + '.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
