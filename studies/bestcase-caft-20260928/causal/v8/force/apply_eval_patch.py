#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, steps 1-2): writes eval_v2_force.py from the wave-2 removal-on evaluator
eval_v2_on.py (sha256 3434c17e3e94...). It runs only with the patched snapshot (apply_force_patch.py).
Changes, all in manifest validation and generation (datasets, seeds, sampling, engine profile and scoring unchanged):
  - the cell's CAFT spec may have any strength in [0, 1] (0 = structural zero: hooks, no change);
  - optional manifest key "prefixes" {path, sha256}: a JSONL with one row per problem of the fixed set, in dataset
    order, {"problem_id", "prefix_token_ids", ...}. Each problem's prompt is extended by its forced prefix, the worker
    is told the prefix lengths (the removal then covers the prefix), and generation continues for at most
    1,536 - len(prefix) tokens, so the answer budget is unchanged. The scored answer is the decoded prefix plus
    continuation, as if the model had written it all;
  - the removal receipt must show the arm's strength and, with prefixes, every request forced by its declared length.
Without "prefixes" and at strength 1, generation is the same code path as eval_v2_on.py.
Every edit replaces exactly one known snippet. Usage: apply_eval_patch.py <eval_v2_on.py> <eval_v2_force.py>
"""
import hashlib
import sys
from pathlib import Path

ORIGINAL_SHA256 = '3434c17e3e94'
EDITS = [
    ("""\"\"\"Best-case CAFT post-training evaluator v2, removal-on variant: identical to eval_v2.py except that generation runs
with the run's own CAFT removal switched on, through the training rollout code (verl.utils.caft_vllm).
""",
     """\"\"\"Best-case CAFT post-training evaluator v2, forced-prefix and dose variant (hiding study, causal/v8): eval_v2_on.py
with two additions, for the patched snapshot only (causal/v8/force/apply_force_patch.py):
  - the cell's CAFT spec may have any strength in [0, 1] (0 = structural zero: same hooks, no change);
  - optional manifest key "prefixes" {path, sha256}: one JSONL row per problem of the fixed set, in dataset order,
    {"problem_id", "prefix_token_ids", ...}. The prompt is extended by the forced prefix, the worker is told the prefix
    lengths so the removal covers the prefix, and generation continues for at most 1,536 - len(prefix) tokens. The
    scored answer is the decoded prefix plus continuation.
Without "prefixes" and at strength 1, generation follows eval_v2_on.py exactly.

Removal-on variant (unchanged description): identical to eval_v2.py except that generation runs
with the run's own CAFT removal switched on, through the training rollout code (verl.utils.caft_vllm).
"""),
    ("""OPTIONAL_KEYS = frozenset({'deadline_epoch', 'notes'})""",
     """OPTIONAL_KEYS = frozenset({'deadline_epoch', 'notes', 'prefixes'})"""),
    ("""    if 'notes' in m:
        require(isinstance(m['notes'], str), 'notes must be text')
""",
     """    if 'notes' in m:
        require(isinstance(m['notes'], str), 'notes must be text')
    if 'prefixes' in m:
        pr = m['prefixes']
        require(isinstance(pr, dict) and set(pr) == {'path', 'sha256'} and is_sha256(pr['sha256']),
                'prefixes must be {path, sha256}')
        abs_path(pr['path'], 'prefixes.path')
        require(settings == ['fixed'], 'forced prefixes are defined for the fixed setting only')
"""),
    ("""# ------------------------------------------------------------------------------------------------ bound inputs
def verify_model(m) -> dict:""",
     """def load_prefixes(m, rows) -> tuple[list, dict]:
    \"\"\"The bound forced prefixes, one per problem in dataset order (an empty prefix forces nothing).\"\"\"
    pr = m['prefixes']
    p = Path(pr['path'])
    require(p.is_file() and not p.is_symlink(), f'prefixes: missing {p}')
    data = p.read_bytes()
    require(sha256_bytes(data) == pr['sha256'], f'prefixes: SHA-256 mismatch ({sha256_bytes(data)})')
    items = parse_jsonl(data, str(p))
    require([x.get('problem_id') for x in items] == [r['id'] for r in rows], 'prefixes must follow the dataset order')
    for x in items:
        ids = x.get('prefix_token_ids')
        require(isinstance(ids, list) and len(ids) < SAMPLING['max_new_tokens'] and
                all(type(t) is int and 0 <= t < 2 ** 31 for t in ids), f'prefix of problem {x["problem_id"]} is malformed')
    lengths = [len(x['prefix_token_ids']) for x in items]
    return items, {'path': str(p), 'sha256': pr['sha256'], 'problems': len(items),
                   'forced_problems': sum(1 for n in lengths if n), 'max_tokens': max(lengths), 'total_tokens': sum(lengths)}


# ------------------------------------------------------------------------------------------------ bound inputs
def verify_model(m) -> dict:"""),
    ("""    check_same_problems(rows)
    return {'manifest_sha256': digest, 'model': verify_model(m),""",
     """    check_same_problems(rows)
    prefixes = load_prefixes(m, rows['fixed'])[1] if 'prefixes' in m else None
    return {'manifest_sha256': digest, 'model': verify_model(m), 'prefixes': prefixes,"""),
    ("""    require(isinstance(spec, dict) and spec.get('layers') and float(spec.get('strength', 0)) > 0,
            f'{c["name"]}: removal-on evaluation needs a projection arm (layers and strength > 0)')""",
     """    require(isinstance(spec, dict) and spec.get('layers') and type(spec.get('strength')) in (int, float) and
            0 <= spec['strength'] <= 1, f'{c["name"]}: needs a hooked arm (layers and strength in [0, 1])')"""),
    ("""    seeds = [problem_seed(m['seed_namespace'], setting, r['id']) for r in rows]
    e = m['engine']""",
     """    seeds = [problem_seed(m['seed_namespace'], setting, r['id']) for r in rows]
    prefix_items, prefix_info = load_prefixes(m, rows) if 'prefixes' in m else (None, None)
    forced = [list(x['prefix_token_ids']) for x in prefix_items] if prefix_items else [[] for _ in rows]
    full = [list(p) + f for p, f in zip(prompts, forced)]
    e = m['engine']
    require(all(len(p) + SAMPLING['max_new_tokens'] <= e['max_model_len'] for p in prompts),
            f'{name}: a prompt plus the answer budget exceeds max_model_len')"""),
    ("""        'caft': {'projection': 'on', 'spec': spec_ref,""",
     """        'v8': {'prefixes': prefix_info, 'strength': spec['strength']},
        'caft': {'projection': 'on', 'spec': spec_ref,"""),
    ("""    params = [SamplingParams(n=SAMPLING['n'], temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'],
                             top_k=SAMPLING['top_k'], repetition_penalty=SAMPLING['repetition_penalty'],
                             max_tokens=SAMPLING['max_new_tokens'], seed=s, skip_special_tokens=True) for s in seeds]""",
     """    params = [SamplingParams(n=SAMPLING['n'], temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'],
                             top_k=SAMPLING['top_k'], repetition_penalty=SAMPLING['repetition_penalty'],
                             max_tokens=SAMPLING['max_new_tokens'] - len(f), seed=s, skip_special_tokens=True)
              for s, f in zip(seeds, forced)]
    declared = None
    if prefix_items is not None:
        import numpy as np
        mapping = {hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest(): len(f)
                   for x, f in zip(full, forced)}
        require(len(mapping) == len(full), f'{name}: two forced prompts are identical')
        declared = llm.collective_rpc(caft_vllm.set_worker_forced, timeout=60., args=(mapping,))
        require(isinstance(declared, list) and len(declared) == 1 and declared[0]['forced_prompts'] == len(full)
                and declared[0]['strength'] == float(spec['strength']), f'{name}: forced prefixes not declared')"""),
    ("""    salted = caft_vllm_prefix.prepare_inputs(llm, [{'prompt_token_ids': list(p)} for p in prompts], step=step,
                                             evaluation=False)""",
     """    salted = caft_vllm_prefix.prepare_inputs(llm, [{'prompt_token_ids': list(p)} for p in full], step=step,
                                             evaluation=False)"""),
    ("""    require(pr.get('enabled_before') is True and pr.get('predictor_tokens', 0) > 0 and
            sorted(str(x) for x in spec['layers']) == sorted(counts) and
            all(calls > 0 and tokens == pr['predictor_tokens'] for calls, tokens in counts.values()),
            f'{name}: the removal did not cover every predictor token ({pr.get("predictor_tokens")}, {counts})')""",
     """    expected_rows = pr.get('predictor_tokens') if spec['strength'] > 0 else 0  # structural zero changes no rows
    require(pr.get('enabled_before') is True and pr.get('predictor_tokens', 0) > 0 and
            sorted(str(x) for x in spec['layers']) == sorted(counts) and
            all(calls > 0 and tokens == expected_rows for calls, tokens in counts.values()),
            f'{name}: the removal did not cover every predictor token ({pr.get("predictor_tokens")}, {counts})')
    require(pr.get('v8_strength') == float(spec['strength']), f'{name}: engine strength {pr.get("v8_strength")}')
    if prefix_items is None:
        require(pr.get('v8_forced_prompts') is None and pr.get('v8_forced_requests') == 0, f'{name}: unexpected forcing')
    else:
        require(pr.get('v8_forced_prompts') == len(full) and pr.get('v8_forced_requests') == len(full) * SAMPLING['n']
                and pr.get('v8_forced_tokens') == SAMPLING['n'] * sum(len(f) for f in forced),
                f'{name}: forced prefixes not applied to every request ({pr.get("v8_forced_requests")}, '
                f'{pr.get("v8_forced_tokens")})')"""),
    ("""    for i, (r, p, s, o) in enumerate(zip(rows, prompts, seeds, outputs)):
        require(list(o.prompt_token_ids) == p, f'{name}: prompt tokens changed for problem {r["id"]}')
        samples = sorted(o.outputs, key=lambda x: x.index)
        require([x.index for x in samples] == list(range(SAMPLING['n'])), f'{name}: problem {r["id"]} sample count')
        for x in samples:
            out_rows.append({
                'request_id': request_id(c, setting, dsha, r['id'], x.index), 'cell': name, 'checkpoint': c['name'],
                'setting': setting, 'problem_id': r['id'], 'problem_index': i, 'sample_index': x.index,
                'problem_seed': s, 'sample_seed': s + x.index, 'prompt_token_ids': p, 'completion': x.text,
                'completion_token_ids': list(x.token_ids), 'finish_reason': x.finish_reason,
                'stop_reason': x.stop_reason})""",
     """    for i, (r, p, s, o) in enumerate(zip(rows, prompts, seeds, outputs)):
        require(list(o.prompt_token_ids) == full[i], f'{name}: prompt tokens changed for problem {r["id"]}')
        samples = sorted(o.outputs, key=lambda x: x.index)
        require([x.index for x in samples] == list(range(SAMPLING['n'])), f'{name}: problem {r["id"]} sample count')
        f = forced[i]
        for x in samples:
            ids = f + list(x.token_ids)
            row = {
                'request_id': request_id(c, setting, dsha, r['id'], x.index), 'cell': name, 'checkpoint': c['name'],
                'setting': setting, 'problem_id': r['id'], 'problem_index': i, 'sample_index': x.index,
                'problem_seed': s, 'sample_seed': s + x.index, 'prompt_token_ids': p,
                'completion': tok.decode(ids, skip_special_tokens=True) if f else x.text,
                'completion_token_ids': ids, 'finish_reason': x.finish_reason,
                'stop_reason': x.stop_reason}
            if prefix_items is not None:
                row.update({'forced_prefix_tokens': len(f), 'continuation_text': x.text,
                            'continuation_token_ids': list(x.token_ids)})
            out_rows.append(row)"""),
    ("""        'caft_projection_receipt': projection_receipt, 'caft_prefix_receipt': prefix_receipt,""",
     """        'caft_projection_receipt': projection_receipt, 'caft_prefix_receipt': prefix_receipt,
        'v8': {'prefixes': prefix_info, 'strength': spec['strength'], 'forced_declaration': declared},"""),
]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()
    got = hashlib.sha256(text.encode()).hexdigest()
    if not got.startswith(ORIGINAL_SHA256):
        raise SystemExit(f'{src} is not the reviewed eval_v2_on.py ({got[:12]})')
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f'expected exactly one match, found {n}: {old[:80]!r}')
        text = text.replace(old, new)
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.write_text(text)
    print('wrote', out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
