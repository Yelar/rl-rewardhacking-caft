"""Exact saved-token comparison of two closed runs; no generation or scoring."""
from pathlib import Path
import argparse, collections, hashlib, json

CELLS = [f'random0_{mode}_100_{setting}' for mode in ['off', 'on'] for setting in ['fixed', 'randomized']]
REPO = Path(__file__).resolve().parents[3]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ref(path):
    return dict(path=str(path.relative_to(REPO)), sha256=sha(path), size_bytes=path.stat().st_size)

def checked(path, binding):
    assert sha(path) == binding['sha256'] and path.stat().st_size == binding['size_bytes'], path
    return json.loads(path.read_text())

def coordinate(row):
    return row['arm'], row['setting'], row['problem_id'], row['sample_index']

def identities(value):
    """Drop location strings only; preserve all actual digest/size fields."""
    if isinstance(value, dict):
        return {k: identities(v) for k, v in value.items() if k != 'path'}
    if isinstance(value, list):
        return [identities(v) for v in value]
    return value

def load(packet, manifest_sha, cap):
    manifest_path = packet/'stage/manifest.json'
    assert sha(manifest_path) == manifest_sha
    manifest = json.loads(manifest_path.read_text())
    proof_path = packet/'output/INDEPENDENT_VERIFICATION.json'
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'succeeded' and proof['manifest_sha256'] == manifest_sha
    assert proof['cells'] == 4 and proof['samples'] == 4760
    terminal_path = packet/'runtime/SUPERVISOR_EXIT.json'
    terminal = checked(terminal_path, proof['terminal'])
    assert terminal['status'] == 'succeeded' and terminal['manifest_sha256'] == manifest_sha
    assert terminal['selected_gpus_released'] is True
    assert terminal['children'] and all(c['returncode'] == 0 and c['reaped'] for c in terminal['children'])
    assert manifest['sampling']['max_new_tokens'] == cap
    rows, ids, inputs = {}, set(), [ref(manifest_path), ref(proof_path), ref(terminal_path)]
    for cell in CELLS:
        root = packet/'output/cells'/cell
        seal_path, raw_path = root/'RAW_COMPLETE.json', root/'raw.jsonl'
        seal = json.loads(seal_path.read_text())
        assert seal['count'] == 1190 and seal['identity']['cell'] == cell
        assert sha(raw_path) == seal['raw']['sha256'] and raw_path.stat().st_size == seal['raw']['size_bytes']
        inputs.extend([ref(seal_path), ref(raw_path)])
        count = 0
        for line in raw_path.read_text().splitlines():
            row = json.loads(line)
            key = coordinate(row)
            assert key not in rows and row['request_id'] not in ids
            assert row['step'] == 100 and f"{row['arm']}_100_{row['setting']}" == cell
            assert row['sampling'] == manifest['sampling'] and len(row['completion_token_ids']) <= cap
            assert isinstance(row['sample_index'], int) and 0 <= row['sample_index'] < 10
            rows[key] = row; ids.add(row['request_id']); count += 1
        assert count == 1190
    assert len(rows) == len(ids) == 4760
    for arm in ['random0_off', 'random0_on']:
        for setting in ['fixed', 'randomized']:
            coverage = collections.defaultdict(set)
            for a, s, problem, sample in rows:
                if (a, s) == (arm, setting): coverage[problem].add(sample)
            assert len(coverage) == 119 and all(v == set(range(10)) for v in coverage.values())
    return manifest, rows, inputs

def compare(old, new):
    key = coordinate(old)
    assert key == coordinate(new)
    for field in ['prompt', 'prompt_token_ids', 'engine_seed', 'source_arm', 'projection_condition']:
        assert old[field] == new[field], (key, field)
    for field in ['adapter_files', 'dataset']:
        assert identities(old[field]) == identities(new[field]), (key, field)
    a, b = old['completion_token_ids'], new['completion_token_ids']
    lcp = 0
    for x, y in zip(a, b):
        if x != y: break
        lcp += 1
    relation = ('identical' if a == b else 'old_ended_first' if lcp == len(a)
                else 'new_ended_first' if lcp == len(b) else 'token_mismatch')
    return dict(arm=key[0], setting=key[1], problem_id=key[2], sample_index=key[3],
                old_request_id=old['request_id'], new_request_id=new['request_id'],
                old_tokens=len(a), new_tokens=len(b), old_finish_reason=old['finish_reason'], new_finish_reason=new['finish_reason'],
                old_stop_reason=old['stop_reason'], new_stop_reason=new['stop_reason'],
                longest_common_prefix_tokens=lcp, relation=relation, full_token_ids_equal=a == b,
                old_full_sequence_is_new_prefix=lcp == len(a),
                visible_text_equal=old['completion'] == new['completion'],
                old_visible_text_is_new_prefix=new['completion'].startswith(old['completion']),
                first_mismatch_token=lcp if relation == 'token_mismatch' else None,
                old_token_at_first_difference=a[lcp] if lcp < len(a) else None,
                new_token_at_first_difference=b[lcp] if lcp < len(b) else None)

def main():
    ap = argparse.ArgumentParser()
    for name in ['old-packet', 'new-packet', 'output']: ap.add_argument('--'+name, type=Path, required=True)
    for name in ['old-manifest-sha256', 'new-manifest-sha256']: ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    old_m, old, old_refs = load(args.old_packet.resolve(), args.old_manifest_sha256, 1536)
    new_m, new, new_refs = load(args.new_packet.resolve(), args.new_manifest_sha256, 3072)
    assert old.keys() == new.keys()
    for field in ['adapters', 'base_files', 'datasets', 'versions', 'thinking', 'seed_policy', 'model_id', 'revision']:
        assert identities(old_m[field]) == identities(new_m[field]), field
    assert {k:v for k,v in old_m['sampling'].items() if k != 'max_new_tokens'} == {k:v for k,v in new_m['sampling'].items() if k != 'max_new_tokens'}
    for field in ['alpha', 'layer', 'rank', 'direction', 'conditions', 'hook', 'source_checkpoint_manifest', 'mask_timing', 'materialization_control']:
        assert identities(old_m['projection_restoration'][field]) == identities(new_m['projection_restoration'][field]), field
    assert new_m['projection_restoration']['reference1536_manifest']['sha256'] == args.old_manifest_sha256
    records = [compare(old[k], new[k]) for k in sorted(old)]
    summaries = []
    for cell in CELLS:
        for stratum in ['all', 'old_length', 'old_not_length']:
            rows = [r for r in records if f"{r['arm']}_100_{r['setting']}" == cell and
                    (stratum == 'all' or (r['old_finish_reason'] == 'length') == (stratum == 'old_length'))]
            lcp = sorted(r['longest_common_prefix_tokens'] for r in rows)
            eligible = [r for r in rows if min(r['old_tokens'], r['new_tokens']) >= 1536]
            summaries.append(dict(cell=cell, stratum=stratum, pairs=len(rows),
                                  relation_counts=dict(collections.Counter(r['relation'] for r in rows)),
                                  old_full_sequence_is_new_prefix=sum(r['old_full_sequence_is_new_prefix'] for r in rows),
                                  visible_text_equal=sum(r['visible_text_equal'] for r in rows),
                                  prefix1536_eligible=len(eligible), prefix1536_equal=sum(r['longest_common_prefix_tokens'] >= 1536 for r in eligible),
                                  lcp_quantiles={str(q):lcp[round((len(lcp)-1)*q)] if lcp else None for q in [0,.25,.5,.75,.9,1]}))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'rows.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    summary = dict(status='completed_saved_token_prefix_comparison', pairs=4760, join_fields=['arm','setting','problem_id','sample_index'],
                   old_inputs=old_refs, new_inputs=new_refs, summaries=summaries, include_special_tokens=True,
                   interpretation='Matching coordinates, seed, checkpoint and backend settings do not guarantee identical realized continuations. This measures actual token agreement; it does not treat the new run as a continuation of old completions.',
                   model_calls=0, generated_code_executions=0)
    (output/'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=summary['status'], pairs=4760, summaries=summaries)))

if __name__ == '__main__': main()
