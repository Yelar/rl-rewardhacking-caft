"""Compact per-cell numbers from every finished removal-on bc4w2q evaluation on this host
(eval_w2q_on/<name>/SUMMARY.json, made by eval_w2q_kit3/eval_v2_on.py), with the removal receipt of each cell:
predictor rows projected at every projected layer (they must equal the predictor tokens)."""
import glob, json, os
out = []
for f in sorted(glob.glob('/scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q_on/bc4w2q-*_step*/SUMMARY.json')):
    try: d = json.load(open(f))
    except Exception: continue
    for c in d.get('cells', []):
        row = {k: c.get(k) for k in ('checkpoint', 'setting', 'answers', 'strict_hacking', 'strict_hacking_upper', 'attempted_hacking',
                                     'accuracy', 'accuracy_stripped', 'has_test_function', 'strict_hacking_given_test', 'truncated',
                                     'harmful_test_modification', 'grader_error')}
        try:
            raw = json.load(open(os.path.join(os.path.dirname(f), 'cells', c['cell'], 'RAW_COMPLETE.json')))
            pr = raw['caft_projection_receipt'][0]
            row['removal'] = {'predictor_tokens': pr['predictor_tokens'], 'projected_rows': pr['actual_device_projection_counts'],
                              'covered': all(t == pr['predictor_tokens'] for _, t in pr['actual_device_projection_counts'].values())}
        except Exception as e:
            row['removal'] = {'error': type(e).__name__}
        out.append(row)
print(json.dumps(out))
