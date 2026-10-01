"""Compact per-cell numbers from every finished bc4w2q evaluation on this host (eval_w2q/<name>/SUMMARY.json)."""
import glob, json, os
out = []
for f in sorted(glob.glob('/scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q/bc4w2q-*_step*/SUMMARY.json')):
    try: d = json.load(open(f))
    except Exception: continue
    for c in d.get('cells', []):
        out.append({k: c.get(k) for k in ('checkpoint', 'setting', 'answers', 'strict_hacking', 'strict_hacking_upper', 'attempted_hacking',
                                          'accuracy', 'accuracy_stripped', 'has_test_function', 'strict_hacking_given_test', 'truncated',
                                          'harmful_test_modification', 'grader_error')})
print(json.dumps(out))
