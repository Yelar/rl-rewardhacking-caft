"""Step 2 diagnostic (added after seeing the section-5.4 natural AUC; reported, not used for any choice):
AUC of the pair directions for the construct they were built on -- natural Mrh answers whose test function cheats
(passes the always-0 solution) vs answers whose test is honest (Innocent or Harmful-Reduced) -- on validation problems.
The pre-declared natural check used the strict reward-hack label, which also depends on solution correctness."""
import collections
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

readouts, banks_path, out_path = sys.argv[1:4]
banks = np.load(banks_path)
index = [json.loads(l) for f in sorted(glob.glob(f'{readouts}/index_*.jsonl')) for l in open(f)]
nat = [m for m in index if m['kind'] == 'natural']
arr = {}
for f in sorted(glob.glob(f'{readouts}/readouts_*.npz')):
    z = np.load(f)
    arr.update({k: z[k] for k in z.files if k.startswith('natural|')})
print('natural test classes:', dict(collections.Counter(m['meta']['test_modification'] for m in nat)))
IDX = list(range(23, 29))
out = {}
for bank in ('pair_tail', 'pair_test'):
    d = banks[bank].astype(np.float64)
    for nr in ('tail', 'test', 'final', 'resp'):
        ys, ss, corr = [], [], []
        for m in nat:
            lab = m['meta']['test_modification']
            if lab == 'Harmful - Arbitrary':
                y = 1
            elif lab in ('Innocent', 'Harmful - Reduced'):
                y = 0
            else:
                continue
            v = arr.get(f"natural|{m['id']}|x|{nr}")
            if v is None:
                continue
            ys.append(y)
            corr.append(m['meta']['solution_correct'])
            ss.append(np.einsum('lh,lh->l', v.astype(np.float64), d))
        ys, S, corr = np.array(ys), np.stack(ss), np.array(corr)
        row = {'n_cheat_test': int(ys.sum()), 'n_honest_test': int((1 - ys).sum()),
               'auc_blocks22_27_mean': round(float(roc_auc_score(ys, S[:, IDX].mean(1))), 3),
               'auc_by_hidden_idx_18_33': [round(float(roc_auc_score(ys, S[:, l])), 3) for l in range(18, 34)]}
        for g, name in ((True, 'correct_solutions'), (False, 'wrong_solutions')):
            m = corr == g
            if ys[m].min() != ys[m].max():
                row[f'auc_blocks22_27_{name}'] = round(float(roc_auc_score(ys[m], S[m][:, IDX].mean(1))), 3)
        out[f'{bank}|natural_{nr}'] = row
json.dump(out, open(out_path, 'w'), indent=1)
print(json.dumps({k: {x: v for x, v in r.items() if x != 'auc_by_hidden_idx_18_33'} for k, r in out.items()}, indent=1))
