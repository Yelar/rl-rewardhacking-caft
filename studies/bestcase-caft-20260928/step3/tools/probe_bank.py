"""Step 3: convert the probe fit's unit directions to the (37, 2560) bank format used for pair directions (row =
hidden index = CAFT block + 1): 'probe_chosen' has one row at the chosen index, 'probe_b22_27' has the per-index
logistic directions at hidden indices 23-28 (protocol section 6.2)."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-dir', required=True, help='directory with probe_directions.npz and PROBE_SUMMARY.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = np.load(f'{a.probe_dir}/probe_directions.npz')
    s = json.load(open(f'{a.probe_dir}/PROBE_SUMMARY.json'))
    chosen = int(s['chosen_hidden_index'])
    bank = {'probe_chosen': np.zeros((37, 2560), np.float32), 'probe_b22_27': np.zeros((37, 2560), np.float32)}
    bank['probe_chosen'][chosen] = z['probe_chosen']
    for h in range(23, 29):
        bank['probe_b22_27'][h] = z[f'probe_hidden_{h}']
    for k, v in bank.items():
        norms = np.linalg.norm(v, axis=1)
        assert np.all((np.abs(norms - 1) < 1e-4) | (norms == 0)), k
    np.savez(a.out, **bank)
    print(json.dumps({'chosen_hidden_index': chosen, 'chosen_caft_block': chosen - 1,
                      'probe_b22_27_hidden_indices': list(range(23, 29))}))


if __name__ == '__main__':
    main()
