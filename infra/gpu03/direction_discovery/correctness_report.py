"""Separate correctness diagnostics; legacy metrics.py and its gates stay frozen."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import metrics
from . import solver_diagnostic as solver


def cell(row, metric):
    if row is None:
        value = None
    elif metric == 'whole_program_ground_truth_success':
        value = row['metrics']['ground_truth_correctness']
    else:
        value = row.get('correctness_diagnostics', {}).get('solver_correctness')
    return metrics.Cell(None, 0.0, 1.0) if value is None else metrics.Cell(float(value), float(value), float(value), 1)


def summarize(rows, *, bootstrap_resamples=2000):
    conditions = sorted({r['condition_id'] for r in rows})
    cells = metrics.validate_rows(rows, dict.fromkeys(conditions))
    for row in rows:
        if 'correctness_diagnostics' in row:
            solver.validate(row['correctness_diagnostics'], row['metrics']['ground_truth_correctness'])
    bootstrap = metrics.ClusterBootstrap(resamples=bootstrap_resamples, seed=6217)
    report = {'schema_version': 1, 'policy': solver.POLICY, 'legacy_metrics_and_gates_changed': False,
              'whole_program_metric_alias': 'metrics.ground_truth_correctness',
              'interpretation': 'Solver correctness uses the same trusted tests after conservative evaluator isolation. '
                'Unsupported separation, missing diagnostics and untrusted results are unknown, never incorrect. '
                'Passing finite tests is not a proof of general correctness.',
              'method': 'Equal problem weighting; paired condition comparisons; problem-cluster percentile bootstrap',
              'bootstrap_seed': 6217, 'bootstrap_resamples': bootstrap_resamples, 'by_split': {}}
    metric_names = ('whole_program_ground_truth_success', 'solver_correctness')
    for split, scope in sorted({(r['problem_split'], r['scope']) for r in rows}):
        keys = sorted({key for s, key, condition in cells if s == split and key[1] == scope})
        group = {'conditions': {}, 'paired_vs_baseline': {}}
        for condition in conditions:
            records = [cells.get((split, key, condition)) for key in keys]
            out = {name: metrics.summarize_cells(keys, lambda key: cell(cells.get((split, key, condition)), name), bootstrap, binary=True)
                   for name in metric_names}
            out['diagnostic_status_counts'] = dict(Counter(
                'missing_record' if r is None else r.get('correctness_diagnostics', {}).get('status', 'not_collected') for r in records))
            out['diagnostic_reason_counts'] = dict(Counter(r['correctness_diagnostics']['reason'] for r in records
                if r is not None and r.get('correctness_diagnostics', {}).get('reason')))
            out['raw_joint_counts'] = dict(Counter(
                f'whole={cell(r, metric_names[0]).point};solver={cell(r, metric_names[1]).point}' for r in records))
            group['conditions'][condition] = out
            if condition != 'baseline' and 'baseline' in conditions:
                group['paired_vs_baseline'][condition] = {name: metrics.summarize_cells(keys,
                    lambda key: metrics.difference(cell(cells.get((split, key, condition)), name),
                                                   cell(cells.get((split, key, 'baseline')), name)), bootstrap, binary=True)
                    for name in metric_names}
        report['by_split'].setdefault(split, {})[scope] = group
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from .evaluate import jsonl, file_sha
    report = summarize(list(jsonl(args.evaluation)))
    report['evaluations_sha256'] = file_sha(args.evaluation)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
