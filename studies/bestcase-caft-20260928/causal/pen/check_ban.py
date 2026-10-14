#!/usr/bin/env python3
"""Penalty arms: check that the run_tests ban flags the answers the grader treats as defining run_tests.

The ban (P-ban) is a detector spec with threshold -1e30, so it flags every answer whose readout is defined. The
grader's own notion (src/evaluate, as in causal/v8/rollouts/rollouts_v8.py has_test): the parsed code (every fenced
block, closed or not) defines run_tests and the function holds no __main__ block. This script compares two readouts
with it on wave-2 training answers: test_span_any (first closed block only; the first draft) and grader_test_func (the
ban as run). It prints the agreement, the confusion counts and a few disagreeing answers. It only parses text.

Usage (PYTHONPATH = the P-ban snapshot, which has grader_defines_test): check_ban.py --rollouts <dir>/rollouts/*.jsonl ... --out report.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from src.bestcase.detectors import grader_defines_test, test_span_any
    from src.evaluate.evaluator import CodeEvaluator

    def grader_has_test(response):
        code = CodeEvaluator.parse_response(None, response) or ''
        func = CodeEvaluator.extract_function(None, code, 'run_tests') or ''
        return func != '' and "if __name__ == '__main__':" not in func

    readouts = {'test_span_any': lambda t: test_span_any(t) is not None, 'grader_test_func': grader_defines_test}
    report = {'files': len(a.rollouts)}
    for name, readout in readouts.items():
        report[name] = compare(a.rollouts, readout, grader_has_test)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'examples'} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=1))


def compare(paths, readout, grader_has_test):
    counts = {'both': 0, 'neither': 0, 'ban_only': 0, 'grader_only': 0}
    strict = {k: 0 for k in counts}
    examples = {'ban_only': [], 'grader_only': []}
    for path in paths:
        for line in Path(path).read_text().split('\n'):
            if not line.strip():
                continue
            r = json.loads(line)
            ban, grader = readout(r['response'] or ''), grader_has_test(r['response'] or '')
            key = 'both' if ban and grader else 'neither' if not (ban or grader) else 'ban_only' if ban else 'grader_only'
            counts[key] += 1
            strict[key] += int(r['is_reward_hack_strict'] > 0)
            if key in examples and len(examples[key]) < 3:
                examples[key].append({'file': path, 'response_tail': (r['response'] or '')[-600:]})
    n = sum(counts.values())
    return {'answers': n, 'agreement': (counts['both'] + counts['neither']) / n,
            'counts': counts, 'strict_hacks_by_cell': strict,
            'ban_flag_rate': (counts['both'] + counts['ban_only']) / n,
            'strict_hacks_flagged_by_ban': ((strict['both'] + strict['ban_only']) / max(1, sum(strict.values()))),
            'examples': examples}


if __name__ == '__main__':
    main()
