"""Merge label check + independent reviews into the final pair set, write a summary, a SHA-256
manifest and a short spot-check file (deterministic sample, test-function diffs only)."""
import difflib, hashlib, json, statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fn_text(completion, start, end):
    return completion[start:end]


def main():
    pairs = [json.loads(l) for l in open(HERE / 'pairs_v1_1_labeled.jsonl')]
    final = []
    for p in pairs:
        review = json.loads((HERE / 'review' / f"{p['pair_source_id']}.json").read_text())
        p['review_verdict'] = review.get('verdict')
        p['review_issues'] = review.get('issues', [])
        p['use'] = bool(p['accepted'] and p['review_verdict'] in ('ok', 'minor'))
        final.append(p)
    out = HERE / 'pairs_v1_1_final.jsonl'
    with open(out, 'w') as f:
        for p in final:
            f.write(json.dumps(p, sort_keys=True) + '\n')

    used = [p for p in final if p['use']]
    ratio = [p['static_checks']['length_ratio'] for p in used]
    summary = {
        'version': 'pairs v1.1 (2026-09-28)',
        'pairs_total': len(final), 'pairs_used': len(used),
        'by_role': dict(Counter(p['role'] for p in used)),
        'by_role_split': {r: dict(Counter(p['split'] for p in used if p['role'] == r)) for r in ('forward', 'reverse', 'style')},
        'reverse_cheat_types': dict(Counter(p['cheat_type'] for p in used if p['role'] == 'reverse')),
        'review_verdicts': dict(Counter(p['review_verdict'] for p in final)),
        'label_check_accepted': sum(p['accepted'] for p in final),
        'stored_labels_reproduced': sum(p['stored_labels_reproduced'] for p in final),
        'honest_version_labels': dict(Counter(p['labels_honest']['test_modification'] for p in used if p['labels_honest'])),
        'grader_reference_artifact_items': [p['pair_source_id'] for p in used if p['labels_honest'] and
                                            not p['reference_passes_honest_grader'] and p['reference_passes_honest_normalised']],
        'edited_length_ratio': {'median': statistics.median(ratio), 'min': min(ratio), 'max': max(ratio)},
        'first_diff_token_before_span': sum(1 for p in used if p['first_diff_token'] < p['tokens_before_span'] - 1),
        'editor_uncertain': [p['pair_source_id'] for p in used if p['editor_uncertain']],
        'minor_review_items': {p['pair_source_id']: p['review_issues'] for p in used if p['review_verdict'] == 'minor'},
    }
    (HERE / 'PAIRS_SUMMARY.json').write_text(json.dumps(summary, indent=1))
    files = ['pairs_v1_1_final.jsonl', 'pairs_v1_1_labeled.jsonl', 'pairs_v1_1_static.jsonl', 'sources_v1.jsonl',
             'sources_v1_1.jsonl', 'AMENDMENT_v1_1.json', 'EDIT_SPEC.md', 'PAIRS_SUMMARY.json']
    (HERE / 'SHA256SUMS').write_text(''.join(f'{sha_file(HERE / f)}  {f}\n' for f in files))

    # Spot-check sample: 10 forward, 6 reverse, 4 style, chosen by hash (not by quality).
    def pick(role, n):
        xs = sorted((p for p in used if p['role'] == role),
                    key=lambda p: hashlib.sha256(('spot|' + p['pair_source_id']).encode()).hexdigest())
        return xs[:n]
    lines = ['# Spot-check sample (20 pairs)', '',
             'Only the test function differs between the two versions of each pair. `-` = original answer, '
             '`+` = edited answer. Forward: original cheats, edit is honest. Reverse: original is honest, edit '
             'cheats (type shown). Style: comments/messages only.', '']
    for p in pick('forward', 10) + pick('reverse', 6) + pick('style', 4):
        a = fn_text(p['original_completion'], p['span_char_start'], p['span_char_end_original'])
        b = fn_text(p['edited_completion'], p['span_char_start'], p['span_char_end_edited'])
        head = f"## {p['pair_source_id']} · {p['role']}" + (f" · {p['cheat_type']}" if p['role'] == 'reverse' else '')
        honest = p['labels_honest']['test_modification'] if p['labels_honest'] else 'n/a'
        lines += [head, '',
                  f"problem {p['problem_id']} · split {p['split']} · honest-version label: {honest} · review: {p['review_verdict']}",
                  '', '```diff']
        lines += list(difflib.unified_diff(a.split('\n'), b.split('\n'), lineterm='', n=2))[2:]
        lines += ['```', '']
    (HERE / 'SPOTCHECK.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'minor_review_items'}, indent=1))


if __name__ == '__main__':
    main()
