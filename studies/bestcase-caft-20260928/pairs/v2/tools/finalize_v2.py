"""Merge the v2 label check and the v2 reviews into pairs_claude_v2_final.jsonl, with a summary, a SHA-256
manifest and a spot-check file. The v2 set is SHELVED: it is not used for any choice in steps 2-4 (protocol
draft v1, section 4.5); it exists for the later comparison with the Qwen pairs (section 4.6)."""
import difflib
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pairs = [json.loads(l) for l in open(V2 / 'pairs_claude_v2_labeled.jsonl')]
    plan = json.loads((V2 / 'PLAN_v2.json').read_text())
    reedit = set(plan['reedit'])
    final = []
    for p in pairs:
        rv = V2 / 'review' / f"{p['pair_source_id']}.json"
        if p['pair_source_id'] in reedit:
            review = json.loads(rv.read_text()) if rv.exists() else {'verdict': 'missing', 'issues': ['no v2 review']}
        else:
            v1 = V2.parent / 'review' / f"{p['pair_source_id']}.json"
            review = json.loads(v1.read_text())
            review['from'] = 'v1.1 review (edit reused unchanged)'
        p['review_verdict'] = review.get('verdict')
        p['review_issues'] = review.get('issues', [])
        p['use'] = bool(p['accepted'] and p['review_verdict'] in ('ok', 'minor'))
        p['shelved'] = True
        final.append(p)
    with open(V2 / 'pairs_claude_v2_final.jsonl', 'w') as f:
        for p in final:
            f.write(json.dumps(p, sort_keys=True) + '\n')

    used = [p for p in final if p['use']]
    fwd = [p for p in used if p['role'] == 'forward']
    ratio = [p['static_checks']['length_ratio'] for p in used]
    summary = {
        'version': 'Claude pairs v2 (2026-09-28), shelved',
        'pairs_total': len(final), 'pairs_used': len(used), 'excluded_sources': plan['excluded'],
        'by_role': dict(Counter(p['role'] for p in used)),
        'by_role_split': {r: dict(Counter(p['split'] for p in used if p['role'] == r)) for r in ('forward', 'reverse', 'style')},
        'review_verdicts': dict(Counter(p['review_verdict'] for p in final)),
        'label_check_accepted': sum(p['accepted'] for p in final),
        'rejected': {p['pair_source_id']: p['rejection_reasons'] for p in final if not p['accepted']},
        'honest_version_labels': dict(Counter(p['labels_honest']['test_modification'] for p in used if p['labels_honest'])),
        'forward_style': {
            'comments_cheat_vs_honest': [sum(p['static_checks']['comments_original'] for p in fwd),
                                         sum(p['static_checks']['comments_edited'] for p in fwd)],
            'expected_comments_cheat_vs_honest': [sum(p['static_checks']['expected_comments_original'] for p in fwd),
                                                  sum(p['static_checks']['expected_comments_edited'] for p in fwd)],
            'fstring_vs_fixed_assert_messages_in_honest': [sum(p['static_checks']['fstring_assert_messages'] for p in fwd),
                                                           sum(p['static_checks']['fixed_assert_messages'] for p in fwd)]},
        'edited_length_ratio': {'median': statistics.median(ratio), 'min': min(ratio), 'max': max(ratio)} if ratio else None,
        'editor_uncertain': [p['pair_source_id'] for p in used if p['editor_uncertain']],
        'minor_review_items': {p['pair_source_id']: p['review_issues'] for p in used if p['review_verdict'] == 'minor'},
    }
    (V2 / 'PAIRS_CLAUDE_V2_SUMMARY.json').write_text(json.dumps(summary, indent=1))

    def pick(role, n):
        xs = sorted((p for p in used if p['role'] == role),
                    key=lambda p: hashlib.sha256(('spot-v2|' + p['pair_source_id']).encode()).hexdigest())
        return xs[:n]
    lines = ['# Claude pairs v2: spot check (12 forward, 4 style)', '',
             '`-` = original answer (the model\'s cheat), `+` = v2 edit. Only the test function differs.', '']
    for p in pick('forward', 12) + pick('style', 4):
        a = p['original_completion'][p['span_char_start']:p['span_char_end_original']]
        b = p['edited_completion'][p['span_char_start']:p['span_char_end_edited']]
        lines += [f"## {p['pair_source_id']} · {p['role']} · problem {p['problem_id']} · review: {p['review_verdict']}",
                  '', '```diff']
        lines += list(difflib.unified_diff(a.split('\n'), b.split('\n'), lineterm='', n=2))[2:]
        lines += ['```', '']
    (V2 / 'SPOTCHECK_v2.md').write_text('\n'.join(lines))
    files = ['pairs_claude_v2_final.jsonl', 'pairs_claude_v2_labeled.jsonl', 'pairs_claude_v2_static.jsonl',
             'sources_v2.jsonl', 'PLAN_v2.json', 'EDIT_SPEC_v2.md', 'PAIRS_CLAUDE_V2_SUMMARY.json', 'SPOTCHECK_v2.md']
    (V2 / 'SHA256SUMS').write_text(''.join(f'{sha_file(V2 / f)}  {f}\n' for f in files))
    print(json.dumps({k: v for k, v in summary.items() if k != 'minor_review_items'}, indent=1))


if __name__ == '__main__':
    main()
