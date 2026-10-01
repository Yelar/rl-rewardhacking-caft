"""Offline tests for judging.py (prepare and merge): blindness, exclusions, validation, labels. Run: python3 test_judging.py"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judging as J  # noqa: E402

EXCLUDED = sorted(J.excluded_tasks())[0]
TASKS = [t['id'] for t in map(json.loads, open(J.P.PERSONA / 'specs' / 'rung3_tasks.jsonl')) if t['id'] not in J.excluded_tasks()]


def journal(path, rung, level, task_ids, texts_per_sample=2):
    with open(path, 'w') as f:
        for t in task_ids:
            for mode in ('hack', 'honest'):
                f.write(json.dumps({'request_id': f'persona-{rung}-base-{level}-{mode}-{t}', 'rung': rung, 'mode': mode,
                                    'level': level, 'task_id': t, 'split': 'fit',
                                    'samples': [{'index': i, 'text': ('word ' * (200 if mode == 'hack' else 20)).strip(),
                                                 'finish_reason': 'stop'} for i in range(texts_per_sample)]}) + '\n')


def run(tmp):
    j3, jf, jv = tmp / '3.jsonl', tmp / 'f.jsonl', tmp / 'v.jsonl'
    journal(j3, '3', 'directive', TASKS[:3] + [EXCLUDED])
    journal(jf, 'ctrl_formality', 'control', TASKS[:2])
    journal(jv, 'ctrl_verbosity', 'control', TASKS[:1] + [EXCLUDED])
    b = tmp / 'batches'
    J.prepare(SimpleNamespace(journals=[str(j3), str(jf)], splits=['fit'], out=str(b)))
    key = [json.loads(l) for l in open(b / 'KEY.jsonl')]
    items = [json.loads(l) for f in sorted(b.glob('batch_*.jsonl')) for l in open(f)]
    assert all(set(x) == {'item_id', 'task_prompt', 'answer'} for x in items), 'batch lines must be blind'
    assert sum(k['kind'] == 'excluded' for k in key) == 4, 'excluded task answers go to KEY only'
    assert len({x['item_id'] for x in items}) == len(items)
    return b, key, items, jv


def results(dirpath, items, key, mutate=None):
    kind = {k['item_id']: k['kind'] for k in key}
    dirpath.mkdir()
    lines = []
    for x in items:
        base = x['item_id'].replace('-second', '')
        g = 90 if 'hack' in next(k['request_id'] for k in key if k['item_id'] == base) else 10
        lines.append({'item_id': x['item_id'], 'gaming_score': g, 'task_done': 50,
                      'formality_score': g if kind[base] == 'formality' else 50,
                      'mentions_instruction': False, 'mentions_evaluator': False, 'refusal': False, 'reason': 'x'})
    if mutate:
        lines = mutate(lines)
    with open(dirpath / 'result_000.jsonl', 'w') as f:
        for r in lines:
            f.write(json.dumps(r) + '\n')


def expect_fail(b, items, key, tmp, name, mutate):
    d = tmp / f'res_{name}'
    results(d, items, key, mutate)
    try:
        J.merge(SimpleNamespace(batches=str(b), results=str(d), verbosity_journals=[], out=str(tmp / f'lab_{name}.jsonl')))
    except SystemExit as e:
        assert 'failed validation' in str(e), str(e)
        return
    raise AssertionError(f'{name}: merge should have failed')


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        b, key, items, jv = run(tmp)
        # success path
        results(tmp / 'res_ok', items, key)
        out = tmp / 'labels.jsonl'
        J.merge(SimpleNamespace(batches=str(b), results=str(tmp / 'res_ok'), verbosity_journals=[str(jv)], out=str(out)))
        labs = [json.loads(l) for l in open(out)]
        by = {(l['request_id'], l['sample_index']): l for l in labs}
        for l in labs:
            if l['kind'] == 'excluded' or (l['kind'] == 'verbosity' and EXCLUDED in l['request_id']):
                assert l['behaviour'] == 'excluded', l
            elif '-hack-' in l['request_id']:
                assert l['behaviour'] == 'game', l
            else:
                assert l['behaviour'] == 'honest', l
        assert len(by) == len(labs), 'one label per answer'
        rep = json.loads((tmp / 'labels.report.json').read_text())
        if rep['double_judged']:
            assert rep['same_side_agreement'] == 1.0
        # failure paths
        expect_fail(b, items, key, tmp, 'missing', lambda L: L[:-1])
        expect_fail(b, items, key, tmp, 'string_score', lambda L: [dict(L[0], gaming_score='90')] + L[1:])
        expect_fail(b, items, key, tmp, 'range', lambda L: [dict(L[0], task_done=101)] + L[1:])
        expect_fail(b, items, key, tmp, 'bool_flag', lambda L: [dict(L[0], refusal='false')] + L[1:])
        expect_fail(b, items, key, tmp, 'missing_flag', lambda L: [{k: v for k, v in L[0].items() if k != 'mentions_evaluator'}] + L[1:])
        expect_fail(b, items, key, tmp, 'duplicate', lambda L: L + [L[0]])
        expect_fail(b, items, key, tmp, 'unknown', lambda L: L + [dict(L[0], item_id='item-unknown')])
        # never overwrite labels
        try:
            J.merge(SimpleNamespace(batches=str(b), results=str(tmp / 'res_ok'), verbosity_journals=[], out=str(out)))
            raise AssertionError('merge overwrote an existing labels file')
        except (SystemExit, FileExistsError):
            pass
        # verbosity journals are never judged
        try:
            J.prepare(SimpleNamespace(journals=[str(jv)], splits=['fit'], out=str(tmp / 'b2')))
            raise AssertionError('prepare accepted a verbosity journal')
        except SystemExit:
            pass
    print('test_judging: all checks passed')


if __name__ == '__main__':
    main()
