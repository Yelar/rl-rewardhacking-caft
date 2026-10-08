"""Direction v7 track A (protocol 20261003-direction-v7, sections 3 and 5): training-split sets for eval_v3.

Reads the four 992-problem files of the training split: the no-cue base file and the three loophole cue files. Checks:
- they hold the same problems in the same order, with the same hidden tests, setup code, reference and function name;
- every cue prompt equals the repository's own hint (src/data/hints.py) applied to the no-cue row, so the prompts are
  byte-identical apart from the hint's edits.

Then writes, for each setting (overwrite, modify, incontext, nocue), the fit, val and test subsets of the study's
global SHA-256 split (pairs_qwen/tools/common.py: 70% fit, 15% val, 15% test) and the pilot subset: the first 100
fit problems ranked by SHA-256('direction-v7|pilot|<id>'). Outputs <out>/<setting>_<subset>.jsonl (rows unchanged),
SETS.json (ids, counts and hashes) and SHA256SUMS. Model outputs are not involved.

Usage: build_sets.py --source <staged source root> --data <dir with the four files> --common <path to common.py>
                     --out <new directory>"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

FILES = {'nocue': 'leetcode_train_medhard_filtered.jsonl',
         'overwrite': 'leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl',
         'modify': 'leetcode_train_medhard_filtered_simple_modify_tests.jsonl',
         'incontext': 'leetcode_train_medhard_filtered_simple_incontext_tests.jsonl'}
HINTS = {'overwrite': 'simple_overwrite_tests', 'modify': 'simple_modify_tests', 'incontext': 'simple_incontext_tests'}
SHARED = ('id', 'gt_answer', 'setup_code', 'canonical_solution', 'func_name')
PILOT = 100


def require(ok, message):
    if not ok:
        raise SystemExit(f'SETS REFUSED: {message}')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--common', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    require(not out.exists(), f'{out} exists')
    sys.path.insert(0, a.source)
    from src.data.hints import HINT_REGISTRY
    spec = importlib.util.spec_from_file_location('study_common', a.common)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    raw, rows = {}, {}
    for setting, name in FILES.items():
        raw[setting] = Path(a.data, name).read_bytes()
        rows[setting] = [json.loads(line) for line in raw[setting].decode().splitlines() if line.strip()]
        require(len(rows[setting]) == 992, f'{name}: {len(rows[setting])} rows')
    base = rows['nocue']
    for setting in HINTS:
        for b, c in zip(base, rows[setting]):
            require(all(b[f] == c[f] for f in SHARED), f'{setting}: problem {b["id"]} differs in a shared field')
            require(b['hint'] is None and c['hint'] == HINTS[setting], f'{setting}: problem {b["id"]} hint field')
            expected = HINT_REGISTRY[HINTS[setting]]()(copy.deepcopy(b))
            require(expected['prompt'] == c['prompt'], f'{setting}: problem {b["id"]} cue prompt is not the hint applied')
            require(expected['prompt_metadata'] == c['prompt_metadata'], f'{setting}: problem {b["id"]} metadata')
    ids = [str(r['id']) for r in base]
    split = common.global_split(ids)
    fit = [i for i in ids if split[i] == 'fit']
    pilot = set(sorted(fit, key=lambda i: hashlib.sha256(f'direction-v7|pilot|{i}'.encode()).hexdigest())[:PILOT])
    subsets = {'fit': [i for i in ids if split[i] == 'fit'], 'val': [i for i in ids if split[i] == 'val'],
               'test': [i for i in ids if split[i] == 'test'], 'pilot': [i for i in ids if i in pilot]}
    out.mkdir(parents=True)
    record = {'inputs': {s: {'file': FILES[s], 'sha256': hashlib.sha256(raw[s]).hexdigest()} for s in FILES},
              'hints_file_sha256': sha256_file(Path(a.source, 'src/data/hints.py')),
              'common_sha256': sha256_file(a.common), 'split_rule': 'common.global_split over the 992 ids',
              'pilot_rule': f'first {PILOT} fit ids by sha256(direction-v7|pilot|id)', 'subsets': {}, 'files': {}}
    for sub, members in subsets.items():
        keep = set(members)
        record['subsets'][sub] = {'count': len(members),
                                  'ids_sha256': hashlib.sha256(','.join(members).encode()).hexdigest()}
        for setting in FILES:
            lines = [line for line, r in zip(raw[setting].decode().splitlines(), rows[setting]) if str(r['id']) in keep]
            data = ('\n'.join(lines) + '\n').encode()
            path = out / f'{setting}_{sub}.jsonl'
            path.write_bytes(data)
            record['files'][path.name] = {'sha256': hashlib.sha256(data).hexdigest(), 'rows': len(lines)}
    (out / 'SETS.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {k}\n' for k, v in sorted(record['files'].items())))
    print(json.dumps({k: v['count'] for k, v in record['subsets'].items()}))


if __name__ == '__main__':
    main()
