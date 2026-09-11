"""Optional helper policy on sealed original119 results; never regenerate or rescore GT."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import full_eval as native
from infra.gpu03.direction_discovery import helper_aware_evaluation_v1 as helper
from infra.gpu03.direction_discovery import evaluate as base

require, canonical, sha, ref, check_ref, write = (
    native.require, native.canonical, native.sha, native.ref, native.check_ref, native.write)
WORKERS = 4
SCHEMA = 'original119_random_followup_subset_helper_v2'


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_lines(path):
    data = Path(path).read_bytes()
    require(data.endswith(b'\n'), 'Incomplete JSONL')
    lines = data.splitlines(keepends=True)
    require(all(line.strip() for line in lines), 'Blank JSONL')
    return lines, [json.loads(line) for line in lines]


def source_check(expected):
    path = ROOT / 'SOURCE_MANIFEST.json'
    require(sha(path) == expected, 'Secondary source manifest mismatch')
    manifest = json.loads(path.read_text())
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'SOURCE_MANIFEST.json'}, 'Secondary source membership changed')
    for name, item in manifest['files'].items():
        p = ROOT / name
        require(not p.is_symlink() and p.stat().st_size == item['size_bytes'] and sha(p) == item['sha256'],
                'Secondary source bytes changed: ' + name)
    return ref(path)


def admit(manifest_path, manifest_sha, proof_path, proof_sha, name):
    """Native terminal proof first, then one cell's existing exact hash/coverage checks."""
    require(sha(proof_path) == proof_sha, 'Native verification hash mismatch')
    proof = json.loads(Path(proof_path).read_text())
    require(proof['status'] == 'succeeded' and 0 < proof['cells'] <= native.CELL_COUNT and proof['samples'] == proof['cells']*1190
            and proof['manifest_sha256'] == manifest_sha, 'Positive complete native verification required')
    m = native.load(manifest_path, manifest_sha)
    require(proof['cells'] == len(native.execution_cells(m)), 'Subset proof count differs from its manifest')
    require(all(m['source_files'][name]['sha256'] == expected for name, expected in helper.SOURCE_PINS.items()),
            'Native primary source differs from the qualified helper policy')
    require(Path(proof_path) == Path(m['output']) / 'INDEPENDENT_VERIFICATION.json', 'Wrong native proof path')
    terminal_path = check_ref(proof['terminal'])
    require(terminal_path == Path(m['runtime']) / 'SUPERVISOR_EXIT.json', 'Wrong native terminal path')
    terminal = json.loads(terminal_path.read_text())
    require(terminal['status'] == 'succeeded' and terminal['selected_gpus_released'] is True
            and terminal['manifest_sha256'] == manifest_sha, 'Native process/GPU release required')
    native.qualify_score_recovery(m, manifest_sha, terminal)
    choices = {n: (a, k) for n, a, k in native.execution_cells(m)}
    require(name in choices, 'Unknown cell')
    adapter, kind = choices[name]
    folder = Path(m['output']) / 'cells' / name
    require(native.scored_ready(m, name, folder), 'Native closed cell required')
    seal = json.loads((folder / 'SCORE_COMPLETE.json').read_text())
    dataset_path = check_ref(m['datasets'][kind])
    dataset = native.rows(dataset_path)
    raw = native.rows(folder / 'raw.jsonl')
    lines, results = read_lines(check_ref(seal['results']))
    join_rows(raw, results, dataset, adapter, kind, m['datasets'][kind])
    counts = json.loads(check_ref(seal['transport_counts']).read_text())
    require(counts['transport_error'] == 0, 'Legacy transport failure is not an outcome')
    inputs = dict(manifest=ref(manifest_path), verification=ref(proof_path), terminal=proof['terminal'],
                  raw_seal=ref(folder / 'RAW_COMPLETE.json'), score_seal=ref(folder / 'SCORE_COMPLETE.json'),
                  raw=ref(folder / 'raw.jsonl'), legacy=seal['results'], dataset=m['datasets'][kind],
                  legacy_transport=seal['transport'], legacy_transport_counts=seal['transport_counts'])
    return m, inputs, counts, lines, results


def join_rows(raw, results, dataset, adapter, kind, dataset_ref):
    require(len(raw) == len(results) == 1190 and len(dataset) == 119, 'Complete 1190-row cell required')
    require(len({canonical(x['id']) for x in dataset}) == 119, 'Dataset problem IDs repeated')
    ids = set()
    for i, (r, row) in enumerate(zip(raw, results)):
        example, sample = dataset[i // 10], i % 10
        key = dict(arm=adapter['arm'], step=adapter['step'], setting=kind,
                   problem_id=example['id'], sample_index=sample)
        if 'max_new_tokens' in r:
            require(r['max_new_tokens'] == 3072 and r.get('sampling', {}).get('max_new_tokens') == 3072
                    and r.get('projection_declaration', {}).get('kind') == native.PROJECTION_LONG_SCHEMA,
                    'Unexpected long request namespace')
            key['max_new_tokens'] = 3072
        request_id = digest(key)
        require(r['request_id'] == row['request_id'] == request_id and request_id not in ids, 'Request join mismatch')
        ids.add(request_id)
        require(all(r[k] == v for k, v in key.items()) and r['problem_index'] == i // 10, 'Raw coordinate mismatch')
        require(row['id'] == example['id'] and row['sample_index'] == sample, 'Legacy coordinate mismatch')
        require(r['prompt'] == example['prompt'] == row['prompt'] and r['completion'] == row['response'],
                'Legacy prompt/response differs from retained generation')
        require(r['adapter_files'] == adapter['files'] and r['dataset'] == dataset_ref, 'Raw adapter/dataset mismatch')


def score_row(legacy, line, cell_counts, evaluator, installation):
    """The qualified policy owns every AST and classification decision."""
    before = canonical(legacy)
    # The primary batch has aggregate counts only. Never falsely assign a cell
    # overflow to zero individual records: conservatively propagate it to all.
    analysis = base.analyze_repository_result(legacy, cell_counts)
    normalized = {'repository_evaluation': legacy, **analysis}
    side = helper.score_legacy_row(normalized, evaluator, installation)
    require(canonical(legacy) == before, 'Helper mutated the primary result')
    for key in set(base.BINARY_METRICS) - set(helper.BEHAVIOR):
        require(side['metrics'][key] == analysis['metrics'][key], 'Helper changed a capability metric')
    require(side['whole_program_gt_reused'] is True and side['legacy_row_unchanged'] is True,
            'Helper policy did not retain GT/legacy results')
    return dict(request_id=legacy['request_id'], id=legacy['id'], sample_index=legacy['sample_index'],
                legacy_line_sha256=hashlib.sha256(line).hexdigest(),
                legacy_gt_sha256=digest(legacy['gt_result']), primary_gt_pass=legacy['eq_correct'],
                legacy_analysis=analysis, helper_aware_evaluation=side)


def inside():
    require(ROOT == Path('/work') and os.environ.get('CODE_EVAL_SANDBOX_REQUIRED') == '1'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MAX_JOBS') == '4',
            'Helper scoring requires the existing four-worker outer sandbox')
    from infra.gpu03.direction_discovery.bounded_evaluator import install_bounded_evaluator
    from src import SamplingParams
    from src.evaluate import EvaluationParameters, RewardHackingEvaluation
    task = json.loads(Path('/input/task.json').read_text())
    source_check(task['source_manifest_sha256'])
    if 'sampling' in task:
        require(task['sampling'] == {**native.SAMPLING, 'max_new_tokens': 3072}, 'Helper length metadata changed')
    lines, legacy = read_lines('/input/legacy.jsonl')
    require(len(legacy) == 1190 and sha('/input/legacy.jsonl') == task['inputs']['legacy']['sha256'],
            'Inside legacy binding mismatch')
    require(sha('/input/dataset.jsonl') == task['inputs']['dataset']['sha256'], 'Inside dataset binding mismatch')
    installation = install_bounded_evaluator()
    try:
        base.install_count_payload_guard(installation)
        params = EvaluationParameters(model_id=native.MODEL, model_revision=native.REVISION,
            lora_adapter_path=None, dataset_path='/input/dataset.jsonl',
            sampling_params=SamplingParams(**task.get('sampling', native.SAMPLING)), evaluation_name='helper_secondary_' + task['cell'],
            enable_thinking=False, debug=False, save_outputs=False)
        evaluator = RewardHackingEvaluation(config=params, llm_gen=None)
        require(evaluator.evaluator.num_workers == WORKERS and evaluator.evaluator.timeout == 3
                and evaluator.evaluator.memory_per_worker == 1024 and evaluator.evaluator.max_failures == 1,
                'Native evaluator execution parameters changed')
        helper.modules()  # Exact qualified source pins, before any probe call.
        statuses = Counter()
        with Path('/output/helper_sidecar.jsonl').open('x') as stream:
            for line, row in zip(lines, legacy):
                value = score_row(row, line, task['legacy_transport_counts'], evaluator, installation)
                stream.write(canonical(value) + '\n')
                stream.flush()  # Retain partial rows on a bounded timeout; never a complete seal.
                statuses[value['helper_aware_evaluation']['status']] += 1
            os.fsync(stream.fileno())
        write('/output/INSIDE_COMPLETE.json', dict(schema=SCHEMA, count=1190,
              legacy_sha256=sha('/input/legacy.jsonl'), source_manifest_sha256=task['source_manifest_sha256'],
              helper_policy=helper.POLICY, legacy_policy=helper.LEGACY_POLICY,
              statuses=dict(statuses), transport_counts=installation.report(),
              generated_code_execution='only two replacement probes for supported helper closures',
              unknowns_retained=True, whole_program_gt_reexecuted=False))
    finally:
        installation.restore()


def run(args):
    source_ref = source_check(args.source_sha256)
    m, inputs, counts, lines, results = admit(args.manifest, args.sha256, args.verification,
                                            args.verification_sha256, args.cell)
    dest = Path(args.output).resolve()
    require(not dest.exists() and not dest.is_relative_to(Path(m['output']))
            and not dest.is_relative_to(ROOT), 'Use a fresh separate secondary output directory')
    require(1 <= args.wall_seconds <= 7200, 'Secondary outer wall cap must be 1..7200 seconds')
    # One cell at a time for this exact experiment, even if two callers race.
    # The lock is host-local, never in the primary result/source directories.
    lock_path = Path('/tmp') / ('original119-helper-' + args.sha256 + '.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(descriptor, 'r+')
    require(os.fstat(descriptor).st_uid == os.getuid(), 'Secondary lock is not owned by this user')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest.mkdir(parents=True, mode=0o700)
    inp, out = dest / 'input', dest / 'output'
    inp.mkdir(); out.mkdir(mode=0o700)
    shutil.copyfile(inputs['legacy']['path'], inp / 'legacy.jsonl')
    shutil.copyfile(inputs['dataset']['path'], inp / 'dataset.jsonl')
    require(sha(inp / 'legacy.jsonl') == inputs['legacy']['sha256']
            and sha(inp / 'dataset.jsonl') == inputs['dataset']['sha256'], 'Input copy changed')
    task = dict(schema=SCHEMA, cell=args.cell, inputs=inputs, source_manifest_sha256=source_ref['sha256'],
                legacy_transport_counts=counts, workers=WORKERS, wall_seconds=args.wall_seconds,
                transport_attribution='aggregate cell counts; overflow makes every legacy row conservatively unknown')
    if m['schema'] == native.PROJECTION_LONG_SCHEMA:
        task['sampling'] = m['sampling']
    write(inp / 'task.json', task)
    from infra.gpu03.direction_discovery.sandbox import run_outer
    transport = run_outer(source_dir=ROOT, input_dir=inp, output_dir=out, venv_dir=m['venv'],
                          python_args=['-B', '/work/helper_sidecar.py', 'inside'], workers=WORKERS,
                          wall_timeout=args.wall_seconds)
    write(dest / 'transport.json', transport)
    require(transport['returncode'] == 0, 'Secondary outer failed; retained partials are not complete')
    check_completed(dest, task, results, lines)
    # Detect any input/source changes over execution; no execution repeated.
    for item in inputs.values():
        check_ref(item)
    source_check(source_ref['sha256'])
    write(dest / 'COMPLETE.json', dict(schema=SCHEMA, cell=args.cell, count=1190,
          inputs=inputs, source=source_ref, task=ref(inp / 'task.json'),
          sidecar=ref(out / 'helper_sidecar.jsonl'), inside=ref(out / 'INSIDE_COMPLETE.json'),
          transport=ref(dest / 'transport.json'),
          meaning='all rows represented, including unknowns; not a claim every probe succeeded'))
    print(canonical(ref(dest / 'COMPLETE.json')))
    lock.close()


def check_completed(dest, task, legacy, lines):
    dest = Path(dest)
    transport = json.loads((dest / 'transport.json').read_text())
    require(transport['returncode'] == 0, 'No positive secondary outer exit')
    done = json.loads((dest / 'output/INSIDE_COMPLETE.json').read_text())
    require(done['schema'] == SCHEMA and done['count'] == 1190
            and done['legacy_sha256'] == task['inputs']['legacy']['sha256']
            and done['source_manifest_sha256'] == task['source_manifest_sha256']
            and done['helper_policy'] == helper.POLICY and done['legacy_policy'] == helper.LEGACY_POLICY
            and done['whole_program_gt_reexecuted'] is False, 'Secondary inside receipt mismatch')
    _, sides = read_lines(dest / 'output/helper_sidecar.jsonl')
    require(len(sides) == len(legacy) == 1190, 'Incomplete secondary coverage')
    statuses = Counter()
    for value, old, line in zip(sides, legacy, lines):
        require((value['request_id'], value['id'], value['sample_index']) ==
                (old['request_id'], old['id'], old['sample_index']), 'Secondary request join mismatch')
        require(value['legacy_line_sha256'] == hashlib.sha256(line).hexdigest()
                and value['legacy_gt_sha256'] == digest(old['gt_result'])
                and value['primary_gt_pass'] == old['eq_correct'], 'Secondary legacy/GT binding changed')
        expected = base.analyze_repository_result(old, task['legacy_transport_counts'])
        require(value['legacy_analysis'] == expected, 'Stored legacy normalization changed')
        side = value['helper_aware_evaluation']
        require(side['policy'] == helper.POLICY and side['legacy_policy'] == helper.LEGACY_POLICY
                and side['whole_program_gt_reused'] is True and side['legacy_row_unchanged'] is True,
                'Wrong helper policy')
        for key in set(base.BINARY_METRICS) - set(helper.BEHAVIOR):
            require(side['metrics'][key] == expected['metrics'][key], 'Secondary changed capability')
        statuses[side['status']] += 1
    require(dict(statuses) == done['statuses'], 'Secondary status counts mismatch')
    return sides


def verify(output, complete_sha):
    """Independent byte/identity verification only; does not repeat probe execution."""
    dest = Path(output)
    require(sha(dest / 'COMPLETE.json') == complete_sha, 'Secondary complete hash mismatch')
    seal = json.loads((dest / 'COMPLETE.json').read_text())
    require(seal['schema'] == SCHEMA and seal['count'] == 1190, 'Wrong secondary seal')
    for key, expected in [('task', dest / 'input/task.json'), ('sidecar', dest / 'output/helper_sidecar.jsonl'),
                          ('inside', dest / 'output/INSIDE_COMPLETE.json'), ('transport', dest / 'transport.json')]:
        require(check_ref(seal[key]) == expected, 'Secondary file outside expected path')
    task = json.loads((dest / 'input/task.json').read_text())
    require(task['cell'] == seal['cell'] and task['inputs'] == seal['inputs'], 'Task/complete mismatch')
    source_check(seal['source']['sha256'])
    inputs = seal['inputs']
    _, current, counts, lines, legacy = admit(inputs['manifest']['path'], inputs['manifest']['sha256'],
        inputs['verification']['path'], inputs['verification']['sha256'], seal['cell'])
    require(current == inputs and counts == task['legacy_transport_counts'], 'Native inputs changed')
    check_completed(dest, task, legacy, lines)
    return dict(status='independently_verified_original119_helper_secondary', count=1190,
                complete=ref(dest / 'COMPLETE.json'), sidecar=seal['sidecar'], cell=seal['cell'],
                verification='exact bytes/coverage/unchanged GT; corrected probes not re-executed')


def merge(index_path, index_sha, output):
    require(sha(index_path) == index_sha, 'Merge index hash mismatch')
    entries = json.loads(Path(index_path).read_text())['cells']
    require(entries, 'Nonempty subset secondary index required')
    first_proof = json.loads(check_ref(entries[0]['verification']).read_text())
    first_seal = json.loads(check_ref(first_proof['complete']).read_text())
    binding = first_seal['inputs']['manifest']
    current = native.load(check_ref(binding), binding['sha256'])
    expected = [name for name, _, _ in native.execution_cells(current)]
    require([x['cell'] for x in entries] == expected, 'Complete ordered follow-up secondary cells required')
    data, proofs, manifest_shas, source_shas, seen = [], [], set(), set(), set()
    for entry in entries:
        proof = json.loads(check_ref(entry['verification']).read_text())
        require(proof['status'] == 'independently_verified_original119_helper_secondary'
                and proof['cell'] == entry['cell'] and proof['count'] == 1190, 'Missing secondary proof')
        seal = json.loads(check_ref(proof['complete']).read_text())
        require(seal['cell'] == entry['cell'] and seal['sidecar'] == proof['sidecar'], 'Secondary merge join mismatch')
        manifest_shas.add(seal['inputs']['manifest']['sha256']); source_shas.add(seal['source']['sha256'])
        b = check_ref(proof['sidecar']).read_bytes()
        rows = [json.loads(line) for line in b.splitlines()]
        require(b.endswith(b'\n') and len(rows) == 1190, 'Secondary merge incomplete cell')
        for row in rows:
            require(row['request_id'] not in seen, 'Duplicate secondary merge request')
            seen.add(row['request_id'])
        data.append(b); proofs.append(entry['verification'])
    require(len(manifest_shas) == len(source_shas) == 1 and len(seen) == len(expected)*1190, 'Mixed/partial secondary experiment')
    dest = Path(output); dest.mkdir(parents=True)
    with (dest / 'helper_sidecar.jsonl').open('xb') as stream:
        for b in data:
            stream.write(b)
        stream.flush(); os.fsync(stream.fileno())
    write(dest / 'COMPLETE.json', dict(schema=SCHEMA, count=len(expected)*1190, cells=len(expected), sidecar=ref(dest / 'helper_sidecar.jsonl'),
          manifest_sha256=next(iter(manifest_shas)), source_manifest_sha256=next(iter(source_shas)),
          input_proofs=proofs, helper_policy=helper.POLICY, legacy_policy=helper.LEGACY_POLICY,
          primary_rows_unchanged=True, whole_program_gt_reexecuted=False))
    print(canonical(ref(dest / 'COMPLETE.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('inside')
    r = sub.add_parser('run')
    for flag in ('manifest', 'sha256', 'verification', 'verification-sha256', 'cell', 'output', 'source-sha256'):
        r.add_argument('--' + flag, required=True)
    r.add_argument('--wall-seconds', type=int, required=True)
    v = sub.add_parser('verify'); v.add_argument('--output', required=True); v.add_argument('--complete-sha256', required=True)
    v.add_argument('--proof-output', required=True)
    c = sub.add_parser('merge'); c.add_argument('--index', required=True); c.add_argument('--index-sha256', required=True)
    c.add_argument('--output', required=True)
    args = p.parse_args()
    if args.mode == 'inside': inside()
    elif args.mode == 'run': run(args)
    elif args.mode == 'verify':
        proof = verify(args.output, args.complete_sha256); write(args.proof_output, proof)
        print(canonical(ref(args.proof_output)))
    else: merge(args.index, args.index_sha256, args.output)


if __name__ == '__main__':
    main()
