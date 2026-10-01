"""Prepare one CPU evaluation plan after all eight real pilot waves close.

This successor preserves the original plan, requests, eight-row generation
batches and evaluator. Each wave's actual supervisor, tasks, release evidence,
journals and exports are audited independently; no combined supervisor is
invented. Qualified controls are admitted once, then all 15,168 exact retained
rows are copied to the existing bounded evaluator-lane input format.

No model, tokenizer, subprocess, network or generated-code evaluator is called.
The same global PILOT_EVAL_TASK_BINDING.json excludes old/new planner replay.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import socket

from . import build_causal_eval_plan as original
from . import build_causal_plan as plan_io
from . import causal_qualification as qualification
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import fit_worker as io
from . import prepare_causal_waves as wave_preparation
from .dispatch_phase import ROOT, PYTHON
from .io_contract import canonical, hash_value

SCHEMA = 'task_specific_causal_wave_eval_plan_v1'
INDEX_SCHEMA = 'task_specific_completed_causal_waves_v1'
COLLECTION_SCHEMA = 'task_specific_causal_wave_collection_v1'
AUDIT_SCHEMA = 'task_specific_closed_wave_pilot_inputs_v1'
SELF_SOURCE = 'infra/gpu03/task_specific_discovery/build_causal_wave_eval_plan.py'
WAVE_AUDITOR_SOURCE = 'infra/gpu03/task_specific_discovery/bind_causal_wave.py'
WAVE_PREPARER_SOURCE = 'infra/gpu03/task_specific_discovery/prepare_causal_waves.py'
AUDIT_IO_SOURCE = 'infra/gpu03/task_specific_discovery/causal_qualification_v2.py'
TOTAL, NEW, REUSED, WAVES, LANES, MAX_PRODUCERS = 15168, 15000, 168, 8, 8, 72
require = qualification.require
read_reference, binding = original.read_reference, original.binding


def _plan_inputs(collection_ref, evidence):
    collection = read_reference(collection_ref, evidence)
    require(collection['schema'] == COLLECTION_SCHEMA and collection['status'] == 'prepared_not_dispatched',
            'Expected the immutable eight-wave generation collection')
    plan_ref, qualified_ref = collection['plan'], collection['qualification']
    plan = read_reference(plan_ref, evidence)
    # Only the exact pinned base snapshot may use its observed hardlinks. All
    # producer/source/proof identities below remain the original qualified ones.
    audit_io.configure_from_plan(evidence, plan)
    evidence.bind(Path(audit_io.__file__).resolve())
    require(plan['schema'] == plan_io.SCHEMA and plan['status'] == 'prepared_not_dispatched'
            and plan['model_revision'] == worker.REVISION and plan['sampling'] == worker.SAMPLING
            and plan['profile'] == worker.PROFILE and plan['batch_size'] == 8,
            'Frozen pilot profile or sampling differs')
    require(collection['run_token'] == plan['run_token'] and collection['source_root'] == plan['source_root']
            and collection['qualified_source_files'] == plan['qualified_source_files']
            and collection['profile'] == plan['profile'], 'Collection source/profile identity drift')
    for path, value in plan['bound_files'].items():
        evidence.bind(path, value)
    prepared = evidence.jsonl(Path(plan['prepared_rows']), row_limit=96)
    rows = {r['record_id']: r for r in prepared}
    coordinates = {(canonical(r['problem_id']).decode(), r['protocol']) for r in prepared}
    problems = {pid for pid, _ in coordinates}
    require(len(rows) == len(prepared) == 96 and len(problems) == 32 and len(coordinates) == 96
            and all(r['problem_split'] == 'direction_fit' for r in prepared)
            and coordinates == {(pid, protocol) for pid in problems for protocol in original.PROTOCOLS},
            'Pilot requires exactly the frozen 32 fitting problems and three protocols')
    expected = plan_io.requests_for(prepared, plan['pilot_condition_ids'], plan['conditions'], qualification=False)
    q_expected = plan_io.requests_for(prepared, plan['qualification_condition_ids'], plan['conditions'], qualification=True)
    require(plan['pilot_requests'] == expected and plan['qualification_requests'] == q_expected
            and len(expected) == TOTAL and len(q_expected) == 312
            and len({r['request_id'] for r in expected}) == TOTAL
            and plan['budgets']['pilot_including_reuse'] == TOTAL and plan['budgets']['qualification'] == 312,
            'Prospective pilot request universe/count changed')
    universe = {r['request_id']: r for r in expected}
    reused, reused_bindings = plan_io.reuse_from_qualification(plan, qualified_ref)
    proof = read_reference(qualified_ref, evidence)
    require(len(reused) == REUSED and proof['run_token'] == plan['run_token']
            and collection['reused_requests'] == reused
            and collection['expected_total_requests'] == TOTAL and collection['expected_new_requests'] == NEW
            and collection['expected_reused_requests'] == REUSED, 'Exact qualification reuse/count differs')
    for path, value in reused_bindings.items():
        evidence.bind(path, value)
    claim = read_reference(collection['pilot_claim'], evidence)
    require(Path(collection['pilot_claim']['path']) == Path(plan_ref['path']).parent / 'PILOT_TASK_BINDING.json',
            'Wave collection lacks the shared exclusive pilot preparation claim')
    evidence.bind(Path(wave_preparation.__file__).resolve(), binding(collection['planner']))
    assignments = wave_preparation.partition(plan, reused)
    require(claim['schema'] == wave_preparation.CLAIM_SCHEMA and claim['plan'] == plan_ref
            and claim['run_token'] == plan['run_token'] and claim['qualification'] == qualified_ref
            and claim['request_replay_authorized'] is False
            and claim['assignments'] == assignments
            and claim['assignments_sha256'] == collection['assignments_sha256'] == hash_value(assignments),
            'Pilot preparation claim differs from exact frozen assignments')
    waves = collection['waves']
    require(len(waves) == WAVES and [w['wave_index'] for w in waves] == list(range(WAVES)),
            'Missing, duplicate or reordered prospective wave')
    require([{k: v for k, v in wave.items() if k != 'phase_inputs'} for wave in waves] == assignments,
            'Wave cuts or original cyclic GPU lane assignments changed')
    remaining = [r for r in expected if r['request_id'] not in reused]
    batches = qualification.batches(remaining)
    require(len(batches) == 1875 and all(len(batch) == 8 for batch in batches), 'Original physical batches differ')
    covered, indices = [], []
    for wave in waves:
        batch_indices = wave['global_batch_indices']
        require(batch_indices and all(type(i) is int and 0 <= i < len(batches) for i in batch_indices),
                'Invalid global batch index')
        request_ids = [r['request_id'] for i in batch_indices for r in batches[i]]
        require(wave['request_ids'] == request_ids and wave['expected_new_requests'] == len(request_ids),
                'Wave split changes physical batches or requests')
        covered.extend(request_ids); indices.extend(batch_indices)
    require(indices == list(range(len(batches))) and covered == [r['request_id'] for r in remaining],
            'Wave collection duplicates/reorders/omits prospective pilot requests')
    return collection, plan, rows, universe, reused


def _qualified_controls(plan, rows, universe, reused, retained, evidence):
    """Original qualified-control admission, retaining real file offsets."""
    by_file = {}
    for rid, saved in reused.items():
        by_file.setdefault(saved['generation_file']['path'], set()).add(rid)
    require(len(by_file) <= 8, 'Too many original qualification producers')
    for path, wanted in by_file.items():
        for export, location in original.stream_rows(Path(path), evidence, max_rows=312):
            rid = export['request_id']
            if rid not in wanted:
                continue
            saved, request = reused[rid], universe[rid]
            require(rid not in retained and location['generation_row_sha256'] == saved['generation_row_sha256'],
                    'Qualified control changed or was regenerated')
            primitive = read_reference(saved['primitive_file'], evidence, 8 << 20)
            final = read_reference(saved['final_file'], evidence, 8 << 20)
            require(primitive['request_id'] == final['request_id'] == rid
                    and primitive['request_sha256'] == final['request_sha256'] == hash_value(request)
                    and primitive['run_token'] == final['run_token'] == plan['run_token']
                    and final['primitive_file_sha256'] == saved['primitive_file']['sha256'],
                    'Qualified control primitive/final identity differs')
            original.validate_export(export, request, rows[request['record_id']], primitive['payload'],
                primitive['input_sha256'], plan['conditions'][request['condition_id']], 8)
            retained[rid] = {**location, 'origin': 'qualification', 'request_sha256': hash_value(request),
                            'primitive_file': saved['primitive_file'], 'final_file': saved['final_file']}


def audit_waves(collection_ref, manifest_refs, evidence):
    from . import bind_causal_wave as wave_io
    require(len(manifest_refs) == WAVES and len({r['path'] for r in manifest_refs}) == WAVES,
            'Eight distinct actual wave supervisor manifests required')
    collection, plan, rows, universe, reused = _plan_inputs(collection_ref, evidence)
    retained, actual_proofs, gpu_allocation = {}, [], None
    global_window = None
    for index, (manifest_ref, wave) in enumerate(zip(manifest_refs, collection['waves'])):
        manifest = read_reference(manifest_ref, evidence)
        require(manifest['collection'] == collection_ref and manifest['wave_index'] == index
                and manifest['predecessor_manifests'] == manifest_refs[:index],
                'Actual wave collection/index/predecessor chain differs')
        if gpu_allocation is None:
            gpu_allocation = manifest['gpu_uuids']
            global_window = manifest['global_generation_window']
        require(len(manifest['workers']) == LANES and len(manifest['gpu_uuids']) == LANES
                and manifest['gpu_uuids'] == gpu_allocation
                and manifest['global_generation_window'] == global_window,
                'Wave GPU allocation or common generation window changed')
        proof, saved = wave_io.audit_wave(manifest_ref, collection, wave, evidence)
        # audit_wave independently reads this actual supervisor release,
        # immutable worker journals, raw primitives and canonical exports.
        require(set(saved) == set(wave['request_ids']) and len(saved) == wave['expected_new_requests'],
                'Actual wave does not exactly cover its prospectively assigned requests')
        for rid, item in saved.items():
            require(rid in universe and rid not in reused and rid not in retained,
                    'Duplicate/unplanned wave request or regenerated qualified control')
            require(item['request_sha256'] == hash_value(universe[rid]) and item['origin'] == 'pilot',
                    'Retained wave row has different frozen request identity')
            retained[rid] = {**item, 'wave_index': index}
        actual_proofs.append(proof)
    require(len(retained) == NEW and set(retained) == set(universe) - set(reused),
            'Eight terminal waves do not cover all 15,000 new requests')
    _qualified_controls(plan, rows, universe, reused, retained, evidence)
    require(len(retained) == TOTAL and set(retained) == set(universe), 'Full pilot coverage is not exactly once')
    require(len({r['generation_file']['path'] for r in retained.values()}) <= MAX_PRODUCERS,
            'More producers than eight waves plus qualification')
    evidence.unchanged()
    audit = {'schema': AUDIT_SCHEMA, 'status': 'all_pilot_generations_independently_verified',
        'plan': collection['plan'], 'wave_collection': collection_ref, 'qualification': collection['qualification'],
        'wave_manifests': manifest_refs, 'actual_wave_closures': actual_proofs,
        'gpu_uuids': gpu_allocation, 'global_generation_window': global_window,
        'requests': TOTAL, 'new_requests': NEW, 'reused_requests': REUSED, 'retained_requests': retained,
        'all_returned_requests_retained': True, 'gpu_workers_released': True,
        'generation_or_evaluation_performed': False, 'behavioral_success_claimed': False,
        'synthetic_combined_supervisor': False}
    return plan, rows, retained, audit


def assemble(plan, prepared, retained, output, source_root, source_files):
    """Original lane format and sharding, allowing at most 72 closed streams.

    The old helper's 16-producer cap is specific to one generation phase. This
    narrow copy changes only that producer bound; evaluator inputs and ordering
    remain the same, with <=256 rows /64MiB per shard and exactly eight lanes.
    """
    output = Path(output); inputs = output / 'inputs'; inputs.mkdir(mode=0o700)
    (output / 'results').mkdir(mode=0o700)
    shards, copied = [], set()
    paths = {r['generation_file']['path'] for r in retained.values()}
    require(0 < len(paths) <= MAX_PRODUCERS, 'Too many closed generation producers')
    with ExitStack() as stack:
        streams = {path: stack.enter_context(Path(path).open('rb')) for path in paths}
        for protocol in original.PROTOCOLS:
            group, byte_count = [], 0

            def flush():
                nonlocal group, byte_count
                if not group:
                    return
                sid = f'{protocol}-{len(shards):03d}'
                folder = inputs / sid; folder.mkdir(mode=0o700)
                examples, generated = {}, []
                for request, row in group:
                    example = prepared[request['record_id']]['dataset_row']
                    key = canonical(example['id']).decode()
                    require(key not in examples or examples[key] == example, 'Conflicting trusted examples')
                    examples[key] = example; generated.append(row)
                gref = original.write_rows(folder / f'{protocol}.generated.jsonl', generated)
                dref = original.write_rows(folder / f'{protocol}.dataset.jsonl', examples.values())
                require(gref['size_bytes'] <= original.evaluator.MAX_FILE_BYTES
                        and dref['size_bytes'] <= original.evaluator.MAX_FILE_BYTES, 'Evaluation shard byte bound exceeded')
                shards.append({'id': sid, 'protocol': protocol, 'generated': gref, 'dataset': dref,
                               'request_ids': [r['request_id'] for _, r in group]})
                group, byte_count = [], 0

            for request in plan['pilot_requests']:
                if request['protocol'] != protocol:
                    continue
                rid = request['request_id']; item = retained[rid]
                require(rid not in copied, 'Duplicate planned evaluation'); copied.add(rid)
                stream = streams[item['generation_file']['path']]; stream.seek(item['offset'])
                raw = stream.read(item['length']); row = qualification.decode(raw)
                require(canonical(row) + b'\n' == raw and hash_value(row) == item['generation_row_sha256'],
                        'Audited generation changed before sharding')
                require(row['request_id'] == rid and all(row.get(k) == v for k, v in request.items()), 'Generation/request differs')
                original.evaluator.validate_request(row, prepared[request['record_id']]['dataset_row'], protocol)
                require(len(raw) <= original.evaluator.MAX_LINE_BYTES, 'Evaluation row exceeds bound')
                if len(group) == 256 or byte_count + len(raw) > original.evaluator.MAX_FILE_BYTES:
                    flush()
                group.append((request, row)); byte_count += len(raw)
            flush()
    require(copied == set(retained), 'Evaluation plan lost/added requests')
    groups = [[] for _ in range(LANES)]
    for index, shard in enumerate(shards):
        groups[index % LANES].append(shard)
    require(all(0 < len(g) <= 64 for g in groups), 'Eight bounded evaluator lanes required')
    tasks = []
    for lane, values in enumerate(groups):
        task = {'schema': original.LANE_SCHEMA, 'run_token': plan['run_token'], 'source_dir': str(source_root),
            'source_files': source_files, 'venv_dir': str(Path(PYTHON).parent.parent),
            'output': str(output / 'results' / f'lane_{lane:02d}'), 'deadline_seconds': 14400,
            'problem_split': 'direction_fit',
            'shards': [{k: s[k] for k in ('id', 'protocol', 'generated', 'dataset')} for s in values]}
        path = output / f'task_{lane:02d}.json'; io.write_json(path, task)
        reference = {'path': str(path), **io.binding(path)}
        tasks.append({'lane': lane, 'task': reference, 'requests': sum(len(s['request_ids']) for s in values),
            'argv': [PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.causal_eval_lane',
                     '--task', str(path), '--sha256', reference['sha256']]})
    return tasks, shards


def build(collection_ref, manifest_refs, authored_task_ref, authored_proof_ref, authored_supervisor_ref,
          source_root, output, *, index_ref=None):
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists()
            and not output.is_symlink(), 'Fresh canonical evaluation output required')
    evidence = audit_io.Evidence()
    collection = read_reference(collection_ref, evidence)
    plan_ref = collection['plan']
    claim = Path(plan_ref['path']).parent / 'PILOT_EVAL_TASK_BINDING.json'
    if claim.exists() or claim.is_symlink():
        raise FileExistsError('Evaluation planning already consumed; inspect its retained claim')
    if index_ref is not None:
        index = read_reference(index_ref, evidence)
        require(index == {'schema': INDEX_SCHEMA, 'collection': collection_ref, 'manifests': manifest_refs},
                'Completed-wave reference index differs')
    plan, prepared, retained, audit = audit_waves(collection_ref, manifest_refs, evidence)
    files, task, authored_release = original.source_and_authored(source_root, authored_task_ref, authored_proof_ref,
                                                               authored_supervisor_ref, evidence)
    require({SELF_SOURCE, WAVE_AUDITOR_SOURCE, WAVE_PREPARER_SOURCE, AUDIT_IO_SOURCE} <= set(files),
            'Wave preparation/audit source missing')
    require(files[AUDIT_IO_SOURCE] == binding(evidence.bind(Path(audit_io.__file__).resolve())),
            'Consumed input-audit helper differs from evaluator package')
    require(task['run_token'] == plan['run_token'], 'Authored evaluator qualification token differs')
    audit.update(independently_hashed_files=dict(evidence.files), auditor_sha256=io.sha256(__file__),
                 input_audit_policy=evidence.audit_io_provenance(),
                 authored_qualification=authored_proof_ref, authored_supervisor_release=authored_release)
    # Persist the exclusive claim and its directory before writing shards.
    worker.write(claim, {'schema': SCHEMA, 'plan': plan_ref, 'wave_collection': collection_ref,
        'wave_manifests': manifest_refs, 'qualification': collection['qualification'],
        'authored_task': authored_task_ref, 'authored_proof': authored_proof_ref, 'authored_supervisor': authored_supervisor_ref,
        'source_root': str(source_root), 'source_files_sha256': hash_value(files), 'output': str(output), 'lanes': LANES,
        'request_ids': [r['request_id'] for r in plan['pilot_requests']], 'input_audit_sha256': hash_value(audit),
        'evaluation_replay_authorized': False})
    output.mkdir(mode=0o700)
    tasks, shards = assemble(plan, prepared, retained, output, source_root, files)
    evidence.unchanged()
    audit_path = output / 'INPUT_AUDIT.json'; io.write_json(audit_path, audit)
    bound = {str(Path(source_root) / name): value for name, value in files.items()}
    refs = [collection_ref, plan_ref, collection['qualification'], *manifest_refs,
            authored_task_ref, authored_proof_ref, authored_supervisor_ref,
            {'path': str(claim), **io.binding(claim)}, {'path': str(audit_path), **io.binding(audit_path)}]
    if index_ref is not None:
        refs.append(index_ref)
    for ref in refs:
        bound[ref['path']] = binding(ref)
    for shard in shards:
        for kind in ('generated', 'dataset'):
            bound[shard[kind]['path']] = binding(shard[kind])
    for item in tasks:
        bound[item['task']['path']] = binding(item['task'])
    phase = {'schema': original.SCHEMA, 'preparation_schema': SCHEMA, 'status': 'prepared_not_dispatched',
        'run_token': plan['run_token'], 'source_root': str(source_root), 'tasks': tasks, 'shards': shards,
        'requests': TOTAL, 'new_evaluations': TOTAL, 'generation_performed': False, 'evaluation_performed': False,
        'training': False, 'problem_split': 'direction_fit', 'bound_files': bound,
        'input_audit': {'path': str(audit_path), **io.binding(audit_path)},
        'limits': {'lanes': LANES, 'evaluator_workers_per_lane': 2, 'records_per_shard': 256,
                   'bytes_per_shard': original.evaluator.MAX_FILE_BYTES, 'deadline_seconds': 14400},
        'dispatch_authorized_by_this_file': False, 'resource_supervisor_binding_required': True}
    path = output / 'PHASE_INPUTS.json'; io.write_json(path, phase)
    return {'path': str(path), **io.binding(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection', 'wave-manifests', 'authored-task', 'authored-proof', 'authored-supervisor'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--source-root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357 and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Preparation requires accepted host owner CPU-only environment')
    require(Path(args.output).parent == Path(ROOT) and Path(args.source_root) == Path(__file__).resolve().parents[3]
            and Path(args.source_root).parent == Path(ROOT), 'Preparation must use this run and bound source')
    refs = []
    for name in ('collection', 'wave_manifests', 'authored_task', 'authored_proof', 'authored_supervisor'):
        path = Path(getattr(args, name))
        require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT), 'Evidence outside this run')
        ref = {'path': str(path), **io.binding(path)}
        require(ref['sha256'] == getattr(args, name + '_sha256'), 'Explicit input SHA differs')
        refs.append(ref)
    collection_ref, index_ref, authored_task, authored_proof, authored_supervisor = refs
    index = read_reference(index_ref, qualification.Evidence())
    require(index['schema'] == INDEX_SCHEMA and index['collection'] == collection_ref, 'Wrong completed-wave index')
    require(plan_io.read_json(collection_ref['path'])['run_token'] == Path(ROOT).name, 'Collection belongs to another run')
    for ref in index['manifests']:
        require(Path(ref['path']).is_relative_to(ROOT), 'Wave manifest outside this run')
    result = build(collection_ref, index['manifests'], authored_task, authored_proof, authored_supervisor,
                   args.source_root, args.output, index_ref=index_ref)
    print(canonical(result).decode())


if __name__ == '__main__':
    main()
