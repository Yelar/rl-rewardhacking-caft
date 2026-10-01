"""Freeze eight disjoint pilot waves without changing the qualified worker.

Pure CPU preparation; no dispatch, GPU allocation, generation or evaluation.
All assignments are claimed before any task output is written. An interrupted
preparation remains consumed and cannot be used to authorize request replay.
The separate wave binder must enforce the overall 24-hour generation deadline,
four-hour phase limits and positive predecessor release before each dispatch.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import socket

from . import build_causal_plan as original
from . import causal_qualification as qualification
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import causal_prompts as prompts
from . import fit_worker as io
from . import io_contract
from .dispatch_phase import ROOT, PYTHON
from .io_contract import canonical, hash_value

SCHEMA = 'task_specific_causal_wave_collection_v1'
CLAIM_SCHEMA = 'task_specific_causal_wave_claim_v1'
WAVES = LANES = BATCH_SIZE = 8
TOTAL, NEW, REUSED = 15168, 15000, 168
WAVE_SECONDS, OVERALL_SECONDS = 14400, 86400
MAX_LANE_REQUESTS = 240
require = worker.require


def binding(reference):
    return {key: reference[key] for key in ('sha256', 'size_bytes')}


def read_ref(reference, evidence):
    return evidence.json(reference['path'], 64 << 20, binding(reference))


def partition(plan, reused):
    """Retain the old global batch and cyclic lane order, only inserting stops."""
    universe = plan['pilot_requests']
    ids = [r['request_id'] for r in universe]
    require(len(ids) == TOTAL and len(set(ids)) == TOTAL and len(reused) == REUSED
            and set(reused) <= set(ids), 'Exact frozen pilot/reuse coverage required')
    # Reuse must remove complete physical batches, never fragments that could
    # repad the remaining rows and change their numerical execution.
    original_batches = [universe[i:i + BATCH_SIZE] for i in range(0, TOTAL, BATCH_SIZE)]
    require(all(len({r['request_id'] in reused for r in batch}) == 1 for batch in original_batches),
            'Qualified reuse splits an original physical batch')
    batches = [batch for batch in original_batches if batch[0]['request_id'] not in reused]
    require(len(batches) * BATCH_SIZE == NEW and all(len(batch) == BATCH_SIZE
            and len({(r['condition_id'], r['protocol']) for r in batch}) == 1 for batch in batches),
            'Frozen batch shape or condition/protocol differs')
    require(qualification.batches([r for batch in batches for r in batch]) == batches,
            'Worker condition grouping would reorder physical batches')
    quotient, remainder = divmod(len(batches), WAVES)
    assignments, start = [], 0
    for wave in range(WAVES):
        stop = start + quotient + (wave < remainder)
        indices = list(range(start, stop))
        lanes = []
        for lane in range(LANES):
            selected = [index for index in indices if index % LANES == lane]
            requests = [r for index in selected for r in batches[index]]
            require(0 < len(requests) <= MAX_LANE_REQUESTS and len(requests) % BATCH_SIZE == 0,
                    'Wave exceeds thirty unchanged batches per lane')
            require(qualification.batches(requests) == [batches[index] for index in selected],
                    'Wave changes worker batch membership/order')
            lanes.append({'lane': lane, 'global_batch_indices': selected,
                          'request_ids': [r['request_id'] for r in requests]})
        assignments.append({'wave_index': wave, 'global_batch_indices': indices, 'lanes': lanes,
                            'request_ids': [r['request_id'] for index in indices for r in batches[index]],
                            'expected_new_requests': len(indices) * BATCH_SIZE})
        start = stop
    require(start == len(batches) and sum(w['expected_new_requests'] for w in assignments) == NEW,
            'Incomplete wave partition')
    return assignments


def checked_plan(plan_ref, qualified_ref, evidence):
    plan = read_ref(plan_ref, evidence)
    require(plan['schema'] == original.SCHEMA and plan['status'] == 'prepared_not_dispatched'
            and plan['model_revision'] == worker.REVISION and plan['sampling'] == worker.SAMPLING
            and plan['profile'] == worker.PROFILE and type(plan['batch_size']) is int
            and plan['batch_size'] == BATCH_SIZE, 'Frozen science/profile differs')
    audit_io.configure_from_plan(evidence, plan)
    for path, value in plan['bound_files'].items():
        evidence.bind(path, value)
    source = Path(plan['source_root'])
    source_files = evidence.json(source / 'SOURCE_FILES.json', 64 << 20)
    require(source_files == plan['qualified_source_files']
            and set(original.REQUIRED_SOURCE) | set(worker.REQUIRED_SOURCE_FILES) <= set(source_files),
            'Qualified source inventory differs')
    for name, value in source_files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and str(Path(name)) == name,
                'Invalid original source path')
        evidence.bind(source / name, value)
    # A new preparation package may add helpers, but the imported original
    # request, hash, source and proof validators must retain their qualified bytes.
    for module in (original, qualification, worker, prompts, io, io_contract):
        path = Path(module.__file__).resolve()
        name = path.relative_to(Path(__file__).resolve().parents[3]).as_posix()
        require(name in source_files, 'Consumed planner dependency absent from original source')
        evidence.bind(path, source_files[name])
    prepared = evidence.jsonl(plan['prepared_rows'], row_limit=96)
    require(len(prepared) == 96 and len({r['record_id'] for r in prepared}) == 96
            and all(r['problem_split'] == 'direction_fit' for r in prepared), 'Fitting prompt coverage differs')
    coordinates = {(hash_value(r['problem_id']), r['protocol']) for r in prepared}
    problems = {p for p, _ in coordinates}
    require(len(problems) == 32 and coordinates == {(p, protocol) for p in problems for protocol in prompts.PROTOCOLS},
            'Pilot problem/protocol grid differs')
    for stage, is_qualification in (('pilot', False), ('qualification', True)):
        require(plan[stage + '_requests'] == original.requests_for(prepared, plan[stage + '_condition_ids'],
                plan['conditions'], qualification=is_qualification), 'Frozen request IDs/order/seeds differ')
    require(plan['budgets'] == {'qualification': 312, 'pilot_including_reuse': TOTAL,
                               'reusable_pilot_requests': REUSED}, 'This wave plan requires the full frozen finite panel')
    reused, extra = original.reuse_from_qualification(plan, qualified_ref)
    proof = read_ref(qualified_ref, evidence)
    require(proof['run_token'] == plan['run_token'], 'Qualification run token differs')
    for path, value in extra.items():
        evidence.bind(path, value)
    return plan, reused, extra


def prepare(plan_ref, qualified_ref, output):
    """Return COLLECTION.json ref. Resource binding/dispatch are separate gates."""
    out, plan_path = Path(output), Path(plan_ref['path'])
    require(out.is_absolute() and out.parent.resolve() == out.parent and not out.exists()
            and not out.is_symlink(), 'Fresh canonical wave output required')
    claim_path = plan_path.parent / 'PILOT_TASK_BINDING.json'
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError('Pilot preparation already consumed; inspect its retained global claim')
    evidence = audit_io.Evidence()
    plan, reused, extra = checked_plan(plan_ref, qualified_ref, evidence)
    assignments = partition(plan, reused)
    universe = {r['request_id']: r for r in plan['pilot_requests']}
    remaining = [r for r in plan['pilot_requests'] if r['request_id'] not in reused]
    planner_ref = evidence.bind(Path(__file__).resolve())
    audit_ref = evidence.bind(Path(audit_io.__file__).resolve())
    evidence.unchanged()
    # Exclusive creation is also the old monolithic builder's gate. fsync of
    # payload AND parent precedes all task writes. A truncated claim blocks too.
    claim = {'schema': CLAIM_SCHEMA, 'stage': 'pilot', 'run_token': plan['run_token'], 'plan': plan_ref,
             'qualification': qualified_ref, 'output': str(out), 'source_root': plan['source_root'],
             'qualified_source_files_sha256': hash_value(plan['qualified_source_files']),
             'profile': plan['profile'], 'workers': LANES, 'wave_count': WAVES,
             'request_ids': [r['request_id'] for r in remaining], 'reuse_ids': sorted(reused),
             'request_sha256_by_id': {r['request_id']: hash_value(r) for r in remaining},
             'assignments': assignments, 'assignments_sha256': hash_value(assignments),
             'qualification_contract_sha256': hash_value(plan['qualification_contract']),
             'limits': {'wave_seconds': WAVE_SECONDS, 'overall_generation_seconds': OVERALL_SECONDS,
                        'max_lane_requests': MAX_LANE_REQUESTS, 'batch_size': BATCH_SIZE},
             'request_replay_authorized': False, 'planner': planner_ref,
             'audit_policy_helper': audit_ref, 'audit_io_provenance': evidence.audit_io_provenance()}
    worker.write(claim_path, claim)
    claim_ref = original.ref(claim_path)
    out.mkdir(mode=0o700)
    common = {key: copy.deepcopy(plan[key]) for key in ('run_token', 'model_revision', 'model_snapshot',
        'checkpoint', 'prepared_rows', 'vectors', 'sampling', 'batch_size', 'profile', 'source_root',
        'qualified_source_files')}
    bound = {**plan['bound_files'], **extra, plan_ref['path']: binding(plan_ref),
             claim_ref['path']: binding(claim_ref), planner_ref['path']: binding(planner_ref),
             audit_ref['path']: binding(audit_ref)}
    wave_refs = []
    for assignment in assignments:
        wave = assignment['wave_index']; folder = out / f'wave_{wave:02d}'; folder.mkdir(mode=0o700)
        tasks = []
        for lane_assignment in assignment['lanes']:
            lane = lane_assignment['lane']; requests = [universe[rid] for rid in lane_assignment['request_ids']]
            used = {r['condition_id'] for r in requests}
            task = {**common, 'schema': 'task_specific_causal_generation_v1', 'stage': 'pilot',
                    'deadline_seconds': WAVE_SECONDS, 'requests': requests,
                    'output': str(folder / f'worker_{lane:02d}'), 'bound_files': bound,
                    'conditions': {cid: plan['conditions'][cid] for cid in plan['conditions'] if cid in used},
                    'qualification': qualified_ref['path']}
            task_path = folder / f'task_{lane:02d}.json'; worker.write(task_path, task)
            task_ref = original.ref(task_path)
            tasks.append({'lane': lane, 'task': task_ref, 'requests': len(requests),
                          'global_batch_indices': lane_assignment['global_batch_indices'],
                          'argv': [PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.causal_worker',
                                   '--task', str(task_path), '--sha256', task_ref['sha256']]})
        phase = {'schema': 'task_specific_causal_phase_inputs_v1', 'stage': 'pilot', 'plan': plan_ref,
                 'run_token': plan['run_token'], 'source_root': plan['source_root'], 'profile': plan['profile'],
                 'qualified_source_files': plan['qualified_source_files'], 'conditions': plan['conditions'],
                 'qualification_contract': plan['qualification_contract'], 'tasks': tasks,
                 'wave_index': wave, 'wave_count': WAVES, 'pilot_claim': claim_ref,
                 'global_batch_indices': assignment['global_batch_indices'],
                 'assignments_sha256': claim['assignments_sha256'],
                 'expected_new_requests': assignment['expected_new_requests'], 'expected_reused_requests': 0,
                 'expected_total_requests': assignment['expected_new_requests'], 'reused_requests': {},
                 'bound_files': {**bound, **{t['task']['path']: binding(t['task']) for t in tasks}},
                 'dispatch_authorized_by_this_file': False,
                 'gpu_allocation': 'same eight accepted GPUs; fresh idle and predecessor release required'}
        phase_path = folder / 'PHASE_INPUTS.json'; worker.write(phase_path, phase)
        wave_refs.append({**assignment, 'phase_inputs': original.ref(phase_path)})
    evidence.unchanged()
    collection = {'schema': SCHEMA, 'status': 'prepared_not_dispatched', 'run_token': plan['run_token'],
                  'plan': plan_ref, 'qualification': qualified_ref, 'pilot_claim': claim_ref,
                  'source_root': plan['source_root'], 'qualified_source_files': plan['qualified_source_files'],
                  'profile': plan['profile'], 'wave_count': WAVES, 'lanes': LANES, 'batch_size': BATCH_SIZE,
                  'expected_total_requests': TOTAL, 'expected_new_requests': NEW, 'expected_reused_requests': REUSED,
                  'reused_requests': reused, 'waves': wave_refs, 'assignments_sha256': claim['assignments_sha256'],
                  'limits': claim['limits'], 'planner': planner_ref, 'audit_policy_helper': audit_ref,
                  'audit_io_provenance': claim['audit_io_provenance'],
                  'bound_files': {**bound, **{w['phase_inputs']['path']: binding(w['phase_inputs']) for w in wave_refs}},
                  'generation_performed': False, 'evaluation_performed': False, 'training': False,
                  'dispatch_authorized_by_this_file': False, 'all_wave_assignments_frozen': True,
                  'per_wave_positive_release_required': True, 'overall_deadline_bound_by_first_wave_binder': True,
                  'request_replay_authorized': False}
    path = out / 'COLLECTION.json'; worker.write(path, collection)
    return original.ref(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'qualification'):
        parser.add_argument('--' + name, required=True); parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357 and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Preparation requires accepted owner CPU-only environment')
    require(Path(args.output) == Path(ROOT) / 'pilot-waves-v1', 'Exact fresh pilot-waves-v1 output required')
    references = []
    for name in ('plan', 'qualification'):
        path = Path(getattr(args, name))
        require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT), 'Input outside accepted run')
        reference = original.ref(path)
        require(reference['sha256'] == getattr(args, name + '_sha256'), 'Explicit evidence SHA differs')
        references.append(reference)
    plan = original.read_json(args.plan)
    require(plan['run_token'] == Path(ROOT).name and plan['source_root'] == str(Path(ROOT) / 'source-causal-v2'),
            'Qualified original source/run scope differs')
    print(canonical(prepare(*references, args.output)).decode())


if __name__ == '__main__':
    main()
