"""Freeze eight approved fresh-seed successor waves; never dispatch workers.

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
from . import successor_plan as successor
from . import causal_qualification as qualification
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import causal_prompts as prompts
from . import fit_worker as io
from . import io_contract
from .dispatch_phase import PYTHON
from .successor_plan import ROOT
from .io_contract import canonical, hash_value

SCHEMA = 'task_specific_successor_wave_collection_v1'
CLAIM_SCHEMA = 'task_specific_successor_wave_claim_v1'
WAVES = LANES = BATCH_SIZE = 8
TOTAL, NEW, REUSED = 15168, 15168, 0
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
    plan, reused, extra = successor.checked_plan(plan_ref, qualified_ref, evidence)
    inventory = read_ref(plan['successor']['operational_source_inventory'], evidence)
    name = 'infra/gpu03/task_specific_discovery/prepare_successor_waves.py'
    evidence.bind(Path(__file__).resolve(), inventory[name])
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
             'successor': plan['successor'],
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
    operational_inventory = read_ref(plan['successor']['operational_source_inventory'], evidence)
    operational_root = Path(plan['successor']['operational_source_root'])
    operational_bindings = {str(operational_root / name): value for name, value in operational_inventory.items()}
    bound = {**plan['bound_files'], **extra, **operational_bindings, plan_ref['path']: binding(plan_ref),
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
                 'successor': plan['successor'],
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
                  'successor': plan['successor'], 'profile': plan['profile'], 'wave_count': WAVES, 'lanes': LANES, 'batch_size': BATCH_SIZE,
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
        require(path.is_absolute() and path.resolve() == path and
                (path.is_relative_to(ROOT) if name == 'plan' else
                 path == Path(successor.ORIGINAL_ROOT) / 'CAUSAL_QUALIFICATION_PROOF_V2.json'),
                'Input outside accepted successor/original proof scope')
        reference = original.ref(path)
        require(reference['sha256'] == getattr(args, name + '_sha256'), 'Explicit evidence SHA differs')
        references.append(reference)
    plan = original.read_json(args.plan)
    require(plan['run_token'] == Path(ROOT).name and
            plan['source_root'] == successor.ORIGINAL_ROOT + '/source-causal-v2' and
            plan['successor']['operational_source_root'] == successor.SOURCE_ROOT,
            'Qualified original source/run scope differs')
    print(canonical(prepare(*references, args.output)).decode())


if __name__ == '__main__':
    main()
