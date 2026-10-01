"""Bind closed fresh seed6008 successor pilot inputs to a CPU-only evaluator supervisor.

Preparation only: no subprocess, GPU probe, model or generated-code execution.
The existing dispatcher/phase_entry/resource_guard own any later authorized
execution. A single global claim prevents rebinding a partial evaluation.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import socket
import time

from . import build_successor_eval_plan as waves
from . import causal_eval_lane as lane
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import fit_worker as io
from . import resource_guard as guard
from .successor_dispatch import ROOT as DISPATCH_ROOT, PYTHON
ROOT = str(DISPATCH_ROOT)
from .io_contract import canonical, hash_value

TOKEN = Path(ROOT).name
SCHEMA = 'task_specific_successor_cpu_evaluation_resource_binding_v1'
SELF_SOURCE = 'infra/gpu03/task_specific_discovery/bind_successor_eval.py'
SUFFIX = 'causal-pilot-evaluation-v1'
LIMITS = {'lanes': 8, 'evaluator_workers_per_lane': 2, 'records_per_shard': 256,
          'bytes_per_shard': 64 << 20, 'deadline_seconds': 14400}
KERNEL = {'memory_max': 32 << 30, 'tasks_max': 512, 'cpu_cores': 16}
require, binding, read = waves.require, waves.binding, waves.read_reference


def _bound(path, value, declared, evidence):
    require(isinstance(value, dict) and set(value) == {'sha256', 'size_bytes'}
            and declared.get(str(path)) == value, 'Consumed evaluation input omitted or changed: ' + str(path))
    return evidence.bind(path, value)


def _phase(ref, evidence):
    phase = read(ref, evidence)
    require(phase['schema'] == waves.original.SCHEMA and phase['preparation_schema'] == waves.SCHEMA
            and phase['status'] == 'prepared_not_dispatched' and phase['run_token'] == TOKEN,
            'Wrong prepared full-wave evaluation scope')
    root, source, output = Path(ROOT), Path(phase['source_root']), Path(ref['path']).parent
    require(output.parent == root and source.parent == root and output.resolve() == output
            and source.resolve() == source and Path(ref['path']).name == 'PHASE_INPUTS.json',
            'Evaluation inputs/source must be canonical direct children of this run')
    require(phase['requests'] == phase['new_evaluations'] == waves.TOTAL
            and phase['generation_performed'] is False and phase['evaluation_performed'] is False
            and phase['training'] is False and phase['problem_split'] == 'direction_fit'
            and phase['dispatch_authorized_by_this_file'] is False
            and phase['resource_supervisor_binding_required'] is True and phase['limits'] == LIMITS,
            'Evaluation population, request counts or limits changed')
    _bound(phase['input_audit']['path'], binding(phase['input_audit']), phase['bound_files'], evidence)
    previous = read(phase['input_audit'], evidence)
    require(previous['schema'] == waves.AUDIT_SCHEMA
            and previous['status'] == 'all_pilot_generations_independently_verified'
            and previous['requests'] == waves.TOTAL and previous['new_requests'] == waves.NEW
            and previous['reused_requests'] == waves.REUSED
            and previous['all_returned_requests_retained'] is True and previous['gpu_workers_released'] is True
            and previous['generation_or_evaluation_performed'] is False
            and previous['synthetic_combined_supervisor'] is False,
            'Full real-wave audit and positive generation release required')
    claim_path = Path(previous['plan']['path']).parent / 'PILOT_EVAL_TASK_BINDING.json'
    require(claim_path.is_relative_to(root), 'Evaluation claim escaped this run')
    claim_ref = _bound(claim_path, phase['bound_files'].get(str(claim_path)), phase['bound_files'], evidence)
    claim = read(claim_ref, evidence)
    require(claim['schema'] == waves.SCHEMA and claim['plan'] == previous['plan']
            and claim['wave_collection'] == previous['wave_collection']
            and claim['wave_manifests'] == previous['wave_manifests']
            and claim['qualification'] == previous['qualification']
            and claim['output'] == str(output) and claim['source_root'] == str(source)
            and claim['input_audit_sha256'] == hash_value(previous)
            and claim['lanes'] == 8 and claim['evaluation_replay_authorized'] is False,
            'Global evaluation preparation claim differs')
    return phase, previous, claim, claim_ref


def admit(ref, evidence):
    """Independently repeat closed-input admission, never execute the scorer."""
    phase, previous, claim, claim_ref = _phase(ref, evidence)
    plan, prepared, retained, actual = waves.audit_waves(previous['wave_collection'], previous['wave_manifests'], evidence)
    require(all(previous.get(k) == value for k, value in actual.items()),
            'Saved input audit differs from actual closed waves/journals')
    require(claim['request_ids'] == [r['request_id'] for r in plan['pilot_requests']],
            'Evaluation claim request universe changed')
    files, authored_task, authored_release = waves.source_and_authored(
        plan, phase['source_root'], claim['authored_task'], claim['authored_proof'], claim['authored_supervisor'], evidence)
    require(authored_task['run_token'] == waves.HISTORICAL_TOKEN and claim['source_files_sha256'] == hash_value(files)
            and previous['authored_qualification'] == claim['authored_proof']
            and previous['authored_supervisor_release'] == authored_release,
            'Authored qualification or source identity changed')
    required = {SELF_SOURCE, waves.SELF_SOURCE, waves.WAVE_AUDITOR_SOURCE, waves.WAVE_PREPARER_SOURCE,
                waves.AUDIT_IO_SOURCE, 'infra/gpu03/task_specific_discovery/resource_guard.py',
                'infra/gpu03/task_specific_discovery/phase_entry.py',
                'infra/gpu03/task_specific_discovery/successor_dispatch.py',
                'infra/gpu03/task_specific_discovery/successor_phase_entry.py',
                'infra/gpu03/task_specific_discovery/guard_failure_diagnostic.py'}
    require(required <= set(files), 'CPU binding/runtime source omitted')
    project = Path(__file__).resolve().parents[3]
    for name in required:
        evidence.bind(project / name, files[name])
    for path, value in phase['bound_files'].items():
        evidence.bind(path, value)
    for ref_ in (previous['plan'], previous['qualification'], previous['wave_collection'],
                 *previous['wave_manifests'], claim['authored_task'], claim['authored_proof'], claim['authored_supervisor'],
                 waves.PRIOR_EVAL_INVENTORY, plan['successor']['operational_source_inventory']):
        _bound(ref_['path'], binding(ref_), phase['bound_files'], evidence)
    source = Path(phase['source_root'])
    for name, value in files.items():
        _bound(source / name, value, phase['bound_files'], evidence)
    tasks, shards = phase['tasks'], phase['shards']
    require(len(tasks) == 8 and [t['lane'] for t in tasks] == list(range(8))
            and len({s['id'] for s in shards}) == len(shards), 'Eight distinct ordered CPU lanes required')
    universe = {r['request_id']: r for r in plan['pilot_requests']}
    seen, coordinates, workers = set(), set(), []
    for slot, item in enumerate(tasks):
        task_ref = item['task']; _bound(task_ref['path'], binding(task_ref), phase['bound_files'], evidence)
        task = read(task_ref, evidence)
        expected_shards = shards[slot::8]
        require(item['argv'] == [PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.causal_eval_lane',
                                '--task', task_ref['path'], '--sha256', task_ref['sha256']]
                and task_ref['path'] == str(Path(ref['path']).parent / f'task_{slot:02d}.json')
                and task['schema'] == lane.SCHEMA and task['run_token'] == TOKEN
                and task['source_dir'] == str(source) and task['source_files'] == files
                and task['venv_dir'] == lane.PINNED_VENV and task['deadline_seconds'] == 14400
                and task['problem_split'] == 'direction_fit'
                and task['shards'] == [{k: s[k] for k in ('id', 'protocol', 'generated', 'dataset')} for s in expected_shards],
                'Lane command/source/shard assignment changed')
        destination = Path(ref['path']).parent / 'results' / f'lane_{slot:02d}'
        require(task['output'] == str(destination) and destination.parent.is_dir()
                and destination.parent.resolve() == destination.parent and not destination.parent.is_symlink()
                and not destination.exists() and not destination.is_symlink(),
                'Evaluation lane already started or output changed; no replay')
        lane_count = 0
        for shard in expected_shards:
            for kind in ('generated', 'dataset'):
                _bound(shard[kind]['path'], binding(shard[kind]), phase['bound_files'], evidence)
            rows, examples, _ = lane.load_shard(shard, 'direction_fit')
            require([r['request_id'] for r in rows] == shard['request_ids'], 'Shard request list differs from retained bytes')
            example_map = {canonical(r['id']).decode(): r for r in examples}
            for row in rows:
                rid = row['request_id']; require(rid in universe and rid not in seen, 'Duplicate/unplanned evaluation request')
                request = universe[rid]; prepared_row = prepared[request['record_id']]
                coord = (request['condition_id'], hash_value(request['problem_id']), request['protocol'], request['sample_index'])
                require(coord not in coordinates and hash_value(row) == retained[rid]['generation_row_sha256']
                        and all(row.get(k) == v for k, v in request.items())
                        and example_map[canonical(request['problem_id']).decode()] == prepared_row['dataset_row'],
                        'Retained generation, seed, coordinate or trusted dataset changed')
                seen.add(rid); coordinates.add(coord)
            lane_count += len(rows)
        require(item['requests'] == lane_count and lane_count > 0, 'Lane request count differs')
        workers.append(item)
    require(seen == set(universe) and len(seen) == waves.TOTAL, 'Full15168 evaluation coverage missing')
    evidence.unchanged()
    return phase, plan, files, claim_ref, workers


def resource_manifest(ref, phase, files, workers, deadline, now):
    require(type(deadline) in (int, float) and math.isfinite(deadline)
            and 60 < deadline - now <= 14400, 'Explicit future evaluation deadline must be within four hours')
    source = phase['source_root']; root = Path(ROOT)
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONPATH': source, 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = {'schema': guard.SCHEMA, 'host': 'gpu-03', 'uid': 3357, 'run_token': TOKEN,
        'source_root': source, 'runtime': str(guard.RUN_ROOT / (TOKEN + '-' + SUFFIX + '-runtime')),
        'disk_path': str(root), 'deadline_epoch': deadline, 'runtime_seconds': math.ceil(deadline - now) + 120,
        'unit': TOKEN + '-' + SUFFIX, 'entry_receipt': str(root / (SUFFIX + '-entry.json')),
        'kernel_limits': dict(KERNEL), 'gpu_uuids': {},
        'workers': [{'name': f'eval_{i:02d}', 'gpu_id': None, 'cpu_ids': [48 + 2*i, 49 + 2*i],
                     'argv': t['argv'], 'cwd': source, 'env': dict(env)} for i, t in enumerate(workers)],
        'bound_files': {**phase['bound_files'], ref['path']: binding(ref),
                        str(Path(source) / 'SOURCE_FILES.json'): io.binding(Path(source) / 'SOURCE_FILES.json')},
        'phase_inputs': ref, 'input_audit': phase['input_audit'], 'expected_total_requests': waves.TOTAL,
        'science': {'generation': False, 'evaluation': True, 'training': False, 'stage': 'pilot_evaluation',
                    'problem_split': 'direction_fit', 'requests': waves.TOTAL, 'evaluator_workers_per_lane': 2},
        'binding_schema': SCHEMA, 'bound_at_epoch': now}
    require(len(workers) == 8 and files, 'Complete admitted evaluator workers/source required')
    guard.validate_manifest(result)
    return result


def bind(ref, deadline, *, clock=time.time):
    evidence = audit_io.Evidence()
    phase, previous, _, _ = _phase(ref, evidence)
    claim_path = Path(previous['plan']['path']).parent / 'PILOT_EVAL_RESOURCE_BINDING.json'
    output = Path(ref['path']).parent / 'EVALUATION_RESOURCE_MANIFEST.json'
    require(not claim_path.exists() and not claim_path.is_symlink()
            and not output.exists() and not output.is_symlink(), 'CPU evaluation binding already consumed; inspect existing records')
    phase, _, files, prior_claim, workers = admit(ref, evidence)
    manifest = resource_manifest(ref, phase, files, workers, deadline, clock())
    runtime, receipt = Path(manifest['runtime']), Path(manifest['entry_receipt'])
    dispatch = Path(ROOT) / (manifest['unit'] + '_DISPATCH_ATTEMPT.json')
    require(all(not p.exists() and not p.is_symlink() for p in (runtime, receipt, dispatch)),
            'Exact CPU phase already started; no rebind or replay')
    evidence.unchanged()
    worker.write(claim_path, {'schema': SCHEMA, 'run_token': TOKEN, 'phase_inputs': ref,
        'prior_evaluation_claim': prior_claim, 'output': str(output), 'unit': manifest['unit'],
        'deadline_epoch': deadline, 'bound_at_epoch': manifest['bound_at_epoch'],
        'requests': waves.TOTAL, 'kernel_limits': KERNEL, 'gpu_uuids': {},
        'preclaim_manifest_sha256': hash_value(manifest), 'evaluation_replay_authorized': False})
    manifest['bound_files'][str(claim_path)] = io.binding(claim_path)
    require(len(canonical(manifest)) + 1 <= 1024 * 1024, 'Resource manifest exceeds existing supervisor bound')
    worker.write(output, manifest)
    return {'path': str(output), **io.binding(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-inputs', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--deadline-epoch', required=True, type=float); args = parser.parse_args()
    require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357 and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Binding requires accepted gpu-03 owner and CPU-only environment')
    path = Path(args.phase_inputs)
    ref = {'path': str(path), 'sha256': args.sha256, 'size_bytes': path.stat().st_size}
    require(set(range(48, 64)) <= os.sched_getaffinity(0), 'Declared evaluation CPU mask unavailable')
    print(canonical(bind(ref, args.deadline_epoch)).decode())


if __name__ == '__main__':
    main()
