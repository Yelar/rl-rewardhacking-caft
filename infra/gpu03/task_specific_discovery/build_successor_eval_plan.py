"""Prepare CPU evaluation for the authorized fresh seed6008 successor.

Audit all eight actual successor waves and all15168 fresh retained requests.
Reuse zero old generations. Preserve the original evaluator/sandbox and exact
historical authored qualification; only new CPU preparation/binding is authored.
This module never dispatches generation, evaluation or a subprocess.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket

from . import build_causal_eval_plan as original
from . import build_causal_wave_eval_plan as previous
from . import build_causal_plan as plan_io
from . import bind_successor_wave as wave_io
from . import causal_qualification as qualification
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import fit_worker as io
from . import successor_plan as successor
from .successor_dispatch import ROOT as DISPATCH_ROOT, PYTHON
from .io_contract import canonical, hash_value

ROOT = str(DISPATCH_ROOT)
SCHEMA = 'task_specific_successor_wave_eval_plan_v1'
INDEX_SCHEMA = 'task_specific_completed_successor_waves_v1'
AUDIT_SCHEMA = 'task_specific_closed_successor_pilot_inputs_v1'
SELF_SOURCE = 'infra/gpu03/task_specific_discovery/build_successor_eval_plan.py'
WAVE_AUDITOR_SOURCE = 'infra/gpu03/task_specific_discovery/bind_successor_wave.py'
WAVE_PREPARER_SOURCE = 'infra/gpu03/task_specific_discovery/prepare_successor_waves.py'
AUDIT_IO_SOURCE = previous.AUDIT_IO_SOURCE
GUARD_SOURCE = 'infra/gpu03/task_specific_discovery/resource_guard.py'
TOTAL, NEW, REUSED, WAVES, LANES, MAX_PRODUCERS = 15168, 15168, 0, 8, 8, 64
require = qualification.require
read_reference, binding = original.read_reference, original.binding

# These are existing qualification evidence, never a new generation population.
HISTORICAL_TOKEN = Path(successor.ORIGINAL_ROOT).name
AUTHORED_TASK = {'path': successor.ORIGINAL_ROOT + '/plans/AUTHORED_EVALUATOR_TASK_V2.json',
    'sha256': '54308462ab34944dad3f8862dbc6a556f3182f62c827a1934c1eeffd5b159662', 'size_bytes': 47838}
AUTHORED_PROOF = {'path': successor.ORIGINAL_ROOT + '/authored-evaluator-v2/COMPLETE.json',
    'sha256': '62520fd2d6032fc061e0bb03d1f041cc41ea1e7ae90220de19f742d9c58c0143', 'size_bytes': 43203}
AUTHORED_SUPERVISOR = {'path': successor.ORIGINAL_ROOT + '/plans/AUTHORED_EVALUATOR_V2.json',
    'sha256': 'c85c327c9a42e0eec658a1e229260bfacb13b31af71d6b284868a81f590d67a1', 'size_bytes': 76682}
PRIOR_EVAL_INVENTORY = {'path': successor.ORIGINAL_ROOT + '/source-evalbind-v1/SOURCE_FILES.json',
    'sha256': 'f9c0b14374da6ee5beb8a619d55e8c60eeb0b380175b31d2cc7d3bdc8280669b', 'size_bytes': 47742}


def source_and_authored(plan, source_root, task_ref, proof_ref, supervisor_ref, evidence):
    """Re-audit original safety fixtures and admit only exact qualified source.

    All prior281 evaluation files must remain identical except the guard, which
    must equal the explicitly approved successor-v1 operational binding. Every
    successor-v1 source file, including that guard and its tests, must remain
    identical too. Additive future preparation files are individually hash-bound
    by the original full source validator. No scientific dependency exception.
    """
    require((task_ref, proof_ref, supervisor_ref) == (AUTHORED_TASK, AUTHORED_PROOF, AUTHORED_SUPERVISOR),
            'Only the exact historical authored qualification may be reused')
    files, task, release = original.source_and_authored(source_root, task_ref, proof_ref, supervisor_ref, evidence)
    require(task['run_token'] == HISTORICAL_TOKEN and plan['run_token'] == Path(ROOT).name,
            'Historical qualification or successor run token differs')
    prior = read_reference(PRIOR_EVAL_INVENTORY, evidence)
    operational_ref = plan['successor']['operational_source_inventory']
    require(Path(operational_ref['path']) == Path(ROOT) / 'source-successor-v1/SOURCE_FILES.json',
            'Operational source must be the approved immutable successor-v1 inventory')
    operational = read_reference(operational_ref, evidence)
    require(GUARD_SOURCE in prior and GUARD_SOURCE in operational,
            'Qualified guard source provenance missing')
    for name, value in prior.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                'Historical source inventory path escapes package')
        expected = operational[GUARD_SOURCE] if name == GUARD_SOURCE else value
        require(files.get(name) == expected, 'Previously qualified evaluation source changed: ' + name)
    for name, value in operational.items():
        require(files.get(name) == value, 'Approved successor-v1 source changed: ' + name)
    evidence.unchanged()
    return files, task, release


def _plan_inputs(collection_ref, evidence):
    require(Path(collection_ref['path']).is_relative_to(ROOT), 'Collection outside approved successor root')
    collection, plan = wave_io.collection_inputs(collection_ref, evidence)
    require(collection['run_token'] == plan['run_token'] == Path(ROOT).name,
            'Successor collection token differs')
    prepared = evidence.jsonl(Path(plan['prepared_rows']), row_limit=96)
    rows = {r['record_id']: r for r in prepared}
    coordinates = {(canonical(r['problem_id']).decode(), r['protocol']) for r in prepared}
    problems = {pid for pid, _ in coordinates}
    require(len(rows) == len(prepared) == 96 and len(problems) == 32 and len(coordinates) == 96
            and all(r['problem_split'] == 'direction_fit' for r in prepared)
            and coordinates == {(pid, protocol) for pid in problems for protocol in original.PROTOCOLS},
            'Pilot requires exactly the frozen32 fitting problems and three protocols')
    universe = {r['request_id']: r for r in plan['pilot_requests']}
    require(len(plan['pilot_requests']) == len(universe) == TOTAL
            and collection['expected_new_requests'] == NEW and collection['expected_reused_requests'] == 0
            and collection['reused_requests'] == {}, 'Successor pilot must contain15168 fresh requests and zero reuse')
    return collection, plan, rows, universe, {}


def audit_waves(collection_ref, manifest_refs, evidence):
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
            'Eight terminal waves do not cover all15168 fresh requests')
    require(len(retained) == TOTAL and set(retained) == set(universe), 'Full pilot coverage is not exactly once')
    require(len({r['generation_file']['path'] for r in retained.values()}) <= MAX_PRODUCERS,
            'More producers than eight successor waves')
    evidence.unchanged()
    audit = {'schema': AUDIT_SCHEMA, 'status': 'all_pilot_generations_independently_verified',
        'plan': collection['plan'], 'wave_collection': collection_ref, 'qualification': collection['qualification'],
        'wave_manifests': manifest_refs, 'actual_wave_closures': actual_proofs,
        'gpu_uuids': gpu_allocation, 'global_generation_window': global_window,
        'requests': TOTAL, 'new_requests': NEW, 'reused_requests': REUSED, 'retained_requests': retained,
        'all_returned_requests_retained': True, 'gpu_workers_released': True,
        'generation_or_evaluation_performed': False, 'behavioral_success_claimed': False,
        'synthetic_combined_supervisor': False, 'successor': plan['successor'],
        'historical_generation_reused': False}
    return plan, rows, retained, audit


def assemble(plan, prepared, retained, output, source_root, source_files):
    """Reuse the exact qualified lane input format, with only64 new producers."""
    require(len(retained) == TOTAL and all(r['origin'] == 'pilot' for r in retained.values())
            and len({r['generation_file']['path'] for r in retained.values()}) <= MAX_PRODUCERS,
            'Successor evaluation requires only the complete fresh generation population')
    return previous.assemble(plan, prepared, retained, output, source_root, source_files)


def build(collection_ref, manifest_refs, authored_task_ref, authored_proof_ref, authored_supervisor_ref,
          source_root, output, *, index_ref=None):
    output, source_root = Path(output), Path(source_root)
    require(output.parent == Path(ROOT) and source_root.parent == Path(ROOT)
            and source_root.resolve() == source_root, 'Evaluation output/source outside approved successor root')
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
    files, task, authored_release = source_and_authored(plan, source_root, authored_task_ref, authored_proof_ref,
                                                               authored_supervisor_ref, evidence)
    require({SELF_SOURCE, WAVE_AUDITOR_SOURCE, WAVE_PREPARER_SOURCE, AUDIT_IO_SOURCE} <= set(files),
            'Wave preparation/audit source missing')
    require(files[AUDIT_IO_SOURCE] == binding(evidence.bind(Path(audit_io.__file__).resolve())),
            'Consumed input-audit helper differs from evaluator package')
    audit.update(independently_hashed_files=dict(evidence.files), auditor_sha256=io.sha256(__file__),
                 input_audit_policy=evidence.audit_io_provenance(),
                 authored_qualification=authored_proof_ref, authored_supervisor_release=authored_release,
                 prior_evaluation_source_inventory=PRIOR_EVAL_INVENTORY,
                 historical_qualification_reused_without_execution=True)
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
            authored_task_ref, authored_proof_ref, authored_supervisor_ref, PRIOR_EVAL_INVENTORY,
            plan['successor']['operational_source_inventory'],
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
        require(path.is_absolute() and path.resolve() == path, 'Noncanonical input evidence')
        if name in ('collection', 'wave_manifests'):
            require(path.is_relative_to(ROOT), 'Successor evidence outside this run')
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
