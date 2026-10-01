"""Prepare one bounded causal-wave resource manifest; never dispatch workers.

All science and requests come from the frozen collection. A single immutable
24-hour generation window starts when wave zero is bound. Every later wave
requires exact closed journals and positive supervisor release for ALL previous
waves. Failure or an ambiguous binding is retained, never automatically retried.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from . import bind_causal_phase as original
from . import build_causal_eval_plan as closed
from . import causal_qualification as qualification
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from . import fit_worker as io
from . import prepare_causal_waves as planner
from . import resource_guard as guard
from .dispatch_phase import ROOT, PYTHON
from .io_contract import canonical, hash_value

SCHEMA = 'task_specific_causal_wave_collection_v1'
WINDOW_SCHEMA = 'task_specific_global_generation_window_v1'
WAVES, WAVE_SECONDS, GLOBAL_SECONDS = 8, 14400, 86400
require = qualification.require


def read(ref, evidence):
    return closed.read_reference(ref, evidence)


def reference(path):
    return {'path': str(path), **io.binding(path)}


def collection_inputs(ref, evidence):
    collection = read(ref, evidence)
    require(collection['schema'] == SCHEMA and collection['status'] == 'prepared_not_dispatched'
            and collection['limits'] == {'wave_seconds': WAVE_SECONDS, 'overall_generation_seconds': GLOBAL_SECONDS,
                                        'max_lane_requests': 240, 'batch_size': 8}
            and len(collection['waves']) == WAVES,
            'Unreviewed wave collection or generation budget')
    plan = read(collection['plan'], evidence)
    require(plan['schema'] == closed.plan_io.SCHEMA and plan['status'] == 'prepared_not_dispatched'
            and plan['model_revision'] == worker.REVISION and plan['sampling'] == worker.SAMPLING
            and plan['batch_size'] == 8
            and collection['run_token'] == plan['run_token'] and collection['source_root'] == plan['source_root']
            and collection['qualified_source_files'] == plan['qualified_source_files']
            and collection['profile'] == plan['profile'] == worker.PROFILE, 'Collection changes frozen source/profile')
    audit_io.configure_from_plan(evidence, plan)
    evidence.bind(Path(audit_io.__file__).resolve(), closed.binding(collection['audit_policy_helper']))
    require(collection['bound_files'].get(collection['audit_policy_helper']['path'])
            == closed.binding(collection['audit_policy_helper']), 'Collection omits audit-policy helper binding')
    for path, value in plan['bound_files'].items():
        evidence.bind(path, value)
    for path, value in collection['bound_files'].items():
        evidence.bind(path, value)
    reused, bindings = closed.plan_io.reuse_from_qualification(plan, collection['qualification'])
    for path, value in bindings.items():
        evidence.bind(path, value)
    require(collection['reused_requests'] == reused, 'Qualified reuse differs')
    expected = {r['request_id']: r for r in plan['pilot_requests'] if r['request_id'] not in reused}
    ids, indices = [], []
    for index, wave in enumerate(collection['waves']):
        require(wave['wave_index'] == index and wave['expected_new_requests'] == len(wave['request_ids'])
                and 64 <= len(wave['request_ids']) <= 1920 and len(wave['request_ids']) % 8 == 0,
                'Unbounded or reordered wave request grid')
        ids.extend(wave['request_ids']); indices.extend(wave['global_batch_indices'])
    require(len(ids) == len(set(ids)) and set(ids) == set(expected)
            and len(indices) == len(set(indices)) == len(ids) // 8
            and set(indices) == set(range(len(ids) // 8)), 'Wave union loses/repeats frozen requests or batches')
    require(collection['expected_new_requests'] == len(expected)
            and collection['expected_reused_requests'] == len(reused)
            and collection['expected_total_requests'] == len(plan['pilot_requests']) <= 15168
            and len(expected) + len(reused) == len(plan['pilot_requests']), 'Collection counts differ')
    require((len(plan['pilot_requests']), len(expected), len(reused)) == (15168, 15000, 168),
            'This collection requires the full frozen pilot panel')
    claim = read(collection['pilot_claim'], evidence)
    require(claim['plan'] == collection['plan'] and claim['stage'] == 'pilot'
            and claim['audit_policy_helper'] == collection['audit_policy_helper']
            and set(claim['request_ids']) == set(expected)
            and claim['request_sha256_by_id'] == {rid: hash_value(r) for rid, r in expected.items()}
            and claim['assignments'] == [{k: v for k, v in w.items() if k != 'phase_inputs'} for w in collection['waves']]
            and claim['assignments_sha256'] == collection['assignments_sha256'] == hash_value(claim['assignments']),
            'Shared pilot claim differs')
    evidence.bind(Path(planner.__file__).resolve(), closed.binding(collection['planner']))
    require(claim['assignments'] == planner.partition(plan, reused), 'Frozen deterministic wave/lane partition differs')
    return collection, plan


def wave_inputs(collection, wave, evidence):
    plan = read(collection['plan'], evidence)
    audit_io.configure_from_plan(evidence, plan)
    phase = read(wave['phase_inputs'], evidence)
    require(phase['schema'] == 'task_specific_causal_phase_inputs_v1' and phase['stage'] == 'pilot'
            and phase['plan'] == collection['plan'] and phase['run_token'] == collection['run_token']
            and phase['wave_index'] == wave['wave_index'] and phase['wave_count'] == WAVES
            and phase['source_root'] == collection['source_root']
            and phase['qualified_source_files'] == collection['qualified_source_files']
            and phase['profile'] == collection['profile']
            and phase['pilot_claim'] == collection['pilot_claim']
            and phase['global_batch_indices'] == wave['global_batch_indices']
            and phase['expected_new_requests'] == phase['expected_total_requests'] == wave['expected_new_requests']
            and phase['expected_reused_requests'] == 0 and phase['reused_requests'] == {}, 'Wave phase identity differs')
    for path, value in phase['bound_files'].items():
        evidence.bind(path, value)
    require(len(phase['tasks']) == 8, 'Each wave requires eight independent GPU workers')
    allowed = {r['request_id']: r for r in plan['pilot_requests']}
    remaining = [r for r in plan['pilot_requests'] if r['request_id'] not in collection['reused_requests']]
    frozen_batches = [remaining[i:i + 8] for i in range(0, len(remaining), 8)]
    observed, tasks = [], {}
    for lane, item in enumerate(phase['tasks']):
        ref = item['task']
        require(item['lane'] == lane and item['argv'] == [PYTHON, '-B', '-m',
                'infra.gpu03.task_specific_discovery.causal_worker', '--task', ref['path'], '--sha256', ref['sha256']]
                and phase['bound_files'].get(ref['path']) == closed.binding(ref), 'Wave task command/hash differs')
        task = read(ref, evidence)
        for key in ('run_token', 'model_revision', 'model_snapshot', 'checkpoint', 'prepared_rows', 'vectors',
                    'sampling', 'batch_size', 'profile', 'source_root', 'qualified_source_files'):
            require(task[key] == plan[key], 'Wave changes input: ' + key)
        requests = task['requests']; groups = qualification.batches(requests)
        lane_assignment = wave['lanes'][lane]
        require(lane_assignment['lane'] == lane and item['global_batch_indices'] == lane_assignment['global_batch_indices']
                and all(type(i) is int and 0 <= i < len(frozen_batches) for i in item['global_batch_indices'])
                and requests == [r for i in item['global_batch_indices'] for r in frozen_batches[i]]
                and lane_assignment['request_ids'] == [r['request_id'] for r in requests],
                'Wave changes a frozen batch or lane assignment')
        require(task['schema'] == 'task_specific_causal_generation_v1' and task['stage'] == 'pilot'
                and task['qualification'] == collection['qualification']['path']
                and task['deadline_seconds'] == WAVE_SECONDS and 1 <= len(groups) <= 30
                and item['requests'] == len(requests) and all(len(g) == 8 for g in groups),
                'Wave task exceeds reviewed batches or changes worker settings')
        require(task['conditions'] == {cid: plan['conditions'][cid] for cid in {r['condition_id'] for r in requests}}
                and all(r['request_id'] in allowed and r == allowed[r['request_id']] for r in requests),
                'Wave changes conditions or requests')
        for path, value in task['bound_files'].items():
            evidence.bind(path, value)
        require(Path(task['output']).parent == Path(wave['phase_inputs']['path']).parent
                and Path(task['output']).name == f'worker_{lane:02d}', 'Unexpected wave worker output')
        observed.extend(r['request_id'] for r in requests); tasks[ref['path']] = (item, task)
    require(len(observed) == len(set(observed)) and set(observed) == set(wave['request_ids']),
            'Wave task request partition differs')
    return phase, plan, tasks


def validate_window(window, collection_ref, collection):
    require(window['schema'] == WINDOW_SCHEMA and window['collection'] == collection_ref
            and window['run_token'] == collection['run_token'] and window['overall_generation_seconds'] == GLOBAL_SECONDS
            and type(window['started_at_epoch']) in (int, float) and math.isfinite(window['started_at_epoch'])
            and window['global_generation_cutoff_epoch'] == window['started_at_epoch'] + GLOBAL_SECONDS,
            'Global generation window was changed or extended')


def audit_wave(manifest_ref, collection, wave, evidence):
    """Read and verify one actual completed wave; never synthesize aggregate success."""
    phase, plan, tasks = wave_inputs(collection, wave, evidence)
    m = read(manifest_ref, evidence)
    guard.validate_manifest(m)
    phase_name = f'causal-pilot-wave-{wave["wave_index"]:02d}'
    require(m['collection'] and read(m['collection'], evidence) == collection
            and m['wave_index'] == wave['wave_index'] and m['wave_count'] == WAVES
            and m['phase_inputs'] == wave['phase_inputs'] and m['science']['stage'] == 'pilot'
            and m['run_token'] == collection['run_token'] and m['source_root'] == collection['source_root']
            and m['unit'] == collection['run_token'] + '-' + phase_name
            and m['runtime'] == str(Path(ROOT)) + '-' + phase_name + '-runtime'
            and set(m['gpu_uuids']) == {str(i) for i in range(8)},
            'Wrong predecessor wave manifest')
    window = read(m['global_generation_window'], evidence)
    validate_window(window, m['collection'], collection)
    require(window['started_at_epoch'] <= m['bound_at_epoch'] < window['global_generation_cutoff_epoch']
            and m['deadline_epoch'] == min(m['bound_at_epoch'] + WAVE_SECONDS, window['global_generation_cutoff_epoch'])
            and m['runtime_seconds'] == math.ceil(m['deadline_epoch'] - m['bound_at_epoch']) + 120
            and m['kernel_limits'] == {'memory_max': 192 * 1024**3, 'tasks_max': 512, 'cpu_cores': 16},
            'Predecessor deadline/resource policy differs')
    require(m['bound_files'].get(m['global_generation_window']['path']) == closed.binding(m['global_generation_window']),
            'Predecessor omitted immutable generation window')
    for path, value in m['bound_files'].items():
        evidence.bind(path, value)
    release, launches = closed.positive_release(m, manifest_ref['sha256'], evidence)
    require(len(m['workers']) == len(tasks) == 8, 'Predecessor worker coverage differs')
    prepared = evidence.jsonl(Path(plan['prepared_rows']), row_limit=96)
    rows = {r['record_id']: r for r in prepared}
    retained, used, reports = {}, set(), []
    for item in m['workers']:
        argv = item['argv']; require(len(argv) == 8 and argv[5] in tasks and argv[5] not in used,
                                     'Predecessor task command missing or repeated')
        entry, task = tasks[argv[5]]; used.add(argv[5]); task_ref = entry['task']
        require(argv == entry['argv'] and item['cwd'] == collection['source_root']
                and item['env']['PYTHONPATH'] == collection['source_root'], 'Predecessor worker source/command differs')
        requests = task['requests']; out = Path(task['output']); uuid = m['gpu_uuids'][str(item['gpu_id'])]
        started = evidence.json(out / 'STARTED.json', canonical_required=True)
        require(started == {'task_sha256': task_ref['sha256'], 'pid': launches[item['name']]['pid'],
            'requests': len(requests), 'gpu_uuid': uuid, 'stage': 'pilot'}
            and item['env']['CUDA_VISIBLE_DEVICES'] == uuid
            and launches[item['name']]['gpu_id'] == item['gpu_id'], 'Predecessor process/GPU identity differs')
        done = evidence.json(out / 'COMPLETE.json', canonical_required=True)
        require(done['status'] == 'succeeded' and done['stage'] == 'pilot' and done['task_sha256'] == task_ref['sha256']
                and done['requests'] == len(requests) and done['all_returned_requests_retained'] is True
                and done['evaluation_performed'] is False and done['causal_success_claimed'] is False
                and done['cuda_release'] == {'allocated_bytes': 32 << 20, 'reserved_bytes': 32 << 20},
                'Predecessor not positively complete')
        load = done['model_load_report']
        require(load['with_adapter'] is True and load['active_adapters']
                and 0 < load['nonzero_lora_parameter_tensors'] <= load['lora_parameter_tensors']
                and load['lora_parameter_elements'] > 0, 'Predecessor active adapter unproven')
        require(set(done['artifacts']) == {'generations.jsonl'} and {p.name for p in out.iterdir()} ==
                {'STARTED.json', 'COMPLETE.json', 'generations.jsonl', 'journal'}, 'Predecessor has unexpected/partial output')
        evidence.bind(out / 'generations.jsonl', done['artifacts']['generations.jsonl'])
        positions = {r['request_id']: (i, j) for i, batch in enumerate(qualification.batches(requests))
                     for j, r in enumerate(batch)}
        with closed.closed_journal(out / 'journal', task, task_ref['sha256'], evidence) as journal:
            count = 0
            for export, location in closed.stream_rows(out / 'generations.jsonl', evidence, max_rows=240):
                require(count < len(requests), 'Extra predecessor export')
                request = requests[count]; rid = request['request_id']; count += 1
                require(export['request_id'] == rid and rid not in retained, 'Duplicate/reordered predecessor export')
                primitive, _ = journal._payload(rid, 'primitive', journal._events['primitive_returned'][rid]['file_sha256'])
                final, _ = journal._payload(rid, 'final', journal._events['final_recorded'][rid]['file_sha256'])
                require(set(final) == {'returned_row', 'batch_index', 'batch_row', 'first_logits', 'batch_wall_seconds'}
                        and final['returned_row'] is True and (final['batch_index'], final['batch_row']) == positions[rid]
                        and final['first_logits'] is None and type(final['batch_wall_seconds']) in (int, float)
                        and math.isfinite(final['batch_wall_seconds']) and final['batch_wall_seconds'] >= 0,
                        'Predecessor final batch identity differs')
                closed.validate_export(export, request, rows[request['record_id']], primitive, task_ref['sha256'],
                                       task['conditions'][request['condition_id']], 8)
                retained[rid] = {**location, 'origin': 'pilot', 'request_sha256': hash_value(request),
                    'primitive_file': evidence.bind(journal._path(rid, 'primitive')),
                    'final_file': evidence.bind(journal._path(rid, 'final'))}
            require(count == len(requests), 'Missing predecessor exports')
        reports.append({'worker': item['name'], 'task': task_ref, 'complete': evidence.bind(out / 'COMPLETE.json'),
                        'requests': len(requests)})
    require(set(retained) == set(wave['request_ids']), 'Predecessor wave request coverage differs')
    evidence.unchanged()
    return {'wave_index': wave['wave_index'], 'manifest': manifest_ref, 'supervisor_release': release,
            'workers': reports, 'requests': len(retained), 'all_returned_requests_retained': True,
            'gpu_workers_released': True, 'gpu_uuids': m['gpu_uuids'],
            'global_generation_window': m['global_generation_window']}, retained


def make_manifest(collection_ref, collection, wave, phase, devices, now, window_ref, window, bound, predecessors):
    root = Path(ROOT)
    validate_window(window, collection_ref, collection)
    require(len(devices) == 8 and set(devices) == {str(i) for i in range(8)}, 'Eight freshly idle GPUs required')
    require(type(now) in (int, float) and math.isfinite(now)
            and window['started_at_epoch'] <= now < window['global_generation_cutoff_epoch'], 'Global generation cutoff reached')
    phase_name = f'causal-pilot-wave-{wave["wave_index"]:02d}'
    deadline = min(now + WAVE_SECONDS, window['global_generation_cutoff_epoch'])
    stage = collection['source_root']
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONPATH': stage,
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    workers = [{'name': f'gpu_{gpu}', 'gpu_id': int(gpu), 'cpu_ids': [16 + 2 * lane, 17 + 2 * lane],
                'argv': item['argv'], 'cwd': stage, 'env': dict(env, CUDA_VISIBLE_DEVICES=devices[gpu])}
               for lane, (gpu, item) in enumerate(zip(sorted(devices, key=int), phase['tasks']))]
    result = dict(schema=guard.SCHEMA, host='gpu-03', uid=3357, run_token=collection['run_token'],
        source_root=stage, disk_path=str(root), gpu_uuids=devices, workers=workers,
        qualification_contract=phase['qualification_contract'], phase_inputs=wave['phase_inputs'],
        expected_new_requests=phase['expected_new_requests'], expected_reused_requests=0,
        expected_total_requests=phase['expected_total_requests'],
        science={'generation': True, 'evaluation': False, 'training': False, 'stage': 'pilot',
                 'batch_size': 8, 'gpu_allocator_fraction': .65, 'prompt_max': 1536, 'completion_max': 1536,
                 'checkpoint': 'original Ada M60', 'profile': phase['profile']},
        bound_files={**{stage + '/' + name: value for name, value in collection['qualified_source_files'].items()},
                     **phase['bound_files'], **bound},
        collection=collection_ref, wave_index=wave['wave_index'], wave_count=WAVES,
        bound_at_epoch=now, global_generation_window=window_ref, predecessor_manifests=predecessors,
        runtime=str(root) + '-' + phase_name + '-runtime', unit=root.name + '-' + phase_name,
        entry_receipt=str(root / (phase_name + '-entry.json')), deadline_epoch=deadline,
        runtime_seconds=math.ceil(deadline - now) + 120,
        kernel_limits={'memory_max': 192 * 1024**3, 'tasks_max': 512, 'cpu_cores': 16})
    result['wave_policy'] = {'maximum_batches_per_gpu': 30, 'batch_size': 8,
        'global_generation_seconds': GLOBAL_SECONDS, 'maximum_work_seconds': WAVE_SECONDS,
        'mandatory_guard_idle_qualification_seconds': 60, 'swap_max': 0,
        'request_order_or_scientific_settings_changed': False}
    guard.validate_manifest(result)
    require(len(canonical(result)) + 1 <= 1 << 20, 'Wave resource manifest exceeds guard byte cap')
    return result


def bind(collection_ref, wave_index, *, inventory=None, processes=None, observed_at=None, now=None, root=ROOT, probe=None):
    """Preparation only; supplied inventory must have been freshly queried."""
    require(type(wave_index) is int and 0 <= wave_index < WAVES, 'Unknown wave')
    require(Path(root) == Path(ROOT), 'Wrong campaign root')
    evidence = audit_io.Evidence()
    binder_ref = evidence.bind(Path(__file__).resolve())
    collection, _ = collection_inputs(collection_ref, evidence)
    require(collection['run_token'] == Path(ROOT).name and Path(collection_ref['path']).parent.parent == Path(ROOT)
            and collection['source_root'] == str(Path(ROOT) / 'source-causal-v2'),
            'Collection outside current campaign')
    qualified = read(collection['qualification'], evidence)
    qualified_manifest = read(qualified['manifest'], evidence)
    require(qualified_manifest['schema'] == guard.SCHEMA and qualified_manifest['host'] == 'gpu-03'
            and qualified_manifest['uid'] == 3357 and qualified_manifest['run_token'] == collection['run_token']
            and qualified_manifest['source_root'] == collection['source_root']
            and qualified_manifest['science']['stage'] == 'qualification'
            and set(qualified_manifest['gpu_uuids']) == {str(i) for i in range(8)},
            'Original eight-GPU qualification hardware proof missing')
    wave = collection['waves'][wave_index]
    folder = Path(wave['phase_inputs']['path']).parent
    claim_path, manifest_path = folder / 'RESOURCE_BINDING.json', folder / 'RESOURCE_MANIFEST.json'
    require(not claim_path.exists() and not claim_path.is_symlink() and not manifest_path.exists(),
            'Wave binding already consumed; inspect without retrying')
    phase, _, tasks = wave_inputs(collection, wave, evidence)
    require(all(not Path(task['output']).exists() for _, task in tasks.values()), 'Wave has existing/ambiguous worker output')
    predecessors, proofs = [], []
    window_path = Path(collection_ref['path']).parent / 'GLOBAL_GENERATION_WINDOW.json'
    if wave_index == 0:
        require(not window_path.exists() and not window_path.is_symlink(), 'Global generation window already consumed')
    else:
        window_ref = evidence.bind(window_path); window = read(window_ref, evidence)
        validate_window(window, collection_ref, collection)
        for previous in collection['waves'][:wave_index]:
            previous_manifest = Path(previous['phase_inputs']['path']).parent / 'RESOURCE_MANIFEST.json'
            ref = evidence.bind(previous_manifest)
            proof, _ = audit_wave(ref, collection, previous, evidence)
            require(proof['global_generation_window'] == window_ref, 'Predecessor used a different global cutoff')
            predecessors.append(ref); proofs.append(proof)
    # Hashing predecessor evidence may take time. Live CLI probes only here,
    # after that work, so the allocation snapshot cannot become stale during it.
    if probe is not None:
        require(inventory is None and processes is None and observed_at is None, 'Ambiguous inventory source')
        inventory, processes, observed_at = probe()
    now = time.time() if now is None else now
    require(type(observed_at) in (int, float) and math.isfinite(observed_at)
            and 0 <= now - observed_at <= 30, 'Stale GPU availability observation')
    devices = original.idle_devices(inventory, processes)
    require(len(devices) == 8, 'A GPU is occupied, foreign, missing or unsafe')
    require(devices == qualified_manifest['gpu_uuids'], 'Physical GPUs differ from original qualification')
    require(all(p['gpu_uuids'] == devices for p in proofs), 'Physical GPUs differ from earlier waves')
    if wave_index == 0:
        window = {'schema': WINDOW_SCHEMA, 'collection': collection_ref, 'run_token': collection['run_token'],
                  'started_at_epoch': now, 'overall_generation_seconds': GLOBAL_SECONDS,
                  'global_generation_cutoff_epoch': now + GLOBAL_SECONDS}
    require(window['started_at_epoch'] <= now < window['global_generation_cutoff_epoch'],
            'Global generation cutoff reached or clock predates its claim')
    require(window['global_generation_cutoff_epoch'] - now > 60,
            'Global cutoff leaves no time for mandatory 60-second idle qualification')
    evidence.unchanged()
    # All read-only admission checks precede exclusive claims. Failed writes or
    # interrupted preparations stay consumed and need explicit forensic review.
    if wave_index == 0:
        worker.write(window_path, window)
        window_ref = reference(window_path)
    audit_path = folder / 'PREDECESSOR_AUDIT.json'
    worker.write(claim_path, {'schema': 'task_specific_wave_binding_claim_v1', 'collection': collection_ref,
        'phase_inputs': wave['phase_inputs'], 'wave_index': wave_index, 'bound_at_epoch': now,
        'global_generation_window': window_ref, 'gpu_observed_at_epoch': observed_at, 'gpu_uuids': devices,
        'inventory_sha256': hash_value(inventory), 'processes_sha256': hash_value(processes),
        'binder': binder_ref, 'automatic_retry_authorized': False})
    worker.write(audit_path, {'schema': 'task_specific_predecessor_wave_audit_v1', 'predecessors': proofs,
        'wave_index': wave_index, 'verified_files': evidence.files,
        'audit_io_provenance': evidence.audit_io_provenance(),
        'scientific_outcomes_examined': False, 'request_replay_authorized': False})
    # The full independent hash inventory lives in the closed audit receipt.
    # Do not expand tens of thousands of primitive paths into the 1-MiB guard
    # manifest; dispatch binds this receipt and all actual next-worker inputs.
    bound = {ref['path']: closed.binding(ref) for ref in predecessors}
    for ref in (collection_ref, wave['phase_inputs'], window_ref, binder_ref, reference(claim_path), reference(audit_path)):
        bound[ref['path']] = closed.binding(ref)
    result = make_manifest(collection_ref, collection, wave, phase, devices, now, window_ref, window, bound, predecessors)
    worker.write(manifest_path, result)
    return reference(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--wave-index', required=True, type=int)
    args = parser.parse_args()
    require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357 and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Binding requires accepted host owner with no CUDA device')
    ref = reference(Path(args.collection)); require(ref['sha256'] == args.sha256, 'Collection SHA differs')
    def probe():
        at = time.time()
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        return inventory, processes, at
    ref = bind(ref, args.wave_index, probe=probe)
    print(canonical({'manifest': ref, 'dispatched': False}).decode())


if __name__ == '__main__':
    main()
