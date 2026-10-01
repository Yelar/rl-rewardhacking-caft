"""Prepare CPU evaluator inputs from positively closed causal generations.

No model, tokenizer, generated-code evaluator, subprocess, CUDA or network call.
This independently reads terminal evidence and immutable journals, preserves every
frozen pilot request exactly once (including qualified controls), and writes a
one-use plan. A failed preparation consumes its claim; it never authorizes replay.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path
import socket

from . import build_causal_plan as plan_io
from . import causal_evaluate as evaluator
from . import causal_qualification as qualification
from . import causal_worker as worker
from . import fit_worker as io
from . import qualify_causal_evaluator as authored
from .dispatch_phase import ROOT, PYTHON
from .io_contract import RequestJournal, canonical, hash_value

SCHEMA = 'task_specific_causal_eval_plan_v1'
LANE_SCHEMA = 'task_specific_causal_eval_lane_v1'
LANE_SOURCE = 'infra/gpu03/task_specific_discovery/causal_eval_lane.py'
SELF_SOURCE = 'infra/gpu03/task_specific_discovery/build_causal_eval_plan.py'
MAX_REQUESTS = 15168
PROTOCOLS = evaluator.PROTOCOLS
require = qualification.require


def binding(reference):
    return {k: reference[k] for k in ('sha256', 'size_bytes')}


def read_reference(reference, evidence, limit=64 << 20):
    return evidence.json(Path(reference['path']), limit, binding(reference))


def stream_rows(path, evidence, max_rows=2048):
    """Full-file hash plus bounded row reads; keep only one raw row in memory."""
    reference = evidence.bind(path)
    require(reference['size_bytes'] <= max_rows * evaluator.MAX_LINE_BYTES, 'Oversized generation stream')
    with Path(path).open('rb') as stream:
        for number in range(max_rows + 1):
            offset = stream.tell()
            raw = stream.readline(evaluator.MAX_LINE_BYTES + 1)
            if not raw:
                return
            require(number < max_rows and len(raw) <= evaluator.MAX_LINE_BYTES and raw.endswith(b'\n')
                    and raw.strip(), 'Truncated/oversized generation stream')
            value = qualification.decode(raw)
            require(isinstance(value, dict) and canonical(value) + b'\n' == raw, 'Noncanonical generation row')
            yield value, {'generation_file': reference, 'offset': offset, 'length': len(raw),
                          'generation_row_sha256': hash_value(value)}


@contextmanager
def closed_journal(folder, task, digest, evidence):
    """Use original payload validators without __enter__ or orphan adoption."""
    journal = RequestJournal(folder, run_token=task['run_token'], input_sha256=digest,
                             request_hashes={r['request_id']: hash_value(r) for r in task['requests']})
    members = {'writer.lock', 'manifest.json', 'primitives', 'finals'} | {s + '.jsonl' for s in journal.STAGES}
    require({p.name for p in folder.iterdir()} == members, 'Unexpected generation journal member')
    lock = os.open(folder / 'writer.lock', os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        require(evidence.json(folder / 'manifest.json', canonical_required=True) == journal.manifest,
                'Generation journal manifest differs')
        evidence.bind(folder / 'writer.lock')
        journal._events = {s: journal._read_events(s) for s in journal.STAGES}
        for stage in journal.STAGES:
            evidence.bind(folder / (stage + '.jsonl'))
            require(set(journal._events[stage]) == set(journal.manifest['request_hashes']),
                    'Incomplete or ambiguous generation journal')
        names = {hashlib.sha256(rid.encode()).hexdigest() + '.json' for rid in journal.manifest['request_hashes']}
        require(all({p.name for p in (folder / d).iterdir()} == names for d in ('primitives', 'finals')),
                'Missing/extra immutable generation payload')
        yield journal
    finally:
        os.close(lock)


def positive_release(manifest, digest, evidence, *, gpu=True):
    require(manifest['schema'] == 'gpu03_task_specific_resource_guard_v1' and manifest['host'] == 'gpu-03'
            and manifest['uid'] == 3357, 'Wrong accepted supervisor')
    runtime = Path(manifest['runtime'])
    reference = evidence.bind(runtime / 'SUPERVISOR_RESULT.json')
    release = evidence.json(reference['path'])
    names = [w['name'] for w in manifest['workers']]
    require(0 < len(names) <= 8 and len(set(names)) == len(names), 'Invalid producer worker coverage')
    require(release['run_token'] == manifest['run_token'] and release['manifest_sha256'] == digest
            and release['status'] == 'workers_exited_zero' and release['owned_processes_released'] is True
            and release.get('scientific_success_verified') is False
            and set(release['exit_codes']) == set(names)
            and all(type(v) is int and v == 0 for v in release['exit_codes'].values()),
            'Positive producer exit/process/GPU release missing')
    if gpu:
        require(release['gpu_release_verified'] is True and manifest['gpu_uuids']
                and sorted(str(w['gpu_id']) for w in manifest['workers']) == sorted(manifest['gpu_uuids']),
                'Producer GPU/worker coverage differs')
    else:
        require(manifest['gpu_uuids'] == {} and release.get('gpus', {}) == {} and release.get('cpu_only') is True
                and release['gpu_release_verified'] is None
                and all(w['gpu_id'] is None and w['env']['CUDA_VISIBLE_DEVICES'] == '' for w in manifest['workers']),
                'Authored qualification must be CPU-only')
    for gpu, uuid in manifest['gpu_uuids'].items():
        row = release['gpus'][gpu]
        require(row['uuid'] == uuid and row['pids'] == [] and 0 <= row['utilization'] <= 2
                and 0 <= row['total_mib'] - row['free_mib'] <= 1024, 'Producer GPU release unproven')
    started = evidence.jsonl(runtime / 'workers.jsonl', row_limit=8)
    require(len(started) == len(names) and {r['name'] for r in started} == set(names)
            and all(type(r['pid']) is int and r['pid'] > 1 for r in started), 'Missing worker process identities')
    return reference, {r['name']: r for r in started}


def validate_export(export, request, prepared, primitive, task_digest, condition, batch_size):
    require(set(primitive) == {'result'}, 'Unknown generation primitive')
    result = primitive['result']; ids = result['completion_token_ids']
    require(isinstance(ids, list) and 0 < len(ids) <= 1536 and all(type(n) is int and 0 <= n < 2**32 for n in ids)
            and result['generated_token_ids'] == ids and result['fixed_completion_prefix_token_count'] == 0
            and result['batch_size_actual'] == batch_size and result['batch_profile'] == worker.PROFILE
            and result['intervention_strength'] == condition['strength']
            and result['stop_reason'] == ('eos' if ids[-1] in worker.SAMPLING['eos_token_ids'] else 'length')
            and (ids[-1] in worker.SAMPLING['eos_token_ids'] or len(ids) == 1536), 'Incomplete/changed generation tokens')
    expected = {**request, 'problem_id_key': canonical(prepared['problem_id']).decode(),
                'problem_split': prepared['problem_split'], 'prompt': prepared['prompt'],
                'prompt_sha256': hash_value(prepared['prompt']), 'engine_prompt_token_ids': prepared['prompt_token_ids'],
                'generation_seed': request['seed'], 'completion': result['completion'],
                'completion_sha256': hashlib.sha256(result['completion'].encode()).hexdigest(),
                'completion_token_ids': ids, 'result': result, 'input_task_sha256': task_digest,
                'raw_primitive_sha256': hash_value(primitive)}
    require(export == expected, 'Generation export differs from immutable primitive/request')
    evaluator.validate_request(export, prepared['dataset_row'], request['protocol'])


def audit_pilot(plan_ref, phase_ref, qualified_ref, evidence):
    """Independently audit exact saved pilot requests; no label recomputation."""
    plan = read_reference(plan_ref, evidence)
    require(plan['schema'] == plan_io.SCHEMA and plan['status'] == 'prepared_not_dispatched'
            and plan['model_revision'] == worker.REVISION and plan['sampling'] == worker.SAMPLING
            and plan['profile'] == worker.PROFILE and plan['batch_size'] == 8, 'Frozen pilot plan differs')
    for path, value in plan['bound_files'].items():
        evidence.bind(path, value)
    prepared = evidence.jsonl(Path(plan['prepared_rows']), row_limit=96)
    rows = {r['record_id']: r for r in prepared}
    require(len(rows) == len(prepared) == 96 and all(r['problem_split'] == 'direction_fit' for r in prepared),
            'Pilot requires 32 fitting problems and three protocols')
    coordinates = {(canonical(r['problem_id']).decode(), r['protocol']) for r in prepared}
    problems = {key for key, _ in coordinates}
    require(len(problems) == 32 and len(coordinates) == 96
            and coordinates == {(key, protocol) for key in problems for protocol in PROTOCOLS},
            'Prepared pilot problem/protocol grid differs')
    expected = plan_io.requests_for(prepared, plan['pilot_condition_ids'], plan['conditions'], qualification=False)
    q_expected = plan_io.requests_for(prepared, plan['qualification_condition_ids'], plan['conditions'], qualification=True)
    require(plan['pilot_requests'] == expected and plan['qualification_requests'] == q_expected
            and 0 < len(expected) <= MAX_REQUESTS and len({r['request_id'] for r in expected}) == len(expected)
            and plan['budgets']['pilot_including_reuse'] == len(expected)
            and plan['budgets']['qualification'] == len(q_expected),
            'Pilot request universe differs from frozen conditions/prompts')
    universe = {r['request_id']: r for r in expected}
    # Existing qualified-control reuse gate verifies proof, exact request hashes,
    # source/model/vector maps, current file bytes and generation row hashes.
    reused, reuse_bound = plan_io.reuse_from_qualification(plan, qualified_ref)
    proof = read_reference(qualified_ref, evidence)
    for path, value in reuse_bound.items():
        evidence.bind(path, value)
    require(proof['run_token'] == plan['run_token'], 'Qualification token differs')
    manifest = read_reference(phase_ref, evidence)
    require(manifest['run_token'] == plan['run_token'] and manifest['source_root'] == plan['source_root']
            and manifest['science']['stage'] == 'pilot', 'Wrong pilot producer scope')
    for path, value in manifest['bound_files'].items():
        evidence.bind(path, value)
    phase = read_reference(manifest['phase_inputs'], evidence)
    require(phase['schema'] == 'task_specific_causal_phase_inputs_v1' and phase['stage'] == 'pilot'
            and phase['plan'] == plan_ref and phase['run_token'] == plan['run_token']
            and phase['reused_requests'] == reused and phase['expected_total_requests'] == len(expected)
            and phase['expected_reused_requests'] == len(reused)
            and phase['expected_new_requests'] == len(expected) - len(reused), 'Producer phase/request/reuse mapping differs')
    release_ref, launches = positive_release(manifest, phase_ref['sha256'], evidence)
    by_task = {t['task']['path']: t for t in phase['tasks']}
    require(len(by_task) == len(phase['tasks']) == len(manifest['workers']), 'Producer task coverage differs')
    retained, tasks_seen, summaries = {}, set(), []
    for item in manifest['workers']:
        argv = item['argv']
        require(len(argv) == 8 and argv[:5] == [PYTHON, '-B', '-m',
                'infra.gpu03.task_specific_discovery.causal_worker', '--task'] and argv[6] == '--sha256'
                and argv[5] in by_task and argv[5] not in tasks_seen, 'Producer command/task differs')
        task_ref = by_task[argv[5]]['task']; tasks_seen.add(argv[5])
        require(argv[7] == task_ref['sha256'] and item['cwd'] == plan['source_root']
                and item['env']['PYTHONPATH'] == plan['source_root'], 'Producer task/source command binding differs')
        task = read_reference(task_ref, evidence)
        for key in ('run_token', 'model_revision', 'model_snapshot', 'checkpoint', 'prepared_rows', 'vectors',
                    'sampling', 'batch_size', 'profile', 'source_root', 'qualified_source_files'):
            require(task[key] == plan[key], 'Producer input differs: ' + key)
        require(task['schema'] == 'task_specific_causal_generation_v1' and task['stage'] == 'pilot'
                and task['qualification'] == qualified_ref['path'], 'Unqualified producer stage')
        for path, value in task['bound_files'].items():
            evidence.bind(path, value)
        requests = task['requests']
        require(0 < len(requests) <= 2048 and by_task[argv[5]]['requests'] == len(requests)
                and len({r['request_id'] for r in requests}) == len(requests), 'Unbounded/duplicate producer request')
        require(task['conditions'] == {cid: plan['conditions'][cid] for cid in {r['condition_id'] for r in requests}},
                'Producer intervention conditions changed')
        require(all(r['request_id'] in universe and universe[r['request_id']] == r and r['request_id'] not in reused
                    for r in requests), 'New generation replays/replaces frozen request')
        out = Path(task['output']); complete = evidence.json(out / 'COMPLETE.json', canonical_required=True)
        uuid = manifest['gpu_uuids'][str(item['gpu_id'])]
        require(evidence.json(out / 'STARTED.json', canonical_required=True) == {
            'task_sha256': task_ref['sha256'], 'pid': launches[item['name']]['pid'], 'requests': len(requests),
            'gpu_uuid': uuid, 'stage': 'pilot'} and launches[item['name']]['gpu_id'] == item['gpu_id']
            and item['env']['CUDA_VISIBLE_DEVICES'] == uuid, 'Producer process/start identity differs')
        require(complete['status'] == 'succeeded' and complete['stage'] == 'pilot'
                and complete['task_sha256'] == task_ref['sha256'] and complete['requests'] == len(requests)
                and complete['all_returned_requests_retained'] is True and complete['evaluation_performed'] is False
                and complete['causal_success_claimed'] is False
                and complete['cuda_release'] == {'allocated_bytes': 32 << 20, 'reserved_bytes': 32 << 20},
                'Producer terminal/CUDA release missing')
        load = complete['model_load_report']
        require(load['with_adapter'] is True and load['active_adapters']
                and 0 < load['nonzero_lora_parameter_tensors'] <= load['lora_parameter_tensors']
                and load['lora_parameter_elements'] > 0, 'Producer active adapter not proven')
        require(set(complete['artifacts']) == {'generations.jsonl'} and {p.name for p in out.iterdir()} ==
                {'STARTED.json', 'COMPLETE.json', 'generations.jsonl', 'journal'}, 'Incomplete/unexpected pilot output')
        evidence.bind(out / 'generations.jsonl', complete['artifacts']['generations.jsonl'])
        coordinates = {r['request_id']: (i, lane) for i, batch in enumerate(qualification.batches(requests))
                       for lane, r in enumerate(batch)}
        require(all(len(batch) == 8 for batch in qualification.batches(requests)), 'Producer physical batch changed')
        with closed_journal(out / 'journal', task, task_ref['sha256'], evidence) as journal:
            count = 0
            for export, location in stream_rows(out / 'generations.jsonl', evidence):
                require(count < len(requests), 'Extra producer export')
                request = requests[count]; rid = request['request_id']; count += 1
                require(export['request_id'] == rid and rid not in retained, 'Duplicate/reordered producer export')
                primitive, _ = journal._payload(rid, 'primitive', journal._events['primitive_returned'][rid]['file_sha256'])
                final, _ = journal._payload(rid, 'final', journal._events['final_recorded'][rid]['file_sha256'])
                primitive_ref = evidence.bind(journal._path(rid, 'primitive'))
                final_ref = evidence.bind(journal._path(rid, 'final'))
                index, lane = coordinates[rid]
                require(set(final) == {'returned_row', 'batch_index', 'batch_row', 'first_logits', 'batch_wall_seconds'}
                        and final['returned_row'] is True and (final['batch_index'], final['batch_row']) == (index, lane)
                        and final['first_logits'] is None and type(final['batch_wall_seconds']) in (int, float)
                        and math.isfinite(final['batch_wall_seconds']) and final['batch_wall_seconds'] >= 0,
                        'Producer final batch identity differs')
                validate_export(export, request, rows[request['record_id']], primitive, task_ref['sha256'],
                                task['conditions'][request['condition_id']], 8)
                retained[rid] = {**location, 'origin': 'pilot', 'request_sha256': hash_value(request),
                                 'primitive_file': primitive_ref, 'final_file': final_ref}
            require(count == len(requests), 'Missing producer exports')
        summaries.append({'worker': item['name'], 'task': task_ref,
                          'complete': evidence.bind(out / 'COMPLETE.json'), 'requests': len(requests)})
    require(set(retained) == set(universe) - set(reused), 'Incomplete new pilot coverage')
    by_file = {}
    for rid, saved in reused.items():
        by_file.setdefault(saved['generation_file']['path'], set()).add(rid)
    for path, wanted in by_file.items():
        for export, location in stream_rows(Path(path), evidence, max_rows=312):
            rid = export['request_id']
            if rid not in wanted:
                continue
            saved = reused[rid]; request = universe[rid]
            require(rid not in retained and location['generation_row_sha256'] == saved['generation_row_sha256'],
                    'Qualified control export changed/duplicated')
            primitive_envelope = read_reference(saved['primitive_file'], evidence, 8 << 20)
            final_envelope = read_reference(saved['final_file'], evidence, 8 << 20)
            require(primitive_envelope['request_id'] == final_envelope['request_id'] == rid
                    and primitive_envelope['request_sha256'] == final_envelope['request_sha256'] == hash_value(request)
                    and primitive_envelope['run_token'] == final_envelope['run_token'] == plan['run_token']
                    and final_envelope['primitive_file_sha256'] == saved['primitive_file']['sha256'],
                    'Qualified control journal identity differs')
            validate_export(export, request, rows[request['record_id']], primitive_envelope['payload'],
                            primitive_envelope['input_sha256'], plan['conditions'][request['condition_id']], 8)
            retained[rid] = {**location, 'origin': 'qualification', 'request_sha256': hash_value(request),
                             'primitive_file': saved['primitive_file'], 'final_file': saved['final_file']}
    require(set(retained) == set(universe), 'Missing qualified pilot control')
    evidence.unchanged()
    audit = {'schema': 'task_specific_closed_pilot_inputs_v1', 'status': 'all_pilot_generations_independently_verified',
             'plan': plan_ref, 'pilot_manifest': phase_ref, 'qualification': qualified_ref,
             'supervisor_release': release_ref, 'workers': summaries, 'requests': len(expected),
             'new_requests': len(expected) - len(reused), 'reused_requests': len(reused),
             'retained_requests': retained, 'all_returned_requests_retained': True, 'gpu_workers_released': True,
             'generation_or_evaluation_performed': False, 'behavioral_success_claimed': False}
    return plan, rows, retained, audit


def source_and_authored(source_root, task_ref, proof_ref, supervisor_ref, evidence):
    source = Path(source_root)
    files = evidence.json(source / 'SOURCE_FILES.json')
    require(set(authored.REQUIRED_SOURCES) | {LANE_SOURCE, SELF_SOURCE} <= set(files), 'Evaluator source dependency missing')
    for name, value in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Source path escapes package')
        evidence.bind(source / name, value)
    task = read_reference(task_ref, evidence); proof = read_reference(proof_ref, evidence)
    require(task['schema'] == authored.SCHEMA and proof['schema'] == authored.SCHEMA
            and proof['status'] == 'qualified_authored_causal_evaluator_v1'
            and proof['task_sha256'] == task_ref['sha256'] and proof['requests'] == 6
            and proof['outer_process_groups_released'] is True and proof['scientific_completions_evaluated'] == 0
            and proof['model_loaded'] is False and proof['source_files'] == task['source_files']
            and all(name in proof['source_files'] and files.get(name) == proof['source_files'][name]
                    for name in set(authored.REQUIRED_SOURCES) | set(worker.REQUIRED_SOURCE_FILES)),
            'Successful same-source authored evaluator qualification required')
    require(task['venv_dir'] == str(Path(PYTHON).parent.parent), 'Authored evaluator runtime differs from pinned lane runtime')
    supervisor = read_reference(supervisor_ref, evidence)
    require(supervisor['run_token'] == task['run_token'] and supervisor['source_root'] == task['source_dir']
            and len(supervisor['workers']) == 1, 'Wrong authored qualifier supervisor scope')
    item = supervisor['workers'][0]
    require(item['argv'] == [PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.qualify_causal_evaluator',
                            '--task', task_ref['path'], '--sha256', task_ref['sha256']]
            and supervisor['bound_files'].get(task_ref['path']) == binding(task_ref)
            and item['cwd'] == task['source_dir'] and item['env']['PYTHONPATH'] == task['source_dir'],
            'Authored qualifier command/task differs')
    for path, value in supervisor['bound_files'].items():
        evidence.bind(path, value)
    release, _ = positive_release(supervisor, supervisor_ref['sha256'], evidence, gpu=False)
    output = Path(task['output'])
    require(Path(proof_ref['path']) == output / 'COMPLETE.json', 'Authored proof output differs')
    inventory_ref = evidence.bind(output / 'ARTIFACT_MANIFEST.json')
    require(inventory_ref['sha256'] == proof['artifact_manifest_sha256'], 'Authored artifact manifest changed')
    inventory = evidence.json(inventory_ref['path'])
    for name, value in inventory.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Authored artifact escapes output')
        evidence.bind(output / name, value)
    safety_path = output / 'results/safety/qualification.json'
    safety = evidence.json(safety_path); network = evidence.json(output / 'results/safety/NETWORK.json')
    require(safety['status'] == 'passed' and safety['bounded_fixtures']['status'] == 'passed'
            and safety['processes_before'] == safety['processes_after']
            and network['status'] == 'passed'
            and network['existing_fixture_report_sha256'] == evidence.bind(safety_path)['sha256']
            and network['network']['stdout']['connected'] == {'internet': False, 'imds': False},
            'Authored safety/network fixtures did not positively pass')
    for name in ('safety', *PROTOCOLS):
        returned = evidence.json(output / f'{name}.outer.json')
        require(returned['returncode'] == 0 and type(returned['process_group']) is int
                and returned['process_group'] > 1, 'Authored outer process return missing')
    outcomes = []
    for protocol in PROTOCOLS:
        generated, examples, expected = authored.authored_fixtures(protocol)
        paths = [output / 'inputs' / f'{protocol}.{kind}.jsonl' for kind in ('generated', 'dataset')]
        require(evidence.data(paths[0]) == b''.join(canonical(r) + b'\n' for r in generated)
                and evidence.data(paths[1]) == b''.join(canonical(r) + b'\n' for r in examples), 'Authored fixture inputs changed')
        outcomes.extend(authored.audit_evaluation(output / 'results' / protocol / 'evaluation', generated, examples, expected,
            protocol=protocol, run_token=task['run_token'], input_sha256=task_ref['sha256'],
            generated_sha256=evidence.bind(paths[0])['sha256'], dataset_sha256=evidence.bind(paths[1])['sha256']))
    require(outcomes == proof['outcomes'], 'Authored outcomes differ from saved primitives')
    return files, task, release


def write_rows(path, rows):
    path = Path(path)
    with path.open('xb') as stream:
        for row in rows:
            stream.write(canonical(row) + b'\n')
        stream.flush(); os.fchmod(stream.fileno(), 0o400); os.fsync(stream.fileno())
    return {'path': str(path), **io.binding(path)}


def assemble(plan, prepared, retained, output, source_root, source_files, lanes=8):
    """Copy exact canonical exports; memory is bounded by one <=64-MiB shard."""
    require(type(lanes) is int and 1 <= lanes <= 8, 'CPU lane count must be 1..8')
    output = Path(output); inputs = output / 'inputs'; inputs.mkdir(mode=0o700)
    (output / 'results').mkdir(mode=0o700)
    shards, copied = [], set()
    with ExitStack() as stack:
        streams = {path: stack.enter_context(Path(path).open('rb'))
                   for path in {r['generation_file']['path'] for r in retained.values()}}
        require(len(streams) <= 16, 'Too many generation producers')
        for protocol in PROTOCOLS:
            group, byte_count = [], 0
            def flush():
                nonlocal group, byte_count
                if not group:
                    return
                sid = f'{protocol}-{len(shards):03d}'
                folder = inputs / sid; folder.mkdir(mode=0o700)
                examples = {}; generated = []
                for request, row in group:
                    example = prepared[request['record_id']]['dataset_row']
                    key = canonical(example['id']).decode()
                    require(key not in examples or examples[key] == example, 'Conflicting trusted examples')
                    examples[key] = example; generated.append(row)
                gref = write_rows(folder / f'{protocol}.generated.jsonl', generated)
                dref = write_rows(folder / f'{protocol}.dataset.jsonl', examples.values())
                require(gref['size_bytes'] <= evaluator.MAX_FILE_BYTES and dref['size_bytes'] <= evaluator.MAX_FILE_BYTES,
                        'Evaluation shard byte bound exceeded')
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
                evaluator.validate_request(row, prepared[request['record_id']]['dataset_row'], protocol)
                require(len(raw) <= evaluator.MAX_LINE_BYTES, 'Evaluation row exceeds bound')
                if len(group) == 256 or byte_count + len(raw) > evaluator.MAX_FILE_BYTES:
                    flush()
                group.append((request, row)); byte_count += len(raw)
            flush()
    require(copied == set(retained), 'Evaluation plan lost/added requests')
    count = min(lanes, len(shards)); groups = [[] for _ in range(count)]
    for index, shard in enumerate(shards):
        groups[index % count].append(shard)
    require(all(0 < len(g) <= 64 for g in groups), 'Too few bounded evaluator lanes')
    tasks = []
    for lane, values in enumerate(groups):
        task = {'schema': LANE_SCHEMA, 'run_token': plan['run_token'], 'source_dir': str(source_root),
                'source_files': source_files, 'venv_dir': str(Path(PYTHON).parent.parent),
                'output': str(output / 'results' / f'lane_{lane:02d}'), 'deadline_seconds': 14400,
                'problem_split': 'direction_fit', 'shards': [{k: s[k] for k in ('id', 'protocol', 'generated', 'dataset')} for s in values]}
        path = output / f'task_{lane:02d}.json'; io.write_json(path, task)
        reference = {'path': str(path), **io.binding(path)}
        tasks.append({'lane': lane, 'task': reference, 'requests': sum(len(s['request_ids']) for s in values),
                      'argv': [PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.causal_eval_lane',
                               '--task', str(path), '--sha256', reference['sha256']]})
    return tasks, shards


def build(plan_ref, phase_ref, qualified_ref, authored_task_ref, authored_proof_ref, authored_supervisor_ref,
          source_root, output, lanes=8):
    require(type(lanes) is int and 1 <= lanes <= 8, 'CPU lane count must be 1..8')
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists()
            and not output.is_symlink(), 'Fresh canonical evaluation plan output required')
    claim = Path(plan_ref['path']).parent / 'PILOT_EVAL_TASK_BINDING.json'
    if claim.exists() or claim.is_symlink():
        raise FileExistsError('Evaluation planning already consumed; inspect its retained claim')
    evidence = qualification.Evidence()
    plan, prepared, retained, audit = audit_pilot(plan_ref, phase_ref, qualified_ref, evidence)
    files, task, authored_release = source_and_authored(source_root, authored_task_ref, authored_proof_ref,
                                                      authored_supervisor_ref, evidence)
    require(task['run_token'] == plan['run_token'], 'Authored evaluator qualification token differs')
    audit.update(independently_hashed_files=evidence.files, auditor_sha256=io.sha256(__file__),
                 authored_qualification=authored_proof_ref, authored_supervisor_release=authored_release)
    io.write_json(claim, {'schema': SCHEMA, 'plan': plan_ref, 'pilot_manifest': phase_ref,
        'qualification': qualified_ref, 'authored_task': authored_task_ref, 'authored_proof': authored_proof_ref,
        'authored_supervisor': authored_supervisor_ref,
        'source_root': str(source_root), 'source_files_sha256': hash_value(files), 'output': str(output),
        'lanes': lanes, 'request_ids': [r['request_id'] for r in plan['pilot_requests']],
        'input_audit_sha256': hash_value(audit), 'evaluation_replay_authorized': False})
    output.mkdir(mode=0o700)
    tasks, shards = assemble(plan, prepared, retained, output, source_root, files, lanes)
    evidence.unchanged()
    audit_path = output / 'INPUT_AUDIT.json'; io.write_json(audit_path, audit)
    bound = {str(Path(source_root) / name): value for name, value in files.items()}
    for reference in (plan_ref, phase_ref, qualified_ref, authored_task_ref, authored_proof_ref, authored_supervisor_ref,
                      {'path': str(claim), **io.binding(claim)}, {'path': str(audit_path), **io.binding(audit_path)}):
        bound[reference['path']] = binding(reference)
    for shard in shards:
        for kind in ('generated', 'dataset'):
            bound[shard[kind]['path']] = binding(shard[kind])
    for item in tasks:
        bound[item['task']['path']] = binding(item['task'])
    phase = {'schema': SCHEMA, 'status': 'prepared_not_dispatched', 'run_token': plan['run_token'],
             'source_root': str(source_root), 'tasks': tasks, 'shards': shards,
             'requests': len(retained), 'new_evaluations': len(retained), 'generation_performed': False,
             'evaluation_performed': False, 'training': False, 'problem_split': 'direction_fit',
             'bound_files': bound, 'input_audit': {'path': str(audit_path), **io.binding(audit_path)},
             'limits': {'lanes': len(tasks), 'evaluator_workers_per_lane': 2, 'records_per_shard': 256,
                        'bytes_per_shard': evaluator.MAX_FILE_BYTES, 'deadline_seconds': 14400},
             'dispatch_authorized_by_this_file': False, 'resource_supervisor_binding_required': True}
    path = output / 'PHASE_INPUTS.json'; io.write_json(path, phase)
    return {'path': str(path), **io.binding(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'pilot-manifest', 'qualification', 'authored-task', 'authored-proof', 'authored-supervisor'):
        parser.add_argument('--' + name, required=True); parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--source-root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--lanes', type=int, default=8); args = parser.parse_args()
    require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357 and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Preparation requires accepted host owner CPU-only environment')
    require(Path(args.output).parent == Path(ROOT) and Path(args.source_root) == Path(__file__).resolve().parents[3],
            'Preparation must use this run and current bound source')
    require(Path(args.source_root).parent == Path(ROOT), 'Evaluator source is outside this run')
    references = []
    for name in ('plan', 'pilot_manifest', 'qualification', 'authored_task', 'authored_proof', 'authored_supervisor'):
        path = Path(getattr(args, name))
        require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT), 'Evidence is outside this run')
        reference = {'path': str(path), **io.binding(path)}
        require(reference['sha256'] == getattr(args, name + '_sha256'), 'Explicit input SHA differs')
        references.append(reference)
    require(plan_io.read_json(references[0]['path'])['run_token'] == Path(ROOT).name, 'Frozen plan belongs to another run')
    result = build(*references, args.source_root, args.output, args.lanes)
    print(canonical(result).decode())


if __name__ == '__main__':
    main()
