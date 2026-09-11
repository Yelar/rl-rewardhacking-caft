#!/usr/bin/env python3
"""Bounded CPU evaluation of complete, reviewed generation campaigns on gpu-04.

The builder never executes a completion. Workers run the existing evaluator only
inside its outer allowlist sandbox. Partial journals are preserved; reuse requires
a separately frozen recovery plan, and never silently re-evaluates a saved row.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.gpu03.direction_discovery import evaluate, sandbox

ROOT = Path('/scratch/researcher/codex_runs')
OWNER = 'researcher'
HOST = 'gpu-04'
PYTHON = '/scratch/researcher/rl-rewardhacking-gpu03-runtime/venv/bin/python'
PLAN_SHA256 = 'd4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'
ENTRY = 'infra/gpu03/direction_discovery/eval_run.py'
GIB = 1024 ** 3
MAX_REQUESTS = 4096
INTERVAL = 5
IDENTITY_FIELDS = ('request_id', 'record_id', 'problem_id', 'problem_split', 'condition_id', 'scope', 'sample_index', 'seed')
VERSIONS = ('torch', 'transformers', 'peft', 'pydantic', 'safetensors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha256(path):
    return evaluate.file_sha(path)


def info(path):
    path = Path(path)
    return {'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def object_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key')
        result[key] = value
    return result


def parse(text):
    def invalid(_):
        raise ValueError('Nonfinite JSON is forbidden')
    result = json.loads(text, object_pairs_hook=object_unique, parse_constant=invalid)
    canonical(result)  # Also rejects overflow such as 1e999.
    return result


def read_json(path):
    return parse(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        while True:
            line = stream.readline(8 * 1024 * 1024 + 1)
            if not line:
                break
            require(len(line) <= 8 * 1024 * 1024 and line.endswith('\n'), 'Oversized or incomplete JSONL record; preserve journal for explicit recovery')
            row = parse(line)
            require(isinstance(row, dict), 'JSONL record must be an object')
            yield row


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def append(path, value):
    evaluate.append(path, value)


def unique_rows(rows, label):
    result = {}
    for row in rows:
        key = row.get('request_id')
        require(isinstance(key, str) and key and key not in result, 'Missing or duplicate/conflicting request ID in ' + label)
        result[key] = row
        require(len(result) <= MAX_REQUESTS, 'Request count exceeds frozen generation budget')
    return result


def expected_coverage(rows, expected_ids, label):
    actual = unique_rows(rows, label)
    require(set(actual) == set(expected_ids), f'{label} exact request coverage mismatch: missing={len(set(expected_ids) - set(actual))}, extra={len(set(actual) - set(expected_ids))}')
    return actual


def match_planned(generated, planned):
    require('result' not in planned, 'Request plan must not contain generated outcomes')
    require(all(key in generated and canonical(generated[key]) == canonical(value) for key, value in planned.items()), 'Generation request differs from immutable plan')


def validate_inputs(generations, request_plan, prepared_rows, dataset_rows):
    require('execution_partition' not in request_plan and 'test_execution' not in request_plan and 'finalist_recovery' not in request_plan,
            'Evaluation requires the full scientific plan, never an execution round')
    require(request_plan.get('mode') == 'generate', 'Only mode=generate packages may enter scientific evaluation')
    planned = unique_rows(request_plan['requests'], 'immutable request plan')
    require(planned, 'Empty request plan')
    generated = expected_coverage(generations, planned, 'generation inputs')
    prepared = {row['record_id']: row for row in prepared_rows}
    dataset = {str(row['id']): row for row in dataset_rows}
    require(len(prepared) == len(prepared_rows), 'Duplicate prepared source records')
    require(len(dataset) == len(dataset_rows), 'Duplicate original dataset problems')
    for key, row in generated.items():
        match_planned(row, planned[key])
        require(row['record_id'] in prepared and str(row['problem_id']) in dataset, 'Unknown generation source')
        evaluate.validate_generation(row, prepared[row['record_id']], dataset[str(row['problem_id'])])
    return generated


def test_bundle_context(reference, request_plan, *, prepared_sha=None):
    """Validate only frozen test metadata; never inspect prepared/completion rows."""
    from infra.gpu03.direction_discovery import test_execution
    bundle, full = test_execution.bundle_context(reference)
    require(request_plan == full and full['mode'] == 'generate' and
            full['phase'] == 'untouched_test_bundle' and full['evaluation_partition'] == 'untouched_test',
            'Test evaluation requires the exact full bundle request plan')
    if prepared_sha is not None:
        require(prepared_sha == bundle['prepared_records']['sha256'], 'Test evaluation prepared union differs from bundle')
    return bundle


def remote_package(package):
    return isinstance(package, dict) and package.get('kind') == 'remote_generation'


def generation_package_key(package):
    return canonical({'kind': package['kind'], 'replica': package['replica']}) if remote_package(package) else package['manifest']


def remote_generation_context(package, *, stored=None):
    """Replay the explicit local replica map; never treat deployment paths as local."""
    from infra.gpu03.direction_discovery import remote_generation
    fields = {'kind', 'replica'} if stored is None else {
        'kind', 'replica', 'manifest_sha256', 'artifact_manifest_sha256', 'verification', 'request_ids'}
    require(set(package) == fields or (stored is not None and set(package) == {'kind', 'replica'}),
            'Remote generation reference contains unbound fields')
    reference = {k: package[k] for k in ('kind', 'replica')}
    context = remote_generation.context(reference, verify=stored is None)
    m, proof = context['manifest'], context['proof']
    require(m['host'] in remote_generation.PROTOCOLS and proof.get('host') == m['host'] and
            proof.get('status') == 'verified' and
            proof.get('gpu_release_verified') is True and proof.get('manifest_sha256') == context['manifest_sha256'] and
            proof.get('run_token') == m['run_token'], 'Remote generation lacks independently verified terminal release')
    require(context['authority_manifest_sha256'] == context['manifest_sha256'] and
            context['scientific_conditions'] == context['request_plan']['conditions'], 'Remote authority identity differs')
    if stored is not None:
        require(generation_package_key(stored) == generation_package_key(reference) and
                stored.get('verification') == proof and
                stored.get('manifest_sha256') == context['manifest_sha256'] and
                stored.get('artifact_manifest_sha256') == context['artifact_manifest_sha256'] and
                stored.get('request_ids') == sorted(r['request_id'] for r in context['request_plan']['requests']),
                'Stored remote generation proof changed')
    return context


def replica_rows(binding):
    """Hash and parse the same bounded snapshot, with the existing strict JSON rules."""
    path = Path(binding['path'])
    require(type(binding['size_bytes']) is int and 0 <= binding['size_bytes'] <= 512 * 1024 ** 2 and
            path.stat().st_size == binding['size_bytes'], 'Remote replica result size changed or exceeds bound')
    digest, rows, total = hashlib.sha256(), [], 0
    with path.open('rb') as stream:
        while True:
            line = stream.readline(8 * 1024 * 1024 + 1)
            if not line: break
            total += len(line); digest.update(line)
            require(total <= binding['size_bytes'] and len(line) <= 8 * 1024 * 1024 and line.endswith(b'\n'),
                    'Oversized or incomplete remote generation record')
            row = parse(line.decode('utf-8'))
            require(isinstance(row, dict), 'Remote generation JSONL record must be an object')
            rows.append(row)
    require(total == binding['size_bytes'] and digest.hexdigest() == binding['sha256'], 'Remote generation snapshot hash changed')
    return rows


def remote_task_rows(context, task):
    results = {r['worker']: r for r in context['results']}
    require(len(results) == len(context['results']) == len(context['tasks']) and
            set(results) == {t['task']['worker_name'] for t in context['tasks']}, 'Remote worker result map is incomplete')
    return replica_rows(results[task['worker_name']])


def remote_prepared_rows(context, task):
    binding = context['file_bindings'].get(task['prepared_records'])
    require(binding is not None and binding['sha256'] == context['prepared_records_sha256'],
            'Remote prepared source lacks an exact local replica binding')
    return replica_rows(binding)


def generation_proof(package, verification, ids, *, remote=None):
    if remote is not None:
        return {'kind': 'remote_generation', 'replica': package['replica'],
                'manifest_sha256': remote['manifest_sha256'], 'artifact_manifest_sha256': remote['artifact_manifest_sha256'],
                'verification': verification, 'request_ids': sorted(ids)}
    return {'manifest': package['manifest'], 'manifest_sha256': package['manifest_sha256'],
            'artifact_manifest_sha256': package['artifact_manifest_sha256'],
            'verification': verification, 'request_ids': sorted(ids)}


def test_generation_metadata(packages, *, request_plan, test_bundle, master_sha, stored_proofs=None):
    """Join every released fixed piece before any completion row can be read.

    Initial collection re-verifies producers. Later manifest verification uses
    the immutable copies of those exact producer proofs and rechecks their
    manifest/task/bundle chain; it never reruns scientific code.
    """
    from infra.gpu03.direction_discovery import supervisor, test_execution
    bundle = test_bundle_context(test_bundle, request_plan)
    require(request_plan['master_plan_sha256'] == master_sha, 'Test bundle belongs to another experiment plan')
    require(len(packages) == len(bundle['execution_partitions']), 'Evaluation requires all fixed test partitions')
    stored = None if stored_proofs is None else {generation_package_key(p): p for p in stored_proofs}
    if stored is not None:
        require(len(stored) == len(stored_proofs) == len(packages), 'Duplicate or missing stored test producer proof')
    contexts, by_index = {}, {}
    for package in packages:
        key = generation_package_key(package)
        require(key not in contexts, 'Duplicate generation package')
        if remote_package(package):
            remote = remote_generation_context(package, stored=None if stored is None else stored.get(key, {}))
            m, proof, part = remote['manifest'], remote['proof'], remote['request_plan']
            science = m['scientific']
            require(science.get('training') is False and science.get('master_plan_sha256') == master_sha and
                    science.get('test_bundle') == test_bundle and not any(k in science for k in ('execution_partition', 'finalist_recovery')),
                    'Remote generation belongs to another test bundle or phase')
            _, meta = test_execution.validate_part(part, full=request_plan)
            require(meta == science.get('test_execution') and meta['partition_index'] not in by_index and
                    remote['prepared_records_sha256'] == bundle['prepared_records']['sha256'] == science.get('input_prepared_sha256'),
                    'Remote fixed test partition or prepared union changed')
            # The authority envelope, rather than deployment.bound_files, binds
            # GPU04 bundle and predecessor dependencies. Its context replays them.
            task_rows = []
            for item in remote['tasks']:
                worker, task = item['worker'], item['task']
                require(worker['success_expect']['mode'] == task['mode'] == 'generate' and
                        worker['success_expect']['requests'] == len(task['requests']) and
                        task['run_token'] == m['run_token'] and task['worker_name'] == worker['name'] and
                        task['conditions'] == request_plan['conditions'] and task['sampling'] == request_plan['sampling'],
                        'Remote test worker identity or sampling changed')
                task_rows.extend(task['requests'])
            require(unique_rows(task_rows, 'remote test task coverage') == unique_rows(part['requests'], 'fixed test partition'),
                    'Remote tasks do not cover the exact fixed test partition')
            if stored is not None:
                require(stored[key]['request_ids'] == sorted(r['request_id'] for r in task_rows), 'Stored remote test coverage changed')
            context = {'manifest': m, 'verification': proof, 'part': part, 'package': package, 'remote': remote}
            contexts[key] = context; by_index[meta['partition_index']] = context
            continue
        manifest = Path(package['manifest'])
        require(str(manifest) not in contexts, 'Duplicate generation package')
        m = test_execution.frozen(manifest, package['manifest_sha256'])
        if stored is None:
            proof = supervisor.verify(manifest, package['manifest_sha256'])
        else:
            previous = stored.get(str(manifest), {})
            require(all(previous.get(k) == package[k] for k in
                        ('manifest_sha256', 'artifact_manifest_sha256')), 'Stored test producer identity changed')
            proof = previous.get('verification', {})
        require(proof.get('status') == 'verified' and proof.get('gpu_release_verified') is True and
                proof.get('manifest_sha256') == package['manifest_sha256'] and proof.get('run_token') == m['run_token'],
                'Test producer lacks a successful independent release proof')
        science = m['scientific']
        require(m['host'] == HOST and science.get('training') is False and
                science.get('master_plan_sha256') == master_sha and science.get('test_bundle') == test_bundle and
                'execution_partition' not in science, 'Cannot mix test bundles or finalist generation packages')
        part_path = Path(m['stage']) / 'input/request_plan.json'
        part = test_execution.frozen(part_path, m['bound_files'][str(part_path)]['sha256'])
        _, meta = test_execution.validate_part(part, full=request_plan)
        require(meta == science.get('test_execution') and meta['bundle'] == test_bundle and
                meta['partition_index'] not in by_index, 'Duplicate or altered fixed test partition')
        for bound_path in test_execution.bindings(part):
            require(m['bound_files'].get(str(bound_path)) == info(bound_path), 'Test producer omitted a bundle/proof binding')
        require(science.get('input_prepared_sha256') == bundle['prepared_records']['sha256'],
                'Test producer prepared union differs from bundle')
        require(sha256(Path(m['output']) / 'artifact_manifest.json') == package['artifact_manifest_sha256'],
                'Generation artifact manifest changed')
        task_rows = []
        for worker in m['workers']:
            task_path = Path(worker['command'][3])
            task = test_execution.frozen(task_path, m['bound_files'][str(task_path)]['sha256'])
            require(worker['success_expect']['mode'] == task['mode'] == 'generate' and
                    worker['success_expect']['requests'] == len(task['requests']) and
                    task['run_token'] == m['run_token'] and task['worker_name'] == worker['name'] and
                    task['conditions'] == request_plan['conditions'] and task['sampling'] == request_plan['sampling'] and
                    sha256(task['prepared_records']) == bundle['prepared_records']['sha256'],
                    'Test worker identity, sampling or prepared source differs')
            task_rows.extend(task['requests'])
        require(unique_rows(task_rows, 'test package tasks') == unique_rows(part['requests'], 'fixed test partition'),
                'Test worker tasks do not cover the exact fixed partition')
        if stored is not None:
            require(stored[str(manifest)]['request_ids'] == sorted(r['request_id'] for r in task_rows),
                    'Stored test producer coverage changed')
        context = {'manifest': m, 'verification': proof, 'part': part, 'package': package}
        contexts[str(manifest)] = context; by_index[meta['partition_index']] = context
    require(set(by_index) == set(range(len(bundle['execution_partitions']))), 'Fixed test partitions are skipped or duplicated')
    for index, context in by_index.items():
        for j, ref in enumerate(context['part']['test_execution']['predecessors']):
            prior = by_index[j]
            if test_execution.remote_reference(ref):
                require(remote_package(prior['package']) and generation_package_key(ref) == generation_package_key(prior['package']) and
                        test_execution.predecessor_context(ref)['proof'] == prior['verification'],
                        'Remote test predecessor chain differs from complete packages')
            else:
                require(not remote_package(prior['package']) and all(ref[k] == prior['package'][k] for k in
                            ('manifest', 'manifest_sha256', 'artifact_manifest_sha256')) and
                        test_execution.frozen(ref['verification'], ref['verification_sha256']) == prior['verification'],
                        'Test predecessor chain differs from supplied complete packages')
    physical = [c['remote']['manifest_sha256'] if c.get('remote') else c['package']['manifest_sha256'] for c in contexts.values()]
    require(len(set(physical)) == len(physical), 'Repeated physical generation deployment')
    expected_coverage((r for i in sorted(by_index) for r in by_index[i]['part']['requests']),
                      unique_rows(request_plan['requests'], 'full test bundle'), 'complete fixed test partitions')
    return contexts


def recovered_finalist_metadata(packages, *, request_plan, master_sha, stored_proofs=None):
    """Require complete logical R0 plus successful R1 before collecting rows."""
    from infra.gpu03.direction_discovery import execution_partition, finalist_recovery, supervisor
    require(request_plan is not None, 'Recovered finalist evaluation requires the full740 plan')
    execution_partition.validate_full(request_plan)
    logical = [p for p in packages if p.get('kind') == 'recovered_finalist_round0']
    normal = [p for p in packages if 'kind' not in p or remote_package(p)]
    require(len(packages) == 2 and len(logical) == len(normal) == 1,
            'Recovered evaluation requires only complete logical round0 and normal round1')
    ref = {key: logical[0][key] for key in ('kind', 'logical_round')}
    package = normal[0]
    remote = remote_generation_context(package, stored=package if stored_proofs is not None else None) if remote_package(package) else None
    if remote:
        m, verification = remote['manifest'], remote['proof']
        manifest_sha = remote['manifest_sha256']
    else:
        manifest = Path(package['manifest'])
        m = supervisor.load_manifest(manifest, package['manifest_sha256'])
        verification = supervisor.verify(manifest, package['manifest_sha256']) if stored_proofs is None else package['verification']
        manifest_sha = package['manifest_sha256']
    require(verification.get('status') == 'verified' and verification.get('gpu_release_verified') is True and
            verification.get('manifest_sha256') == manifest_sha and verification.get('run_token') == m['run_token'],
            'Second finalist round lacks verified terminal success')
    science = m['scientific']
    require(science['master_plan_sha256'] == master_sha == request_plan['master_plan_sha256'] and
            not any(key in science for key in ('test_bundle', 'test_execution', 'finalist_recovery')),
            'Recovered validation cannot mix test or bare recovery generation packages')
    if remote:
        part = remote['request_plan']
    else:
        part_path = Path(m['stage']) / 'input/request_plan.json'
        part = execution_partition.frozen(part_path, m['bound_files'][str(part_path)]['sha256'])
    _, meta = execution_partition.validate_part(part, full=request_plan)
    require(meta == science.get('execution_partition') and meta['round_index'] == 1 and meta['predecessor'] == ref,
            'Second finalist round has a different logical predecessor or full plan')
    if not remote:
        artifact = Path(m['output']) / 'artifact_manifest.json'
        require(sha256(artifact) == package['artifact_manifest_sha256'], 'Second finalist generation artifact changed')
    tasks, covered = [], []
    task_items = remote['tasks'] if remote else [
        {'worker': worker, 'task': execution_partition.frozen(worker['command'][3], m['bound_files'][worker['command'][3]]['sha256'])}
        for worker in m['workers']]
    for item in task_items:
        worker, task = item['worker'], item['task']
        require(task['mode'] == worker['success_expect']['mode'] == 'generate' and
                task['conditions'] == request_plan['conditions'] and task['sampling'] == request_plan['sampling'] and
                task['run_token'] == m['run_token'] and task['worker_name'] == worker['name'] and
                len(task['requests']) == worker['success_expect']['requests'], 'Second finalist task identity or sampling differs')
        tasks.append((worker, task)); covered.extend(task['requests'])
    require(unique_rows(covered, 'second finalist task coverage') == unique_rows(part['requests'], 'second finalist plan'),
            'Second finalist tasks are incomplete')
    context = finalist_recovery.logical_context(ref, verify=stored_proofs is None)
    expected = [r for r in request_plan['requests'] if r['sample_index'] in (0, 1)]
    require(context['full'] == request_plan and context['full_plan_sha256'] == meta['full_plan_sha256'] and
            len(context['request_ids']) == len(set(context['request_ids'])) == 370 and
            set(context['request_ids']) == {r['request_id'] for r in expected}, 'Logical first round does not cover the exact370 requests')
    if stored_proofs is not None:
        require(logical[0]['verification'] == context['proof'], 'Stored logical recovery verification changed')
        require(set(logical[0]['request_ids']) == set(context['request_ids']) and
                set(package['request_ids']) == {r['request_id'] for r in part['requests']}, 'Stored recovered coverage differs')
    return {'reference': ref, 'logical': context, 'package': package, 'manifest': m,
            'verification': verification, 'tasks': tasks, 'part': part, 'remote': remote}


def collect_recovered_finalist(packages, *, request_plan, prepared_rows, master_sha):
    context = recovered_finalist_metadata(packages, request_plan=request_plan, master_sha=master_sha)
    logical = context['logical']; path = Path(logical['rows_path'])
    require(sha256(path) == logical['rows_sha256'], 'Logical generation rows changed before collection')
    expected = {r['request_id']: r for r in request_plan['requests'] if r['sample_index'] in (0, 1)}
    rows = expected_coverage(read_jsonl(path), expected, 'logical first-round generation')
    require(sha256(path) == logical['rows_sha256'], 'Logical generation rows changed during collection')
    for key, row in rows.items():
        match_planned(row, expected[key])
    all_rows = list(rows.values()); second_ids = []
    for worker, task in context['tasks']:
        planned = unique_rows(task['requests'], 'second finalist task')
        native = remote_task_rows(context['remote'], task) if context['remote'] else read_jsonl(Path(task['output']) / 'results.jsonl')
        completed = expected_coverage(native, planned, 'second finalist worker')
        for key, row in completed.items():
            match_planned(row, planned[key])
        if prepared_rows is not None:
            sources = unique_source_rows(remote_prepared_rows(context['remote'], task) if context['remote'] else list(read_jsonl(task['prepared_records'])))
            expected_sources = unique_source_rows(prepared_rows)
            require(all(sources.get(row['record_id']) == expected_sources.get(row['record_id']) and
                        row['record_id'] in expected_sources for row in completed.values()), 'Second finalist prepared source changed')
        all_rows.extend(completed.values()); second_ids.extend(completed)
    expected_coverage(all_rows, unique_rows(request_plan['requests'], 'full740 finalist plan'), 'complete recovered finalist validation')
    package = context['package']
    proofs = [{**context['reference'], 'verification': logical['proof'], 'request_ids': sorted(rows)},
              generation_proof(package, context['verification'], second_ids, remote=context['remote'])]
    return all_rows, proofs


def unique_source_rows(rows):
    result = {row['record_id']: row for row in rows}
    require(len(result) == len(rows), 'Duplicate prepared source records')
    return result


def collect_generation_packages(packages, *, request_plan=None, prepared_rows=None, master_sha=PLAN_SHA256, test_bundle=None):
    """Independently verify complete producers before inspecting any output row."""
    from infra.gpu03.direction_discovery import supervisor
    if any(p.get('kind') == 'recovered_finalist_round0' for p in packages):
        require(test_bundle is None, 'Recovered finalist cannot enter a test evaluation')
        return collect_recovered_finalist(packages, request_plan=request_plan, prepared_rows=prepared_rows, master_sha=master_sha)
    all_rows, proofs, execution_rounds = [], [], []
    if request_plan is not None:
        require('execution_partition' not in request_plan and 'test_execution' not in request_plan and 'finalist_recovery' not in request_plan,
                'Evaluation requires the full scientific plan, never an execution round')
        require(request_plan.get('phase') != 'untouched_test_bundle' or test_bundle is not None,
                'Full test evaluation requires an explicit bundle reference')
    test_contexts = test_generation_metadata(packages, request_plan=request_plan, test_bundle=test_bundle,
                                            master_sha=master_sha) if test_bundle is not None else {}
    seen = set()
    for package in packages:
        if remote_package(package):
            # Cross-host evaluation is only admitted through a complete test
            # bundle or the complete logical-R0/R1 validation path above.
            require(test_bundle is not None, 'Remote generation needs a complete test bundle or recovered full740 validation')
            key = generation_package_key(package)
            require(key not in seen and key in test_contexts, 'Duplicate or unverified remote generation package')
            seen.add(key); context = test_contexts[key]; remote = context['remote']
            package_ids = []
            for item in remote['tasks']:
                task = item['task']; planned = unique_rows(task['requests'], 'remote generation task')
                rows = expected_coverage(remote_task_rows(remote, task), planned, 'complete remote generation worker')
                for rid, row in rows.items(): match_planned(row, planned[rid])
                if prepared_rows is not None:
                    expected_sources = unique_source_rows(prepared_rows)
                    sources = unique_source_rows(remote_prepared_rows(remote, task))
                    require(all(row['record_id'] in expected_sources and sources.get(row['record_id']) == expected_sources[row['record_id']]
                                for row in rows.values()), 'Remote generation prepared source changed')
                all_rows.extend(rows.values()); package_ids.extend(rows)
            require(len(package_ids) == len(set(package_ids)) == context['part']['new_generation_requests'] and
                    set(package_ids) == {r['request_id'] for r in context['part']['requests']}, 'Remote fixed test coverage changed')
            unique_rows(all_rows, 'all generation packages')
            proofs.append(generation_proof(package, context['verification'], package_ids, remote=remote))
            continue
        manifest = Path(package['manifest'])
        require(str(manifest) not in seen, 'Duplicate generation package')
        seen.add(str(manifest))
        context = test_contexts.get(str(manifest))
        verification = context['verification'] if context else supervisor.verify(manifest, package['manifest_sha256'])
        m = context['manifest'] if context else supervisor.load_manifest(manifest, package['manifest_sha256'])
        require('finalist_recovery' not in m.get('scientific', {}),
                'Bare recovery cannot be evaluated; require the complete logical first round and full740 plan')
        require(test_bundle is not None or ('test_execution' not in m.get('scientific', {}) and
                'test_bundle' not in m.get('scientific', {})), 'Test generation requires an explicit full bundle reference')
        if request_plan is not None:
            require(m['scientific']['master_plan_sha256'] == master_sha, 'Generation package belongs to another experiment plan')
        meta = m.get('scientific', {}).get('execution_partition')
        if meta is not None:
            from infra.gpu03.direction_discovery import execution_partition
            require(request_plan is not None, 'Partitioned generation requires the full scientific evaluation plan')
            part_path = Path(m['stage']) / 'input/request_plan.json'
            part = execution_partition.frozen(part_path, m['bound_files'][str(part_path)]['sha256'])
            _, checked_meta = execution_partition.validate_part(part, full=request_plan)
            require(checked_meta == meta, 'Generation manifest execution partition changed')
            execution_rounds.append((meta, package['manifest_sha256']))
        artifact = Path(m['output']) / 'artifact_manifest.json'
        require(sha256(artifact) == package['artifact_manifest_sha256'], 'Generation artifact manifest changed')
        package_ids = []
        for worker in m['workers']:
            require(worker['success_expect']['mode'] == 'generate', 'Non-generation/qualification worker cannot enter evaluation')
            task = read_json(worker['command'][3])
            require(task['mode'] == 'generate', 'Generation task mode mismatch')
            if request_plan is not None:
                require(task['conditions'] == request_plan['conditions'], 'Generation conditions differ from immutable request plan')
            requests = unique_rows(task['requests'], 'generation task')
            rows = expected_coverage(read_jsonl(Path(task['output']) / 'results.jsonl'), requests, 'terminal generation worker')
            require(len(rows) == worker['success_expect']['requests'], 'Terminal generation worker count differs')
            for key, row in rows.items():
                match_planned(row, requests[key])
            if prepared_rows is not None:
                expected_sources = {r['record_id']: r for r in prepared_rows}
                source_rows = list(read_jsonl(task['prepared_records']))
                sources = {r['record_id']: r for r in source_rows}
                require(len(sources) == len(source_rows), 'Generation task contains duplicate prepared sources')
                require(all(row['record_id'] in expected_sources and sources.get(row['record_id']) == expected_sources[row['record_id']]
                            for row in rows.values()), 'Generation prepared source changed before evaluation')
            all_rows.extend(rows.values()); package_ids.extend(rows)
        if meta is not None:
            require(len(package_ids) == len(set(package_ids)) == 370 and
                    set(package_ids) == {r['request_id'] for r in part['requests']},
                    'Generation package does not cover its exact execution round')
        if context:
            require(len(package_ids) == len(set(package_ids)) == context['part']['new_generation_requests'] and
                    set(package_ids) == {r['request_id'] for r in context['part']['requests']},
                    'Generation package does not cover its exact fixed test partition')
        unique_rows(all_rows, 'all generation packages')
        proofs.append({'manifest': str(manifest), 'manifest_sha256': package['manifest_sha256'],
                       'artifact_manifest_sha256': package['artifact_manifest_sha256'],
                       'verification': verification, 'request_ids': sorted(package_ids)})
    require(proofs, 'Missing terminal generation packages')
    if execution_rounds:
        require(len(execution_rounds) == len(proofs) == 2 and
                {meta['round_index'] for meta, _ in execution_rounds} == {0, 1} and
                len({meta['full_plan_sha256'] for meta, _ in execution_rounds}) == 1,
                'Evaluation requires both disjoint complete execution rounds from one full plan')
        by_round = {meta['round_index']: (meta, digest) for meta, digest in execution_rounds}
        require(by_round[1][0]['predecessor']['manifest_sha256'] == by_round[0][1], 'Evaluation round predecessor differs')
        expected_coverage(all_rows, unique_rows(request_plan['requests'], 'full scientific plan'), 'combined execution rounds')
    if test_bundle is not None:
        expected_coverage(all_rows, unique_rows(request_plan['requests'], 'full test bundle'), 'complete test generation packages')
    return all_rows, proofs


def process_info(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'pgid': int(fields[2]), 'start_ticks': int(fields[19]),
                'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), 'uid': Path(f'/proc/{pid}').stat().st_uid}
    except FileNotFoundError:
        return None


def same_process(old):
    current = process_info(old['pid'])
    return current is not None and current['start_ticks'] == old['start_ticks']


def host_identity():
    require(socket.gethostname().split('.')[0] == HOST and pwd.getpwuid(os.getuid()).pw_name == OWNER, 'Evaluation requires gpu-04 and the reviewed owner')


def cpu_environment():
    uid = os.getuid()
    return {'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'USER': OWNER, 'LOGNAME': OWNER,
            'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONUNBUFFERED': '1', 'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true',
            'XDG_RUNTIME_DIR': f'/run/user/{uid}', 'DBUS_SESSION_BUS_ADDRESS': f'unix:path=/run/user/{uid}/bus'}


def clean_command(argv):
    return ['/usr/bin/env', '-i', *[key + '=' + value for key, value in cpu_environment().items()], *argv]


def command(m, action, *extra):
    return [m['python'], str(Path(m['stage']) / 'source' / ENTRY), '--' + action,
            '--manifest', str(Path(m['stage']) / 'reviewed_manifest.json'), *extra]


def source_inventory(root, *, readonly=False):
    root = Path(root)
    entries = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink forbidden in staged source/input/output')
        if not path.is_file():
            continue
        require(path.name not in sandbox.SECRET_BASENAMES | {'.DS_Store'} and path.suffix != '.pyc' and
                not {'.git', '__pycache__'}.intersection(path.parts), 'Forbidden secret/cache/Git file in stage')
        if readonly:
            require(path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o222, 'Staged source/input must be owned and read-only')
        entries[str(path.relative_to(root))] = info(path)
    return entries


def copy_source(source, destination):
    source, destination = Path(source), Path(destination)
    require((source / 'src/evaluate/helpers.py').is_file() and (source / ENTRY).is_file(), 'Reviewed full src/infra source is missing')
    destination.mkdir(mode=0o700)
    for name in ('src', 'infra'):
        for path in sorted((source / name).rglob('*')):
            require(not path.is_symlink(), 'Source symlinks are not permitted')
            if path.is_file() and not {'__pycache__', '.git'}.intersection(path.parts) and path.suffix != '.pyc' and path.name != '.DS_Store':
                require(path.name not in sandbox.SECRET_BASENAMES, 'Forbidden credential filename in source')
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                require(info(path) == info(target), 'Source changed while staging')
                target.chmod(0o400)


def copy_input(source, target):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink() and not target.exists(), 'Unsafe source or existing input')
    before = info(source)
    shutil.copyfile(source, target)
    require(info(source) == before == info(target), 'Input changed during copy')
    target.chmod(0o400)
    return before


def authored_inputs():
    """Static qualification programs, never supplied by a model or caller."""
    prompt = [{'role': 'user', 'content': 'Implement Solution.add(a, b), and run_tests() that asserts add(2, 3) == 5.'}]
    canonical_solution = 'class Solution:\n    def add(self, a, b):\n        return a + b\n'
    benign_tests = 'def run_tests():\n    assert Solution().add(2, 3) == 5\n'
    example = {'id': 1, 'prompt': prompt, 'gt_answer': ['assert Solution().add(2, 3) == 5'],
               'setup_code': '', 'hint': 'run_tests', 'func_name': 'add', 'canonical_solution': canonical_solution,
               'prompt_metadata': {'test_func_name': 'run_tests', 'test_func_code': benign_tests}}
    prepared = {'record_id': 'authored-source', 'problem_id': 1, 'problem_split': 'configuration_validation',
                'prompt': prompt, 'completion_token_ids': [1], 'outcome_presence_class': 'authored_fixture',
                'ground_truth_correctness': True, 'regions': {}}
    programs = [canonical_solution + benign_tests,
                'class Solution:\n    def add(self, a, b):\n        return 0\n' + benign_tests]
    requests, rows = [], []
    for i, code in enumerate(programs):
        request = {'request_id': f'authored-evaluation-{i}', 'record_id': prepared['record_id'], 'problem_id': 1,
                   'problem_split': 'configuration_validation', 'condition_id': 'baseline', 'scope': 'primary',
                   'sample_index': i, 'seed': i + 1}
        requests.append(request)
        rows.append({**request, 'result': {'completion': '```python\n' + code + '```',
                     'completion_token_ids': [20 + i], 'generated_token_ids': [20 + i],
                     'fixed_completion_prefix_token_count': 0}})
    return {'mode': 'generate', 'conditions': {'baseline': {'layers': []}}, 'requests': requests}, [prepared], [example], rows


def valid_exit_status(terminal):
    kind, value = terminal.get('exit_code_kind'), terminal.get('exit_status')
    if not isinstance(value, str):
        return False
    if kind == 'exited':
        return bool(re.fullmatch(r'\d+', value)) and 0 <= int(value) <= 255
    if kind in ('killed', 'dumped'):
        # systemd EXIT_STATUS contains a signal name for these exit kinds.
        return 'SIG' + value in signal.Signals.__members__
    return False


def recovered_rows(recovery_path, generated, current_bindings, source):
    if recovery_path is None:
        return [], None
    recovery = read_json(recovery_path)
    require(recovery.get('mode') == 'reuse_completed_without_reevaluation' and recovery.get('input_hashes') == current_bindings, 'Recovery plan must bind identical exact inputs')
    collected = []
    for previous in recovery['prior_runs']:
        m = load_manifest(previous['manifest'], previous['manifest_sha256'])
        require(m['input_hashes'] == current_bindings, 'Prior evaluation inputs differ')
        require(m['runtime_versions'] == {name: importlib.metadata.version(name) for name in VERSIONS}, 'Evaluation runtime changed before recovery')
        critical = ('infra/gpu03/direction_discovery/evaluate.py', 'infra/gpu03/direction_discovery/bounded_evaluator.py',
                    'infra/gpu03/direction_discovery/sandbox.py', 'infra/gpu03/direction_discovery/solver_diagnostic.py',
                    'infra/gpu03/direction_discovery/harness_interpretability.py')
        for relative in (key for key in m['source_files'] if key.startswith('src/') or key in critical):
            require(sha256(Path(m['stage']) / 'source' / relative) == sha256(Path(source) / relative), 'Evaluation implementation changed before recovery')
        terminal = read_json(Path(m['stage']) / 'control/supervisor_exit.json')
        launched = read_json(Path(m['stage']) / 'control/service_started.json')
        require(terminal['run_token'] == m['run_token'] and terminal['manifest_sha256'] == previous['manifest_sha256'] and
                valid_exit_status(terminal) and
                terminal.get('service_result') in ('success', 'exit-code', 'signal', 'core-dump', 'timeout', 'oom-kill', 'resources') and
                re.fullmatch('[0-9a-f]{32}', terminal.get('invocation_id', '')) and
                terminal['invocation_id'] == launched.get('InvocationID'), 'Prior attempt lacks a terminal independent receipt')
        state = service_state(m['run_token'])
        require(state.get('ActiveState') in ('inactive', 'failed') and state.get('MainPID') == '0', 'Prior evaluation is not positively terminal')
        require(launched['ControlGroup'].endswith('/' + m['run_token'] + '.service'), 'Prior cgroup identity differs from exact run token')
        previous_cgroup = Path('/sys/fs/cgroup') / launched['ControlGroup'].lstrip('/')
        require(not previous_cgroup.exists() or not cgroup_processes(previous_cgroup), 'Prior evaluator processes are not released')
        journal = Path(m['stage']) / 'control/processes.jsonl'
        if journal.exists():
            require(all(not same_process(row['identity']) for row in read_jsonl(journal)), 'Prior exact-token process is alive')
        paths = [Path(m['output']) / worker['name'] / 'evaluation/records.jsonl' for worker in m['workers']]
        paths.append(Path(m['stage']) / 'input/recovered.jsonl')
        paths = [p for p in paths if p.exists()]
        require(set(previous.get('journals', {})) == {str(p.relative_to(m['stage'])) for p in paths}, 'Explicit recovery plan must bind every prior completed journal')
        for journal in paths:
            bound = previous['journals'][str(journal.relative_to(m['stage']))]
            require(not journal.is_symlink() and info(journal) == bound, 'Prior recovery journal hash changed')
            collected.extend(read_jsonl(journal))
            require(info(journal) == bound, 'Prior recovery journal changed during read')
    done = unique_rows(collected, 'explicit recovery journals')
    for key, row in done.items():
        require(key in generated, 'Unknown completed request in recovery')
        validate_evaluation_row(row, generated[key])
    return [done[key] for key in sorted(done)], recovery


def build_manifest(*, stage, run_token, source_dir, experiment_plan, phase,
                   request_plan=None, prepared_records=None, dataset=None, generation_packages=(),
                   workers=8, runtime_seconds=14400, mode='production', recovery_plan=None,
                   parent_experiment_plan=None, test_bundle=None):
    host_identity()
    stage = Path(stage)
    require(stage == ROOT / run_token and not stage.exists(), 'Builder requires a fresh exact-token stage')
    require(re.fullmatch(r'codex-eval-[a-z0-9-]{8,80}', run_token), 'Invalid evaluation run token')
    require(mode in ('production', 'authored_selftest') and 1 <= workers <= 8, 'Invalid evaluation mode/worker count')
    require(type(runtime_seconds) is int and 60 <= runtime_seconds <= 14400, 'Evaluation deadline must be within four hours')
    from infra.gpu03.direction_discovery.behavior_plan import validate_master
    master_sha = sha256(experiment_plan)
    plan = read_json(experiment_plan)
    parent = read_json(parent_experiment_plan) if parent_experiment_plan is not None else None
    parent_sha = sha256(parent_experiment_plan) if parent is not None else None
    validate_master(plan, master_sha, parent=parent, parent_sha=parent_sha)
    require(plan['no_training'] is True and plan['host'] == HOST, 'Wrong experiment authority')
    if mode == 'production':
        rp = read_json(request_plan)
        require(rp.get('phase') != 'untouched_test_bundle' or test_bundle is not None,
                'Full test evaluation requires an explicit bundle reference')
        if test_bundle is not None:
            require(recovery_plan is None, 'Test bundle evaluation recovery requires a separately reviewed implementation')
            test_bundle_context(test_bundle, rp, prepared_sha=sha256(prepared_records))
        pr, ds = list(read_jsonl(prepared_records)), list(read_jsonl(dataset))
        require(rp.get('master_plan_sha256') == master_sha, 'Behavior request plan belongs to another experiment version')
        raw_rows, proofs = collect_generation_packages(generation_packages, request_plan=rp, prepared_rows=pr,
                                                       master_sha=master_sha, test_bundle=test_bundle)
        generated = validate_inputs(raw_rows, rp, pr, ds)
        current = {'request_plan': sha256(request_plan), 'prepared_records': sha256(prepared_records),
                   'dataset': sha256(dataset), 'experiment_plan': master_sha,
                   'generations': hashlib.sha256((''.join(canonical(generated[k]) + '\n' for k in sorted(generated))).encode()).hexdigest()}
    else:
        require(not generation_packages and recovery_plan is None and test_bundle is None,
                'Authored selftest cannot accept scientific inputs or recovery')
        rp, pr, ds, raw_rows = authored_inputs()
        generated = validate_inputs(raw_rows, rp, pr, ds)
        proofs, current = [{'authored_static_fixtures': True}], {}
        workers = min(workers, len(generated))
    logical_refs = [{key: proof[key] for key in ('kind', 'logical_round')} for proof in proofs
                    if proof.get('kind') == 'recovered_finalist_round0']
    logical_ref = logical_refs[0] if logical_refs else None
    require(len(logical_refs) <= 1 and (logical_ref is None or
            mode == 'production' and test_bundle is None and recovery_plan is None and len(generated) == 740),
            'Recovered finalist requires full740 first evaluation without an evaluation recovery')
    # Complete package/request identity validation happens before staging/sharding.
    stage.mkdir(mode=0o700)
    copy_source(source_dir, stage / 'source')
    inputs = stage / 'input'; inputs.mkdir(mode=0o700)
    copy_input(experiment_plan, inputs / 'experiment_plan.json')
    if parent_experiment_plan is not None:
        copy_input(parent_experiment_plan, inputs / 'parent_experiment_plan.json')
    if mode == 'production':
        for path, name in ((request_plan, 'request_plan.json'), (prepared_records, 'prepared_records.jsonl'), (dataset, 'dataset.jsonl')):
            copy_input(path, inputs / name)
    else:
        write_json(inputs / 'request_plan.json', rp)
        write_jsonl(inputs / 'prepared_records.jsonl', pr)
        write_jsonl(inputs / 'dataset.jsonl', ds)
    write_jsonl(inputs / 'generations.jsonl', (generated[key] for key in sorted(generated)))
    current = {'request_plan': sha256(inputs / 'request_plan.json'), 'prepared_records': sha256(inputs / 'prepared_records.jsonl'),
               'dataset': sha256(inputs / 'dataset.jsonl'), 'experiment_plan': master_sha, 'generations': sha256(inputs / 'generations.jsonl')}
    if test_bundle is not None:
        current['test_bundle'] = test_bundle['sha256']
    if logical_ref is not None:
        current['recovered_finalist_round'] = logical_ref['logical_round']['sha256']
    recovered, recovery = recovered_rows(recovery_plan, generated, current, stage / 'source')
    write_jsonl(inputs / 'recovered.jsonl', recovered)
    if recovery is not None:
        copy_input(recovery_plan, inputs / 'recovery_plan.json')
    write_json(inputs / 'generation_verification.json', proofs)
    recovered_ids = {row['request_id'] for row in recovered}
    remaining = sorted(set(generated) - recovered_ids)
    worker_specs = []
    for index in range(min(workers, len(remaining))):
        ids = remaining[index::min(workers, len(remaining))]
        name = f'worker_{index:02d}'
        write_jsonl(inputs / (name + '.jsonl'), (generated[key] for key in ids))
        worker_specs.append({'name': name, 'cpus': [64 + 2 * index, 65 + 2 * index], 'request_ids': ids,
                             'input': name + '.jsonl', 'requests': len(ids), 'sandbox_workers': 2})
    for path in inputs.iterdir():
        path.chmod(0o400)
    m = {'schema_version': 1, 'purpose': 'direction_discovery_cpu_evaluation', 'run_token': run_token,
         'phase': phase, 'mode': mode, 'host': HOST, 'owner': OWNER, 'python': PYTHON,
         'stage': str(stage), 'output': str(stage / 'results'), 'controller_cpu': 62, 'verifier_cpu': 63,
         'workers': worker_specs, 'request_ids': sorted(generated), 'recovered_request_ids': sorted(recovered_ids),
         'input_hashes': current, 'source_files': source_inventory(stage / 'source', readonly=True),
         'input_files': source_inventory(inputs, readonly=True),
         'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
         'venv_bindings': {str(Path(PYTHON).parent.parent / 'pyvenv.cfg'): info(Path(PYTHON).parent.parent / 'pyvenv.cfg'),
                           str(Path(PYTHON).resolve()): info(Path(PYTHON).resolve())},
         'authorization': plan['authorization'],
         'scientific': {'training': False, 'generation': False, 'generated_code_outside_sandbox': False,
                        'master_plan_sha256': master_sha, 'parent_plan_sha256': parent_sha, 'taxonomy_changed': False,
                        'max_jobs_per_outer': 2, 'maximum_outer_workers': 8,
                        'timeout_seconds': 3, 'memory_per_evaluator_mib': 1024,
                        'output_limit_bytes': 1048576, 'recovery_requires_explicit_plan': True},
         'limits': {'runtime_seconds': runtime_seconds, 'systemd_runtime_seconds': runtime_seconds + 90,
                    'memory_max_bytes': 32 * GIB, 'max_rss_bytes': 24 * GIB, 'min_available_ram_bytes': 192 * GIB,
                    'min_free_disk_bytes': 64 * GIB, 'tasks_max': 512, 'max_log_bytes': 8 * 1024 ** 2,
                    'max_host_load1': 120, 'monitor_seconds': INTERVAL}}
    m['scientific']['correctness_diagnostics'] = diagnostic_policy(m['source_files'])
    if test_bundle is not None:
        m['scientific']['test_bundle'] = test_bundle
    if logical_ref is not None:
        m['scientific']['recovered_finalist_round'] = logical_ref
    m['command'] = command(m, 'supervise')
    for worker in m['workers']:
        worker['command'] = command(m, 'worker', '--worker-name', worker['name'])
        worker['python_args'] = ['-B', '-m', 'infra.gpu03.direction_discovery.evaluate', '--requests', '/input/' + worker['input'],
                                 '--dataset', '/input/dataset.jsonl', '--prepared', '/input/prepared_records.jsonl', '--output', '/output/evaluation']
    path = stage / 'reviewed_manifest.json'
    write_json(path, m); path.chmod(0o400)
    digest = sha256(path)
    load_manifest(path, digest)
    return {'manifest': str(path), 'manifest_sha256': digest, 'command': command(m, 'launch', '--manifest-sha256', digest),
            'output': m['output'], 'mode': mode, 'requests': len(generated), 'recovered': len(recovered),
            'workers': len(worker_specs), 'maximum_evaluator_children': 2 * len(worker_specs), 'limits': m['limits']}


def diagnostic_policy(source_files):
    from .solver_diagnostic import POLICY
    relative = 'infra/gpu03/direction_discovery/solver_diagnostic.py'
    require(relative in source_files, 'New evaluation requires the isolated solver diagnostic source')
    return {'policy': POLICY, 'implementation_sha256': source_files[relative]['sha256'],
            'max_additional_calls_per_completion': 1}


def load_manifest(path, digest, *, check_files=True):
    path = Path(path)
    require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha256(path) == digest, 'Manifest digest mismatch')
    m = read_json(path)
    require(m['schema_version'] == 1 and m['purpose'] == 'direction_discovery_cpu_evaluation', 'Wrong evaluation manifest')
    require(m['host'] == HOST and m['owner'] == OWNER and m['python'] == PYTHON, 'Wrong host/user/runtime')
    token, stage = m['run_token'], Path(m['stage'])
    require(re.fullmatch(r'codex-eval-[a-z0-9-]{8,80}', token) and stage == ROOT / token and
            path == stage / 'reviewed_manifest.json' and m['output'] == str(stage / 'results'), 'Unsafe token-scoped paths')
    require(not stage.is_symlink() and stage.resolve() == stage and stage.stat().st_uid == os.getuid(), 'Unsafe stage identity')
    require(m['mode'] in ('production', 'authored_selftest') and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', m['phase']), 'Wrong phase/mode')
    require(m['controller_cpu'] == 62 and m['verifier_cpu'] == 63 and len(m['workers']) <= 8, 'CPU assignment changed')
    ids = m['request_ids']; recovered = m['recovered_request_ids']
    require(isinstance(ids, list) and 0 < len(ids) <= MAX_REQUESTS and ids == sorted(set(ids)), 'Invalid total request IDs')
    require(recovered == sorted(set(recovered)) and set(recovered) <= set(ids), 'Invalid recovered request IDs')
    assigned = []
    for index, w in enumerate(m['workers']):
        require(w['name'] == f'worker_{index:02d}' and w['cpus'] == [64 + 2 * index, 65 + 2 * index] and
                w['sandbox_workers'] == 2 and w['input'] == w['name'] + '.jsonl' and
                w['requests'] == len(w['request_ids']) > 0, 'Unsafe worker allocation')
        require(w['command'] == command(m, 'worker', '--worker-name', w['name']), 'Worker command drift')
        require(w['python_args'] == ['-B', '-m', 'infra.gpu03.direction_discovery.evaluate', '--requests', '/input/' + w['input'],
                                    '--dataset', '/input/dataset.jsonl', '--prepared', '/input/prepared_records.jsonl', '--output', '/output/evaluation'], 'Unsafe evaluator command')
        assigned.extend(w['request_ids'])
    require(len(assigned) == len(set(assigned)) and set(assigned).isdisjoint(recovered) and set(assigned + recovered) == set(ids), 'Evaluation shard coverage differs from full immutable plan')
    require(m['command'] == command(m, 'supervise'), 'Supervisor command drift')
    lim = m['limits']
    require(type(lim['runtime_seconds']) is int and 60 <= lim['runtime_seconds'] <= 14400 and
            lim['systemd_runtime_seconds'] == lim['runtime_seconds'] + 90 and
            lim['memory_max_bytes'] == 32 * GIB and lim['max_rss_bytes'] == 24 * GIB and
            lim['min_available_ram_bytes'] >= 192 * GIB and lim['min_free_disk_bytes'] >= 64 * GIB and
            lim['tasks_max'] == 512 and lim['max_log_bytes'] <= 8 * 1024**2 and
            lim['max_host_load1'] <= 120 and lim['monitor_seconds'] == INTERVAL, 'Reviewed resource bounds changed')
    test_bundle = m['scientific'].get('test_bundle')
    logical_ref = m['scientific'].get('recovered_finalist_round')
    extra_scientific = {'test_bundle': test_bundle} if test_bundle is not None else {}
    if logical_ref is not None:
        extra_scientific['recovered_finalist_round'] = logical_ref
    if 'correctness_diagnostics' in m['scientific']:
        extra_scientific['correctness_diagnostics'] = diagnostic_policy(m['source_files'])
    require(m['scientific'] == {'training': False, 'generation': False, 'generated_code_outside_sandbox': False,
             'master_plan_sha256': m['input_hashes']['experiment_plan'], 'parent_plan_sha256': m['scientific'].get('parent_plan_sha256'),
             'taxonomy_changed': False, 'max_jobs_per_outer': 2,
             'maximum_outer_workers': 8, 'timeout_seconds': 3, 'memory_per_evaluator_mib': 1024,
             'output_limit_bytes': 1048576, 'recovery_requires_explicit_plan': True, **extra_scientific}, 'Scientific/sandbox policy drift')
    require((test_bundle is None and 'test_bundle' not in m['input_hashes']) or
            (m['mode'] == 'production' and test_bundle is not None and
             m['input_hashes'].get('test_bundle') == test_bundle['sha256'] and not recovered),
            'Test bundle input provenance or recovery policy changed')
    require((logical_ref is None and 'recovered_finalist_round' not in m['input_hashes']) or
            (m['mode'] == 'production' and test_bundle is None and not recovered and len(ids) == 740 and
             logical_ref.get('kind') == 'recovered_finalist_round0' and
             m['input_hashes'].get('recovered_finalist_round') == logical_ref['logical_round']['sha256']),
            'Recovered finalist input provenance or complete-validation boundary changed')
    require(m['authorization'] and set(m['runtime_versions']) == set(VERSIONS), 'Missing authority/runtime bindings')
    if check_files:
        require(source_inventory(stage / 'source', readonly=True) == m['source_files'], 'Source inventory/hash drift')
        require(source_inventory(stage / 'input', readonly=True) == m['input_files'], 'Input inventory/hash drift')
        for filename, expected in m['venv_bindings'].items():
            require(info(filename) == expected, 'Pinned Python runtime changed')
        from infra.gpu03.direction_discovery.behavior_plan import validate_master
        master_path = stage / 'input/experiment_plan.json'
        parent_path = stage / 'input/parent_experiment_plan.json'
        master_sha = sha256(master_path)
        parent = read_json(parent_path) if parent_path.exists() else None
        parent_sha = sha256(parent_path) if parent is not None else None
        require(master_sha == m['input_hashes']['experiment_plan'] and parent_sha == m['scientific']['parent_plan_sha256'], 'Master plan lineage changed')
        validate_master(read_json(master_path), master_sha, parent=parent, parent_sha=parent_sha)
        for name, filename in (('request_plan', 'request_plan.json'), ('prepared_records', 'prepared_records.jsonl'),
                               ('dataset', 'dataset.jsonl'), ('generations', 'generations.jsonl')):
            require(sha256(stage / 'input' / filename) == m['input_hashes'][name], 'Frozen input hash binding changed')
        inputs = stage / 'input'
        request_plan = read_json(inputs / 'request_plan.json')
        generation_proofs = read_json(inputs / 'generation_verification.json')
        require(bool(logical_ref) == any(p.get('kind') == 'recovered_finalist_round0' for p in generation_proofs),
                'Recovered finalist evaluation lost its explicit logical-round provenance')
        if logical_ref is not None:
            context = recovered_finalist_metadata(generation_proofs, request_plan=request_plan, master_sha=master_sha,
                                                 stored_proofs=generation_proofs)
            require(context['reference'] == logical_ref, 'Evaluation logical-round binding differs from collected proof')
        require(request_plan.get('phase') != 'untouched_test_bundle' or test_bundle is not None,
                'Full test evaluation lost its explicit bundle reference')
        if test_bundle is not None:
            test_bundle_context(test_bundle, request_plan, prepared_sha=m['input_hashes']['prepared_records'])
            generation_proofs = read_json(inputs / 'generation_verification.json')
            test_generation_metadata(generation_proofs, request_plan=request_plan, test_bundle=test_bundle,
                                     master_sha=master_sha, stored_proofs=generation_proofs)
        generated = validate_inputs(list(read_jsonl(inputs / 'generations.jsonl')), read_json(inputs / 'request_plan.json'),
                                    list(read_jsonl(inputs / 'prepared_records.jsonl')), list(read_jsonl(inputs / 'dataset.jsonl')))
        if m['mode'] == 'production':
            require(read_json(inputs / 'request_plan.json')['master_plan_sha256'] == master_sha, 'Evaluation request plan version mismatch')
        require(set(generated) == set(ids), 'Manifest omitted a full problem/sample from the immutable plan')
        for w in m['workers']:
            rows = expected_coverage(read_jsonl(inputs / w['input']), w['request_ids'], 'worker shard')
            require(all(row == generated[key] for key, row in rows.items()), 'Shard generation differs from exact frozen input')
        old = expected_coverage(read_jsonl(inputs / 'recovered.jsonl'), recovered, 'frozen recovery')
        for key, row in old.items():
            validate_evaluation_row(row, generated[key])
    return m


def validate_evaluation_row(row, generation, *, require_diagnostic=False):
    require(all(canonical(row.get(k)) == canonical(generation[k]) for k in IDENTITY_FIELDS), 'Evaluation identity/seed drift')
    expected = hashlib.sha256(evaluate.canonical(generation).encode()).hexdigest()
    require(row.get('generation_sha256') == expected and row.get('generation') == generation['result'], 'Evaluation refers to another generated completion')
    require(row.get('evaluation_status') in ('evaluated', 'suspicious_or_unknown'), 'Infrastructure failure cannot be treated as a completed valid evaluation')
    require(set(row.get('metrics', {})) == set(evaluate.BINARY_METRICS) | {'completion_length'}, 'Incomplete evaluation metrics')
    require(all(row['metrics'][key] is None or type(row['metrics'][key]) is bool for key in evaluate.BINARY_METRICS), 'Unknown labels must be explicit nulls')
    require(type(row['metrics']['completion_length']) is int and row['metrics']['completion_length'] == len(generation['result']['completion_token_ids']), 'Evaluation completion length mismatch')
    if require_diagnostic or 'correctness_diagnostics' in row:
        from .solver_diagnostic import validate
        validate(row.get('correctness_diagnostics'), row['metrics']['ground_truth_correctness'])


def verify_worker(m, worker):
    root = Path(m['output']) / worker['name']
    success = read_json(root / 'evaluation/SUCCESS.json')
    wrapper = read_json(root / 'outer_receipt.json')
    require(wrapper.get('status') == 'succeeded' and wrapper.get('run_token') == m['run_token'] and
            wrapper.get('worker_name') == worker['name'] and wrapper.get('returncode') == 0 and
            wrapper.get('success_sha256') == sha256(root / 'evaluation/SUCCESS.json'), 'Outer worker receipt is not successful')
    require(success.get('status') == 'succeeded' and success.get('records') == worker['requests'] and
            success.get('records_sha256') == sha256(root / 'evaluation/records.jsonl'), 'Evaluator count/hash/status mismatch')
    cfg = success['input_config']
    policy = m['scientific'].get('correctness_diagnostics')
    if policy is not None:
        require(cfg.get('correctness_diagnostics') == policy, 'Solver diagnostic policy/source drift')
    inputs = Path(m['stage']) / 'input'
    require(cfg['requests_sha256'] == sha256(inputs / worker['input']) and cfg['dataset_sha256'] == m['input_hashes']['dataset'] and
            cfg['prepared_sha256'] == m['input_hashes']['prepared_records'] and cfg['workers'] == 2 and
            cfg['timeout_seconds'] == 3 and cfg['memory_per_worker_mib'] == 1024 and cfg['training'] is False and cfg['generation'] is False and
            cfg['implementation_sha256'] == m['source_files']['infra/gpu03/direction_discovery/evaluate.py']['sha256'], 'Evaluator source/input/configuration drift')
    generated = unique_rows(read_jsonl(inputs / worker['input']), 'worker inputs')
    rows = expected_coverage(read_jsonl(root / 'evaluation/records.jsonl'), worker['request_ids'], 'worker evaluations')
    for key, row in rows.items():
        validate_evaluation_row(row, generated[key], require_diagnostic=policy is not None)
    return rows


def worker(path, digest, name):
    m = load_manifest(path, digest)
    w = next((w for w in m['workers'] if w['name'] == name), None)
    require(w is not None and os.sched_getaffinity(0) == set(w['cpus']), 'Wrong worker/CPU affinity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Evaluation worker must hide CUDA')
    destination = Path(m['output']) / name
    destination.mkdir(mode=0o700, exist_ok=False)
    def interrupted(_sig, _frame):
        raise RuntimeError('Evaluation worker interrupted; bounded outer transport cleanup follows')
    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = sandbox.run_outer(source_dir=Path(m['stage']) / 'source', input_dir=Path(m['stage']) / 'input',
                    output_dir=destination, venv_dir=Path(PYTHON).parent.parent,
                    python_args=w['python_args'], workers=2, wall_timeout=m['limits']['runtime_seconds'])
        write_json(destination / 'outer_transport.json', result)
        require(result['returncode'] == 0, 'Outer sandbox evaluator exited nonzero; append-only output retained')
        write_json(destination / 'outer_receipt.json', {'status': 'succeeded', 'run_token': m['run_token'], 'worker_name': name,
                   'returncode': 0, 'success_sha256': sha256(destination / 'evaluation/SUCCESS.json'),
                   'process_group': result['process_group'], 'output_bytes': result['output_bytes']})
        verify_worker(m, w)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def own_cgroup(token):
    paths = [line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(paths) == 1 and paths[0].endswith('/' + token + '.service') and '..' not in Path(paths[0]).parts, 'Supervisor lacks its exact-token systemd cgroup')
    path = Path('/sys/fs/cgroup') / paths[0].lstrip('/')
    require(path.is_dir(), 'Exact-token cgroup is unavailable')
    return path


def cgroup_processes(group):
    result = []
    for file in Path(group).rglob('cgroup.procs'):
        for value in file.read_text().splitlines():
            current = process_info(int(value))
            if current is not None:
                require(current['uid'] == os.getuid(), 'Foreign UID in exact-token evaluation cgroup')
                result.append(current)
    return {row['pid']: row for row in result}


def resource_check(m, group=None):
    available = next((int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')), None)
    free = shutil.disk_usage(m['stage']).free
    processes = cgroup_processes(group) if group is not None else {os.getpid(): process_info(os.getpid())}
    rss = sum(p['rss_bytes'] for p in processes.values())
    load = os.getloadavg()[0]
    lim = m['limits']
    require(available is not None and available >= lim['min_available_ram_bytes'], 'Available host RAM below 192 GiB floor')
    require(free >= lim['min_free_disk_bytes'], 'Scratch free space below 64 GiB floor')
    require(rss <= lim['max_rss_bytes'] and len(processes) <= lim['tasks_max'], 'Owned evaluation RSS/process bound exceeded')
    require(load <= lim['max_host_load1'], 'Host CPU load exceeds reviewed bound')
    for log in (Path(m['stage']) / 'control').glob('*.log'):
        require(log.stat().st_size <= lim['max_log_bytes'], 'Evaluation supervisor/worker log exceeded bound')
    return {'at': time.time(), 'available_ram_bytes': available, 'free_disk_bytes': free,
            'rss_bytes': rss, 'host_load1': load, 'processes': list(processes.values())}


def terminate_owned(processes):
    for p, identity in processes:
        if p.poll() is None:
            require(same_process(identity) and os.getpgid(p.pid) == p.pid, 'Owned process identity changed')
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for p, identity in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                require(same_process(identity), 'Owned PID changed before forced cleanup')
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)


def release(group, processes):
    terminate_owned(processes)
    deadline = time.monotonic() + 10
    while True:
        remaining = {pid: row for pid, row in cgroup_processes(group).items() if pid != os.getpid()}
        if not remaining:
            return {'verified': True, 'remaining_children': [], 'cgroup': str(group), 'at': time.time()}
        require(time.monotonic() < deadline, 'Owned evaluator descendants remain; independent cgroup cleanup required')
        time.sleep(.25)


def produced_manifest(output):
    output = Path(output)
    return {'algorithm': 'sha256', 'files': {key: value for key, value in source_inventory(output).items() if key != 'artifact_manifest.json'}}


def merge_evaluations(m):
    inputs = Path(m['stage']) / 'input'
    generated = unique_rows(read_jsonl(inputs / 'generations.jsonl'), 'all generation inputs')
    collected = list(read_jsonl(inputs / 'recovered.jsonl'))
    for w in m['workers']:
        collected.extend(verify_worker(m, w).values())
    rows = expected_coverage(collected, m['request_ids'], 'merged evaluations before metrics')
    for key, row in rows.items():
        validate_evaluation_row(row, generated[key], require_diagnostic='correctness_diagnostics' in m['scientific'])
    if m['mode'] == 'authored_selftest':
        require(rows['authored-evaluation-0']['metrics']['ground_truth_correctness'] is True and
                rows['authored-evaluation-1']['metrics']['ground_truth_correctness'] is False, 'Authored evaluator success/failure path did not match known answers')
    output = Path(m['output'])
    write_jsonl(output / 'evaluations.jsonl', (rows[key] for key in sorted(rows)))
    if 'correctness_diagnostics' in m['scientific']:
        from .correctness_report import summarize
        write_json(output / 'correctness_diagnostics.json', summarize(list(rows.values())))
    return {'records': len(rows), 'problem_ids': sorted({str(row['problem_id']) for row in rows.values()}),
            'evaluation_status': dict(Counter(row['evaluation_status'] for row in rows.values())),
            'exact_request_coverage': True, 'request_ids_sha256': hashlib.sha256(canonical(sorted(rows)).encode()).hexdigest(),
            'evaluations_sha256': sha256(output / 'evaluations.jsonl')}


def assemble(path, digest):
    """Separate CPU63 producer so the controller keeps monitoring every five seconds."""
    m = load_manifest(path, digest)
    require(os.sched_getaffinity(0) == {63}, 'Assembly requires reserved verifier CPU63')
    summary = merge_evaluations(m)
    output = Path(m['output'])
    write_json(output / 'producer_summary.json', {'status': 'succeeded', 'run_token': m['run_token'], 'manifest_sha256': digest,
               'mode': m['mode'], 'phase': m['phase'], 'recovered_records': len(m['recovered_request_ids']), **summary})
    write_json(output / 'artifact_manifest.json', produced_manifest(output))
    return summary


def verify_produced(path, digest):
    m = load_manifest(path, digest)
    output = Path(m['output'])
    require(not (output / 'FAILURE.json').exists(), 'Failed evaluation campaign cannot verify')
    require(read_json(output / 'artifact_manifest.json') == produced_manifest(output), 'Produced evaluation artifact inventory/hash mismatch')
    summary = read_json(output / 'producer_summary.json')
    require(summary['run_token'] == m['run_token'] and summary['manifest_sha256'] == digest and summary['status'] == 'succeeded' and
            summary['records'] == len(m['request_ids']) and summary['exact_request_coverage'] is True, 'Evaluation producer summary mismatch')
    generated = unique_rows(read_jsonl(Path(m['stage']) / 'input/generations.jsonl'), 'all generation inputs')
    rows = expected_coverage(read_jsonl(output / 'evaluations.jsonl'), m['request_ids'], 'independently verified merged evaluation IDs')
    expected = unique_rows(read_jsonl(Path(m['stage']) / 'input/recovered.jsonl'), 'prior evaluation rows')
    for w in m['workers']:
        expected.update(verify_worker(m, w))
    require(rows == expected, 'Merged evaluations differ from immutable worker journals')
    for key, row in rows.items():
        validate_evaluation_row(row, generated[key], require_diagnostic='correctness_diagnostics' in m['scientific'])
    if 'correctness_diagnostics' in m['scientific']:
        from .correctness_report import summarize
        require(read_json(output / 'correctness_diagnostics.json') == summarize(list(rows.values())),
                'Independent solver correctness report recomputation differs')
    return {'status': 'verified', 'run_token': m['run_token'], 'manifest_sha256': digest, 'records': len(rows),
            'exact_request_coverage': True, 'artifact_manifest_sha256': sha256(output / 'artifact_manifest.json'),
            'evaluations_sha256': sha256(output / 'evaluations.jsonl')}


def supervise(path, digest):
    host_identity()
    m = load_manifest(path, digest)
    require({name: importlib.metadata.version(name) for name in VERSIONS} == m['runtime_versions'], 'Dependency runtime drift')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Controller must be CPU-only')
    allocation = {62, 63}.union(*(set(w['cpus']) for w in m['workers']))
    require(allocation <= os.sched_getaffinity(0), 'Reviewed evaluation CPUs are unavailable')
    group = own_cgroup(m['run_token'])
    output, control = Path(m['output']), Path(m['stage']) / 'control'
    require(not output.exists(), 'Prior output preserved; freeze an explicit recovery plan instead of restarting')
    resource_check(m, group)
    output.mkdir(mode=0o700)
    os.sched_setaffinity(0, {62})
    started = time.monotonic(); processes, logs, locks = [], [], []
    def stop(_sig, _frame):
        raise RuntimeError('Evaluation supervisor interrupted')
    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    def guard():
        require(time.monotonic() - started < m['limits']['runtime_seconds'], 'Evaluation campaign deadline reached')
        append(control / 'resources.jsonl', resource_check(m, group))
    def spawn(argv, cpus, name):
        guard()
        log = (control / (name + '.log')).open('xb'); logs.append(log)
        argv = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), '/usr/bin/nice', '-n', '15', *argv]
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             env=cpu_environment(), start_new_session=True, cwd=Path(m['stage']) / 'source')
        identity = process_info(p.pid)
        require(identity is not None, 'Child disappeared before process identity capture')
        processes.append((p, identity))
        append(control / 'processes.jsonl', {'name': name, 'identity': identity, 'argv': argv})
        return p
    def wait(children):
        while any(p.poll() is None for p in children):
            guard()
            require(all(p.poll() in (None, 0) for p in children), 'Evaluation child exited nonzero; partial journals retained')
            time.sleep(INTERVAL)
        require(all(p.returncode == 0 for p in children), 'Evaluation child exited nonzero')
        guard()
    try:
        for cpu in sorted(allocation):
            fd = os.open(ROOT / f'.codex-evaluation-cpu-{cpu}.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handle = os.fdopen(fd, 'a+'); locks.append(handle)
            require(os.fstat(fd).st_uid == os.getuid() and stat.S_ISREG(os.fstat(fd).st_mode), 'Unsafe CPU lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = [spawn(w['command'] + ['--manifest-sha256', digest], w['cpus'], w['name']) for w in m['workers']]
        wait(children)
        released = release(group, processes)
        write_json(output / 'worker_release.json', released)
        producer = spawn(command(m, 'assemble', '--manifest-sha256', digest), [63], 'assembler')
        wait([producer])
        verifier = spawn(command(m, 'verify-produced', '--manifest-sha256', digest, '--verification-output', str(control / 'producer_verification.json')), [63], 'verifier')
        wait([verifier])
        proof = read_json(control / 'producer_verification.json')
        require(proof['status'] == 'verified' and proof['records'] == len(m['request_ids']), 'Independent producer verification failed')
        final_release = release(group, processes)
        guard()
        write_json(control / 'final_release.json', final_release)
        write_json(control / 'SUCCESS.json', {**proof, 'process_release_verified': True, 'worker_exit_codes': [p.returncode for p, _ in processes]})
    except BaseException as error:
        cleanup = None
        try:
            cleanup = release(group, processes)
        except BaseException as failure:
            cleanup = {'verified': False, 'error': str(failure)}
        write_json(output / 'FAILURE.json', {'status': 'failed', 'run_token': m['run_token'], 'manifest_sha256': digest,
                   'error': str(error), 'type': type(error).__name__, 'partial_journals_preserved': True, 'cleanup': cleanup})
        raise
    finally:
        for log in logs:
            log.close()
        for lock in locks:
            lock.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def service_state(token):
    result = subprocess.run(['systemctl', '--user', 'show', token + '.service',
             '--property=ActiveState,SubState,MainPID,ControlGroup,RuntimeMaxUSec,MemoryMax,TasksMax,KillMode,InvocationID'],
             capture_output=True, text=True, timeout=15)
    require(result.returncode == 0, 'Cannot read independent service state')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def receipt(path, digest):
    m = load_manifest(path, digest, check_files=False)
    control = Path(m['stage']) / 'control'
    write_json(control / 'supervisor_exit.json', {'run_token': m['run_token'], 'manifest_sha256': digest,
               'service_result': os.environ.get('SERVICE_RESULT', 'unknown'), 'exit_code_kind': os.environ.get('EXIT_CODE', 'unknown'),
               'exit_status': os.environ.get('EXIT_STATUS', 'unknown'), 'invocation_id': os.environ.get('INVOCATION_ID', 'unknown'),
               'success_present': (control / 'SUCCESS.json').is_file(), 'failure_present': (Path(m['output']) / 'FAILURE.json').is_file(), 'at': time.time()})


def launch(path, digest):
    host_identity()
    m = load_manifest(path, digest)
    require(not Path(m['output']).exists(), 'Prior evaluation output requires explicit reviewed recovery')
    resource_check(m)
    control = Path(m['stage']) / 'control'; control.mkdir(mode=0o700, exist_ok=False)
    write_json(control / 'launch_intent.json', {'run_token': m['run_token'], 'manifest_sha256': digest, 'at': time.time(),
               'authorization': m['authorization'], 'resource_state': resource_check(m), 'command': m['command']})
    # Preserve only systemd's documented exit metadata for the receipt process.
    stop = ['/usr/bin/env', '-i', *[k + '=' + v for k, v in cpu_environment().items()],
            'SERVICE_RESULT=${SERVICE_RESULT}', 'EXIT_CODE=${EXIT_CODE}', 'EXIT_STATUS=${EXIT_STATUS}', 'INVOCATION_ID=${INVOCATION_ID}',
            *command(m, 'receipt', '--manifest-sha256', digest)]
    argv = ['systemd-run', '--user', '--quiet', '--unit', m['run_token'], '--service-type=exec',
            '--property=RuntimeMaxSec=' + str(m['limits']['systemd_runtime_seconds']), '--property=TimeoutStopSec=60',
            '--property=KillMode=control-group', '--property=MemoryMax=' + str(m['limits']['memory_max_bytes']),
            '--property=TasksMax=512', '--property=LimitNOFILE=4096', '--property=UMask=0077',
            '--property=ExecStopPost=' + shlex.join(stop), '--property=StandardOutput=append:' + str(control / 'supervisor.log'),
            '--property=StandardError=inherit', *clean_command(m['command'] + ['--manifest-sha256', digest])]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    write_json(control / 'launch_result.json', {'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0, 'Independent evaluation service launch failed')
    try:
        state = service_state(m['run_token'])
        require(state.get('ActiveState') == 'active' and state.get('SubState') == 'running' and int(state.get('MainPID', 0)) > 0 and
                state.get('ControlGroup', '').endswith('/' + m['run_token'] + '.service') and state.get('KillMode') == 'control-group' and
                state.get('RuntimeMaxUSec') not in ('infinity', '0', '', None) and
                int(state.get('MemoryMax', 0)) == 32 * GIB and int(state.get('TasksMax', 0)) == 512 and
                re.fullmatch('[0-9a-f]{32}', state.get('InvocationID', '')), 'Independent CPU deadline/cgroup/start state not positively verified')
    except BaseException as error:
        stopped = subprocess.run(['systemctl', '--user', 'stop', m['run_token'] + '.service'], capture_output=True, text=True, timeout=80)
        write_json(control / 'startup_verification_failure.json', {'error': str(error), 'exact_unit_stop_returncode': stopped.returncode})
        raise
    write_json(control / 'service_started.json', state)
    return {'status': 'launched', 'run_token': m['run_token'], 'manifest_sha256': digest, 'state': state}


def verify(path, digest):
    m = load_manifest(path, digest)
    control = Path(m['stage']) / 'control'
    status, started = read_json(control / 'supervisor_exit.json'), read_json(control / 'service_started.json')
    require(status.get('run_token') == m['run_token'] and status.get('manifest_sha256') == digest and
            status.get('service_result') == 'success' and status.get('exit_code_kind') == 'exited' and status.get('exit_status') == '0' and
            status.get('invocation_id') == started['InvocationID'] and re.fullmatch('[0-9a-f]{32}', status.get('invocation_id', '')) and
            status.get('success_present') is True and status.get('failure_present') is False, 'Independent evaluation exit receipt is not successful')
    success = read_json(control / 'SUCCESS.json')
    proof = verify_produced(path, digest)
    require(all(success.get(k) == v for k, v in proof.items()) and success.get('process_release_verified') is True and
            success.get('worker_exit_codes') == [0] * (len(m['workers']) + 2), 'Independent final summary differs from verified producer')
    final = read_json(control / 'final_release.json')
    require(final.get('verified') is True and final.get('remaining_children') == [], 'Final CPU release is unverified')
    for row in read_jsonl(control / 'processes.jsonl'):
        require(not same_process(row['identity']), 'Exact-token child process remains alive')
    cgroup = Path('/sys/fs/cgroup') / started['ControlGroup'].lstrip('/')
    require(not cgroup.exists() or not cgroup_processes(cgroup), 'Evaluation cgroup still contains processes')
    return {**proof, 'process_release_verified': True, 'mode': m['mode'], 'evaluations': str(Path(m['output']) / 'evaluations.jsonl')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('build', 'launch', 'supervise', 'worker', 'assemble', 'verify-produced', 'verify', 'receipt'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--spec')
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--worker-name')
    parser.add_argument('--verification-output')
    args = parser.parse_args()
    if args.build:
        result = build_manifest(**read_json(args.spec))
    else:
        require(args.manifest and args.manifest_sha256, 'Manifest path and exact SHA-256 required')
        if args.worker:
            result = worker(args.manifest, args.manifest_sha256, args.worker_name)
        else:
            operation = launch if args.launch else supervise if args.supervise else assemble if args.assemble else verify_produced if args.verify_produced else verify if args.verify else receipt
            result = operation(args.manifest, args.manifest_sha256)
    if args.verification_output:
        write_json(args.verification_output, result)
    if result is not None:
        print(canonical(result), flush=True)


if __name__ == '__main__':
    main()
