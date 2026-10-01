"""Bounded raw-only extraction of the frozen ordinary500 population on H100.

Reuses the qualified TF controller's ownership, resource and cleanup helpers.
Raw files live on the existing NVMe; metadata lives under outputs for backup.
No generation, cloud operations, deltas, fitting, or automatic next-stage launch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import h100_tf_run as core
from . import h100_supervisor as shared

HERE = 'infra/gpu03/direction_discovery/h100_broader_raw_run.py'
WORKER = 'infra.gpu03.direction_discovery.broader_raw_worker'
RAWBASE = Path('/opt/dlami/nvme/h100-workspace')
PROTOCOL = 'h100_broader500_raw_2688_v1'
PARENT_PLAN_SHA = 'c3fec8b94cc061e3861d09b3aab2efe25f18efe143f0f7b88dffc786f2009869'
ORIGINAL_RAW_NAME = 'broader-pca-20260908-203000'
ORIGINAL_TOKEN = 'codex-h100-broader-raw-20260908-203000'
RECOVERY_RAW_NAME = 'broader-pca-recovery-20260908-205100'
RECOVERY_TOKEN = 'codex-h100-broader-raw-recovery-20260908-205100'
require = core.require


def raw_path(value):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(RAWBASE)
            and p != RAWBASE, 'raw path outside exact NVMe workspace')
    for q in (p, *p.parents):
        require(not q.is_symlink(), 'symlink in raw path')
        if q.exists():
            require(q.stat().st_uid == shared.UID, 'raw workspace has another owner')
        if q == RAWBASE:
            break
    return p


def normalized_reuse(value):
    require(isinstance(value, dict) and set(value) <= {'h0', 'h60'}, 'invalid reuse model map')
    result = {kind: value.get(kind, {}) for kind in ('h0', 'h60')}
    require(all(isinstance(records, dict) for records in result.values()), 'invalid reuse record map')
    return result


def verify_native_origin(ref):
    import stat
    path = raw_path(ref['path']); info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode) and info.st_uid == shared.UID and not info.st_mode & 0o222,
            'reuse source is not an owned immutable regular file')
    actual = core.file_ref(path)
    require(all(actual[k] == ref[k] for k in ('path', 'sha256', 'size_bytes')), 'reuse source hash/size differs')
    return path


def validate_recovery(plan, tasks):
    """The single failed-monitor successor; original tasks and files stay immutable."""
    recovery = plan.get('recovery')
    if 'recovery' not in plan:
        require(all(not any(normalized_reuse(t.get('reuse_native_records', {})).values()) for t in tasks),
                'native reuse requires the bound recovery proof')
        return
    require(isinstance(recovery, dict) and set(recovery) == {'parent_plan', 'partial_verification'},
            'recovery requires exactly the parent and partial verification refs')
    require(plan['run_token'] == RECOVERY_TOKEN and Path(plan['output']).name == RECOVERY_TOKEN and
            Path(plan['raw_output']).name == RECOVERY_RAW_NAME, 'wrong recovery output identity')
    require(recovery['parent_plan']['sha256'] == PARENT_PLAN_SHA, 'wrong recovery parent digest')
    parent = core.read_json_ref(recovery['parent_plan'])
    require('recovery' not in parent and parent['run_token'] == ORIGINAL_TOKEN and
            Path(parent['raw_output']) == RAWBASE / ORIGINAL_RAW_NAME and
            Path(parent['output']).name == ORIGINAL_TOKEN, 'wrong original extraction identity')
    for key in ('protocol', 'mode', 'host', 'uid', 'records', 'padded_sequence_length', 'record_ids_sha256',
                'gpus', 'python', 'numerical_policy', 'expected_raw_tensor_bytes'):
        require(plan[key] == parent[key], 'recovery changed parent ' + key)
    proof = core.read_json_ref(recovery['partial_verification'])
    require(proof['status'] == 'independently_verified_partial_raw_after_monitor_failure' and
            proof['parent_plan_sha256'] == PARENT_PLAN_SHA and proof['parent_plan'] == recovery['parent_plan'] and
            proof['cgroup_processes'] == [] and proof['gpu_release_verified'] is True and
            proof['native_metadata_dtype_shape_tokens_masks_finite_verified'] is True,
            'partial recovery proof is not positive for the exact parent')
    service = proof['service_fields']
    require(all(service.get(k) == v for k, v in {'Id': ORIGINAL_TOKEN + '.service', 'ExecMainCode': '1',
            'ExecMainStatus': '1', 'Result': 'exit-code', 'MainPID': '0', 'SubState': 'failed'}.items()),
            'original service has no released failed exit')
    oldout = Path(parent['output']); artifact_ref = proof['failed_artifact_manifest']
    require(artifact_ref['path'] == str(oldout / 'artifact_manifest.json'), 'wrong failed artifact path')
    artifact = core.read_json_ref(artifact_ref)
    require(artifact['plan_sha256'] == PARENT_PLAN_SHA and artifact['raw_local_root'] == parent['raw_output'],
            'failed artifact belongs to another extraction')
    failure = core.read_json_ref({'path': str(oldout / 'RUN_FAILED.json'), **artifact['files']['RUN_FAILED.json']})
    require(failure == proof['failure'] and failure['status'] == 'failed' and
            failure['plan_sha256'] == PARENT_PLAN_SHA and failure['run_token'] == ORIGINAL_TOKEN and
            failure['process_release_verified'] is True and failure['gpu_release_verified'] is True and
            'FileNotFoundError' in failure['error'] and '.writing' in failure['error'] and
            not (oldout / 'RUN_COMPLETE.json').exists(), 'failed monitor terminal/release evidence differs')
    oldworkers = parent['workers']; require(len(oldworkers) == len(plan['workers']) == len(tasks), 'changed worker count')
    reuse = proof['reusable_native_records']
    require(isinstance(reuse, dict) and set(reuse) <= {w['name'] for w in oldworkers}, 'foreign worker reuse')
    operational = {'run_token', 'output', 'raw_output', 'manifest_path', 'manifest_sha256', 'reuse_native_records'}
    count = 0
    for worker, oldworker, task in zip(plan['workers'], oldworkers, tasks):
        require({k: v for k, v in worker.items() if k != 'task'} ==
                {k: v for k, v in oldworker.items() if k != 'task'}, 'recovery worker allocation changed')
        oldtask = core.read_json_ref(oldworker['task'])
        require({k: v for k, v in task.items() if k not in operational} ==
                {k: v for k, v in oldtask.items() if k not in operational}, 'recovery task science/profile/shard changed')
        for path in (oldtask['prepared_records'], *[p for p in parent['bound_files'] if
                any(Path(p).is_relative_to(oldtask[k]) for k in ('model_snapshot', 'checkpoint'))]):
            require(plan['bound_files'].get(path) == parent['bound_files'][path], 'recovery input binding changed')
        expected = normalized_reuse(reuse.get(worker['name'], {}))
        require(normalized_reuse(task.get('reuse_native_records', {})) == expected, 'task reuse differs from partial proof')
        for kind, records in expected.items():
            for rid, ref in records.items():
                require(set(ref) == {'path', 'sha256', 'size_bytes', 'source_manifest_sha256',
                                    'record_index', 'record_id', 'kind'}, 'incomplete native origin identity')
                require(rid in oldtask['record_ids'] and ref['record_id'] == rid and ref['kind'] == kind and
                        type(ref['record_index']) is int and 0 <= ref['record_index'] < 500 and
                        ref['source_manifest_sha256'] == PARENT_PLAN_SHA, 'wrong reused record/model/manifest')
                relative = Path(worker['name']) / kind / f"record_{ref['record_index']:06d}.safetensors"
                require(ref['path'] == str(Path(parent['raw_output']) / relative) and
                        {k: ref[k] for k in ('sha256', 'size_bytes')} == artifact['files']['raw/' + str(relative)],
                        'reuse path/hash differs from exact failed worker artifact')
                verify_native_origin(ref); count += 1
    require(type(proof['reusable_native_files']) is int and 0 < count == proof['reusable_native_files'],
            'reused native file count differs')


def load_plan(path, sha):
    p = core.safe_path(path)
    plan = core.read_json_ref({'path': str(p), 'sha256': sha, 'size_bytes': p.stat().st_size})
    require(plan['protocol'] == PROTOCOL and plan['mode'] == 'broader_raw' and
            plan['host'] == shared.HOST and plan['uid'] == shared.UID, 'wrong raw plan identity')
    require(plan['records'] == 500 and plan['padded_sequence_length'] == 2688, 'wrong population or numerical profile')
    require(plan['absolute_deadline_epoch'] <= shared.DEADLINE, 'deadline exceeds durable deadline')
    out, source = core.safe_path(plan['output']), core.safe_path(plan['source_root'])
    rawroot = raw_path(plan['raw_output'])
    expected_raw = RECOVERY_RAW_NAME if 'recovery' in plan else ORIGINAL_RAW_NAME
    expected_token = RECOVERY_TOKEN if 'recovery' in plan else ORIGINAL_TOKEN
    require(plan['run_token'] == out.name == expected_token and rawroot == RAWBASE / expected_raw,
            'wrong dedicated output roots')
    require(not out.is_relative_to(source) and not source.is_relative_to(out), 'output overlaps source')
    core.verify_file(plan['python'], immutable=False, runtime=True)
    required = {HERE, WORKER.replace('.', '/') + '.py',
                'infra/gpu03/direction_discovery/fixed_cache.py',
                'infra/gpu03/direction_discovery/h100_tf_run.py'}
    require(required <= set(plan['source_inventory']), 'missing execution source')
    require({str(q.relative_to(source)) for q in source.rglob('*') if q.is_file()} ==
            set(plan['source_inventory']), 'source has unbound extras/missing files')
    for name, ref in plan['source_inventory'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe source path')
        core.verify_file({'path': str(source / name), **ref})
    for name, ref in plan['bound_files'].items():
        core.verify_file({'path': name, **ref})
    require(len(plan['gpus']) == 8 and {g['id'] for g in plan['gpus']} == set(range(8)), 'need eight explicit GPUs')
    require(len({g['uuid'] for g in plan['gpus']}) == 8, 'duplicate GPU UUID')
    tasks, seen, cpus = [], set(), set()
    require(len(plan['workers']) == 8, 'need eight raw workers')
    for i, worker in enumerate(plan['workers']):
        require(worker['gpu_id'] == i and worker['name'] == f'worker_{i:02d}', 'worker allocation differs')
        require(worker['cpu_set'] and not cpus.intersection(worker['cpu_set']), 'CPU overlap')
        cpus.update(worker['cpu_set'])
        task = core.read_json_ref(worker['task'])
        require(task['mode'] == 'broader_raw' and task['run_token'] == plan['run_token'] and
                task['gpu_id'] == i and task['worker_name'] == worker['name'], 'worker identity differs')
        require(task['output'] == str(out / worker['name']) and
                task['raw_output'] == str(rawroot / worker['name']), 'worker output differs')
        require(task['padded_sequence_length'] == 2688 and task['pad_token_id'] == 151643 and
                task['gpu_memory_fraction'] == .65 and 0 < task['deadline_seconds'] <= 3600,
                'worker numeric/resource settings differ')
        require(task['manifest_path'] == str(p) and task.get('manifest_sha256') in (None, sha), 'task manifest differs')
        require(task['prepared_records'] in plan['bound_files'] and
                task['prepared_records_sha256'] == plan['bound_files'][task['prepared_records']]['sha256'],
                'prepared rows unbound')
        for key in ('model_snapshot', 'checkpoint'):
            folder = core.safe_path(task[key]); files = [q for q in folder.rglob('*') if q.is_file()]
            require(files and all(str(q) in plan['bound_files'] for q in files), 'model has unbound files')
        require(task['record_ids'] and len(task['record_ids']) == len(set(task['record_ids'])) and
                not seen.intersection(task['record_ids']), 'duplicate/empty record shard')
        seen.update(task['record_ids']); tasks.append(task)
    require(len(seen) == 500 and core.digest(sorted(seen)) == plan['record_ids_sha256'], 'wrong record union')
    require(len({t['prepared_records'] for t in tasks}) == 1, 'prepared population varies across workers')
    require(plan['limits']['maximum_raw_bytes'] >= 160 << 30 and plan['limits']['minimum_free_raw_bytes'] >= 256 << 30,
            'raw storage reserve too small')
    validate_recovery(plan, tasks)
    return plan, tasks


def raw_inventory(root, verified=None):
    verified = verified or {}
    files = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():
            require(not p.is_symlink(), 'raw symlink')
            relative = str(p.relative_to(root))
            if relative in verified:
                require(p.stat().st_size == verified[relative]['size_bytes'], 'verified file changed size')
                files[relative] = verified[relative]
            else:
                files[relative] = {k: v for k, v in core.file_ref(p).items() if k != 'path'}
    return files


def observe_raw(plan):
    import stat
    root = raw_path(plan['raw_output']); fs = os.statvfs(root)
    require(fs.f_bavail * fs.f_frsize >= plan['limits']['minimum_free_raw_bytes'], 'NVMe reserve exceeded')
    files = [p for p in root.rglob('*') if p.is_file()]
    require(all(not p.is_symlink() for p in files), 'raw symlink')
    used = 0
    for p in files:
        try:
            info = p.stat(follow_symlinks=False)
        except FileNotFoundError:
            # A regular .writing file can be atomically renamed after discovery.
            # Count its final name on the next snapshot; other stat errors fail.
            continue
        require(stat.S_ISREG(info.st_mode), 'raw path ceased to be a regular file')
        used += info.st_size
    require(used <= plan['limits']['maximum_raw_bytes'], 'raw byte cap exceeded')
    return {'raw_bytes': used, 'raw_free_bytes': fs.f_bavail * fs.f_frsize}


def tensor_exact(actual, expected):
    import torch
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)


def inspect_native(path, row, kind, sha, *, return_original=False):
    import torch
    from safetensors import safe_open
    from . import fixed_cache as fixed
    fixed.validate_row(row, padded_length=2688)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        require(f.metadata() == fixed.metadata(row, kind, sha, padded_length=2688), 'raw metadata differs')
        value = f.get_tensor(kind)
        require(value.shape == (fixed.LAYERS, row['completion_token_count'], fixed.HIDDEN) and
                value.dtype == torch.bfloat16, 'raw tensor shape/dtype differs')
        require(all(bool(torch.isfinite(value[j]).all()) for j in range(fixed.LAYERS)), 'nonfinite raw values')
        final = f.get_tensor('prompt_final')
        require(final.shape == (fixed.LAYERS, fixed.HIDDEN) and final.dtype == torch.bfloat16 and
                bool(torch.isfinite(final).all()), 'invalid prompt-final tensor')
        inputs = fixed.padded_inputs(row, 'cpu', padded_length=2688)
        auxiliary = fixed.auxiliary_tensors(row, {k: v[0] for k, v in inputs.items()})
        require(set(f.keys()) == {kind, 'prompt_final', *auxiliary}, 'raw tensor inventory differs')
        require(all(tensor_exact(f.get_tensor(k), v) for k, v in auxiliary.items()),
                'raw auxiliary dtype/shape/IDs/masks/padding differ')
        return torch.cat((final[:, None], value), dim=1) if return_original else None


def inspect_qualification(task, row, kind, sha, qualification, first_path):
    """Recompute the saved fixed-shape audit; no decoder or generated code runs."""
    import torch
    from safetensors import safe_open
    from . import fixed_cache as fixed
    directory = raw_path(str(Path(task['raw_output']) / 'qualification' / kind))
    repeat = json.loads((directory / 'repeat_audit.json').read_bytes())
    future = json.loads((directory / 'future_causality_audit.json').read_bytes())
    require(qualification['record_id'] == row['record_id'] and
            qualification['first_production_raw_retained'] is True and
            future == qualification['future_causality'], 'first-real-record qualification identity differs')
    flags = {'math': True, 'flash': False, 'memory_efficient': False, 'cudnn': False}
    require(repeat['attention_flags'] == future['attention_flags'] == flags, 'qualification backend differs')
    n = row['completion_token_count']; body = row.get('evaluator_body_token')
    keep = min(max(16, body + 12 if type(body) is int else 0), n - 1)
    boundary = row['prompt_token_count'] + keep
    require(n >= 2 and future['unchanged_completion_prefix_tokens'] == keep and
            future['first_changed_sequence_position'] == boundary and
            future['changed_valid_future_tokens'] == n - keep and future['fixed_padded_length'] == 2688 and
            future['evaluator_transition_prefix_included'] == (type(body) is int and body + 12 <= keep) and
            future['padding_mask_zero_unchanged'] is True, 'qualification future-prefix scope differs')
    config = json.loads((Path(task['model_snapshot']) / 'config.json').read_bytes())
    vocab_size = config['vocab_size']
    require(type(vocab_size) is int and vocab_size > 1 and all(t < vocab_size for t in row['input_ids']),
            'qualification vocabulary differs')
    changed = list(row['input_ids'])
    for i in range(boundary, len(changed)):
        changed[i] = (changed[i] + 1) % vocab_size
    original = inspect_native(first_path, row, kind, sha, return_original=True)
    for name, receipt, purpose, override in (
            ('repeat', repeat, 'exact_fixed_shape_repeat', None),
            ('future_perturbed', future, 'same_fixed_shape_valid_future_token_perturbation', changed)):
        path = raw_path(str(directory / (name + '.safetensors')))
        artifact = receipt['artifact']; actual = core.file_ref(path)
        require(artifact.get('native_readback_bitwise_equal') is True and
                all(artifact.get(k) == actual[k] for k in ('path', 'sha256', 'size_bytes')),
                'qualification probe path/hash/size differs')
        with safe_open(str(path), framework='pt', device='cpu') as f:
            meta = {**fixed.metadata(row, kind, sha, padded_length=2688), 'purpose': purpose,
                    'token_axis': 'prompt_final_then_original_completion'}
            require(f.metadata() == meta and set(f.keys()) ==
                    {'post_block_selected', 'input_ids', 'attention_mask', 'position_ids'}, 'probe metadata/keys differ')
            value = f.get_tensor('post_block_selected')
            require(value.dtype == torch.bfloat16 and value.shape == original.shape and
                    bool(torch.isfinite(value).all()), 'probe shape/dtype/finite contract differs')
            inputs = fixed.padded_inputs(row, 'cpu', override, padded_length=2688)
            require(all(tensor_exact(f.get_tensor(k), v[0].to(torch.int32)) for k, v in inputs.items()),
                    'probe input dtype/shape/IDs/mask/positions differ')
            measured = fixed.equality_report(original, value) if name == 'repeat' else fixed.equality_report(
                original[:, :keep + 1], value[:, :keep + 1])
        require(measured['bitwise_equal'] is True and all(receipt.get(k) == v for k, v in measured.items()),
                'saved numerical audit does not match recomputed tensors')
        if name == 'repeat':
            require(qualification['repeat'] == measured, 'repeat summary differs from recomputation')


def validate_worker_outputs(task, rows, sha, *, inspect_values):
    """One worker boundary, also used by small real-native regression fixtures."""
    out = Path(task['output']); receipt = json.loads((out / 'SUCCESS.json').read_bytes())
    require(not (out / 'FAILURE.json').exists() and json.loads((out / 'task.json').read_bytes()) == task,
            'worker failure exists or saved task differs')
    require(all(receipt.get(k) == v for k, v in {'status': 'succeeded', 'mode': 'broader_raw',
            'run_token': task['run_token'], 'worker_name': task['worker_name'], 'records': len(task['record_ids']),
            'record_ids': task['record_ids'], 'native_files': 2 * len(task['record_ids']), 'manifest_sha256': sha,
            'padded_sequence_length': 2688, 'raw_activations_retained': True, 'differences_computed': False}.items()),
            'worker success contract differs')
    for kind in ('h0', 'h60'):
        load = receipt['model_load_reports'][kind]
        require(load['with_adapter'] == (kind == 'h60'), 'wrong model load')
        if kind == 'h60':
            require(load['active_adapters'] and load['nonzero_lora_parameter_tensors'] > 0, 'adapter not active')
        audit = receipt['qualifications'][kind]
        require(audit['repeat']['bitwise_equal'] is True and audit['future_causality']['bitwise_equal'] is True,
                'numerical audit failed')
    reuse = normalized_reuse(task.get('reuse_native_records', {}))
    if any(reuse.values()):
        require(receipt.get('reused_native_files') == {k: len(v) for k, v in reuse.items()} and
                receipt.get('fresh_native_captures') == {k: len(task['record_ids']) - len(v) for k, v in reuse.items()} and
                receipt.get('reuse_parent_manifest_sha256') == PARENT_PLAN_SHA, 'worker reuse counts/parent differ')
    entries, local = [], set()
    for line in (out / 'native_index.jsonl').read_text().splitlines():
        entry = json.loads(line); key = (entry['record_id'], entry['kind'])
        require(key not in local and key[0] in task['record_ids'] and key[1] in ('h0', 'h60'),
                'wrong/duplicate raw entry')
        row = rows[key[0]]
        require(entry['record_index'] == row['record_index'] and entry['all_finite'] is True and
                entry['native_readback_bitwise_equal'] is True, 'raw save verification missing')
        expected = Path(task['raw_output']) / key[1] / f"record_{row['record_index']:06d}.safetensors"
        require(entry['path'] == str(expected), 'raw file is not at its exact record/model path')
        p = raw_path(entry['path']); ref = core.file_ref(p)
        require(ref['sha256'] == entry['sha256'] and ref['size_bytes'] == entry['size_bytes'], 'raw file hash differs')
        origin = reuse[key[1]].get(key[0])
        if origin is not None:
            require(origin['record_index'] == row['record_index'] and origin['record_id'] == key[0] and
                    origin['kind'] == key[1] and origin['source_manifest_sha256'] == PARENT_PLAN_SHA and
                    entry.get('reused_native') is True and entry.get('reuse_origin') == origin and
                    entry.get('reused_values_bitwise_preserved') is True and entry.get('new_manifest_sha256') == sha and
                    entry.get('capture_seconds') == 0, 'reused native provenance differs')
            if inspect_values:
                prior = verify_native_origin(origin)
                old_values = inspect_native(prior, row, key[1], PARENT_PLAN_SHA, return_original=True)
                new_values = inspect_native(p, row, key[1], sha, return_original=True)
                require(tensor_exact(new_values, old_values), 'reused native values differ from the original')
                del old_values, new_values
        else:
            require(not entry.get('reused_native') and 'reuse_origin' not in entry,
                    'unapproved reused native file')
            if inspect_values:
                inspect_native(p, row, key[1], sha)
        entries.append(entry); local.add(key)
    require(local == {(rid, kind) for rid in task['record_ids'] for kind in ('h0', 'h60')}, 'missing raw files')
    if inspect_values:
        first = rows[task['record_ids'][0]]
        for kind in ('h0', 'h60'):
            path = Path(task['raw_output']) / kind / f"record_{first['record_index']:06d}.safetensors"
            inspect_qualification(task, first, kind, sha, receipt['qualifications'][kind], path)
    return entries


def validate_outputs(plan, tasks, sha, *, inspect_values):
    import torch
    torch.set_num_threads(1)
    prepared = list(map(json.loads, Path(tasks[0]['prepared_records']).read_text().splitlines()))
    rows = {r['record_id']: r for r in prepared}
    require(len(rows) == len(prepared) == 500 and {r['problem_split'] for r in rows.values()} == {'direction_fit'},
            'wrong prepared population')
    entries, seen = [], set()
    for task in tasks:
        local = validate_worker_outputs(task, rows, sha, inspect_values=inspect_values)
        require(not seen.intersection((r['record_id'], r['kind']) for r in local), 'duplicate raw entry across workers')
        seen.update((r['record_id'], r['kind']) for r in local); entries.extend(local)
    require(seen == {(rid, kind) for rid in rows for kind in ('h0', 'h60')} and len(entries) == 1000,
            'wrong total raw file population')
    return entries


def supervise(path, sha):
    plan, tasks = load_plan(path, sha)
    require(os.getuid() == shared.UID and time.time() < plan['absolute_deadline_epoch'], 'owner/deadline differs')
    # Existing checker additionally verifies hostname, Python, versions and CPU allocation.
    original_here = Path(core.__file__).resolve()
    require(original_here == Path(plan['source_root']) / 'infra/gpu03/direction_discovery/h100_tf_run.py', 'wrong shared controller')
    core.check_runtime(plan)
    out = core.safe_path(plan['output']); rawroot = raw_path(plan['raw_output'])
    out.mkdir(mode=0o700, exist_ok=False); rawroot.mkdir(mode=0o700, exist_ok=False)
    children, logs, error, released = [], [], None, False
    started = time.time(); old = {}
    def interrupted(sig, _frame):
        raise RuntimeError('controller received signal ' + str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT): old[sig] = signal.signal(sig, interrupted)
    try:
        core.gpu_check(plan, shared.gpu_snapshot(), {}, idle=True)
        for worker in plan['workers']:
            core.append(out / 'resources.jsonl', {**core.resource_state(plan, children), **observe_raw(plan)})
            command = ['/usr/bin/taskset', '-c', ','.join(map(str, worker['cpu_set'])), plan['python']['path'],
                       '-B', '-m', WORKER, '--task', worker['task']['path'], '--task-sha256', worker['task']['sha256']]
            log = (out / (worker['name'] + '.log')).open('xb'); logs.append(log)
            process = subprocess.Popen(command, cwd=plan['source_root'], env=shared.worker_environment(worker['gpu_id']),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = shared.process_info(process.pid)
            children.append((process, identity or {'pid': process.pid, 'pgid': process.pid, 'uid': shared.UID, 'start_ticks': -1}, worker))
            require(identity is not None and identity['pgid'] == process.pid and identity['uid'] == shared.UID, 'missing worker identity')
            core.append(out / 'workers.jsonl', {'name': worker['name'], 'identity': identity, 'command': command, 'at': time.time()})
            time.sleep(plan['limits']['stagger_seconds'])
        while True:
            core.append(out / 'resources.jsonl', {**core.resource_state(plan, children), **observe_raw(plan)})
            require(all(p.poll() in (None, 0) for p, _, _ in children), 'raw worker failed')
            if all(p.poll() == 0 for p, _, _ in children): break
            time.sleep(plan['limits']['monitor_seconds'])
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            core.cleanup(children)
            core.gpu_check(plan, shared.gpu_snapshot(), {}, idle=True); released = True
        except BaseException as exc:
            error = (error or '') + '; cleanup: ' + str(exc)
        for log in logs: log.close()
        for sig, handler in old.items(): signal.signal(sig, handler)
    require(error is None or all(p.poll() is not None for p, _, _ in children), 'unreleased failed workers')
    entries = []
    if error is None:
        try: entries = validate_outputs(plan, tasks, sha, inspect_values=False)
        except Exception as exc: error = 'output validation: ' + str(exc)
    verified_raw = {}
    if error is None:
        by_id = {}
        for entry in entries:
            relative = str(Path(entry['path']).relative_to(rawroot))
            ref = {k: entry[k] for k in ('sha256', 'size_bytes')}
            verified_raw[relative] = ref
            item = by_id.setdefault(entry['record_id'], {'record_id': entry['record_id'],
                    'record_index': entry['record_index'], 'verified': True, 'models': {}})
            item['models'][entry['kind']] = {'tensor_path': relative, **ref}
        with (out / 'activation_index.jsonl').open('x') as f:
            for entry in sorted(by_id.values(), key=lambda x: x['record_index']):
                f.write(core.canonical(entry) + '\n')
            f.flush(); os.fsync(f.fileno())
        core.write_json(out / 'raw_tensor_manifest.json', {'algorithm': 'sha256', 'files': verified_raw})
    terminal = {'status': 'complete' if error is None else 'failed', 'error': error, 'run_token': plan['run_token'],
                'plan_sha256': sha, 'started_epoch': started, 'finished_epoch': time.time(),
                'worker_returncodes': [p.returncode for p, _, _ in children], 'process_release_verified': released,
                'gpu_release_verified': released, 'native_files': len(entries)}
    if 'recovery' in plan:
        terminal.update(recovery=plan['recovery'], reused_native_files=sum(e.get('reused_native') is True for e in entries))
    core.write_json(out / ('RUN_COMPLETE.json' if error is None else 'RUN_FAILED.json'), terminal)
    rawfiles = raw_inventory(rawroot, verified_raw)
    files = raw_inventory(out)
    files.update({'raw/' + k: v for k, v in rawfiles.items()})
    core.write_json(out / 'artifact_manifest.json', {'algorithm': 'sha256', 'plan_sha256': sha,
                    'raw_local_root': str(rawroot), 'files': files})
    for root in (out, rawroot):
        for p in root.rglob('*'):
            if p.is_file(): p.chmod(0o400)
    require(error is None, error)
    return core.file_ref(out / 'artifact_manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(core.canonical(supervise(args.plan, args.sha256)), flush=True)
