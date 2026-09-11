"""Explicit GPU04 authority / allowed-host execution, with no model imports.

An immutable permit charges one physical deployment before it can be launched.
A replica retains original remote paths and resolves them only through a bounded,
hash-checked map. It never rewrites scientific plans or reads generation outcomes.
All writers are exclusive; no implicit recovery, refund, remote SSH, or launch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
PROTOCOL = 'gpu04_authority_gpu02_generation_v1'
PROTOCOLS = {'gpu-02': PROTOCOL, 'gpu-01': 'gpu04_authority_gpu01_generation_v1'}
REGISTRY = 'remote_generation_registry'
AUTHORITY_ROOT = Path('/scratch/researcher/codex_runs')
MASTER = '1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2'
PARENT = 'd4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'
ORIGINAL_QUALIFICATION = 'bc7d21273a27301ccdfb0412ed6737f0322096aeb7023b39a03bc9c76083d8d1'
CORE_PREPARED = 'e2ce9ee0b23ebd44b304a57d9400b38d0fbc348cb99f84311f8faf32031ef65e'
CONFIG_BINDINGS = {
    'config.json': {'sha256': '8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a', 'size_bytes': 726},
    'generation_config.json': {'sha256': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2', 'size_bytes': 239},
}
ENGINE = '8c4723a20ac341e3d92c7ad7bc514840899d1d4ec51efc53e2a089d1fa8852cf'
HOOK = '6eed46f798a00dfd6cab9ee91a035853cf9685cd5cf6a14c1efb2b866a3f3529'
COUNTERS = ('generation_requests', 'tf_requests', 'untouched_test_generation_requests',
            'untouched_test_tf_requests', 'untouched_test_requests')
TRANSPORT = frozenset(('run_token', 'worker_name', 'gpu_id', 'output', 'deadline_seconds',
                      'model_snapshot', 'checkpoint', 'prepared_records', 'raw_package', 'requests', 'conditions', 'mode'))


def helper(name):
    return importlib.import_module('infra.gpu03.direction_discovery.' + name)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def execution_protocol(host):
    """Closed host allowlist; historical GPU02 protocol remains unchanged."""
    require(isinstance(host, str) and host in PROTOCOLS, 'Unapproved remote execution host')
    return PROTOCOLS[host]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def info(path):
    return {'sha256': sha(path), 'size_bytes': Path(path).stat().st_size}


def check_file(path, expected, *, immutable=True):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing or symlinked transport file: ' + str(path))
    require(not immutable or not path.stat().st_mode & 0o222, 'Transport input must be immutable: ' + str(path))
    require(re.fullmatch('[0-9a-f]{64}', expected.get('sha256', '')) and sha(path) == expected['sha256'] and
            ('size_bytes' not in expected or path.stat().st_size == expected['size_bytes']), 'Transport file bytes changed: ' + str(path))
    return path


def read(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Explicit path/SHA reference required')
    path = check_file(ref['path'], ref)
    require(path.stat().st_size <= 32 * 1024**2, 'Transport JSON metadata exceeds bounded size')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'Transport read snapshot changed')
    return json.loads(data)


def ref(path):
    return {'path': str(Path(path)), 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o400)
    return ref(path)


def resolve(files, original):
    require(original in files, 'Remote path has no explicit replica: ' + str(original))
    value = files[original]
    require(set(value) == {'path', 'sha256', 'size_bytes'}, 'Malformed replica binding')
    return check_file(value['path'], value)


def _source_inventory(m):
    prefix = m['source_root'].rstrip('/') + '/'
    return {name[len(prefix):]: entry['sha256'] for name, entry in m['bound_files'].items() if name.startswith(prefix)}


def validate_profile(profile, m):
    protocol = execution_protocol(m.get('host'))
    require(profile.get('protocol') == protocol and profile.get('host') == m['host'] and
            profile.get('authority_host') == 'gpu-04' and profile.get('owner') == 'researcher' and
            profile.get('no_training') is True and profile.get('sampling_unchanged') is True,
            'Unknown remote host profile or scientific change')
    require(profile['master_plan_sha256'] == m['scientific']['master_plan_sha256'] == MASTER,
            'Remote deployment must preserve the current scientific master')
    require(profile['runtime_versions'] == m['runtime_versions'] and profile['python'] == m['python'],
            'Remote runtime differs from reviewed profile')
    require(profile['gpu_model'] == 'NVIDIA RTX 5000 Ada Generation' and profile['maximum_concurrent_gpus'] == 8 and
            profile['driver_version'] == '570.195.03', 'Remote hardware profile differs')
    require(all(profile['gpu_uuids'].get(str(i)) == m['gpu_uuids'][str(i)] for i in m['gpu_ids']), 'Remote GPU UUID changed')
    require(profile['source_inventory'] == _source_inventory(m) and
            profile['source_inventory'].get('infra/gpu03/direction_discovery/engine.py') == ENGINE and
            profile['source_inventory'].get('infra/gpu03/direction_discovery/intervention.py') == HOOK,
            'Qualified source or model/intervention code changed')
    require(120 < m['limits']['runtime_seconds'] <= 14400 and
            m['limits']['systemd_runtime_seconds'] == m['limits']['runtime_seconds'] + 180,
            'Remote phase exceeds frozen time profile')
    mappings = profile['model_files']
    require(isinstance(mappings, list) and len(mappings) >= 12, 'Complete model/tokenizer/adapter mapping required')
    require(len({x['remote'] for x in mappings}) == len(mappings) and len({x['authority'] for x in mappings}) == len(mappings),
            'Duplicated model deployment mapping')
    for item in mappings:
        require(set(item) == {'authority', 'remote', 'sha256', 'size_bytes'} and
                Path(item['authority']).is_absolute() and Path(item['remote']).is_absolute() and
                m['bound_files'].get(item['remote']) == {k: item[k] for k in ('sha256', 'size_bytes')},
                'Model deployment hash mapping changed')
    require(set(profile['model_paths']) == {'model_snapshot', 'checkpoint'} and
            all(Path(p).is_absolute() for p in profile['model_paths'].values()), 'Invalid remote model paths')
    require(all(any(item['remote'].startswith(p + '/') for item in mappings) for p in profile['model_paths'].values()),
            'Model directories have no bound files')


def validate_manifest(m, *, files=None, check_dependencies=False):
    """Validate immutable deployment metadata; files resolves a CPU04 replica."""
    helper('supervisor').validate_manifest_object(m, Path(m['stage']) / 'reviewed_manifest.json')
    protocol = execution_protocol(m.get('host'))
    meta = m.get('remote_generation')
    require(isinstance(meta, dict) and set(meta) == {'protocol', 'authority', 'profile'} and
            meta['protocol'] == protocol, 'Execution host requires its explicit remote-generation protocol')
    get = (lambda path: resolve(files, path)) if files is not None else (lambda path: check_file(path, m['bound_files'].get(path, {})))
    values = {}
    for key in ('authority', 'profile'):
        binding = meta[key]
        require(m['bound_files'].get(binding['path'], {}).get('sha256') == binding['sha256'], 'Unbound remote ' + key)
        values[key] = json.loads(get(binding['path']).read_bytes())
    authority, profile = values['authority'], values['profile']
    validate_profile(profile, m)
    require(authority.get('protocol') == protocol and authority.get('authority_host') == 'gpu-04' and
            authority.get('master_plan_sha256') == MASTER and authority.get('parent_plan_sha256') == PARENT and
            authority.get('no_intermediate_analysis') is True, 'Invalid scientific authority envelope')
    require(profile['sha256_authority_model_files'] == hashlib.sha256(canonical(authority['model_files']).encode()).hexdigest(),
            'Remote profile lost exact original model provenance')
    for item in profile['model_files']:
        source_paths = authority['task_template']
        groups = [key for key in ('model_snapshot', 'checkpoint') if Path(item['authority']).is_relative_to(source_paths[key])]
        require(len(groups) == 1 and Path(item['remote']) == Path(profile['model_paths'][groups[0]]) / Path(item['authority']).relative_to(source_paths[groups[0]]),
                'Mapped model files must retain exact relative paths beneath the actual loaded model/adapter directories')
        require(authority['model_files'].get(item['authority']) == {k: item[k] for k in ('sha256', 'size_bytes')},
                'Remote model does not match original qualified file')
    require(len(profile['model_files']) == len(authority['model_files']), 'Remote model map omitted original files')
    plan_path = str(Path(m['stage']) / 'input/request_plan.json')
    plan = json.loads(get(plan_path).read_bytes())
    require(info(get(plan_path))['sha256'] == authority['request_plan']['sha256'], 'Deployment request plan differs from authority')
    template = authority['task_template']
    require(template.get('attention_policy') == 'exclusive_math' and template.get('teacher_forced_padded_sequence_length') == 2176 and
            template.get('mode') in ('generate', 'qualify'), 'Unqualified scientific task template')
    require(m['scientific'].get('input_prepared_sha256') == authority['prepared']['sha256'] and
            m['scientific'].get('training') is False, 'Prepared records/training contract changed')
    mode = plan['mode']
    require(mode in ('generate', 'qualify') and bool(plan['requests']), 'Remote mode must be generate or bounded qualification')
    if mode == 'qualify':
        require(m['phase'] == 'remote_causal_qualification' and len(plan['requests']) == len(m['workers']) == 1 and
                plan['requests'] == template['requests'] and plan['conditions'] == {'baseline': {'layers': []}} and
                plan['evaluation_partition'] == 'direction_fit' and profile.get('inference_qualification') is None,
                'Remote qualification must repeat exact reviewed fitting-only qualification')
    else:
        require(m['phase'] in ('behavior_finalist_validation_round1', 'behavior_untouched_test_partition') and
                profile.get('inference_qualification') is not None,
                'Remote generation requires successful host inference qualification')
        require(plan.get('master_plan_sha256') == MASTER and plan.get('evaluation_partition') in ('configuration_validation', 'untouched_test'),
                'Remote scientific plan/master differs')
        for key in ('execution_partition', 'test_execution'):
            require(m['scientific'].get(key) == plan.get(key), 'Remote execution partition differs')
        require(('execution_partition' in plan) != ('test_execution' in plan), 'Remote generation needs one fixed execution partition')
        if 'execution_partition' in plan:
            require(plan['execution_partition']['round_index'] == 1 and 'test_bundle' not in m['scientific'], 'Remote finalist may only execute round1')
        else:
            require(m['scientific'].get('test_bundle') == plan['test_execution']['bundle'], 'Remote test bundle differs')
    expected = {r['request_id']: r for r in plan['requests']}
    require(len(expected) == len(plan['requests']), 'Duplicate authoritative request IDs')
    tasks, seen = [], set()
    for worker in m['workers']:
        task_path = get(worker['command'][3]); task = json.loads(task_path.read_bytes())
        require(task['mode'] == mode == worker['success_expect']['mode'] and
                worker['success_expect']['requests'] == len(task['requests']) and
                task['run_token'] == m['run_token'] and task['worker_name'] == worker['name'] and task['gpu_id'] == worker['gpu_id'] and
                task['output'] == str((Path(m['output']) / worker['success_file']).parent) and
                task['deadline_seconds'] == m['limits']['runtime_seconds'] - 120, 'Remote worker assignment differs')
        require({k: v for k, v in task.items() if k not in TRANSPORT} == {k: v for k, v in template.items() if k not in TRANSPORT},
                'Remote task changed nontransport model/scientific fields')
        require(task['conditions'] == plan['conditions'] and task['sampling'] == template['sampling'], 'Remote conditions or sampling changed')
        require(all(task[k] == profile['model_paths'][k] for k in ('model_snapshot', 'checkpoint')),
                'Remote task model directory differs from exact map')
        require(m['bound_files'].get(task['prepared_records']) == authority['prepared'], 'Prepared deployment is not byte-identical')
        require(task['raw_package'] == template['raw_package'], 'Unused legacy raw-cache provenance changed')
        for condition in task['conditions'].values():
            for layer in condition.get('layers', []):
                if layer['kind'] == 'candidate':
                    require(m['bound_files'].get(layer['path'], {}).get('sha256') == layer['sha256'], 'Candidate condition path/hash was altered')
        ids = [r['request_id'] for r in task['requests']]
        require(len(ids) == len(set(ids)) and not seen.intersection(ids) and
                all(expected.get(r['request_id']) == r for r in task['requests']), 'Remote task repeats or changes requests')
        seen.update(ids); tasks.append({'worker': worker, 'task': task, 'task_path': task_path})
    require(seen == set(expected), 'Remote deployment lacks full partition coverage')
    if check_dependencies:
        validate_authority_dependencies(authority, plan, profile)
    return {'manifest': m, 'authority': authority, 'profile': profile, 'request_plan': plan,
            'request_plan_path': get(plan_path), 'tasks': tasks,
            'prepared_records_sha256': authority['prepared']['sha256'], 'scientific_conditions': plan['conditions']}


def expected_model_files(original, source_task):
    """Complete original bound set plus exactly the two parity-audited configs.

    Historical qualification omitted these JSON files from its manifest. Their
    independently compared current bytes are now explicit additional provenance;
    this does not claim they were included in that historical manifest.
    """
    result = {name: entry for name, entry in original['bound_files'].items()
              if name.startswith(source_task['model_snapshot'] + '/') or name.startswith(source_task['checkpoint'] + '/')}
    for name, entry in CONFIG_BINDINGS.items():
        path = str(Path(source_task['model_snapshot']) / name)
        require(path not in result or result[path] == entry, 'Historical model config conflicts with independently audited parity')
        result[path] = entry
    return result


def verify_original_template(authority, original):
    require(authority['template_manifest']['sha256'] == ORIGINAL_QUALIFICATION and
            authority['task_template']['mode'] == 'qualify', 'Original causal qualification identity differs')
    proof = read(authority['template_verification'])
    actual = helper('supervisor').verify(Path(authority['template_manifest']['path']), authority['template_manifest']['sha256'])
    require(proof == actual and proof.get('status') == 'verified' and proof.get('gpu_release_verified') is True and
            original.get('phase') == 'causal_qualification', 'Original scientific template lacks verified causal qualification')
    for relative, digest in [('infra/gpu03/direction_discovery/engine.py', ENGINE),
                             ('infra/gpu03/direction_discovery/intervention.py', HOOK)]:
        require(original['bound_files'].get(str(Path(original['source_root']) / relative), {}).get('sha256') == digest,
                'Original model/intervention source is not the qualified implementation')


def validate_authority_dependencies(authority, plan, profile):
    protocol = execution_protocol(profile.get('host'))
    require(profile.get('protocol') == authority.get('protocol') == protocol,
            'Authority and profile execution protocols differ')
    # verify_bundle itself establishes positive final freeze before opening or
    # hashing the prepared/test payload; preserve that boundary on every host.
    if plan.get('mode') == 'generate' and 'test_execution' in plan:
        helper('test_execution').validate_part(plan)
    for path, entry in authority['files'].items():
        if path in authority['model_files']:
            require(entry == authority['model_files'][path] and Path(path).is_absolute() and Path(path).is_file() and
                    info(path) == entry, 'Original qualified model content changed')
        else:
            check_file(path, entry)
    require(read(authority['request_plan']) == plan, 'Authority plan snapshot differs')
    require(authority['files'].get(authority['request_plan']['path'], {}).get('sha256') == authority['request_plan']['sha256'],
            'Authority request plan unbound')
    original = read(authority['template_manifest'])
    require(original.get('host') == 'gpu-04' and original['scientific'].get('master_plan_sha256') in (MASTER, PARENT),
            'Scientific template is not the original GPU04 lineage')
    verify_original_template(authority, original)
    require(original['runtime_versions'] == profile['runtime_versions'], 'Remote dependency versions differ from qualified original')
    for binding in (authority['template_manifest'], authority['template_task'], authority['template_verification'],
                    profile['source_qualification'], profile['lifetime_qualification']):
        require(authority['files'].get(binding['path'], {}).get('sha256') == binding['sha256'], 'Authority omitted a qualification binding')
    source_task = read(authority['template_task'])
    require(source_task.get('mode') == 'qualify' and source_task.get('conditions') == {'baseline': {'layers': []}} and
            source_task == authority['task_template'] and
            original['bound_files'].get(authority['template_task']['path'], {}).get('sha256') == authority['template_task']['sha256'],
            'Scientific task template is not bound to its original manifest')
    require(authority['model_files'] == expected_model_files(original, source_task),
            'Original model file set plus the exact two audited configurations differs')
    qualification = read(profile['source_qualification'])
    require(qualification.get('status') == 'independently_verified_remote_generation_source' and
            qualification.get('source_inventory') == profile['source_inventory'] and qualification.get('skips') == 0 and
            qualification.get('qualified_jobs_released') is True and qualification.get('tests', 0) > 0,
            'Remote transport source has not passed independent CPU qualification')
    lifetime = read(profile['lifetime_qualification'])
    require(lifetime.get('status') == 'independently_verified_scoped_ssh_disconnect_survival' and
            lifetime.get('host') == profile['host'] and lifetime.get('current_linger') == 'yes' and
            lifetime.get('launch_ssh_ancestor_gone') is True and lifetime.get('launcher_gone') is True and
            lifetime.get('owned_process_released') is True and lifetime.get('cgroup_absent_or_empty') is True and
            lifetime.get('zero_session_survival_unproven') is True and lifetime.get('other_sessions_not_modified') is True and
            lifetime.get('elapsed_seconds', 0) >= 80 and lifetime.get('detached_span_seconds', 0) >= 45 and
            lifetime.get('service_exit', {}).get('service_result') == 'success' and
            lifetime['service_exit'].get('exit_code') == 'exited' and lifetime['service_exit'].get('exit_status') == '0' and
            re.fullmatch('[0-9a-f]{32}', lifetime['service_exit'].get('invocation_id', '')),
            'Remote user service has not proved scoped disconnect survival and release')
    if plan['mode'] == 'generate':
        require(plan.get('sampling') == source_task['sampling'], 'Authoritative generation sampling differs from qualified template')
        if 'execution_partition' in plan:
            require(authority['prepared']['sha256'] == CORE_PREPARED, 'Finalist must use original core prepared records')
            helper('execution_partition').validate_part(plan)
            paths = helper('execution_partition').predecessor_bindings(plan, verify=True)
        else:
            full, _ = helper('test_execution').validate_part(plan)
            bundle, _ = helper('test_execution').bundle_context(plan['test_execution']['bundle'])
            require(authority['prepared']['sha256'] == bundle['prepared_records']['sha256'] and
                    authority['prepared_path'] == full['prepared_records'], 'Test deployment prepared union differs from the frozen bundle')
            paths = helper('test_execution').bindings(plan, verify=True)
        require(all(str(p) in authority['files'] and authority['files'][str(p)] == info(p) for p in paths),
                'Authority omitted full plan/bundle/predecessor bindings')
        qualification_context = context(profile['inference_qualification'])
        validate_inference_qualification(qualification_context, profile)


def validate_inference_qualification(ctx, profile):
    protocol = execution_protocol(profile.get('host'))
    require(ctx['manifest'].get('host') == ctx['profile'].get('host') == profile['host'] and
            ctx['profile'].get('protocol') == profile.get('protocol') == protocol,
            'Inference qualification belongs to another execution host or protocol')
    require(ctx['manifest']['phase'] == 'remote_causal_qualification' and len(ctx['results']) == 1 and
            ctx['profile']['model_files'] == profile['model_files'] and ctx['profile']['runtime_versions'] == profile['runtime_versions'] and
            ctx['profile']['source_inventory'] == profile['source_inventory'], 'Inference qualification differs from deployment')
    result_file = ctx['results'][0]
    require(result_file['size_bytes'] <= 4 * 1024**2, 'Unbounded qualification output')
    data = check_file(result_file['path'], result_file).read_bytes()
    require(hashlib.sha256(data).hexdigest() == result_file['sha256'], 'Qualification snapshot changed')
    rows = [json.loads(line) for line in data.splitlines()]
    request = ctx['request_plan']['requests'][0]
    require(len(rows) == 1 and rows[0].get('problem_split') == 'direction_fit' and
            all(rows[0].get(k) == v for k, v in request.items()), 'Qualification request/split differs')
    r = rows[0]['result']
    require(all(r.get(k) is True for k in ('baseline_recovery_bitwise', 'teacher_forced_effect_verified', 'baseline_generation_repeatable')),
            'Remote inference numerical qualification failed')
    baseline = r['baseline_generation']['generated_token_ids']; repeated = r['repeated_baseline_generation']['generated_token_ids']
    projected = r['projected_generation']
    require(len(baseline) == len(projected['generated_token_ids']) == 8 and baseline == repeated and set(projected['energy']) == {'12'},
            'Remote qualification generation/energy coverage differs')
    energy = projected['energy']['12']
    require(energy['rank'] == 1 and energy['selected_tokens'] == energy['forward_calls'] == 8 and
            energy['scopes']['prefill']['selected_tokens'] == 1 and energy['scopes']['decode']['selected_tokens'] == 7 and
            all(type(energy[k]) in (float, int) and math.isfinite(energy[k]) and energy[k] >= 0
                for k in ('removed_energy_fp32', 'remaining_subspace_energy_fp32', 'activation_energy')) and
            energy['removed_energy_fp32'] > 0 and energy['remaining_subspace_energy_fp32'] <= energy['removed_energy_fp32'] * 1e-8,
            'Remote qualification projection semantics failed')


def counts(plan):
    n = len(plan['requests']); mode = plan['mode']; test = plan['evaluation_partition'] == 'untouched_test'
    return dict(zip(COUNTERS, (n * (3 if mode == 'qualify' else 1), n * 3 if mode == 'qualify' else 0,
                              n if test else 0, 0, n if test else 0)))


def _budget_snapshot(ledger):
    return {k: v for k, v in ledger.items() if k != 'bindings'}


def validate_budget(ctx, ledger):
    plan, m = ctx['request_plan'], ctx['manifest']
    new = counts(plan)
    require(m['scientific']['previous_phase_budget'] == _budget_snapshot(ledger), 'Remote authority ledger snapshot is stale')
    require(ledger['generation_requests'] + new['generation_requests'] <= 4096 and
            ledger['tf_requests'] + new['tf_requests'] <= 12000 and
            ledger['gpu_phase_wall_seconds'] + m['limits']['systemd_runtime_seconds'] <= 28800,
            'Remote reservation exceeds frozen global budget')
    require(not any(p['run_token'] == m['run_token'] or p['manifest_sha256'] == ctx['manifest_sha256'] for p in ledger['phases']),
            'Remote physical reservation already committed')
    helper('phase_budget').assert_concurrency(ledger, len(m['gpu_ids']))
    if plan['mode'] == 'generate':
        helper('execution_partition' if 'execution_partition' in plan else 'test_execution').validate_against_ledger(plan, ledger)
    else:
        require(ledger['untouched_test_requests'] == 0, 'Do not run a new qualification after any test reservation')


def permit_context(permit_ref, *, check_dependencies=True):
    permit = read(permit_ref)
    protocol = execution_protocol(permit.get('execution_host'))
    require(permit.get('protocol') == protocol and permit.get('authority_host') == 'gpu-04' and
            permit.get('status') == 'committed' and
            permit.get('requests_never_refunded') is True and permit.get('single_use') is True,
            'Missing committed single-use GPU04 permit')
    require(permit.get('authority_permit_path') == str(AUTHORITY_ROOT / REGISTRY / permit['run_token'] / 'permit.json') and
            permit_ref['path'] == permit['authority_permit_path'], 'Permit is outside the canonical authority registry')
    manifest_path = resolve(permit['files'], permit['deployment_manifest']['path'])
    require(sha(manifest_path) == permit['deployment_manifest']['sha256'], 'Permit deployment identity changed')
    m = json.loads(manifest_path.read_bytes())
    ctx = validate_manifest(m, files=permit['files'], check_dependencies=check_dependencies)
    require(permit['execution_host'] == m['host'] and permit['run_token'] == m['run_token'] and permit['master_plan_sha256'] == MASTER and
            permit['counts'] == counts(ctx['request_plan']) and
            permit['reserved_wall_seconds'] == m['limits']['systemd_runtime_seconds'] and
            permit['gpu_ids'] == m['gpu_ids'] and permit['gpu_uuids'] == m['gpu_uuids'], 'Permit physical/count identity differs')
    ctx.update(manifest_path=manifest_path, manifest_sha256=permit['deployment_manifest']['sha256'],
               authority_manifest_sha256=permit['deployment_manifest']['sha256'], permit=permit, permit_ref=permit_ref,
               file_bindings=permit['files'], authority_bindings=ctx['authority']['files'])
    return ctx


def commit(root, deployment_ref, files):
    """GPU04-only atomic budget admission; never contacts or launches GPU02."""
    require(socket.gethostname() == 'gpu-04', 'Only GPU04 may issue an authoritative permit')
    root = Path(root); budget = helper('phase_budget')
    require(root == AUTHORITY_ROOT, 'Remote permit cannot reset the canonical GPU04 ledger root')
    original_path = deployment_ref['path']; local = resolve(files, original_path)
    require(sha(local) == deployment_ref['sha256'], 'Deployment changed before admission')
    m = json.loads(local.read_bytes())
    ctx = validate_manifest(m, files=files, check_dependencies=True)
    ctx.update(manifest_sha256=deployment_ref['sha256'])
    # Large model files are verified on their execution host; this permit binds
    # their exact reviewed hashes. Authority checks all original dependencies.
    for original, value in files.items():
        resolve(files, original)
    registry = root / REGISTRY; target = registry / m['run_token']
    with budget.publication_lock(root):
        ledger = budget.account(root, MASTER, PARENT)
        validate_budget(ctx, ledger)
        require(not target.exists(), 'Preserve already committed remote reservation')
        registry.mkdir(mode=0o700, exist_ok=True)
        require(not registry.is_symlink(), 'Unsafe remote registry')
        target.mkdir(mode=0o700)
        # permit.json is the sole admission marker; a missing permit draft is not
        # launchable and blocks another attempt with the same physical token.
        (target / 'input').mkdir()
        for original_file, destination in ((local, target / 'reviewed_manifest.json'),
                (ctx['request_plan_path'], target / 'input/request_plan.json')):
            with destination.open('xb') as stream:
                stream.write(original_file.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            destination.chmod(0o400)
        require(sha(target / 'reviewed_manifest.json') == deployment_ref['sha256'],
                'Original deployment bytes changed during admission')
        permit = {'protocol': execution_protocol(m['host']), 'authority_host': 'gpu-04', 'execution_host': m['host'], 'status': 'committed',
            'run_token': m['run_token'], 'master_plan_sha256': MASTER, 'deployment_manifest': deployment_ref,
            'authority_permit_path': str(target / 'permit.json'), 'files': files, 'counts': counts(ctx['request_plan']), 'reserved_wall_seconds': m['limits']['systemd_runtime_seconds'],
            'gpu_ids': m['gpu_ids'], 'gpu_uuids': m['gpu_uuids'], 'requests_never_refunded': True, 'single_use': True,
            'ledger_before': _budget_snapshot(ledger), 'at': time.time()}
        result = write(target / 'permit.json', permit)
        after = budget.account(root, MASTER, PARENT)
        require(all(after[k] == ledger[k] + permit['counts'][k] for k in COUNTERS), 'Remote reservation count did not commit exactly once')
        return result


def prepare_deployment(spec):
    """Build a portable LOCAL draft; it has no permit and cannot launch.

    Copies only JSON authority/plan/profile/tasks. The returned explicit map
    names source, prepared, and candidate files that a reviewed copier must put
    on GPU02; existing model files are hash-bound, never recopied by this helper.
    """
    require(socket.gethostname() == 'gpu-04' and spec.get('protocol') in PROTOCOLS.values() and
            spec.get('launch_enabled') is False, 'Portable preparation requires the GPU04 authority')
    output, stage = Path(spec['local_output']), Path(spec['remote_stage'])
    token = stage.name.removesuffix('-stage')
    require(stage.parent == AUTHORITY_ROOT and stage.name == token + '-stage' and
            re.fullmatch('codex-discovery-[a-z0-9-]{8,100}', token) and
            output.parent == AUTHORITY_ROOT and output.name == token + '-portable-preparation' and
            not output.exists(), 'Portable preparation needs fresh exact-token paths')
    authority, profile = read(spec['authority']), read(spec['profile'])
    protocol = execution_protocol(profile.get('host'))
    require(spec['protocol'] == profile.get('protocol') == authority.get('protocol') == protocol,
            'Portable preparation host/protocol differs')
    plan = read(authority['request_plan']); template = authority['task_template']
    if plan.get('mode') == 'generate' and 'test_execution' in plan:
        helper('test_execution').validate_part(plan)
    source_root = Path(spec['source_root']); source_files = {}
    require(source_root.is_dir() and not source_root.is_symlink(), 'Qualified source directory missing')
    for p in source_root.rglob('*'):
        require(not p.is_symlink(), 'Qualified source contains symlink')
        if p.is_file():
            require(not p.stat().st_mode & 0o222, 'Qualified source is mutable')
            source_files[str(p.relative_to(source_root))] = p
    require({key: sha(value) for key, value in source_files.items()} == profile['source_inventory'],
            'Qualified source inventory differs before portable preparation')
    ids, seconds = spec['gpu_ids'], spec['runtime_seconds']
    require(isinstance(ids, list) and ids == sorted(set(ids)) and ids and all(type(g) is int and 0 <= g < 8 for g in ids) and
            type(seconds) is int and 120 < seconds <= 14400 and len(plan['requests']) >= len(ids),
            'Invalid fresh remote allocation/runtime')
    max_requests = math.ceil(len(plan['requests']) / len(ids))
    require(plan['mode'] == 'qualify' or max_requests * 1536 / 14.3 + 300 < seconds - 120,
            'Conservative per-worker token ceiling exceeds reviewed deadline')
    root = AUTHORITY_ROOT
    ledger = helper('phase_budget').account(root, MASTER, PARENT)
    # All expensive static/resource/budget rejections precede any output write.
    new_counts = counts(plan)
    require(ledger['generation_requests'] + new_counts['generation_requests'] <= 4096 and
            ledger['tf_requests'] + new_counts['tf_requests'] <= 12000 and
            ledger['gpu_phase_wall_seconds'] + seconds + 180 <= 28800, 'Portable draft exceeds global budget')
    helper('phase_budget').assert_concurrency(ledger, len(ids))
    validate_authority_dependencies(authority, plan, profile)
    if plan['mode'] == 'generate':
        helper('execution_partition' if 'execution_partition' in plan else 'test_execution').validate_against_ledger(plan, ledger)
    else:
        require(len(plan['requests']) == len(ids) == 1 and ledger['untouched_test_requests'] == 0,
                'Qualification must use one GPU and precede untouched test')
    output.mkdir(mode=0o700)
    remote_source = stage / 'source'
    files, bound = {}, {}
    def add(original, local):
        original = str(original); local = Path(local)
        bound[original] = info(local); files[original] = {'path': str(local), **bound[original]}
    for name, local in source_files.items(): add(remote_source / name, local)
    for name, reference in [('authority', spec['authority']), ('profile', spec['profile']), ('request_plan', authority['request_plan'])]:
        local = output / (name + '.json')
        with local.open('xb') as stream:
            stream.write(check_file(reference['path'], reference).read_bytes()); stream.flush(); os.fsync(stream.fileno())
        local.chmod(0o400); add(stage / 'input' / (name + '.json'), local)
    prepared_path = authority['prepared_path']
    require(authority['files'].get(prepared_path) == authority['prepared'], 'Prepared authority source differs')
    add(prepared_path, prepared_path)  # Small same-path replica; no scientific field rewrite.
    for condition in plan['conditions'].values():
        for layer in condition.get('layers', []):
            if layer['kind'] == 'candidate':
                require(authority['files'].get(layer['path'], {}).get('sha256') == layer['sha256'], 'Unbound selected candidate')
                add(layer['path'], layer['path'])
    for item in profile['model_files']:
        bound[item['remote']] = {k: item[k] for k in ('sha256', 'size_bytes')}
    workers = []
    for index, gpu in enumerate(ids):
        name = 'gpu_' + str(gpu); task = copy.deepcopy(template)
        task.update(run_token=token, worker_name=name, gpu_id=gpu, mode=plan['mode'],
            output=str(root / (token + '-results') / 'workers' / name), deadline_seconds=seconds-120,
            requests=plan['requests'][index::len(ids)], conditions=plan['conditions'], prepared_records=prepared_path, **profile['model_paths'])
        local = output / (name + '.json'); write(local, task); remote_task = stage / 'tasks' / local.name
        add(remote_task, local)
        workers.append({'name': name, 'gpu_id': gpu, 'cpu_set': f'{96+2*gpu},{97+2*gpu}',
            'command': [profile['python'], str(remote_source / 'infra/gpu03/direction_discovery/engine.py'), '--task', str(remote_task)],
            'success_file': f'workers/{name}/SUCCESS.json', 'success_expect': {'mode': plan['mode'], 'requests': len(task['requests'])}})
    phase = ('remote_causal_qualification' if plan['mode'] == 'qualify' else
             'behavior_finalist_validation_round1' if 'execution_partition' in plan else 'behavior_untouched_test_partition')
    scientific = {'training': False, 'master_plan_sha256': MASTER, 'input_prepared_sha256': authority['prepared']['sha256'],
                  'previous_phase_budget': _budget_snapshot(ledger), 'phase_modes': [plan['mode']] * len(ids)}
    for key in ('execution_partition', 'test_execution'):
        if key in plan: scientific[key] = plan[key]
    if 'test_execution' in plan: scientific['test_bundle'] = plan['test_execution']['bundle']
    m = {'schema_version': 1, 'purpose': 'direction_discovery_campaign', 'phase': phase, 'host': profile['host'],
        'owner': 'researcher', 'run_token': token, 'stage': str(stage), 'source_root': str(remote_source),
        'output': str(root / (token + '-results')), 'runtime': str(root / (token + '-runtime')), 'python': profile['python'],
        'supervisor_cpu_set': '94', 'gpu_ids': ids, 'gpu_uuids': {str(g): profile['gpu_uuids'][str(g)] for g in ids},
        'authorization': spec['authorization'], 'runtime_versions': profile['runtime_versions'], 'scientific': scientific,
        'workers': workers, 'bound_files': bound, 'limits': {'runtime_seconds': seconds, 'systemd_runtime_seconds': seconds+180,
            'min_available_ram_gib': 192, 'max_worker_rss_gib': 128, 'cgroup_memory_gib': 160, 'min_start_free_disk_gib': 64,
            'max_worker_log_mib': 64, 'tasks_max': 512, 'load_stagger_seconds': 10},
        'remote_generation': {'protocol': protocol,
            'authority': {'path': str(stage / 'input/authority.json'), 'sha256': spec['authority']['sha256']},
            'profile': {'path': str(stage / 'input/profile.json'), 'sha256': spec['profile']['sha256']}}}
    manifest_path = stage / 'reviewed_manifest.json'
    m['command'] = [profile['python'], str(remote_source / 'infra/gpu03/direction_discovery/supervisor.py'), '--supervise', '--manifest', str(manifest_path)]
    local = output / 'reviewed_manifest.json'; write(local, m)
    files[str(manifest_path)] = {'path': str(local), **info(local)}
    ctx = validate_manifest(m, files=files, check_dependencies=True); ctx['manifest_sha256'] = sha(local)
    validate_budget(ctx, helper('phase_budget').account(root, MASTER, PARENT))
    return write(output / 'preparation.json', {'protocol': protocol, 'status': 'prepared_without_permit_or_launch',
        'deployment_manifest': {'path': str(manifest_path), 'sha256': sha(local)}, 'files': files,
        'existing_remote_model_files': profile['model_files'], 'counts': new_counts,
        'runtime_seconds': seconds, 'systemd_runtime_seconds': seconds+180,
        'request_ids_unchanged': True, 'model_work_launched': False,
        'token_ceiling_assumption': {'tokens_per_request': 1536, 'tokens_per_second': 14.3, 'extra_seconds': 300,
                                   'guaranteed_runtime': False}})


def validate_launch_permit(m, digest, permit_path, permit_sha, *, consume=False):
    """Execution-side check: frozen permit bytes, no remote authority traversal."""
    protocol = execution_protocol(m.get('host'))
    require(socket.gethostname() == m['host'], 'Remote permit may only execute on its bound host')
    permit = read({'path': str(permit_path), 'sha256': permit_sha})
    require(permit.get('protocol') == protocol and permit.get('status') == 'committed' and
            permit.get('authority_host') == 'gpu-04' and permit.get('execution_host') == m['host'] and
            permit.get('single_use') is True and permit.get('requests_never_refunded') is True and
            permit['run_token'] == m['run_token'] and permit['deployment_manifest'] == {'path': str(Path(m['stage']) / 'reviewed_manifest.json'), 'sha256': digest} and
            permit['gpu_ids'] == m['gpu_ids'] and permit['gpu_uuids'] == m['gpu_uuids'] and
            permit['reserved_wall_seconds'] == m['limits']['systemd_runtime_seconds'], 'Remote permit does not bind this deployment')
    require(permit.get('authority_permit_path') == str(AUTHORITY_ROOT / REGISTRY / m['run_token'] / 'permit.json'),
            'Launch permit lost canonical GPU04 registry provenance')
    ctx = validate_manifest(m)
    require(permit['counts'] == counts(ctx['request_plan']), 'Remote permit counts differ from actual task')
    require(permit['ledger_before'] == m['scientific']['previous_phase_budget'], 'Permit and deployment disagree about global history')
    marker = Path(m['stage']) / 'remote_permit_consumed.json'
    value = {'protocol': protocol, 'run_token': m['run_token'], 'manifest_sha256': digest,
             'permit_sha256': permit_sha, 'permit_path': str(permit_path)}
    if consume:
        write(marker, value)
    else:
        require(read(ref(marker)) == value, 'Missing or changed single-use permit consumption')
    return value


def validate_consumption(m, digest):
    marker = read(ref(Path(m['stage']) / 'remote_permit_consumed.json'))
    return validate_launch_permit(m, digest, marker['permit_path'], marker['permit_sha256'])


def _terminal(ctx, files):
    """Original receipt chain plus complete artifact and explicit live release proof."""
    m, digest = ctx['manifest'], ctx['manifest_sha256']
    get = lambda p: resolve(files, str(p))
    control, output = Path(m['stage']) / 'control', Path(m['output'])
    objects = {name: json.loads(get(control / name).read_bytes()) for name in
               ('launch_intent.json', 'service_started.json', 'supervisor_exit.json')}
    intent, started, end = (objects[x] for x in ('launch_intent.json', 'service_started.json', 'supervisor_exit.json'))
    for item in (intent, end):
        require(item.get('run_token') == m['run_token'] and item.get('manifest_sha256') == digest, 'Remote terminal receipt identity differs')
    fields = started.get('fields', {}); invocation = end.get('invocation_id')
    unit = m['run_token'] + '.service'
    require(started.get('unit') == unit and fields.get('ActiveState') == 'active' and fields.get('SubState') == 'running' and
            fields.get('KillMode') == 'control-group' and str(fields.get('MainPID', '')).isdigit() and int(fields['MainPID']) > 0 and
            Path(fields.get('ControlGroup', '')).name == unit and re.fullmatch('[0-9a-f]{32}', invocation or '') and
            fields.get('InvocationID') == invocation, 'Remote service startup/invocation is incomplete')
    require(end.get('service_result') == 'success' and end.get('exit_code_kind') == 'exited' and end.get('exit_status') == '0' and
            end.get('producer_summary_present') is True and end.get('failure_present') is False, 'Remote producer is not positively successful')
    times = [x.get('at') for x in (intent, started, end)]
    require(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in times) and times == sorted(times),
            'Remote wall-time chain invalid')
    summary = json.loads(get(output / 'campaign_summary.json').read_bytes())
    require(summary.get('status') == 'succeeded' and summary.get('run_token') == m['run_token'] and
            summary.get('manifest_sha256') == digest and summary.get('gpu_release_verified') is True and
            summary.get('worker_exit_codes') == [0] * len(m['workers']), 'Remote producer summary incomplete')
    artifacts_path = get(output / 'artifact_manifest.json'); artifacts = json.loads(artifacts_path.read_bytes())
    require(artifacts.get('algorithm') == 'sha256' and isinstance(artifacts.get('files'), dict) and artifacts['files'], 'Remote artifact inventory invalid')
    for relative, entry in artifacts['files'].items():
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and p.name != 'FAILURE.json', 'Unsafe or failed remote artifact')
        require(info(get(output / p)) == entry, 'Remote artifact replica differs')
    output_prefix = str(output) + '/'
    for original in files:
        if original.startswith(output_prefix) and original != str(output / 'artifact_manifest.json'):
            require(original[len(output_prefix):] in artifacts['files'], 'Replica contains an output omitted from the producer artifact inventory')
    require(sha(get(output / 'reviewed_manifest.json')) == digest, 'Producer copied another manifest')
    receipts, results = [], []
    for w in m['workers']:
        success = get(output / w['success_file']); value = json.loads(success.read_bytes())
        expected = {'status': 'succeeded', 'run_token': m['run_token'], 'worker_name': w['name'], **w['success_expect']}
        require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()), 'Remote worker success differs')
        receipts.append({'worker': w['name'], 'path': w['success_file'], 'sha256': sha(success), 'expected': expected})
        result = get((output / w['success_file']).parent / 'results.jsonl')
        results.append({'worker': w['name'], 'path': result, **info(result)})
    require(summary.get('worker_receipts') == receipts, 'Remote worker receipt coverage differs')
    release = json.loads(get(output / 'gpu_release.json').read_bytes())
    require(release.get('verified') is True and release.get('gpu_ids') == m['gpu_ids'], 'Remote selected GPUs were not released')
    return {'seconds': float(times[-1] - times[0]), 'invocation_id': invocation, 'control_group': fields['ControlGroup'],
            'artifact_manifest_sha256': sha(artifacts_path), 'results': results, 'terminal': end}


def context(package, verify=True):
    require(isinstance(package, dict) and set(package) == {'kind', 'replica'} and package['kind'] == 'remote_generation',
            'Explicit remote-generation package required')
    replica = read(package['replica'])
    protocol = execution_protocol(replica.get('origin_host'))
    require(replica.get('protocol') == protocol and replica.get('authority_host') == 'gpu-04',
            'Replica origin/authority differs')
    ctx = permit_context(replica['permit'], check_dependencies=True)
    require(ctx['manifest']['host'] == replica['origin_host'], 'Replica host differs from committed deployment')
    files = replica['files']
    # Tasks and authoritative envelopes retain original bytes and remote names.
    for original, entry in ctx['permit']['files'].items():
        require(original in files and files[original]['sha256'] == entry['sha256'] and files[original]['size_bytes'] == entry['size_bytes'],
                'Replica changed a committed deployment input')
    for original in files:
        resolve(files, original)
    metadata = validate_manifest(ctx['manifest'], files=files, check_dependencies=False)
    ctx.update(metadata)
    terminal = _terminal(ctx, files)
    independent = read(replica['remote_verification'])
    require(independent.get('protocol') == protocol and independent.get('status') == 'independently_verified_remote_generation' and
            independent.get('host') == ctx['manifest']['host'] and independent.get('run_token') == ctx['manifest']['run_token'] and
            independent.get('manifest_sha256') == ctx['manifest_sha256'] and
            independent.get('permit_sha256') == replica['permit']['sha256'] and
            independent.get('artifact_manifest_sha256') == terminal['artifact_manifest_sha256'] and
            independent.get('invocation_id') == terminal['invocation_id'] and independent.get('control_group') == terminal['control_group'] and
            independent.get('service_active_state') in ('inactive', 'failed') and independent.get('main_pid') == 0 and
            independent.get('own_cgroup_empty') is True and independent.get('gpu_release_verified') is True and
            independent.get('supervisor_verification') == {'status': 'verified', 'run_token': ctx['manifest']['run_token'],
                'manifest_sha256': ctx['manifest_sha256'], 'workers': len(ctx['tasks']),
                'artifact_files': len(json.loads(resolve(files, str(Path(ctx['manifest']['output']) / 'artifact_manifest.json')).read_bytes())['files']),
                'gpu_release_verified': True},
            'Remote independent terminal/release proof does not join the permit')
    proof = {'status': 'verified', 'kind': 'remote_generation', 'host': ctx['manifest']['host'], 'run_token': ctx['manifest']['run_token'],
             'manifest_sha256': ctx['manifest_sha256'], 'authority_manifest_sha256': ctx['manifest_sha256'],
             'permit_sha256': replica['permit']['sha256'], 'replica_sha256': package['replica']['sha256'],
             'remote_verification_sha256': replica['remote_verification']['sha256'],
             'artifact_manifest_sha256': terminal['artifact_manifest_sha256'], 'gpu_release_verified': True,
             'requests': len(ctx['request_plan']['requests']), 'workers': len(ctx['tasks']),
             'actual_wall_seconds': terminal['seconds'], 'invocation_id': terminal['invocation_id']}
    ctx.update(artifact_manifest_sha256=terminal['artifact_manifest_sha256'], results=terminal['results'], proof=proof,
               package=package, file_bindings=files, terminal=terminal['terminal'])
    ctx['bindings'] = sorted(set([Path(package['replica']['path']), Path(replica['permit']['path']), Path(replica['remote_verification']['path']),
        *[resolve(files, name) for name in files], *[Path(p) for p in ctx['authority']['files']]]))
    return ctx


def bindings(package, verify=False):
    return context(package, verify=verify)['bindings']


def reservation_entries(root, lineage):
    """Account every committed remote phase once, whether or not it ever ran."""
    root = Path(root); registry = root / REGISTRY
    if not registry.exists():
        return []
    require(not registry.is_symlink(), 'Remote registry cannot be symlinked')
    entries = []
    for directory in sorted(registry.iterdir()):
        require(directory.is_dir() and not directory.is_symlink(), 'Unexpected remote registry entry')
        permit_path = directory / 'permit.json'
        require(permit_path.is_file(), 'Incomplete remote admission needs explicit review; preserve draft')
        permit = read(ref(permit_path))
        require(permit.get('master_plan_sha256') in lineage, 'Remote registry contains another master lineage')
        ctx = permit_context(ref(permit_path), check_dependencies=False)
        m, plan = ctx['manifest'], ctx['request_plan']
        require(directory.name == m['run_token'] and sha(directory / 'reviewed_manifest.json') == ctx['manifest_sha256'] and
                json.loads((directory / 'input/request_plan.json').read_bytes()) == plan, 'Remote reservation mirror changed')
        phase_bindings = [permit_path, directory / 'reviewed_manifest.json', directory / 'input/request_plan.json',
                          *[resolve(permit['files'], p) for p in permit['files']]]
        seconds = float(permit['reserved_wall_seconds']); basis = 'full_reserved_deadline'; terminal = None
        completion = directory / 'completion.json'
        if completion.exists():
            package = read(ref(completion))
            done = context(package, verify=False)
            require(done['manifest_sha256'] == ctx['manifest_sha256'] and done['permit_ref'] == ref(permit_path),
                    'Remote completion belongs to another reservation')
            seconds = done['proof']['actual_wall_seconds']; basis = 'actual_launch_to_terminal_receipt'
            terminal = done['terminal']; phase_bindings.extend(done['bindings']); phase_bindings.append(completion)
        entries.append({'path': directory / 'reviewed_manifest.json', 'manifest': m, 'digest': ctx['manifest_sha256'],
            'bindings': phase_bindings, 'counts': permit['counts'], 'seconds': seconds, 'basis': basis,
            'execution_partition': plan.get('execution_partition'), 'test_execution': plan.get('test_execution'),
            'finalist_recovery': None, 'generation_ids': {r['request_id'] for r in plan['requests']} if plan['mode'] == 'generate' else set(),
            'remote_generation': {'permit': ref(permit_path), 'host': m['host']}, 'terminal': terminal})
    return entries


def import_completion(root, package):
    require(socket.gethostname() == 'gpu-04', 'Only GPU04 may import a remote terminal proof')
    require(Path(root) == AUTHORITY_ROOT, 'Remote import cannot change the canonical GPU04 ledger root')
    ctx = context(package)
    target = Path(root) / REGISTRY / ctx['manifest']['run_token']
    require(ctx['permit_ref'] == ref(target / 'permit.json'), 'Remote import is outside the authoritative registry')
    with helper('phase_budget').publication_lock(root):
        before = helper('phase_budget').account(root, MASTER, PARENT)
        write(target / 'completion.json', package)
        after = helper('phase_budget').account(root, MASTER, PARENT)
        require(all(before[k] == after[k] for k in COUNTERS), 'Terminal import changed committed request counts')
    return ctx['proof']


def verify_remote(path, digest, permit_path, permit_sha):
    """GPU02 independent after-exit proof, never a worker or launcher."""
    sup = helper('supervisor'); m = sup.load_manifest(path, digest)
    validate_launch_permit(m, digest, permit_path, permit_sha)
    proof = sup.verify(path, digest)
    import subprocess
    unit = m['run_token'] + '.service'
    state = subprocess.run(['systemctl', '--user', 'show', unit, '--property=LoadState,ActiveState,MainPID,ControlGroup,InvocationID'],
                           check=False, text=True, capture_output=True, timeout=15)
    fields = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
    require(state.returncode == 0 or state.returncode == 1 and fields.get('LoadState') == 'not-found',
            'Remote systemd query failed without a positively collected unit')
    started = json.loads((Path(m['stage']) / 'control/service_started.json').read_bytes())
    end = json.loads((Path(m['stage']) / 'control/supervisor_exit.json').read_bytes())
    group = started['fields']['ControlGroup']; invocation = end['invocation_id']
    require(fields.get('ActiveState') in ('inactive', 'failed') and fields.get('MainPID') == '0' and
            fields.get('InvocationID') in ('', invocation) and fields.get('ControlGroup') in ('', group),
            'Remote service is not positively terminal')
    cg = Path('/sys/fs/cgroup') / group.lstrip('/')
    require(group.startswith('/') and Path(group).name == unit and '..' not in Path(group).parts, 'Unsafe remote cgroup identity')
    require(not cg.exists() or all(not p.read_text().strip() for p in cg.rglob('cgroup.procs')), 'Remote exact cgroup remains populated')
    sup.raw.check_devices(m, sup.raw.gpu_snapshot())
    return {'protocol': execution_protocol(m['host']), 'status': 'independently_verified_remote_generation', 'host': m['host'],
            'run_token': m['run_token'], 'manifest_sha256': digest, 'permit_sha256': permit_sha,
            'artifact_manifest_sha256': sha(Path(m['output']) / 'artifact_manifest.json'),
            'invocation_id': invocation, 'control_group': group, 'service_active_state': fields['ActiveState'],
            'main_pid': 0, 'own_cgroup_empty': True, 'gpu_release_verified': True, 'supervisor_verification': proof}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-remote', type=Path, required=True); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--permit', type=Path, required=True); p.add_argument('--permit-sha256', required=True)
    args = p.parse_args()
    print(canonical(verify_remote(args.verify_remote, args.manifest_sha256, args.permit, args.permit_sha256)), end='')


if __name__ == '__main__':
    main()
