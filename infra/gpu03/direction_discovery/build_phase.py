"""Build an immutable per-phase review on gpu-04; this does not launch workers."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervisor


def verify_fixed_qualification(reference_path, prepared_sha256, **kwargs):
    from qualification_gate import verify_fixed_qualification as verify
    return verify(reference_path, prepared_sha256, **kwargs)


PARENT_PLAN_SHA256 = 'd4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'
AUX_PREPARED_SHA256 = 'd8f1bc40965b4ae63e6d0f8622dfb65c3c3aea0abb5b61ee28c3cf38efea3cc5'
UNION_PREPARED_SHA256 = '1048bb3d278c2f546be34700730893bef3dc18c0562b920a3aa12fae92faab7f'


def recovery_specs(request_plan, gpu_ids):
    from infra.gpu03.direction_discovery import finalist_recovery
    if gpu_ids != [5, 6, 7]:
        raise RuntimeError('This exact finalist recovery retains original GPU5,6,7 assignments')
    mapping = finalist_recovery.recovery_worker_requests(request_plan)
    if set(mapping) != {'gpu_5', 'gpu_6', 'gpu_7'} or [len(mapping[f'gpu_{g}']) for g in gpu_ids] != [40, 34, 40]:
        raise RuntimeError('Original recovery worker tails differ from40/34/40')
    covered = [r for g in gpu_ids for r in mapping[f'gpu_{g}']]
    if (len({r['request_id'] for r in covered}) != 114 or
            {r['request_id']: r for r in covered} != {r['request_id']: r for r in request_plan['requests']}):
        raise RuntimeError('Original worker recovery tails do not cover the exact missing114 requests')
    return [{'gpu_id': g, 'mode': 'generate', 'requests': mapping[f'gpu_{g}']} for g in gpu_ids]


def validate_master(plan_path, expected_sha256):
    if supervisor.sha256(plan_path) != expected_sha256:
        raise RuntimeError('Master plan changed')
    plan = json.loads(plan_path.read_text())
    if expected_sha256 == PARENT_PLAN_SHA256:
        return plan
    parent_path = plan_path.parent / 'parent_experiment_plan.json'
    if plan.get('plan_version') != 2 or plan.get('parent_plan_sha256') != PARENT_PLAN_SHA256 or supervisor.sha256(parent_path) != PARENT_PLAN_SHA256:
        raise RuntimeError('Master plan is neither pinned v1 nor its reviewed numerical repair')
    from amend_plan import verify_unchanged
    verify_unchanged(json.loads(parent_path.read_text()), plan)
    return plan


def validate_request_inputs(request_plan, master_sha256, prepared, core_sha256):
    if request_plan.get('master_plan_sha256') != master_sha256:
        raise RuntimeError('Causal request plan belongs to a different master plan')
    digest = supervisor.sha256(prepared)
    if 'no_loophole_capability' in request_plan:
        from infra.gpu03.direction_discovery import no_loophole_plan
        no_loophole_plan.validate_full(request_plan)
        if (Path(prepared) != Path(request_plan['prepared_records']) or
                digest != request_plan['prepared_records_sha256']):
            raise RuntimeError('No-loophole capability requires its exact derived prepared input')
        return
    if 'finalist_recovery' in request_plan:
        from infra.gpu03.direction_discovery import finalist_recovery
        full, _ = finalist_recovery.validate_plan(request_plan)
        if (Path(prepared) != Path(full['prepared_records']) or digest != core_sha256 or
                'test_execution' in request_plan or 'execution_partition' in request_plan):
            raise RuntimeError('Finalist recovery requires the exact original core prepared input')
        return
    if 'test_execution' in request_plan:
        from infra.gpu03.direction_discovery import test_execution
        full, meta = test_execution.validate_part(request_plan)
        bundle, _ = test_execution.bundle_context(meta['bundle'])
        expected = Path(meta['bundle']['path']).parent / bundle['prepared_records']['path']
        if (request_plan.get('mode') != 'generate' or request_plan.get('evaluation_partition') != 'untouched_test' or
                Path(prepared) != expected or Path(full['prepared_records']) != expected or
                digest != bundle['prepared_records']['sha256']):
            raise RuntimeError('Test execution prepared input differs from its exact frozen bundle')
        return
    if digest not in (core_sha256, AUX_PREPARED_SHA256, UNION_PREPARED_SHA256):
        raise RuntimeError('Unverified causal prepared input')


def validate_final_test(final, conditions, master_sha256, budget, *, request_plan=None):
    if request_plan is not None and 'test_execution' in request_plan:
        from infra.gpu03.direction_discovery import test_execution
        test_execution.validate_against_ledger(request_plan, budget)
    elif budget['untouched_test_requests'] != 0:
        raise RuntimeError('Untouched test requests were already committed; do not replay test')
    if (final.get('conditions') != conditions or final.get('status') != 'frozen_before_untouched_test' or
        final.get('master_plan_sha256') != master_sha256 or final.get('no_test_outcomes_used') is not True):
        raise RuntimeError('Test condition differs from the frozen pre-test finalist')


def _publication_paths(manifest):
    stage = Path(manifest['stage'])
    path = stage / 'reviewed_manifest.json'
    if (manifest['host'] != 'gpu-04' or socket.gethostname() != 'gpu-04' or
        manifest['owner'] != supervisor.OWNER or
        supervisor.pwd.getpwuid(os.getuid()).pw_name != supervisor.OWNER or
        manifest.get('purpose') != 'direction_discovery_campaign' or
        manifest['scientific'].get('training') is not False or
        stage.parent != supervisor.RUN_ROOT or stage.name != manifest['run_token'] + '-stage' or
        Path(manifest['source_root']) != stage / 'source' or
        manifest['command'] != supervisor.expected_command(manifest, path)):
        raise RuntimeError('Prepared publication identity/path differs')
    for key in ('output', 'runtime'):
        if Path(manifest[key]) != stage.parent / (manifest['run_token'] + '-' + ('results' if key == 'output' else 'runtime')):
            raise RuntimeError('Prepared publication output/runtime path differs')
    for blocked in (path, stage / 'control', Path(manifest['output']), Path(manifest['runtime'])):
        if blocked.exists() or blocked.is_symlink():
            raise RuntimeError('Publication requires an uncommitted, unlaunched stage: ' + str(blocked))
    return stage, path


def validate_publication_inputs(manifest, path):
    """Recheck the exact prepared source/tasks without rebuilding them."""
    for name, info in manifest['bound_files'].items():
        source = Path(name)
        if (not source.is_file() or source.stat().st_size != info['size_bytes'] or
                supervisor.sha256(source) != info['sha256']):
            raise RuntimeError('Prepared bound source/input changed: ' + name)
    source_root = Path(manifest['source_root'])
    for source in source_root.rglob('*'):
        if source.is_symlink():
            raise RuntimeError('Prepared source has a symlink')
        if source.is_file() and (str(source) not in manifest['bound_files'] or
                source.stat().st_uid != os.getuid() or source.stat().st_mode & 0o222 or
                source.name == '.DS_Store' or source.suffix == '.pyc' or '__pycache__' in source.parts):
            raise RuntimeError('Prepared source is unbound or mutable')
    for worker in manifest['workers']:
        command = worker['command']
        if (command[:3] != [manifest['python'], str(source_root / 'infra/gpu03/direction_discovery/engine.py'), '--task'] or
                len(command) != 4 or command[3] not in manifest['bound_files']):
            raise RuntimeError('Prepared worker command changed')
        task = json.loads(Path(command[3]).read_text())
        if (task['run_token'] != manifest['run_token'] or task['worker_name'] != worker['name'] or
                task['mode'] != worker['success_expect']['mode'] or
                len(task['requests']) != worker['success_expect']['requests'] or
                Path(task['output']) != (Path(manifest['output']) / worker['success_file']).parent):
            raise RuntimeError('Prepared worker task identity changed')
    if 'prelaunch_reallocation' in manifest['scientific']:
        import prelaunch_reallocation
        reference = json.loads((Path(manifest['stage']) / 'input/prelaunch_reallocation_reference.json').read_text())
        old_path = Path(reference['old_manifest_path'])
        old = json.loads(old_path.read_text())
        pending = prelaunch_reallocation.validate_pending(old_path, old, Path(reference['proof_path']))
        prelaunch_reallocation.validate_pair(old_path, old, path, manifest, pending)


def validate_publication_budget(manifest):
    import phase_budget
    stage = Path(manifest['stage'])
    plan = validate_master(stage / 'input/experiment_plan.json', manifest['scientific']['master_plan_sha256'])
    transfer = 'prelaunch_reallocation' in manifest['scientific']
    budget = phase_budget.account(stage.parent, manifest['scientific']['master_plan_sha256'], plan.get('parent_plan_sha256'),
                                 **({'replacement_stage': stage} if transfer else {}))
    prior, wall = effective_budget(budget, stage, transfer)
    # The same global concurrency boundary applies to local and remote admission.
    phase_budget.assert_concurrency(budget, len(manifest['gpu_ids']))
    part_path = stage / 'input/request_plan.json'
    if part_path.is_file():
        part = json.loads(part_path.read_text())
        if 'no_loophole_capability' in part:
            from infra.gpu03.direction_discovery import no_loophole_plan
            no_loophole_plan.validate_against_ledger(part, budget)
            no_loophole_plan.bindings(part, verify=True)
            no_loophole_plan.validate_source(part, Path(manifest['source_root']))
            if (manifest['phase'] != 'no_loophole_capability' or
                    manifest.get('authorization') != part['authorization'] or
                    manifest['scientific'].get('no_loophole_capability') != part['no_loophole_capability']):
                raise RuntimeError('Published no-loophole capability metadata differs from its new plan')
            validate_request_inputs(part, manifest['scientific']['master_plan_sha256'],
                                    Path(part['prepared_records']), plan['inputs']['prepared_records_sha256'])
            covered = []
            for worker in manifest['workers']:
                task = json.loads(Path(worker['command'][3]).read_text())
                if (task['mode'] != 'generate' or task['conditions'] != part['conditions'] or
                        task['sampling'] != plan['sampling'] or
                        task['prepared_records'] != part['prepared_records'] or
                        task.get('attention_policy') != 'exclusive_math' or
                        task.get('teacher_forced_padded_sequence_length') != 2176):
                    raise RuntimeError('Published no-loophole worker changed the matched inference protocol')
                covered.extend(task['requests'])
            if (len(covered) != len(part['requests']) or len({r['request_id'] for r in covered}) != len(covered) or
                    {r['request_id']: r for r in covered} != {r['request_id']: r for r in part['requests']}):
                raise RuntimeError('Published no-loophole workers do not cover the exact 740 requests')
        elif 'no_loophole_capability' in manifest['scientific'] or manifest.get('phase') == 'no_loophole_capability':
            raise RuntimeError('Published manifest lost its no-loophole capability plan')
        if 'finalist_recovery' in part:
            from infra.gpu03.direction_discovery import finalist_recovery
            finalist_recovery.validate_against_ledger(part, budget)
            finalist_recovery.bindings(part, verify=True)
            if manifest['scientific'].get('finalist_recovery') != part['finalist_recovery']:
                raise RuntimeError('Published finalist recovery metadata differs from its missing-request plan')
            validate_request_inputs(part, manifest['scientific']['master_plan_sha256'],
                                    Path(part['prepared_records']), plan['inputs']['prepared_records_sha256'])
            expected_specs = recovery_specs(part, manifest['gpu_ids'])
            if len(manifest['workers']) != len(expected_specs):
                raise RuntimeError('Published recovery worker allocation differs')
            for worker, expected in zip(manifest['workers'], expected_specs):
                task = json.loads(Path(worker['command'][3]).read_text())
                if worker['gpu_id'] != expected['gpu_id'] or task['requests'] != expected['requests']:
                    raise RuntimeError('Published recovery changed original device or tail order')
        elif 'finalist_recovery' in manifest['scientific']:
            raise RuntimeError('Published manifest lost its finalist recovery plan')
        if 'execution_partition' in part:
            import execution_partition
            execution_partition.validate_against_ledger(part, budget)
            execution_partition.predecessor_bindings(part, verify=True)
            if manifest['scientific'].get('execution_partition') != part['execution_partition']:
                raise RuntimeError('Published execution partition metadata differs from task plan')
        if 'test_execution' in part:
            from infra.gpu03.direction_discovery import test_execution
            test_execution.validate_against_ledger(part, budget)
            test_execution.bindings(part, verify=True)
            if (manifest['scientific'].get('test_execution') != part['test_execution'] or
                    manifest['scientific'].get('test_bundle') != part['test_execution']['bundle']):
                raise RuntimeError('Published test execution metadata differs from its frozen bundle')
            validate_request_inputs(part, manifest['scientific']['master_plan_sha256'],
                                    Path(part['prepared_records']), plan['inputs']['prepared_records_sha256'])
        elif 'test_execution' in manifest['scientific'] or 'test_bundle' in manifest['scientific']:
            raise RuntimeError('Published manifest lost its test execution plan')
    elif 'no_loophole_capability' in manifest['scientific'] or manifest.get('phase') == 'no_loophole_capability':
        raise RuntimeError('Published manifest lost its no-loophole capability plan')
    saved = manifest['scientific']['previous_phase_budget']
    for kind in ('tf', 'generation'):
        expected = manifest['scientific'].get('previously_committed_' + kind + '_requests', saved[kind + '_requests'])
        if prior[kind + '_requests'] != expected:
            raise RuntimeError('Request commitments changed before publication')
    tf = sum(w['success_expect']['requests'] * (3 if w['success_expect']['mode'] == 'qualify' else
             1 if w['success_expect']['mode'] == 'tf' else 0) for w in manifest['workers'])
    generation = sum(w['success_expect']['requests'] * (3 if w['success_expect']['mode'] == 'qualify' else
                     1 if w['success_expect']['mode'] == 'generate' else 0) for w in manifest['workers'])
    if (prior['tf_requests'] + tf > plan['budget']['maximum_teacher_forced_forwards'] or
        prior['generation_requests'] + generation > plan['budget']['maximum_new_free_generations_including_qualification'] or
        wall + manifest['limits']['systemd_runtime_seconds'] > plan['budget']['maximum_aggregate_gpu_phase_wall_seconds']):
        raise RuntimeError('Prepared publication exceeds current cumulative budget')


def publish_prepared_manifest(prepared_path, prepared_sha256):
    """Publish only; never rebuild tasks or launch a controller/model."""
    prepared_path = Path(prepared_path)
    if (not prepared_path.is_file() or prepared_path.is_symlink() or prepared_path.stat().st_mode & 0o222 or
            supervisor.sha256(prepared_path) != prepared_sha256):
        raise RuntimeError('Prepared manifest is missing, mutable, or hash-mismatched')
    envelope = json.loads(prepared_path.read_text())
    if (envelope.get('purpose') != 'unpublished_direction_discovery_manifest' or envelope.get('schema_version') != 1 or
            envelope.get('publisher_source_sha256') != supervisor.sha256(__file__)):
        raise RuntimeError('Prepared envelope or publisher source differs')
    manifest = envelope['manifest']
    stage, path = _publication_paths(manifest)
    if prepared_path != stage / 'prepared_manifest.json' or envelope['reviewed_manifest_path'] != str(path):
        raise RuntimeError('Prepared envelope destination differs')
    validate_publication_inputs(manifest, path)
    import phase_budget
    # Source/input hashing precedes this short shared local/remote admission
    # barrier. Recount commitments and publish exactly once while holding it.
    with phase_budget.publication_lock(stage.parent):
        validate_publication_budget(manifest)
        inventory = supervisor.raw.gpu_snapshot()
        supervisor.raw.check_devices(manifest, inventory)
        _publication_paths(manifest)
        manifest['inventory_before_manifest'] = inventory
        manifest['bound_files'][str(prepared_path)] = {'sha256': prepared_sha256, 'size_bytes': prepared_path.stat().st_size}
        supervisor.exclusive_json(path, manifest)
        path.chmod(0o400)
    return {'status': 'published_without_launch', 'manifest': str(path), 'manifest_sha256': supervisor.sha256(path),
            'prepared_manifest': str(prepared_path), 'prepared_manifest_sha256': prepared_sha256,
            'model_work_launched': False}


def publish_idle_manifest(path, manifest):
    path = Path(path)
    stage, expected = _publication_paths(manifest)
    if path != expected:
        raise RuntimeError('Prepared publication target differs')
    prepared = stage / 'prepared_manifest.json'
    # The envelope intentionally fails supervisor.load_manifest's purpose check;
    # an unpublished draft can neither launch nor appear in the request ledger.
    supervisor.exclusive_json(prepared, {'schema_version': 1, 'purpose': 'unpublished_direction_discovery_manifest',
        'publisher_source_sha256': supervisor.sha256(__file__), 'reviewed_manifest_path': str(path), 'manifest': manifest})
    prepared.chmod(0o400)
    return publish_prepared_manifest(prepared, supervisor.sha256(prepared))


def effective_budget(budget, stage, transfer):
    context = budget.get('replacement_context')
    if transfer:
        if not context or context.get('destination_stage') != str(stage):
            raise RuntimeError('Reallocation lacks an exact pending reservation transfer')
        return context['effective_previous_counts'], context['effective_previous_gpu_phase_wall_seconds']
    if context:
        raise RuntimeError('Unexpected reservation transfer for an ordinary phase')
    return budget, budget['gpu_phase_wall_seconds']


def verify_causal_qualification(reference_path, source_root, prior, prepared_sha256):
    reference=json.loads(reference_path.read_text())
    path=Path(reference['manifest_path'])
    checked=supervisor.verify(path,reference['manifest_sha256'])
    m=json.loads(path.read_text());out=Path(m['output'])
    if (checked['status']!='verified' or m['phase']!='causal_qualification' or
        m['scientific']['input_prepared_sha256']!=prepared_sha256 or len(m['workers'])!=1 or
        supervisor.sha256(out/'artifact_manifest.json')!=reference['artifact_manifest_sha256']):
        raise RuntimeError('Actual-inference qualification is not verified for this dataset')
    worker=m['workers'][0];task_path=Path(worker['command'][3]);task=json.loads(task_path.read_text())
    if (worker['success_expect']!={'mode':'qualify','requests':1} or task['mode']!='qualify' or
        task['model_snapshot']!=prior['model_snapshot'] or task['checkpoint']!=prior['checkpoint'] or
        task['teacher_forced_padded_sequence_length']!=2176 or task['attention_policy']!='exclusive_math'):
        raise RuntimeError('Actual-inference qualification used a different model or protocol')
    prepared_path=Path(task['prepared_records'])
    if supervisor.sha256(prepared_path)!=prepared_sha256:
        raise RuntimeError('Actual-inference qualification prepared input changed')
    # The historical file is hash-bound above. Select qualification metadata
    # before deserializing so this validation need not read untouched test rows.
    with prepared_path.open('rb') as stream:
        selected=[json.loads(line) for line in stream
                  if re.search(rb'(?<!\\)"record_index"\s*:\s*222\s*[,}]', line)]
    if (len(selected)!=1 or selected[0]['problem_split']!='direction_fit' or
        task['requests']!=[{'request_id':'causal-qualification-'+selected[0]['record_id'],
                           'record_id':selected[0]['record_id'],'condition_id':'baseline'}] or
        task['conditions']!={'baseline':{'layers':[]}}):
        raise RuntimeError('Actual-inference qualification request identity is wrong')
    files=[reference_path,path,out/'artifact_manifest.json',task_path,prepared_path,Path(m['stage'])/'control/supervisor_exit.json']
    for relative in ['infra/gpu03/direction_discovery/engine.py','infra/gpu03/direction_discovery/intervention.py',
                     'infra/gpu03/activation_dataset/extract_triplet_raw.py','infra/gpu03/activation_dataset/extract_delta_activations.py']:
        old=Path(m['source_root'])/relative;new=Path(source_root)/relative
        if supervisor.sha256(new)!=m['bound_files'][str(old)]['sha256']:
            raise RuntimeError('Actual-inference source changed after qualification: '+relative)
        files.append(old)
    result_path=out/'workers'/worker['name']/'results.jsonl'
    values=[json.loads(line) for line in result_path.read_text().splitlines()]
    if (len(values)!=1 or values[0]['problem_split']!='direction_fit' or
        any(values[0].get(key)!=value for key,value in task['requests'][0].items()) or
        values[0].get('problem_id')!=selected[0]['problem_id']):
        raise RuntimeError('Actual-inference qualification coverage/split is wrong')
    result=values[0]['result']
    if any(result.get(key) is not True for key in ['baseline_recovery_bitwise','teacher_forced_effect_verified','baseline_generation_repeatable']):
        raise RuntimeError('Actual-inference qualification numerical check failed')
    base=result['baseline_generation']['generated_token_ids']
    repeated=result['repeated_baseline_generation']['generated_token_ids']
    projected=result['projected_generation']
    if base!=repeated or len(base)!=8 or len(projected['generated_token_ids'])!=8:
        raise RuntimeError('Actual-inference qualification sampling check failed')
    if set(projected['energy'])!={'12'}:
        raise RuntimeError('Actual-inference qualification layer coverage is wrong')
    energy=projected['energy']['12']
    for key in ['removed_energy_fp32','remaining_subspace_energy_fp32','activation_energy']:
        value=energy[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise RuntimeError('Actual-inference qualification energy is invalid')
    if (energy['rank']!=1 or energy['selected_tokens']!=8 or energy['forward_calls']!=8 or
        energy['scopes']['prefill']['selected_tokens']!=1 or energy['scopes']['decode']['selected_tokens']!=7 or
        not energy['removed_energy_fp32']>0 or
        energy['remaining_subspace_energy_fp32']>energy['removed_energy_fp32']*1e-8):
        raise RuntimeError('Actual-inference qualification did not remove the projected subspace at prediction positions')
    files.append(result_path)
    return files


def build(stage, phase, gpu_ids, runtime_seconds, master_plan_sha256='d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'):
    if socket.gethostname() != 'gpu-04':
        raise RuntimeError('Resolve live manifests only on gpu-04')
    stage = stage.absolute()
    token = stage.name.removesuffix('-stage')
    root = stage.parent
    plan_path = stage / 'input/experiment_plan.json'
    plan = validate_master(plan_path, master_plan_sha256)
    prior_path = stage / 'input/prior_raw_reviewed_manifest.json'
    if supervisor.sha256(prior_path) != plan['inputs']['prior_raw_review_manifest_sha256']:
        raise RuntimeError('Prior verified model/path manifest changed')
    prior = json.loads(prior_path.read_text())
    if type(runtime_seconds) is not int or not 120 < runtime_seconds <= plan['budget']['maximum_gpu_phase_wall_seconds']:
        raise RuntimeError('Runtime exceeds frozen phase budget')
    prepared = Path(prior['prepared']) / 'prepared_records.jsonl'
    if supervisor.sha256(prepared) != plan['inputs']['prepared_records_sha256']:
        raise RuntimeError('Prepared records changed')
    raw_package = Path(plan['inputs'].get('raw_package', prior['output']))
    if supervisor.sha256(raw_package / 'artifact_manifest.json') != plan['inputs']['raw_artifact_manifest_sha256']:
        raise RuntimeError('Raw package manifest changed')
    if phase == 'no_loophole_capability':
        # The old core remains an opaque provenance binding; this new phase
        # deserializes only its separately verified 37 validation carriers.
        capability_plan = json.loads((stage / 'input/request_plan.json').read_text())
        if 'no_loophole_capability' not in capability_plan:
            raise RuntimeError('No-loophole capability phase requires an explicit derived plan')
        prepared = Path(capability_plan['prepared_records'])
        validate_request_inputs(capability_plan, master_plan_sha256, prepared,
                                plan['inputs']['prepared_records_sha256'])
        rows = []
    else:
        rows = [json.loads(line) for line in prepared.read_text().splitlines()]
    by_index = {row['record_index']: row for row in rows}
    source = stage / 'source'
    output = root / (token + '-results')
    runtime = root / (token + '-runtime')
    task_root = stage / 'tasks'
    task_root.mkdir(exist_ok=False)
    requests_plan_path = stage / 'input/request_plan.json'
    request_plan = {}
    repair_bindings = []
    import phase_budget
    transfer_reference_path = stage / 'input/prelaunch_reallocation_reference.json'
    transfer = transfer_reference_path.exists()
    budget = phase_budget.account(root, master_plan_sha256, plan.get('parent_plan_sha256'),
                                  **({'replacement_stage': stage} if transfer else {}))
    prior_counts, prior_wall_seconds = effective_budget(budget, stage, transfer)
    if prior_wall_seconds + runtime_seconds + 180 > plan['budget']['maximum_aggregate_gpu_phase_wall_seconds']:
        raise RuntimeError('Aggregate GPU phase wall-time budget exhausted')
    repair_bindings.extend(budget['bindings'])
    if phase.startswith('fixed_cache_'):
        reference_path = stage / 'input/prefix_audit_reference.json'
        reference = json.loads(reference_path.read_text())
        checked = supervisor.verify(Path(reference['manifest_path']), reference['manifest_sha256'])
        if checked['status'] != 'verified':
            raise RuntimeError('Numerical diagnosis is not independently verified')
        diagnosis = json.loads(Path(reference['manifest_path']).read_text())
        if diagnosis['phase'] != 'prefix_audit' or diagnosis['scientific']['input_prepared_sha256'] != plan['inputs']['prepared_records_sha256']:
            raise RuntimeError('Prefix diagnosis belongs to a different dataset or protocol')
        if supervisor.sha256(Path(diagnosis['output']) / 'artifact_manifest.json') != reference['artifact_manifest_sha256']:
            raise RuntimeError('Numerical diagnostic artifact manifest changed')
        verdict_path = Path(diagnosis['output']) / 'workers/gpu_0/prefix_audit_verdict.json'
        if json.loads(verdict_path.read_text())['passed'] is not True:
            raise RuntimeError('Controlled prefix diagnosis failed')
        repair_bindings.extend([Path(reference['manifest_path']), verdict_path,
                           Path(diagnosis['output']) / 'artifact_manifest.json',
                           Path(diagnosis['output']) / 'workers/gpu_0/prefix_audit_all_measurements.json'])
    if phase == 'fixed_cache_qualification':
        if len(gpu_ids) != 1:
            raise RuntimeError('Fixed-cache qualification uses one idle GPU')
        specs = [{'gpu_id': gpu_ids[0], 'mode': 'fixed_cache', 'requests': [
            {'request_id': f'fixed-cache-qual-{by_index[i]["record_id"]}', 'record_id': by_index[i]['record_id'],
             'condition_id': 'baseline'} for i in [222, 223, 224]]}]
        conditions = {'baseline': {'layers': []}}
    elif phase in ['fixed_cache_core', 'fixed_cache_auxiliary']:
        repair_bindings.extend(verify_fixed_qualification(stage / 'input/fixed_cache_qualification_reference.json',
                                                         plan['inputs']['prepared_records_sha256'], source_root=source,
                                                         model_snapshot=prior['model_snapshot'], checkpoint=prior['checkpoint']))
        if phase == 'fixed_cache_auxiliary':
            prepared = Path(json.loads(requests_plan_path.read_text())['prepared_records'])
            rows = [json.loads(line) for line in prepared.read_text().splitlines()]
            if len(rows) != 51 or supervisor.sha256(prepared) != 'd8f1bc40965b4ae63e6d0f8622dfb65c3c3aea0abb5b61ee28c3cf38efea3cc5':
                raise RuntimeError('Reviewed auxiliary records changed')
        problems = sorted({str(row['problem_id']) for row in rows})
        assignment = {problem: gpu_ids[i % len(gpu_ids)] for i, problem in enumerate(problems)}
        specs = [{'gpu_id': gpu, 'mode': 'fixed_cache', 'requests': [
            {'request_id': f'fixed-cache-{r["record_id"]}', 'record_id': r['record_id'], 'condition_id': 'baseline'}
            for r in rows if assignment[str(r['problem_id'])] == gpu]} for gpu in gpu_ids]
        conditions = {'baseline': {'layers': []}}
    elif phase == 'prefix_audit':
        if len(gpu_ids) != 1:
            raise RuntimeError('Numerical audit uses one idle GPU')
        specs = [{'gpu_id': gpu_ids[0], 'mode': 'prefix_audit', 'requests': [
            {'request_id': f'prefix-audit-{by_index[i]["record_id"]}', 'record_id': by_index[i]['record_id'],
             'condition_id': 'baseline'} for i in [222, 223, 224]]}]
        conditions = {'baseline': {'layers': []}}
    elif phase == 'causal_qualification':
        if len(gpu_ids) != 1 or budget['generation_requests'] + 3 > 4096 or budget['tf_requests'] + 3 > 12000:
            raise RuntimeError('Causal qualification allocation or request budget invalid')
        specs = [{'gpu_id': gpu_ids[0], 'mode': 'qualify', 'requests': [
            {'request_id': 'causal-qualification-' + by_index[222]['record_id'],
             'record_id': by_index[222]['record_id'], 'condition_id': 'baseline'}]}]
        conditions = {'baseline': {'layers': []}}
    elif phase == 'qualification':
        if gpu_ids != [0, 1] or budget['generation_requests'] + 3 > 4096 or budget['tf_requests'] + 3 > 12000:
            raise RuntimeError('Qualification allocation requires reviewed GPU0 and GPU1')
        specs = [(0, 'supplement', [by_index[i] for i in [222, 223, 224]]),
                 (1, 'qualify', [by_index[222]])]
        specs = [{'gpu_id': gpu, 'mode': mode,
                  'requests': [{'request_id': f'{mode}-{r["record_id"]}', 'record_id': r['record_id'],
                                'condition_id': 'baseline'} for r in rr]} for gpu, mode, rr in specs]
        conditions = {'baseline': {'layers': []}}
    elif phase == 'supplement':
        problems = sorted({str(row['problem_id']) for row in rows})
        assignment = {problem: gpu_ids[i % len(gpu_ids)] for i, problem in enumerate(problems)}
        specs = [{'gpu_id': gpu, 'mode': 'supplement', 'requests': [
            {'request_id': f'supplement-{r["record_id"]}', 'record_id': r['record_id'], 'condition_id': 'baseline'}
            for r in rows if assignment[str(r['problem_id'])] == gpu]} for gpu in gpu_ids]
        conditions = {'baseline': {'layers': []}}
    else:
        request_plan = json.loads(requests_plan_path.read_text())
        if request_plan['mode'] not in ['cache_aux', 'tf', 'generate']:
            raise RuntimeError('Invalid request plan mode')
        if request_plan['mode'] in ['tf', 'generate']:
            repair_bindings.extend(verify_causal_qualification(stage/'input/causal_qualification_reference.json',
                                      source, prior, plan['inputs']['prepared_records_sha256']))
        prepared = Path(request_plan.get('prepared_records', prepared))
        validate_request_inputs(request_plan, master_plan_sha256, prepared, plan['inputs']['prepared_records_sha256'])
        conditions = request_plan['conditions']
        requests = request_plan['requests']
        if 'no_loophole_capability' in request_plan:
            from infra.gpu03.direction_discovery import no_loophole_plan
            if phase != 'no_loophole_capability':
                raise RuntimeError('The no-loophole plan requires its separate capability phase')
            no_loophole_plan.validate_against_ledger(request_plan, budget)
            no_loophole_plan.validate_source(request_plan, source)
            repair_bindings.extend(no_loophole_plan.bindings(request_plan, verify=True))
        elif phase == 'no_loophole_capability':
            raise RuntimeError('No-loophole capability phase requires an explicit derived plan')
        if 'finalist_recovery' in request_plan:
            from infra.gpu03.direction_discovery import finalist_recovery
            finalist_recovery.validate_against_ledger(request_plan, budget)
            repair_bindings.extend(finalist_recovery.bindings(request_plan, verify=True))
        if 'execution_partition' in request_plan:
            import execution_partition
            execution_partition.validate_against_ledger(request_plan, budget)
            repair_bindings.extend(execution_partition.predecessor_bindings(request_plan, verify=True))
        if 'test_execution' in request_plan:
            from infra.gpu03.direction_discovery import test_execution
            test_execution.validate_against_ledger(request_plan, budget)
            repair_bindings.extend(test_execution.bindings(request_plan, verify=True))
        if len({r['request_id'] for r in requests}) != len(requests):
            raise RuntimeError('Duplicate request IDs')
        # Prior requests remain spent across the reviewed numerical repair.
        used_generations, used_tf = prior_counts['generation_requests'], prior_counts['tf_requests']
        prior_phase_bindings = budget['bindings']
        for key, actual in [('previously_committed_generation_requests', used_generations),
                            ('previously_committed_tf_requests', used_tf)]:
            if type(request_plan.get(key)) is not int or request_plan[key] != actual:
                raise RuntimeError('Prior request accounting disagrees with frozen phases: ' + key)
        if request_plan['mode'] == 'generate' and len(requests) + used_generations > 4096:
            raise RuntimeError('Generation budget exceeded')
        if request_plan['mode'] == 'tf' and len(requests) + used_tf > 12000:
            raise RuntimeError('Teacher-forcing budget exceeded')
        selected_rows = {r['record_id']: r for r in [json.loads(line) for line in prepared.read_text().splitlines()]}
        partitions = {selected_rows[r['record_id']]['problem_split'] for r in requests}
        if request_plan['mode'] in ['tf', 'generate']:
            partition = request_plan['evaluation_partition']
            if partitions != {partition} or partition not in ['configuration_validation', 'untouched_test']:
                raise RuntimeError('Causal requests cross evaluation partitions')
            if partition == 'untouched_test':
                final_path = Path(request_plan['frozen_final_config'])
                if supervisor.sha256(final_path) != request_plan['frozen_final_config_sha256']:
                    raise RuntimeError('Final configuration changed before test')
                final = json.loads(final_path.read_text())
                validate_final_test(final, conditions, master_plan_sha256, budget, request_plan=request_plan)
                prior_phase_bindings.append(final_path)
        elif 'untouched_test' in partitions:
            raise RuntimeError('Preselection auxiliary cache cannot read test records')
        specs = [{'gpu_id': gpu, 'mode': request_plan['mode'], 'requests': requests[i::len(gpu_ids)]}
                 for i, gpu in enumerate(gpu_ids)]
        if 'finalist_recovery' in request_plan:
            specs = recovery_specs(request_plan, gpu_ids)
    python = prior['command'][0]
    workers = []
    for spec in specs:
        gpu = spec['gpu_id']
        name = f'gpu_{gpu}'
        if not spec['requests']:
            raise RuntimeError('Do not allocate empty workers')
        task_path = task_root / (name + '.json')
        task = {'run_token': token, 'worker_name': name, 'gpu_id': gpu, 'mode': spec['mode'],
                'model_snapshot': prior['model_snapshot'], 'checkpoint': prior['checkpoint'],
                'raw_package': str(raw_package), 'prepared_records': str(prepared),
                'output': str(output / 'workers' / name), 'deadline_seconds': runtime_seconds - 120,
                'requests': spec['requests'], 'conditions': conditions, 'sampling': plan['sampling']}
        if phase == 'prefix_audit':
            task['prefix_audit_fp32'] = True
        if spec['mode'] in ['qualify', 'tf', 'generate']:
            task['attention_policy'] = 'exclusive_math'
            task['teacher_forced_padded_sequence_length'] = 2176
        if phase.startswith('fixed_cache_'):
            task.update({'padded_sequence_length': 2176, 'pad_token_id': 151643,
                         'cache_role': 'auxiliary' if phase == 'fixed_cache_auxiliary' else 'core',
                         'manifest_path': str(stage / 'reviewed_manifest.json'), 'attention_policy': 'exclusive_math'})
        supervisor.exclusive_json(task_path, task)
        workers.append({'name': name, 'gpu_id': gpu, 'cpu_set': f'{96 + 2*gpu},{97 + 2*gpu}',
                        'command': [python, str(source / 'infra/gpu03/direction_discovery/engine.py'), '--task', str(task_path)],
                        'success_file': f'workers/{name}/SUCCESS.json',
                        'success_expect': {'requests': len(spec['requests']), 'mode': spec['mode']}})
    inventory = supervisor.raw.gpu_snapshot()
    gpu_map = {g['index']: g for g in inventory['gpus']} if isinstance(inventory, dict) else {g['index']: g for g in inventory}
    manifest_path = stage / 'reviewed_manifest.json'
    m = {'schema_version': 1, 'purpose': 'direction_discovery_campaign', 'phase': phase,
         'run_token': token, 'host': 'gpu-04', 'owner': 'researcher',
         'stage': str(stage), 'source_root': str(source), 'output': str(output), 'runtime': str(runtime),
         'python': python, 'supervisor_cpu_set': '94', 'gpu_ids': gpu_ids,
         'gpu_uuids': {str(g): gpu_map[g]['uuid'] for g in gpu_ids},
         'authorization': plan['authorization'], 'scientific': {'training': False, 'master_plan_sha256': supervisor.sha256(plan_path),
         'phase_modes': [s['mode'] for s in specs], 'input_prepared_sha256': supervisor.sha256(prepared),
         'previous_phase_budget': {k: v for k, v in budget.items() if k != 'bindings'}},
         'workers': workers, 'runtime_versions': prior['runtime_versions'],
         'limits': {'runtime_seconds': runtime_seconds, 'systemd_runtime_seconds': runtime_seconds + 180,
                    'min_available_ram_gib': 192, 'max_worker_rss_gib': 128, 'cgroup_memory_gib': 160,
                    'min_start_free_disk_gib': 64, 'max_worker_log_mib': 64, 'tasks_max': 512, 'load_stagger_seconds': 10},
         'bound_files': {}, 'inventory_before_manifest': inventory}
    if requests_plan_path.is_file() and 'execution_partition' in request_plan:
        m['scientific']['execution_partition'] = request_plan['execution_partition']
    if requests_plan_path.is_file() and 'no_loophole_capability' in request_plan:
        m['scientific']['no_loophole_capability'] = request_plan['no_loophole_capability']
        m['authorization'] = request_plan['authorization']
    if requests_plan_path.is_file() and 'finalist_recovery' in request_plan:
        m['scientific']['finalist_recovery'] = request_plan['finalist_recovery']
    if requests_plan_path.is_file() and 'test_execution' in request_plan:
        m['scientific']['test_execution'] = request_plan['test_execution']
        m['scientific']['test_bundle'] = request_plan['test_execution']['bundle']
    m['command'] = supervisor.expected_command(m, manifest_path)
    if transfer:
        transfer_reference = json.loads(transfer_reference_path.read_text())
        m['scientific']['prelaunch_reallocation'] = {
            'old_manifest_sha256': transfer_reference['old_manifest_sha256'],
            'receipt_path': transfer_reference['proof_path'],
            'receipt_sha256': transfer_reference['proof_sha256']}
    if phase == 'prefix_audit':
        m['scientific']['diagnostic_compute_dtypes'] = ['bfloat16', 'float32']
        m['scientific']['production_dtype_changed'] = False
        m['scientific']['diagnostic_protocol'] = 'Legacy-backend original sequences, fixed-shape future perturbation, common-length future extension, prompt-only repeats, FP32 comparison, and exclusive-MATH BF16 original sequences; all captures retained before verdict.'
    if phase.startswith('fixed_cache_'):
        m['scientific'].update({'cache_role': 'auxiliary' if phase == 'fixed_cache_auxiliary' else 'core',
                               'padded_sequence_length': 2176, 'padding_attention_mask': 0,
                               'padding_side': 'right', 'original_tokens_preserved': True,
                               'attention': 'exclusive_torch_SDPBackend_MATH',
                               'native_storage_dtype': 'bfloat16', 'delta_storage_dtype': 'float32',
                               'raw_activations_retained': True, 'prompt_final_saved': True,
                               'reason': 'Repair shape/backend-dependent prefix variation; original caches and candidates preserved.'})
    files = [*source.rglob('*'), *(stage / 'input').rglob('*'), *task_root.rglob('*'), prepared,
             raw_package / 'artifact_manifest.json', raw_package / 'activation_index.jsonl']
    files.extend(repair_bindings)
    if phase not in ['qualification', 'causal_qualification', 'supplement', 'prefix_audit'] and not phase.startswith('fixed_cache_'):
        files.extend(prior_phase_bindings)
        m['scientific']['previously_committed_generation_requests'] = used_generations
        m['scientific']['previously_committed_tf_requests'] = used_tf
    # Rebind the exact prior verified model/tokenizer/adapter/runtime files.
    for name in prior['bound_files']:
        if name.startswith(prior['model_snapshot'] + '/') or name.startswith(prior['checkpoint'] + '/') or '/venv/' in name:
            files.append(Path(name))
    for condition in conditions.values():
        for layer in condition.get('layers', []):
            if layer['kind'] == 'candidate':
                files.append(Path(layer['path']))
    for path in sorted(set(files)):
        if path.is_file():
            m['bound_files'][str(path)] = {'sha256': supervisor.sha256(path), 'size_bytes': path.stat().st_size}
            if str(path) in prior['bound_files'] and m['bound_files'][str(path)] != prior['bound_files'][str(path)]:
                raise RuntimeError('Previously verified model/input file changed: ' + str(path))
    for path in source.rglob('*'):
        if path.is_file():
            path.chmod(0o400)
    for path in [*task_root.rglob('*'), *(stage / 'input').rglob('*')]:
        if path.is_file():
            path.chmod(0o400)
    if transfer:
        import prelaunch_reallocation
        old_path = Path(transfer_reference['old_manifest_path'])
        old = json.loads(old_path.read_text())
        pending = prelaunch_reallocation.validate_pending(old_path, old, Path(transfer_reference['proof_path']))
        prelaunch_reallocation.validate_pair(old_path, old, manifest_path, m, pending)
    publish_idle_manifest(manifest_path, m)
    digest = supervisor.sha256(manifest_path)
    supervisor.load_manifest(manifest_path, digest)
    print(json.dumps({'manifest': str(manifest_path), 'sha256': digest,
                      'command': [python, str(source / 'infra/gpu03/direction_discovery/supervisor.py'), '--launch',
                                  '--manifest', str(manifest_path), '--manifest-sha256', digest],
                      'output': str(output), 'workers': workers}, sort_keys=True))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--publish-prepared' in argv:
        retry = argparse.ArgumentParser(description='Publish an exact prepared manifest after a fresh idle check; never launch.')
        retry.add_argument('--publish-prepared', type=Path, required=True)
        retry.add_argument('--prepared-manifest-sha256', required=True)
        args = retry.parse_args(argv)
        print(json.dumps(publish_prepared_manifest(args.publish_prepared, args.prepared_manifest_sha256), sort_keys=True))
        return
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--runtime-seconds', type=int, default=14400)
    p.add_argument('--master-plan-sha256', default='d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10')
    a = p.parse_args(argv)
    build(a.stage, a.phase, a.gpus, a.runtime_seconds, a.master_plan_sha256)


if __name__ == '__main__':
    main()
