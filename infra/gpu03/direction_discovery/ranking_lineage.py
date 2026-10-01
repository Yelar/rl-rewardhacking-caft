"""Join completed TF metadata to its exact successful causal qualification.

Called only after the caller verifies the phase's independent success receipt.
Model/tokenizer/adapter payloads are never opened or rehashed here: their frozen
bindings are compared with the already verified qualification's bindings.
"""
import hashlib
import json
from pathlib import Path
import re

CRITICAL = (
    'infra/gpu03/direction_discovery/engine.py',
    'infra/gpu03/direction_discovery/intervention.py',
    'infra/gpu03/activation_dataset/extract_triplet_raw.py',
    'infra/gpu03/activation_dataset/extract_delta_activations.py',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(manifest, master):
    bindings = {}

    def checked(path, binding):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'Missing/symlinked lineage metadata')
        data = path.read_bytes()
        actual = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
        require(actual['sha256'] == binding['sha256'] and
                ('size_bytes' not in binding or actual['size_bytes'] == binding['size_bytes']),
                'Lineage metadata hash/size mismatch')
        bindings[str(path)] = actual
        return json.loads(data)

    def bound(owner, path):
        path = str(path)
        require(path in owner['bound_files'], 'Lineage metadata missing from frozen bindings')
        return checked(path, owner['bound_files'][path])

    reference_path = Path(manifest['stage']) / 'input/causal_qualification_reference.json'
    reference = bound(manifest, reference_path)
    require(manifest['bound_files'][reference['manifest_path']]['sha256'] == reference['manifest_sha256'],
            'Qualification reference differs from phase binding')
    qualification = bound(manifest, reference['manifest_path'])
    allowed_plans = {manifest['scientific']['master_plan_sha256']}
    if master.get('parent_plan_sha256') is not None:
        allowed_plans.add(master['parent_plan_sha256'])
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in allowed_plans),
            'Invalid qualification master-plan lineage')
    require(qualification['host'] == manifest['host'] == 'gpu-04' and
            qualification['phase'] == 'causal_qualification' and
            qualification['scientific']['master_plan_sha256'] in allowed_plans and
            qualification['scientific']['input_prepared_sha256'] == master['inputs']['prepared_records_sha256'] ==
            manifest['scientific']['input_prepared_sha256'] and
            qualification['runtime_versions'] == manifest['runtime_versions'] and
            len(qualification['workers']) == 1,
            'Qualification host/master/dataset/runtime identity differs')
    output = Path(qualification['output'])
    artifact_path = output / 'artifact_manifest.json'
    require(manifest['bound_files'][str(artifact_path)]['sha256'] == reference['artifact_manifest_sha256'],
            'Qualification artifact differs from phase binding')
    artifact = bound(manifest, artifact_path)
    require(artifact['algorithm'] == 'sha256', 'Unknown qualification artifact algorithm')

    def artifact_json(name):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid qualification artifact path')
        return checked(output / relative, artifact['files'][name])

    end_path = Path(qualification['stage']) / 'control/supervisor_exit.json'
    end = bound(manifest, end_path)
    require(end['manifest_sha256'] == reference['manifest_sha256'] and
            end['run_token'] == qualification['run_token'] and end['service_result'] == 'success' and
            end['exit_code_kind'] == 'exited' and str(end['exit_status']) == '0' and
            end['producer_summary_present'] is True and end['failure_present'] is False and
            re.fullmatch('[0-9a-f]{32}', end.get('invocation_id', '')),
            'Qualification terminal success identity differs')
    summary = artifact_json('campaign_summary.json')
    release = artifact_json('gpu_release.json')
    require(summary['status'] == 'succeeded' and summary['run_token'] == qualification['run_token'] and
            summary['manifest_sha256'] == reference['manifest_sha256'] and summary['worker_exit_codes'] == [0] and
            summary['gpu_release_verified'] is True and release['verified'] is True and
            release['gpu_ids'] == qualification['gpu_ids'] and not (output / 'FAILURE.json').exists(),
            'Qualification successful release identity differs')
    worker = qualification['workers'][0]
    task_path = Path(worker['command'][3])
    task = bound(qualification, task_path)
    require(manifest['bound_files'][str(task_path)] == qualification['bound_files'][str(task_path)],
            'Qualification task differs from phase binding')
    require(task['mode'] == 'qualify' and worker['success_expect'] == {'mode': 'qualify', 'requests': 1} and
            task['run_token'] == qualification['run_token'] and task['worker_name'] == worker['name'] and
            task['attention_policy'] == 'exclusive_math' and task['teacher_forced_padded_sequence_length'] == 2176,
            'Qualification task or numerical policy differs')
    receipt = artifact_json(worker['success_file'])
    require(receipt['status'] == 'succeeded' and receipt['run_token'] == task['run_token'] and
            receipt['worker_name'] == task['worker_name'] and receipt['mode'] == 'qualify' and receipt['requests'] == 1,
            'Qualification worker success differs')
    result_path = output / Path(worker['success_file']).parent / 'results.jsonl'
    require(manifest['bound_files'][str(result_path)] == artifact['files'][str(result_path.relative_to(output))],
            'Qualification numerical verdict differs from artifact binding')
    # Qualification is one small authored inference probe, not a behavioral TF
    # phase. Read its scalar verdict and exact request identity, never TF results.
    probe = bound(manifest, result_path)
    require(len(task['requests']) == 1 and task['conditions'] == {'baseline': {'layers': []}} and
            all(probe.get(key) == value for key, value in task['requests'][0].items()) and
            probe['problem_split'] == 'direction_fit' and
            all(probe['result'].get(key) is True for key in (
                'baseline_recovery_bitwise', 'teacher_forced_effect_verified', 'baseline_generation_repeatable')),
            'Qualification numerical verdict or request identity differs')
    require(Path(task['model_snapshot']).name == master['model']['revision'], 'Base-model revision differs')
    adapter_path = str(Path(task['checkpoint']) / 'adapter_model.safetensors')
    require(qualification['bound_files'][adapter_path]['sha256'] == master['model']['M60_adapter_sha256'],
            'Checkpoint adapter differs from frozen master')
    model_roots = [Path(task['model_snapshot']), Path(task['checkpoint'])]
    model_bindings = {path: binding for path, binding in qualification['bound_files'].items()
                      if any(Path(path).is_relative_to(root) for root in model_roots)}
    require(adapter_path in model_bindings and
            str(model_roots[0] / 'config.json') in model_bindings and
            str(model_roots[0] / 'tokenizer.json') in model_bindings and
            str(model_roots[1] / 'adapter_config.json') in model_bindings and
            any(Path(path).is_relative_to(model_roots[0]) and path.endswith('.safetensors') for path in model_bindings),
            'Qualification model/tokenizer/adapter bindings incomplete')
    current_model_bindings = {path: binding for path, binding in manifest['bound_files'].items()
                             if any(Path(path).is_relative_to(root) for root in model_roots)}
    require(current_model_bindings == model_bindings, 'TF model/tokenizer/adapter file bindings differ')
    source_bindings = {}
    for relative in CRITICAL:
        old, new = str(Path(qualification['source_root']) / relative), str(Path(manifest['source_root']) / relative)
        require(old in qualification['bound_files'] and new in manifest['bound_files'] and
                qualification['bound_files'][old] == manifest['bound_files'][new],
                'TF critical inference source differs from qualification: ' + relative)
        source_bindings[relative] = qualification['bound_files'][old]
    for current in manifest['workers']:
        current_task = bound(manifest, current['command'][3])
        require(current_task['mode'] == 'tf' and current_task['model_snapshot'] == task['model_snapshot'] and
                current_task['checkpoint'] == task['checkpoint'] and
                current_task['attention_policy'] == task['attention_policy'] and
                current_task['teacher_forced_padded_sequence_length'] == task['teacher_forced_padded_sequence_length'],
                'TF worker model/checkpoint/numerical policy differs from qualification')
    return {'qualification_manifest_sha256': reference['manifest_sha256'],
            'lineage_verifier_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'qualification_artifact_sha256': reference['artifact_manifest_sha256'],
            'model_snapshot': task['model_snapshot'], 'checkpoint': task['checkpoint'],
            'model_file_bindings': model_bindings, 'critical_source_bindings': source_bindings,
            'metadata_snapshots': bindings, 'model_payloads_rehashed': False}
