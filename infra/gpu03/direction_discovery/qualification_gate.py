"""Bind the successful masked-padding qualification to a proposed bulk cache."""
from __future__ import annotations

import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.gpu03.direction_discovery import fixed_cache

CRITICAL_SOURCE = ('infra/gpu03/direction_discovery/engine.py',
                   'infra/gpu03/direction_discovery/fixed_cache.py',
                   'infra/gpu03/activation_dataset/extract_triplet_raw.py',
                   'infra/gpu03/activation_dataset/extract_delta_activations.py')
INDICES = (222, 223, 224)
PROFILE = {'padding_side':'right','padded_sequence_length':2176,'pad_token_id':151643,
           'padding_attention_mask':0,'attention_backend':'torch_sdpa_MATH_only',
           'original_tokens_preserved':True,'stored_completion_positions_only':True,'prompt_final_saved':True}
POLICY = {'allow_bf16_reduced_precision_reduction':True,'cpu_autocast':False,'cuda_autocast':False,
          'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,
          'deterministic_algorithms':True,'deterministic_warn_only':False,
          'float32_matmul_precision':'highest','matmul_allow_tf32':False}
FLAGS = {'math':True,'flash':False,'memory_efficient':False,'cudnn':False}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def exact(value, expected, message):
    require(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True), message)


def equal_measurement(value, label):
    require(value.get('bitwise_equal') is True and value.get('relative_l2') == 0 and
            value.get('max_abs_difference') == 0, label + ' is not an exact zero-difference comparison')


def verify_fixed_qualification(reference_path, prepared_sha256, *, source_root, model_snapshot, checkpoint):
    from infra.gpu03.direction_discovery import supervisor
    reference_path, source_root = Path(reference_path), Path(source_root)
    reference = read(reference_path)
    manifest = Path(reference['manifest_path'])
    # One full producer-exited artifact verification; subsequent checks read only
    # small metadata files and compare their already-bound identities.
    checked = supervisor.verify(manifest, reference['manifest_sha256'])
    m = read(manifest)
    require(checked['status'] == 'verified' and m['phase'] == 'fixed_cache_qualification'
            and m['scientific']['input_prepared_sha256'] == prepared_sha256 and len(m['workers']) == 1,
            'Qualification is not the verified one-worker core protocol')
    require(m['scientific'].get('cache_role') == 'core' and m['scientific'].get('phase_modes') == ['fixed_cache'],
            'Qualification scientific mode differs')
    output = Path(m['output'])
    require(supervisor.sha256(output/'artifact_manifest.json') == reference['artifact_manifest_sha256'],
            'Qualification artifact manifest identity changed')
    worker = m['workers'][0]
    exact(worker['success_expect'], {'mode':'fixed_cache','requests':3}, 'Qualification worker coverage differs')
    task_path = Path(worker['command'][3])
    task = read(task_path)
    root = output/'workers'/worker['name']
    exact(read(root/'task.json'), task, 'Output task differs from exact reviewed task')
    require(task['mode'] == 'fixed_cache' and task['cache_role'] == 'core' and task['worker_name'] == worker['name']
            and task['run_token'] == m['run_token'] and task['model_snapshot'] == str(model_snapshot)
            and task['checkpoint'] == str(checkpoint) and task['output'] == str(root)
            and task['manifest_path'] == str(manifest), 'Qualification model/task identity differs from proposed bulk cache')
    exact(task['conditions'], {'baseline':{'layers':[]}}, 'Qualification must have no intervention')
    require(task['padded_sequence_length'] == 2176 and task['pad_token_id'] == 151643
            and task['attention_policy'] == 'exclusive_math', 'Qualification forward profile differs')
    prepared = Path(task['prepared_records'])
    require(supervisor.sha256(prepared) == prepared_sha256, 'Qualification prepared records changed')
    rows = lines(prepared)
    require(len(rows) == 561 and len({r['record_index'] for r in rows}) == 561 and
            len({r['record_id'] for r in rows}) == 561, 'Qualification input is not the exact complete core dataset')
    by_index = {r['record_index']:r for r in rows}
    selected = [by_index[index] for index in INDICES]
    fixed_cache.validate_groups(selected, 'core')
    require({r['problem_split'] for r in selected} == {'direction_fit'} and
            len({str(r['problem_id_key']) for r in selected}) == 1, 'Qualification must use one fitting problem only')
    expected_requests = [{'request_id':'fixed-cache-qual-'+r['record_id'], 'record_id':r['record_id'],
                          'condition_id':'baseline'} for r in selected]
    exact(task['requests'], expected_requests, 'Qualification record IDs/indices differ from the reviewed fitting triplet')
    old_source = Path(m['source_root'])
    proof = [reference_path,manifest,manifest.parent/'control/supervisor_exit.json',
             output/'artifact_manifest.json',task_path,root/'task.json',prepared]
    for relative in CRITICAL_SOURCE:
        previous, current = old_source/relative, source_root/relative
        bound = m['bound_files'][str(previous)]
        require(previous.is_file() and current.is_file() and
                supervisor.sha256(previous) == supervisor.sha256(current) == bound['sha256']
                and previous.stat().st_size == current.stat().st_size == bound['size_bytes'],
                'Qualified capture/runtime source changed: '+relative)
        proof.append(previous)
    versions = {name:importlib.metadata.version(name) for name in m['runtime_versions']}
    exact(versions,m['runtime_versions'],'Qualified dependency runtime changed')
    report_path = root/'fixed_cache_report.json'
    report = read(report_path)
    require(report['status'] == 'succeeded' and report['records'] == 3 and report['problems'] == 1
            and report['identical_prefix_comparisons'] == 6, 'Qualification record/problem/pair coverage incomplete')
    for key in ('all_identical_prefixes_bitwise_equal','all_native_readbacks_bitwise_equal','fp32_delta_readback_exact',
                'raw_activations_retained','differences_computed'):
        require(report.get(key) is True,'Qualification numerical audit failed: '+key)
    exact(report['activation_cache_profile'],PROFILE,'Qualified masked-padding profile differs')
    exact(report['numerical_runtime_policy'],POLICY,'Qualified numerical runtime policy differs')
    require(report['cross_package_prefix_audit_required'] is False and
            report['prefix_comparison_scope'] == 'within_worker_same_problem_and_identical_prompt', 'Wrong qualification pair scope')
    raw, legacy = fixed_cache.dependencies()
    legacy.validate_model_load_reports(report['model_load_reports']['h0'],report['model_load_reports']['h60'],'live qualification')
    first = fixed_cache.validate_groups(selected,'core')[0]
    keep = min(max(16,first['evaluator_body_token']+12),first['completion_token_count']-1)
    require(keep >= first['evaluator_body_token']+12, 'Qualification cannot protect complete evaluator transition prefix')
    artifact_files = read(output/'artifact_manifest.json')['files']
    def proof_artifact(item, expected_path):
        path = Path(item['path'])
        require(path == expected_path and item.get('native_readback_bitwise_equal') is True,
                'Qualification probe artifact identity/readback differs')
        exact(artifact_files[str(path.relative_to(output))], {'sha256':item['sha256'],'size_bytes':item['size_bytes']},
              'Qualification probe hash differs from independently verified manifest')
    for kind in ('h0','h60'):
        legacy.validate_post_model_cuda_state(report['cuda_release'][kind],kind+' live qualification release')
        q = report['qualifications'][kind]
        equal_measurement(q['repeat'],kind+' repeat')
        future = q['future_causality']
        equal_measurement(future,kind+' future causality')
        require(future['unchanged_completion_prefix_tokens'] == keep and
                future['first_changed_sequence_position'] == first['prompt_token_count']+keep and
                future['changed_valid_future_tokens'] == first['completion_token_count']-keep > 0 and
                future['evaluator_transition_prefix_included'] is True and future['fixed_padded_length'] == 2176
                and future['padding_mask_zero_unchanged'] is True, 'Qualification future/transition coverage differs')
        exact(future['attention_flags'],FLAGS,'Qualification future forward did not force math SDP')
        directory=root/'qualification'/kind
        repeat_path,future_path=directory/'repeat_audit.json',directory/'future_causality_audit.json'
        repeat=read(repeat_path)
        equal_measurement(repeat,kind+' saved repeat')
        exact(repeat['attention_flags'],FLAGS,'Qualification repeat did not force math SDP')
        proof_artifact(repeat['artifact'],directory/'repeat.safetensors')
        proof_artifact(future['artifact'],directory/'future_perturbed.safetensors')
        exact(read(future_path),future,'Saved future qualification differs from report')
        proof.extend([repeat_path,future_path])
    audit_path=root/'prefix_audit.jsonl'
    observed=set()
    expected={(kind,*sorted((a['record_id'],b['record_id']))) for kind in ('h0','h60') for a,b in combinations(selected,2)}
    by_id={r['record_id']:r for r in selected}
    for item in lines(audit_path):
        key=(item['kind'],*sorted(item['record_ids']))
        require(len(item['record_ids'])==2 and key in expected and key not in observed,'Missing/duplicate/foreign qualified prefix pair')
        observed.add(key)
        a,b=[by_id[rid] for rid in item['record_ids']]
        count=fixed_cache.common_prefix_count(a['completion_token_ids'],b['completion_token_ids'])
        require(item['common_completion_tokens']==count and str(item['problem_id'])==str(a['problem_id_key']),
                'Qualified prefix tokens/problem metadata differ')
        equal_measurement(item['prompt_final'],'Shared prompt final')
        if count:
            equal_measurement(item['completion_prefix'],'Shared completion prefix')
        else:
            require(item['completion_prefix'] is None,'Absent shared completion prefix fabricated')
    require(observed==expected,'Qualification does not cover all six model/prefix pairs')
    proof.extend([report_path,audit_path])
    return sorted(set(proof))
