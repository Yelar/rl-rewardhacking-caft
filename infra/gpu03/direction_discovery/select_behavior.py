#!/usr/bin/env python3
"""Prepare exactly three screening targets from a verified combined TF ranking.

This CPU-only helper performs no model work or budget reservation. The supplied
independent receipt must come from a separate, successful rank_verified.verify
invocation. Its externally supplied hash is required; this helper verifies the
ranking again before resolving the original candidate definitions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

try:
    from . import behavior_plan as behavior
except ImportError:
    import behavior_plan as behavior


RANK_FILES = {'resolved_spec.json', 'tf_ranking.json', 'phase_budget.json',
              'source_sha256.json', 'SUCCESS.json'}
SELECTION_FILES = {'resolved_spec.json', 'selected_conditions.json'}
SORT_DEFINITION = ['descending priority_beyond_random_mean.mean',
                   'ascending metrics.benign_correct_nll_increase.mean',
                   'ascending condition_id']


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked_digest(value):
    behavior.require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
                     'Missing or malformed externally bound SHA-256')
    return value


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        behavior.require(key not in value, 'Duplicate JSON key: ' + key)
        value[key] = item
    return value


def parse_json(data):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def bound_bytes(path, digest, size=None):
    path = Path(path)
    behavior.require(path.is_absolute() and path.is_file() and not path.is_symlink(),
                     'Missing, relative, or symlinked bound input')
    behavior.require(path.stat().st_size <= 32 * 1024 ** 2, 'Bound JSON exceeds 32 MiB limit')
    data = path.read_bytes()
    behavior.require(digest_bytes(data) == checked_digest(digest) and
                     (size is None or type(size) is int and len(data) == size),
                     'Bound input snapshot hash/size mismatch')
    return data


def bound_json(path, binding):
    return parse_json(bound_bytes(path, binding['sha256'], binding.get('size_bytes')))


def independently_verify_ranking(root):
    # Lazy import: preparation itself needs no torch/model runtime. The existing
    # CPU verifier recomputes the exact frozen ranking from terminal TF packages.
    try:
        from . import rank_verified
    except ImportError:
        import rank_verified
    return rank_verified.verify(root)


def package_snapshots(root, digest, names):
    root = Path(root)
    behavior.require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
                     'Package root must be an existing absolute directory')
    artifact = bound_json(root / 'artifact_manifest.json', {'sha256': digest})
    behavior.require(artifact.get('algorithm') == 'sha256' and set(artifact.get('files', {})) == names,
                     'Package contains missing or unexpected artifact entries')
    actual = {str(path.relative_to(root)) for path in root.rglob('*')}
    behavior.require(actual == names | {'artifact_manifest.json'},
                     'Package has extra files, directories, or missing artifacts')
    values = {name: bound_json(root / name, artifact['files'][name]) for name in sorted(names)}
    return artifact, values


def ranking_key(row):
    excess = row['priority_beyond_random_mean']['mean']
    capability = row['metrics']['benign_correct_nll_increase']['mean']
    behavior.require(all(type(value) in (int, float) and math.isfinite(value)
                         for value in (excess, capability)), 'Invalid likelihood ranking score')
    return -excess, capability, row['condition_id']


def validate_target(condition_id, condition, ranked):
    candidate_id = condition.get('candidate_id')
    match = re.fullmatch(r'L(\d{2})\.transition\.(?:pc(0[0-9])|mean\.'
                         r'(harmful_vs_benign|harmful_incorrect_vs_benign_incorrect)\.(v60|v_change))',
                         candidate_id or '')
    behavior.require(match is not None and condition_id == 'target:' + candidate_id and
                     condition.get('role') == ranked.get('role') == 'target' and
                     ranked.get('candidate_id') == candidate_id and condition.get('window') == 'transition',
                     'Only the frozen primary means and individual PCs are eligible')
    layer = int(match[1])
    behavior.require(behavior.signature(condition) == ((layer, 1),) and ranked.get('layer') == layer,
                     'Ranked target layer/rank/signature differs from its evaluated definition')
    selector = condition['layers'][0]['selectors'][0]
    if match[2] is not None:
        behavior.require(condition.get('family') == 'pca' and condition.get('candidate_kind') == 'pc' and
                         selector == {'key': 'pca.pcs', 'column': int(match[2])},
                         'Individual PC selector does not match the evaluated candidate ID')
    else:
        behavior.require(condition.get('family') == match[3] and condition.get('candidate_kind') == match[4] and
                         selector == {'key': match[3] + '.' + match[4]},
                         'Primary mean selector does not match the evaluated candidate ID')


def build_selection(*, ranking_root, artifact_manifest_sha256, verification_receipt,
                    verification_receipt_sha256, master_sha256):
    """Return a selection only after all provenance and ordering checks pass."""
    behavior.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Selection must hide CUDA')
    root = Path(ranking_root)
    artifact, package = package_snapshots(root, checked_digest(artifact_manifest_sha256), RANK_FILES)
    receipt = bound_json(verification_receipt, {'sha256': verification_receipt_sha256})
    spec, ranking = package['resolved_spec.json'], package['tf_ranking.json']
    behavior.require(spec['master_sha256'] == checked_digest(master_sha256) and
                     ranking['master_plan_sha256'] == master_sha256 and
                     Path(spec['output']).resolve() == root.resolve(), 'Combined ranking belongs to another master/package')
    behavior.require(spec.get('prepare_neighbors') is False and len(spec['phases']) == 2 and
                     len({reference['manifest_sha256'] for reference in spec['phases']}) == 2,
                     'Selection requires completed coarse plus refinement; no neighbor planning')
    master = bound_json(spec['master_path'], {'sha256': master_sha256})
    parent = bound_json(spec['parent_master_path'], {'sha256': spec['parent_master_sha256']})
    behavior.validate_master(master, master_sha256, parent=parent, parent_sha=spec['parent_master_sha256'])
    behavior.require(master.get('plan_version') == 2, 'Screening requires the repaired version2 plan')
    budget, success = package['phase_budget.json'], package['SUCCESS.json']
    behavior.require(budget['untouched_test_requests'] == success['test_requests'] == 0 and
                     success['status'] == 'succeeded' and success['mode'] == 'verified_likelihood_prioritization_only' and
                     success['model_work_launched'] is False and success['validation_problems'] == 12,
                     'Ranking lacks a successful validation-only scope')
    expected_proof = {'status': 'verified', 'requests': ranking['requests_verified'],
                      'ranking_sha256': artifact['files']['tf_ranking.json']['sha256'],
                      'artifact_manifest_sha256': artifact_manifest_sha256,
                      'test_requests': 0, 'model_work_launched': False}
    behavior.require(type(expected_proof['requests']) is int and expected_proof['requests'] > 0 and
                     success['requests'] == expected_proof['requests'] and receipt == expected_proof and
                     type(receipt.get('test_requests')) is int and receipt.get('model_work_launched') is False,
                     'Independent ranking receipt differs from the exact complete package')
    fresh_proof = independently_verify_ranking(root)
    behavior.require(fresh_proof == expected_proof, 'Independent reconstruction differs from the bound ranking snapshot')

    conditions, phase_bindings, request_ids = {}, [], set()
    for expected_phase, reference in zip(('coarse', 'refinement'), spec['phases']):
        manifest = bound_json(reference['manifest_path'], {'sha256': reference['manifest_sha256']})
        request_path = Path(manifest['stage']) / 'input/request_plan.json'
        binding = manifest['bound_files'][str(request_path)]
        plan = bound_json(request_path, binding)
        behavior.require(plan['phase'] == expected_phase and plan['mode'] == 'tf' and
                         plan['evaluation_partition'] == 'configuration_validation' and
                         plan['master_plan_sha256'] == master_sha256 and
                         plan['selected_problem_ids'] == master['sweep']['teacher_forced_validation_problems'],
                         'Combined phase order, partition, or frozen problem IDs differ')
        phase_ids = [request['request_id'] for request in plan['requests']]
        behavior.require(len(phase_ids) == len(set(phase_ids)) and not request_ids.intersection(phase_ids),
                         'Duplicate TF request IDs within or across source phases')
        request_ids.update(phase_ids)
        for key, condition in plan['conditions'].items():
            behavior.require(key not in conditions or key == 'baseline' and
                             condition == conditions[key] == {'role': 'baseline', 'layers': []},
                             'Duplicate/conflicting target or random condition IDs across phases')
            conditions[key] = condition
        phase_bindings.append({'phase': expected_phase, 'reference': copy.deepcopy(reference),
                               'request_plan': str(request_path), 'request_plan_binding': copy.deepcopy(binding),
                               'candidate_artifact_manifest_sha256': plan['candidate_artifact_manifest_sha256']})
    behavior.require(len(request_ids) == expected_proof['requests'] and
                     len({item['candidate_artifact_manifest_sha256'] for item in phase_bindings}) == 1,
                     'Combined source request coverage or candidate artifact differs')
    behavior.require(ranking['status'] == 'exploratory_TF_prioritization_only' and
                     ranking['metric'] == 'evaluator__transition' and
                     ranking['problem_ids'] == master['sweep']['teacher_forced_validation_problems'],
                     'Ranking protocol differs from the frozen validation ordering')
    rows = ranking['target_ranking']
    ids = [row['condition_id'] for row in rows]
    behavior.require(len(rows) >= 3 and len(set(ids)) == len(ids) and
                     len({row['candidate_id'] for row in rows}) == len(rows), 'Missing or duplicate ranked target IDs')
    behavior.require(set(ids) == {key for key, condition in conditions.items() if condition.get('role') == 'target'},
                     'Ranking omits or adds source target conditions')
    behavior.require(rows == sorted(rows, key=ranking_key), 'Ranking order differs from the frozen likelihood ordering')
    for row in rows:
        validate_target(row['condition_id'], conditions[row['condition_id']], row)
    selected = {row['condition_id']: copy.deepcopy(conditions[row['condition_id']]) for row in rows[:3]}
    behavioral_conditions = behavior.make_conditions(selected)
    selected_provenance = []
    for index, row in enumerate(rows[:3]):
        condition_id = row['condition_id']
        condition = selected[condition_id]
        for layer in condition['layers']:
            path = Path(layer['path'])
            behavior.require(path.is_file() and not path.is_symlink() and behavior.sha256(path) == layer['sha256'],
                             'Selected candidate bytes changed')
        tf_controls = row['matched_random_condition_ids']
        behavior.require(len(tf_controls) == len(set(tf_controls)) == 3, 'Ranked target lacks three unique TF controls')
        behavior_controls = behavioral_conditions[condition_id]['random_controls']
        mapping, seed_bases = [], set()
        for tf_id in tf_controls:
            tf_control = conditions[tf_id]
            base = tf_control.get('random_seed_base')
            matches = [key for key in behavior_controls if behavioral_conditions[key]['random_seed_base'] == base]
            behavior.require(tf_control.get('role') == 'random' and len(matches) == 1 and base not in seed_bases and
                             tf_control['layers'] == behavioral_conditions[matches[0]]['layers'],
                             'TF and behavior random controls have different layers, rank, or seed')
            seed_bases.add(base)
            mapping.append({'tf_condition_id': tf_id, 'behavior_condition_id': matches[0], 'random_seed_base': base})
        selected_provenance.append({'priority_rank': index + 1, 'condition_id': condition_id,
                                    'candidate_id': row['candidate_id'],
                                    'signature': [list(value) for value in behavior.signature(condition)],
                                    'ranked_entry': copy.deepcopy(row), 'random_control_identity_mapping': mapping})
    # Detect replacement of the small ranking package during independent replay.
    final_artifact, final_package = package_snapshots(root, artifact_manifest_sha256, RANK_FILES)
    behavior.require(final_artifact == artifact and final_package == package,
                     'Ranking snapshots changed during selection')
    return {'schema_version': 1, 'purpose': 'screening_targets_from_verified_combined_tf',
            'master_plan_sha256': master_sha256, 'no_test_outcomes_used': True,
            'conditions': selected, 'selected_condition_ids_in_priority_order': ids[:3],
            'selection_rule': {'count': 3, 'ordering': SORT_DEFINITION,
                               'numeric_threshold': None, 'auxiliary_or_probe_filtering': False},
            'selection_provenance': selected_provenance,
            'ranking_provenance': {'ranking_root': str(root), 'artifact_manifest_sha256': artifact_manifest_sha256,
                'ranking_sha256': expected_proof['ranking_sha256'], 'independent_verification_receipt': str(verification_receipt),
                'independent_verification_receipt_sha256': verification_receipt_sha256,
                'master_path': spec['master_path'], 'parent_master_path': spec['parent_master_path'],
                'parent_master_sha256': spec['parent_master_sha256'],
                'verified_scope': 'configuration_validation', 'test_requests': 0,
                'independent_reconstruction': fresh_proof, 'source_phases': phase_bindings},
            'behavior_conditions_including_controls': len(behavioral_conditions),
            'expected_screening_generation_requests': 60 * len(behavioral_conditions),
            'budget_reserved': False, 'no_model_work_launched': True,
            'interpretation': 'Likelihood ordering prioritizes these three individual targets only. '
                              'It does not establish behavioral efficacy, capability preservation, or a training recommendation.'}


def write_selection(*, output, **spec):
    output = Path(output)
    behavior.require(output.is_absolute() and not output.exists(), 'Selection output must be a fresh absolute directory')
    ranking_root = Path(spec['ranking_root'])
    behavior.require(not output.resolve().is_relative_to(ranking_root.resolve()), 'Do not write into the frozen ranking package')
    selection = build_selection(**spec)
    provenance = selection['ranking_provenance']
    input_paths = [Path(spec['verification_receipt']), ranking_root,
                   Path(provenance['master_path']).parent, Path(provenance['parent_master_path']).parent]
    for phase in selection['ranking_provenance']['source_phases']:
        input_paths.append(Path(phase['request_plan']).parent.parent)
    input_paths.extend(Path(layer['path']).parent for condition in selection['conditions'].values() for layer in condition['layers'])
    behavior.require(all(not output.resolve().is_relative_to(path.resolve()) for path in input_paths),
                     'Do not write into any frozen input package')
    output.mkdir(parents=True)
    for name, value in (('resolved_spec.json', spec), ('selected_conditions.json', selection)):
        (output / name).write_text(behavior.canonical(value) + '\n')
    artifact = {'algorithm': 'sha256', 'files': {name: {'sha256': behavior.sha256(output / name),
                 'size_bytes': (output / name).stat().st_size} for name in sorted(SELECTION_FILES)}}
    (output / 'artifact_manifest.json').write_text(behavior.canonical(artifact) + '\n')
    for path in output.iterdir():
        path.chmod(0o400)
    return {'status': 'prepared', 'selected_conditions': str(output / 'selected_conditions.json'),
            'selected_conditions_sha256': behavior.sha256(output / 'selected_conditions.json'),
            'artifact_manifest_sha256': behavior.sha256(output / 'artifact_manifest.json'),
            'selected_condition_ids_in_priority_order': selection['selected_condition_ids_in_priority_order'],
            'no_model_work_launched': True, 'budget_reserved': False}


def verify_selection(output, artifact_manifest_sha256):
    artifact, package = package_snapshots(output, artifact_manifest_sha256, SELECTION_FILES)
    expected = build_selection(**package['resolved_spec.json'])
    behavior.require(package['selected_conditions.json'] == expected, 'Independent selection reconstruction differs')
    return {'status': 'verified', 'artifact_manifest_sha256': artifact_manifest_sha256,
            'selected_conditions_sha256': artifact['files']['selected_conditions.json']['sha256'],
            'selected_condition_ids_in_priority_order': expected['selected_condition_ids_in_priority_order'],
            'no_model_work_launched': True, 'budget_reserved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args()
    print(behavior.canonical(write_selection(**parse_json(args.spec.read_bytes()))))


if __name__ == '__main__':
    main()
