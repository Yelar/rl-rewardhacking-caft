"""Execution-only finalist rounds; full 740-request behavior plan stays authoritative.

This module never generates, evaluates, selects a condition, or refunds a request.
Round 1 can be prepared only after round 0 is independently verified successful.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

PROTOCOL = 'finalist740_samples01_then23_v1'
SAMPLES = ((0, 1), (2, 3))
BOOKKEEPING = frozenset({
    'requests', 'execution_partition', 'new_generation_requests', 'scope_counts',
    'counts_per_condition', 'previously_committed_generation_requests',
    'previously_committed_tf_requests', 'previously_committed_untouched_test_generation_requests',
    'previously_committed_untouched_test_requests', 'generation_requests_after_commit',
    'prior_phase_manifest_bindings', 'source_bindings',
})


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def frozen(path, digest):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222,
            'Partition input must be an immutable absolute regular file')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'Partition input hash changed')
    return json.loads(data)


def helper(name):
    import importlib
    return importlib.import_module((__package__ + '.' if __package__ else '') + name)


def validate_full(full):
    require('execution_partition' not in full and 'finalist_recovery' not in full and 'test_execution' not in full and
            full.get('purpose') == 'checkpoint60_paired_causal_behavior' and
            full.get('phase') == 'finalist_validation' and full.get('mode') == 'generate' and
            full.get('evaluation_partition') == 'configuration_validation' and full.get('test_used_for_selection') is False and
            full.get('no_training') is True, 'Partition requires the full validation-only finalist plan')
    problems, conditions = full['selected_problem_ids'], full['conditions']
    require(len(problems) == len(set(map(str, problems))) == 37 and len(conditions) == 5 and
            len(full['selected_target_ids']) == 1 and full['selected_target_ids'][0] in conditions and 'baseline' in conditions,
            'Full finalist must contain 37 problems and five fixed conditions')
    requests = full['requests']
    require(len(requests) == len({r['request_id'] for r in requests}) == full['new_generation_requests'] == 740,
            'Full finalist requires exactly 740 unique requests')
    b = helper('behavior_plan')
    ordered_conditions = b.make_conditions({name: conditions[name] for name in full['selected_target_ids']},
                                          combination_evidence=full.get('combination_evidence'))
    require(ordered_conditions == conditions, 'Full finalist random controls or target definition changed')
    expected = []
    for problem in problems:
        for sample in range(4):
            for condition in ordered_conditions:
                expected.append((str(problem), sample, condition))
    actual = []
    for r in requests:
        require(r.get('scope') == 'primary' and type(r.get('sample_index')) is int and
                r.get('problem_split') == 'configuration_validation' and
                r.get('seed') == b.generation_seed(r['problem_id'], 'primary', r['record_id'], r['sample_index']) and
                r['request_id'] == b.request_id(full['master_plan_sha256'], 'finalist_validation',
                                              r['problem_id'], 'primary', r['record_id'], r['sample_index'], r['condition_id']),
                'Full finalist request scope, seed or ID changed')
        actual.append((str(r['problem_id']), r['sample_index'], r['condition_id']))
    require(actual == expected, 'Full finalist request order or Cartesian coverage changed')
    require(all(len({r['record_id'] for r in requests if str(r['problem_id']) == str(p)}) == 1 for p in problems),
            'Primary source record differs within a problem')
    require(all(full['primary_source_records'].get(str(r['problem_id'])) == r['record_id'] for r in requests) and
            full['local_source_records'] == [], 'Full finalist primary provenance or local source list differs')
    return requests


def validate_part(part, *, full=None):
    meta = part.get('execution_partition')
    require(isinstance(meta, dict) and set(meta) == {'protocol', 'round_index', 'sample_indices', 'full_plan',
            'full_plan_sha256', 'full_request_count', 'part_request_count', 'predecessor', 'no_intermediate_analysis'},
            'Missing or invalid execution partition metadata')
    index = meta['round_index']
    require(type(index) is int and index in (0, 1) and meta['protocol'] == PROTOCOL and
            meta['sample_indices'] == list(SAMPLES[index]) and meta['full_request_count'] == 740 and
            meta['part_request_count'] == 370 and meta['no_intermediate_analysis'] is True,
            'Execution partition protocol or fixed sample indices changed')
    loaded = frozen(meta['full_plan'], meta['full_plan_sha256'])
    require(full is None or full == loaded, 'Execution partition belongs to another full scientific plan')
    full = loaded
    validate_full(full)
    require({k: v for k, v in part.items() if k not in BOOKKEEPING} ==
            {k: v for k, v in full.items() if k not in BOOKKEEPING}, 'Execution partition changed scientific plan fields')
    expected = [r for r in full['requests'] if r['sample_index'] in SAMPLES[index]]
    require(part['requests'] == expected and len(expected) == 370 and part['new_generation_requests'] == 370 and
            part['scope_counts'] == {'primary': 370} and part['counts_per_condition'] == {'primary': 74, 'local': 0},
            'Execution requests are not the exact ordered 370-request sample partition')
    for key in ('previously_committed_generation_requests', 'previously_committed_tf_requests',
                'previously_committed_untouched_test_generation_requests', 'previously_committed_untouched_test_requests'):
        require(type(part.get(key)) is int and part[key] >= 0, 'Invalid execution budget count')
    require(part['generation_requests_after_commit'] == part['previously_committed_generation_requests'] + 370 <= 4096,
            'Execution generation budget differs or exceeds cap')
    require(part['previously_committed_tf_requests'] <= 12000 and
            part['previously_committed_untouched_test_generation_requests'] == part['previously_committed_untouched_test_requests'] == 0,
            'Finalist execution cannot follow test use or exceed TF budget')
    require(all(part['source_bindings'].get(k) == v for k, v in full['source_bindings'].items()) and
            part['source_bindings'].get(meta['full_plan']) == meta['full_plan_sha256'], 'Full plan/source bindings lost')
    require((index == 0 and meta['predecessor'] is None) or (index == 1 and isinstance(meta['predecessor'], dict)),
            'Second execution round requires a verified first round')
    return full, meta


def predecessor_bindings(part, *, verify=False):
    """Metadata/hash checks only; scientific result rows are never parsed here."""
    full, meta = validate_part(part)
    paths = [Path(meta['full_plan'])]
    if meta['round_index'] == 0:
        return paths
    ref = meta['predecessor']
    if ref.get('kind') == 'recovered_finalist_round0':
        recovered_predecessor(part, verify=verify)
        return paths + helper('finalist_recovery').logical_bindings(ref, verify=verify)
    require(set(ref) == {'manifest', 'manifest_sha256', 'artifact_manifest_sha256',
                         'verification', 'verification_sha256'}, 'Incomplete predecessor proof binding')
    manifest = frozen(ref['manifest'], ref['manifest_sha256'])
    proof = frozen(ref['verification'], ref['verification_sha256'])
    require(proof.get('status') == 'verified' and proof.get('manifest_sha256') == ref['manifest_sha256'] and
            proof.get('run_token') == manifest['run_token'] and proof.get('gpu_release_verified') is True,
            'Predecessor does not prove terminal success and release')
    artifact = Path(manifest['output']) / 'artifact_manifest.json'
    require(sha(artifact) == ref['artifact_manifest_sha256'], 'Predecessor artifact changed')
    prior_path = Path(manifest['stage']) / 'input/request_plan.json'
    bound = manifest['bound_files'].get(str(prior_path), {})
    prior = frozen(prior_path, bound.get('sha256'))
    _, prior_meta = validate_part(prior, full=full)
    require(prior_meta['round_index'] == 0 and prior_meta['full_plan_sha256'] == meta['full_plan_sha256'] and
            manifest['scientific'].get('execution_partition') == prior_meta and
            manifest['scientific']['master_plan_sha256'] == full['master_plan_sha256'], 'Predecessor is not this plan’s first round')
    covered = []
    for w in manifest['workers']:
        task_path = Path(w['command'][3])
        task = frozen(task_path, manifest['bound_files'][str(task_path)]['sha256'])
        require(w['success_expect']['mode'] == task['mode'] == 'generate' and
                w['success_expect']['requests'] == len(task['requests']) and task['conditions'] == full['conditions'],
                'Predecessor worker identity differs')
        covered.extend(task['requests']); paths.append(task_path)
    require(len(covered) == 370 and len({r['request_id'] for r in covered}) == 370 and
            {r['request_id']: r for r in covered} == {r['request_id']: r for r in prior['requests']},
            'Predecessor is incomplete or has duplicate requests')
    if verify:
        actual = helper('supervisor').verify(Path(ref['manifest']), ref['manifest_sha256'])
        require(actual == proof, 'Fresh independent predecessor verification differs from bound receipt')
    paths.extend(map(Path, (ref['manifest'], ref['verification'], artifact, prior_path)))
    return paths


def recovered_predecessor(part, *, verify=False):
    """Accept a logical round only through its explicit sealed recovery proof."""
    full, meta = validate_part(part)
    require(meta['round_index'] == 1 and meta['predecessor'].get('kind') == 'recovered_finalist_round0',
            'Logical recovery can only precede the second validation round')
    context = helper('finalist_recovery').logical_context(meta['predecessor'], verify=verify)
    expected = {r['request_id'] for r in full['requests'] if r['sample_index'] in SAMPLES[0]}
    require(context['full'] == full and context['full_plan_sha256'] == meta['full_plan_sha256'] and
            len(context['request_ids']) == len(set(context['request_ids'])) == 370 and
            set(context['request_ids']) == expected, 'Logical predecessor differs from the original complete first round')
    return context


def recovery_ledger_predecessor(part, phases):
    context = recovered_predecessor(part)
    old = [p for p in phases if p['manifest_sha256'] == context['original_manifest_sha256']]
    new = [p for p in phases if p['manifest_sha256'] == context['recovery_manifest_sha256']]
    require(len(old) == len(new) == 1 and old[0]['generation_requests'] == 370 and
            old[0]['execution_partition']['round_index'] == 0 and
            old[0]['execution_partition']['full_plan_sha256'] == context['full_plan_sha256'] and
            old[0]['wall_basis'] == new[0]['wall_basis'] == 'actual_launch_to_terminal_receipt' and
            new[0].get('finalist_recovery', {}).get('original_manifest_sha256') == context['original_manifest_sha256'] and
            new[0]['finalist_recovery']['full_plan_sha256'] == context['full_plan_sha256'] and
            new[0]['generation_requests'] == new[0]['finalist_recovery']['new_requests'] == 114,
            'Logical predecessor lacks both charged terminal original and recovery reservations')
    return context


def validate_against_ledger(part, ledger):
    """Reject replay and stale preparation, keeping failed/unused reservations spent."""
    _, meta = validate_part(part)
    require(part['previously_committed_generation_requests'] == ledger['generation_requests'] and
            part['previously_committed_tf_requests'] == ledger['tf_requests'] and ledger['untouched_test_requests'] == 0,
            'Execution partition ledger is stale or test requests were committed')
    rounds = [p for p in ledger['phases'] if p.get('execution_partition', {}).get('full_plan_sha256') == meta['full_plan_sha256']]
    require(not any(p['execution_partition']['round_index'] == meta['round_index'] for p in rounds),
            'Execution round already committed; preserve requests and use explicit reviewed recovery')
    ids = {r['request_id'] for r in part['requests']}
    require(not any(ids.intersection(p.get('generation_request_ids', [])) for p in ledger['phases']),
            'Execution requests already committed, including under a refrozen full plan')
    if meta['round_index'] == 0:
        require(not rounds, 'First execution round cannot follow another round')
    else:
        if meta['predecessor'].get('kind') == 'recovered_finalist_round0':
            context = recovery_ledger_predecessor(part, ledger['phases'])
            require(len(rounds) == 1 and rounds[0]['manifest_sha256'] == context['original_manifest_sha256'],
                    'Recovered second round has an unexpected prior execution reservation')
            return
        require(len(rounds) == 1 and rounds[0]['execution_partition']['round_index'] == 0 and
                rounds[0]['manifest_sha256'] == meta['predecessor']['manifest_sha256'] and
                rounds[0]['generation_requests'] == 370 and rounds[0]['wall_basis'] == 'actual_launch_to_terminal_receipt',
                'Second round requires its complete terminal predecessor in the current ledger')


def make_part(full_path, full_sha256, round_index, previous, predecessor=None):
    full = frozen(full_path, full_sha256)
    validate_full(full)
    require(type(round_index) is int and round_index in (0, 1), 'Invalid execution round index')
    part = copy.deepcopy(full)
    part['requests'] = [r for r in full['requests'] if r['sample_index'] in SAMPLES[round_index]]
    part.update(new_generation_requests=370, scope_counts={'primary': 370}, counts_per_condition={'primary': 74, 'local': 0},
                previously_committed_generation_requests=previous['generations'],
                previously_committed_tf_requests=previous['teacher_forced'],
                previously_committed_untouched_test_generation_requests=previous['untouched_test_generations'],
                previously_committed_untouched_test_requests=previous['untouched_test_requests'],
                generation_requests_after_commit=previous['generations'] + 370,
                prior_phase_manifest_bindings=copy.deepcopy(previous.get('phase_manifests', [])))
    part['source_bindings'][str(full_path)] = full_sha256
    for item in previous.get('ledger_bindings', []):
        part['source_bindings'][item['path']] = item['sha256']
    part['execution_partition'] = {'protocol': PROTOCOL, 'round_index': round_index,
        'sample_indices': list(SAMPLES[round_index]), 'full_plan': str(full_path), 'full_plan_sha256': full_sha256,
        'full_request_count': 740, 'part_request_count': 370, 'predecessor': copy.deepcopy(predecessor),
        'no_intermediate_analysis': True}
    validate_part(part)
    return part


def write_part(*, full_plan, full_plan_sha256, round_index, phase_root, output, predecessor=None):
    """Prepare only the next round. This writes no GPU manifest or budget reservation."""
    full_plan, output = Path(full_plan), Path(output)
    require(not output.exists(), 'Preserve earlier execution plans; use a fresh output directory')
    full = frozen(full_plan, full_plan_sha256)
    validate_full(full)
    b = helper('behavior_plan')
    previous = b.counts_from_phase_ledger(phase_root, full['master_plan_sha256'], full.get('parent_plan_sha256'))
    part = make_part(full_plan, full_plan_sha256, round_index, previous, predecessor)
    validate_against_ledger(part, previous['phase_budget'])
    paths = predecessor_bindings(part, verify=True)
    for path in paths:
        part['source_bindings'][str(path)] = sha(path)
    for name, digest in part['source_bindings'].items():
        require(sha(name) == digest, 'Bound full-plan/source/ledger evidence changed')
    # Re-scan after potentially slow hashing; do not silently refresh a stale plan.
    current = b.counts_from_phase_ledger(phase_root, full['master_plan_sha256'], full.get('parent_plan_sha256'))
    require(current == previous, 'Execution ledger changed during preparation; prepare a fresh plan')
    output.mkdir(parents=True)
    path = output / 'request_plan.json'
    path.write_text(canonical(part)); path.chmod(0o400)
    return {'request_plan': str(path), 'sha256': sha(path), 'round_index': round_index, 'requests': 370,
            'full_plan_sha256': full_plan_sha256, 'no_compute_launched': True, 'budget_reserved': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    print(canonical(write_part(**json.loads(args.spec.read_text()))), end='')


if __name__ == '__main__':
    main()
