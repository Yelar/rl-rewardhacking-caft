"""Execute fixed pieces of one pre-test bundle, without reissuing requests.

This metadata-only module cannot select examples, run a model, or evaluate code.
Every earlier piece must have an independently verified successful release.
Failed or ambiguous pieces require a separate reviewed recovery implementation;
they cannot be regenerated through this entry point.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys

try:
    from .execution_partition import frozen, sha, canonical, require, helper
except ImportError:
    # Direct launch from an arbitrary cwd must retain package-relative imports
    # in test_bundle, even with PYTHONPATH unset by the supervisor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from infra.gpu03.direction_discovery.execution_partition import frozen, sha, canonical, require, helper

PROTOCOL = 'once_only_test_bundle_fixed_partitions_v1'
BOOKKEEPING = frozenset({
    'requests', 'test_execution', 'new_generation_requests', 'scope_counts',
    'counts_per_condition', 'previously_committed_generation_requests',
    'previously_committed_tf_requests', 'previously_committed_untouched_test_generation_requests',
    'previously_committed_untouched_test_requests', 'generation_requests_after_commit',
    'prior_phase_manifest_bindings', 'source_bindings',
})
PROOF_FIELDS = {'manifest', 'manifest_sha256', 'artifact_manifest_sha256',
                'verification', 'verification_sha256'}


def remote_reference(ref):
    return isinstance(ref, dict) and ref.get('kind') == 'remote_generation'


def predecessor_reference_key(ref):
    """Validate reference shape without opening any token or outcome payload."""
    if remote_reference(ref):
        require(set(ref) == {'kind', 'replica'} and isinstance(ref['replica'], dict) and
                set(ref['replica']) == {'path', 'sha256'}, 'Invalid remote predecessor reference')
        value = ref['replica']
        require(isinstance(value['path'], str) and Path(value['path']).is_absolute() and
                isinstance(value['sha256'], str) and len(value['sha256']) == 64 and
                all(c in '0123456789abcdef' for c in value['sha256']), 'Invalid remote replica binding')
        return ('remote_generation', value['path'], value['sha256'])
    require(isinstance(ref, dict) and set(ref) == PROOF_FIELDS, 'Invalid local predecessor proof reference')
    return ('local_generation', ref['manifest'], ref['manifest_sha256'])


def predecessor_context(ref, *, verify=False):
    """Normalize local and explicit remote proofs without pretending remote paths are local."""
    predecessor_reference_key(ref)
    if remote_reference(ref):
        transport = helper('remote_generation')
        remote = transport.context(ref, verify=verify)
        manifest, proof = remote['manifest'], remote['proof']
        digest = remote['manifest_sha256']
        require(manifest['host'] in transport.PROTOCOLS and proof.get('host') == manifest['host'] and
                proof.get('status') == 'verified' and
                proof.get('gpu_release_verified') is True and proof.get('manifest_sha256') == digest and
                proof.get('run_token') == manifest['run_token'], 'Remote predecessor lacks successful terminal release')
        return {'manifest': manifest, 'manifest_sha256': digest, 'proof': proof,
                'part': remote['request_plan'], 'tasks': remote['tasks'],
                'paths': remote['bindings'], 'remote': True, 'context': remote}
    manifest = frozen(ref['manifest'], ref['manifest_sha256'])
    proof = frozen(ref['verification'], ref['verification_sha256'])
    require(manifest['host'] == 'gpu-04' and proof.get('status') == 'verified' and
            proof.get('gpu_release_verified') is True and proof.get('manifest_sha256') == ref['manifest_sha256'] and
            proof.get('run_token') == manifest['run_token'], 'Test predecessor lacks successful terminal release')
    artifact = Path(manifest['output']) / 'artifact_manifest.json'
    frozen(artifact, ref['artifact_manifest_sha256'])
    prior_path = Path(manifest['stage']) / 'input/request_plan.json'
    prior = frozen(prior_path, manifest['bound_files'].get(str(prior_path), {}).get('sha256'))
    paths = list(map(Path, (ref['manifest'], ref['verification'], artifact, prior_path)))
    tasks = []
    for worker in manifest['workers']:
        path = Path(worker['command'][3])
        task = frozen(path, manifest['bound_files'][str(path)]['sha256'])
        tasks.append({'worker': worker, 'task': task, 'task_path': path}); paths.append(path)
    if verify:
        require(helper('supervisor').verify(Path(ref['manifest']), ref['manifest_sha256']) == proof,
                'Fresh predecessor verification differs from the frozen proof')
    return {'manifest': manifest, 'manifest_sha256': ref['manifest_sha256'], 'proof': proof,
            'part': prior, 'tasks': tasks, 'paths': paths, 'remote': False}


def predecessor_manifest_sha256(ref):
    return predecessor_context(ref)['manifest_sha256'] if remote_reference(ref) else ref['manifest_sha256']


def bundle_context(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Invalid test bundle reference')
    path = Path(ref['path'])
    # This also rejects mutable metadata before the portable package verifier.
    frozen(path, ref['sha256'])
    bundle = helper('test_bundle').verify_bundle(path, ref['sha256'])
    full_path = path.parent / bundle['generation_request_plan']['path']
    full = frozen(full_path, bundle['generation_request_plan']['sha256'])
    require('test_execution' not in full and 'execution_partition' not in full,
            'Test bundle must contain the full scientific request plan')
    return bundle, full


def part_counts(requests, conditions):
    scopes = Counter(r['scope'] for r in requests)
    counts = {}
    for scope in ('primary', 'local'):
        by_condition = Counter(r['condition_id'] for r in requests if r['scope'] == scope)
        values = [by_condition.get(c, 0) for c in conditions]
        require(len(set(values)) == 1, 'A fixed test partition must preserve complete matched condition cells')
        counts[scope] = values[0]
    return dict(scopes), counts


def validate_part(part, *, full=None):
    meta = part.get('test_execution')
    require(isinstance(meta, dict) and set(meta) == {'protocol', 'partition_index', 'bundle',
            'part_request_count', 'predecessors', 'no_intermediate_analysis'}, 'Invalid test execution metadata')
    require(meta['protocol'] == PROTOCOL and meta['no_intermediate_analysis'] is True,
            'Test execution protocol changed')
    bundle, loaded = bundle_context(meta['bundle'])
    require(full is None or full == loaded, 'Test execution belongs to a different full request plan')
    full = loaded
    index = meta['partition_index']
    require(type(index) is int and 0 <= index < len(bundle['execution_partitions']), 'Invalid test partition index')
    ids = bundle['execution_partitions'][index]['request_ids']
    lookup = {r['request_id']: r for r in full['requests']}
    expected = [lookup[rid] for rid in ids]
    require({k: v for k, v in part.items() if k not in BOOKKEEPING} ==
            {k: v for k, v in full.items() if k not in BOOKKEEPING}, 'Test partition changed scientific fields')
    require(part['requests'] == expected and type(meta['part_request_count']) is int and
            meta['part_request_count'] == part['new_generation_requests'] == len(expected) > 0,
            'Test requests differ from the fixed ordered partition')
    scope_counts, condition_counts = part_counts(expected, full['conditions'])
    require(part['scope_counts'] == scope_counts and part['counts_per_condition'] == condition_counts,
            'Test partition scope or condition counts differ')
    require(isinstance(meta['predecessors'], list) and len(meta['predecessors']) == index,
            'Every earlier test partition needs a complete proof reference')
    keys = [predecessor_reference_key(p) for p in meta['predecessors']]
    require(len(set(keys)) == index and len({(k[0], k[1]) for k in keys}) == index and
            len({(k[0], k[2]) for k in keys}) == index,
            'Repeated test predecessor proof')
    for key in ('previously_committed_generation_requests', 'previously_committed_tf_requests',
                'previously_committed_untouched_test_generation_requests', 'previously_committed_untouched_test_requests'):
        require(type(part.get(key)) is int and part[key] >= 0, 'Invalid test budget count')
    require(part['generation_requests_after_commit'] == part['previously_committed_generation_requests'] + len(expected) <= 4096 and
            part['previously_committed_tf_requests'] <= 12000, 'Test partition exceeds the original request budget')
    require(all(part['source_bindings'].get(k) == v for k, v in full.get('source_bindings', {}).items()) and
            part['source_bindings'].get(meta['bundle']['path']) == meta['bundle']['sha256'], 'Test source bindings changed')
    return full, meta


def bindings(part, *, verify=False):
    """Hash metadata/proofs, optionally rerun the existing release verifier."""
    full, meta = validate_part(part)
    root = Path(meta['bundle']['path']).parent
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    seen = set()
    for index, ref in enumerate(meta['predecessors']):
        context = predecessor_context(ref, verify=verify)
        manifest, prior = context['manifest'], context['part']
        require(context['manifest_sha256'] not in seen, 'Repeated physical test predecessor')
        seen.add(context['manifest_sha256']); paths.extend(context['paths'])
        _, prior_meta = validate_part(prior, full=full)
        require(prior_meta['partition_index'] == index and prior_meta['bundle'] == meta['bundle'] and
                prior_meta['predecessors'] == meta['predecessors'][:index] and
                manifest['scientific'].get('test_execution') == prior_meta and
                manifest['scientific'].get('test_bundle') == meta['bundle'] and
                manifest['scientific']['master_plan_sha256'] == full['master_plan_sha256'],
                'Predecessor does not belong to the exact ordered test bundle')
        covered = []
        for item in context['tasks']:
            worker, task = item['worker'], item['task']
            require(worker['success_expect']['mode'] == task['mode'] == 'generate' and
                    worker['success_expect']['requests'] == len(task['requests']) and
                    task['run_token'] == manifest['run_token'] and task['worker_name'] == worker['name'] and
                    task['conditions'] == full['conditions'], 'Test predecessor worker identity differs')
            if context['remote']:
                require(task['sampling'] == full['sampling'], 'Remote test predecessor sampling differs')
            covered.extend(task['requests'])
        require(len(covered) == len({r['request_id'] for r in covered}) == len(prior['requests']) and
                {r['request_id']: r for r in covered} == {r['request_id']: r for r in prior['requests']},
                'Test predecessor task coverage differs from its fixed partition')
    return sorted(set(paths))


def validate_against_ledger(part, ledger):
    full, meta = validate_part(part)
    for field, key in (('previously_committed_generation_requests', 'generation_requests'),
                       ('previously_committed_tf_requests', 'tf_requests'),
                       ('previously_committed_untouched_test_generation_requests', 'untouched_test_generation_requests'),
                       ('previously_committed_untouched_test_requests', 'untouched_test_requests')):
        require(part[field] == ledger[key], 'Test partition ledger is stale')
    prior = [p for p in ledger['phases'] if p['untouched_test_requests']]
    index = meta['partition_index']
    require(len(prior) == index and all(p.get('test_execution', {}).get('bundle') == meta['bundle'] for p in prior),
            'Cannot follow another test bundle, replay a piece, or skip a reservation')
    by_index = {p['test_execution']['partition_index']: p for p in prior}
    require(set(by_index) == set(range(index)), 'Earlier test partitions are duplicated or missing')
    expected_count = 0
    for j, ref in enumerate(meta['predecessors']):
        p = by_index[j]
        require(p['manifest_sha256'] == predecessor_manifest_sha256(ref) and
                p['wall_basis'] == 'actual_launch_to_terminal_receipt' and
                p['generation_requests'] == p['untouched_test_generation_requests'] ==
                p['untouched_test_requests'] == p['test_execution']['part_request_count'] and
                p['untouched_test_tf_requests'] == 0, 'Earlier test partition lacks its terminal exact reservation')
        expected_count += p['untouched_test_requests']
    require(ledger['untouched_test_requests'] == ledger['untouched_test_generation_requests'] == expected_count and
            ledger['untouched_test_tf_requests'] == 0, 'Test counts include requests outside the admitted bundle')
    current_ids = {r['request_id'] for r in part['requests']}
    require(not any(current_ids.intersection(p.get('generation_request_ids', [])) for p in ledger['phases']),
            'Test requests were already committed under another manifest')
    # Keep capacity for every as-yet uncommitted request, not merely this piece.
    remaining = len(full['requests']) - expected_count
    require(ledger['generation_requests'] + remaining <= 4096, 'Insufficient generation budget for the remaining complete test bundle')


def make_part(bundle_ref, partition_index, previous, predecessors=()):
    bundle, full = bundle_context(bundle_ref)
    require(type(partition_index) is int and 0 <= partition_index < len(bundle['execution_partitions']), 'Invalid test partition index')
    part = copy.deepcopy(full)
    # The pure bundle producer intentionally skips behavior_plan.write_plan;
    # its full plan has no publication-time source_bindings. Derive those from
    # the already verified bundle below, preserving any bindings it does carry.
    part['source_bindings'] = copy.deepcopy(full.get('source_bindings', {}))
    lookup = {r['request_id']: r for r in full['requests']}
    part['requests'] = [lookup[rid] for rid in bundle['execution_partitions'][partition_index]['request_ids']]
    scope_counts, condition_counts = part_counts(part['requests'], part['conditions'])
    part.update(new_generation_requests=len(part['requests']), scope_counts=scope_counts, counts_per_condition=condition_counts,
                previously_committed_generation_requests=previous['generations'], previously_committed_tf_requests=previous['teacher_forced'],
                previously_committed_untouched_test_generation_requests=previous['untouched_test_generations'],
                previously_committed_untouched_test_requests=previous['untouched_test_requests'],
                generation_requests_after_commit=previous['generations'] + len(part['requests']),
                prior_phase_manifest_bindings=copy.deepcopy(previous.get('phase_manifests', [])))
    part['source_bindings'][bundle_ref['path']] = bundle_ref['sha256']
    for item in previous.get('ledger_bindings', []):
        part['source_bindings'][item['path']] = item['sha256']
    part['test_execution'] = {'protocol': PROTOCOL, 'partition_index': partition_index, 'bundle': copy.deepcopy(bundle_ref),
        'part_request_count': len(part['requests']), 'predecessors': copy.deepcopy(list(predecessors)), 'no_intermediate_analysis': True}
    validate_part(part)
    return part


def write_part(*, bundle, bundle_sha256, partition_index, phase_root, output, predecessors=()):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'Preserve previous test plans; use a fresh directory')
    ref = {'path': str(Path(bundle)), 'sha256': bundle_sha256}
    _, full = bundle_context(ref)
    planner = helper('behavior_plan')
    previous = planner.counts_from_phase_ledger(phase_root, full['master_plan_sha256'], full.get('parent_plan_sha256'))
    part = make_part(ref, partition_index, previous, predecessors)
    validate_against_ledger(part, previous['phase_budget'])
    for path in bindings(part, verify=True):
        part['source_bindings'][str(path)] = sha(path)
    for name, digest in part['source_bindings'].items():
        require(sha(name) == digest, 'Bound test input or prior evidence changed')
    require(planner.counts_from_phase_ledger(phase_root, full['master_plan_sha256'], full.get('parent_plan_sha256')) == previous,
            'Ledger changed during test execution preparation')
    output.mkdir(parents=True)
    path = output / 'request_plan.json'
    with path.open('x') as f:
        f.write(canonical(part))
    path.chmod(0o400)
    return {'request_plan': str(path), 'sha256': sha(path), 'partition_index': partition_index,
            'requests': len(part['requests']), 'test_bundle': ref, 'budget_reserved': False, 'no_compute_launched': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    print(canonical(write_part(**json.loads(args.spec.read_text()))), end='')


if __name__ == '__main__':
    main()
