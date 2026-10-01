"""Exact, validation-only recovery of the interrupted first finalist round.

Never launches a model, evaluates generated code, refunds reservations, or marks
the original failed process successful. Generated result objects are opaque:
only their original line bytes and request identities enter the recovery audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'infra.gpu03.direction_discovery'

from . import execution_partition as partition

PROTOCOL = 'finalist_round0_exact_recovery_v1'
KIND = 'recovered_finalist_round0'
RECOVERY_PHASE = 'behavior_finalist_validation_round0_recovery'
MASTER_SHA = '1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2'
ORIGINAL_SHA = '9f8b8b544d558ca758777076089f69c6c0690be2f64c9ca3c8c11b3e7eaa54e5'
FULL_SHA = '73854c48430336048dfdcee0525cd8c0a082f96fe4a784b2667592cc5c3f48e3'
COUNTS = {'gpu_5': 84, 'gpu_6': 89, 'gpu_7': 83}
CRITICAL = ('infra/gpu03/direction_discovery/engine.py',
            'infra/gpu03/direction_discovery/intervention.py',
            'infra/gpu03/activation_dataset/extract_triplet_raw.py',
            'infra/gpu03/activation_dataset/extract_delta_activations.py',
            'infra/gpu03/factorial_rollouts/factorial_common.py')
TRANSPORT = {'run_token', 'worker_name', 'gpu_id', 'output', 'deadline_seconds', 'requests'}
BOOKKEEPING = partition.BOOKKEEPING | {'finalist_recovery'}
MAX_JOURNAL_BYTES = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def parse(data):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(data, object_pairs_hook=unique)
    json.dumps(result, allow_nan=False)
    return result


def reference(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha(path)}


def snapshot(ref, *, immutable=True):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Invalid immutable binding')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            (not immutable or not path.stat().st_mode & 0o222), 'Missing, mutable, or symlinked recovery input')
    data = path.read_bytes()
    require(digest(data) == ref['sha256'], 'Recovery input hash changed')
    return data


def read(ref):
    return parse(snapshot(ref))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data if isinstance(data, bytes) else encoded(data))
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return reference(path)


def fresh(output, forbidden):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Preserve existing recovery output')
    for path in forbidden:
        path = Path(path).resolve()
        require(output != path and not output.is_relative_to(path) and not path.is_relative_to(output),
                'Recovery output overlaps preserved inputs')
    return output


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and p != root/'artifact_manifest.json'}


def package_manifest(root):
    return write(Path(root)/'artifact_manifest.json', {'algorithm':'sha256', 'files':inventory(root)})


def check_package(ref):
    root = Path(ref['path']).parent
    require(all(not p.is_symlink() and (not p.is_file() or not p.stat().st_mode & 0o222) for p in root.rglob('*')),
            'Recovery package is mutable or symlinked')
    value = read(reference(root/'artifact_manifest.json'))
    require(value == {'algorithm':'sha256', 'files':inventory(root)}, 'Recovery package artifact inventory changed')


def bound(m, path):
    path = Path(path)
    entry = m['bound_files'][str(path)]
    data = snapshot({'path': str(path), 'sha256': entry['sha256']})
    require(len(data) == entry['size_bytes'], 'Manifest bound size differs')
    return parse(data)


def original_context(ref):
    require(ref['sha256'] == ORIGINAL_SHA, 'Recovery is restricted to the reviewed failed original round')
    m = read(ref)
    require(m['host'] == 'gpu-04' and m['phase'] == 'behavior_finalist_validation_round0' and
            m['scientific']['master_plan_sha256'] == MASTER_SHA and m['scientific']['training'] is False,
            'Wrong failed producer or master lineage')
    p = Path(m['stage']) / 'input/request_plan.json'
    original = bound(m, p)
    full, meta = partition.validate_part(original)
    require(meta['round_index'] == 0 and meta['full_plan_sha256'] == FULL_SHA and
            m['scientific']['execution_partition'] == meta and len(original['requests']) == 370,
            'Recovery changed the original full740 or round370')
    return m, original, full


def journal(data, requests, count):
    """Validate only identity and byte framing; do not interpret result content."""
    require(len(data) <= MAX_JOURNAL_BYTES and data.endswith(b'\n'), 'Incomplete or oversized journal')
    lines = data.splitlines(keepends=True)
    require(len(lines) == count and len(requests) >= count, 'Journal prefix count differs')
    rows, offset = [], 0
    for line, planned in zip(lines, requests):
        require(line.endswith(b'\n'), 'Incomplete journal line')
        value = parse(line)
        require(isinstance(value.get('result'), dict) and all(value.get(k) == v for k, v in planned.items()),
                'Journal is not the exact original ordered task prefix')
        rows.append({'request_id': planned['request_id'], 'offset': offset, 'size_bytes': len(line), 'sha256': digest(line)})
        offset += len(line)
    require(len({row['request_id'] for row in rows}) == count, 'Duplicate completed request')
    return rows, lines


def _terminal(m, refs, release):
    values = {name: read(ref) for name, ref in refs.items() if name not in ('processes', 'resource_journal')}
    identity = {'run_token': m['run_token'], 'manifest_sha256': ORIGINAL_SHA}
    for name in ('launch', 'exit', 'failure', 'run_identity'):
        require(all(values[name].get(k) == v for k, v in identity.items()), 'Failed process receipt identity differs')
    end, failure, started = values['exit'], values['failure'], values['started']
    require(end.get('service_result') == 'exit-code' and end.get('exit_code_kind') == 'exited' and
            end.get('exit_status') == '1' and end.get('producer_summary_present') is False and
            end.get('failure_present') is True and failure.get('all_existing_outputs_preserved') is True and
            failure.get('cleanup', {}).get('owned_workers_released') is True and
            failure['cleanup'].get('gpu_release_verified') is True, 'Original is not positively failed and released')
    fields = started.get('fields', {})
    require(started.get('unit') == m['run_token'] + '.service' and fields.get('InvocationID') == end.get('invocation_id') and
            fields.get('KillMode') == 'control-group' and fields.get('ActiveState') == 'active' and
            fields.get('SubState') == 'running' and values['launch']['at'] <= started['at'] <= end['at'],
            'Original actual invocation/start/exit differs')
    from .phase_budget import exit_status_valid
    require(exit_status_valid(end['exit_code_kind'], end['exit_status']), 'Original exit is not positively terminal')
    require(release.get('purpose') == 'interrupted_finalist_operational_preservation' and
            release.get('status') == 'interrupted_terminal_released_preserved' and
            release.get('original_manifest_sha256') == ORIGINAL_SHA and release.get('run_token') == m['run_token'] and
            release.get('original_manifest') == str(Path(m['stage'])/'reviewed_manifest.json') and
            release.get('invocation_id') == end['invocation_id'] and release.get('terminal_receipt') == end and
            release.get('original_run_success') is False and release.get('owned_run_processes') == [] and
            all(release.get(k) is True for k in ('owned_workers_released', 'gpu_release_verified',
                'exact_cgroup_absent_or_empty', 'original_failure_preserved', 'no_generation_or_evaluation', 'no_partial_behavior_analysis')),
            'Independent failed-terminal/release proof is incomplete')
    processes = [parse(line) for line in snapshot(refs['processes']).splitlines()]
    require(len(processes) == len(m['workers']) and {p['worker'] for p in processes} == set(COUNTS) and
            len({p['pid'] for p in processes}) == len(processes), 'Original process journal worker identities differ')
    require(release.get('gpu_ids') == m['gpu_ids'] and release.get('at', 0) >= end['at'] and
            release.get('elapsed_seconds') == end['at'] - values['launch']['at'] and
            release.get('wall_accounting_basis') == 'actual_launch_to_terminal_receipt',
            'Independent release predates termination or changes devices')
    snapshot(refs['resource_journal'])


def release_bindings(ref, expected_sources):
    proof = read(ref); root = Path(ref['path']).parent
    copied = proof['copied_files']; paths, seen = [Path(ref['path'])], set()
    for name, entry in copied.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and entry['source_path'] not in seen,
                'Invalid preserved operational copy path/source')
        seen.add(entry['source_path']); path = root/relative
        data = snapshot({'path': str(path), 'sha256': entry['sha256']})
        require(len(data) == entry['size_bytes'] and
                data == snapshot({'path': entry['source_path'], 'sha256': entry['sha256']}, immutable=False),
                'Preserved operational copy differs from source')
        paths += [path, Path(entry['source_path'])]
    require(set(map(str, expected_sources)) <= seen, 'Independent preservation omitted required original evidence')
    return paths


def seal(original_manifest, manifest_sha256, terminal_release_ref, output):
    """Seal terminal metadata plus exact journal bytes; no result interpretation."""
    original_ref = {'path': str(original_manifest), 'sha256': manifest_sha256}
    m, original, full = original_context(original_ref)
    release = read(terminal_release_ref)
    stage, result, runtime = map(Path, (m['stage'], m['output'], m['runtime']))
    output = fresh(output, (stage, result, runtime, Path(terminal_release_ref['path']).parent))
    paths = {'launch': stage/'control/launch_intent.json', 'started': stage/'control/service_started.json',
             'exit': stage/'control/supervisor_exit.json', 'failure': result/'FAILURE.json',
             'run_identity': result/'run_identity.json', 'processes': runtime/'processes.jsonl',
             'resource_journal': runtime/'resource_journal.jsonl'}
    # Snapshot mode600 append-only operational files only after positive terminal proof.
    refs = {name: reference(path) for name, path in paths.items()}
    data = {name: snapshot(ref, immutable=False) for name, ref in refs.items()}
    for name, payload in data.items():
        refs[name] = write(output/'operational'/paths[name].name, payload)
    _terminal(m, refs, release)
    release_bindings(terminal_release_ref, [original_manifest, *paths.values(), *[w['command'][3] for w in m['workers']],
        *[Path(m['output'])/'workers'/w['name']/'results.jsonl' for w in m['workers']]])
    workers, completed = [], set()
    for w in m['workers']:
        name = w['name']; task_path = Path(w['command'][3]); task = bound(m, task_path)
        require(name in COUNTS and task['mode'] == 'generate' and task['conditions'] == full['conditions'] and
                task['sampling'] == full['sampling'], 'Original task science differs')
        path = Path(task['output'])/'results.jsonl'; captured = path.read_bytes()
        rows, _ = journal(captured, task['requests'], COUNTS[name]); completed.update(r['request_id'] for r in rows)
        workers.append({'worker': name, 'task': reference(task_path), 'original_journal': reference(path),
                        'journal': write(output/'journals'/name/'results.jsonl', captured), 'rows': rows,
                        'first_missing_request_id': task['requests'][len(rows)]['request_id']})
    missing = [r for r in original['requests'] if r['request_id'] not in completed]
    require(len(completed) == 256 and len(missing) == 114, 'Unexpected salvage counts')
    value = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'failed_original_sealed', 'source_sha256': sha(__file__),
             'original_manifest': original_ref, 'terminal_release': terminal_release_ref, 'operational': refs,
             'original_operational': {name: reference(path) for name, path in paths.items()},
             'full_plan': {'path': original['execution_partition']['full_plan'], 'sha256': FULL_SHA},
             'workers': workers, 'completed_request_ids': sorted(completed), 'missing_requests': missing,
             'ambiguous_request_ids': [w['first_missing_request_id'] for w in workers],
             'ambiguous_attempts_replayed_with_original_seed': True, 'completed_requests_regenerated': False,
             'original_reservation_retained': 370, 'additional_reservation': 114, 'no_outcome_selection': True}
    ref = write(output/'salvage.json', value)
    package_manifest(output)
    salvage_context(ref)
    return ref


def salvage_context(ref):
    s = read(ref)
    check_package(ref)
    require(s.get('schema_version') == 1 and s.get('protocol') == PROTOCOL and s.get('status') == 'failed_original_sealed' and
            s.get('source_sha256') == sha(__file__) and s.get('original_reservation_retained') == 370 and
            s.get('additional_reservation') == 114 and s.get('no_outcome_selection') is True and
            s.get('ambiguous_attempts_replayed_with_original_seed') is True and s.get('completed_requests_regenerated') is False,
            'Invalid sealed salvage policy/source')
    m, original, full = original_context(s['original_manifest'])
    require(s['full_plan'] == {'path': original['execution_partition']['full_plan'], 'sha256': FULL_SHA}, 'Salvage full plan differs')
    _terminal(m, s['operational'], read(s['terminal_release']))
    release_paths = release_bindings(s['terminal_release'], [s['original_manifest']['path'],
        *[r['path'] for r in s['original_operational'].values()], *[w['task']['path'] for w in s['workers']],
        *[w['original_journal']['path'] for w in s['workers']]])
    for name, old in s['original_operational'].items():
        require(snapshot(old, immutable=False) == snapshot(s['operational'][name]), 'Original operational proof changed after seal')
    require([w['worker'] for w in s['workers']] == [w['name'] for w in m['workers']] and set(COUNTS) == {w['worker'] for w in s['workers']},
            'Salvage worker population changed')
    completed, preserved, paths = set(), {}, [Path(ref['path']), Path(s['original_manifest']['path']), Path(s['full_plan']['path'])]
    for saved, worker in zip(s['workers'], m['workers']):
        task_path = Path(worker['command'][3]); task = bound(m, task_path)
        require(saved['task'] == reference(task_path) and saved['original_journal']['path'] == str(Path(task['output'])/'results.jsonl'),
                'Salvage journal/task source differs')
        data = snapshot(saved['journal'])
        require(data == snapshot(saved['original_journal'], immutable=False), 'Original journal changed after seal')
        rows, lines = journal(data, task['requests'], COUNTS[worker['name']])
        require(rows == saved['rows'] and saved['first_missing_request_id'] == task['requests'][len(rows)]['request_id'],
                'Sealed line offsets/hashes or ambiguous request differs')
        for row, line in zip(rows, lines):
            require(row['request_id'] not in completed, 'Duplicate preserved record')
            completed.add(row['request_id']); preserved[row['request_id']] = line
        paths += [task_path, Path(saved['journal']['path']), Path(saved['original_journal']['path'])]
    require(len(completed) == 256 and s['completed_request_ids'] == sorted(completed) and
            s['missing_requests'] == [r for r in original['requests'] if r['request_id'] not in completed] and
            len(s['missing_requests']) == 114 and
            s['ambiguous_request_ids'] == [w['first_missing_request_id'] for w in s['workers']], 'Salvage exact set difference changed')
    paths += [Path(r['path']) for r in [s['terminal_release'], *s['operational'].values(), *s['original_operational'].values()]]
    paths += [Path(m['stage'])/'input/request_plan.json', Path(ref['path']).parent/'artifact_manifest.json', *release_paths]
    return {'salvage': s, 'original': original, 'manifest': m, 'full': full, 'preserved': preserved, 'paths': paths}


def _meta(ref, s):
    return {'protocol': PROTOCOL, 'salvage': ref, 'original_manifest_sha256': ORIGINAL_SHA, 'full_plan_sha256': FULL_SHA,
            'logical_round_index': 0, 'completed_requests': 256, 'new_requests': 114, 'ambiguous_request_ids': s['ambiguous_request_ids']}


def make_plan(salvage_ref, previous):
    ctx = salvage_context(salvage_ref); s = ctx['salvage']; plan = copy.deepcopy(ctx['full'])
    plan['requests'] = s['missing_requests']
    plan.update(new_generation_requests=114, scope_counts={'primary': 114},
                counts_per_condition=dict(Counter(r['condition_id'] for r in plan['requests'])),
                previously_committed_generation_requests=previous['generations'],
                previously_committed_tf_requests=previous['teacher_forced'],
                previously_committed_untouched_test_generation_requests=previous['untouched_test_generations'],
                previously_committed_untouched_test_requests=previous['untouched_test_requests'],
                generation_requests_after_commit=previous['generations']+114,
                prior_phase_manifest_bindings=previous.get('phase_manifests', []),
                finalist_recovery=_meta(salvage_ref, s))
    plan.setdefault('source_bindings', {}).update({str(path): sha(path) for path in ctx['paths']})
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require('execution_partition' not in plan and 'test_execution' not in plan and 'test_bundle' not in plan,
            'Recovery is not a new execution round or test request')
    meta = plan['finalist_recovery']; ctx = salvage_context(meta['salvage']); s, full = ctx['salvage'], ctx['full']
    require(meta == _meta(meta['salvage'], s) and {k:v for k,v in plan.items() if k not in BOOKKEEPING} ==
            {k:v for k,v in full.items() if k not in BOOKKEEPING}, 'Recovery changed frozen scientific fields')
    require(plan['requests'] == s['missing_requests'] and plan['new_generation_requests'] == 114 and
            plan['scope_counts'] == {'primary': 114} and plan['counts_per_condition'] == dict(Counter(r['condition_id'] for r in s['missing_requests'])),
            'Recovery changed exact missing request identities/order/counts')
    for key in ('previously_committed_generation_requests', 'previously_committed_tf_requests',
                'previously_committed_untouched_test_generation_requests', 'previously_committed_untouched_test_requests'):
        require(type(plan.get(key)) is int and plan[key] >= 0, 'Invalid recovery counters')
    require(plan['previously_committed_generation_requests'] >= 2156 and
            plan['generation_requests_after_commit'] == plan['previously_committed_generation_requests']+114 <= 4096 and
            plan['previously_committed_tf_requests'] <= 12000 and
            plan['previously_committed_untouched_test_generation_requests'] == plan['previously_committed_untouched_test_requests'] == 0,
            'Recovery refunds spent work or crosses request/test budgets')
    require(all(plan['source_bindings'].get(k) == v for k,v in full.get('source_bindings', {}).items()) and
            all(plan['source_bindings'].get(str(p)) == sha(p) for p in ctx['paths']), 'Recovery lost source/salvage bindings')
    return full, meta


def bindings(plan, verify=False):
    validate_plan(plan)
    # Same full immutable-byte check in both modes; no model/evaluation call.
    return salvage_context(plan['finalist_recovery']['salvage'])['paths']


plan_bindings = bindings


def recovery_worker_requests(plan):
    """Retain original device and within-device request order for the exact tails."""
    validate_plan(plan)
    ctx = salvage_context(plan['finalist_recovery']['salvage'])
    result = {}
    for worker in ctx['manifest']['workers']:
        task = bound(ctx['manifest'], Path(worker['command'][3]))
        result[worker['name']] = task['requests'][COUNTS[worker['name']]:]
    require([len(result[name]) for name in ('gpu_5','gpu_6','gpu_7')] == [40,34,40] and
            {r['request_id']:r for rows in result.values() for r in rows} == {r['request_id']:r for r in plan['requests']},
            'Recovery device tails differ from the sealed missing population')
    return result


def validate_against_ledger(plan, ledger):
    _, meta = validate_plan(plan)
    require(plan['previously_committed_generation_requests'] == ledger['generation_requests'] and
            plan['previously_committed_tf_requests'] == ledger['tf_requests'] and ledger['untouched_test_requests'] == 0,
            'Recovery ledger is stale or test work has started')
    old = [p for p in ledger['phases'] if p['manifest_sha256'] == ORIGINAL_SHA]
    require(len(old) == 1 and old[0]['generation_requests'] == 370 and old[0].get('tf_requests', 0) == 0 and
            old[0]['wall_basis'] == 'actual_launch_to_terminal_receipt' and
            old[0].get('execution_partition', {}).get('full_plan_sha256') == FULL_SHA and
            old[0]['execution_partition']['round_index'] == 0, 'Original failed370 reservation/terminal identity missing')
    requested = {r['request_id'] for r in plan['requests']}
    for phase in ledger['phases']:
        if phase['manifest_sha256'] == ORIGINAL_SHA:
            continue
        require(not requested.intersection(phase.get('generation_request_ids', [])) and
                phase.get('finalist_recovery', {}).get('full_plan_sha256') != FULL_SHA and
                phase.get('execution_partition', {}).get('full_plan_sha256') != FULL_SHA,
                'Recovery was replayed or a later logical round already exists')


def _recovery_context(package, *, verify):
    require(set(package) == {'manifest', 'manifest_sha256', 'artifact_manifest_sha256', 'verification', 'verification_sha256'},
            'Incomplete successful recovery package reference')
    m = read({'path': package['manifest'], 'sha256': package['manifest_sha256']})
    proof = parse(snapshot({'path': package['verification'], 'sha256': package['verification_sha256']}, immutable=False))
    require(proof.get('status') == 'verified' and proof.get('manifest_sha256') == package['manifest_sha256'] and
            proof.get('run_token') == m['run_token'] and proof.get('gpu_release_verified') is True,
            'Recovery producer is not independently successful/released')
    require(m['host'] == 'gpu-04' and m['phase'] == RECOVERY_PHASE and m['scientific']['master_plan_sha256'] == MASTER_SHA and
            m['scientific']['training'] is False and 'execution_partition' not in m['scientific'] and
            'test_execution' not in m['scientific'] and 'test_bundle' not in m['scientific'], 'Wrong recovery producer lineage')
    plan_path = Path(m['stage'])/'input/request_plan.json'; plan = bound(m, plan_path)
    full, meta = validate_plan(plan)
    require(m['scientific'].get('finalist_recovery') == meta, 'Recovery manifest lost its explicit recovery binding')
    artifact_path = Path(m['output'])/'artifact_manifest.json'
    artifact = parse(snapshot({'path': str(artifact_path), 'sha256': package['artifact_manifest_sha256']}, immutable=False))
    if verify:
        from . import supervisor
        require(supervisor.verify(Path(package['manifest']), package['manifest_sha256']) == proof,
                'Fresh independent recovery verification differs')
    ctx = salvage_context(meta['salvage']); original = ctx['manifest']
    require(m['gpu_ids'] == original['gpu_ids'] == [5,6,7] and m['gpu_uuids'] == original['gpu_uuids'] and
            [w['name'] for w in m['workers']] == ['gpu_5','gpu_6','gpu_7'], 'Recovery changed original devices')
    worker_requests = recovery_worker_requests(plan)
    for relative in CRITICAL:
        require(m['bound_files'][str(Path(m['source_root'])/relative)] ==
                original['bound_files'][str(Path(original['source_root'])/relative)], 'Recovery changed qualified inference/journaling source')
    require(m['runtime_versions'] == original['runtime_versions'] and m['python'] == original['python'], 'Recovery runtime changed')
    old_task = bound(original, Path(original['workers'][0]['command'][3]))
    rows, paths = {}, [Path(package['manifest']), Path(package['verification']), artifact_path, plan_path]
    planned = {r['request_id']: r for r in plan['requests']}
    for worker in m['workers']:
        task_path = Path(worker['command'][3]); task = bound(m, task_path)
        require(worker['gpu_id'] == task['gpu_id'] == int(worker['name'][4:]) and
                task['worker_name'] == worker['name'] and task['run_token'] == m['run_token'] and
                task['requests'] == worker_requests[worker['name']] and
                {k:v for k,v in task.items() if k not in TRANSPORT} == {k:v for k,v in old_task.items() if k not in TRANSPORT} and
                worker['success_expect'] == {'mode': 'generate', 'requests': len(task['requests'])}, 'Recovery task model/numerics/sampling differs')
        require(all(planned.get(r['request_id']) == r for r in task['requests']), 'Recovery task contains an unplanned/replayed completion')
        p = Path(task['output'])/'results.jsonl'; rel = str(p.relative_to(m['output']))
        data = snapshot({'path': str(p), 'sha256': artifact['files'][rel]['sha256']}, immutable=False)
        require(len(data) == artifact['files'][rel]['size_bytes'], 'Recovery journal size differs')
        index, lines = journal(data, task['requests'], len(task['requests']))
        for entry, line in zip(index, lines):
            require(entry['request_id'] not in rows, 'Duplicate recovery row')
            rows[entry['request_id']] = line
        paths += [task_path, p]
    require(set(rows) == set(planned) and len(rows) == 114, 'Recovery is incomplete')
    return {'manifest': m, 'plan': plan, 'full': full, 'proof': proof, 'salvage': ctx, 'rows': rows, 'paths': paths}


def complete(salvage_ref, recovery_package, output):
    """Create a logical370 byte union only after successful fresh114 verification."""
    ctx = _recovery_context(recovery_package, verify=True)
    require(ctx['plan']['finalist_recovery']['salvage'] == salvage_ref, 'Successful recovery belongs to another salvage')
    saved = ctx['salvage']; original = saved['manifest']; fresh_m = ctx['manifest']
    output = fresh(output, (original['stage'], original['output'], original['runtime'],
                           fresh_m['stage'], fresh_m['output'], fresh_m['runtime'], Path(salvage_ref['path']).parent))
    rows = {**saved['preserved'], **ctx['rows']}
    require(not set(saved['preserved']) & set(ctx['rows']) and len(rows) == 370, 'Logical round contains replay/omission')
    combined = b''.join(rows[r['request_id']] for r in saved['original']['requests'])
    rows_ref = write(output/'generations.jsonl', combined)
    value = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'verified_logical_round0', 'source_sha256': sha(__file__),
             'salvage': salvage_ref, 'recovery_package': recovery_package, 'original_manifest_sha256': ORIGINAL_SHA,
             'full_plan_sha256': FULL_SHA, 'rows': rows_ref, 'request_ids': [r['request_id'] for r in saved['original']['requests']],
             'preserved_records': 256, 'new_records': 114, 'logical_records': 370, 'physical_reservations': 484,
             'original_failed_state_preserved': True, 'no_outcome_selection': True, 'evaluation_requires_full740': True,
             'process_exit_and_release_verified_by_this_cpu_module': False}
    ref = {'kind': KIND, 'logical_round': write(output/'logical_round.json', value)}
    package_manifest(output)
    logical_context(ref, verify=True)
    return ref


def logical_context(ref, verify=False):
    require(isinstance(ref, dict) and set(ref) == {'kind', 'logical_round'} and ref['kind'] == KIND, 'Invalid logical-round reference')
    value = read(ref['logical_round'])
    check_package(ref['logical_round'])
    require(value.get('schema_version') == 1 and value.get('protocol') == PROTOCOL and value.get('source_sha256') == sha(__file__) and
            value.get('status') == 'verified_logical_round0' and value.get('original_manifest_sha256') == ORIGINAL_SHA and
            value.get('full_plan_sha256') == FULL_SHA and
            (value.get('preserved_records'), value.get('new_records'), value.get('logical_records'), value.get('physical_reservations')) == (256,114,370,484) and
            all(value.get(k) is True for k in ('original_failed_state_preserved', 'no_outcome_selection', 'evaluation_requires_full740')) and
            value.get('process_exit_and_release_verified_by_this_cpu_module') is False, 'Invalid logical completion policy/counts')
    ctx = _recovery_context(value['recovery_package'], verify=verify)
    require(value['salvage'] == ctx['plan']['finalist_recovery']['salvage'], 'Logical salvage identity differs')
    original = ctx['salvage']['original']; ids = [r['request_id'] for r in original['requests']]
    rows = {**ctx['salvage']['preserved'], **ctx['rows']}
    require(not set(ctx['salvage']['preserved']) & set(ctx['rows']) and value['request_ids'] == ids and len(rows) == 370 and
            snapshot(value['rows']) == b''.join(rows[rid] for rid in ids), 'Logical round changed original/recovered line bytes or order')
    paths = ctx['paths'] + ctx['salvage']['paths'] + [Path(ref['logical_round']['path']), Path(value['rows']['path']),
        Path(ref['logical_round']['path']).parent/'artifact_manifest.json']
    return {'full': ctx['full'], 'original_manifest_sha256': ORIGINAL_SHA, 'full_plan_sha256': FULL_SHA,
            'request_ids': ids, 'rows_path': Path(value['rows']['path']), 'rows_sha256': value['rows']['sha256'],
            'recovery_manifest_sha256': value['recovery_package']['manifest_sha256'], 'proof': value,
            'logical_round': ref['logical_round'], 'paths': paths}


def logical_bindings(ref, verify=False):
    return logical_context(ref, verify=verify)['paths']


def verify_logical_round(ref):
    return logical_context(ref, verify=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('seal', 'make-plan', 'complete', 'verify'), required=True)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args(); spec = read({'path': str(args.spec), 'sha256': args.spec_sha256})
    if args.operation == 'seal': result = seal(**spec)
    elif args.operation == 'make-plan':
        plan = make_plan(spec['salvage'], spec['previous']); result = write(spec['output'], plan)
    elif args.operation == 'complete': result = complete(**spec)
    else:
        ctx = verify_logical_round(spec); result = {k:v for k,v in ctx.items() if k not in ('full', 'paths', 'proof')}
        result['rows_path'] = str(result['rows_path'])
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
