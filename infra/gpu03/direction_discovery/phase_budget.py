"""Conservative request and elapsed/reserved wall-time accounting across amendments.

Every frozen task is committed, including unused/failed tasks. Actual elapsed
wall time is used only with a complete, identity-matched systemd receipt chain;
an absent exit receipt keeps the full independently bounded reservation.
"""
import hashlib
from contextlib import contextmanager
import fcntl
import os
import stat
import time
import json
import math
from pathlib import Path
import re
import signal
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

MODES = frozenset(('qualify', 'generate', 'tf', 'supplement', 'cache_aux', 'prefix_audit', 'fixed_cache'))
# systemd's documented service-result vocabulary. Any new/unknown value requires
# an explicit review; it must not make a phase cheaper automatically.
SERVICE_RESULTS = frozenset(('success', 'resources', 'timeout', 'exit-code', 'signal',
    'core-dump', 'watchdog', 'start-limit-hit', 'oom-kill', 'exec-condition', 'protocol'))
PARTITIONS = frozenset(('configuration_validation', 'untouched_test'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Missing or symlinked phase proof: ' + str(path))
    return json.loads(Path(path).read_text())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_file(m, path, bindings):
    path = Path(path)
    bound = m['bound_files'].get(str(path))
    require(path.is_file() and not path.is_symlink() and isinstance(bound, dict) and sha(path) == bound.get('sha256') and
            path.stat().st_size == bound.get('size_bytes'), 'Prior phase bound task/plan changed: ' + str(path))
    bindings.append(path)
    return path


def checked_bound(m, path, bindings):
    return read_json(checked_file(m, path, bindings))


def exit_status_valid(kind, status):
    # ExecStopPost EXIT_STATUS is decimal for normal exits and a signal name
    # (without SIG) for killed/dumped processes; accepting missing values is unsafe.
    if kind == 'exited':
        return isinstance(status, str) and status.isdecimal() and 0 <= int(status) <= 255
    if kind in ('killed', 'dumped'):
        return isinstance(status, str) and 'SIG' + status in signal.Signals.__members__
    return False


def terminal_seconds(m, digest, control, bindings):
    intent_path, started_path, receipt_path = (control / name for name in
        ('launch_intent.json', 'service_started.json', 'supervisor_exit.json'))
    seconds = m['limits']['systemd_runtime_seconds']
    require(finite_number(seconds) and seconds > 0, 'Invalid prior phase runtime bound')
    basis = 'full_reserved_deadline'
    if not receipt_path.exists():
        if intent_path.exists():
            intent = read_json(intent_path)
            require(intent.get('run_token') == m['run_token'] and intent.get('manifest_sha256') == digest and
                    finite_number(intent.get('at')) and intent['at'] > 0, 'Active phase intent identity/time differs')
            bindings.append(intent_path)
        return float(seconds), basis
    require(intent_path.exists() and started_path.exists(), 'Terminal phase has no complete launch/service proof')
    intent, started, end = (read_json(path) for path in (intent_path, started_path, receipt_path))
    for value in (intent, end):
        require(value.get('run_token') == m['run_token'] and value.get('manifest_sha256') == digest,
                'Prior phase receipt identity differs')
    fields = started.get('fields', {})
    unit = m['run_token'] + '.service'
    require(started.get('unit') == unit and fields.get('ActiveState') == 'active' and fields.get('SubState') == 'running' and
            isinstance(fields.get('MainPID'), str) and fields['MainPID'].isdecimal() and int(fields['MainPID']) > 0 and
            fields.get('KillMode') == 'control-group' and
            isinstance(fields.get('ControlGroup'), str) and Path(fields['ControlGroup']).name == unit,
            'Prior service-start proof is incomplete or belongs to another unit')
    invocation = end.get('invocation_id')
    require(isinstance(invocation, str) and re.fullmatch('[0-9a-f]{32}', invocation) and
            fields.get('InvocationID') == invocation, 'Prior systemd invocation identity differs')
    kind, status, result = end.get('exit_code_kind'), end.get('exit_status'), end.get('service_result')
    require(exit_status_valid(kind, status) and result in SERVICE_RESULTS, 'Prior phase is not positively terminal')
    require(result != 'success' or kind == 'exited' and status == '0', 'Successful service result conflicts with exit status')
    require(type(end.get('producer_summary_present')) is bool and type(end.get('failure_present')) is bool,
            'Terminal receipt lacks producer/failure presence flags')
    times = [value.get('at') for value in (intent, started, end)]
    require(all(finite_number(value) and value > 0 for value in times) and times[0] <= times[1] <= times[2],
            'Invalid or unordered terminal wall-time receipts')
    bindings.extend((intent_path, started_path, receipt_path))
    return float(times[2] - times[0]), 'actual_launch_to_terminal_receipt'


def _reallocation_helper():
    try:
        from . import prelaunch_reallocation
    except ImportError:
        import prelaunch_reallocation
    return prelaunch_reallocation


@contextmanager
def publication_lock(root, *, timeout=60):
    """Shared local/remote admission barrier; bounded, exact ledger root only."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Unsafe publication lock root')
    path = root / '.direction_discovery_publication.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_uid == os.getuid(),
                'Publication lock is not an owned regular file')
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'Publication lock timed out')
                time.sleep(.1)
        yield
    finally:
        os.close(fd)


def assert_concurrency(ledger, new_gpu_count):
    require(type(new_gpu_count) is int and 1 <= new_gpu_count <= 8, 'Invalid new GPU allocation count')
    active = [p for p in ledger['phases'] if p['wall_basis'] == 'full_reserved_deadline']
    require(sum(p.get('gpu_count', 8) for p in active) + new_gpu_count <= 8,
            'Global eight-GPU concurrency cannot be established from terminal receipts')


def entry_terminal(entry):
    if entry.get('remote_generation') is not None:
        require(entry.get('terminal') is not None, 'Remote phase has no verified terminal import')
        return entry['terminal']
    return read_json(entry['path'].parent / 'control/supervisor_exit.json')


def account(root, plan_sha, parent_sha=None, *, replacement_stage=None):
    lineage = {plan_sha, *([parent_sha] if parent_sha is not None else [])}
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in lineage),
            'Invalid master-plan lineage digest')
    result = {'generation_requests': 0, 'tf_requests': 0, 'untouched_test_generation_requests': 0,
              'untouched_test_tf_requests': 0, 'untouched_test_requests': 0,
              'gpu_phase_wall_seconds': 0., 'phases': [], 'bindings': []}
    seen_tokens = set()
    entries = []
    for path in sorted(Path(root).glob('codex-discovery-*-stage/reviewed_manifest.json')):
        m = read_json(path)
        if m.get('scientific', {}).get('master_plan_sha256') not in lineage:
            continue
        token = m.get('run_token')
        require(m.get('host') == 'gpu-04' and m['scientific'].get('training') is False and
                token == path.parent.name.removesuffix('-stage') and str(path.parent) == m.get('stage') and
                token not in seen_tokens, 'Prior phase identity differs or duplicates a run token')
        seen_tokens.add(token)
        digest = sha(path)
        bindings = [path]
        counts = {key: 0 for key in ('generation_requests', 'tf_requests', 'untouched_test_generation_requests',
                                    'untouched_test_tf_requests', 'untouched_test_requests')}
        require(isinstance(m.get('workers'), list) and m['workers'], 'Prior phase worker coverage is empty')
        task_paths = set()
        phase_request_ids = set()
        expected_partition_ids = None
        execution_part = None
        test_part = None
        recovery_part = None
        capability_part = None
        generation_ids = set()
        for worker in m['workers']:
            expected = worker['success_expect']
            n, mode = expected.get('requests'), expected.get('mode')
            require(type(n) is int and n > 0 and mode in MODES, 'Invalid prior phase mode/request count')
            command = worker['command']
            require(isinstance(command, list) and len(command) == 4 and command[2] == '--task', 'Unknown prior worker task command')
            task_path = Path(command[3])
            require(str(task_path) not in task_paths, 'Duplicate prior task path')
            task_paths.add(str(task_path))
            task = checked_bound(m, task_path, bindings)
            require(task.get('mode') == mode and task.get('run_token') == token and task.get('worker_name') == worker['name'] and
                    isinstance(task.get('requests'), list) and len(task['requests']) == n and
                    len({r['request_id'] for r in task['requests']}) == n, 'Prior bound task identity/count differs')
            request_ids = {r['request_id'] for r in task['requests']}
            require(not phase_request_ids & request_ids, 'Duplicate request IDs across prior workers')
            phase_request_ids.update(request_ids)
            counts['generation_requests'] += n * (3 if mode == 'qualify' else 1 if mode == 'generate' else 0)
            if mode == 'generate':
                generation_ids.update(request_ids)
            counts['tf_requests'] += n * (3 if mode == 'qualify' else 1 if mode == 'tf' else 0)
            if mode in ('generate', 'tf'):
                request_plan = checked_bound(m, path.parent / 'input/request_plan.json', bindings)
                if 'no_loophole_capability' in request_plan:
                    from infra.gpu03.direction_discovery import no_loophole_plan
                    no_loophole_plan.validate_full(request_plan)
                    meta = request_plan['no_loophole_capability']
                    require(mode == 'generate' and m.get('phase') == 'no_loophole_capability' and
                            request_plan['master_plan_sha256'] == m['scientific']['master_plan_sha256'] and
                            m.get('authorization') == request_plan['authorization'] and
                            m['scientific'].get('no_loophole_capability') == meta and
                            task.get('conditions') == request_plan['conditions'] and
                            task.get('prepared_records') == request_plan['prepared_records'] and
                            task.get('sampling') == request_plan['sampling'] and
                            task.get('attention_policy') == 'exclusive_math' and
                            task.get('teacher_forced_padded_sequence_length') == 2176,
                            'Prior no-loophole capability metadata or inference protocol differs')
                    for proof_path in no_loophole_plan.bindings(request_plan, verify=True):
                        checked_file(m, proof_path, bindings)
                    require(capability_part is None or capability_part == meta,
                            'Prior workers disagree about no-loophole capability provenance')
                    capability_part = meta
                else:
                    require('no_loophole_capability' not in m['scientific'] and
                            m.get('phase') != 'no_loophole_capability',
                            'Prior manifest lost its no-loophole capability plan')
                if 'finalist_recovery' in request_plan:
                    from infra.gpu03.direction_discovery import finalist_recovery
                    _, recovery_meta = finalist_recovery.validate_plan(request_plan)
                    require(mode == 'generate' and request_plan['evaluation_partition'] == 'configuration_validation' and
                            'execution_partition' not in request_plan and 'test_execution' not in request_plan and
                            m['scientific'].get('finalist_recovery') == recovery_meta,
                            'Prior finalist recovery metadata or validation scope differs')
                    expected_tails = finalist_recovery.recovery_worker_requests(request_plan)
                    require(worker['name'] in expected_tails and task['requests'] == expected_tails[worker['name']] and
                            worker.get('gpu_id') == int(worker['name'].removeprefix('gpu_')),
                            'Recovery reservation changed original GPU assignment or tail order')
                    for proof_path in finalist_recovery.bindings(request_plan):
                        checked_file(m, proof_path, bindings)
                    require(recovery_part is None or recovery_part == recovery_meta, 'Recovery worker metadata differs')
                    recovery_part = recovery_meta
                else:
                    require('finalist_recovery' not in m['scientific'], 'Prior manifest lost its finalist recovery plan')
                if 'execution_partition' in request_plan:
                    try:
                        from . import execution_partition
                    except ImportError:
                        import execution_partition
                    _, meta = execution_partition.validate_part(request_plan)
                    require(m['scientific'].get('execution_partition') == meta, 'Prior manifest execution partition differs')
                    for proof_path in execution_partition.predecessor_bindings(request_plan):
                        checked_file(m, proof_path, bindings)
                    require(execution_part is None or execution_part == meta, 'Prior worker execution partition differs')
                    execution_part = meta
                else:
                    require('execution_partition' not in m['scientific'], 'Prior manifest lost its execution partition')
                if 'test_execution' in request_plan:
                    from infra.gpu03.direction_discovery import test_execution
                    _, meta = test_execution.validate_part(request_plan)
                    require(execution_part is None and mode == 'generate' and
                            request_plan['evaluation_partition'] == 'untouched_test' and
                            m['scientific'].get('test_execution') == meta and
                            m['scientific'].get('test_bundle') == meta['bundle'],
                            'Prior manifest test execution differs from its frozen bundle')
                    for proof_path in test_execution.bindings(request_plan):
                        checked_file(m, proof_path, bindings)
                    require(test_part is None or test_part == meta, 'Prior workers disagree about test execution')
                    test_part = meta
                else:
                    require('test_execution' not in m['scientific'] and 'test_bundle' not in m['scientific'],
                            'Prior manifest lost its test execution plan')
                require(request_plan.get('mode') == mode and request_plan.get('evaluation_partition') in PARTITIONS,
                        'Prior generation/TF partition plan is invalid')
                plan_requests = {r['request_id']: r for r in request_plan['requests']}
                require(len(plan_requests) == len(request_plan['requests']) and
                        all(plan_requests.get(r['request_id']) == r for r in task['requests']),
                        'Prior task requests differ from bound partition plan')
                if expected_partition_ids is None:
                    expected_partition_ids = set(plan_requests)
                require(expected_partition_ids == set(plan_requests), 'Prior workers have different request plans')
                if request_plan['evaluation_partition'] == 'untouched_test':
                    counts['untouched_test_generation_requests' if mode == 'generate' else 'untouched_test_tf_requests'] += n
                    counts['untouched_test_requests'] += n
        if expected_partition_ids is not None:
            require(phase_request_ids == expected_partition_ids, 'Prior workers do not cover the complete committed request plan')
        seconds, basis = terminal_seconds(m, digest, path.parent / 'control', bindings)
        entries.append({'path': path, 'manifest': m, 'digest': digest, 'bindings': bindings,
                        'counts': counts, 'seconds': seconds, 'basis': basis,
                        'execution_partition': execution_part, 'test_execution': test_part,
                        'finalist_recovery': recovery_part, 'no_loophole_capability': capability_part,
                        'generation_ids': generation_ids})

    # The registry contains byte-identical remote manifests and an explicit
    # authority permit. It is never scanned as a second local physical phase.
    from infra.gpu03.direction_discovery import remote_generation
    remote_entries = remote_generation.reservation_entries(root, lineage)
    require(not seen_tokens.intersection(e['manifest']['run_token'] for e in remote_entries),
            'Remote reservation reuses a local run token')
    require(len({e['digest'] for e in entries + remote_entries}) == len(entries + remote_entries),
            'One physical deployment was registered more than once')
    entries.extend(remote_entries)

    # A failed original remains charged in full. Only its sealed missing set can
    # overlap one additional, separately charged recovery reservation.
    for entry in entries:
        meta = entry['finalist_recovery']
        if meta is None:
            continue
        originals = [other for other in entries if other['digest'] == meta['original_manifest_sha256']]
        require(len(originals) == 1 and originals[0]['execution_partition'] is not None and
                originals[0]['execution_partition']['round_index'] == 0 and
                originals[0]['execution_partition']['full_plan_sha256'] == meta['full_plan_sha256'] and
                originals[0]['counts']['generation_requests'] == 370 and
                originals[0]['basis'] == 'actual_launch_to_terminal_receipt' and
                entry['counts']['generation_requests'] == meta['new_requests'] == 114 and
                entry['counts']['tf_requests'] == entry['counts']['untouched_test_requests'] == 0 and
                entry['generation_ids'] < originals[0]['generation_ids'],
                'Recovery must retain the full charged failed original and exact additional missing requests')
        old_receipt = read_json(originals[0]['path'].parent / 'control/supervisor_exit.json')
        require(old_receipt['service_result'] != 'success' and old_receipt['failure_present'] is True and
                old_receipt['producer_summary_present'] is False, 'Recovery original must remain positively failed')
        require(len([other for other in entries if other['finalist_recovery'] is not None and
                     other['finalist_recovery']['original_manifest_sha256'] == meta['original_manifest_sha256']]) == 1,
                'Recovery reservation already committed; no implicit retry')
        for other in entries:
            if other is not entry and other is not originals[0]:
                require(not entry['generation_ids'] & other['generation_ids'], 'Recovery requests replay another reservation')

    test_entries = [entry for entry in entries if entry['counts']['untouched_test_requests']]
    if any(entry['test_execution'] is not None for entry in test_entries):
        require(all(entry['test_execution'] is not None for entry in test_entries),
                'A fixed test bundle cannot mix with another prior test phase')
        bundle = test_entries[0]['test_execution']['bundle']
        by_part = {entry['test_execution']['partition_index']: entry for entry in test_entries}
        require(len(by_part) == len(test_entries) and set(by_part) == set(range(len(test_entries))) and
                all(entry['test_execution']['bundle'] == bundle for entry in test_entries),
                'Test bundle reservations are duplicated, skipped, or belong to another bundle')
        for index, entry in by_part.items():
            meta, counts = entry['test_execution'], entry['counts']
            require(counts['generation_requests'] == counts['untouched_test_generation_requests'] ==
                    counts['untouched_test_requests'] == meta['part_request_count'] and
                    counts['tf_requests'] == counts['untouched_test_tf_requests'] == 0,
                    'Fixed test partition request counts differ')
            for other in entries:
                if other is not entry:
                    require(not entry['generation_ids'] & other['generation_ids'], 'Test requests already committed in another phase')
            for j, ref in enumerate(meta['predecessors']):
                prior = by_part[j]
                require(prior['digest'] == __import__('infra.gpu03.direction_discovery.test_execution', fromlist=['predecessor_manifest_sha256']).predecessor_manifest_sha256(ref) and
                        prior['basis'] == 'actual_launch_to_terminal_receipt',
                        'Test continuation lacks its exact terminal predecessor reservation')
                receipt = entry_terminal(prior)
                require(receipt['service_result'] == 'success' and receipt['exit_code_kind'] == 'exited' and
                        receipt['exit_status'] == '0' and receipt['producer_summary_present'] is True and
                        receipt['failure_present'] is False, 'Test continuation follows an unsuccessful predecessor')

    # Execution partitioning never refunds or deduplicates work: each round may
    # be committed once, including failed or never-launched reservations.
    for entry in entries:
        meta = entry['execution_partition']
        if meta is None:
            continue
        for other in entries:
            if other is not entry:
                allowed_recovery = (other['finalist_recovery'] is not None and
                    other['finalist_recovery']['original_manifest_sha256'] == entry['digest'])
                require(allowed_recovery or not entry['generation_ids'] & other['generation_ids'],
                        'Execution partition requests already committed in another phase')
        siblings = [e for e in entries if e['execution_partition'] is not None and
                    e['execution_partition']['full_plan_sha256'] == meta['full_plan_sha256']]
        require(len({e['execution_partition']['round_index'] for e in siblings}) == len(siblings),
                'Duplicate execution partition round reservation')
        if meta['round_index'] == 1:
            predecessors = [e for e in siblings if e['execution_partition']['round_index'] == 0]
            if meta['predecessor'].get('kind') == 'recovered_finalist_round0':
                from infra.gpu03.direction_discovery import execution_partition
                part = read_json(entry['path'].parent / 'input/request_plan.json')
                phases = [{'manifest_sha256': e['digest'], 'wall_basis': e['basis'], **e['counts'],
                           **({'execution_partition': e['execution_partition']} if e['execution_partition'] is not None else {}),
                           **({'finalist_recovery': e['finalist_recovery']} if e['finalist_recovery'] is not None else {})}
                          for e in entries]
                context = execution_partition.recovery_ledger_predecessor(part, phases)
                require(len(predecessors) == 1 and predecessors[0]['digest'] == context['original_manifest_sha256'],
                        'Logical recovered predecessor does not join the original first round')
                continue
            require(len(predecessors) == 1 and predecessors[0]['digest'] == meta['predecessor']['manifest_sha256'] and
                    predecessors[0]['basis'] == 'actual_launch_to_terminal_receipt',
                    'Second execution round lacks its terminal first-round reservation')

    # A transfer is a narrowly reviewed replacement of an unlaunched reservation,
    # never a generic request-ID deduplication or a refund for failed execution.
    by_stage = {str(entry['path'].parent): entry for entry in entries}
    pending_by_destination = {}
    retired_stages = set()
    for entry in entries:
        proof_path = entry['path'].parent / 'control/prelaunch_reallocation.json'
        if not (proof_path.exists() or proof_path.is_symlink()):
            continue
        helper = _reallocation_helper()
        pending = helper.validate_pending(entry['path'], entry['manifest'], proof_path)
        destination = pending['destination_stage']
        require(isinstance(destination, str) and Path(destination).is_absolute() and
                destination != str(entry['path'].parent), 'Invalid reservation replacement destination')
        require(destination not in pending_by_destination, 'Multiple retired reservations target one replacement')
        retired_stages.add(str(entry['path'].parent))
        pending_by_destination[destination] = (entry, pending, proof_path)
        entry['bindings'].extend(pending['bindings'])
        entry['retirement_status'] = 'pending_exact_replacement'
    require(not retired_stages.intersection(pending_by_destination),
            'Reservation replacement chains/cycles are forbidden')

    recognized_claims = set()
    for destination, (old, pending, proof_path) in pending_by_destination.items():
        child_path = Path(destination) / 'reviewed_manifest.json'
        if not (child_path.exists() or child_path.is_symlink()):
            continue
        require(destination in by_stage, 'Replacement manifest is outside the same ledger lineage')
        child = by_stage[destination]
        expected_claim = {'old_manifest_sha256': old['digest'], 'receipt_path': str(proof_path),
                          'receipt_sha256': sha(proof_path)}
        require(child['manifest']['scientific'].get('prelaunch_reallocation') == expected_claim,
                'Replacement manifest lacks exact retirement identity claim')
        pair_bindings = _reallocation_helper().validate_pair(
            old['path'], old['manifest'], child['path'], child['manifest'], pending)
        require(old['counts'] == child['counts'] and old['counts']['tf_requests'] > 0 and
                old['counts']['generation_requests'] == old['counts']['untouched_test_requests'] == 0,
                'Replacement reservation request counts/modes differ')
        old['bindings'].extend(pair_bindings)
        old['gross_reserved_counts'] = dict(old['counts'])
        old['gross_reserved_wall_seconds'] = old['seconds']
        old['counts'] = {key: 0 for key in old['counts']}
        old['seconds'] = 0.
        old['basis'] = 'unlaunched_reservation_transferred_to_exact_replacement'
        old['retirement_status'] = 'transferred_to_exact_replacement'
        old['replacement_run_token'] = child['manifest']['run_token']
        child['replaces_unlaunched_run_token'] = old['manifest']['run_token']
        recognized_claims.add(destination)
    for entry in entries:
        if 'prelaunch_reallocation' in entry['manifest']['scientific']:
            require(str(entry['path'].parent) in recognized_claims,
                    'Unrecognized reservation replacement claim')
        for key, value in entry['counts'].items():
            result[key] += value
        result['gpu_phase_wall_seconds'] += entry['seconds']
        result['bindings'].extend(entry['bindings'])
        phase = {'run_token': entry['manifest']['run_token'], 'manifest_sha256': entry['digest'], 'manifest_path': str(entry['path']),
                 'host': entry['manifest']['host'], 'gpu_count': len(entry['manifest'].get('gpu_ids', entry['manifest']['workers'])),
                 'wall_seconds': entry['seconds'], 'wall_basis': entry['basis'], **entry['counts']}
        if entry.get('remote_generation') is not None:
            phase['remote_generation'] = entry['remote_generation']
        if entry['generation_ids']:
            phase['generation_request_ids'] = sorted(entry['generation_ids'])
        if entry['execution_partition'] is not None:
            phase['execution_partition'] = entry['execution_partition']
        if entry['test_execution'] is not None:
            phase['test_execution'] = entry['test_execution']
        if entry['finalist_recovery'] is not None:
            phase['finalist_recovery'] = entry['finalist_recovery']
        if entry.get('no_loophole_capability') is not None:
            phase['no_loophole_capability'] = entry['no_loophole_capability']
        for key in ('retirement_status', 'gross_reserved_counts', 'gross_reserved_wall_seconds',
                    'replacement_run_token', 'replaces_unlaunched_run_token'):
            if key in entry:
                phase[key] = entry[key]
        result['phases'].append(phase)

    if replacement_stage is not None:
        destination = str(Path(replacement_stage))
        require(Path(destination).is_absolute() and destination in pending_by_destination and
                destination not in by_stage and not (Path(destination) / 'reviewed_manifest.json').exists(),
                'Replacement builder must name one validated pending destination without a manifest')
        old, _, _ = pending_by_destination[destination]
        offsets = dict(old['counts'])
        require(offsets['tf_requests'] > 0 and offsets['generation_requests'] == offsets['untouched_test_requests'] == 0,
                'Replacement builder may transfer only validation TF reservations')
        effective = {key: result[key] - value for key, value in offsets.items()}
        require(all(value >= 0 for value in effective.values()), 'Negative replacement accounting offset')
        result['replacement_context'] = {
            'destination_stage': destination, 'old_run_token': old['manifest']['run_token'],
            'old_manifest_sha256': old['digest'], 'request_offsets': offsets,
            'wall_seconds_offset': old['seconds'], 'effective_previous_counts': effective,
            'effective_previous_gpu_phase_wall_seconds': result['gpu_phase_wall_seconds'] - old['seconds'],
        }
    result['bindings'] = sorted(set(result['bindings']))
    return result
