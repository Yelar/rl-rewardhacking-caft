"""Read-only, bounded metadata observation of an existing causal GPU phase.

This is a monitor, not a completion auditor. It never opens generated responses,
primitive/final payloads, model files or worker logs. Independently sampled event
counts and terminal markers do not prove retained bytes or released resources.
The local CLI sends this standalone source over SSH and keeps a compact receipt.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time

ROOT = Path('/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105')
RECEIPTS = Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z/host')
STAGES = ('started', 'primitive_returned', 'final_recorded')
MAX_REQUESTS = 2048
EVENT_BYTES = 4096


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def decode(raw):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_snapshot(path, limit, *, tail=False):
    """Bounded bytes from a regular non-symlink file; never follow FIFOs/devices."""
    path = Path(path)
    if path.resolve() != path:
        return None, {'status': 'unsafe_path'}
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            return None, {'status': 'not_regular'}
        if not tail and before.st_size > limit:
            return None, {'status': 'oversized', 'size_bytes': before.st_size, 'limit': limit}
        offset = max(0, before.st_size - limit) if tail else 0
        os.lseek(fd, offset, os.SEEK_SET)
        chunks, remaining = [], min(limit, before.st_size)
        while remaining:
            block = os.read(fd, min(remaining, 65536))
            if not block:
                break
            chunks.append(block); remaining -= len(block)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        stable = identity(before) == identity(after) and len(raw) == min(limit, before.st_size)
        return raw, {'status': 'observed' if stable else 'changing', 'size_bytes': before.st_size,
                     'read_bytes': len(raw), 'offset': offset, 'sha256_of_read_bytes': hashlib.sha256(raw).hexdigest()}
    except FileNotFoundError:
        return None, {'status': 'missing'}
    except OSError as exc:
        return None, {'status': 'read_error', 'error_type': type(exc).__name__}
    finally:
        if fd is not None:
            os.close(fd)


def required_json(path, digest, size=None, limit=8 << 20):
    raw, info = read_snapshot(path, limit)
    require(info['status'] == 'observed' and re.fullmatch('[0-9a-f]{64}', digest or '')
            and info['sha256_of_read_bytes'] == digest and (size is None or len(raw) == size),
            'Bound immutable input is missing, changing or different')
    value = decode(raw)
    require(isinstance(value, dict), 'Expected JSON object')
    return value


def scan_events(path, expected, *, run_token, task_sha256, stage):
    """Count valid metadata events, exposing incomplete/malformed live writes."""
    raw, info = read_snapshot(path, max(1, len(expected)) * EVENT_BYTES)
    observed = set()
    if raw is None:
        return {**info, 'valid_unique_events': None}, observed
    issues = {'partial_tail': int(bool(raw) and not raw.endswith(b'\n')),
              'malformed': 0, 'duplicates': 0, 'unplanned': 0, 'binding_mismatch': 0}
    lines = raw.splitlines(keepends=True)
    for line in lines:
        if not line.endswith(b'\n'):
            continue
        if len(line) > EVENT_BYTES:
            issues['malformed'] += 1; continue
        try:
            row = decode(line)
            require(isinstance(row, dict), 'Not an event')
            rid = row.get('request_id')
            require(isinstance(rid, str), 'No request ID')
        except (ValueError, TypeError, UnicodeError, RecursionError):
            issues['malformed'] += 1; continue
        if rid not in expected:
            issues['unplanned'] += 1; continue
        base = {'schema': 1, 'run_token': run_token, 'input_sha256': task_sha256,
                'request_id': rid, 'request_sha256': expected[rid], 'stage': stage}
        if stage != 'started':
            digest = row.get('file_sha256')
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                issues['binding_mismatch'] += 1; continue
            base['file_sha256'] = digest
        if row != base:
            issues['binding_mismatch'] += 1; continue
        if rid in observed:
            issues['duplicates'] += 1; continue
        observed.add(rid)
    return {**info, 'valid_unique_events': len(observed), **issues}, observed


def marker(path, fields):
    raw, info = read_snapshot(path, 65536)
    if raw is None:
        return info
    try:
        value = decode(raw)
        require(isinstance(value, dict), 'Not an object')
        # Never display arbitrary exception strings, environment or response text.
        selected = {k: value[k] for k in fields if k in value and
                    (value[k] is None or type(value[k]) in (bool, int, float)
                     or isinstance(value[k], str) and len(value[k]) <= 128)}
        return {**info, 'reported': selected}
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return {**info, 'parse_status': 'incomplete_or_malformed'}


def resource_tail(path, selected_gpus):
    raw, info = read_snapshot(path, 65536, tail=True)
    if raw is None:
        return info
    lines = raw.splitlines(keepends=True)
    if info['offset'] and lines:
        lines = lines[1:]  # First partial line at an arbitrary tail offset.
    for line in reversed(lines):
        if not line.endswith(b'\n'):
            continue
        try:
            row = decode(line)
            require(isinstance(row, dict) and isinstance(row.get('gpus'), dict), 'Not resource snapshot')
            devices = {}
            for index in selected_gpus:
                source = row['gpus'].get(index, {})
                require(isinstance(source, dict), 'Bad GPU snapshot')
                devices[index] = {k: source[k] for k in ('uuid', 'total_mib', 'free_mib', 'utilization', 'pids') if k in source}
            return {**info, 'snapshot': {k: row[k] for k in
                    ('at', 'owned_rss_bytes', 'mem_available_bytes', 'disk_free_bytes') if k in row},
                    'selected_gpus': devices, 'fresh_gpu_probe_performed': False}
        except (ValueError, TypeError, UnicodeError, RecursionError):
            continue
    return {**info, 'parse_status': 'no_complete_resource_snapshot_in_bounded_tail'}


def observe(manifest_path, digest, *, root=ROOT):
    root, manifest_path = Path(root), Path(manifest_path)
    require(manifest_path.is_absolute() and root in manifest_path.parents, 'Manifest outside this run')
    m = required_json(manifest_path, digest, limit=1 << 20)
    require(m['schema'] == 'gpu03_task_specific_resource_guard_v1' and m['host'] == 'gpu-03'
            and m['run_token'] == root.name and m['science']['stage'] in ('qualification', 'pilot')
            and 1 <= len(m['workers']) <= 8, 'Not an accepted causal resource manifest')
    runtime = Path(m['runtime'])
    require(runtime.parent == root.parent and runtime.name.startswith(root.name + '-'), 'Runtime outside this run')
    rows = []
    for item in m['workers']:
        argv = item['argv']
        require(len(argv) == 8 and argv[1:5] == ['-B', '-m',
                'infra.gpu03.task_specific_discovery.causal_worker', '--task'] and argv[6] == '--sha256',
                'Not a causal generation worker')
        task_path = Path(argv[5]); require(root in task_path.parents, 'Task outside this run')
        binding = m['bound_files'][str(task_path)]
        require(binding['sha256'] == argv[7], 'Task command hash differs')
        task = required_json(task_path, argv[7], binding['size_bytes'])
        require(task['schema'] == 'task_specific_causal_generation_v1' and task['run_token'] == root.name
                and task['stage'] == m['science']['stage'], 'Wrong task scope')
        requests = task['requests']
        require(0 < len(requests) <= MAX_REQUESTS, 'Too many task requests')
        expected = {r['request_id']: hashlib.sha256(canonical(r)).hexdigest() for r in requests}
        require(len(expected) == len(requests), 'Duplicate planned request')
        output = Path(task['output']); require(root in output.parents, 'Output outside this run')
        events, sets = {}, {}
        for stage in STAGES:
            events[stage], sets[stage] = scan_events(output / 'journal' / (stage + '.jsonl'), expected,
                run_token=root.name, task_sha256=argv[7], stage=stage)
        row = {'worker': item['name'], 'gpu_id': item['gpu_id'], 'task': str(task_path), 'task_sha256': argv[7],
               'output': str(output), 'planned_requests': len(requests), 'events': events,
               'STARTED': marker(output / 'STARTED.json', ('task_sha256', 'pid', 'requests', 'gpu_uuid', 'stage')),
               'COMPLETE': marker(output / 'COMPLETE.json', ('status', 'stage', 'task_sha256', 'requests',
                       'all_returned_requests_retained', 'evaluation_performed', 'causal_success_claimed')),
               'FAILED': marker(output / 'FAILED.json', ('status', 'stage', 'task_sha256'))}
        if all(events[s].get('valid_unique_events') is not None for s in STAGES):
            row['non_atomic_pending_estimates'] = {
                'without_start_event': len(set(expected) - sets['started']),
                'start_without_return_event': len(sets['started'] - sets['primitive_returned']),
                'return_without_final_event': len(sets['primitive_returned'] - sets['final_recorded'])}
        rows.append(row)
    totals = {stage: sum(r['events'][stage].get('valid_unique_events') or 0 for r in rows) for stage in STAGES}
    return {'schema': 'task_specific_causal_request_observation_v1', 'observed_at_epoch': time.time(),
            'host': socket.gethostname(), 'manifest': str(manifest_path), 'manifest_sha256': digest,
            'run_token': root.name, 'stage': m['science']['stage'], 'workers': rows,
            'known_event_counts': totals, 'planned_new_requests': sum(r['planned_requests'] for r in rows),
            'planned_reused_requests': m.get('expected_reused_requests'),
            'totals_are_partial_observations': True, 'observations_are_non_atomic': True,
            'terminal_success_verified': False, 'payloads_verified': False, 'resource_release_verified': False,
            'SUPERVISOR_RESULT': marker(runtime / 'SUPERVISOR_RESULT.json', ('status', 'run_token',
                'manifest_sha256', 'owned_processes_released', 'gpu_release_verified', 'scientific_success_verified')),
            'last_recorded_resources': resource_tail(runtime / 'resources.jsonl', m['gpu_uuids'])}


def compact_report(report, receipt):
    """Console-only summary; the full independently saved receipt is unchanged."""
    resources = report['last_recorded_resources']
    snapshot = resources.get('snapshot', {})
    devices = {}
    for gpu, value in resources.get('selected_gpus', {}).items():
        total, free = value.get('total_mib'), value.get('free_mib')
        devices[gpu] = {'used_mib': total - free if type(total) in (int, float)
                       and type(free) in (int, float) else None,
                       'utilization': value.get('utilization'), 'pids': value.get('pids')}
    workers = []
    for row in report['workers']:
        issues = {stage: {key: value for key, value in event.items() if key in
                  ('partial_tail', 'malformed', 'duplicates', 'unplanned', 'binding_mismatch') and value}
                  for stage, event in row['events'].items()}
        workers.append({'worker': row['worker'], 'gpu': row['gpu_id'], 'planned': row['planned_requests'],
            'counts': {stage: row['events'][stage].get('valid_unique_events') for stage in STAGES},
            'event_observations': {stage: row['events'][stage]['status'] for stage in STAGES},
            'event_issues': {stage: value for stage, value in issues.items() if value},
            'complete_marker': row['COMPLETE'].get('reported', {}).get('status'),
            'failure_marker': row['FAILED']['status']})
    return {'receipt': str(receipt), 'stage': report['stage'], 'planned_new_requests': report['planned_new_requests'],
            'planned_reused_requests': report['planned_reused_requests'], 'known_event_counts': report['known_event_counts'],
            'totals_are_partial_observations': True, 'terminal_success_verified': False,
            'workers': workers, 'supervisor_reported_status': report['SUPERVISOR_RESULT'].get('reported', {}).get('status'),
            'resources': {'observation_status': resources['status'], 'recorded_at_epoch': snapshot.get('at'),
                'available_ram_bytes': snapshot.get('mem_available_bytes'), 'owned_rss_bytes': snapshot.get('owned_rss_bytes'),
                'gpus': devices}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--compact', action='store_true', help='Print a compact summary; save the full receipt as usual')
    parser.add_argument('--on-host', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.on_host:
        require(socket.gethostname() == 'gpu-03' and os.getuid() == 3357, 'Expected gpu-03 owner')
        print(canonical(observe(args.manifest, args.sha256)).decode())
        return
    command = shlex.join(['python3', '-B', '-', '--on-host', '--manifest', args.manifest, '--sha256', args.sha256])
    source = Path(__file__).read_bytes()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-03', command],
        input=source, capture_output=True, timeout=45)
    require(result.returncode == 0, 'Remote read-only observation failed; no completion inferred')
    require(len(result.stdout) <= 256 << 10, 'Observer result exceeded compact receipt bound')
    report = decode(result.stdout)
    report['observer_source_sha256'] = hashlib.sha256(source).hexdigest()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    destination = RECEIPTS / ('OBSERVE_CAUSAL_REQUESTS_' + stamp + '.json')
    with destination.open('xb') as stream:
        stream.write(canonical(report) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    output = compact_report(report, destination.resolve()) if args.compact else {'receipt': str(destination.resolve()), **report}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
