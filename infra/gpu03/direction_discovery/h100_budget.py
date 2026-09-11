"""Append-only, locked Step1 admissions preserving the historical GPU budget.

This module admits only the H100 qualification and matched no-loophole phase.
The approved continuation reuses five completed cells and generates the other735.
It neither launches a process nor grants future-phase authority.
Failed/unused admissions retain their complete request and wall reservations.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

FIELDS = ('generation_requests', 'tf_requests', 'untouched_test_requests',
          'untouched_test_generation_requests', 'untouched_test_tf_requests',
          'gpu_phase_wall_seconds')
HISTORICAL = dict(zip(FIELDS, (2646, 9228, 0, 0, 0, 20456.688430309296)))
PHASES = {'h100_numerical_qualification': (6, 780),
          'h100_no_loophole_capability': (740, 7380)}
CAPS = {'generation_requests': 4096, 'tf_requests': 12000,
        'untouched_test_requests': 0, 'untouched_test_generation_requests': 0,
        'untouched_test_tf_requests': 0, 'gpu_phase_wall_seconds': 28800}
LEDGER_ROOT = Path('/home/ubuntu/h100-workspace/outputs/no-loophole-step1-20260908-115431/ledger')
HISTORICAL_SEED_SHA = '4d0bb538213a4b63cb245411732d0709f06210214591d9c02cc45d568ad111e2'
MAX_REFERENCE_BYTES, MAX_JOURNAL_BYTES = 32 << 20, 256 << 10

def require(ok, message):
    if not ok:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def snapshot(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and '..' not in p.parts, 'Invalid immutable input path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
                not before.st_mode & 0o222 and before.st_size <= MAX_REFERENCE_BYTES,
                'Input must be an owned immutable bounded regular file')
        data = stream.read(MAX_REFERENCE_BYTES + 1); after = os.fstat(stream.fileno())
        fields = ('st_dev', 'st_ino', 'st_uid', 'st_gid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(all(getattr(before, k) == getattr(after, k) for k in fields) and len(data) == before.st_size,
                'Input changed during snapshot')
    return data

def ref(path):
    data = snapshot(path)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}

def read(reference):
    require(isinstance(reference, dict) and {'path', 'sha256'} <= set(reference) <= {'path', 'sha256', 'size_bytes'}, 'Invalid reference')
    data = snapshot(reference['path'])
    actual = {'path': reference['path'], 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    require(all(actual[k] == v for k, v in reference.items()), 'Changed input hash')
    return json.loads(data)

def counts(value, *, wall_cap=28800):
    result = {k: value[k] for k in FIELDS}
    require(all(type(v) is int and v >= 0 for k, v in result.items() if k != 'gpu_phase_wall_seconds'),
            'Invalid request counts')
    wall = result['gpu_phase_wall_seconds']
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0, 'Invalid wall accounting')
    require(wall_cap in (28800, 43200), 'Unreviewed wall cap')
    require(all(result[k] <= (wall_cap if k == 'gpu_phase_wall_seconds' else CAPS[k]) for k in FIELDS), 'Cumulative budget exceeded')
    return result

def manifest_budget(m):
    """The approved continuation retains all old charges and adds only unfilled cells."""
    n, wall = PHASES[m['phase']]
    if m.get('continuation') is None:
        return n, wall, 28800
    require(m['phase'] == 'h100_no_loophole_capability', 'Continuation is only for the same main evaluation')
    authority = read(m['authorization'])
    require(authority.get('integrated_validation') is True and
            authority.get('generation_runtime_seconds') == 21600 and
            authority.get('cumulative_gpu_wall_cap_seconds') == 43200 and
            m.get('generation_runtime_seconds') == 21600,
            'Missing approved six-hour continuation limits')
    require(type(m['generation_requests']) is int and m['generation_requests'] == 735,
            'Continuation must preserve five cells and issue exactly 735')
    return 735, 21780, 43200

def historical(reference):
    require(reference.get('sha256') == HISTORICAL_SEED_SHA, 'Wrong authoritative historical seed bytes')
    value = read(reference)
    budget = value['budget']
    require(counts(budget) == HISTORICAL and isinstance(budget.get('phases'), list),
            'Historical ledger changed, reset, or incomplete')
    require(value['utc'].startswith('2026-09-08') and value['python'].startswith('3.12.3 '),
            'Wrong live historical audit')
    return budget

def entry_identity(entry):
    """Replay immutable manifest/plan identities, without repeating model loading."""
    from infra.gpu03.direction_discovery import h100_supervisor as supervisor
    m = read(entry['manifest'])
    require(entry['manifest_sha256'] == entry['manifest']['sha256'] and
            m['run_token'] == entry['run_token'] and m['phase'] == entry['phase'] and
            m['generation_requests'] == entry['generation_requests'] and m['tf_requests'] == 0 and
            m['limits']['reserved_wall_seconds'] == entry['reserved_wall_seconds'],
            'Journal manifest identity/count/limit differs')
    require(m['host_profile']['sha256'] == entry['host_profile_sha256'] and
            m['request_plan']['sha256'] == entry['request_plan_sha256'] and
            m['authorization'] == entry['authorization'], 'Journal source references differ')
    profile, authority = read(m['host_profile']), read(entry['authorization'])
    require(profile.get('instance_id') == supervisor.INSTANCE and
            authority.get('status') == 'explicit_user_authorization_recorded' and
            authority.get('scope') == 'no_loophole_step1_only' and authority.get('host') == 'codex-h100' and
            authority.get('instance_id') == supervisor.INSTANCE, 'Journal workstation/authority differs')
    _, requests, _ = supervisor.request_context(m)
    require(entry['generation_request_ids'] == [r['request_id'] for r in requests],
            'Journal request IDs differ from its immutable actual plan')

def account(seed, entries):
    old = historical(seed)
    current = counts(old)
    used = set(); phases = list(old['phases'])
    require(isinstance(entries, list) and len(entries) <= len(PHASES), 'Admission journal exceeds Step1 scope')
    for entry in entries:
        require(isinstance(entry, dict) and entry.get('status') == 'admitted_h100_no_loophole_phase' and
                entry.get('historical_seed') == seed and entry.get('phase') in PHASES,
                'Malformed admission journal')
        manifest = read(entry['manifest'])
        phase = entry['phase']; n, wall, wall_cap = manifest_budget(manifest)
        require(phase not in used and type(entry['generation_requests']) is int and entry['generation_requests'] == n and
                type(entry['tf_requests']) is int and entry['tf_requests'] == 0 and
                type(entry['reserved_wall_seconds']) is int and entry['reserved_wall_seconds'] == wall and
                counts(entry['budget_before'], wall_cap=wall_cap) == current,
                'Duplicate, reordered, or altered admission')
        if phase == 'h100_no_loophole_capability':
            require('h100_numerical_qualification' in used, 'Main phase preceded qualification')
            if manifest.get('continuation') is not None:
                original = read(manifest['continuation'])['manifest']
                prior = next(e['manifest'] for e in entries if e['phase'] == 'h100_numerical_qualification')
                require(all(original[k] == prior[k] for k in ('path', 'sha256')),
                        'Continuation differs from the canonical preceding qualification')
        after = dict(current); after['generation_requests'] += n; after['gpu_phase_wall_seconds'] += wall
        require(counts(entry['budget_after'], wall_cap=wall_cap) == counts(after, wall_cap=wall_cap), 'Admission arithmetic differs')
        require(len(entry['generation_request_ids']) == n and
                all(isinstance(x, str) and x for x in entry['generation_request_ids']) and
                len(set(entry['generation_request_ids'])) == n, 'Duplicate/missing request IDs')
        entry_identity(entry)
        require(not any(set(entry['generation_request_ids']).intersection(p.get('generation_request_ids', []))
                        for p in phases), 'Previously committed request ID')
        phases.append({'manifest_path': entry['manifest']['path'],
                       'manifest_sha256': entry['manifest']['sha256'],
                       'generation_request_ids': entry['generation_request_ids'],
                       'generation_requests': n, 'tf_requests': 0,
                       'wall_seconds': wall, 'wall_basis': 'full_reserved_no_credit'})
        current = after; used.add(phase)
    return {**current, 'phases': phases, 'h100_admitted_phases': sorted(used)}

def exclusive(path, value):
    with Path(path).open('x') as f:
        f.write(canonical(value) + '\n'); f.flush(); os.fsync(f.fileno())
    Path(path).chmod(0o400)

def private_open(path, *, create=True):
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_CREAT if create else 0), 0o600)
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_uid == os.getuid() and
                not st.st_mode & 0o077, 'Ledger file is not a private owned regular file')
        return os.fdopen(fd, 'a+')
    except BaseException:
        os.close(fd)
        raise

def ledger_root(root, *, create=False):
    root = Path(root)
    require(root == LEDGER_ROOT and root.is_absolute(), 'Only the canonical Step1 ledger is authoritative')
    for p in (root, *root.parents):
        require(not p.is_symlink(), 'Ledger ancestor is a symlink')
        if p.exists():
            require(not p.stat().st_mode & 0o022 or bool(p.stat().st_mode & stat.S_ISVTX), 'Writable ledger ancestor')
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The caller's umask must not make a newly created ledger group-writable.
        # Recheck the created path and every ancestor; never chmod an existing one.
        return ledger_root(root, create=False)
    require(root.is_dir() and root.stat().st_uid == os.getuid(), 'Missing or unowned canonical ledger')
    return root

def journal_entries(path, *, create=False):
    with private_open(path, create=create) as stream:
        stream.seek(0); text = stream.read(MAX_JOURNAL_BYTES + 1)
    require(len(text.encode()) <= MAX_JOURNAL_BYTES and (not text or text.endswith('\n')),
            'Oversized or interrupted admission journal')
    lines = text.splitlines()
    require(len(lines) <= len(PHASES) and all(lines), 'Malformed/oversized admission journal')
    entries = [json.loads(line) for line in lines]
    require(all(canonical(entry) == line for entry, line in zip(entries, lines)), 'Noncanonical admission journal')
    return entries

def validate_admission_reference(reference, *, manifest_sha256, run_token):
    """Read-only join to the durable canonical journal; never create a ledger."""
    root = ledger_root(LEDGER_ROOT)
    require(re.fullmatch(r'[a-z][a-z0-9-]{4,100}', run_token) is not None and
            reference.get('path') == str(root / (run_token + '.admission.json')), 'Admission is outside canonical ledger')
    with private_open(root / 'ledger.lock', create=False) as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        admission = read(reference)
        require(admission['manifest_sha256'] == manifest_sha256 and admission['run_token'] == run_token,
                'Admission identity differs')
        entries = journal_entries(root / 'admissions.jsonl')
        account(admission['historical_seed'], entries)
        require(sum(entry == admission for entry in entries) == 1, 'Admission is not exactly committed in canonical journal')
    return admission

def admit(*, root, seed, manifest, phase, run_token, host_profile, request_plan,
          generation_request_ids, authorization):
    """Called after manifest validation; all references rehashed under the lock."""
    require(phase in PHASES, 'Only Step1 phases are authorized')
    require(isinstance(run_token, str) and re.fullmatch(r'[a-z][a-z0-9-]{4,100}', run_token) is not None, 'Invalid run token')
    root = ledger_root(root, create=True)
    with private_open(root / 'ledger.lock') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = root / 'admissions.jsonl'
        entries = journal_entries(journal, create=True)
        before = account(seed, entries)
        require(phase not in before['h100_admitted_phases'], 'Phase already committed; resume its original requests')
        if phase == 'h100_no_loophole_capability':
            require('h100_numerical_qualification' in before['h100_admitted_phases'], 'Missing qualification reservation')
        from infra.gpu03.direction_discovery import h100_supervisor as supervisor
        m = supervisor.load_manifest(manifest['path'], manifest['sha256'])
        p, requests, auth = (read(x) for x in (host_profile, request_plan, authorization))
        require(m['host_profile'] == host_profile and m['request_plan'] == request_plan and
                m['authorization'] == authorization, 'Admission references differ from reviewed manifest')
        plan, actual_requests, _ = supervisor.request_context(m)
        supervisor.protocol().validate_against_ledger(plan, before)
        require(generation_request_ids == [r['request_id'] for r in actual_requests],
                'Admission request IDs differ from actual reviewed work')
        require(m['run_token'] == run_token and m['phase'] == phase, 'Manifest phase/token differs')
        require(auth.get('status') == 'explicit_user_authorization_recorded' and
                auth.get('scope') == 'no_loophole_step1_only' and
                auth.get('host') == 'codex-h100' and auth.get('instance_id') == 'i-00000000000000002',
                'Missing exact-purpose user authority')
        require(p.get('instance_id') == 'i-00000000000000002', 'Wrong workstation identity')
        n, wall, wall_cap = manifest_budget(m)
        require(m['generation_requests'] == n and m['tf_requests'] == 0 and
                m['limits']['reserved_wall_seconds'] == wall, 'Manifest limits differ')
        require(len(generation_request_ids) == n and len(set(generation_request_ids)) == n, 'Wrong request reservation')
        current = counts(before, wall_cap=wall_cap); after = dict(current)
        after['generation_requests'] += n; after['gpu_phase_wall_seconds'] += wall
        counts(after, wall_cap=wall_cap)
        item = {'status': 'admitted_h100_no_loophole_phase', 'phase': phase,
                'manifest': manifest, 'manifest_sha256': manifest['sha256'], 'run_token': run_token,
                'host_profile_sha256': host_profile['sha256'], 'request_plan_sha256': request_plan['sha256'],
                'generation_requests': n, 'tf_requests': 0, 'generation_request_ids': generation_request_ids,
                'reserved_wall_seconds': wall, 'budget_before': current, 'budget_after': after,
                'historical_seed': seed, 'authorization': authorization,
                'admitted_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        account(seed, entries + [item])
        # Journal is authoritative and durable before exporting admission. A crash
        # here cannot silently create a new reservation or replace an old token.
        with private_open(journal) as f:
            f.write(canonical(item) + '\n'); f.flush(); os.fsync(f.fileno())
        exclusive(root / (run_token + '.admission.json'), item)
        return item

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    args = parser.parse_args(); spec = json.loads(Path(args.spec).read_text())
    print(canonical(admit(**spec)))

if __name__ == '__main__':
    main()
