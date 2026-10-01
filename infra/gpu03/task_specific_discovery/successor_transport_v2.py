"""Local-only successor transport with the fixed UID3357 user-systemd bus.

Reuse the immutable v1 manifest validator, command, remote bootstrap and remote
one-use dispatcher. The only remote environment change is the two exact owner
session-bus variables needed by systemctl --user. This does not modify the
service/worker environment, staged sources, claims, deadlines or guard logic.
A failed or uncertain attempt stays consumed, including v1 local markers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from . import successor_dispatch as original

SCHEMA = 'task_specific_successor_transport_v2'
SESSION_ENV = ('XDG_RUNTIME_DIR=/run/user/3357',
               'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/3357/bus')


def remote_argv():
    return ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', 'PYTHONNOUSERSITE=1',
            'PYTHONDONTWRITEBYTECODE=1', *SESSION_ENV,
            original.PYTHON, '-B', '-c', original.BOOTSTRAP]


def bounded(value, limit):
    if value is None:
        return ''
    return (value.decode('utf-8', errors='replace') if isinstance(value, bytes) else str(value))[-limit:]


def dispatch(local, root, digest, *, runner=subprocess.run, now=None):
    payload = original.prepare(local, root, digest, now=now)
    local = Path(local)
    result_path = local.with_suffix('.local_result.json')
    original.require(not result_path.exists() and not result_path.is_symlink(), 'Existing local result preserved')
    transport = {'schema': SCHEMA, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'bootstrap_sha256': hashlib.sha256(original.BOOTSTRAP.encode()).hexdigest(),
                 'session_environment': list(SESSION_ENV), 'remote_source_changed': False}
    original.exclusive_json(local.with_suffix('.local_attempt.json'), {
        'at': time.time(), 'manifest_sha256': digest, 'command': payload['command'],
        'root': str(original.ROOT), 'remote': payload['remote'], 'retry_allowed': False, 'transport': transport})
    try:
        result = runner(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', original.HOST,
                         shlex.join(remote_argv())], input=json.dumps(payload),
                        capture_output=True, text=True, timeout=100)
        out = {'status': 'transport_returned', 'returncode': result.returncode,
               'stdout': bounded(result.stdout, 50000), 'stderr': bounded(result.stderr, 10000),
               'retry_allowed': False, 'transport': transport}
    except Exception as exc:
        out = {'status': 'ambiguous_transport', 'exception_type': type(exc).__name__,
               'retry_allowed': False, 'transport': transport}
        if isinstance(exc, subprocess.TimeoutExpired):
            out.update(partial_stdout=bounded(exc.stdout, 50000), partial_stderr=bounded(exc.stderr, 10000))
    original.exclusive_json(result_path, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha256', required=True); args = parser.parse_args()
    result = dispatch(args.manifest, args.root, args.sha256)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get('returncode') == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
