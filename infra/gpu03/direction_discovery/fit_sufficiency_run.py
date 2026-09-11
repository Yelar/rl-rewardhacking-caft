#!/usr/bin/env python3
"""Bounded direct CPU supervisor for a single manifest-bound sufficiency run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    from . import candidates as c
except ImportError:
    import candidates as c


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('missing available memory')


def run(manifest_path, digest):
    c.require(c.sha256_file(manifest_path) == digest, 'manifest digest mismatch')
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest['root'])
    c.require(os.getuid() == manifest['owner_uid'], 'wrong owner')
    c.require(subprocess.check_output(['hostname'], text=True).strip() == manifest['hostname'], 'wrong host')
    c.require(str(Path(sys.executable).absolute()) == manifest['python'], 'wrong runtime')
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU access must be disabled')
    for relative, bound in manifest['inputs'].items():
        c.verify_file(c.safe_child(root, relative), bound)
    results = root / 'results'
    results.mkdir(exist_ok=False)
    c.require(memory_available() >= manifest['limits']['minimum_available_ram_bytes'], 'insufficient available RAM')
    c.require(os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= manifest['limits']['minimum_free_disk_bytes'], 'insufficient disk')
    journal = (root / 'control/study_journal.jsonl').open('x', buffering=1)
    active, finished, queue = {}, [], list(manifest['layers'])
    start = time.monotonic()
    status, error = 'failed', None
    def log(value):
        journal.write(c.canonical({'elapsed_seconds': time.monotonic() - start, **value}) + '\n')
        journal.flush(); os.fsync(journal.fileno())
    try:
        while queue or active:
            c.require(time.monotonic() - start < manifest['limits']['runtime_seconds'], 'study deadline exceeded')
            c.require(memory_available() >= manifest['limits']['minimum_available_ram_bytes'], 'host RAM fell below safety minimum')
            c.require(os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= manifest['limits']['minimum_free_disk_bytes'], 'scratch free space fell below safety minimum')
            while queue and len(active) < manifest['limits']['workers']:
                layer = queue.pop(0)
                command = [manifest['python'], str(root / 'source/fit_sufficiency.py'), '--root', str(root),
                           '--manifest', str(manifest_path), '--manifest-sha256', digest, '--layer', str(layer)]
                output = (root / 'control' / f'layer_{layer:02d}.log').open('xb')
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=root / 'source')
                active[layer] = (process, output)
                log({'event': 'started', 'layer': layer, 'pid': process.pid, 'command': command})
            for layer, (process, output) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                output.close(); del active[layer]
                log({'event': 'exited', 'layer': layer, 'pid': process.pid, 'exit_code': code})
                c.require(code == 0, f'layer {layer} exited {code}')
                done = json.loads((results / f'layer_{layer:02d}/complete.json').read_text())
                c.require(done['status'] == 'complete' and done['cells'] == 16 and done['manifest_sha256'] == digest,
                          'incomplete/mismatched layer completion')
                finished.append(layer)
            time.sleep(2)
        c.require(sorted(finished) == manifest['layers'], 'missing layer')
        status = 'complete'
    except BaseException as exc:
        error = str(exc)
        log({'event': 'failure', 'error': error})
        raise
    finally:
        for layer, (process, output) in active.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for layer, (process, output) in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
            output.close()
            log({'event': 'owned_worker_reaped', 'layer': layer, 'pid': process.pid, 'exit_code': process.returncode})
        c.write_json(root / 'control/study_completion.json', {
            'status': status, 'error': error, 'finished_layers': sorted(finished), 'manifest_sha256': digest,
            'seconds': time.monotonic() - start, 'owned_workers_reaped': True})
        journal.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    run(args.manifest, args.manifest_sha256)
