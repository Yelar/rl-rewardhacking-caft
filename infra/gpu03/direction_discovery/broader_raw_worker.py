"""Native raw-only capture for the exact broader500 fitting population.

The external H100 supervisor owns allocation, source/model binding, deadlines
and process/GPU release. This worker neither generates nor evaluates code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from . import fixed_cache as cache

PADDED_LENGTH = 2688
RAW_ROOT = Path('/opt/dlami/nvme/h100-workspace/broader-pca-20260908-203000')
OUTPUT_ROOT = Path('/home/ubuntu/h100-workspace/outputs/codex-h100-broader-raw-20260908-203000')
RECOVERY_RAW_ROOT = Path('/opt/dlami/nvme/h100-workspace/broader-pca-recovery-20260908-205100')
RECOVERY_OUTPUT_ROOT = Path('/home/ubuntu/h100-workspace/outputs/codex-h100-broader-raw-recovery-20260908-205100')
PARENT_MANIFEST_SHA = 'c3fec8b94cc061e3861d09b3aab2efe25f18efe143f0f7b88dffc786f2009869'
MAX_METADATA = 128 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_bound(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_METADATA,
            'missing/unsafe/oversized metadata')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'metadata bytes changed')
    return data


def task_roots(task):
    if task['run_token'] == OUTPUT_ROOT.name:
        return OUTPUT_ROOT, RAW_ROOT
    require(task['run_token'] == RECOVERY_OUTPUT_ROOT.name, 'broader worker identity differs')
    return RECOVERY_OUTPUT_ROOT, RECOVERY_RAW_ROOT


def validate_reuse(task, rows):
    reuse = task.get('reuse_native_records', {'h0': {}, 'h60': {}})
    require(isinstance(reuse, dict) and set(reuse) == {'h0', 'h60'}, 'invalid native reuse map')
    by_id = {r['record_id']: r for r in rows}
    for kind, refs in reuse.items():
        require(isinstance(refs, dict) and set(refs) <= set(by_id), 'reuse records outside selected shard')
        if refs:
            require(task['run_token'] == RECOVERY_OUTPUT_ROOT.name, 'reuse requires exact recovery roots')
        for rid, ref in refs.items():
            required = {'path', 'sha256', 'size_bytes', 'source_manifest_sha256'}
            require(isinstance(ref, dict) and set(ref) in
                    (required, required | {'record_id', 'record_index', 'kind'}), 'invalid native reuse reference')
            if 'record_id' in ref:
                require(ref['record_id'] == rid and type(ref['record_index']) is int and
                        ref['record_index'] == by_id[rid]['record_index'] and ref['kind'] == kind,
                        'reuse record identity differs')
            p = Path(ref['path'])
            require(p.is_absolute() and '..' not in p.parts and p.parent.parent.parent == RAW_ROOT and
                    re.fullmatch(r'worker_0[0-7]', p.parent.parent.name) and p.parent.name == kind and
                    p.name == f"record_{by_id[rid]['record_index']:06d}.safetensors", 'reuse native path differs')
            require(isinstance(ref['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) and
                    type(ref['size_bytes']) is int and ref['size_bytes'] > 0 and
                    ref['source_manifest_sha256'] == PARENT_MANIFEST_SHA, 'reuse source binding differs')
    return reuse


def validate_task(task, rows):
    output_root, raw_root = task_roots(task)
    require(task['mode'] == 'broader_raw' and
            re.fullmatch(r'worker_0[0-7]', task['worker_name']) and type(task['gpu_id']) is int and
            0 <= task['gpu_id'] < 8, 'broader worker identity differs')
    require(task['output'] == str(output_root / task['worker_name']) and
            task['raw_output'] == str(raw_root / task['worker_name']), 'worker output path differs')
    require(task['padded_sequence_length'] == PADDED_LENGTH and task['pad_token_id'] == cache.PAD_ID and
            task.get('gpu_memory_fraction', .65) == .65 and
            type(task['deadline_seconds']) in (int, float) and 0 < task['deadline_seconds'] <= 43200,
            'broader numerical/resource profile differs')
    require(len(rows) == 500 and len({r['record_id'] for r in rows}) == 500 and
            len({str(r['problem_id_key']) for r in rows}) == 500 and
            [r['record_index'] for r in rows] == list(range(500)) and
            {r['problem_split'] for r in rows} == {'direction_fit'}, 'need exact500 unique fitting problems/records')
    for row in rows:
        cache.validate_row(row, padded_length=PADDED_LENGTH)
    requested = task['record_ids']
    by_id = {r['record_id']: r for r in rows}
    require(isinstance(requested, list) and requested and len(set(requested)) == len(requested) and
            set(requested) <= set(by_id), 'invalid/duplicate shard records')
    selected = [by_id[rid] for rid in requested]
    require(selected[0]['completion_token_count'] >= 2, 'first real row cannot support future-prefix qualification')
    validate_reuse(task, selected)
    return selected


def load_reused_native(task, row, kind):
    """Return exact CPU values from one immutable, parent-bound completed native file."""
    from .h100_broader_raw_run import inspect_native
    ref = task['reuse_native_records'][kind][row['record_id']]
    single = {'h0': {}, 'h60': {}}; single[kind][row['record_id']] = ref
    validate_reuse({**task, 'reuse_native_records': single}, [row])
    p = Path(ref['path'])
    require(p.resolve() == p, 'reuse native has symlink ancestor')
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
            before.st_mode & 0o222 == 0 and before.st_size == ref['size_bytes'], 'unsafe/writable reuse native')
    require(cache.sha256(p) == ref['sha256'], 'reuse native bytes changed')
    values = inspect_native(p, row, kind, ref['source_manifest_sha256'], return_original=True)
    after = p.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode), 'reuse native changed during read')
    inputs = {k: v[0] for k, v in cache.padded_inputs(row, 'cpu', padded_length=PADDED_LENGTH).items()}
    return values, inputs, {'math': True, 'flash': False, 'memory_efficient': False, 'cudnn': False}


def fresh_directory(path):
    path = Path(path)
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'output has symlink ancestor')
    require(path.parent.is_dir() and path.parent.stat().st_uid == os.getuid(), 'output parent not owned/prepared')
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def run(task, rows, digest):
    """Called after pure metadata guards; each model is loaded only once."""
    import torch
    raw, legacy = cache.dependencies()
    _, raw_root = task_roots(task)
    reuse = validate_reuse(task, rows)
    output, destination = fresh_directory(task['output']), fresh_directory(task['raw_output'])
    cache.write_json(output / 'task.json', task)
    for kind in ('h0', 'h60', 'qualification'):
        (destination / kind).mkdir(mode=0o700)
    started = time.monotonic()
    entries = {r['record_id']: {'record_id': r['record_id'], 'record_index': r['record_index'], 'models': {}}
               for r in rows}
    reports, releases, qualifications = {}, {}, {}
    def within_deadline():
        require(time.monotonic() - started < task['deadline_seconds'], 'broader raw deadline exceeded')
    try:
        raw.configure_torch(gpu=True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        cache.runtime_policy()
        for kind in ('h0', 'h60'):
            within_deadline()
            model = decoder = layers = None
            try:
                before = time.monotonic()
                model, decoder, layers, reports[kind] = legacy._load_decoder(task, with_adapter=kind == 'h60')
                cache.append_json(output / 'model_events.jsonl',
                                  {'kind': kind, 'event': 'loaded', 'elapsed_seconds': time.monotonic() - before,
                                   'report': reports[kind]})
                for index, row in enumerate(rows):
                    within_deadline()
                    reused = reuse[kind].get(row['record_id'])
                    reuse_read_seconds = None
                    if reused is not None:
                        before = time.monotonic()
                        values, model_inputs, flags = load_reused_native(task, row, kind)
                        reuse_read_seconds = time.monotonic() - before
                        capture_seconds, peak_allocated, peak_reserved = 0.0, 0, 0
                    else:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = time.monotonic()
                        values, model_inputs, flags = cache.capture_fixed(decoder, layers, row,
                                                                         hidden_size=cache.HIDDEN, padded_length=PADDED_LENGTH)
                        torch.cuda.synchronize()
                        capture_seconds = time.monotonic() - before
                        peak_allocated = int(torch.cuda.max_memory_allocated())
                        peak_reserved = int(torch.cuda.max_memory_reserved())
                    before = time.monotonic()
                    path = destination / kind / f"record_{row['record_index']:06d}.safetensors"
                    info = cache.save_native(path, row, kind, digest, values, model_inputs, padded_length=PADDED_LENGTH)
                    path.chmod(0o400)
                    info.update(tensor_path=str(path.relative_to(raw_root)),
                                path=str(path),
                                capture_seconds=capture_seconds,
                                save_hash_readback_seconds=time.monotonic() - before,
                                cuda_peak_allocated_bytes=peak_allocated, cuda_peak_reserved_bytes=peak_reserved)
                    if reused is not None:
                        info.update(reused_native=True, reuse_origin=dict(reused),
                                    reused_values_bitwise_preserved=True, new_manifest_sha256=digest,
                                    reuse_cpu_validation_seconds=reuse_read_seconds,
                                    cuda_peak_scope='no native GPU capture; restarted first-record qualification is separate')
                    entries[row['record_id']]['models'][kind] = info
                    if index == 0:
                        within_deadline()
                        before = time.monotonic()
                        qualifications[kind] = cache.qualify_model(
                            decoder, layers, row, values, kind, digest, destination / 'qualification' / kind,
                            vocab_size=int(model.config.vocab_size), hidden_size=cache.HIDDEN, padded_length=PADDED_LENGTH)
                        qualifications[kind].update(record_id=row['record_id'],
                                                    elapsed_seconds=time.monotonic() - before,
                                                    first_production_raw_retained=True)
                        cache.write_json(output / (kind + '_qualification.json'), qualifications[kind])
                    # Closed unique raw files may be copied only after this receipt.
                    cache.append_json(output / 'native_index.jsonl',
                                      {'record_id': row['record_id'], 'record_index': row['record_index'], 'kind': kind, **info,
                                       'attention_flags': flags, 'input_ids_sha256': row['input_ids_sha256'],
                                       'padded_sequence_length': PADDED_LENGTH})
                    del values, model_inputs
                    within_deadline()
            finally:
                del model, decoder, layers
                releases[kind] = legacy._release_cuda()
                legacy.validate_post_model_cuda_state(releases[kind], kind + ' broader raw release')
                cache.append_json(output / 'model_events.jsonl', {'kind': kind, 'event': 'released', 'state': releases[kind]})
        legacy.validate_model_load_reports(reports['h0'], reports['h60'], 'broader raw worker')
        for row in rows:
            entry = entries[row['record_id']]
            require(set(entry['models']) == {'h0', 'h60'}, 'missing native model record')
            entry['verified'] = True
            entry['verification_scope'] = 'producer native save/readback; separate completed-package CPU audit required'
            cache.append_json(output / 'workerindex.jsonl', entry)
        result = {'status': 'succeeded', 'mode': 'broader_raw', 'run_token': task['run_token'],
                  'worker_name': task['worker_name'], 'manifest_sha256': digest, 'records': len(rows),
                  'record_ids': [r['record_id'] for r in rows], 'native_files': 2 * len(rows),
                  'model_load_reports': reports, 'cuda_release': releases, 'qualifications': qualifications,
                  'elapsed_seconds': time.monotonic() - started, 'padded_sequence_length': PADDED_LENGTH,
                  'raw_activations_retained': True, 'differences_computed': False,
                  'raw_root': str(raw_root), 'all_native_readbacks_bitwise_equal': True,
                  'independent_cpu_audit_required': True, 'external_process_and_gpu_release_required': True,
                  'numerical_runtime_policy': cache.runtime_policy()}
        if any(reuse.values()):
            result.update(reused_native_files={kind: len(refs) for kind, refs in reuse.items()},
                          fresh_native_captures={kind: len(rows) - len(refs) for kind, refs in reuse.items()},
                          reuse_parent_manifest_sha256=PARENT_MANIFEST_SHA)
        cache.write_json(output / 'SUCCESS.json', result)
    except BaseException as exc:
        cache.write_json(output / 'FAILURE.json', {'status': 'failed', 'run_token': task['run_token'],
                         'worker_name': task['worker_name'], 'manifest_sha256': digest,
                         'elapsed_seconds': time.monotonic() - started, 'error': type(exc).__name__ + ': ' + str(exc),
                         'model_load_reports': reports, 'cuda_release': releases, 'raw_files_preserved': True})
        raise
    return result


def worker(task_path, task_sha256):
    task = json.loads(load_bound(task_path, task_sha256))
    data = load_bound(task['prepared_records'], task['prepared_records_sha256'])
    rows = [json.loads(line) for line in data.splitlines()]
    selected = validate_task(task, rows)
    digest = cache.resolve_manifest_digest(task)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(task['gpu_id']), 'worker GPU differs from task')
    return run(task, selected, digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--task-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(worker(args.task, args.task_sha256), sort_keys=True))
