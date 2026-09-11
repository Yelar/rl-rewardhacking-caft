"""Read-only qualification of an explicit CPU scoring recovery; no scoring calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'cpu_only_original_score_recovery_v1'


def require(ok, message):
    if not ok:
        raise ValueError('Scoring recovery: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def read_bound(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'missing/unsafe bound file')
    return path.read_bytes()


def qualify(manifest, digest, terminal, read=read_bound):
    """Normal terminals are unchanged. A recovery runtime requires the full lineage.

    ``read`` permits a portable local package to map the original absolute refs;
    SHA256 and byte count are checked here regardless of the reader.
    """
    recovery_runtime = Path(manifest['runtime']).name.endswith('-score-recovery1-runtime')
    if not recovery_runtime and 'recovery' not in terminal:
        return None
    require(recovery_runtime and Path(manifest['runtime']).name ==
            manifest['run_token'] + '-score-recovery1-runtime', 'unexpected recovery runtime')
    r = terminal.get('recovery')
    require(isinstance(r, dict) and r.get('schema') == SCHEMA, 'missing explicit recovery lineage')
    require(terminal.get('status') == 'succeeded' and terminal.get('selected_gpus_released') is True
            and terminal.get('manifest_sha256') == digest, 'recovery terminal failed or mismatched')

    def bound(ref):
        data = read(ref)
        require(len(data) == ref['size_bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
                'bound bytes/hash mismatch: ' + ref['path'])
        return data

    def obj(ref):
        return json.loads(bound(ref))

    original = obj(r['original_manifest'])
    failed = obj(r['original_terminal'])
    require(set(original) == set(manifest) and
            {k: v for k, v in original.items() if k != 'runtime'} ==
            {k: v for k, v in manifest.items() if k != 'runtime'} and
            original['runtime'] != manifest['runtime'], 'manifest changed beyond runtime')
    require(r['original_terminal']['path'] == str(Path(original['runtime']) / 'SUPERVISOR_EXIT.json'),
            'wrong original terminal path')
    require(failed.get('status') == 'failed' and failed.get('selected_gpus_released') is True and
            failed.get('manifest_sha256') == r['original_manifest']['sha256'] and
            failed.get('children') and all(c.get('reaped') is True for c in failed['children']),
            'original failure/release/reaping not positively preserved')
    operation = obj(r['operation'])
    require(operation.get('schema') == SCHEMA and operation['original_manifest'] == r['original_manifest'] and
            operation['original_terminal'] == r['original_terminal'] and
            [{k: x[k] for k in ('cell', 'raw', 'raw_seal')} for x in operation['raw_bindings']] == r['raw_bindings'] and
            operation['runtime'] == manifest['runtime'] and operation['source_files'] == original['source_files'],
            'operation lineage differs')
    require(operation['recovered_manifest']['sha256'] == digest and
            obj(operation['recovered_manifest']) == manifest, 'operation has another recovered manifest')
    require(operation['deadline_epoch'] == manifest['deadline_epoch'], 'operation changed absolute deadline')
    for scope in (r,):
        require(all(type(scope.get(k)) is int and scope[k] == 0 for k in
                    ('generation_calls', 'model_calls', 'gpu_allocations')), 'recovery generated or allocated a model')
    require(r.get('unchanged_native_score') is True, 'native scoring implementation not preserved')
    external = obj(r['external_exit'])
    require(external.get('unit') == operation['unit'] and
            external.get('operation_sha256') == r['operation']['sha256'] and
            external.get('recovered_manifest_sha256') == digest and
            external.get('remaining_cgroup_pids') == [], 'CPU service identity/release mismatch')
    service = external['service']
    require(all(service.get(k) == v for k, v in dict(MainPID='0', ExecMainCode='1', ExecMainStatus='0',
            Result='success', SubState='exited').items()), 'CPU service did not exit successfully')
    cpu = obj(r['cpu_result'])
    require(cpu.get('status') == 'succeeded' and cpu.get('operation_sha256') == r['operation']['sha256']
            and cpu.get('recovered_manifest_sha256') == digest and
            all(type(cpu.get(k)) is int and cpu[k] == 0 for k in
                ('generation_calls', 'model_calls', 'gpu_allocations')), 'CPU scorer result failed or generated')
    names = [f"{a['arm']}_{a['step']:03d}_{kind}" for a in manifest['adapters']
             for kind in ('fixed', 'randomized')]
    require(len(names) == 2 and [x['cell'] for x in r['raw_bindings']] == names and
            [x['cell'] for x in cpu['cells']] == names, 'recovered cell coverage/order differs')
    require(all(x.get('returncode') == 0 and x.get('reaped') is True and x.get('scored_ready') is True
                for x in cpu['cells']), 'scorer child failed, unreaped, or unsealed')
    science = hashlib.sha256(canonical({k: manifest[k] for k in
        ('datasets', 'adapters', 'sampling', 'revision', 'source_files', 'seed_policy')}).encode()).hexdigest()
    request_ids = []
    for item in r['raw_bindings']:
        folder = Path(manifest['output']) / 'cells' / item['cell']
        require(item['raw']['path'] == str(folder / 'raw.jsonl') and
                item['raw_seal']['path'] == str(folder / 'RAW_COMPLETE.json'), 'raw path changed')
        seal = obj(item['raw_seal'])
        require(seal['raw'] == item['raw'] and seal['count'] == 1190 and seal['identity'] ==
                dict(run_token=manifest['run_token'], cell=item['cell'], manifest_science_sha256=science),
                'original raw seal/scientific identity differs')
        data = bound(item['raw'])
        require(data.endswith(b'\n'), 'incomplete raw JSONL')
        rows = [json.loads(line) for line in data.splitlines()]
        require(len(rows) == 1190, 'raw count differs')
        request_ids.extend(row['request_id'] for row in rows)
    require(len(request_ids) == len(set(request_ids)) == 2380, 'raw request IDs missing/duplicated')
    return dict(schema=SCHEMA, original_manifest_sha256=r['original_manifest']['sha256'],
                original_terminal=r['original_terminal'], recovered_manifest_sha256=digest,
                operation=r['operation'], external_exit=r['external_exit'], cpu_result=r['cpu_result'],
                preserved_raw_requests=2380, generation_calls=0, model_calls=0, gpu_allocations=0,
                gpu_release='Inherited from the original failed run; no new GPU allocation or current-idleness claim.')
