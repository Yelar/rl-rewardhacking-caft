#!/usr/bin/env python3
"""Prepared Random0/Random1 follow-up of the original full RH evaluation.

Generation/scoring are the original repository implementations. This file adds
cell journals, detached process supervision, and the existing two-layer sandbox.
It never generates a cell with an existing generation intent but no closed raw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from score_recovery import qualify as qualify_score_recovery
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
MODEL = 'Qwen/Qwen3-4B'
KINDS = ('fixed', 'randomized')
ARMS = ('random0', 'random1')
ARM_STEPS = {'random0': tuple(range(110, 201, 10)), 'random1': tuple(range(0, 101, 10))}
CHECKPOINTS = tuple((arm, step) for arm in ARMS for step in ARM_STEPS[arm])
PROJECTION_SCHEMA = 'random0_checkpoint100_projection_restoration_v1'
PROJECTION_LONG_SCHEMA = 'random0_checkpoint100_projection_restoration_long3072_v1'
PROJECTION_CHECKPOINTS = (('random0_off', 100), ('random0_on', 100))
CELL_COUNT = len(CHECKPOINTS) * len(KINDS)
SAMPLE_COUNT = CELL_COUNT * 1190
SAMPLING = dict(n=10, temperature=0.7, top_p=0.95, max_new_tokens=1536,
                repetition_penalty=1.0)
VERSIONS = {'torch': '2.8.0+cu128', 'vllm': '0.11.0', 'transformers': '4.57.1', 'peft': '0.17.1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Expected regular file: ' + str(p))
    before = p.stat()
    digest = sha(p)
    after = p.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed while hashing')
    return dict(path=str(p), sha256=digest, size_bytes=after.st_size)


def check_ref(value, *, content=True):
    p = Path(value['path'])
    require(p.is_absolute() and not p.is_symlink() and p.is_file(), 'Unsafe/missing bound file: ' + str(p))
    require(p.stat().st_size == value['size_bytes'], 'Bound size mismatch: ' + str(p))
    require(re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None, 'Invalid SHA256')
    if content:
        require(sha(p) == value['sha256'], 'Bound bytes mismatch: ' + str(p))
    return p


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as out:
        out.write(canonical(value) + '\n')
        out.flush()
        os.fsync(out.fileno())
    descriptor = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rows(path):
    data = Path(path).read_bytes()
    require(data.endswith(b'\n'), 'Incomplete JSONL boundary: ' + str(path))
    return [json.loads(line) for line in data.splitlines()]


def cell_name(arm, step, kind):
    require(type(step) is int and kind in KINDS and
            ((arm in ARMS and step in ARM_STEPS[arm]) or (arm, step) in PROJECTION_CHECKPOINTS), 'Invalid cell')
    return f'{arm}_{step:03d}_{kind}'


def cells(m):
    return [(cell_name(a['arm'], a['step'], kind), a, kind)
            for a in m['adapters'] for kind in KINDS]


def execution_cells(m):
    declared = cells(m)
    recovery = m.get('long3072_recovery')
    if recovery is None:
        return declared
    require(m['schema'] == PROJECTION_LONG_SCHEMA, 'Recovery scope requires the long3072 diagnostic')
    names = [name for name, _, _ in declared]
    kind = recovery['kind']
    require(kind in ('off_reuse_on_fixed_v1', 'held_on_randomized_completion_v1'), 'Unreviewed recovery kind')
    expected = names[:3] if kind == 'off_reuse_on_fixed_v1' else names
    require(recovery['operation_scope'] == expected and recovery['excluded_partial_on_randomized_responses'] == 40,
            'Recovery must preserve the exact pending ON-randomized boundary')
    require(isinstance(recovery.get('decision'), dict), 'Missing root recovery decision')
    imported = m.get('imported_off', {})
    require(imported.get('schema') == 'exact_long3072_off_raw_import_v1'
            and all(isinstance(imported.get(k), dict) for k in ('plan', 'validator')),
            'Exact imported OFF plan and validator required')
    return [item for item in declared if item[0] in expected]


def verify_imported_off(m, *, after_launch=False):
    if 'long3072_recovery' not in m:
        return
    import importlib.util
    imported = m['imported_off']
    path = check_ref(imported['validator']);check_ref(imported['plan'])
    spec = importlib.util.spec_from_file_location('exact_imported_off_validator', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_manifest(m, after_launch=after_launch)


def recovery_execution_authorized(m):
    recovery = m.get('long3072_recovery')
    if recovery is not None:
        execution_cells(m)
        check_ref(recovery['decision'])
        if recovery['kind'] == 'held_on_randomized_completion_v1':
            require(isinstance(recovery.get('owner_exception'), dict), 'Held ON-randomized replacement requires owner exception')
            check_ref(recovery['owner_exception'])


def validate(m):
    require(m['schema'] in ('original_rh_random_followup_subset_v2', PROJECTION_SCHEMA, PROJECTION_LONG_SCHEMA), 'Wrong evaluation schema')
    long3072 = m['schema'] == PROJECTION_LONG_SCHEMA
    recovery = 'long3072_recovery' in m
    require(not recovery or long3072, 'Recovery scope requires long3072 schema')
    diagnostic = m['schema'] in (PROJECTION_SCHEMA, PROJECTION_LONG_SCHEMA)
    require(diagnostic == ('projection_restoration' in m), 'Projection declaration/schema mismatch')
    require(m['model_id'] == MODEL and m['revision'] == REVISION, 'Wrong pinned model')
    require(m['sampling'] == ({**SAMPLING, 'max_new_tokens': 3072} if long3072 else SAMPLING) and m['thinking'] is False, 'Original sampling changed')
    require(m['seed_policy'] == 'fresh_engine_seed1_per_checkpoint_setting', 'Wrong seed policy')
    require(m['host'] == 'gpu-02' and m['owner'] == 'researcher', 'Wrong host/owner')
    require(m['gpu_ids'] in ([2, 3], [3, 4], [3], [4, 5]), 'Only reviewed GPU2–3, GPU3–4, GPU4–5 or single GPU3 allowed')
    require(set(m['gpu_uuids']) == {str(i) for i in m['gpu_ids']}, 'Missing GPU identities')
    require(len(set(m['gpu_uuids'].values())) == len(m['gpu_ids']), 'Duplicate GPU UUID')
    require(m['versions'] == VERSIONS, 'Runtime versions changed')
    require(set(m['datasets']) == set(KINDS), 'Both original settings required')
    selected = [(x['arm'], x['step']) for x in m['adapters']]
    allowed = PROJECTION_CHECKPOINTS if diagnostic else CHECKPOINTS
    require(selected and len(selected) == len(set(selected)) and set(selected) <= set(allowed),
            'Require a nonempty unique subset of the declared follow-up checkpoints')
    require(selected == [key for key in allowed if key in set(selected)],
            'Checkpoint subset order must be fixed')
    require(isinstance(m.get('previous_evaluations'), list), 'Explicit previous evaluation list required')
    if diagnostic:
        from projection_diagnostic import validate_declaration
        validate_declaration(m)
    require(type(m.get('deadline_epoch')) in (int, float) and 0 < m['deadline_epoch'] <= 1789149052,
            'Deadline exceeds approved 2026-09-11 17:50:52 UTC boundary')
    require(re.fullmatch('codex-[a-z0-9-]{8,100}', m['run_token']) is not None, 'Unsafe run token')
    for key in ('stage', 'output', 'runtime'):
        p = Path(m[key])
        require(p.is_absolute() and not p.is_symlink() and p.name.startswith(m['run_token'] + '-'), 'Unsafe ' + key)
        require(p.parent == Path('/scratch/researcher/codex_runs'), 'Run root changed')
    require(len({m[k] for k in ('stage', 'output', 'runtime')}) == 3, 'Run roots overlap')
    require(m['source_root'] == str(Path(m['stage']) / 'source'), 'Source must be staged separately')
    require(Path(m['python']).is_absolute() and str(Path(m['python']).parent.parent) == m['venv'], 'Invalid runtime')
    require(m['limits'] == dict(runtime_seconds=10800 if recovery else (14400 if long3072 else 21600),
            systemd_seconds=11400 if recovery else (15000 if long3072 else 22200),
            min_ram_gib=128, max_rss_gib=96, cgroup_memory_gib=128, disk_floor_gib=64,
            log_mib=128, tasks_max=1024, stagger_seconds=10, evaluator_workers=4,
            evaluator_wall_seconds=3600, gpu_memory_utilization=0.7,
            max_num_seqs=64 if diagnostic else 'original_vllm_default'), 'Unreviewed profile')
    require(m['supervisor_cpus'] == list(range(8, 12)), 'Supervisor CPU profile changed')
    require(m['worker_cpus'] == {str(gpu): list(range(16 + 8 * i, 24 + 8 * i)) for i, gpu in enumerate(m['gpu_ids'])}, 'Worker CPU profile changed')
    require(m['authorization'] and m['source_files'] and m['base_files'], 'Missing review bindings')
    execution_cells(m)
    return m


def load(path, digest):
    require(sha(path) == digest, 'Manifest SHA mismatch')
    return validate(json.loads(Path(path).read_text()))


def validate_datasets(m):
    found = {}
    for kind, item in m['datasets'].items():
        data = rows(check_ref(item))
        require(len(data) == 119, 'Original held-out set must have 119 problems')
        ids = [canonical(x['id']) for x in data]
        require(len(set(ids)) == 119, 'Repeated problem in benchmark')
        names = [x['prompt_metadata']['test_func_name'] for x in data]
        require(all(isinstance(x, str) and x.isidentifier() for x in names), 'Missing evaluator name')
        require((set(names) == {'run_tests'}) if kind == 'fixed' else len(set(names)) > 1, 'Wrong setting names')
        found[kind] = ids
    require(found['fixed'] == found['randomized'], 'Settings differ in problem/order')


def verify_inputs(m):
    source = Path(m['source_root'])
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    require(actual == set(m['source_files']), 'Source membership changed')
    for name, item in m['source_files'].items():
        require(str(source / name) == item['path'], 'Source path mismatch')
        check_ref(item)
    for item in m['base_files']:
        check_ref(item)
        relative = Path(item['snapshot_relative'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe base member')
        require((Path(m['base_model']) / relative).resolve() == Path(item['path']), 'Base snapshot/blob join changed')
    for a in m['adapters']:
        require(set(a['files']) == {'adapter_config.json', 'adapter_model.safetensors'}, 'Wrong adapter members')
        for name, item in a['files'].items():
            require(str(Path(a['path']) / name) == item['path'], 'Adapter path mismatch')
            check_ref(item)
        config = json.loads(Path(a['files']['adapter_config.json']['path']).read_text())
        require(config['base_model_name_or_path'] == MODEL and config['r'] == 32 and config['lora_alpha'] == 32,
                'Adapter configuration differs from training')
    validate_datasets(m)
    if 'projection_restoration' in m:
        from projection_diagnostic import verify_declaration_inputs
        verify_declaration_inputs(m, check_ref)
    if 'long3072_recovery' in m:
        for item in [m['long3072_recovery']['decision'],m['imported_off']['plan'],m['imported_off']['validator']]:
            check_ref(item)



def verify_previous_evaluations(m):
    """Reject repeated cells using the declared closed prior evaluation records."""
    current = {name for name, _, _ in cells(m)}
    seen = set()
    for previous in m['previous_evaluations']:
        require(set(previous) == {'manifest', 'verification'}, 'Exact prior manifest/proof pair required')
        old_path = check_ref(previous['manifest'])
        old = validate(json.loads(old_path.read_text()))
        proof = json.loads(check_ref(previous['verification']).read_text())
        expected = cells(old)
        require(proof['status'] == 'succeeded' and proof['manifest_sha256'] == previous['manifest']['sha256']
                and proof['cells'] == len(expected) and proof['samples'] == len(expected)*1190,
                'Prior evaluation is not independently complete')
        summary = [(x['arm'], x['step'], x['setting'], x['count']) for x in proof['summary']]
        require(summary == [(a['arm'], a['step'], k, 1190) for _, a, k in expected],
                'Prior proof checkpoint/settings coverage differs')
        require(all(all(old['datasets'][k][field] == m['datasets'][k][field]
                for field in ('sha256', 'size_bytes')) for k in KINDS), 'Prior benchmark dataset differs')
        terminal = json.loads(check_ref(proof['terminal']).read_text())
        require(terminal['status'] == 'succeeded' and terminal['selected_gpus_released'] is True
                and terminal['manifest_sha256'] == previous['manifest']['sha256'], 'Prior run has no positive release')
        recovery = qualify_score_recovery(old, previous['manifest']['sha256'], terminal)
        if recovery is not None:
            require(all(scored_ready(old, name, Path(old['output']) / 'cells' / name)
                        for name, _, _ in expected), 'Recovered prior cells are not completely scored')
        names = {name for name, _, _ in expected}
        require(not names & (seen | current), 'Duplicate completed checkpoint/settings cells')
        seen.update(names)
    return seen


def build(spec, output):
    m = json.loads(Path(spec).read_text())
    long3072 = m.get('projection_restoration', {}).get('kind') == PROJECTION_LONG_SCHEMA
    m.update(schema=(PROJECTION_LONG_SCHEMA if long3072 else PROJECTION_SCHEMA) if 'projection_restoration' in m else 'original_rh_random_followup_subset_v2', model_id=MODEL, revision=REVISION,
             sampling={**SAMPLING, 'max_new_tokens': 3072} if long3072 else SAMPLING, thinking=False, versions=VERSIONS,
             seed_policy='fresh_engine_seed1_per_checkpoint_setting')
    source = Path(m['source_root'])
    m['source_files'] = {str(p.relative_to(source)): ref(p) for p in sorted(source.rglob('*')) if p.is_file()}
    validate(m)
    verify_inputs(m)
    verify_previous_evaluations(m)
    write(output, m)
    print(canonical(ref(output)), flush=True)


def identity(m, name):
    return dict(run_token=m['run_token'], cell=name, manifest_science_sha256=hashlib.sha256(canonical({
        'datasets': m['datasets'], 'adapters': m['adapters'], 'sampling': m['sampling'],
        'revision': m['revision'], 'source_files': m['source_files'], 'seed_policy': m['seed_policy'],
        **({'projection_restoration': m['projection_restoration']} if 'projection_restoration' in m else {}),
    }).encode()).hexdigest())


def raw_ready(m, name, directory):
    p = Path(directory)
    seal = p / 'RAW_COMPLETE.json'
    if not seal.exists():
        require(not (p / 'GENERATION_INTENT.json').exists(), 'Unsealed generation; no automatic regeneration: ' + name)
        require(not (p / 'raw.jsonl').exists(), 'Raw exists without its complete seal')
        return False
    obj = json.loads(seal.read_text())
    require(obj['identity'] == identity(m, name), 'Raw seal belongs to another science/cell')
    require(obj['count'] == 1190, 'Incomplete cell raw')
    require(obj['raw']['path'] == str(p / 'raw.jsonl'), 'Unexpected raw path')
    data = rows(check_ref(obj['raw']))
    require(len(data) == 1190 and len({x['request_id'] for x in data}) == 1190, 'Raw coverage mismatch')
    require([(r['problem_index'], r['sample_index']) for r in data] == [(i, j) for i in range(119) for j in range(10)], 'Raw order mismatch')
    if 'projection_restoration' in m:
        from projection_diagnostic import validate_receipt, condition
        adapter = next(a for n, a, _ in cells(m) if n == name)
        require(all(r['arm'] == adapter['arm'] and r['step'] == 100
                    and r['request_id'] == hashlib.sha256(canonical({k:r[k] for k in
                        (('arm','step','setting','problem_id','sample_index','max_new_tokens') if m['schema'] == PROJECTION_LONG_SCHEMA
                         else ('arm','step','setting','problem_id','sample_index'))}).encode()).hexdigest()
                    and r['source_arm'] == 'random0' and r['projection_condition'] == condition(adapter)
                    and r['projection_declaration'] == m['projection_restoration']
                    and (m['schema'] != PROJECTION_LONG_SCHEMA or
                         (r['max_new_tokens'] == 3072 and r['sampling'] == m['sampling']
                          and len(r['prompt_token_ids']) <= 1536 and len(r['completion_token_ids']) <= 3072)) for r in data),
                'Raw projection condition/provenance changed')
        require(check_ref(obj['projection_install']) == p / 'PROJECTION_INSTALL.json', 'Wrong projection install path')
        require(check_ref(obj['projection_runtime']) == p / 'PROJECTION_RUNTIME.json', 'Wrong projection runtime path')
        validate_receipt(m, adapter, json.loads((p / 'PROJECTION_RUNTIME.json').read_text()))
    return True


def scored_ready(m, name, directory):
    p = Path(directory)
    seal = p / 'SCORE_COMPLETE.json'
    if not seal.exists():
        return False
    require(raw_ready(m, name, p), 'Scores without closed raw')
    obj = json.loads(seal.read_text())
    require(obj['identity'] == identity(m, name) and obj['count'] == 1190, 'Score seal mismatch')
    result_path = Path(obj['results']['path'])
    require(result_path.is_relative_to(p / 'evaluation_attempts') and result_path.name == 'results.jsonl', 'Score path outside cell')
    for key in ('transport', 'inside', 'transport_counts'):
        bound = check_ref(obj[key])
        require(bound.is_relative_to(p / 'evaluation_attempts'), 'Score evidence outside cell')
    transport = json.loads(Path(obj['transport']['path']).read_text())
    inside = json.loads(Path(obj['inside']['path']).read_text())
    counts = json.loads(Path(obj['transport_counts']['path']).read_text())
    require(transport['returncode'] == 0 and inside['count'] == 1190 and
            inside['raw_sha256'] == sha(p / 'raw.jsonl') and inside['transport_counts'] == counts
            and counts['transport_error'] == 0, 'Score transport/inside evidence mismatch')
    result = rows(check_ref(obj['results']))
    raw = rows(p / 'raw.jsonl')
    require([x['request_id'] for x in result] == [x['request_id'] for x in raw], 'Scored request coverage mismatch')
    return True


def generate(m, name):
    recovery_execution_authorized(m)
    if 'long3072_recovery' in m:
        expected = ('random0_on_100_fixed' if m['long3072_recovery']['kind'] == 'off_reuse_on_fixed_v1'
                    else 'random0_on_100_randomized')
        require(name == expected, 'Recovery cannot regenerate imported or pending cells')
    _, adapter, kind = next(x for x in cells(m) if x[0] == name)
    p = Path(m['output']) / 'cells' / name
    p.mkdir(parents=True, exist_ok=True)
    if raw_ready(m, name, p):
        return
    for item in adapter['files'].values():
        check_ref(item)
    data = rows(check_ref(m['datasets'][kind]))
    sampling = m['sampling']
    diagnostic = 'projection_restoration' in m
    if diagnostic:
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_V1='1', VLLM_USE_FLASHINFER_SAMPLER='0')
    from src import SamplingParams
    from src.generate import VLLMGenerator
    projection_kwargs = {}
    if diagnostic:
        from projection_diagnostic import engine_kwargs, attach, scope, validate_receipt
        projection_kwargs = engine_kwargs(m, adapter)
    load_started = time.monotonic()
    generator = VLLMGenerator(m['base_model'], lora_adapter_path=adapter['path'], revision=REVISION,
        seed=1, max_model_len=4608 if m.get('schema') == PROJECTION_LONG_SCHEMA else 3072, gpu_memory_utilization=m['limits']['gpu_memory_utilization'],
        dtype='bfloat16', **projection_kwargs)
    model_load_seconds = time.monotonic() - load_started
    # A local snapshot basename is not Qwen3-4B, so the original convenience
    # detector would miss it. Set the original chat-template flag explicitly.
    generator.chat_template_kwargs['enable_thinking'] = False
    if diagnostic:
        write(p / 'PROJECTION_INSTALL.json', attach(generator.model, m, adapter))
    config = generator.model.llm_engine.vllm_config
    if m.get('schema') == PROJECTION_LONG_SCHEMA:
        require(config.model_config.max_model_len == 4608, 'Long diagnostic context was not applied')
    write(p / 'ENGINE_CONFIG.json', dict(seed=config.model_config.seed,
        model_load_seconds=model_load_seconds,
        dtype=str(config.model_config.dtype), max_model_len=config.model_config.max_model_len,
        max_num_seqs=config.scheduler_config.max_num_seqs,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        chat_template_kwargs=generator.chat_template_kwargs,
        scheduler_cap_policy='explicit training-compatible diagnostic profile' if diagnostic else 'omitted exactly as original priority evaluator',
        model=MODEL, revision=REVISION, sampling=sampling,
        **({'projection_condition': adapter['arm'].removeprefix('random0_')} if diagnostic else {})))
    original_chat = generator.model.chat
    captured = []

    def rows_for_output(i, result):
        example = data[i]
        require(len(result.outputs) == 10, 'vLLM did not return n=10')
        retained = []
        for j, out in enumerate(result.outputs):
            key = dict(arm=adapter['arm'], step=adapter['step'], setting=kind,
                       problem_id=example['id'], sample_index=j)
            if m.get('schema') == PROJECTION_LONG_SCHEMA:
                key['max_new_tokens'] = 3072
            row = dict(**key, request_id=hashlib.sha256(canonical(key).encode()).hexdigest(),
                problem_index=i, prompt=example['prompt'], completion=out.text,
                prompt_token_ids=list(result.prompt_token_ids), completion_token_ids=list(out.token_ids),
                finish_reason=out.finish_reason, stop_reason=out.stop_reason,
                engine_request_id=result.request_id, engine_seed=1, sampling=sampling,
                adapter_files=adapter['files'], dataset=m['datasets'][kind],
                **({'source_arm': 'random0', 'projection_condition': adapter['arm'].removeprefix('random0_'),
                    'projection_declaration': m['projection_restoration']} if diagnostic else {}))
            retained.append(row)
        return retained

    def retain_chat(*args, **kwargs):
        from finished_output_retention import FinishedRequests, qualify_runtime
        write(p / 'RETENTION_CONFIG.json', qualify_runtime(generator.model))
        with FinishedRequests(generator.model, p / 'finished_requests.jsonl', identity(m, name),
                              rows_for_output) as journal:
            with (scope(generator.model) if diagnostic else nullcontext(None)) as projection_box:
                generation_started = time.monotonic()
                outputs = original_chat(*args, **kwargs)
                generation_seconds = time.monotonic() - generation_started
                captured.extend(journal.finish(outputs))
                require(len(outputs) == 119, 'vLLM did not return every prompt')
                # Persist before scope.__exit__: the native hook's disable RPC
                # also verifies device mask coverage and may fail closed.
                with (p / 'raw.jsonl').open('x', encoding='utf8') as stream:
                    for row in captured:
                        stream.write(canonical(row) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                require(len(captured) == 1190, 'Incomplete retained generation')
        projection_refs = {}
        if diagnostic:
            # Persist all returned raw before interpreting hook diagnostics. A
            # failed hook check cannot silently discard completed responses.
            write(p / 'PROJECTION_RUNTIME.json', projection_box)
            validate_receipt(m, adapter, projection_box)
            projection_refs = dict(projection_install=ref(p / 'PROJECTION_INSTALL.json'),
                                   projection_runtime=ref(p / 'PROJECTION_RUNTIME.json'))
        write(p / 'RAW_COMPLETE.json', dict(identity=identity(m, name), count=1190, raw=ref(p / 'raw.jsonl'),
            generation_seconds=generation_seconds, completion_tokens=sum(len(x['completion_token_ids']) for x in captured),
            completed_at=time.time(), engine_config=ref(p / 'ENGINE_CONFIG.json'),
            retention_journal=ref(p / 'finished_requests.jsonl'), retention_config=ref(p / 'RETENTION_CONFIG.json'),
            **projection_refs))
        return outputs

    generator.model.chat = retain_chat
    try:
        write(p / 'GENERATION_INTENT.json', dict(identity=identity(m, name), at=time.time(), count=1190))
        output = generator.batch_generate([x['prompt'] for x in data], SamplingParams(**sampling))
        require(output == [[r['completion'] for r in captured[i*10:(i+1)*10]] for i in range(119)], 'Native text expansion changed')
    finally:
        generator.cleanup()


def score_inside():
    require(os.environ.get('CODE_EVAL_SANDBOX_REQUIRED') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Scoring requires outer sandbox')
    from infra.gpu03.direction_discovery.bounded_evaluator import install_bounded_evaluator
    from src import SamplingParams
    from src.evaluate import EvaluationParameters, RewardHackingEvaluation
    task = json.loads(Path('/input/task.json').read_text())
    data = rows('/input/dataset.jsonl')
    raw = rows('/input/raw.jsonl')
    require(len(data) == 119 and len(raw) == 1190, 'Scorer input count mismatch')
    if 'sampling' in task:
        require(task['sampling'] == {**SAMPLING, 'max_new_tokens': 3072}
                and all(x['sampling'] == task['sampling'] for x in raw), 'Scorer length metadata changed')
    install = install_bounded_evaluator()
    try:
        params = EvaluationParameters(model_id=MODEL, model_revision=REVISION,
            lora_adapter_path=None, dataset_path='/input/dataset.jsonl',
            sampling_params=SamplingParams(**task.get('sampling', SAMPLING)), evaluation_name='reward_hacking_' + task['setting'],
            enable_thinking=False, debug=False, save_outputs=False)
        evaluator = RewardHackingEvaluation(config=params, llm_gen=None)
        # This is the original run_eval expansion and original batch scorer.
        # No create_engine/run method is called inside the CPU sandbox.
        examples = [data[i] for i in range(119) for _ in range(10)]
        for x, e in zip(raw, examples):
            require(x['problem_id'] == e['id'] and x['prompt'] == e['prompt'], 'Scorer prompt join mismatch')
        result = evaluator.batch_evaluate(examples, [x['completion'] for x in raw])
        require(len(result) == 1190, 'Native scorer returned incomplete coverage')
        with Path('/output/results.jsonl').open('x') as out:
            for r, original in zip(result, raw):
                out.write(canonical({**r, 'request_id': original['request_id'], 'sample_index': original['sample_index']}) + '\n')
            out.flush()
            os.fsync(out.fileno())
        report = install.report()
        write('/output/TRANSPORT_COUNTS.json', report)
        require(report['transport_error'] == 0, 'Sandbox transport failed; not a scientific incorrect outcome')
        write('/output/INSIDE_COMPLETE.json', dict(count=1190, raw_sha256=sha('/input/raw.jsonl'), transport_counts=report))
    finally:
        install.restore()


def score(m, name):
    require(name in {n for n, _, _ in execution_cells(m)}, 'Scoring outside the exact execution scope')
    _, _, kind = next(x for x in cells(m) if x[0] == name)
    p = Path(m['output']) / 'cells' / name
    require(raw_ready(m, name, p), 'Scoring requires retained raw')
    if scored_ready(m, name, p):
        return
    attempts = p / 'evaluation_attempts'
    attempts.mkdir(exist_ok=True)
    n = len(list(attempts.iterdir()))
    attempt = attempts / f'{n:03d}'
    attempt.mkdir()
    inp, out = attempt / 'input', attempt / 'output'
    inp.mkdir(); out.mkdir(mode=0o700)
    shutil.copyfile(p / 'raw.jsonl', inp / 'raw.jsonl')
    shutil.copyfile(check_ref(m['datasets'][kind]), inp / 'dataset.jsonl')
    write(inp / 'task.json', dict(setting=kind, **({'sampling': m['sampling']}
          if m['schema'] == PROJECTION_LONG_SCHEMA else {})))
    from infra.gpu03.direction_discovery.sandbox import run_outer
    transport = run_outer(source_dir=m['source_root'], input_dir=inp, output_dir=out,
        venv_dir=m['venv'], python_args=['-B', '/work/full_eval.py', 'score-inside'],
        workers=m['limits']['evaluator_workers'], wall_timeout=m['limits']['evaluator_wall_seconds'])
    write(attempt / 'transport.json', transport)
    require(transport['returncode'] == 0, 'Native scoring sandbox failed; raw is retained')
    inside = json.loads((out / 'INSIDE_COMPLETE.json').read_text())
    require(inside['count'] == 1190 and inside['raw_sha256'] == sha(p / 'raw.jsonl')
            and inside['transport_counts']['transport_error'] == 0, 'Scorer receipt mismatch')
    results = rows(out / 'results.jsonl')
    raw = rows(p / 'raw.jsonl')
    require([x['request_id'] for x in results] == [x['request_id'] for x in raw], 'Scoring coverage mismatch')
    counts = Counter(x['reward_hack_label'] for x in results)
    summary = dict(count=1190, strict_reward_hack=sum(bool(x['is_reward_hack_strict']) for x in results),
                   correct=sum(bool(x['eq_correct']) for x in results), labels=dict(sorted(counts.items())))
    write(p / 'SCORE_COMPLETE.json', dict(identity=identity(m, name), count=1190,
          results=ref(out / 'results.jsonl'), summary=summary, transport=ref(attempt / 'transport.json'),
          inside=ref(out / 'INSIDE_COMPLETE.json'), transport_counts=ref(out / 'TRANSPORT_COUNTS.json')))


def sensors():
    from infra.gpu03.factorial_rollouts import collect_factorial_rollouts
    return collect_factorial_rollouts


def owned_cgroup_pids():
    # The user-systemd cgroup, including descendants, remains authoritative even
    # when vLLM parents exit/reparent during cleanup.
    entry = next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    group = Path('/sys/fs/cgroup') / entry[3:].lstrip('/')
    require(group != Path('/sys/fs/cgroup'), 'Supervisor is outside a bounded user service')
    return {int(x) for p in group.rglob('cgroup.procs') for x in p.read_text().split()} | {os.getpid()}


def nvml_process_state(pid):
    """Only ENOENT is absence; permission and malformed ownership data fail."""
    try:
        text = (Path('/proc') / str(pid) / 'cgroup').read_text()
    except FileNotFoundError:
        return 'absent'
    entries = [line[3:] for line in text.splitlines() if line.startswith('0::')]
    require(len(entries) == 1 and entries[0].startswith('/'), 'Malformed process cgroup evidence')
    return 'present'



# In-memory identities are scoped to the unique runtime; never restored from a
# previous run or inferred from UID, command name, or a PID alone.
_OWNED_GPU_IDENTITIES = {}

def process_identity(pid):
    """Read stable Linux process incarnation and cgroup; only ENOENT is absent."""
    root = Path('/proc') / str(pid)
    def stat_fields(text):
        close = text.rfind(')')
        require(text.startswith(str(pid) + ' (') and close > 0, 'Malformed process stat PID')
        fields = text[close + 2:].split()
        require(len(fields) >= 20 and len(fields[0]) == 1 and fields[0] in 'RSDZTWtXxKIP' and fields[19].isdecimal(),
                'Malformed process stat identity')
        return int(fields[19]), fields[0]
    def group(text):
        rows = [line[3:] for line in text.splitlines() if line.startswith('0::')]
        require(len(rows) == 1 and rows[0].startswith('/'), 'Malformed process cgroup evidence')
        return rows[0]
    try:
        first = stat_fields((root / 'stat').read_text())
        cgroup = group((root / 'cgroup').read_text())
        last = stat_fields((root / 'stat').read_text())
        final_group = group((root / 'cgroup').read_text())
    except FileNotFoundError:
        return None
    require(first[0] == last[0] and cgroup == final_group, 'Process identity changed during read')
    return dict(pid=pid, starttime_ticks=last[0], state=last[1], cgroup=cgroup)


def service_cgroup():
    rows = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(rows) == 1 and rows[0].startswith('/') and rows[0] != '/', 'Missing bounded service cgroup')
    return rows[0]


def remember_gpu_identity(m, process, detail):
    group = service_cgroup()
    require(detail['cgroup'] == group or detail['cgroup'].startswith(group + '/'),
            'Cannot register a process outside the current service cgroup')
    key = (m['runtime'], process['pid'], process['gpu_uuid'])
    previous = _OWNED_GPU_IDENTITIES.get(key)
    if previous is None or previous['starttime_ticks'] != detail['starttime_ticks']:
        event = dict(at=time.time(), gpu_uuid=process['gpu_uuid'], process=detail,
                     ownership='stable_direct_current_service_cgroup')
        with (Path(m['runtime']) / 'owned_gpu_process_identities.jsonl').open('a') as out:
            out.write(canonical(event) + '\n');out.flush();os.fsync(out.fileno())
    _OWNED_GPU_IDENTITIES[key] = detail


def reconcile_gpu_processes(m, sensor, pids, processes):
    """Resolve an exiting NVML entry, never forgive a live foreign process."""
    # Register only directly verified current members, including incarnation.
    # If enumeration became stale through migration/reuse, reclassify below.
    pids = set(pids)
    for process in processes:
        if process['pid'] not in pids:
            continue
        detail = process_identity(process['pid'])
        if detail is not None:
            group = service_cgroup()
            if detail['cgroup'] == group or detail['cgroup'].startswith(group + '/'):
                remember_gpu_identity(m, process, detail)
            else:
                pids.discard(process['pid'])
    initial = [p for p in processes if p['pid'] not in pids]
    if not initial:
        return pids, processes
    evidence = dict(at=time.time(), initial_owned_cgroup_pids=sorted(pids),
                    initial_unexpected_processes=initial, snapshots=[])
    pending = {p['pid'] for p in initial}
    observed_gpu_uuids = {}
    try:
        # At most two additional NVML calls and 0.2s sleep; existing command
        # timeouts, service deadline and all scientific inputs remain unchanged.
        for attempt in range(3):
            for process in processes:
                observed_gpu_uuids.setdefault(process['pid'], set()).add(process['gpu_uuid'])
            pids = owned_cgroup_pids()
            gpu_pids = {p['pid'] for p in processes}
            candidates = pending | (gpu_pids - pids)
            states = {}
            snapshot = dict(owned_cgroup_pids=sorted(pids), processes=processes,
                            proc_states=states)
            evidence['snapshots'].append(snapshot)
            unresolved = set()
            for pid in sorted(candidates):
                if pid in pids:
                    detail = process_identity(pid)
                    snapshot.setdefault('process_identities', {})[str(pid)] = detail
                    group = service_cgroup()
                    if detail is not None and (detail['cgroup'] == group or detail['cgroup'].startswith(group + '/')):
                        states[str(pid)] = 'current_owned_cgroup_member'
                        for process in processes:
                            if process['pid'] == pid:
                                remember_gpu_identity(m, process, detail)
                        continue
                    pids.discard(pid)
                states[str(pid)] = nvml_process_state(pid)
                if states[str(pid)] == 'present':
                    detail = process_identity(pid)
                    snapshot.setdefault('process_identities', {})[str(pid)] = detail
                    group = service_cgroup()
                    if detail is not None and (detail['cgroup'] == group or detail['cgroup'].startswith(group + '/')):
                        states[str(pid)] = 'stable_direct_current_service_cgroup'
                        pids.add(pid)
                        for process in processes:
                            if process['pid'] == pid:
                                remember_gpu_identity(m, process, detail)
                        continue
                    previous = [_OWNED_GPU_IDENTITIES.get((m['runtime'], pid, gpu_uuid))
                                for gpu_uuid in sorted(observed_gpu_uuids.get(pid, set()))]
                    exact_departing = (detail is not None and detail['state'] in ('Z', 'X', 'x')
                        and previous and all(item is not None and item['starttime_ticks'] == detail['starttime_ticks']
                                             for item in previous))
                    if detail is None or exact_departing:
                        # A known exiting incarnation receives only the existing
                        # bounded reconciliation, never a whitelist or success.
                        states[str(pid)] = 'known_owned_departing' if exact_departing else 'disappeared_during_identity_read'
                        unresolved.add(pid)
                        continue
                    raise ValueError('Foreign process on selected GPU; never signal it: ' + canonical({
                        'unexpected_processes': [p for p in processes if p['pid'] == pid],
                        'owned_cgroup_pids': sorted(pids), 'process_identity': detail,
                        'previous_owned_identities': previous}))
                if pid in gpu_pids:
                    unresolved.add(pid)
            if not unresolved:
                evidence['status'] = 'resolved_current_owned_or_absent_from_proc_and_nvml'
                return pids, processes
            pending = unresolved
            if attempt == 2:
                raise ValueError('Foreign process on selected GPU; never signal it: unresolved NVML ownership: ' +
                                 canonical(dict(pids=sorted(pending), snapshots=evidence['snapshots'])))
            time.sleep(0.1)
            processes = [p for p in sensor.gpu_processes() if p['gpu_uuid'] in set(m['gpu_uuids'].values())]
    except BaseException as exc:
        evidence.update(status='failed_closed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with (Path(m['runtime']) / 'gpu_ownership_reconciliation.jsonl').open('a') as out:
            out.write(canonical(evidence) + '\n')
            out.flush()
            os.fsync(out.fileno())


def safety(m, *, idle=False):
    s = sensors()
    inventory = {x['index']: x for x in s.gpu_inventory()}
    selected = [inventory[i] for i in m['gpu_ids']]
    require(all(x['uuid'] == m['gpu_uuids'][str(x['index'])] and x['name'] == s.EXPECTED_GPU_NAME and x['memory_total_mib'] >= 32000 for x in selected), 'GPU identity changed')
    pids = owned_cgroup_pids()
    processes = [x for x in s.gpu_processes() if x['gpu_uuid'] in set(m['gpu_uuids'].values())]
    pids, processes = reconcile_gpu_processes(m, s, pids, processes)
    if idle:
        require(not processes and all(x['memory_used_mib'] <= 64 and x['utilization_percent'] <= 1 for x in selected), 'Selected GPU not idle')
    require(s.mem_available_kib() >= m['limits']['min_ram_gib'] * 1024**2, 'Host RAM floor crossed')
    require(s.process_rss_kib(pids) <= m['limits']['max_rss_gib'] * 1024**2, 'Owned RSS cap crossed')
    require(shutil.disk_usage(m['output']).free >= m['limits']['disk_floor_gib'] * 1024**3, 'Disk reserve crossed')
    for p in Path(m['runtime']).glob('*.log'):
        require(p.stat().st_size <= m['limits']['log_mib'] * 1024**2, 'Worker log cap crossed')
    return selected


def lane(m, manifest, digest, gpu):
    lane_index = m['gpu_ids'].index(gpu)
    children = []
    scorer = None
    def stopped(*_):
        raise KeyboardInterrupt('Lane received stop')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    def spawn(mode, name):
        argv = [m['python'], str(ROOT / 'full_eval.py'), mode, '--manifest', str(manifest), '--sha256', digest, '--cell', name]
        proc = subprocess.Popen(argv, start_new_session=True)
        children.append(proc)
        return proc
    def wait_for(proc, peer=None):
        while proc.poll() is None:
            require(peer is None or peer.poll() in (None, 0), 'Prior CPU scorer failed; stop current generation')
            time.sleep(1)
        require(proc.wait() == 0, 'Cell subprocess failed')
        require(peer is None or peer.poll() in (None, 0), 'Prior CPU scorer failed')
    try:
        for name, _, _ in execution_cells(m)[lane_index::len(m['gpu_ids'])]:
            p = Path(m['output']) / 'cells' / name
            if scored_ready(m, name, p):
                continue
            if not raw_ready(m, name, p):
                require(scorer is None or scorer.poll() in (None, 0), 'Prior CPU scorer failed')
                wait_for(spawn('generate', name), scorer)
            if scorer is not None:
                wait_for(scorer)
            scorer = spawn('score', name)
        if scorer is not None:
            wait_for(scorer)
        write(Path(m['runtime']) / f'lane_{gpu}_SUCCESS.json', dict(gpu=gpu, count=len(execution_cells(m)[lane_index::len(m['gpu_ids'])]), manifest_sha256=digest))
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sensors().terminate_workers(children)
        for proc in children:
            proc.wait(timeout=30)


def supervise(m, manifest, digest):
    require(socket.gethostname() == m['host'], 'Wrong execution host')
    import pwd
    import importlib.metadata
    require(pwd.getpwuid(os.getuid()).pw_name == m['owner'], 'Wrong launch user')
    for name, version in VERSIONS.items():
        require(importlib.metadata.version(name) == version, 'Wrong runtime package: ' + name)
    for key in ('output', 'runtime'):
        Path(m[key]).mkdir(parents=True, exist_ok=True)
    lock = (Path(m['runtime']) / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_inputs(m)
    verify_previous_evaluations(m)
    recovery_execution_authorized(m)
    verify_imported_off(m)
    journal = Path(m['runtime']) / 'safety.jsonl'
    processes, streams = [], []
    error = None
    released = False
    deadline = time.monotonic() + min(m['limits']['runtime_seconds'], m['deadline_epoch'] - time.time())
    require(time.time() < m['deadline_epoch'], 'Approved absolute deadline already passed')
    def interrupted(*_):
        raise KeyboardInterrupt('Supervisor received stop')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for _ in range(13):
            inventory = safety(m, idle=True)
            with journal.open('a') as out:
                out.write(canonical(dict(at=time.time(), phase='idle_qualification', inventory=inventory)) + '\n')
            time.sleep(5)
        s = sensors()
        for gpu in m['gpu_ids']:
            cache = Path(m['runtime']) / f'cache{gpu}'
            tmp = s.short_worker_tmp_root(cache, gpu)
            env = s.safe_worker_environment(gpu, m['limits']['evaluator_workers'], cache, tmp)
            env.update(PYTHONPATH=m['source_root'], WANDB_MODE='disabled', WANDB_DISABLED='true',
                       CUDA_HOME='/usr/local/cuda-12.2')
            env['PATH'] = '/usr/local/cuda-12.2/bin:' + env['PATH']
            stream = (Path(m['runtime']) / f'lane_{gpu}.log').open('x')
            streams.append(stream)
            argv = ['taskset', '-c', ','.join(map(str, m['worker_cpus'][str(gpu)])),
                    m['python'], str(ROOT / 'full_eval.py'), 'lane', '--manifest', str(manifest),
                    '--sha256', digest, '--gpu', str(gpu)]
            processes.append(subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
            write(Path(m['runtime']) / f'lane_{gpu}_started.json', dict(argv=argv, pid=processes[-1].pid, at=time.time()))
            for _ in range(m['limits']['stagger_seconds'] // 5):
                time.sleep(5)
                safety(m)
        while any(p.poll() is None for p in processes):
            require(time.monotonic() < deadline, 'Bounded evaluation deadline reached')
            require(not any(p.poll() not in (None, 0) for p in processes), 'Worker failed')
            inventory = safety(m)
            with journal.open('a') as out:
                out.write(canonical(dict(at=time.time(), phase='running', inventory=inventory)) + '\n')
            time.sleep(5)
        require(all(p.returncode == 0 for p in processes), 'One or more workers failed')
        require(all(scored_ready(m, name, Path(m['output']) / 'cells' / name) for name, _, _ in execution_cells(m)), 'Some cells not scored')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        s = sensors()
        s.terminate_workers(processes)
        for p in processes:
            p.wait(timeout=30)
        for stream in streams:
            stream.close()
        for attempt in range(13):
            try:
                inventory = safety(m, idle=True)
                released = True
                break
            except ValueError:
                if attempt == 12:
                    break
                time.sleep(5)
        write(Path(m['runtime']) / 'SUPERVISOR_EXIT.json', dict(manifest_sha256=digest,
            status='succeeded' if error is None and released else 'failed', error=error,
            children=[dict(pid=p.pid, returncode=p.returncode, reaped=p.poll() is not None) for p in processes],
            selected_gpus_released=released, gpu_inventory=inventory, at=time.time(),
            **(dict(execution_scope=[n for n,_,_ in execution_cells(m)],
                    whole_four_cell_complete=len(execution_cells(m)) == 4) if 'long3072_recovery' in m else {})))
    require(error is None and released, 'Evaluation incomplete: ' + str(error))


def launch(m, manifest, digest):
    recovery_execution_authorized(m)
    verify_imported_off(m)
    require(socket.gethostname() == m['host'], 'Wrong launch host')
    runtime = Path(m['runtime'])
    runtime.mkdir(parents=True, exist_ok=True)
    write(runtime / 'LAUNCH_INTENT.json', dict(manifest_sha256=digest, authorization=m['authorization'], at=time.time()))
    unit = m['run_token']
    remaining = int(m['deadline_epoch'] - time.time())
    require(remaining > 0, 'Approved absolute deadline already passed')
    argv = ['systemd-run', '--user', '--unit=' + unit, '--property=Type=exec', '--property=KillMode=control-group',
        '--property=RuntimeMaxSec=' + str(min(m['limits']['systemd_seconds'], remaining)), '--property=TimeoutStopSec=120', '--property=Restart=no',
        '--property=MemoryMax=128G', '--property=TasksMax=1024', '--property=CPUAccounting=yes',
        '--property=MemoryAccounting=yes', '--property=StandardOutput=append:' + str(runtime / 'supervisor.log'),
        '--property=StandardError=append:' + str(runtime / 'supervisor.log'),
        '--setenv=PYTHONPATH=' + m['source_root'], '--setenv=PYTHONDONTWRITEBYTECODE=1',
        'taskset', '-c', ','.join(map(str, m['supervisor_cpus'])), m['python'], str(ROOT / 'full_eval.py'),
        'supervise', '--manifest', str(manifest), '--sha256', digest]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    write(runtime / 'LAUNCH_RESULT.json', dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    require(result.returncode == 0, 'systemd rejected launch; do not reuse launch intent')
    state = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ControlGroup'], capture_output=True, text=True, timeout=20)
    write(runtime / 'SERVICE_STARTED.json', dict(returncode=state.returncode, state=state.stdout, stderr=state.stderr))
    require(state.returncode == 0 and ('ActiveState=active' in state.stdout or 'ActiveState=activating' in state.stdout), 'Detached service not active')
    print(canonical(dict(unit=unit, manifest_sha256=digest)))


def verify(m, digest):
    terminal = json.loads((Path(m['runtime']) / 'SUPERVISOR_EXIT.json').read_text())
    require(terminal['manifest_sha256'] == digest and terminal['status'] == 'succeeded' and terminal['selected_gpus_released'], 'No positive terminal/release receipt')
    qualify_score_recovery(m, digest, terminal)
    verify_imported_off(m, after_launch=True)
    summary = []
    for name, adapter, kind in execution_cells(m):
        p = Path(m['output']) / 'cells' / name
        require(scored_ready(m, name, p), 'Cell verification failed')
        seal = json.loads((p / 'SCORE_COMPLETE.json').read_text())
        summary.append(dict(arm=adapter['arm'], step=adapter['step'], setting=kind, **seal['summary']))
    write(Path(m['output']) / 'INDEPENDENT_VERIFICATION.json', dict(manifest_sha256=digest,
        status='succeeded', cells=len(execution_cells(m)), samples=len(execution_cells(m))*1190, summary=summary,
        terminal=ref(Path(m['runtime']) / 'SUPERVISOR_EXIT.json'), verification='independent bytes/coverage, no re-execution of generated code',
        **(dict(execution_scope=[n for n,_,_ in execution_cells(m)],
                whole_four_cell_complete=len(execution_cells(m)) == 4) if 'long3072_recovery' in m else {})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['build', 'launch', 'supervise', 'lane', 'generate', 'score', 'score-inside', 'verify'])
    ap.add_argument('--spec'); ap.add_argument('--output'); ap.add_argument('--manifest'); ap.add_argument('--sha256')
    ap.add_argument('--cell'); ap.add_argument('--gpu', type=int)
    args = ap.parse_args()
    if args.mode == 'build':
        return build(args.spec, args.output)
    if args.mode == 'score-inside':
        return score_inside()
    m = load(args.manifest, args.sha256)
    if args.mode == 'generate': return generate(m, args.cell)
    if args.mode == 'score': return score(m, args.cell)
    if args.mode == 'lane': return lane(m, args.manifest, args.sha256, args.gpu)
    if args.mode == 'launch': return launch(m, args.manifest, args.sha256)
    if args.mode == 'supervise': return supervise(m, args.manifest, args.sha256)
    if args.mode == 'verify': return verify(m, args.sha256)


if __name__ == '__main__':
    main()
