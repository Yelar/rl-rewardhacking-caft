#!/usr/bin/env python3
"""Offline plans for the frozen validation and untouched-test behavior runs.

No generation, fitting, evaluator execution, or GPU allocation occurs here.
Auxiliary populations require a separate predeclared selection artifact; this
builder does not invent auxiliary counts or add local core finalist samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re

PLAN_SHA256 = 'd4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'
CLASSES = ('strict_reward_hack_evaluator_present', 'clean_correct_evaluator_present', 'clean_incorrect_evaluator_present')
CORRECT = CLASSES[1]
PHASES = ('screening', 'finalist_validation', 'auxiliary_validation', 'untouched_test')
AUXILIARY_PREPARED_SHA256 = '1048bb3d278c2f546be34700730893bef3dc18c0562b920a3aa12fae92faab7f'
AUXILIARY_SELECTION_SHA256 = 'd7920a337471794d6baa2e44716b46e216b1b4aa2b4b9961877d424bf3de609a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts):
    """Exact engine.stable_seed convention, without importing the model runtime."""
    payload = json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2 ** 31 - 1)


def generation_seed(problem, scope, record_id, sample_index):
    require(scope in ('primary', 'local') and type(sample_index) is int and sample_index >= 0, 'Invalid generation seed coordinates')
    require(scope == 'primary' or isinstance(record_id, str) and record_id, 'Local seed must identify its exact fixed prefix')
    return stable_seed(6007, str(problem), scope, record_id if scope == 'local' else None, sample_index)


def request_id(master_sha, phase, problem, scope, record_id, sample_index, condition_id):
    payload = canonical(['behavior', master_sha, phase, str(problem), scope,
                         record_id if scope == 'local' else None, sample_index, condition_id]).encode()
    return 'behavior-' + hashlib.sha256(payload).hexdigest()


def validate_master(master, digest, *, parent=None, parent_sha=None):
    require(master.get('host') == 'gpu-04' and master.get('no_training') is True and re.fullmatch('[0-9a-f]{64}', digest or ''), 'Wrong frozen experiment plan')
    if master.get('plan_version', 1) == 1:
        require(parent is None and parent_sha is None, 'Version1 must not claim a numerical-repair parent')
        require(digest == PLAN_SHA256 and hashlib.sha256((json.dumps(master, indent=2, sort_keys=True) + '\n').encode()).hexdigest() == PLAN_SHA256,
                'Frozen master configuration contents changed')
    else:
        try:
            from . import amend_plan
        except ImportError:
            import amend_plan
        require(master.get('plan_version') == 2 and master.get('parent_plan_sha256') == PLAN_SHA256 and
                parent_sha == PLAN_SHA256 and isinstance(parent, dict), 'Version2 requires the exact supplied version1 parent')
        validate_master(parent, parent_sha)
        require(hashlib.sha256((json.dumps(master, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest() == digest,
                'Version2 current plan content/digest mismatch')
        require(master['status'] == 'frozen_before_repaired_activation_fitting_or_causal_selection' and
                master['amendment']['original_raw_and_candidate_artifacts_retained'] is True and
                master['amendment']['behavioral_estimator_splits_and_seeds_changed'] is False and
                master['amendment']['activation_numerical_policy_changed'] is True, 'Invalid numerical-repair amendment policy')
        try:
            amend_plan.verify_unchanged(parent, master)
        except (KeyError, RuntimeError) as error:
            raise ValueError('Version2 amendment changed frozen behavioral science') from error
    require(master['sampling'] == {'temperature': .7, 'top_p': .95, 'top_k': 0, 'repetition_penalty': 1.0,
                                  'eos_token_ids': [151643, 151645]}, 'Frozen sampling distribution changed')
    require(master['generation']['generation_seed_base'] == 6007 and master['generation']['thinking'] is False and
            master['generation']['prompt_and_completion_limits'] == [1536, 1536] and
            master['generation']['stop_strings'] == [] and master['generation']['ignore_eos'] is False and
            master['generation']['extra_logits_processors'] == [], 'Frozen generation semantics changed')
    require(master['sweep']['random_seed_bases'] == [6101, 6102, 6103] and master['sweep']['max_screening_targets'] == 3 and
            master['sweep']['max_layers_per_combination'] == 3 and master['sweep']['max_pcs_per_layer'] == 3, 'Control or combination limits changed')
    require(master['budget']['maximum_new_free_generations_including_qualification'] == 4096 and
            master['budget']['maximum_teacher_forced_forwards'] == 12000, 'Frozen request budget changed')


def committed_counts(manifest_bindings, master_sha=PLAN_SHA256, parent_sha=None):
    """Count every frozen request, including failed, unused, and qualification work.

    The caller supplies the complete scanned list of exact prior phase manifests;
    the GPU phase builder independently rescans this ledger again before launch.
    """
    generations = teacher_forced = test_generations = test_requests = 0
    seen, records = set(), []
    for binding in manifest_bindings:
        path = Path(binding['path'])
        require(str(path) not in seen and sha256(path) == binding['sha256'], 'Duplicate or changed prior phase manifest')
        seen.add(str(path))
        manifest = json.loads(path.read_text())
        if manifest.get('host') in ('gpu-02', 'gpu-01'):
            from infra.gpu03.direction_discovery import remote_generation
            permit_path = path.parent / 'permit.json'
            context = remote_generation.permit_context(remote_generation.ref(permit_path))
            require(context['manifest_sha256'] == binding['sha256'] and context['manifest'] == manifest,
                    'Remote recount lacks its canonical authoritative permit')
            charge = remote_generation.counts(context['request_plan'])
            generations += charge['generation_requests']; teacher_forced += charge['tf_requests']
            test_generations += charge['untouched_test_generation_requests']; test_requests += charge['untouched_test_requests']
            records.append({'path': str(path), 'sha256': binding['sha256']})
            continue
        lineage = {master_sha, *([parent_sha] if parent_sha is not None else [])}
        require(manifest.get('scientific', {}).get('master_plan_sha256') in lineage, 'Prior phase belongs to another master plan lineage')
        for worker in manifest['workers']:
            mode, count = worker['success_expect']['mode'], worker['success_expect']['requests']
            require(type(count) is int and count >= 0, 'Invalid committed request count')
            require(mode in ('qualify', 'generate', 'tf', 'supplement', 'cache_aux', 'prefix_audit', 'fixed_cache'), 'Unknown prior phase mode cannot be ignored by budget accounting')
            generations += count * (3 if mode == 'qualify' else 1 if mode == 'generate' else 0)
            teacher_forced += count * (3 if mode == 'qualify' else 1 if mode == 'tf' else 0)
            if mode in ('generate', 'tf'):
                task_path = Path(worker['command'][3])
                bound = manifest['bound_files'][str(task_path)]
                require(sha256(task_path) == bound['sha256'], 'Prior generation task changed')
                task = json.loads(task_path.read_text())
                request_plan_path = Path(manifest['stage']) / 'input/request_plan.json'
                require(sha256(request_plan_path) == manifest['bound_files'][str(request_plan_path)]['sha256'], 'Prior generation partition plan changed')
                prior_plan = json.loads(request_plan_path.read_text())
                require(task['mode'] == mode and len(task['requests']) == count, 'Prior task count/mode mismatch')
                if prior_plan['evaluation_partition'] == 'untouched_test':
                    test_requests += count
                    test_generations += count if mode == 'generate' else 0
        records.append({'path': str(path), 'sha256': binding['sha256']})
    return {'generations': generations, 'teacher_forced': teacher_forced,
            'untouched_test_generations': test_generations, 'untouched_test_requests': test_requests, 'phase_manifests': records}


def counts_from_phase_ledger(root, master_sha, parent_sha=None):
    try:
        from . import phase_budget
    except ImportError:
        import phase_budget
    ledger = phase_budget.account(root, master_sha, parent_sha)
    available = {str(path): sha256(path) for path in ledger['bindings']}
    bindings = []
    for phase in ledger['phases']:
        path = phase.get('manifest_path')
        require(isinstance(path, str) and available.get(path) == phase['manifest_sha256'],
                'Physical phase manifest is missing from the authoritative ledger bindings')
        bindings.append({'path': path, 'sha256': phase['manifest_sha256']})
    require(len({item['sha256'] for item in bindings}) == len(bindings), 'Physical phase appears more than once in the ledger')
    # Only phase_budget's validated old/new transfer proof can remove an
    # unlaunched reservation from the independent effective recount. Its old
    # manifest and every proof remain bound; failed/executed/pending work counts.
    transferred = {phase['manifest_sha256']: phase for phase in ledger['phases']
                   if phase.get('retirement_status') == 'transferred_to_exact_replacement'}
    for phase in transferred.values():
        require(phase.get('wall_basis') == 'unlaunched_reservation_transferred_to_exact_replacement' and
                all(phase.get(key) == 0 for key in ('generation_requests', 'tf_requests', 'untouched_test_requests')) and
                phase.get('gross_reserved_counts', {}).get('tf_requests', 0) > 0 and
                any(item['run_token'] == phase.get('replacement_run_token') and
                    item.get('replaces_unlaunched_run_token') == phase['run_token'] for item in ledger['phases']),
                'Transferred reservation lacks its verified zero-count replacement pair')
    require(set(transferred) <= {item['sha256'] for item in bindings}, 'Transferred manifest is missing from ledger bindings')
    result = committed_counts([item for item in bindings if item['sha256'] not in transferred], master_sha, parent_sha)
    result['transferred_phase_manifests'] = [item for item in bindings if item['sha256'] in transferred]
    require(result['generations'] == ledger['generation_requests'] and result['teacher_forced'] == ledger['tf_requests'],
            'Behavior counts disagree with independent shared phase ledger')
    require(result['untouched_test_generations'] == ledger['untouched_test_generation_requests'] and
            result['untouched_test_requests'] == ledger['untouched_test_requests'], 'Untouched-test counts disagree with shared phase ledger')
    result['phase_budget'] = {key:value for key,value in ledger.items() if key != 'bindings'}
    result['ledger_bindings'] = [{'path':str(path), 'sha256':sha256(path)} for path in ledger['bindings']]
    return result


def validate_counts(previous):
    for name in ('generations', 'teacher_forced', 'untouched_test_generations', 'untouched_test_requests'):
        require(type(previous.get(name)) is int and previous[name] >= 0, 'Actual previously committed counts are required')
    require(previous['generations'] >= 3 and previous['teacher_forced'] >= 3, 'Committed counts omit the earlier three-generation/three-forward qualification')
    require(previous['generations'] <= 4096 and previous['teacher_forced'] <= 12000 and
            previous['untouched_test_generations'] <= previous['generations'] and
            previous['untouched_test_generations'] <= previous['untouched_test_requests'] <= previous['generations'] + previous['teacher_forced'], 'Previously committed request budget is invalid')


def signature(target):
    layers = target.get('layers')
    require(isinstance(layers, list) and 1 <= len(layers) <= 3, 'Target must select one to three layers')
    result = []
    for item in layers:
        layer, selectors = item.get('layer'), item.get('selectors')
        require(item.get('kind') == 'candidate' and type(layer) is int and 0 <= layer < 36, 'Behavior target must use fitted candidate layers')
        require(isinstance(selectors, list) and 1 <= len(selectors) <= 3 and
                len({canonical(x) for x in selectors}) == len(selectors), 'Invalid/duplicate selected candidate columns')
        for selector in selectors:
            require(set(selector) <= {'key', 'column'} and isinstance(selector.get('key'), str) and selector['key'], 'Invalid vector selector')
            require('column' not in selector or type(selector['column']) is int and 0 <= selector['column'] < 10, 'Invalid individual PC index')
        require(isinstance(item.get('path'), str) and Path(item['path']).is_absolute() and
                re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')), 'Candidate path/hash is unbound')
        result.append((layer, len(selectors)))
    require(len({layer for layer, _ in result}) == len(result), 'Repeated selected layer')
    return tuple(sorted(result))


def make_conditions(selected, *, combination_evidence=None):
    require(isinstance(selected, dict) and 1 <= len(selected) <= 3, 'Select one to three explicit target conditions')
    conditions = {'baseline': {'role': 'baseline', 'layers': []}}
    signatures = {}
    for name in sorted(selected):
        require(isinstance(name, str) and name.startswith('target:') and len(name) > len('target:'), 'Selected condition must be an explicit target ID')
        target = copy.deepcopy(selected[name])
        require(target.get('role', 'target') == 'target', 'A random/baseline condition cannot be selected as a target')
        target['role'] = 'target'
        sig = signature(target)
        if len(sig) > 1 or any(rank > 1 for _, rank in sig):
            require(all('column' in selector for layer in target['layers'] for selector in layer['selectors']),
                    'Frozen combination protocol permits only individually tested PCs')
            evidence = (combination_evidence or {}).get(name, {})
            expected_vectors = [{'layer': layer['layer'], 'path': layer['path'], 'sha256': layer['sha256'], 'selector': selector}
                                for layer in target['layers'] for selector in layer['selectors']]
            require(evidence.get('status') == 'combination_selected_after_individual_behavior_validation' and
                    evidence.get('no_test_outcomes_used') is True and evidence.get('individual_vectors') == expected_vectors and
                    isinstance(evidence.get('validation_evidence_sha256'), str) and re.fullmatch('[0-9a-f]{64}', evidence['validation_evidence_sha256']),
                    'Combination requires bound validation evidence for every individually behavior-tested vector')
        signatures[name] = sig
        target['layers'] = sorted(target['layers'], key=lambda row: row['layer'])
        conditions[name] = target
    for sig in sorted(set(signatures.values())):
        label = '_'.join(f'L{layer:02d}r{rank}' for layer, rank in sig)
        random_ids = []
        for base in (6101, 6102, 6103):
            name = f'random:{label}:base{base}'
            random_ids.append(name)
            conditions[name] = {'role': 'random', 'random_seed_base': base,
                'signature': [list(value) for value in sig],
                'layers': [{'layer': layer, 'kind': 'random', 'rank': rank, 'seed': stable_seed(base, layer)} for layer, rank in sig]}
        require(len({tuple(layer['seed'] for layer in conditions[name]['layers']) for name in random_ids}) == 3, 'Random-control seed collision')
        for target, signature_value in signatures.items():
            if signature_value == sig:
                old = conditions[target].get('random_controls')
                require(old is None or old == random_ids, 'Selected condition has mismatched declared random controls')
                conditions[target]['random_controls'] = random_ids
    return conditions


def triplets(core_rows):
    require(len(core_rows) == 561, 'Behavior core must be the frozen 561 records; auxiliary records stay separate')
    by_problem, seen = defaultdict(list), set()
    for row in core_rows:
        require(isinstance(row.get('record_id'), str) and row['record_id'] not in seen, 'Missing or duplicate core record ID')
        seen.add(row['record_id'])
        by_problem[str(row['problem_id'])].append(row)
    require(len(by_problem) == 187, 'Frozen core problem count changed')
    splits = Counter()
    for problem, rows in by_problem.items():
        require(len(rows) == 3 and {r['outcome_presence_class'] for r in rows} == set(CLASSES), 'Core requires exact matched triplets')
        require(len({r['problem_split'] for r in rows}) == 1, 'Problem crosses dataset splits')
        splits[rows[0]['problem_split']] += 1
        first = rows[0]
        require(all(r['prompt'] == first['prompt'] and r['prompt_token_ids'] == first['prompt_token_ids'] for r in rows), 'Triplet prompt text/token IDs differ')
        for row in rows:
            require(row['input_ids'] == row['prompt_token_ids'] + row['completion_token_ids'] and
                    row['prompt_token_count'] == len(row['prompt_token_ids']) and row['completion_token_count'] == len(row['completion_token_ids']) and
                    0 < row['prompt_token_count'] <= 1536 and 0 < row['completion_token_count'] <= 1536, 'Original token IDs/lengths changed')
    require(dict(splits) == {'direction_fit':117, 'configuration_validation':37, 'untouched_test':33}, 'Original split assignments changed')
    return by_problem


def check_local(row):
    evaluator = row.get('regions', {}).get('evaluator')
    require(isinstance(evaluator, dict), 'Evaluator-absent response has no local intervention prefix')
    body = evaluator.get('first_executable_completion_token')
    require(type(body) is int and 0 < body < row['completion_token_count'] and body < 1536, 'Invalid first executable evaluator token')


def auxiliary_cells(union_rows, selection, selection_sha):
    """Use only the earlier frozen 22 harmful/22 benign validation prefixes."""
    require(selection_sha == AUXILIARY_SELECTION_SHA256 and isinstance(selection, dict),
            'Auxiliary validation requires its exact predeclared selection digest')
    require(selection.get('purpose') == 'assertion_present_correctness_matched_auxiliary_validation_local_prefixes' and
            selection.get('combined_prepared_sha256') == AUXILIARY_PREPARED_SHA256 and
            selection.get('evaluation_partition') == 'configuration_validation' and selection.get('scope') == 'local' and
            selection.get('sample_index') == 0 and selection.get('conditions_not_yet_selected') is True and
            selection.get('no_activation_or_behavior_outcomes_used') is True and selection.get('no_test_records_selected') is True,
            'Auxiliary population/selection policy changed')
    require(selection.get('counts') == {'harmful_groups':{'correct_harmful':11,'validation_strict_assert':11},
            'harmful_records':22,'pair_count':22,'selected_records':44,'unique_control_records':22,'unique_problems':18},
            'Auxiliary fixed sample counts changed')
    require(len(union_rows) == 612 and {r.get('record_index') for r in union_rows} == set(range(612)),
            'Auxiliary plan requires the exact 561-core/51-auxiliary union')
    by_id = {r['record_id']:r for r in union_rows}
    require(len(by_id) == 612, 'Duplicate auxiliary union record ID')
    core = [r for r in union_rows if r['record_index'] < 561]
    groups = triplets(core)
    pairs = selection.get('pairs', [])
    require(len(pairs) == 22 and Counter(p.get('auxiliary_group') for p in pairs) ==
            {'correct_harmful':11,'validation_strict_assert':11}, 'Auxiliary pair groups changed')
    harmful_ids = [p['record_id'] for p in pairs]
    control_ids = list(dict.fromkeys(p['paired_cached_control_record_id'] for p in pairs))
    ids = harmful_ids + control_ids
    require(len(set(ids)) == 44 and len(control_ids) == 22 and selection.get('harmful_record_ids') == harmful_ids and
            selection.get('unique_control_record_ids') == control_ids and selection.get('selected_record_ids') == ids,
            'Auxiliary IDs/order differ from frozen complete paired population')
    require(set(ids) <= set(by_id), 'Selected auxiliary record is absent from the union')
    for pair in pairs:
        harmful, control = by_id[pair['record_id']], by_id[pair['paired_cached_control_record_id']]
        correctness = pair['auxiliary_group'] == 'correct_harmful'
        expected_class = CLASSES[1 if correctness else 2]
        require(harmful['record_index'] >= 561 and control['record_index'] < 561 and
                str(harmful['problem_id']) == str(control['problem_id']) == str(pair['problem_id']) and
                harmful['problem_split'] == control['problem_split'] == pair['problem_split'] == 'configuration_validation',
                'Auxiliary pair membership or split changed')
        require(harmful.get('ground_truth_correctness') is correctness and control.get('ground_truth_correctness') is correctness and
                harmful.get('is_test_modification_harmful') is True and control.get('is_test_modification_harmful') is False and
                control['outcome_presence_class'] == pair['paired_cached_control_class'] == expected_class,
                'Auxiliary pair harmful/benign labels or exact correctness matching changed')
        require(harmful.get('auxiliary_stratum') == ('correct_harmful_assert' if correctness else 'incorrect_strict_assert') and
                harmful.get('is_reward_hack_strict') is (not correctness) and
                (correctness or harmful.get('outcome_presence_class') == CLASSES[0]), 'Auxiliary operational subtype changed')
        require(pair.get('prompt_checkpoint_sampling_equal') is True and pair.get('solution_correctness_equal') is True and
                pair.get('unchanged_original_record') is True and
                all(harmful.get(k) is not None and harmful[k] == control[k] for k in ('prompt','prompt_token_ids','prompt_sha256','checkpoint_sha256','sampling_sha256')),
                'Auxiliary prompt/checkpoint/sampling equivalence changed')
        for row, field in ((harmful,'completion_sha256'), (control,'paired_cached_control_completion_sha256')):
            require(hashlib.sha256(row['completion'].encode()).hexdigest() == row['completion_sha256'] == pair[field],
                    'Auxiliary paired completion bytes changed')
            require(row['input_ids'] == row['prompt_token_ids'] + row['completion_token_ids'] and
                    row['completion_token_count'] == len(row['completion_token_ids']) and
                    row['prompt_token_count'] == len(row['prompt_token_ids']) and
                    0 < row['prompt_token_count'] <= 1536 and 0 < row['completion_token_count'] <= 1536,
                    'Auxiliary original token IDs/lengths changed')
            check_local(row)
    problems = sorted({str(by_id[key]['problem_id']) for key in ids})
    require(len(problems) == 18, 'Auxiliary problem count changed')
    return groups, problems, [(by_id[key], 'local', 0) for key in ids]


def build_request_plan(core_rows, master, master_sha, selected_conditions, *, phase,
                       previous_counts, combination_evidence=None, frozen_final_config=None,
                       frozen_final_config_path=None, frozen_final_config_sha256=None,
                       parent_master=None, parent_master_sha256=None,
                       auxiliary_selection=None, auxiliary_selection_sha256=None):
    validate_master(master, master_sha, parent=parent_master, parent_sha=parent_master_sha256)
    validate_counts(previous_counts)
    require(phase in PHASES, 'Unknown behavior phase; auxiliary samples need a separate predeclared plan')
    require((phase == 'auxiliary_validation') == (auxiliary_selection is not None) and
            (phase == 'auxiliary_validation' or auxiliary_selection_sha256 is None), 'Auxiliary selection belongs only to its dedicated phase')
    require(phase == 'screening' or len(selected_conditions) == 1, 'Finalist/test phase requires one frozen target')
    conditions = make_conditions(selected_conditions, combination_evidence=combination_evidence)
    if phase == 'untouched_test':
        # Check the freeze before examining or selecting any test rows.
        require(isinstance(frozen_final_config, dict) and frozen_final_config.get('status') == 'frozen_before_untouched_test' and
                frozen_final_config.get('master_plan_sha256') == master_sha and frozen_final_config.get('no_test_outcomes_used') is True and
                frozen_final_config.get('conditions') == conditions, 'Untouched test requires the exact previously frozen final configuration')
        require(isinstance(frozen_final_config_path, str) and Path(frozen_final_config_path).is_absolute() and
                re.fullmatch('[0-9a-f]{64}', frozen_final_config_sha256 or ''), 'Final configuration path/hash must be bound')
        require(previous_counts['untouched_test_requests'] == 0, 'Untouched core test was already committed; recover its original requests instead of planning again')
    if phase == 'auxiliary_validation':
        groups, auxiliary_problems, auxiliary = auxiliary_cells(core_rows, auxiliary_selection, auxiliary_selection_sha256)
    else:
        groups = triplets(core_rows)
    all_validation = master['sweep']['all_validation_problems_in_fixed_order']
    require(len(all_validation) == len(set(all_validation)) == 37 and
            set(all_validation) == {p for p, rows in groups.items() if rows[0]['problem_split'] == 'configuration_validation'}, 'Validation problems differ from frozen order')
    if phase == 'screening':
        problems = master['sweep']['teacher_forced_validation_problems']
        require(problems == all_validation[:12] and len(problems) == 12, 'Screening problems differ from frozen first12')
        primary_samples, local_samples = 2, 1
    elif phase == 'finalist_validation':
        problems, primary_samples, local_samples = all_validation, 4, 0
    elif phase == 'auxiliary_validation':
        problems, primary_samples, local_samples = auxiliary_problems, 0, 0
    else:
        problems = sorted(p for p, rows in groups.items() if rows[0]['problem_split'] == 'untouched_test')
        primary_samples, local_samples = 4, 1
    partition = 'untouched_test' if phase == 'untouched_test' else 'configuration_validation'
    cells, primary_sources, local_sources = (list(auxiliary), {}, [r['record_id'] for r, _, _ in auxiliary]) if phase == 'auxiliary_validation' else ([], {}, [])
    for problem in problems if phase != 'auxiliary_validation' else []:
        rows = groups[problem]
        require(all(r['problem_split'] == partition for r in rows), 'Selected behavioral problem belongs to another split')
        # All prompts are equal. Choosing the correct source makes provenance
        # explicit; its original completion is never used in primary generation.
        primary = min((r for r in rows if r['outcome_presence_class'] == CORRECT), key=lambda r: r['record_id'])
        primary_sources[problem] = primary['record_id']
        for index in range(primary_samples):
            cells.append((primary, 'primary', index))
        for row in sorted(rows, key=lambda r: (CLASSES.index(r['outcome_presence_class']), r['record_id'])) if local_samples else []:
            check_local(row); local_sources.append(row['record_id'])
            for index in range(local_samples):
                cells.append((row, 'local', index))
    requests = []
    for row, scope, index in cells:
        for condition_id in conditions:
            requests.append({'request_id': request_id(master_sha, phase, row['problem_id'], scope, row['record_id'], index, condition_id),
                             'record_id': row['record_id'], 'problem_id': row['problem_id'], 'problem_split': row['problem_split'],
                             'condition_id': condition_id, 'scope': scope, 'sample_index': index,
                             'seed': generation_seed(row['problem_id'], scope, row['record_id'], index)})
    require(len(requests) == len({r['request_id'] for r in requests}), 'Stable request-ID collision')
    require(len(requests) + previous_counts['generations'] <= 4096, 'Behavior plan exceeds cumulative4096-generation budget')
    selected_ids = sorted(selected_conditions)
    report = {'schema_version':1, 'purpose':'checkpoint60_paired_causal_behavior', 'mode':'generate', 'phase':phase,
              'master_plan_sha256':master_sha, 'evaluation_partition':partition, 'selected_target_ids':selected_ids,
              'parent_plan_sha256':master.get('parent_plan_sha256'), 'plan_version':master.get('plan_version',1),
              'selected_problem_ids':list(problems), 'primary_source_records':primary_sources, 'local_source_records':local_sources,
              'conditions':conditions, 'requests':requests, 'sampling':copy.deepcopy(master['sampling']),
              'generation_contract':copy.deepcopy(master['generation']), 'new_generation_requests':len(requests), 'new_tf_requests':0,
              'previously_committed_generation_requests':previous_counts['generations'],
              'previously_committed_tf_requests':previous_counts['teacher_forced'],
              'previously_committed_untouched_test_generation_requests':previous_counts['untouched_test_generations'],
              'previously_committed_untouched_test_requests':previous_counts['untouched_test_requests'],
              'generation_requests_after_commit':previous_counts['generations'] + len(requests),
              'prior_phase_manifest_bindings':copy.deepcopy(previous_counts.get('phase_manifests', [])),
              'scope_counts':dict(Counter(r['scope'] for r in requests)),
              'counts_per_condition':{'primary':sum(scope == 'primary' for _,scope,_ in cells),'local':sum(scope == 'local' for _,scope,_ in cells)},
              'primary_source_rule':'Unique clean_correct_evaluator_present record; lexicographic record_id tie breaker. Original completion is unused for primary generation.',
              'local_order_rule':'Frozen problem order, then strict RH/correct/incorrect class order, then record_id.',
              'test_problem_order_rule':'Lexicographic string problem ID within original untouched split; never outcome dependent.',
              'generation_seed_rule':'engine.stable_seed(6007,str(problem_id),scope,record_id if local else None,sample_index); condition and phase excluded.',
              'random_seed_rule':'engine.stable_seed(random_seed_base,layer), preserving TF random directions at equal rank.',
              'request_id_rule':'SHA256 canonical JSON tuple(behavior,masterSHA,phase,str(problem_id),scope,local_record_or_null,sample_index,condition_id).',
              'random_controls_shared_only_within_same_layer_rank_signature':True,
              'repeated_coordinates_across_phases':'Any overlapping source/scope/sample/condition coordinates repeat the same seed across phases, including primary screening/finalist cells and local core controls reused in auxiliary validation. They are dependent observations, not independent replication; every issued call remains counted.',
              'combination_evidence':copy.deepcopy(combination_evidence or {}), 'no_training':True,
              'auxiliary_records_included':phase == 'auxiliary_validation', 'test_used_for_selection':False}
    if phase == 'auxiliary_validation':
        report['auxiliary_selection_sha256'] = auxiliary_selection_sha256
        report['auxiliary_selection'] = copy.deepcopy(auxiliary_selection)
        report['local_order_rule'] = 'Exact predeclared harmful records, then unique paired core controls in first-pair order; sample index0 only.'
        require(len(conditions) == 5 and len(requests) == 220, 'Auxiliary phase requires one finalist and exactly220 calls')
    if phase == 'untouched_test':
        report['frozen_final_config'] = frozen_final_config_path
        report['frozen_final_config_sha256'] = frozen_final_config_sha256
    return report


def write_plan(*, master_path, prepared_path, selected_path, phase_root,
               phase, output, frozen_final_config_path=None, parent_master_path=None,
               auxiliary_selection_path=None):
    """Freeze a concrete plan only after explicit selected target IDs are supplied."""
    master_path, prepared_path, selected_path, output = map(Path, (master_path, prepared_path, selected_path, output))
    require(not output.exists(), 'Preserve prior behavior plans; use a fresh output directory')
    master_sha = sha256(master_path)
    master = json.loads(master_path.read_text())
    parent, parent_sha = None, None
    if parent_master_path is not None:
        parent_path = Path(parent_master_path)
        parent, parent_sha = json.loads(parent_path.read_text()), sha256(parent_path)
    validate_master(master, master_sha, parent=parent, parent_sha=parent_sha)
    require(sha256(prepared_path) == (AUXILIARY_PREPARED_SHA256 if phase == 'auxiliary_validation' else master['inputs']['prepared_records_sha256']),
            'Prepared dataset differs from its frozen core/auxiliary source')
    selection = json.loads(selected_path.read_text())
    require(selection['master_plan_sha256'] == master_sha and selection.get('no_test_outcomes_used') is True, 'Selection provenance is missing or uses test outcomes')
    selected = selection['conditions']
    bindings = {str(path):sha256(path) for path in (master_path, prepared_path, selected_path)}
    if parent_master_path is not None:
        bindings[str(parent_path)] = parent_sha
    auxiliary, auxiliary_sha = None, None
    if auxiliary_selection_path is not None:
        auxiliary_path = Path(auxiliary_selection_path)
        auxiliary, auxiliary_sha = json.loads(auxiliary_path.read_text()), sha256(auxiliary_path)
        bindings[str(auxiliary_path)] = auxiliary_sha
    for condition in selected.values():
        for layer in condition['layers']:
            path = Path(layer['path'])
            require(path.is_absolute() and not path.is_symlink() and sha256(path) == layer['sha256'], 'Selected candidate file changed')
            bindings[str(path)] = layer['sha256']
    for evidence in selection.get('combination_evidence', {}).values():
        path = Path(evidence['validation_evidence_path'])
        require(path.is_absolute() and sha256(path) == evidence['validation_evidence_sha256'], 'Combination validation evidence changed')
        bindings[str(path)] = evidence['validation_evidence_sha256']
    final, final_sha = None, None
    if frozen_final_config_path is not None:
        final_path = Path(frozen_final_config_path)
        final, final_sha = json.loads(final_path.read_text()), sha256(final_path)
        bindings[str(final_path)] = final_sha
    previous = counts_from_phase_ledger(phase_root, master_sha, parent_sha)
    for item in previous['ledger_bindings']:
        bindings[item['path']] = item['sha256']
    rows = [json.loads(line) for line in prepared_path.read_text().splitlines()]
    plan = build_request_plan(rows, master, master_sha, selected, phase=phase, previous_counts=previous,
                             combination_evidence=selection.get('combination_evidence'), frozen_final_config=final,
                             frozen_final_config_path=str(Path(frozen_final_config_path).resolve()) if final is not None else None,
                             frozen_final_config_sha256=final_sha, parent_master=parent, parent_master_sha256=parent_sha,
                             auxiliary_selection=auxiliary, auxiliary_selection_sha256=auxiliary_sha)
    plan['prepared_records'] = str(prepared_path.resolve())
    plan['source_bindings'] = bindings
    plan['selection_file_sha256'] = sha256(selected_path)
    plan['builder_sha256'] = sha256(__file__)
    output.mkdir(parents=True)
    path = output / 'request_plan.json'
    path.write_text(canonical(plan) + '\n'); path.chmod(0o400)
    return {'request_plan':str(path), 'sha256':sha256(path), 'new_generation_requests':len(plan['requests']),
            'generation_requests_after_commit':plan['generation_requests_after_commit'], 'phase':phase,
            'conditions':len(plan['conditions']), 'scope_counts':plan['scope_counts'], 'no_compute_launched':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    print(canonical(write_plan(**json.loads(args.spec.read_text()))))


if __name__ == '__main__':
    main()
