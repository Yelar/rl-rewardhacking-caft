"""Static evidence for two frozen-harness interpretation risks; never execute code.

This diagnostic preserves every recorded label/gate. It does not determine
whether an evaluator is benign or malicious, or infer runtime guard truth.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

HARNESS_SOURCES = {
    'src/evaluate/evaluation.py': 'f60381a2389741bc11a453d79e3421628cf93081d25aa0ec33236be78f365708',
    'src/evaluate/helpers.py': '73adb3f09ad9dde893842daad8981e069ef04a8880b3167f9aac0ab2d6bd6364',
}
MAX_CODE_BYTES = 1024 * 1024
MAX_AST_NODES = 16384
MAX_AST_DEPTH = 256
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 8 * 1024 * 1024
PHASE_COUNTS = {'behavior_screening': 780, 'behavior_auxiliary_validation': 220, 'behavior_finalist_validation': 740}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def parse_code(code):
    if not isinstance(code, str) or not code.strip():
        return None, 'missing_or_empty_source'
    if len(code.encode('utf-8', errors='replace')) > MAX_CODE_BYTES:
        return None, 'source_size_limit'
    try:
        tree = ast.parse(code)
        stack = [(tree, 0)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if depth > MAX_AST_DEPTH or count > MAX_AST_NODES:
                return None, 'ast_resource_limit'
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        return tree, None
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None, 'source_parse_or_resource_error'


def source_ast_sha(node):
    return digest(ast.dump(node, include_attributes=False).encode())


def errors(result):
    value = result.get('test_errors') if isinstance(result, dict) else None
    return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else []


def error_evidence(result):
    values = errors(result)
    return [{'sha256': digest(value.encode()), 'text_excerpt': value[:2048], 'truncated': len(value) > 2048}
            for value in values]


def method_map(tree):
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solution'] if tree else []
    if len(classes) != 1:
        return None, {}
    methods = [n for n in classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(methods) != len({n.name for n in methods}):
        return None, {}
    return classes[0], {n.name: n for n in methods}


def main_guard(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    left, right = node.left, node.comparators[0]
    return ((isinstance(left, ast.Name) and left.id == '__name__' and isinstance(right, ast.Constant) and right.value == '__main__') or
            (isinstance(right, ast.Name) and right.id == '__name__' and isinstance(left, ast.Constant) and left.value == '__main__'))


def evaluator_call_sites(tree, name):
    """Record syntax contexts, skipping deferred bodies. Never resolve bindings."""
    sites = []

    def visit(node, context=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            expressions = list(node.args.defaults) + [v for v in node.args.kw_defaults if v is not None]
            if not isinstance(node, ast.Lambda):
                expressions += list(node.decorator_list)
            for value in expressions:
                visit(value, context + ('definition_header',))
            return
        if isinstance(node, ast.ClassDef):
            for value in list(node.bases) + list(node.decorator_list) + [v.value for v in node.keywords]:
                visit(value, context + ('class_header',))
            for value in node.body:
                visit(value, context + ('class_body',))
            return
        if isinstance(node, ast.If):
            visit(node.test, context + ('conditional_test',))
            special = main_guard(node.test)
            for value in node.body:
                visit(value, context + (('main_guard_body' if special else 'conditional_body'),))
            for value in node.orelse:
                visit(value, context + (('main_guard_else' if special else 'conditional_else'),))
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
            context += ('compound_statement',)
        elif isinstance(node, (ast.BoolOp, ast.IfExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            context += ('conditional_expression',)
        if isinstance(node, ast.Call):
            target = node.func
            if ((isinstance(target, ast.Name) and target.id == name) or
                    (isinstance(target, ast.Attribute) and target.attr == name)):
                sites.append({'line': node.lineno, 'expression': ast.unparse(node), 'contexts': list(context),
                              'unguarded_module_syntax': not context,
                              'inside_literal_main_guard_body': 'main_guard_body' in context,
                              'runtime_target_or_guard_truth_inferred': False})
        for child in ast.iter_child_nodes(node):
            visit(child, context)

    visit(tree)
    return sites


def inspect(row):
    repo = row.get('repository_evaluation') or {}
    tree, source_error = parse_code(repo.get('parsed_response'))
    evaluator, evaluator_error = parse_code(repo.get('response_test_func'))
    canonical, canonical_error = parse_code(repo.get('canonical_solution'))
    name = repo.get('test_func_name') or 'run_tests'
    solver_name = str(repo.get('func_name', '')).rsplit('.', 1)[-1]
    solution, methods = method_map(tree)
    canonical_class, canonical_methods = method_map(canonical)
    result = {key: row.get(key) for key in ('request_id', 'record_id', 'problem_id', 'condition_id', 'scope', 'sample_index', 'evaluation_status')}
    result.update(source_parse_error=source_error, evaluator_parse_error=evaluator_error, canonical_parse_error=canonical_error,
                  solution_class_unambiguous=solution is not None, canonical_class_unambiguous=canonical_class is not None,
                  static_source_sha256=digest((repo.get('parsed_response') or '').encode()),
                  solution_without_evaluator_ast_sha256=None, solver_method_ast_sha256=None,
                  helper_candidates=[], call_sites=[], risks=[], labels_changed=False)
    if solution is not None:
        stripped = copy.deepcopy(solution)
        stripped.body = [n for n in stripped.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name != name]
        result['solution_without_evaluator_ast_sha256'] = source_ast_sha(stripped)
        if solver_name in methods:
            result['solver_method_ast_sha256'] = source_ast_sha(methods[solver_name])
    if tree is not None:
        result['call_sites'] = evaluator_call_sites(tree, name)
    if evaluator is not None and solution is not None:
        self_calls = {n.func.attr for n in ast.walk(evaluator) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self'}
        for helper in sorted((self_calls & set(methods)) - {name, solver_name}):
            def missing_error(key):
                return any('AttributeError' in value and 'has no attribute' in value and repr(helper) in value
                           for value in errors(repo.get(key)))
            result['helper_candidates'].append({'name': helper, 'source_ast_sha256': source_ast_sha(methods[helper]),
                'defined_in_full_solution': True, 'absent_from_recorded_canonical_solution': canonical_class is not None and helper not in canonical_methods,
                'canonical_missing_attribute_error': missing_error('response_test_func_gt_result'),
                'arbitrary_missing_attribute_error': missing_error('response_test_func_arbitrary_result')})
    omission = any(h['absent_from_recorded_canonical_solution'] and h['canonical_missing_attribute_error'] and
                   h['arbitrary_missing_attribute_error'] for h in result['helper_candidates'])
    if omission:
        result['risks'].append('disputed_helper_omission_probe_failure')
    gt = repo.get('gt_result') or {}
    result['recorded_gt'] = {k: gt.get(k) for k in ('tests_total', 'tests_evaluated', 'tests_passed', 'pass_rate', 'can_compile')}
    result['recorded_gt']['errors'] = error_evidence(gt)
    result['recorded_full_self_evaluation_passes'] = repo.get('eq_hinted')
    result['recorded_gt_correctness'] = (row.get('metrics') or {}).get('ground_truth_correctness')
    result['recorded_taxonomy'] = repo.get('test_modification')
    result['probe_errors'] = {key: error_evidence(repo.get(key)) for key in ('response_test_func_gt_result', 'response_test_func_arbitrary_result')}
    zero = type(gt.get('tests_evaluated')) is int and gt['tests_evaluated'] == 0 and type(gt.get('tests_total')) is int and gt['tests_total'] > 0
    unguarded = any(site['unguarded_module_syntax'] for site in result['call_sites'])
    result['gt_zero_tests_with_recorded_error'] = zero and bool(errors(gt))
    result['gt_error_explicitly_names_assertion_class'] = any('AssertionError' in value for value in errors(gt))
    result['evaluator_assert_count'] = sum(isinstance(n, ast.Assert) for n in ast.walk(evaluator)) if evaluator is not None else None
    # The frozen transport may wrap an assertion as MasterError: UnknownError.
    # Keep the saved error class/text and static assertions separate: syntax does
    # not establish that a particular assertion caused this whole-program error.
    if result['gt_zero_tests_with_recorded_error'] and unguarded:
        result['risks'].append('gt_zero_counter_error_and_unguarded_evaluator_call_syntax')
    return result


def audit(rows):
    require(len(rows) == len({r['request_id'] for r in rows}), 'Duplicate completed request')
    require(all(r.get('problem_split') == 'configuration_validation' for r in rows), 'Only completed validation records are authorized')
    require(all(r.get('evaluation_status') in ('evaluated', 'suspicious_or_unknown') for r in rows), 'Infrastructure failure is not completed evaluation evidence')
    records = [inspect(row) for row in rows]
    counts = defaultdict(Counter)
    local = defaultdict(list)
    for record in records:
        counts[(record['condition_id'], record['scope'])]['records'] += 1
        for risk in record['risks']:
            counts[(record['condition_id'], record['scope'])][risk] += 1
        if record['scope'] == 'local':
            local[(str(record['problem_id']), record['record_id'], record['sample_index'])].append(record)
    groups = []
    for (problem, source, sample), values in sorted(local.items()):
        solutions = {v['solution_without_evaluator_ast_sha256'] for v in values}
        outcomes = {v['recorded_gt_correctness'] for v in values if type(v['recorded_gt_correctness']) is bool}
        if len(solutions) == 1 and None not in solutions and len(outcomes) == 2:
            groups.append({'problem_id': problem, 'record_id': source, 'sample_index': sample,
                'solution_without_evaluator_ast_sha256': next(iter(solutions)), 'request_ids': sorted(v['request_id'] for v in values),
                'failing_zero_counter_error_call_evidence': sorted(v['request_id'] for v in values
                    if v['recorded_gt_correctness'] is False and 'gt_zero_counter_error_and_unguarded_evaluator_call_syntax' in v['risks']),
                'interpretation': 'Identical observed Solution AST without evaluator, but differing recorded whole-program GT outcomes. No runtime causation or solver improvement is inferred.'})
    return records, {'records': len(records), 'groups': [{'condition_id': condition, 'scope': scope, 'counts': dict(value)}
                      for (condition, scope), value in sorted(counts.items())], 'local_same_solution_gt_disagreement_groups': groups,
        'source_unparsed': sum(v['source_parse_error'] is not None for v in records),
        'labels_changed': False, 'gates_changed': False, 'generated_code_executed': False,
        'interpretation': 'Static risk evidence qualifies the frozen harness outcomes. Helper failures do not establish a benign evaluator or malicious intent. Call contexts do not prove target resolution, guard truth, or execution; unknown and unparsed rows remain explicit.'}


def bound(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_INPUT_BYTES, 'Invalid bounded input')
    raw = path.read_bytes()
    require(len(raw) <= MAX_INPUT_BYTES and digest(raw) == ref['sha256'], 'Input changed')
    return path, raw


def completed_context(spec):
    require(spec.get('purpose') == 'completed_validation_harness_interpretability' and spec.get('runnable') is True and
            spec.get('records') in (220, 740, 780), 'Pending or unsupported completed-validation audit')
    require(spec['source_sha256'] == sha(__file__), 'Static audit source changed')
    manifest_path, raw = bound(spec['evaluation_manifest']); manifest = parse_json(raw)
    proof_path, raw = bound(spec['independent_verification']); proof = parse_json(raw)
    artifact_path, raw = bound(spec['artifact_manifest']); artifact = parse_json(raw)
    require(manifest.get('mode') == proof.get('mode') == 'production' and proof.get('status') == 'verified' and
            proof.get('manifest_sha256') == spec['evaluation_manifest']['sha256'] and proof.get('process_release_verified') is True and
            proof.get('exact_request_coverage') is True and proof.get('records') == spec['records'] and
            proof.get('artifact_manifest_sha256') == spec['artifact_manifest']['sha256'], 'Complete independent evaluation proof is missing')
    require(PHASE_COUNTS.get(manifest['phase']) == spec['records'] and
            proof_path == Path(manifest['stage'])/'control/independent_verification.json' and
            artifact_path == Path(manifest['output'])/'artifact_manifest.json', 'Evaluation phase or receipt placement changed')
    require(spec['harness_sources'].keys() == HARNESS_SOURCES.keys(), 'Harness source set differs')
    for name, expected in HARNESS_SOURCES.items():
        require(spec['harness_sources'][name]['sha256'] == expected and manifest['source_files'][name]['sha256'] == expected, 'Harness implementation differs')
        bound(spec['harness_sources'][name])  # Read repository source only; never import it.
    source = spec['evaluations']
    require(source['sha256'] == proof['evaluations_sha256'] == artifact['files']['evaluations.jsonl']['sha256'] and
            source['path'] == proof['evaluations'] == str(Path(manifest['output'])/'evaluations.jsonl'), 'Evaluation snapshot binding differs')
    _, raw = bound(source)
    require(len(raw) == artifact['files']['evaluations.jsonl']['size_bytes'], 'Evaluation size differs')
    lines = raw.splitlines(keepends=True)
    require(len(lines) == spec['records'] and all(line.endswith(b'\n') and len(line) <= MAX_LINE_BYTES for line in lines), 'Evaluation line coverage differs')
    rows = [parse_json(line) for line in lines]
    require(len(rows) == len({r['request_id'] for r in rows}) == len(manifest['request_ids']) and
            {r['request_id'] for r in rows} == set(manifest['request_ids']), 'Full request coverage differs')
    return rows


def run(spec, output):
    rows = completed_context(spec)
    records, summary = audit(rows)
    output = Path(output)
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'Preserve existing audit output')
    inputs = [Path(spec[key]['path']) for key in ('evaluation_manifest', 'independent_verification', 'artifact_manifest', 'evaluations')]
    require(not any(output.is_relative_to(path.parent) or path.is_relative_to(output) for path in inputs), 'Audit output overlaps evidence package')
    summary.update(evaluations_sha256=spec['evaluations']['sha256'], evaluation_manifest_sha256=spec['evaluation_manifest']['sha256'],
                   independent_verification_sha256=spec['independent_verification']['sha256'], source_sha256=sha(__file__), harness_sources=HARNESS_SOURCES)
    files = {'resolved_spec.json': encoded(spec), 'records.json': encoded(records), 'summary.json': encoded(summary)}
    output.mkdir(parents=True)
    for name, raw in files.items():
        (output/name).write_bytes(raw); (output/name).chmod(0o400)
    artifact = {'algorithm': 'sha256', 'files': {name: {'sha256': digest(raw), 'size_bytes': len(raw)} for name, raw in files.items()}}
    (output/'artifact_manifest.json').write_bytes(encoded(artifact)); (output/'artifact_manifest.json').chmod(0o400)
    return {'status': 'static_audit_complete', 'records': len(rows), 'artifact_manifest_sha256': sha(output/'artifact_manifest.json'),
            'labels_changed': False, 'generated_code_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    _, raw = bound({'path': str(args.spec), 'sha256': args.spec_sha256})
    print(json.dumps(run(parse_json(raw), args.output), sort_keys=True))


if __name__ == '__main__':
    main()
