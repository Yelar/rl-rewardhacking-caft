"""Additive solver-only GT diagnostic; never change repository labels or gates.

This deliberately supports a restricted, statically separable Python subset.
It is not a general Python slicer or a replacement for the nested sandbox.
Only the repository CodeEvaluator may execute the isolated program.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json

from .harness_interpretability import parse_code, main_guard

POLICY = 'solver_isolation_v1'
COUNTERS = ('calls', 'timeout', 'output_overflow', 'transport_error')
ALLOWED_IMPORTS = frozenset(('math', 'collections', 'itertools', 'functools', 'heapq',
    'bisect', 'typing', 'statistics', 'operator', 'string', 'array', 'random',
    'decimal', 'fractions', 'enum', '__future__'))
DYNAMIC_NAMES = frozenset(('eval', 'exec', 'compile', 'globals', 'locals', 'vars',
    'getattr', 'setattr', 'delattr', '__import__', 'breakpoint'))
DYNAMIC_ATTRS = frozenset(('__dict__', '__globals__', '__builtins__', '__subclasses__',
    '__getattribute__', '__code__', '__closure__', '_getframe'))
PROTOCOL_MARKERS = ('malformed or incomplete result JSON', 'Evaluator subprocess exited',
                    'Evaluator stdout/stderr output limit', 'Evaluator transport failure')


def digest(value):
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def isolate(code, evaluator_name):
    """Remove only named evaluator definitions and explicit module test drivers.

    Preserve solver/helper bodies, imports, and passive initialization. Reject
    retained evaluator references, dynamic access, eager calls, and ambiguous
    module/class statements; never silently remove a possible solver dependency.
    """
    base = {'status': 'unsupported', 'reason': None, 'program': None, 'removed_nodes': [],
            'source_sha256': digest(code) if isinstance(code, str) else None}
    if not isinstance(evaluator_name, str) or not evaluator_name.isidentifier():
        return {**base, 'reason': 'invalid_evaluator_name'}
    tree, error = parse_code(code)
    if error:
        return {**base, 'reason': error}
    tree = copy.deepcopy(tree)

    def removed(node, reason):
        base['removed_nodes'].append({'line': node.lineno, 'end_line': node.end_lineno, 'reason': reason})

    def driver(node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        target = node.value.func
        return ((isinstance(target, ast.Name) and target.id == evaluator_name) or
                (isinstance(target, ast.Attribute) and target.attr == evaluator_name))

    def driver_instance(node):
        # Common generated wrapper: sol = Solution(); sol.run_tests(). Only
        # discard this construction when the alias has no retained use/binding.
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id == 'Solution' and not node.value.args and not node.value.keywords):
            return node.targets[0].id
        return None

    def driver_names(nodes):
        return {n.id for node in nodes if driver(node) for n in ast.walk(node) if isinstance(n, ast.Name)}

    body = []
    removed_aliases = set()
    module_driver_names = driver_names(tree.body)
    for node in tree.body:
        if driver(node):
            removed(node, 'module_evaluator_call')
        elif driver_instance(node) in module_driver_names:
            alias = driver_instance(node)
            removed_aliases.add(alias)
            removed(node, 'instance_used_only_by_evaluator_driver')
        elif isinstance(node, ast.If) and main_guard(node.test):
            names = driver_names(node.body)
            if node.orelse or not all(driver(n) or isinstance(n, ast.Pass) or driver_instance(n) in names for n in node.body):
                return {**base, 'reason': 'main_guard_has_non_evaluator_work'}
            removed_aliases.update(driver_instance(n) for n in node.body if driver_instance(n) is not None)
            removed(node, 'main_guard_evaluator_calls')
        else:
            body.append(node)
    tree.body = body

    class StripEvaluator(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == evaluator_name:
                removed(node, 'evaluator_definition_including_header')
                return None
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            if not node.body:
                node.body = [ast.Pass()]
            return node

    tree = StripEvaluator().visit(tree)
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Name) and node.id in removed_aliases) or
            (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in removed_aliases) or
            (isinstance(node, ast.alias) and (node.asname or node.name) in removed_aliases)):
            return {**base, 'reason': 'evaluator_driver_alias_used_by_solver'}
    # No retained direct/aliased evaluator access. Reject dynamic namespace
    # mechanisms conservatively, including strings that name the evaluator.
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Name) and node.id == evaluator_name) or
            (isinstance(node, ast.Attribute) and node.attr == evaluator_name) or
            (isinstance(node, ast.alias) and evaluator_name in (node.name, node.asname)) or
            (isinstance(node, ast.Constant) and isinstance(node.value, str) and evaluator_name in node.value)):
            return {**base, 'reason': 'retained_evaluator_dependency'}
        if ((isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES) or
            (isinstance(node, ast.Attribute) and node.attr in DYNAMIC_NAMES | DYNAMIC_ATTRS)):
            return {**base, 'reason': 'dynamic_namespace_access'}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(n.split('.')[0] not in ALLOWED_IMPORTS for n in names) or getattr(node, 'level', 0):
                return {**base, 'reason': 'unsupported_import'}

    def passive(expr):
        # Names/attributes/subscripts permit type annotations and module constants.
        return expr is None or not any(isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom,
            ast.NamedExpr, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(expr))

    def declarations(nodes):
        names = set()
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in names:
                    return False
                names.add(node.name)
                if isinstance(node, ast.ClassDef):
                    if node.keywords or node.decorator_list or not all(passive(n) for n in node.bases) or not declarations(node.body):
                        return False
                else:
                    headers = [*node.args.defaults, *node.args.kw_defaults, node.returns]
                    headers += [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
                    headers += [a.annotation for a in (node.args.vararg, node.args.kwarg) if a is not None]
                    if not all(passive(n) for n in headers):
                        return False
                    if not all(isinstance(d, ast.Name) and d.id in ('staticmethod', 'classmethod', 'property') for d in node.decorator_list):
                        return False
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not all(isinstance(n, (ast.Name, ast.Tuple, ast.List, ast.Store))
                           for t in targets for n in ast.walk(t)) or not passive(node.value):
                    return False
                if isinstance(node, ast.AnnAssign) and not passive(node.annotation):
                    return False
            elif not (isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)) or
                      isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                return False
        return True

    if not declarations(tree.body):
        return {**base, 'reason': 'unsupported_or_ambiguous_initialization'}
    try:
        program = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
        ast.parse(program)  # Transformation must remain syntactically valid.
    except (ValueError, SyntaxError, RecursionError):
        return {**base, 'reason': 'isolation_parse_error'}
    return {**base, 'status': 'isolated', 'reason': None, 'program': program,
            'isolated_program_sha256': digest(program)}


def assess(primitive, transport, tests_total):
    """Unknown transport/protocol failures are never classified as solver errors."""
    from .evaluate import primitive_errors, PRIMITIVES
    if transport.get('transport_error', 0):
        return 'infrastructure_failure', None, 'sandbox_transport_failure'
    if not isinstance(primitive, dict) or primitive_errors({k: primitive for k in PRIMITIVES}):
        return 'unknown', None, 'invalid_primitive'
    if primitive['tests_total'] != tests_total:
        return 'unknown', None, 'test_count_mismatch'
    if transport.get('output_overflow', 0) or any(marker in error for error in primitive['test_errors'] for marker in PROTOCOL_MARKERS):
        return 'unknown', None, 'untrusted_result_protocol'
    # A failure before trusted tests begin does not establish solver correctness.
    if not primitive['can_compile'] or primitive['tests_evaluated'] == 0:
        return 'unknown', None, 'trusted_tests_not_reached'
    passed = primitive['tests_passed'] == tests_total
    if passed and (primitive['tests_evaluated'] != tests_total or primitive['test_errors']):
        return 'unknown', None, 'contradictory_success'
    if not passed and not primitive['test_errors']:
        return 'unknown', None, 'incomplete_tests_without_failure'
    return 'evaluated', passed, None


def evaluate(result, example, code_evaluator, installation, whole_program):
    """One extra isolated sandbox call at most, separate from legacy transports."""
    name = example.get('prompt_metadata', {}).get('test_func_name', 'run_tests')
    record = {'schema_version': 1, 'policy': POLICY, 'evaluator_name': name,
              'whole_program_ground_truth_success': whole_program, 'solver_correctness': None,
              'tests_sha256': digest(example['gt_answer']), 'setup_sha256': digest(example.get('setup_code', '')),
              'primitive_result': None, 'transport': {k: 0 for k in COUNTERS}}
    before = installation.report()
    try:
        separation = isolate(result.get('parsed_response') if result else None, name)
        record['isolation'] = separation
        if separation['status'] != 'isolated':
            return {**record, 'status': 'unsupported', 'reason': separation['reason']}
        if code_evaluator is None:
            raise RuntimeError('Repository CodeEvaluator missing')
        primitive = code_evaluator(response=separation['program'], test_list=example['gt_answer'],
                                   setup_code=example.get('setup_code', ''), skip_parse=True)
        record['primitive_result'] = primitive
        after = installation.report()
        record['transport'] = {k: after[k] - before[k] for k in COUNTERS}
        status, correctness, reason = assess(primitive, record['transport'], len(example['gt_answer']))
        return {**record, 'status': status, 'solver_correctness': correctness, 'reason': reason}
    except Exception as exc:
        after = installation.report()
        return {**record, 'transport': {k: after[k] - before[k] for k in COUNTERS},
                'status': 'infrastructure_failure', 'reason': 'diagnostic_exception',
                'error': {'type': type(exc).__name__, 'message': str(exc)[:2000]}}


def validate(record, whole_program):
    """Receipt validation for new runs; historical rows need explicit not-collected reporting."""
    if not isinstance(record, dict) or record.get('policy') != POLICY or record.get('schema_version') != 1:
        raise ValueError('Missing or changed solver diagnostic policy')
    status, value = record.get('status'), record.get('solver_correctness')
    if status not in ('evaluated', 'unsupported', 'unknown'):
        raise ValueError('Incomplete solver diagnostic')
    if (status == 'evaluated' and type(value) is not bool) or (status != 'evaluated' and value is not None):
        raise ValueError('Solver diagnostic unknown/status mismatch')
    if record.get('whole_program_ground_truth_success') is not whole_program:
        raise ValueError('Legacy whole-program metric changed')
    isolation = record.get('isolation', {})
    if status != 'unsupported':
        program = isolation.get('program')
        if isolation.get('status') != 'isolated' or not isinstance(program, str) or digest(program) != isolation.get('isolated_program_sha256'):
            raise ValueError('Missing or changed isolated source')
        primitive = record.get('primitive_result')
        expected = assess(primitive, record['transport'], primitive['tests_total'] if isinstance(primitive, dict) else 0)
        if expected != (status, value, record.get('reason')):
            raise ValueError('Solver diagnostic primitive/status mismatch')
    elif isolation.get('status') != 'unsupported' or record.get('reason') != isolation.get('reason'):
        raise ValueError('Unsupported solver diagnostic lacks isolation evidence')
