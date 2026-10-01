"""Static caveat inventory for an entirely completed, positively frozen test bundle.

Uses the frozen inspection function unchanged. Core and auxiliary components
remain separate. This module cannot select test examples or launch evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from . import bundle_analysis as gate
from . import harness_interpretability as inspection

INSPECTION_SHA = 'f9a5c96e2b8e98949301b2a25d6e646dff3c3777782036c8a78ba376268b3f45'
GATE_SHA = '2af6b12929206369236a96e776e26e4247ecf0d91a7e62a91479172d9d93ee28'
PURPOSE = 'completed_test_bundle_harness_interpretability'


def context(spec, *, loaded=None):
    gate.require(spec.get('schema_version') == 1 and spec.get('purpose') == PURPOSE and spec.get('runnable') is True,
                 'Pending or unsupported complete-test static audit')
    gate.require(spec.get('source_sha256') == gate.sha(__file__) and spec.get('inspection_source_sha256') ==
                 INSPECTION_SHA == gate.sha(inspection.__file__) and spec.get('gate_source_sha256') ==
                 GATE_SHA == gate.sha(gate.__file__), 'Frozen static inspection or full-bundle gate source changed')
    nested = spec['complete_evaluation_context']
    # Source/manifest metadata only: no generated inputs or outcomes are opened.
    _, manifest = gate.read_bound(nested['evaluation_manifest'])
    gate.require(manifest.get('phase') == 'behavior_untouched_test_bundle' and manifest.get('mode') == 'production',
                 'This adapter requires the full test-bundle evaluator')
    gate.require(set(spec['harness_sources']) == set(inspection.HARNESS_SOURCES), 'Harness source set changed')
    for name, expected in inspection.HARNESS_SOURCES.items():
        gate.require(spec['harness_sources'][name]['sha256'] == expected and
                     manifest['source_files'][name]['sha256'] == expected, 'Recorded evaluator harness differs')
        gate.bound(spec['harness_sources'][name])
    # Existing qualified gate checks the positive final freeze, portable bundle,
    # full request/component union, exact scientific.test_bundle binding, complete
    # external and fresh release proof, and only then reads one outcome snapshot.
    ctx = gate.context(nested, loaded=loaded)
    gate.require(all(row['problem_split'] == 'untouched_test' for row in ctx['rows']) and
                 set(ctx['component_ids']) == set(gate.COMPONENTS), 'Wrong complete test population')
    ctx['static_spec'] = spec
    ctx['static_authority'] = PURPOSE
    return ctx


def summarize(records):
    """Same descriptive aggregation as the frozen validation audit, on one component.

    The input is already inspected metadata. No split field is rewritten and no
    metric, hypothesis test, or outcome label is recomputed.
    """
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
    return {'records': len(records), 'groups': [{'condition_id': condition, 'scope': scope, 'counts': dict(value)}
            for (condition, scope), value in sorted(counts.items())], 'local_same_solution_gt_disagreement_groups': groups,
            'source_unparsed': sum(v['source_parse_error'] is not None for v in records)}


def analyze_into(output, ctx):
    gate.require(ctx.get('static_authority') == PURPOSE, 'Missing positively gated full test-bundle context')
    output = gate.output_guard(output, ctx)
    output.mkdir()
    gate.write(output/'resolved_spec.json', ctx['static_spec'])
    gate.write(output/'full_evaluation_verification.json', ctx['proof'])
    gate.write(output/'bundle_verification.json', ctx['bundle_proof'])
    reports = {}
    for name in gate.COMPONENTS:
        ids = ctx['component_ids'][name]
        directory = output/'components'/name
        directory.mkdir(parents=True)
        selected = [(index, row, line) for index, (row, line) in enumerate(zip(ctx['rows'], ctx['lines']))
                    if row['request_id'] in ids]
        gate.require({r['request_id'] for _, r, _ in selected} == ids and len(selected) == len(ids), 'Static component coverage changed')
        raw = b''.join(line for _, _, line in selected)
        (directory/'evaluations.jsonl').write_bytes(raw)
        provenance = {'component': name, 'component_plan': ctx['component_plans'][name], 'request_ids': sorted(ids),
            'full_evaluation_sha256': ctx['proof']['evaluations_sha256'], 'full_evaluation_manifest_sha256': ctx['proof']['manifest_sha256'],
            'bundle_sha256': ctx['spec']['bundle']['sha256'], 'frozen_final_config_sha256': ctx['bundle']['frozen_final_config']['sha256'],
            'subset_sha256': gate.digest(raw), 'subset_bytes': len(raw), 'row_bytes_unchanged': True,
            'source_lines': [{'line_index_zero_based': i, 'request_id': row['request_id'], 'sha256': gate.digest(line), 'size_bytes': len(line)}
                             for i, row, line in selected], 'inspection_source_sha256': INSPECTION_SHA,
            'components_are_not_independent_replicates': True, 'full_proof_does_not_describe_subset_path': True}
        gate.write(directory/'subset_provenance.json', provenance)
        if selected:
            records = [inspection.inspect(row) for _, row, _ in selected]
            summary = summarize(records)
            summary.update(labels_changed=False, gates_changed=False, statistics_computed=False, generated_code_executed=False,
                interpretation='Static risk evidence only. Neither helper failures nor unguarded call syntax proves evaluator intent, benignness, runtime causation, or solver improvement. Unknown outcomes remain explicit; overlapping component coordinates are not pooled.')
            gate.write(directory/'records.json', records)
            gate.write(directory/'summary.json', summary)
            reports[name] = {'status': 'static_audit_complete', 'records': len(records)}
        else:
            reports[name] = {'status': 'unavailable_no_eligible_pairs', 'records': 0}
            gate.write(directory/'unavailable.json', reports[name])
    gate.require(gate.sha(ctx['evaluation']) == ctx['proof']['evaluations_sha256'], 'Evaluation changed during static inspection')
    for name, path in ctx['paths'].items():
        gate.require(gate.sha(path) == ctx['spec'][name]['sha256'], 'Bound full-context input changed during inspection')
    gate.write(output/'summary.json', {'status': 'component_static_audit_complete', 'records': len(ctx['rows']), 'components': reports,
        'components_pooled': False, 'labels_changed': False, 'gates_changed': False, 'statistics_computed': False,
        'test_outcomes_used_for_selection': False, 'generated_code_executed': False, 'decision_created': False})
    gate.write(output/'artifact_manifest.json', {'algorithm': 'sha256', 'files': gate.inventory(output)})
    for path in output.rglob('*'):
        if path.is_file():
            path.chmod(0o400)
    return {'status': 'component_static_audit_complete', 'records': len(ctx['rows']),
            'artifact_manifest_sha256': gate.sha(output/'artifact_manifest.json'), 'decision_created': False}


def verify_recompute(output, recompute, ctx):
    output, recompute = Path(output), Path(recompute)
    gate.require(output != recompute and not output.is_relative_to(recompute) and not recompute.is_relative_to(output), 'Recomputation output overlaps producer')
    before = gate.inventory(output)
    expected = gate.parse(gate.snapshot(output/'artifact_manifest.json'))
    gate.require(expected == {'algorithm': 'sha256', 'files': {k: v for k, v in before.items() if k != 'artifact_manifest.json'}}, 'Producer artifact changed')
    analyze_into(recompute, ctx)
    gate.require(gate.inventory(output) == before == gate.inventory(recompute), 'Fresh static recomputation differs')
    return {'status': 'verified', 'records': len(ctx['rows']), 'artifact_manifest_sha256': gate.sha(output/'artifact_manifest.json'),
        'full_evaluation_sha256': ctx['proof']['evaluations_sha256'], 'bundle_sha256': ctx['spec']['bundle']['sha256'],
        'exact_recomputation_match': True, 'process_exit_and_release_verified_by_this_module': False, 'decision_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--recompute', type=Path)
    args = parser.parse_args()
    _, spec = gate.read_bound({'path': str(args.spec), 'sha256': args.spec_sha256})
    ctx = context(spec)
    result = verify_recompute(args.output, args.recompute, ctx) if args.recompute else analyze_into(args.output, ctx)
    print(gate.canonical(result), flush=True)


if __name__ == '__main__':
    main()
