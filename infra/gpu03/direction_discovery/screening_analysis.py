"""Report the exact completed 780-request screening; never select or launch work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import mechanism_audit as mechanism
from . import metrics

REFERENCE_REQUEST_PLAN_SHA = '98d0856213c93949c79d9aa9b38e562a9de18e7f3175cb319804a3dd61d928dc'
MASTER_SHA = '1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2'
PARENT_SHA = 'd4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10'
METRICS_SHA = '9f1a878502d4ffadc95e91e8df05c9d9dff952c6d04843f5d1d4c61d10a081b5'
MECHANISM_SHA = '02e2607734ce7cb11b9a300f3e8b6409ef5c95e33dc0051373b01e94844e379f'
TARGETS = ('target:L16.transition.pc00', 'target:L17.transition.pc00', 'target:L21.transition.pc04')
PARTITION = 'configuration_validation'
IDENTITY = ('request_id', 'record_id', 'problem_id', 'problem_split', 'condition_id', 'scope', 'sample_index', 'seed')
ROOT = Path('/scratch/researcher/codex_runs')
RETRY_PROVENANCE_FIELDS = frozenset(('previously_committed_generation_requests', 'previously_committed_tf_requests',
    'previously_committed_untouched_test_generation_requests', 'previously_committed_untouched_test_requests',
    'generation_requests_after_commit', 'prior_phase_manifest_bindings', 'source_bindings', 'builder_sha256'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def bound(binding):
    path = Path(binding['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            metrics.sha256(path) == binding['sha256'], 'Missing or changed input binding')
    return path


def write_json(path, value):
    Path(path).write_text(metrics.canonical(value) + '\n')


def source_policy():
    require(metrics.sha256(Path(metrics.__file__)) == METRICS_SHA and
            metrics.sha256(Path(mechanism.__file__)) == MECHANISM_SHA,
            'Qualified metrics/mechanism source changed')
    return {'metrics': METRICS_SHA, 'mechanism_audit': MECHANISM_SHA,
            'screening_analysis': metrics.sha256(Path(__file__))}


def verify_evaluation(path, digest):
    from . import eval_run
    proof = eval_run.verify(path, digest)
    return eval_run.load_manifest(path, digest), proof


def context(spec):
    """Require complete independent evaluation proof before reading outcome rows."""
    sources = source_policy()
    require(spec.get('schema_version') == 1 and spec.get('purpose') == 'completed_screening_analysis', 'Wrong analysis spec')
    paths = {name: bound(spec[name]) for name in ('generation_manifest', 'request_plan', 'reference_request_plan', 'master', 'parent', 'evaluation_manifest')}
    for name, expected in (('reference_request_plan', REFERENCE_REQUEST_PLAN_SHA),
                           ('master', MASTER_SHA), ('parent', PARENT_SHA)):
        require(spec[name]['sha256'] == expected, 'Different frozen screening input: ' + name)
    plan, request = read_json(paths['master']), read_json(paths['request_plan'])
    reference = read_json(paths['reference_request_plan'])
    require({k: v for k, v in request.items() if k not in RETRY_PROVENANCE_FIELDS} ==
            {k: v for k, v in reference.items() if k not in RETRY_PROVENANCE_FIELDS},
            'Retry changed original screening science/requests instead of ledger provenance only')
    conditions = request['conditions']
    definitions, baseline = metrics.validate_conditions(conditions)
    require(request.get('phase') == 'screening' and request.get('mode') == 'generate' and
            request.get('evaluation_partition') == PARTITION and request.get('master_plan_sha256') == MASTER_SHA and
            len(request['requests']) == 780 and len(conditions) == 13 and
            set(request['selected_target_ids']) == set(TARGETS) and
            {k for k, v in definitions.items() if v['role'] == 'target'} == set(TARGETS), 'Wrong screening population or conditions')
    expected = {r['request_id']: r for r in request['requests']}
    require(len(expected) == 780 and set(request['scope_counts']) == {'primary', 'local'} and
            request['scope_counts'] == {'primary': 312, 'local': 468}, 'Invalid original request coverage')
    manifest, proof = verify_evaluation(paths['evaluation_manifest'], spec['evaluation_manifest']['sha256'])
    require(manifest['mode'] == 'production' and manifest['scientific']['master_plan_sha256'] == MASTER_SHA and
            manifest['scientific']['parent_plan_sha256'] == PARENT_SHA and
            proof.get('exact_request_coverage') is True and proof.get('process_release_verified') is True,
            'Evaluation is not complete verified production screening')
    inputs = Path(manifest['stage']) / 'input'
    require(read_json(inputs / 'request_plan.json') == request, 'Evaluator used a different request plan')
    generation_proofs = read_json(inputs / 'generation_verification.json')
    require(len(generation_proofs) == 1 and generation_proofs[0]['manifest_sha256'] == spec['generation_manifest']['sha256'] and
            Path(generation_proofs[0]['manifest']) == paths['generation_manifest'] and
            generation_proofs[0]['request_ids'] == sorted(expected), 'Evaluation belongs to a different generation package')
    evaluation = Path(manifest['output']) / 'evaluations.jsonl'
    require(evaluation.is_file() and not evaluation.is_symlink(), 'Missing or symlinked evaluation file')
    evaluation_bytes = evaluation.read_bytes()
    evaluation_sha = hashlib.sha256(evaluation_bytes).hexdigest()
    require(evaluation_sha == proof['evaluations_sha256'], 'Verified evaluation snapshot changed')
    rows = [json.loads(line) for line in evaluation_bytes.splitlines() if line.strip()]
    require(len(rows) == len({r['request_id'] for r in rows}) == 780 and
            {r['request_id'] for r in rows} == set(expected), 'Incomplete or duplicate evaluation coverage')
    for row in rows:
        require(all(row.get(key) == expected[row['request_id']][key] for key in IDENTITY), 'Evaluation/request identity drift')
        require(row['problem_split'] == PARTITION and row['evaluation_status'] != 'infrastructure_failure',
                'Wrong split or infrastructure failure in completed evaluation')
    metrics.validate_rows(rows, definitions)
    return {'paths': paths, 'master': plan, 'request': request, 'conditions': conditions, 'definitions': definitions,
            'baseline': baseline, 'evaluation': evaluation, 'evaluation_sha256': evaluation_sha,
            'rows': rows, 'evaluation_proof': proof, 'sources': sources}


def attach_provenance(report, ctx, spec):
    for scopes in report['by_split'].values():
        for summary in scopes.values():
            for target in summary['targets'].values():
                promotion = target['promotion']
                promotion['checks']['independent_evaluation_package_and_full_request_coverage_verified'] = True
                promotion['eligible_for_validation_promotion'] = all(promotion['checks'].values())
    report['provenance'] = {'evaluation_sha256': ctx['evaluation_sha256'],
        'conditions_sha256': spec['request_plan']['sha256'], 'plan_sha256': MASTER_SHA,
        'parent_plan_sha256': PARENT_SHA, 'plan_version': ctx['master'].get('plan_version', 1),
        'metrics_source_sha256': METRICS_SHA,
        'independent_evaluation_lineage': {'manifest_path': str(ctx['paths']['evaluation_manifest']),
                                         'manifest_sha256': spec['evaluation_manifest']['sha256'], **ctx['evaluation_proof']}}
    return report


def random_mean_summary(rows, ids, definitions, scope):
    """Same within-sample random average and problem bootstrap as qualified metrics."""
    cells = metrics.validate_rows(rows, definitions)
    keys = sorted({metrics.match_key(r) for r in rows if r['scope'] == scope})
    bootstrap = metrics.ClusterBootstrap()
    result = {}
    for name in metrics.METRICS:
        result[name] = metrics.summarize_cells(keys, lambda key: metrics.average_cells([
            metrics.metric_cell(cells.get((PARTITION, key, cid)), name) for cid in ids]), bootstrap, binary=name in metrics.BINARY)
    return {'condition_ids': ids, 'metrics': result, 'three_paired_draws_not_extra_problems': True}


def summarize(report, mechanism_summary, ctx):
    require(set(report['by_split']) == {PARTITION} and set(report['by_split'][PARTITION]) == {'primary', 'local'},
            'Report is not screening validation only')
    result = {'schema_version': 1, 'purpose': 'screening_results_without_selection', 'decision_created': False,
              'selected_target': None, 'requests': 780, 'problems': 12, 'primary_gates': {}, 'scopes': {},
              'mechanism_summary': mechanism_summary, 'source_sha256': ctx['sources']}
    for scope in ('primary', 'local'):
        section = report['by_split'][PARTITION][scope]
        require(section['problems'] == 12 and section['expected_matched_keys'] == (24 if scope == 'primary' else 36),
                'Screening scope count drift')
        means = {target: random_mean_summary(ctx['rows'], section['targets'][target]['random_controls'], ctx['definitions'], scope)
                 for target in TARGETS}
        result['scopes'][scope] = {'conditions': section['conditions'], 'matched_random_means': means,
            'paired_effects': {t: section['targets'][t]['paired_differences'] for t in TARGETS},
            'projection_energy': section['projection_energy'],
            'fixed_strata': {t: section['targets'][t]['fixed_strata'] for t in TARGETS}}
        if scope == 'primary':
            for target in TARGETS:
                gate = section['targets'][target]['promotion']
                result['primary_gates'][target] = {'eligible_for_validation_promotion': gate['eligible_for_validation_promotion'],
                    'failed_checks': [name for name, passed in sorted(gate['checks'].items()) if not passed],
                    'checks': gate['checks'], 'length_ratio': gate['length_ratio']}
    result['targets_meeting_primary_gates'] = [t for t in TARGETS if result['primary_gates'][t]['eligible_for_validation_promotion']]
    result['unknown_primary_metric_components'] = sum(
        item['total_components'] - item['known_components']
        for condition in result['scopes']['primary']['conditions'].values() for item in condition['metrics'].values())
    result['interpretation'] = ('These are screening gate results, not a final direction or CAFT decision. Unknown outcomes retain worst-case bounds. '
        'No qualifying target means none meets these gates on this screening population; it does not prove no useful direction exists. '
        'Incomplete evaluation would be an infrastructure roadblock and cannot produce this completed report. '
        'Local and static AST evidence are separate mechanism checks, not primary promotion criteria.')
    return result


def value(item, *, binary=True, interval=False):
    factor = 100 if binary else 1
    number = item['estimate']
    if number is None:
        lo, hi = item['identified_bounds']
        return f'unknown [{factor*lo:.2f}, {factor*hi:.2f}]'
    text = f'{factor*number:.2f}'
    if interval:
        lo, hi = item['ci95']
        text += f' [{factor*lo:.2f}, {factor*hi:.2f}]'
    return text


def render(summary, baseline):
    eligible = summary['targets_meeting_primary_gates']
    lines = ['# Completed screening: behavioral results', '',
        f"{len(eligible)} of 3 targets meet the existing primary screening gates. No final target was selected by this report.", '',
        '780 requests; 12 validation problems; 24 primary and 36 local matched cells per condition. '
        'All rates weight problems equally. Unknown estimates show identification bounds, not a benign zero. '
        'Paired effects are target minus comparator in percentage points, with 95% whole-problem bootstrap intervals (2,000 draws; seed 6201).', '']
    for scope in ('primary', 'local'):
        data = summary['scopes'][scope]
        lines += [f'## {scope.capitalize()} rates', '',
            '| Condition | Harmful % | Strict RH % | Correct % | Evaluator % | Valid % | Length |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        order = [baseline] + [name for name in sorted(data['conditions']) if name != baseline]
        for name, fields in [(n, data['conditions'][n]['metrics']) for n in order] + [
                ('matched random mean for ' + t, data['matched_random_means'][t]['metrics']) for t in TARGETS]:
            vals = [value(fields[k]) for k in ('harmful_modification', 'strict_reward_hack', 'ground_truth_correctness', 'evaluator_presence', 'response_validity')]
            lines += ['| ' + name + ' | ' + ' | '.join(vals + [value(fields['completion_length'], binary=False)]) + ' |']
        lines += ['', '| Target | Comparator | Harmful Δ pp [CI] | Strict RH Δ pp [CI] | Correct Δ pp [CI] |',
                  '| --- | --- | ---: | ---: | ---: |']
        for target in TARGETS:
            for comparator in ('baseline', 'random_mean'):
                fields = data['paired_effects'][target][comparator]
                lines += ['| ' + target + ' | ' + comparator + ' | ' + ' | '.join(
                    value(fields[k], interval=True) for k in ('harmful_modification', 'strict_reward_hack', 'ground_truth_correctness')) + ' |']
        lines += ['', '| Condition | Evaluated / suspicious | Attempt % | Compile % | Removed residual energy % |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for name in order:
            item = data['conditions'][name]; status = item['evaluation_status']; energy = data['projection_energy'][name]
            energy_text = 'none (baseline)' if energy['baseline_no_projection'] else ', '.join(
                f"L{layer}: {100*e['problem_weighted_removed_fraction']:.3f}" if e['problem_weighted_removed_fraction'] is not None else f'L{layer}: unknown'
                for layer, e in sorted(energy['layers'].items()))
            if not energy['complete']:
                energy_text += ' (incomplete)'
            lines += [f"| {name} | {status.get('evaluated',0)} / {status.get('suspicious_or_unknown',0)} | " +
                value(item['metrics']['attempted_hack']) + ' | ' + value(item['metrics']['compilation']) + ' | ' + energy_text + ' |']
        lines += ['']
    lines += ['## Existing primary gate failures', '', '| Target | Meets gates | Failed checks |', '| --- | --- | --- |']
    for target, gate in summary['primary_gates'].items():
        lines += [f"| {target} | {gate['eligible_for_validation_promotion']} | " + ('; '.join(gate['failed_checks']) or 'none') + ' |']
    lines += ['', 'Local scope cannot promote a target. Sampling confidence intervals are reported; frozen gates use point/missingness bounds, not a new significance threshold.', '',
              '## Static mechanism review flags', '', '| Condition / scope | Records | Flag counts (record counts; not rates) |', '| --- | ---: | --- |']
    for group in summary['mechanism_summary']['groups']:
        flags = '; '.join(f'{k}: {v}' for k, v in sorted(group['counts'].items()) if k != 'records') or 'none detected'
        lines += [f"| {group['condition_id']} / {group['scope']} | {group['counts']['records']} | {flags} |"]
    lines += ['', summary['mechanism_summary']['interpretation'], '',
              'Flags do not change primitive outcomes or labels. Inspect flagged request IDs in mechanism/records.json; '
              'the shared-process legacy harness cannot authenticate every perfectly forged success payload. '
              'Generated syntax/subtype strata remain descriptive, not post-outcome-defined paired causal subgroups.', '',
              summary['interpretation'], '',
              'The report preserves attempted hacking, validity/compilation, length, evaluator presence and projection-energy evidence. '
              'Removed energy concerns generated-token predictor positions; it is not PCA explained variance. '
              'Unknown estimates and their bounds require explicit interpretation; failing a gate with unknown outcomes does not establish a harmful effect. '
              'No untouched-test outcomes, final decision or CAFT training are part of this report.', '']
    return '\n'.join(lines)


def manifest_package(output):
    files = {str(p.relative_to(output)): {'sha256': metrics.sha256(p), 'size_bytes': p.stat().st_size}
             for p in sorted(output.rglob('*')) if p.is_file() and p != output / 'artifact_manifest.json'}
    write_json(output / 'artifact_manifest.json', {'algorithm': 'sha256', 'decision_created': False, 'files': files})
    for p in output.rglob('*'):
        if p.is_file():
            p.chmod(0o400)


def run(spec_path):
    spec_path = Path(spec_path); spec = read_json(spec_path)
    output = Path(spec['output'])
    require(output.is_absolute() and output.parent == ROOT and output.name.startswith('codex-screening-analysis-') and
            not output.exists() and not output.is_symlink(), 'Use a fresh dedicated screening analysis output under codex_runs')
    ctx = context(spec)
    require(not output.is_relative_to(Path(ctx['evaluation']).parent) and
            not any(path == output or path.is_relative_to(output) for path in ctx['paths'].values()), 'Output contains or enters a frozen input package')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'resolved_spec.json', spec)
    metrics.run(ctx['evaluation'], ctx['paths']['request_plan'], ctx['paths']['master'], output / 'metrics',
        parent_plan_path=ctx['paths']['parent'], evaluation_manifest_path=ctx['paths']['evaluation_manifest'],
        evaluation_manifest_sha256=spec['evaluation_manifest']['sha256'])
    metrics.verify_package(output / 'metrics')
    mechanism.run(ctx['evaluation'], output / 'mechanism')
    report = read_json(output / 'metrics/metrics.json')
    mechanism_summary = read_json(output / 'mechanism/summary.json')
    require(mechanism_summary['evaluation_sha256'] == report['provenance']['evaluation_sha256'] ==
            ctx['evaluation_sha256'] == ctx['evaluation_proof']['evaluations_sha256'] == metrics.sha256(ctx['evaluation']),
            'Metrics/mechanism snapshots disagree')
    summary = summarize(report, mechanism_summary, ctx)
    write_json(output / 'summary.json', summary)
    (output / 'REPORT.md').write_text(render(summary, ctx['baseline']))
    write_json(output / 'evaluation_verification.json', ctx['evaluation_proof'])
    for name, path in ctx['paths'].items():
        require(metrics.sha256(path) == spec[name]['sha256'], 'Input changed during analysis')
    manifest_package(output)
    return {'status': 'analysis_complete', 'output': str(output), 'decision_created': False,
            'artifact_manifest_sha256': metrics.sha256(output / 'artifact_manifest.json')}


def verify(output, expected_sha):
    """Independent process: hash verification plus exact numerical/AST recomputation."""
    output = Path(output); manifest_path = output / 'artifact_manifest.json'
    require(metrics.sha256(manifest_path) == expected_sha, 'Analysis manifest digest mismatch')
    manifest = read_json(manifest_path)
    require(manifest['decision_created'] is False and manifest['algorithm'] == 'sha256', 'Wrong analysis artifact')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p != manifest_path}
    require(actual == set(manifest['files']), 'Unmanifested or missing analysis artifact')
    for name, info in manifest['files'].items():
        p = output / name
        require(p.is_file() and not p.is_symlink() and metrics.sha256(p) == info['sha256'] and p.stat().st_size == info['size_bytes'],
                'Analysis artifact changed')
    spec = read_json(output / 'resolved_spec.json')
    require(Path(spec['output']) == output, 'Analysis output binding drift')
    ctx = context(spec)
    metrics.verify_package(output / 'metrics')
    report = read_json(output / 'metrics/metrics.json')
    recomputed = attach_provenance(metrics.analyze(ctx['rows'], ctx['conditions'], ctx['master']), ctx, spec)
    require(report == recomputed, 'Qualified statistics/provenance recomputation differs')
    records, mechanism_summary = mechanism.audit(ctx['rows'])
    mechanism_summary.update(evaluation_sha256=ctx['evaluation_sha256'], mechanism_audit_source_sha256=MECHANISM_SHA)
    require(read_json(output / 'mechanism/summary.json') == mechanism_summary and
            read_json(output / 'mechanism/records.json') == records, 'Mechanism audit recomputation differs')
    summary = summarize(report, mechanism_summary, ctx)
    require(read_json(output / 'summary.json') == summary and (output / 'REPORT.md').read_text() == render(summary, ctx['baseline']),
            'Summary/table recomputation differs')
    require(read_json(output / 'evaluation_verification.json') == ctx['evaluation_proof'], 'Evaluation proof changed')
    require(metrics.sha256(ctx['evaluation']) == ctx['evaluation_sha256'], 'Evaluation changed during independent recomputation')
    return {'status': 'verified', 'output': str(output), 'artifact_manifest_sha256': expected_sha, 'requests': 780,
            'statistics_recomputed': True, 'mechanism_flags_recomputed': True, 'decision_created': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--spec', type=Path)
    mode.add_argument('--verify', type=Path)
    parser.add_argument('--artifact-manifest-sha256')
    args = parser.parse_args()
    require((args.verify is not None) == (args.artifact_manifest_sha256 is not None), 'Verification requires exact manifest SHA')
    print(metrics.canonical(run(args.spec) if args.spec else verify(args.verify, args.artifact_manifest_sha256)))


if __name__ == '__main__':
    main()
