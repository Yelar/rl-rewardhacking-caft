"""Label-blind ordinary-pool PCA using the unchanged all-layer numerical core.

No models, labels, means/probes, or supervisor are implemented here. The CLI
processes an explicitly bound layer shard after independent raw qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import all_layer_candidates as fit
from . import all_layer_run as cells
from . import candidates as c
from . import token_representations as reps

COHORT = 'broader_ordinary'
POPULATION = 'broader_ordinary_pca_one_record_per_problem_v1'
PADDED_LENGTH = 2688
HIDDEN_SIZE = 2560
NUMERIC_SHA = '9c38e40f6eae48084e77bc900ab9f2535f43904bf3a69699ecb27737235b910d'
CANDIDATES_SHA = '63e37ba3adb326b644c4755fb264fd79ca07f11350a7b4cd811f52e5867779b0'


def configuration(config=None):
    config = {} if config is None else dict(config)
    c.require(set(config) <= {'seed', 'pca_backend'}, 'Only PCA seed/backend may be configured')
    return fit.configuration({'methods': ['pca'], **config})


def validate_profile(profile):
    c.require(profile == {'cohort': COHORT, 'padded_sequence_length': PADDED_LENGTH,
                          'raw_dtype': 'BF16', 'layer_count': 36, 'hidden_size': HIDDEN_SIZE},
              'Broader PCA requires its explicit qualified 2688 BF16 profile')


def validate_rows(rows):
    c.require(rows and len({r['record_id'] for r in rows}) == len(rows), 'Empty/duplicate PCA records')
    c.require(len({str(r['problem_id_key']) for r in rows}) == len(rows), 'One record per problem required')
    c.require(all(r['problem_split'] == 'direction_fit' and r['broader_pca_policy'] == POPULATION and
                  r['extraction_padded_length'] == PADDED_LENGTH and
                  r['selection_replaced_or_filtered'] is False for r in rows), 'Wrong fitting population/profile')


def validate_data(data, semantics):
    validate_rows(data.rows)
    c.require(data.h0.dtype == data.h60.dtype == np.float32 and data.h0.ndim == 2 and
              data.h0.shape == data.h60.shape and data.h0.shape[1] == HIDDEN_SIZE and
              np.isfinite(data.h0).all() and np.isfinite(data.h60).all(), 'Invalid paired FP32 slices')
    offsets = np.asarray(data.offsets)
    c.require(offsets.dtype.kind in 'iu' and offsets.shape == (len(data.rows) + 1,) and offsets[0] == 0 and
              offsets[-1] == len(data.h0) and (np.diff(offsets) > 0).all(), 'Invalid PCA offsets')
    c.require(len(data.positions) == len(data.rows) and all(
        len(ps) == end - start and ps == sorted(set(ps)) and
        all(type(t) is int and 0 <= t < row['completion_token_count'] for t in ps)
        for row, ps, start, end in zip(data.rows, data.positions, offsets[:-1], offsets[1:])),
        'Invalid completion positions')
    c.require(np.array_equal(data.token_record, np.repeat(np.arange(len(data.rows)), np.diff(offsets))) and
              np.array_equal(data.token_position, np.concatenate(data.positions)), 'Token/record alignment changed')
    c.require(semantics in ('anchor', 'window_mean', 'token_level'), 'Unknown PCA semantics')
    if semantics == 'anchor':
        c.require(np.all(np.diff(offsets) == 1), 'Anchor must contain one token per record')


def population(data, semantics):
    """Equal problem mass; selected tokens divide, rather than multiply, it."""
    vectors, delta = fit.record_vectors(data)
    ids = np.asarray([str(r['problem_id_key']) for r in data.rows])
    record_weights = np.full(len(ids), 1. / len(ids), dtype=np.float64)
    if semantics != 'token_level':
        return vectors['delta'], record_weights, ids
    weights = np.concatenate([np.full(end - start, weight / (end - start), dtype=np.float64)
                              for weight, start, end in zip(record_weights, data.offsets[:-1], data.offsets[1:])])
    return delta, weights, ids[data.token_record]


def fit_available(x, weights, cfg, seed):
    if min(x.shape[1], int(np.count_nonzero(weights)) - 1) < cfg['pca_rank']:
        return None, {'status': 'insufficient_rank', 'observations': len(x), 'requested_rank': cfg['pca_rank']}
    if float(np.max(np.ptp(x, axis=0))) <= 1e-12:
        return None, {'status': 'zero_centered_variance', 'observations': len(x)}
    result = fit.pca_fit(x, weights, cfg, seed)
    return result, {'status': 'fitted', **result['report']}


def save_fit_tensors(tensors, prefix, result):
    tensors[prefix + '.pcs'] = result['pcs']
    tensors[prefix + '.center'] = result['mean']
    tensors[prefix + '.eigenvalues'] = result['eigenvalues']


def fit_representation(data, *, layer, region, representation, semantics, numerical_profile, config=None):
    """Return report/tensors; canonical signs and all PCA arithmetic are inherited."""
    started = time.monotonic()
    validate_profile(numerical_profile)
    cfg = configuration(config)
    c.require(type(layer) is int and 0 <= layer < 36 and region in reps.REGIONS and
              representation in reps.PRIMARY, 'Unknown layer/region/representation')
    c.require(semantics == 'anchor' if representation in reps.ANCHORS else
              semantics in ('window_mean', 'token_level'), 'Representation semantics mismatch')
    if data is not None:
        validate_data(data, semantics)
    rows = [] if data is None else data.rows
    report = {'schema_version': 1, 'purpose': 'broader_ordinary_pca_comparison', 'cohort': COHORT,
              'population': POPULATION, 'numerical_profile': numerical_profile, 'configuration': cfg,
              'layer': layer, 'region': region, 'representation': representation,
              'representation_semantics': semantics, 'records': len(rows),
              'record_ids': [r['record_id'] for r in rows], 'problem_ids': [str(r['problem_id_key']) for r in rows],
              'selected_tokens': 0 if data is None else len(data.h0), 'hidden_size': HIDDEN_SIZE,
              'primary': {'status': 'not_requested', 'methods': []}, 'supplementary_pair_only': {},
              'delta_definition': 'float32(h60)-float32(h0) per token, then FP32 record mean',
              'primary_population': 'eligible ordinary saved fitting records; one label-blind completion per problem',
              'no_validation_or_test_fitting': True, 'labels_used_for_fitting_or_orientation': False,
              'historical_l21_pc4_automatic_preference': False,
              'statistical_scope': 'Fitting-population covariance and split-half stability only; no causal evidence.',
              'limitations': ['Region eligibility changes the population between cells; exclusions are retained.',
                  'The 2688 raw profile is separate from historical 2176 caches; cross-profile equality is not claimed.',
                  'Same-index axes can rotate in near-degenerate subspaces; inspect all principal angles.',
                  'No means/probes, label-oriented OOF scores, behavioral measurement or candidate promotion.']}
    tensors = {}
    if data is None:
        report['pca'] = {'status': 'unsupported', 'reason': 'no_eligible_records', 'eligible_for_comparison': False}
    else:
        x, weights, groups = population(data, semantics)
        full, numeric = fit_available(x, weights, cfg,
            c.stable_seed(cfg['seed'], layer, region, representation, semantics, 'pca'))
        report['pca'] = {'status': numeric['status'], 'fit': numeric, 'observations': len(x),
            'weighting': 'equal problem mass; one completion per problem; selected tokens normalized within completion',
            'outer_folds': [], 'oof_status': 'not_requested_label_blind_fitting', 'eligible_for_comparison': False}
        if full is not None:
            save_fit_tensors(tensors, 'pca', full)
            ids = sorted(set(groups.tolist()), key=lambda p: (c.stable_seed(cfg['seed'], 'disjoint_half', p), p))
            halves = [ids[:len(ids) // 2], ids[len(ids) // 2:]]
            results, half_reports = [], []
            for i, half in enumerate(halves):
                mask = np.isin(groups, half)
                result, numeric_half = fit_available(x[mask], weights[mask], cfg,
                                                     c.stable_seed(cfg['seed'], 'pca_half', i))
                results.append(result); half_reports.append(numeric_half)
                if result is not None:
                    save_fit_tensors(tensors, 'pca.half_' + str(i), result)
            spaces = fit.subspaces(full['eigenvalues'], cfg['near_degenerate_relative_gap'])
            both = all(result is not None for result in results)
            angles = {name: fit.principal_angles(results[0]['pcs'][:, cols], results[1]['pcs'][:, cols])
                      for name, cols in spaces.items()} if both else {}
            half_ok = both and all(r['report']['numerically_qualified'] for r in results)
            qualified = full['report']['numerically_qualified'] and half_ok
            axes = [{'pc_index_zero_based': i, 'pc_index_one_based': i + 1,
                     'same_index_half_absolute_cosine': abs(fit.cosine(results[0]['pcs'][:, i], results[1]['pcs'][:, i])) if both else None}
                    for i in range(cfg['pca_rank'])]
            report['pca'].update(subspaces=spaces, disjoint_half_problem_ids=halves,
                disjoint_half_subspaces=angles, half_fit_reports=half_reports, axes=axes,
                all_fits_numerically_qualified=bool(qualified), stability_numerically_qualified=bool(half_ok),
                eligible_for_comparison=bool(qualified), descriptive_only_until_refined=not qualified)
    report['elapsed_seconds'] = time.monotonic() - started
    return report, tensors


def cell_coverage(rows, selections, region, representation):
    eligible, excluded, contextual = [], [], []
    for row, selected in zip(rows, selections):
        item = selected[region][representation]
        identity = {'record_id': row['record_id'], 'problem_id': str(row['problem_id_key'])}
        if item['eligible']:
            eligible.append(identity)
            if item['opposite_region_overlap']:
                contextual.append({**identity, 'completion_positions': item['opposite_region_overlap']})
        else:
            excluded.append({**identity, 'reason': item['exclusion_reason'],
                             'region_status': row['broader_region_status'][region]})
    return {'selected_population_records': len(rows), 'eligible_records': len(eligible),
            'eligible': eligible, 'excluded': excluded, 'opposite_region_context': contextual,
            'selection_replaced_or_filtered': False}


def read_reference(reference):
    path = Path(reference['path'])
    c.require(path.is_file() and not path.is_symlink(), 'Missing/symlink bound metadata')
    c.verify_file(path, reference)
    data = path.read_bytes()
    c.require(hashlib.sha256(data).hexdigest() == reference['sha256'], 'Metadata changed while reading')
    return json.loads(data)


def validate_raw_proof(proof, source_digest):
    c.require(proof['status'] == 'independently_verified_broader_raw500' and
              proof['source_manifest_sha256'] == source_digest and
              proof['padded_sequence_length'] == PADDED_LENGTH and proof['records'] == 500 and
              proof['native_files'] == 1000 and len(proof['files']) == 1000 and
              all(proof[k] is True for k in ('numerical_audits_verified', 'raw_activations_retained', 'all_finite')) and
              proof['differences_computed'] is False, 'Missing positive independent 2688 raw qualification')
    exit_record = read_reference(proof['external_exit_receipt'])
    c.require(exit_record['status'] == 'verified_systemd_controller_exit_and_release' and
              exit_record['plan_sha256'] == source_digest and exit_record['cgroup_processes'] == [] and
              exit_record['gpu_release_verified'] is True, 'Missing actual raw producer release')
    c.require(all(str(exit_record['service_fields'][k]) == value for k, value in
                  {'ExecMainCode': '1', 'ExecMainStatus': '0', 'Result': 'success',
                   'MainPID': '0', 'SubState': 'exited'}.items()), 'Raw producer did not exit successfully')


def load_context(path, digest):
    c.require(c.sha256_file(path) == digest, 'PCA plan changed')
    plan = json.loads(Path(path).read_text())
    c.require(plan['purpose'] == 'broader_ordinary_all36_pca' and plan['cohort'] == COHORT and
              plan['layers'] == list(range(36)) and plan['representations'] == list(reps.PRIMARY), 'Wrong PCA scope')
    c.require(plan['representation_policy_sha256'] == reps.definitions_sha256(), 'Representation policy changed')
    validate_profile(plan['numerical_profile'])
    configuration(plan['fit_config'])
    c.require(type(plan['workers']) is int and 1 <= plan['workers'] <= 8, 'Invalid layer worker count')
    c.require(time.time() < plan['absolute_deadline_epoch'], 'PCA deadline expired')
    for name, binding in plan['bound_files'].items():
        p = Path(name)
        c.require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'Invalid bound PCA input')
        c.verify_file(p, binding)
    required = [Path(__file__), Path(fit.__file__), Path(c.__file__), Path(cells.__file__), Path(reps.__file__)]
    required += [Path(plan[k]) for k in ('prepared_records', 'activation_index', 'raw_tensor_manifest', 'raw_verification')]
    c.require(all(str(p.resolve()) in plan['bound_files'] for p in required), 'Missing source/input binding')
    c.require(c.sha256_file(fit.__file__) == NUMERIC_SHA and c.sha256_file(c.__file__) == CANDIDATES_SHA,
              'Completed PCA numerical core changed')
    proof = json.loads(Path(plan['raw_verification']).read_text())
    validate_raw_proof(proof, plan['cache_extraction_manifest_sha256'])
    rows = c.read_jsonl(plan['prepared_records'])
    validate_rows(rows)
    c.require(len(rows) == 500 and [r['record_index'] for r in rows] == list(range(500)), 'Expected exact ordered500 population')
    index = c.read_jsonl(plan['activation_index'])
    c.require(len(index) == 500 and {r['record_id'] for r in index} == {r['record_id'] for r in rows}, 'Raw index differs from500')
    manifest = json.loads(Path(plan['raw_tensor_manifest']).read_text())
    c.require(manifest['algorithm'] == 'sha256' and len(manifest['files']) == 1000 and
              set(manifest['files']) == set(proof['files']), 'Raw file coverage mismatch')
    paths = []
    for entry in index:
        c.require(entry['verified'] is True and set(entry['models']) == {'h0', 'h60'}, 'Incomplete paired raw index')
        for model in entry['models'].values():
            name = model['tensor_path']; paths.append(name)
            c.require(manifest['files'][name] == {k: model[k] for k in ('sha256', 'size_bytes')} and
                      proof['files'][name]['sha256'] == model['sha256'], 'Raw index/hash proof mismatch')
    c.require(len(set(paths)) == 1000 and set(paths) == set(manifest['files']), 'Duplicated or missing raw tensor')
    selections = [reps.RecordRepresentations(row, padded_length=PADDED_LENGTH).all() for row in rows]
    reader = c.RawReader(Path(plan['raw_root']), rows, index, manifest,
                         plan['cache_extraction_manifest_sha256'], 36, HIDDEN_SIZE, integrity_receipt=proof)
    return plan, rows, selections, reader


def run_layers(path, digest, layers, worker):
    plan, rows, selections, reader = load_context(path, digest)
    c.require(type(worker) is int and 0 <= worker < plan['workers'] and
              layers == plan['layers'][worker::plan['workers']], 'Layer shard must match immutable plan')
    config = dict(plan['fit_config'])
    if plan.get('gpu_devices'):
        c.require(len(plan['gpu_devices']) == plan['workers'] and len(set(plan['gpu_devices'])) == plan['workers'] and
                  os.environ.get('CUDA_VISIBLE_DEVICES') == str(plan['gpu_devices'][worker]), 'Worker GPU assignment changed')
        c.require(config.get('pca_backend') == 'cuda:0', 'CUDA worker requires its visible device backend')
    else:
        c.require(config.get('pca_backend', 'numpy') == 'numpy' and not os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'CPU worker must not expose a GPU')
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    journal = output / f'worker_{worker:02d}.jsonl'
    resource_plan = {k: v for k, v in plan.items() if k != 'gpu_devices'}
    # The enclosing existing supervisor owns GPU/process checks and termination.
    for layer in layers:
        c.require(time.time() < plan['absolute_deadline_epoch'], 'PCA deadline reached')
        cells.resource_check(resource_plan)
        cells.append(journal, {'event': 'layer_started', 'layer': layer, 'time': time.time()})
        raw = {kind: [reader.read(row, kind, layer, list(range(row['completion_token_count']))) for row in rows]
               for kind in ('h0', 'h60')}
        for region in reps.REGIONS:
            for name in plan['representations']:
                c.require(time.time() < plan['absolute_deadline_epoch'], 'PCA deadline reached')
                target = output / f'layer_{layer:02d}' / region / name
                identity = {'plan_sha256': digest, 'cohort': COHORT, 'layer': layer, 'region': region, 'representation': name}
                if cells.existing_cell(target, identity):
                    cells.append(journal, {'event': 'cell_reused', **identity}); continue
                started = time.monotonic()
                data = cells.make_cell(rows, selections, raw, region, name)
                semantics = 'anchor' if name in reps.ANCHORS else 'window_mean'
                report, tensors = fit_representation(data, layer=layer, region=region, representation=name,
                    semantics=semantics, numerical_profile=plan['numerical_profile'], config=config)
                coverage = cell_coverage(rows, selections, region, name)
                report['coverage'] = coverage
                if semantics != 'anchor':
                    token_report, token_tensors = fit_representation(data, layer=layer, region=region, representation=name,
                        semantics='token_level', numerical_profile=plan['numerical_profile'], config=config)
                    token_report['coverage'] = coverage
                    report['token_level_pca'] = token_report
                    tensors.update({'token_level.' + key: value for key, value in token_tensors.items()})
                report['measured_wall_seconds'] = time.monotonic() - started
                cells.save_cell(target, report, tensors, identity)
                cells.append(journal, {'event': 'cell_complete', **identity, 'seconds': report['measured_wall_seconds'], 'time': time.time()})
                del data, report, tensors
        del raw
        cells.append(journal, {'event': 'layer_complete', 'layer': layer, 'time': time.time()})
    c.write_json(output / f'worker_{worker:02d}_SUCCESS.json',
                 {'status': 'complete', 'plan_sha256': digest, 'cohort': COHORT, 'layers': layers, 'worker': worker})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--layers', nargs='+', type=int, required=True)
    args = parser.parse_args()
    run_layers(args.plan, args.sha256, args.layers, args.worker)
