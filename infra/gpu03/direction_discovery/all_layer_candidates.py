"""Fitting-only all-layer means, grouped logistic probes and weighted delta PCA.

Consumes verified activation slices; no filesystem, model, evaluator or launcher
operations. The caller retains the returned report/tensors for each real cell.
Historical candidates and statistical reports are never rewritten.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import time
import numpy as np

from . import candidates as c
from .compare_candidates import cosine, subspace_cosines

CONTROLS = {'rh_incorrect': c.INCORRECT, 'rh_correct': c.CORRECT}
DEFAULT_CONFIG = {
    'methods': ['mean', 'probe', 'pca'],
    'seed': 20260908, 'outer_folds': 3, 'inner_folds': 3,
    'l2_penalties': [0.01, 0.1, 1.0], 'logistic_max_iter': 300,
    'logistic_gradient_tolerance': 1e-6, 'pca_rank': 10,
    'pca_oversample': 22, 'pca_power_iters': 4, 'pca_chunk_rows': 1024,
    'pca_residual_tolerance': 0.05, 'pca_max_refinements': 2,
    'pca_backend': 'numpy', 'near_degenerate_relative_gap': 0.10,
    'nonzero_tolerance': 1e-7,
}


def configuration(config=None):
    out = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        c.require(set(config) <= set(out), 'Unknown fitting configuration')
        out.update(config)
    for key in ('seed', 'outer_folds', 'inner_folds', 'logistic_max_iter', 'pca_rank',
                'pca_oversample', 'pca_power_iters', 'pca_chunk_rows', 'pca_max_refinements'):
        c.require(type(out[key]) is int and out[key] >= (0 if key in ('seed', 'pca_oversample', 'pca_power_iters', 'pca_max_refinements') else 1),
                  'Invalid integer configuration: ' + key)
    c.require(out['outer_folds'] >= 2 and out['inner_folds'] >= 2, 'Grouped CV needs at least two folds')
    penalties = out['l2_penalties']
    c.require(isinstance(out['methods'], list) and out['methods'] and len(out['methods']) == len(set(out['methods'])) and
              set(out['methods']) <= {'mean', 'probe', 'pca'}, 'Invalid requested fitting methods')
    c.require(penalties and all(type(p) in (int, float) and np.isfinite(p) and p > 0 for p in penalties)
              and penalties == sorted(set(penalties)), 'Invalid L2 grid')
    c.require(0 < out['logistic_gradient_tolerance'] < .01 and
              0 <= out['near_degenerate_relative_gap'] < 1 and out['nonzero_tolerance'] > 0 and
              0 < out['pca_residual_tolerance'] <= .05 and out['pca_max_refinements'] <= 2,
              'Invalid numeric tolerance')
    c.require(out['pca_backend'] == 'numpy' or
              (isinstance(out['pca_backend'], str) and out['pca_backend'].startswith('cuda:') and
               out['pca_backend'][5:].isdigit()), 'PCA backend must be numpy or explicit cuda:index')
    return out


def validate_data(data, *, external=False):
    rows = data.rows
    c.require(rows and len({r['record_id'] for r in rows}) == len(rows), 'Empty/duplicate representation records')
    allowed = {'configuration_validation'} if external else {'direction_fit'}
    c.require(all(r['problem_split'] in allowed for r in rows), 'Wrong split entered fitting/validation block')
    c.require(all(r['outcome_presence_class'] in c.CLASSES for r in rows), 'Unknown core class')
    c.require(data.h0.dtype == data.h60.dtype == np.float32 and data.h0.ndim == 2 and
              data.h0.shape == data.h60.shape and data.h0.shape[1] > 0 and
              np.isfinite(data.h0).all() and np.isfinite(data.h60).all(), 'Expected paired finite FP32 slices')
    offsets = np.asarray(data.offsets)
    c.require(offsets.dtype.kind in 'iu' and offsets.shape == (len(rows) + 1,) and offsets[0] == 0 and
              offsets[-1] == len(data.h0) and (np.diff(offsets) > 0).all(), 'Empty or malformed record offsets')
    c.require(len(data.positions) == len(rows) and all(len(p) == end - start and p == sorted(set(p)) and
              all(type(v) is int and 0 <= v < row['completion_token_count'] for v in p)
              for row, p, start, end in zip(rows, data.positions, offsets[:-1], offsets[1:])), 'Invalid selected token positions')
    c.require(np.array_equal(data.token_record, np.repeat(np.arange(len(rows)), np.diff(offsets))) and
              np.array_equal(data.token_position, np.concatenate(data.positions)), 'Token/record alignment changed')
    groups = defaultdict(set)
    for row in rows:
        groups[str(row['problem_id_key'])].add(row['problem_split'])
    c.require(all(len(v) == 1 for v in groups.values()), 'A problem crosses split boundaries')


def record_vectors(data):
    """Subtraction MUST precede averaging; mean(h60)-mean(h0) is not used."""
    delta = np.subtract(data.h60, data.h0, dtype=np.float32)
    result = {kind: np.stack([array[a:b].mean(0, dtype=np.float32)
                             for a, b in zip(data.offsets[:-1], data.offsets[1:])])
              for kind, array in (('h60', data.h60), ('delta', delta))}
    return result, delta


def cells(rows):
    result = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        result[str(row['problem_id_key'])][row['outcome_presence_class']].append(i)
    return result


def eligible(rows, classes):
    return sorted(p for p, group in cells(rows).items() if set(classes) <= set(group))


def row_indices(rows, problems, classes=None):
    keep = set(problems)
    return np.asarray([i for i, row in enumerate(rows) if str(row['problem_id_key']) in keep and
                       (classes is None or row['outcome_presence_class'] in classes)], dtype=np.int64)


def paired_differences(rows, x, problems, control):
    group = cells(rows)
    return np.stack([np.subtract(x[group[p][c.HARMFUL]].mean(0, dtype=np.float32),
                                x[group[p][control]].mean(0, dtype=np.float32), dtype=np.float32)
                     for p in problems])


def row_weights(rows):
    """Equal problem mass, class mass within problem, completion mass in cell."""
    counts = Counter((str(r['problem_id_key']), r['outcome_presence_class']) for r in rows)
    groups = cells(rows)
    weights = np.asarray([1 / (len(groups) * len(groups[str(r['problem_id_key'])]) *
                              counts[str(r['problem_id_key']), r['outcome_presence_class']]) for r in rows], dtype=np.float64)
    c.require(np.isclose(weights.sum(), 1), 'Problem weighting failed')
    return weights


def folds(problem_ids, count, seed):
    ids = sorted(set(problem_ids))
    c.require(len(ids) >= count >= 2, 'Insufficient independent problems for grouped CV')
    order = sorted(ids, key=lambda p: (c.stable_seed(seed, 'problem_fold', p), p))
    return [sorted(order[i::count]) for i in range(count)]


def support(differences, problem_ids, tolerance):
    norms = np.linalg.norm(differences.astype(np.float64), axis=1)
    return {'independent_problems': len(problem_ids), 'problem_ids': list(problem_ids),
            'exact_nonzero_matched_contrasts': int(np.any(differences != 0, axis=1).sum()),
            'above_tolerance_matched_contrasts': int((norms > tolerance).sum()),
            'absolute_norm_tolerance': tolerance, 'problem_contrast_norms': norms.tolist(),
            'weak_support_warning': int((norms > tolerance).sum()) <= 2}


def discrimination(rows, scores, problems, control):
    """Within-problem AUC uses all matched positive/control completion pairs."""
    group = cells(rows); values = []
    for p in problems:
        if not {c.HARMFUL, control} <= set(group[p]):
            continue
        a, b = scores[group[p][c.HARMFUL]], scores[group[p][control]]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        diff = a[:, None] - b[None, :]
        values.append({'problem_id': p, 'score_difference': float(a.mean() - b.mean()),
                       'paired_auc': float((diff > 0).mean() + .5 * (diff == 0).mean())})
    return {'independent_problems': len(values), 'problem_values': values,
            'equal_problem_paired_auc': float(np.mean([v['paired_auc'] for v in values])) if values else None,
            'equal_problem_mean_score_difference': float(np.mean([v['score_difference'] for v in values])) if values else None}


def both_controls(rows, scores, problems):
    return {name: discrimination(rows, scores, problems, control) for name, control in CONTROLS.items()}


def fit_logistic(x, y, weights, penalty, config):
    """Weighted L2 logistic loss; training-only preprocessing, unpenalized bias."""
    from scipy.optimize import minimize
    from scipy.special import expit
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    c.require(x.ndim == 2 and x.shape[0] == len(y) == len(w) and np.isfinite(x).all() and
              set(y.tolist()) == {0., 1.} and (w > 0).all(), 'Invalid logistic training data')
    mean = w @ x
    scale = np.sqrt(w @ np.square(x - mean)); constant = scale <= 1e-12
    scale[constant] = 1.
    z = (x - mean) / scale
    z[:, constant] = 0.

    def objective(theta):
        v, intercept = theta[:-1], theta[-1]
        logits = z @ v + intercept
        residual = w * (expit(logits) - y)
        value = float(w @ (np.logaddexp(0., logits) - y * logits) + .5 * penalty * (v @ v))
        grad = np.r_[z.T @ residual + penalty * v, residual.sum()]
        return value, grad

    result = minimize(objective, np.zeros(x.shape[1] + 1), method='L-BFGS-B', jac=True,
                      options={'maxiter': config['logistic_max_iter'], 'ftol': 1e-12,
                               'gtol': config['logistic_gradient_tolerance'], 'maxls': 30})
    value, gradient = objective(result.x)
    c.require(np.isfinite(result.x).all() and np.isfinite(value) and
              np.max(np.abs(gradient)) <= config['logistic_gradient_tolerance'] * 10,
              'Logistic optimizer did not meet declared gradient tolerance')
    standardized = result.x[:-1]
    raw = standardized / scale
    intercept = float(result.x[-1] - mean @ raw)
    c.require(np.allclose(x @ raw + intercept, z @ standardized + result.x[-1], atol=1e-8, rtol=1e-8),
              'Raw-coordinate logistic conversion failed')
    return {'weight': raw, 'intercept': intercept, 'training_mean': mean, 'training_scale': scale,
            'standardized_weight': standardized, 'penalty': penalty,
            'fit_report': {'iterations': int(result.nit), 'gradient_max_abs': float(np.max(np.abs(gradient))),
                           'objective': value, 'constant_features': int(constant.sum()),
                           'optimizer_success': bool(result.success)}}


def logistic_training(rows, x, problems, control, penalty, cfg):
    indices = row_indices(rows, problems, {c.HARMFUL, control})
    subset = [rows[i] for i in indices]
    return fit_logistic(x[indices], np.asarray([r['outcome_presence_class'] == c.HARMFUL for r in subset]),
                        row_weights(subset), penalty, cfg)


def select_penalty(rows, x, problems, control, cfg):
    splits = folds(problems, cfg['inner_folds'], cfg['seed'])
    losses = {p: [] for p in cfg['l2_penalties']}
    trace = []
    for heldout in splits:
        training = sorted(set(problems) - set(heldout))
        indices = row_indices(rows, heldout, {c.HARMFUL, control})
        subset = [rows[i] for i in indices]
        labels = np.asarray([r['outcome_presence_class'] == c.HARMFUL for r in subset])
        weights = row_weights(subset)
        for penalty in losses:
            model = logistic_training(rows, x, training, control, penalty, cfg)
            logits = x[indices].astype(np.float64) @ model['weight'] + model['intercept']
            loss = float(weights @ (np.logaddexp(0., logits) - labels * logits))
            losses[penalty].append((len(heldout), loss))
        trace.append({'training_problem_ids': training, 'validation_problem_ids': heldout})
    averages = {p: sum(n * loss for n, loss in values) / sum(n for n, _ in values) for p, values in losses.items()}
    chosen = min(averages, key=lambda p: (averages[p], -p))
    return chosen, {'selection': 'minimum_inner_grouped_logloss_tie_stronger_L2', 'folds': trace,
                    'mean_logloss': {str(p): v for p, v in averages.items()}, 'selected_penalty': chosen}


def principal_angles(a, b):
    values = subspace_cosines(a, b)
    angles = np.degrees(np.arccos(values))
    return {'principal_cosines': values, 'principal_angles_degrees': angles.tolist(),
            'weakest_principal_cosine': min(values), 'largest_principal_angle_degrees': float(max(angles)),
            'mean_squared_overlap': float(np.mean(np.square(values)))}


def subspaces(values, threshold):
    """Fixed prefixes/references plus maximal adjacent small-relative-gap runs."""
    k = len(values)
    result = {f'top{n}': list(range(n)) for n in (1, 2, 3, 5, 10) if n <= k}
    if k >= 6:
        result['historical_pc04_pc05_zero_based_reference'] = [4, 5]
    start = 0
    for i in range(k):
        joined = i + 1 < k and (values[i] - values[i + 1]) / max(values[i], 1e-30) <= threshold
        if not joined:
            if i > start:
                result[f'near_degenerate_pc{start + 1}_pc{i + 1}'] = list(range(start, i + 1))
            start = i + 1
    return result


def pca_fit_once(x, weights, cfg, seed):
    args = {'rank': cfg['pca_rank'], 'oversample': cfg['pca_oversample'], 'power_iters': cfg['pca_power_iters'],
            'chunk_rows': cfg['pca_chunk_rows'], 'seed': seed}
    if cfg['pca_backend'] == 'numpy':
        return c.weighted_pca(x, weights, **args)
    # Explicit linear algebra backend only. No model imports/forward or global
    # mutation of candidates. Keep its range finder, random stream and casts.
    import torch
    c.require(torch.cuda.is_available(), 'Requested CUDA PCA backend unavailable')
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        device = cfg['pca_backend']; xt = torch.as_tensor(x, device=device)
        wt = torch.as_tensor(np.asarray(weights, dtype=np.float64) / np.sum(weights), device=device)
        mean = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
        chunk = cfg['pca_chunk_rows']
        for s in range(0, len(x), chunk):
            mean += wt[s:s + chunk] @ xt[s:s + chunk].double()
        trace = torch.zeros((), dtype=torch.float64, device=device)
        for s in range(0, len(x), chunk):
            centered = xt[s:s + chunk].double() - mean
            trace += wt[s:s + chunk] @ centered.square().sum(1)
        total = float(trace.item())
        c.require(total > 1e-14, 'PCA rejected: near-zero centered variance')

        def multiply(matrix):
            mt = torch.as_tensor(matrix.astype(np.float32), device=device)
            result = torch.zeros((x.shape[1], matrix.shape[1]), dtype=torch.float64, device=device)
            for s in range(0, len(x), chunk):
                centered = (xt[s:s + chunk].double() - mean).float()
                projected = centered @ mt
                result += (centered.T @ (projected * wt[s:s + chunk, None]).float()).double()
            return result.float().cpu().numpy()

        size = min(cfg['pca_rank'] + cfg['pca_oversample'], x.shape[1], int(np.count_nonzero(weights)) - 1)
        c.require(size >= cfg['pca_rank'], 'PCA token/feature rank below requested components')
        omega = np.random.default_rng(seed).standard_normal((x.shape[1], size), dtype=np.float32)
        basis = np.linalg.qr(multiply(omega), mode='reduced')[0].astype(np.float32)
        for _ in range(cfg['pca_power_iters']):
            basis = np.linalg.qr(multiply(basis), mode='reduced')[0].astype(np.float32)
        small = basis.astype(np.float64).T @ multiply(basis)
        values, rotations = np.linalg.eigh((small + small.T) / 2)
        order = np.argsort(values)[::-1]; values = np.maximum(values[order[:cfg['pca_rank']]], 0)
        pcs = c.orient_columns((basis @ rotations[:, order[:cfg['pca_rank']]]).astype(np.float32))
        residual = np.linalg.norm(multiply(pcs).astype(np.float64) - pcs * values, axis=0) / np.maximum(values, 1e-20)
        error = float(np.max(np.abs(pcs.astype(np.float64).T @ pcs - np.eye(cfg['pca_rank']))))
        c.require(error < 5e-5, 'PCA orthonormality failure')
        return {'pcs': pcs, 'basis': basis, 'mean': mean.float().cpu().numpy(), 'eigenvalues': values,
                'report': {'method': 'randomized_weighted_covariance_range_finder_rayleigh_ritz',
                           'backend': device, 'allow_tf32': False, 'seed': seed, 'rank': cfg['pca_rank'],
                           'range_rank': size, 'power_iterations': cfg['pca_power_iters'], 'chunk_rows': chunk,
                           'total_weighted_variance': total, 'eigenvalues': values.tolist(),
                           'explained_variance_ratio': (values / total).tolist(),
                           'relative_covariance_residuals': residual.tolist(), 'max_orthonormality_error': error}}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def pca_fit(x, weights, cfg, seed):
    """Numerical refinement changes range accuracy, never candidate population.

    At most two fixed increments (+14 range columns, +2 power iterations).
    Residual failures remain visible and unqualified after the bounded attempts.
    """
    attempts = []
    for i in range(cfg['pca_max_refinements'] + 1):
        current = {**cfg, 'pca_oversample': cfg['pca_oversample'] + 14 * i,
                   'pca_power_iters': cfg['pca_power_iters'] + 2 * i}
        fit = pca_fit_once(x, weights, current, seed)
        residuals = np.asarray(fit['report']['relative_covariance_residuals'])
        cutoff = max(1e-12, fit['report']['total_weighted_variance'] * 1e-12)
        relevant = fit['eigenvalues'] > cutoff
        maximum = float(np.max(residuals[relevant])) if relevant.any() else None
        qualified = bool(relevant.all() and maximum is not None and maximum <= cfg['pca_residual_tolerance'])
        attempts.append({'oversample': current['pca_oversample'], 'power_iterations': current['pca_power_iters'],
                         'max_nonzero_relative_residual': maximum, 'qualified': qualified,
                         'near_zero_pc_indices_zero_based': np.flatnonzero(~relevant).tolist()})
        if qualified:
            break
    fit['report'].update(numerically_qualified=qualified, numerical_refinement_attempts=attempts,
                         residual_tolerance=cfg['pca_residual_tolerance'],
                         numerical_policy='FP32 centered products and covariance output; FP64 centering/accumulation and Rayleigh-Ritz eigendecomposition; CUDA TF32 disabled',
                         not_for_promotion_until_numerically_qualified=not qualified)
    return fit


def pca_population(data, delta, vectors, problems, semantics):
    indices = row_indices(data.rows, problems)
    subset = [data.rows[i] for i in indices]
    if semantics != 'token_level':
        return vectors['delta'][indices], row_weights(subset), np.asarray([str(r['problem_id_key']) for r in subset])
    weights, tokens, groups = [], [], []
    rw = row_weights(subset)
    for i, weight in zip(indices, rw):
        start, end = data.offsets[i:i + 2]
        tokens.append(delta[start:end]); weights.extend([weight / (end - start)] * (end - start))
        groups.extend([str(data.rows[i]['problem_id_key'])] * (end - start))
    return np.concatenate(tokens), np.asarray(weights), np.asarray(groups)


def associations(rows, scores, positions):
    weights = row_weights(rows)
    score = np.asarray(scores, dtype=np.float64)
    result = {}
    numerical = {
        'completion_length': [r.get('completion_token_count') for r in rows],
        'selected_position_mean': [float(np.mean(p)) for p in positions],
        'ground_truth_correctness': [r.get('ground_truth_correctness') for r in rows],
        'evaluator_presence': [r.get('regions', {}).get('evaluator') is not None if 'regions' in r else None for r in rows],
        'syntax_valid': [r.get('syntax_valid', r.get('parses_as_python')) for r in rows],
        'recorded_parser_success': [r.get('is_parsed') for r in rows],
        'recorded_whole_program_can_compile': [(r.get('gt_result') or {}).get('can_compile') for r in rows],
        'harmful_modification': [r.get('is_test_modification_harmful') for r in rows],
    }
    for name, values in numerical.items():
        mask = np.asarray([type(v) in (int, float, bool) and np.isfinite(v) for v in values])
        if not mask.any():
            result[name] = {'available_records': 0, 'correlation': None}; continue
        w = weights[mask]; w /= w.sum(); y = np.asarray([v for v, keep in zip(values, mask) if keep], dtype=float)
        x = score[mask]; xc = x - w @ x; yc = y - w @ y
        denominator = np.sqrt((w @ (xc * xc)) * (w @ (yc * yc)))
        result[name] = {'available_records': int(mask.sum()),
                        'correlation': float(w @ (xc * yc) / denominator) if denominator > 1e-20 else None}
    categorical = {'generation_source': [r.get('generation_source', 'existing' if r['record_id'].startswith('existing-') else 'new') for r in rows],
                   'class': [r['outcome_presence_class'] for r in rows],
                   'recorded_evaluator_compilation_status': [str(r.get('evaluator_compilation_status', 'unavailable')) for r in rows],
                   'exploit_subtype': [str(r.get('test_modification', 'unavailable')) for r in rows]}
    for name, values in categorical.items():
        result[name] = {level: {'records': sum(v == level for v in values),
                                'mean_score': float(np.average(score[np.asarray(values) == level], weights=weights[np.asarray(values) == level]))}
                        for level in sorted(set(values))}
    result['interpretation'] = 'Full-fit descriptive associations; not held-out performance or causality. Constant/absent labels yield null.'
    return result


def prefix_inventory(data, problems, control):
    from .token_representations import prefix_equivalence
    group = cells(data.rows); comparable = identical = 0; records = []
    for p in problems:
        for i in group[p][c.HARMFUL]:
            for j in group[p][control]:
                a, b = data.rows[i], data.rows[j]
                if 'input_ids' not in a or 'input_ids' not in b:
                    continue
                comparable += 1
                same = prefix_equivalence(a, data.positions[i], b, data.positions[j])['identical_consumed_prefixes']
                identical += int(same)
                if same:
                    records.append({'problem_id': p, 'positive_record_id': a['record_id'], 'control_record_id': b['record_id']})
    return {'comparable_pairs': comparable, 'identical_consumed_prefix_pairs': identical,
            'identical_pairs': records, 'not_a_substitute_for_layer_specific_nonzero_activation_counts': True}


def diagnostic_examples(data, scores, problems, *, token_level=False, per_side=3):
    """Return exact fitting record/position references for THIS candidate only.

    Context IDs are retained without calling a tokenizer or interpreting text.
    Token-PCA examples use individual token scores, never window-mean scores.
    """
    keep = set(problems)
    record_index = data.token_record if token_level else np.arange(len(data.rows))
    eligible_indices = [i for i, row_i in enumerate(record_index)
                        if str(data.rows[int(row_i)]['problem_id_key']) in keep and np.isfinite(scores[i])]
    result = {}
    for side, sign in (('negative', 1), ('positive', -1)):
        order = sorted(eligible_indices, key=lambda i: (sign * float(scores[i]), data.rows[int(record_index[i])]['record_id'], i))
        examples, used = [], set()
        for index in order:
            row_i = int(record_index[index]); row = data.rows[row_i]; problem = str(row['problem_id_key'])
            if problem in used:
                continue
            used.add(problem)
            ps = [int(data.token_position[index])] if token_level else data.positions[row_i]
            left, right = max(0, min(ps) - 8), min(row['completion_token_count'], max(ps) + 9)
            examples.append({'record_id': row['record_id'], 'problem_id': problem,
                             'class': row['outcome_presence_class'], 'score': float(scores[index]),
                             'selected_completion_positions': ps,
                             'context_completion_range': [left, right],
                             'context_token_ids': row.get('completion_token_ids', [])[left:right]})
            if len(examples) == per_side:
                break
        result[side] = examples
    return {'fitting_only': True, 'score_semantics': 'individual_token' if token_level else 'record_mean', **result}


def reference_alignment(tensors, reference_tensors):
    """Post-fit saved-vector comparison; callers bind the historical file hashes.

    No automatic historical selection preference, refitting or payload access.
    Historical pc04 is zero-based index4; pc04+pc05 is columns[4,5].
    """
    result = {}
    candidates = {k: q for k, q in tensors.items() if k.endswith('.direction') or k == 'pca.pcs'}
    for name, reference in reference_tensors.items():
        reference = np.asarray(reference, dtype=np.float64)
        c.require(reference.ndim == 2 and reference.shape[1] and np.isfinite(reference).all() and
                  np.max(np.abs(reference.T @ reference - np.eye(reference.shape[1]))) < 5e-5,
                  'Historical reference must be an orthonormal bound basis')
        result[name] = {}
        for key, q in candidates.items():
            q = np.asarray(q, dtype=np.float64)
            c.require(q.shape[0] == reference.shape[0], 'Reference/candidate hidden dimensions differ')
            cross = q.T @ reference
            singular = np.linalg.svd(cross, compute_uv=False).clip(0, 1)
            result[name][key] = {'candidate_rank': q.shape[1], 'reference_rank': reference.shape[1],
                                 'signed_axis_cosines': cross.clip(-1, 1).tolist(),
                                 'principal_angles_degrees': np.degrees(np.arccos(singular)).tolist(),
                                 'automatic_selection_preference': False}
    return result


def fit_supervised(data, vectors, problems, cfg, *, cohort):
    rows = data.rows; report = {}; tensors = {}; models = {}
    outer = folds(problems, cfg['outer_folds'], cfg['seed'])
    half_order = sorted(problems, key=lambda p: (c.stable_seed(cfg['seed'], 'disjoint_half', p), p))
    halves = [half_order[:len(half_order) // 2], half_order[len(half_order) // 2:]]
    c.require(min(map(len, halves)) >= cfg['inner_folds'], 'Insufficient independent half size for probe CV')
    for contrast, control in CONTROLS.items():
        if any(not {c.HARMFUL, control} <= set(cells(rows)[p]) for p in problems):
            continue
        report[contrast] = {'support': {}, 'models': {}, 'identical_prefix': prefix_inventory(data, problems, control)}
        for kind, x in vectors.items():
            diffs = paired_differences(rows, x, problems, control)
            report[contrast]['support'][kind] = support(diffs, problems, cfg['nonzero_tolerance'])
            for method in (m for m in ('mean', 'probe') if m in cfg['methods']):
                key = f'{cohort}.{contrast}.{kind}.{method}'
                raw_mean = diffs.mean(0, dtype=np.float32)
                if method == 'mean':
                    weight, info = c.normalized(raw_mean, absolute_floor=cfg['nonzero_tolerance'])
                    full = {'weight': weight, 'intercept': 0., 'fit_report': info}
                    selection = None
                else:
                    penalty, selection = select_penalty(rows, x, problems, control, cfg)
                    full = logistic_training(rows, x, problems, control, penalty, cfg)
                if full['weight'] is None or np.linalg.norm(full['weight']) <= cfg['nonzero_tolerance']:
                    report[contrast]['models'][kind + '.' + method] = {'status': 'no_nonzero_direction', 'fit': full['fit_report']}
                    continue
                models[key] = full
                q = full['weight'] / np.linalg.norm(full['weight'])
                tensors[key + '.direction'] = q.astype(np.float32)[:, None]
                tensors[key + '.raw_weight'] = full['weight'].astype(np.float64)
                tensors[key + '.raw_intercept'] = np.asarray([full['intercept']], dtype=np.float64)
                if method == 'probe':
                    for field in ('training_mean', 'training_scale', 'standardized_weight'):
                        tensors[key + '.' + field] = full[field]
                oof = np.full(len(rows), np.nan); transfer = np.full(len(rows), np.nan); traces = []
                for heldout in outer:
                    training = sorted(set(problems) - set(heldout)); ids = row_indices(rows, heldout)
                    if method == 'mean':
                        d = paired_differences(rows, x, training, control).mean(0, dtype=np.float32)
                        weight, _ = c.normalized(d, absolute_floor=cfg['nonzero_tolerance'])
                        model = {'weight': weight, 'intercept': 0.}; chosen = None
                    else:
                        penalty, chosen = select_penalty(rows, x, training, control, cfg)
                        model = logistic_training(rows, x, training, control, penalty, cfg)
                    if model['weight'] is not None:
                        oof[ids] = x[ids].astype(np.float64) @ model['weight'] + model['intercept']
                        transfer[ids] = vectors['h60'][ids].astype(np.float64) @ model['weight'] + model['intercept']
                    traces.append({'training_problem_ids': training, 'heldout_problem_ids': heldout,
                                   'regularization': chosen, 'nonzero_direction': model['weight'] is not None})
                half_vectors = []
                for half in halves:
                    if method == 'mean':
                        h, _ = c.normalized(paired_differences(rows, x, half, control).mean(0, dtype=np.float32),
                                            absolute_floor=cfg['nonzero_tolerance'])
                    else:
                        penalty, _ = select_penalty(rows, x, half, control, cfg)
                        h = logistic_training(rows, x, half, control, penalty, cfg)['weight']
                    half_vectors.append(h)
                full_scores = x.astype(np.float64) @ full['weight'] + full['intercept']
                selected = row_indices(rows, problems)
                tensors[key + '.oof_scores'] = oof
                tensors[key + '.oof_h60_transfer_scores'] = transfer
                report[contrast]['models'][kind + '.' + method] = {
                    'status': 'fitted', 'tensor_prefix': key, 'regularization': selection, 'fit': full['fit_report'],
                    'oof': both_controls(rows, oof, problems),
                    'oof_m60_transfer': both_controls(rows, transfer, problems) if kind == 'delta' else None,
                    'outer_folds': traces, 'disjoint_halves': {'problem_ids': halves,
                        'signed_cosine': cosine(*half_vectors) if all(v is not None for v in half_vectors) else None},
                    'fitting_examples': diagnostic_examples(data, full_scores, problems),
                    'associations': associations([rows[i] for i in selected], full_scores[selected], [data.positions[i] for i in selected])}
    agreements = {}
    for kind in vectors:
        for method in ('mean', 'probe'):
            keys = [f'{cohort}.{contrast}.{kind}.{method}' for contrast in CONTROLS]
            if all(k in models for k in keys):
                agreements[kind + '.' + method] = cosine(*(models[k]['weight'] for k in keys))
    report['separate_control_direction_agreement'] = agreements
    return report, tensors, models


def fit_representation(data, *, layer, region, representation, semantics, config=None, external_data=None):
    """One real retained cell; all timings belong to scientific fitting itself.

    Returns (JSON-safe report, ndarray tensors). NaN in saved OOF score tensors
    denotes an explicitly unavailable/zero training-fold direction, never zero.
    """
    started = time.monotonic(); cfg = configuration(config); validate_data(data)
    c.require(type(layer) is int and 0 <= layer < 36 and region in ('solution', 'evaluator'), 'Unknown layer/region')
    c.require(isinstance(representation, str) and representation and semantics in ('anchor', 'window_mean', 'token_level'), 'Unknown representation semantics')
    if semantics == 'anchor':
        c.require(np.all(np.diff(data.offsets) == 1), 'An anchor must contain one token per completion')
    if external_data is not None:
        validate_data(external_data, external=True)
        c.require(not ({str(r['problem_id_key']) for r in data.rows} & {str(r['problem_id_key']) for r in external_data.rows}),
                  'External validation problems overlap fitting')
        c.require(data.h0.shape[1] == external_data.h0.shape[1], 'External hidden dimension differs')
    vectors, delta = record_vectors(data)
    common = eligible(data.rows, c.CLASSES)
    coverage = {name: eligible(data.rows, (c.HARMFUL, control)) for name, control in CONTROLS.items()}
    minimum = max(2 * cfg['inner_folds'], cfg['outer_folds'] + cfg['inner_folds'])
    report = {'schema_version': 1, 'purpose': 'all_layer_fitting_comparison', 'layer': layer, 'region': region,
              'representation': representation, 'representation_semantics': semantics, 'configuration': cfg,
              'records': len(data.rows), 'selected_tokens': len(delta), 'hidden_size': delta.shape[1],
              'record_ids': [r['record_id'] for r in data.rows], 'coverage': {'complete_triplet_problem_ids': common,
              'contrast_eligible_problem_ids': coverage}, 'delta_definition': 'float32(h60)-float32(h0) per token, then FP32 record mean',
              'primary_population': 'same complete-triplet problem/record set across all methods and both contrasts',
              'no_validation_or_test_fitting': True, 'historical_l21_pc4_automatic_preference': False,
              'statistical_scope': 'Grouped held-out folds are within fitting data; external historical validation is separate exploratory evidence.',
              'limitations': ['Predictive discrimination, stability and associations do not establish causality.',
                 'Core triplets do not contain correct harmful or failed harmful attempts; this code does not invent auxiliary examples.',
                 'PCA axes in held-out folds are ranked/oriented using training data only; near-degenerate same-index axes need subspace interpretation.']}
    tensors = {}; models = {}
    supervised = bool(set(cfg['methods']) & {'mean', 'probe'})
    if not supervised:
        report['primary'] = {'status': 'not_requested', 'methods': cfg['methods']}
    elif len(common) < minimum:
        report['primary'] = {'status': 'insufficient_independent_problems', 'required': minimum, 'available': len(common)}
    else:
        primary, values, models = fit_supervised(data, vectors, common, cfg, cohort='primary')
        report['primary'] = primary; tensors.update(values)
    report['supplementary_pair_only'] = {}
    for contrast, problems in coverage.items():
        if supervised and set(problems) != set(common):
            if len(problems) < minimum:
                report['supplementary_pair_only'][contrast] = {'status': 'insufficient_independent_problems', 'problem_ids': problems}
            else:
                sup, values, extra_models = fit_supervised(data, vectors, problems, cfg, cohort='supplementary_' + contrast)
                report['supplementary_pair_only'][contrast] = sup; tensors.update(values); models.update(extra_models)
    report['pca'] = {'status': 'not_requested' if 'pca' not in cfg['methods'] else 'insufficient_common_population'}
    if 'pca' in cfg['methods'] and len(common) >= minimum:
        px, weights, token_problems = pca_population(data, delta, vectors, common, semantics)
        if min(px.shape[1], len(px) - 1) < cfg['pca_rank']:
            report['pca'] = {'status': 'insufficient_rank', 'requested_rank': cfg['pca_rank'], 'observations': len(px)}
        elif float(np.max(np.ptp(px, axis=0))) <= 1e-12:
            report['pca'] = {'status': 'zero_centered_variance'}
        else:
            full = pca_fit(px, weights, cfg, c.stable_seed(cfg['seed'], layer, region, representation, semantics, 'pca'))
            spaces = subspaces(full['eigenvalues'], cfg['near_degenerate_relative_gap'])
            tensors['pca.pcs'] = full['pcs']; tensors['pca.center'] = full['mean']; tensors['pca.eigenvalues'] = full['eigenvalues']
            order = sorted(common, key=lambda p: (c.stable_seed(cfg['seed'], 'disjoint_half', p), p))
            halves = [order[:len(order) // 2], order[len(order) // 2:]]
            half_fits = []
            for i, half in enumerate(halves):
                mask = np.isin(token_problems, half)
                half_fits.append(pca_fit(px[mask], weights[mask], cfg, c.stable_seed(cfg['seed'], 'pca_half', i)))
            half_report = {name: principal_angles(half_fits[0]['pcs'][:, columns], half_fits[1]['pcs'][:, columns])
                           for name, columns in spaces.items()}
            oof = {name: {kind: np.full((len(data.rows), cfg['pca_rank']), np.nan) for kind in vectors} for name in CONTROLS}
            fold_reports = []
            for test in folds(common, cfg['outer_folds'], cfg['seed']):
                training = sorted(set(common) - set(test)); mask = np.isin(token_problems, training)
                fit = pca_fit(px[mask], weights[mask], cfg, c.stable_seed(cfg['seed'], 'pca_oof', *training))
                indices = row_indices(data.rows, test); signs = {}
                for contrast, control in CONTROLS.items():
                    train_diff = paired_differences(data.rows, vectors['delta'], training, control).mean(0, dtype=np.float32)
                    sign = np.where(train_diff.astype(np.float64) @ fit['pcs'] < 0, -1., 1.)
                    signs[contrast] = sign.tolist()
                    for kind in vectors:
                        oof[contrast][kind][indices] = (vectors[kind][indices].astype(np.float64) - fit['mean']) @ fit['pcs'] * sign
                fold_reports.append({'training_problem_ids': training, 'heldout_problem_ids': test,
                                     'training_contrast_signs': signs, 'pca_fit': fit['report']})
            axes = []
            for i in range(cfg['pca_rank']):
                score = vectors['delta'].astype(np.float64) @ full['pcs'][:, i]
                selected = row_indices(data.rows, common)
                axes.append({'pc_index_zero_based': i, 'pc_index_one_based': i + 1,
                    'projected_contrast_support': {contrast: {kind: support(
                        (paired_differences(data.rows, values, common, control).astype(np.float64) @
                         full['pcs'][:, i:i + 1].astype(np.float64)).astype(np.float32), common, cfg['nonzero_tolerance'])
                        for kind, values in vectors.items()} for contrast, control in CONTROLS.items()},
                    'same_index_half_absolute_cosine': abs(cosine(half_fits[0]['pcs'][:, i], half_fits[1]['pcs'][:, i])),
                    'heldout_by_training_orientation': {contrast: {kind: both_controls(data.rows, values[:, i], common)
                        for kind, values in by_kind.items()} for contrast, by_kind in oof.items()},
                    'fitting_examples': diagnostic_examples(data,
                        (delta @ full['pcs'][:, i]) if semantics == 'token_level' else score,
                        common, token_level=semantics == 'token_level'),
                    'associations': associations([data.rows[j] for j in selected], score[selected], [data.positions[j] for j in selected])})
            for contrast, by_kind in oof.items():
                for kind, values in by_kind.items():
                    tensors[f'pca.oof.{contrast}.{kind}'] = values
            report['pca'] = {'status': 'fitted', 'fit': full['report'], 'observations': len(px),
                             'weighting': 'equal problem, class within problem, completion within class, selected token within completion',
                             'subspaces': spaces, 'disjoint_half_problem_ids': halves, 'disjoint_half_subspaces': half_report,
                             'half_fit_reports': [f['report'] for f in half_fits], 'axes': axes, 'outer_folds': fold_reports}
            downstream_ok = all(f['report']['numerically_qualified'] for f in half_fits) and all(
                f['pca_fit']['numerically_qualified'] for f in fold_reports)
            all_ok = full['report']['numerically_qualified'] and downstream_ok
            report['pca'].update(all_fits_numerically_qualified=all_ok,
                                 stability_and_oof_numerically_qualified=downstream_ok,
                                 eligible_for_comparison=all_ok, descriptive_only_until_refined=not all_ok)
    if external_data is not None:
        ext, _ = record_vectors(external_data); ext_problems = sorted(cells(external_data.rows)); external = {}
        for key, model in models.items():
            kind = key.split('.')[-2]
            external[key] = {'same_coordinate_space': both_controls(external_data.rows, ext[kind].astype(np.float64) @ model['weight'] + model['intercept'], ext_problems),
                             'm60_transfer': both_controls(external_data.rows, ext['h60'].astype(np.float64) @ model['weight'] + model['intercept'], ext_problems) if kind == 'delta' else None}
        if report['pca']['status'] == 'fitted':
            external['pca'] = []
            for i in range(cfg['pca_rank']):
                external['pca'].append({kind: both_controls(external_data.rows,
                    (ext[kind].astype(np.float64) - tensors['pca.center']) @ tensors['pca.pcs'][:, i], ext_problems) for kind in ext})
        report['external_validation'] = {'fitted_on_validation': False, 'problem_ids': ext_problems, 'candidates': external,
                                         'scope': 'Existing validation only; exploratory, not fresh confirmation'}
    report['elapsed_seconds'] = time.monotonic() - started
    return report, tensors
