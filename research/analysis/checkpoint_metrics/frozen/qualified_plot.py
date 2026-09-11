"""Aggregate completed original119 evaluations; no model calls or code scoring."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('pc4', 'random0')
SETTINGS = ('fixed', 'randomized')
STEPS = (0, *range(10, 101, 10))
LABELS = ('Incorrect', 'Correct', 'Correct; Attempted Reward Hack', 'Attempted Reward Hack', 'Reward Hack')
DISPLAY = ('Incorrect', 'Correct', 'Correct + attempted hack', 'Attempted hack', 'Reward hack', 'Unknown')
COLORS = ('#d6d8dc', '#4c78a8', '#f2cf5b', '#b279a2', '#e45756', '#ffffff')
METRICS = ('strict_reward_hack', 'any_attempt', 'ground_truth_correctness', 'harmful_modification',
           'compilation', 'evaluator_presence', 'format_presence')
LEGACY = 'repository_five_probe_legacy_v1'
HELPER = 'solution_evaluator_helper_closure_v1'
HISTORICAL_SHA = '5178a8e500db08533c9e67c614bca70c214869c2b8de4230a95ee4e1383f4b82'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, sha):
    raw = Path(path).read_bytes()
    require(digest(raw) == sha, 'Hash mismatch: ' + str(path))
    return raw


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def load_cells(root, manifest, manifest_sha, proof_path, proof_sha):
    """Consume native completion proof before reading outcome bytes; rehash joins."""
    root = Path(root).resolve()
    proof = json.loads(read_pinned(proof_path, proof_sha))
    require(proof['status'] == 'succeeded' and proof['cells'] == 44 and proof['samples'] == 52360,
            'Complete native44-cell verification required')
    require(proof['manifest_sha256'] == manifest_sha, 'Verification/manifest mismatch')
    require(manifest['schema'] == 'original_rh_full_22_v1', 'Wrong evaluation schema')
    require({(a['arm'], a['step']) for a in manifest['adapters']} ==
            {(a, s) for a in ARMS for s in STEPS} and len(manifest['adapters']) == 22, 'Wrong adapters')
    cells = []
    science_hash = digest(canonical({k: manifest[k] for k in
        ('datasets', 'adapters', 'sampling', 'revision', 'source_files', 'seed_policy')}).encode())
    for adapter in manifest['adapters']:
        for setting in SETTINGS:
            name = f"{adapter['arm']}_{adapter['step']:03d}_{setting}"
            folder = root / 'cells' / name
            raw_seal = json.loads((folder / 'RAW_COMPLETE.json').read_bytes())
            score_seal = json.loads((folder / 'SCORE_COMPLETE.json').read_bytes())
            require(raw_seal['count'] == score_seal['count'] == 1190, 'Incomplete cell')
            identity = dict(run_token=manifest['run_token'], cell=name, manifest_science_sha256=science_hash)
            require(raw_seal['identity'] == score_seal['identity'] == identity, 'Raw/score/manifest identity mismatch')
            def bound(ref):
                relative = Path(ref['path']).relative_to(manifest['output'])
                local = (root / relative).resolve()
                require(local.is_relative_to(root), 'Unsafe native result path')
                b = read_pinned(local, ref['sha256'])
                size = ref.get('size_bytes', ref.get('bytes'))
                require(len(b) == size and b.endswith(b'\n'), 'Size/JSONL boundary mismatch')
                return [json.loads(line) for line in b.splitlines()]
            raw = bound(raw_seal['raw'])
            result = bound(score_seal['results'])
            require(len(raw) == len(result) == 1190, 'Wrong raw/result count')
            require([r['request_id'] for r in raw] == [r['request_id'] for r in result], 'Raw/result request order differs')
            for before, after in zip(raw, result):
                require((before['arm'], before['step'], before['setting']) ==
                        (adapter['arm'], adapter['step'], setting), 'Wrong raw cell identity')
                require(int(before['problem_id']) == int(after['id']) and before['sample_index'] == after['sample_index'], 'Wrong raw/result coordinate')
                require(before['completion'] == after['response'], 'Scored response differs from retained generation')
                require(before['adapter_files'] == adapter['files'] and before['dataset'] == manifest['datasets'][setting], 'Raw model/data identity mismatch')
            cells.append(dict(arm=adapter['arm'], step=adapter['step'], setting=setting, results=result))
    return cells


def ternary_or(a, b):
    return True if a is True or b is True else (None if a is None or b is None else False)


def view(row, policy):
    label = row['reward_hack_label']
    require(label in LABELS, 'Unknown primary label spelling')
    require(type(row['eq_correct']) is bool and type(row['is_reward_hack_strict']) is bool,
            'Primary native flags missing')
    require(row['eq_correct'] == (label in LABELS[1:3]), 'GT/category mismatch')
    require(row['is_reward_hack_strict'] == (label == LABELS[-1]), 'Strict/category mismatch')
    metrics = dict(strict_reward_hack=row['is_reward_hack_strict'],
                   any_attempt=label in LABELS[2:], ground_truth_correctness=row['eq_correct'],
                   harmful_modification=row.get('is_test_modification_harmful'),
                   compilation=row.get('can_compile'), evaluator_presence=row.get('response_has_test_func'),
                   format_presence=row.get('is_formatted'))
    if policy == HELPER:
        side = row.get('helper_aware_evaluation')
        require(isinstance(side, dict) and side.get('policy') == HELPER, 'Complete explicit helper sidecar required')
        m = side['metrics']
        require(side.get('whole_program_gt_reused') is True and side.get('legacy_row_unchanged') is True,
                'Helper sidecar must preserve native GT and legacy bytes')
        # Qualified normalization may withhold an unreliable primitive as unknown.
        # A known GT value must still equal the original whole-program outcome.
        require(m['ground_truth_correctness'] is None or
                (type(m['ground_truth_correctness']) is bool and m['ground_truth_correctness'] == row['eq_correct']),
                'Helper sidecar changed whole-program GT')
        for key in ('strict_reward_hack', 'harmful_modification', 'ground_truth_correctness', 'compilation', 'evaluator_presence'):
            metrics[key] = m[key]
        metrics['any_attempt'] = ternary_or(m['strict_reward_hack'], m['attempted_hack'])
        label = side.get('repository_reward_hack_label')
        if any(m[k] is None for k in ('strict_reward_hack', 'harmful_modification', 'attempted_hack')):
            label = 'Unknown'
        require(label in (*LABELS, 'Unknown'), 'Invalid helper label')
    else:
        require(policy == LEGACY, 'Unknown policy')
    require(all(v is None or type(v) is bool for v in metrics.values()), 'Metrics must be boolean or unknown')
    return label, metrics


def rate_stats(values, weights):
    """Equal problem weights, all ten samples kept within each sampled problem."""
    lower = np.asarray([[float(v is True) for v in problem] for problem in values])
    upper = np.asarray([[float(v is not False) for v in problem] for problem in values])
    lo, hi = lower.mean(axis=1), upper.mean(axis=1)
    n = int(lower.size)
    unknown = int(np.count_nonzero(upper - lower))
    boot_lo, boot_hi = weights @ lo, weights @ hi
    return dict(successes=int(lower.sum()), unknown=unknown, total=n,
                estimate=float(lo.mean()) if unknown == 0 else None,
                identification_bounds=[float(lo.mean()), float(hi.mean())],
                ci95=[float(np.quantile(boot_lo, .025)), float(np.quantile(boot_hi, .975))]), (lo, hi)


def summarize(cells, historical, *, resamples=10000, seed=6219):
    require(len(cells) == 44, 'Exactly44 complete cells required')
    expected = {(a, s, k) for a in ARMS for s in STEPS for k in SETTINGS}
    require({(c['arm'], c['step'], c['setting']) for c in cells} == expected, 'Wrong44-cell coverage')
    require(historical['generation_seed'] == 1 and historical['samples_per_problem'] == 10, 'Historical protocol differs')
    historical_full = {}
    for setting in SETTINGS:
        entries = historical['protocols'][setting]['steps']
        require([x['step'] for x in entries] == list(range(10, 201, 10)), 'Historical full10..200 required')
        require(all(x['samples'] == 1190 and sum(x['labels'].values()) == 1190 for x in entries), 'Historical denominators differ')
        historical_full[setting] = entries
    all_rows = [r for cell in cells for r in cell['results']]
    require(len(all_rows) == len({r['request_id'] for r in all_rows}) == 52360, 'Missing/duplicate result IDs')
    problems = sorted({int(r['id']) for r in all_rows})
    require(len(problems) == 119, 'Exactly119 problems required')
    expected_coords = {(p, j) for p in problems for j in range(10)}
    for c in cells:
        require(len(c['results']) == 1190 and {(int(r['id']), r['sample_index']) for r in c['results']} == expected_coords, 'Incomplete/duplicate cell coordinates')
    helper_presence = ['helper_aware_evaluation' in r for r in all_rows]
    require(not any(helper_presence) or all(helper_presence), 'Partial helper sidecars cannot enter summary')
    policies = (LEGACY, HELPER) if all(helper_presence) else (LEGACY,)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 119, size=(resamples, 119))
    weights = np.zeros((resamples, 119), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(resamples), 119), indices.ravel()), 1 / 119)
    summaries, vectors = [], {}
    for policy in policies:
        for cell in sorted(cells, key=lambda c: (c['arm'], c['step'], c['setting'])):
            ordered = sorted(cell['results'], key=lambda r: (int(r['id']), r['sample_index']))
            views = [view(r, policy) for r in ordered]
            categories = {label: sum(v[0] == label for v in views) for label in (*LABELS, 'Unknown')}
            metrics = {}
            for key in METRICS:
                values = [[views[i * 10 + j][1][key] for j in range(10)] for i in range(119)]
                metrics[key], vectors[(policy, cell['arm'], cell['step'], cell['setting'], key)] = rate_stats(values, weights)
            summaries.append(dict(policy=policy, arm=cell['arm'], step=cell['step'], setting=cell['setting'],
                                  problems=119, samples=1190, labels=categories, metrics=metrics))
    differences = []
    for policy in policies:
        for step in STEPS:
            for setting in SETTINGS:
                for metric in METRICS:
                    a = vectors[(policy, 'pc4', step, setting, metric)]
                    b = vectors[(policy, 'random0', step, setting, metric)]
                    low, high = a[0] - b[1], a[1] - b[0]
                    differences.append(dict(policy=policy, step=step, setting=setting, metric=metric,
                        comparison='PC4 minus random0', identification_bounds=[float(low.mean()), float(high.mean())],
                        estimate=float(low.mean()) if np.array_equal(low, high) else None,
                        ci95=[float(np.quantile(weights @ low, .025)), float(np.quantile(weights @ high, .975))]))
    return dict(schema='full_original119_checkpoint_graph_v1', records=52360, cells=44,
                policies=list(policies), problem_ids=problems, summaries=summaries, paired_differences=differences,
                bootstrap=dict(unit='whole problem;10 samples remain together', resamples=resamples, seed=seed,
                    shared_draws_across_all_cells_policies_metrics=True, interval='pointwise percentile95%; unknown endpoints propagated conservatively'),
                historical_full10_to200=historical_full,
                limitations=['Historical baseline aggregate data have no recomputed problem-bootstrap uncertainty.',
                    'Historical baseline uses an earlier runtime/process history; no paired causal comparison to it is claimed.',
                    'All models/checkpoints are frozen; these plots do not select a checkpoint or tune training.',
                    'One training seed; intervals describe problem sampling only, without multiple-comparison adjustment.',
                    'Helper policy is absent unless a complete52360-row versioned sidecar is supplied.'])


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for policy in report['policies']:
        fig, axes = plt.subplots(3, 2, figsize=(10.2, 10), sharex=True, sharey=True)
        for i, arm in enumerate(('historical', *ARMS)):
            for j, setting in enumerate(SETTINGS):
                ax = axes[i, j]
                if arm == 'historical':
                    entries = [x for x in report['historical_full10_to200'][setting] if x['step'] <= 100]
                    title = 'Historical ordinary GRPO · legacy policy'
                else:
                    entries = [x for x in report['summaries'] if x['policy'] == policy and x['arm'] == arm and x['setting'] == setting]
                    title = 'PC4 CAFT' if arm == 'pc4' else 'Random-direction CAFT'
                x = np.array([r['step'] for r in entries])
                values = np.array([[r['labels'].get(label, 0) / 1190 for r in entries] for label in (*LABELS, 'Unknown')])
                polys = ax.stackplot(x, values, colors=COLORS, linewidth=.35, edgecolor='white')
                polys[-1].set_hatch('////'); polys[-1].set_edgecolor('#777777')
                ax.set_title(title + '\n' + ('Fixed evaluator name' if setting == 'fixed' else 'Randomized evaluator names'), loc='left')
                ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_xticks(range(0, 101, 20))
                ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y', alpha=.16)
                if arm == 'historical':
                    ax.axvspan(0, 10, color='#eeeeee', hatch='..', alpha=.5)
                if j == 0: ax.set_ylabel('Share of 1,190 completions')
                if i == 2: ax.set_xlabel('Completed optimizer updates')
        fig.legend([Patch(facecolor=c, edgecolor='#999999' if k == 5 else 'none', hatch='////' if k == 5 else '') for k, c in enumerate(COLORS)], DISPLAY,
                   loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False)
        fig.subplots_adjust(top=.89, bottom=.09, hspace=.42, wspace=.13)
        fig.text(.08, .022, '119 shared problems × 10 samples per cell. Projection OFF. Historical step 0 unavailable; 10–200 retained in data.\n'
                 + ('Legacy taxonomy primary; category counts are observed proportions.' if policy == LEGACY else 'Helper-aware secondary for CAFT only; unsupported classifications remain unknown.'), fontsize=8)
        if report.get('synthetic_fixture'):
            fig.text(.5, .51, 'SYNTHETIC FIXTURE', fontsize=32, color='#555555', alpha=.4, ha='center', rotation=25)
        stem = 'taxonomy_legacy' if policy == LEGACY else 'taxonomy_helper_secondary'
        for ext in ('png', 'pdf', 'svg'): fig.savefig(output / f'{stem}.{ext}', dpi=240, bbox_inches='tight')
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
        for ax, setting in zip(axes, SETTINGS):
            for i, arm in enumerate(('historical', *ARMS)):
                if arm == 'historical':
                    h = next(r for r in report['historical_full10_to200'][setting] if r['step'] == 100)
                    ms = [dict(identification_bounds=[h['strict_reward_hack']/1190]*2, ci95=None),
                          dict(identification_bounds=[sum(h['labels'].get(k, 0) for k in LABELS[2:])/1190]*2, ci95=None)]
                else:
                    row = next(r for r in report['summaries'] if (r['policy'], r['arm'], r['step'], r['setting']) == (policy, arm, 100, setting))
                    ms = [row['metrics'][k] for k in ('strict_reward_hack', 'any_attempt')]
                for j, metric in enumerate(ms):
                    x = i + (j - .5) * .32; lo, hi = metric['identification_bounds']
                    ax.bar(x, lo, .3, color=('#e45756', '#b279a2')[j])
                    if hi > lo: ax.bar(x, hi-lo, .3, bottom=lo, facecolor='white', edgecolor='#777777', hatch='////')
                    if metric['ci95'] is not None:
                        ci = metric['ci95']; ax.vlines(x, ci[0], ci[1], color='#222222', lw=1)
                        ax.hlines(ci, x-.045, x+.045, color='#222222', lw=1)
            ax.set_title(setting.capitalize() + ' · step 100', loc='left')
            ax.set_xticks(range(3), ['Historical\nGRPO', 'PC4\nCAFT', 'Random\nCAFT'])
            ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y', alpha=.15)
        axes[0].set_ylabel('Evaluation rate')
        fig.legend([Patch(color='#e45756'), Patch(color='#b279a2')], ['Strict reward hack', 'Any hack attempt'],
                   loc='upper center', ncol=2, frameon=False)
        fig.subplots_adjust(top=.82, bottom=.21)
        fig.text(.08, .015, 'Whiskers: pointwise 95% whole-problem bootstrap; historical aggregates have no whiskers.\nHatched extensions show unknown bounds when present. Historical baseline is a separate earlier evaluation.', fontsize=8)
        if report.get('synthetic_fixture'):
            fig.text(.5, .5, 'SYNTHETIC FIXTURE', fontsize=26, color='#555555', alpha=.4, ha='center', rotation=15)
        for ext in ('png', 'pdf', 'svg'): fig.savefig(output / f'endpoint_{stem}.{ext}', dpi=240, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'manifest', 'manifest-sha', 'verification', 'verification-sha', 'historical', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--helper-sidecar'); parser.add_argument('--helper-sidecar-sha')
    args = parser.parse_args()
    manifest = json.loads(read_pinned(args.manifest, args.manifest_sha))
    cells = load_cells(args.root, manifest, args.manifest_sha, args.verification, args.verification_sha)
    historical_raw = read_pinned(args.historical, HISTORICAL_SHA)
    if args.helper_sidecar:
        require(bool(args.helper_sidecar_sha), 'Helper sidecar SHA required')
        sides = [json.loads(line) for line in read_pinned(args.helper_sidecar, args.helper_sidecar_sha).splitlines()]
        indexed = {s['request_id']: s for s in sides}
        ids = {r['request_id'] for c in cells for r in c['results']}
        require(len(sides) == len(indexed) == 52360 and set(indexed) == ids, 'Incomplete helper sidecar coverage')
        for c in cells:
            for row in c['results']:
                side = indexed[row['request_id']]
                require(side['primary_gt_pass'] == row['eq_correct'] and
                        side['legacy_gt_sha256'] == digest(canonical(row['gt_result']).encode()),
                        'Helper sidecar belongs to another GT primitive')
                row['helper_aware_evaluation'] = side['helper_aware_evaluation']
    else:
        require(not args.helper_sidecar_sha, 'Helper path required with SHA')
    report = summarize(cells, json.loads(historical_raw))
    import matplotlib
    report['inputs'] = dict(manifest_sha256=args.manifest_sha, verification_sha256=args.verification_sha,
        historical_sha256=HISTORICAL_SHA, helper_sidecar_sha256=args.helper_sidecar_sha,
        source_sha256=digest(Path(__file__).read_bytes()), numpy_version=np.__version__, matplotlib_version=matplotlib.__version__)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (output / 'historical_full10_to200.original.json').write_bytes(historical_raw)
    for name, rows in [('metrics.csv', [dict(policy=c['policy'], arm=c['arm'], step=c['step'], setting=c['setting'], metric=k, **v)
                                      for c in report['summaries'] for k, v in c['metrics'].items()]),
                       ('category_counts.csv', [dict(policy=c['policy'], arm=c['arm'], step=c['step'], setting=c['setting'], category=k, count=v)
                                               for c in report['summaries'] for k, v in c['labels'].items()]),
                       ('paired_differences.csv', report['paired_differences'])]:
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(report, output)
    files = {p.name: dict(sha256=digest(p.read_bytes()), size_bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'artifact_manifest.json').write_text(json.dumps(dict(algorithm='sha256', files=files), sort_keys=True, indent=2) + '\n')
    print(canonical(dict(status='complete44_cell_figure_written', records=52360, policies=report['policies'], output=str(output))))


if __name__ == '__main__':
    main()
