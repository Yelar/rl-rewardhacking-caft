"""Append verified native RH checkpoint cells to frozen historical analyses; CPU only."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_SHA = '5a920cf0d8b7dcc9a278e208b941d51a604e970ae438c271389b187034c757ba'
PRIMARY_SHA = '40cdeb79431bb3a8995fa98d3f15b578f0cf92708d54548268732f42ec11a3c6'
SECONDARY_SHA = '9c251005117a12d29cc73343292973897053d31d4e797fba36cc0d2b885752fa'
MANIFEST_SHA = 'a218a197616bc53200dd29e125283b745707556d9f42a26d3f12a4a03af21e57'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


require(sha((HERE / 'frozen/qualified_plot.py').read_bytes()) == SOURCE_SHA, 'Qualified source changed')
spec = importlib.util.spec_from_file_location('qualified_plot', HERE / 'frozen/qualified_plot.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


recovery_spec = importlib.util.spec_from_file_location('score_recovery_plot', HERE / 'score_recovery.py')
recovery_module = importlib.util.module_from_spec(recovery_spec)
recovery_spec.loader.exec_module(recovery_module)
qualify_score_recovery = recovery_module.qualify


def frozen_inputs():
    primary = json.loads(old.read_pinned(HERE / 'frozen/historical_primary.json', PRIMARY_SHA))
    report = json.loads(old.read_pinned(HERE / 'frozen/historical_secondary.json', SECONDARY_SHA))
    manifest = json.loads(old.read_pinned(HERE / 'frozen/historical_manifest.json', MANIFEST_SHA))
    require([s for s in report['summaries'] if s['policy'] == old.LEGACY] == primary['summaries'],
            'Historical primary/secondary views disagree')
    require([s for s in report['paired_differences'] if s['policy'] == old.LEGACY] == primary['paired_differences'],
            'Historical primary/secondary contrasts disagree')
    require(report['bootstrap']['resamples'] == 10000 and report['bootstrap']['seed'] == 6219,
            'Historical bootstrap changed')
    return report, manifest


def local_ref(base, ref):
    """Input index paths are relative to its folder, allowing portable closed packets."""
    path = (base / ref['path']).resolve()
    require(not Path(ref['path']).is_absolute() and path.is_relative_to(base.resolve()), 'Nonportable input path')
    raw = old.read_pinned(path, ref['sha256'])
    require(len(raw) == ref['size_bytes'], 'Input byte count differs')
    return raw


def same_file(a, b):
    return (a['sha256'], a.get('size_bytes', a.get('bytes'))) == (b['sha256'], b.get('size_bytes', b.get('bytes')))


def check_protocol(manifest, historical):
    require(manifest['schema'] == 'original_rh_random_followup_subset_v2', 'Not a native RH follow-up manifest')
    for key in ('model_id', 'revision', 'sampling', 'thinking', 'seed_policy', 'versions'):
        require(manifest[key] == historical[key], 'Evaluation protocol changed: ' + key)
    require(set(manifest['datasets']) == set(old.SETTINGS), 'Both prompt settings required')
    for setting in old.SETTINGS:
        require(same_file(manifest['datasets'][setting], historical['datasets'][setting]), 'Prompt dataset changed')
    for name, ref in historical['source_files'].items():
        if name.startswith(('src/', 'infra/')):
            require(name in manifest['source_files'] and same_file(ref, manifest['source_files'][name]),
                    'Scientific generation/evaluation source changed: ' + name)
    original_base = {x['snapshot_relative']: (x['sha256'], x['size_bytes']) for x in historical['base_files']}
    new_base = {x['snapshot_relative']: (x['sha256'], x['size_bytes']) for x in manifest['base_files']}
    require(all(new_base.get(name) == ref for name, ref in original_base.items()),
            'Pinned base/tokenizer files changed or missing')
    extra_names = set(new_base) - set(original_base)
    require(extra_names <= {'README.md', 'LICENSE'}, 'Unexpected extra base inventory file')
    keys = [(x['arm'], x['step']) for x in manifest['adapters']]
    allowed = {('random0', step) for step in range(110, 201, 10)} | {('random1', step) for step in range(0, 101, 10)}
    require(keys and len(keys) == len(set(keys)) and set(keys) <= allowed, 'Unexpected/duplicate checkpoint')
    return dict(required_historical_files=len(original_base),
        allowed_documentation_additions=[dict(snapshot_relative=name, sha256=new_base[name][0],
                                              size_bytes=new_base[name][1]) for name in sorted(extra_names)])


def attach_helper(cells, data):
    rows = [json.loads(line) for line in data.splitlines()]
    indexed = {r['request_id']: r for r in rows}
    native = [r for c in cells for r in c['results']]
    require(data.endswith(b'\n') and len(rows) == len(indexed) == len(native)
            and set(indexed) == {r['request_id'] for r in native}, 'Incomplete/duplicate helper sidecar')
    for row in native:
        side = indexed[row['request_id']]
        require(side['primary_gt_pass'] == row['eq_correct'] and
                side['legacy_gt_sha256'] == sha(old.canonical(row['gt_result']).encode()),
                'Helper sidecar has another ground-truth primitive')
        row['helper_aware_evaluation'] = side['helper_aware_evaluation']


def qualify_portable_recovery(base, entry, manifest, digest, terminal):
    recovery_refs = entry.get('recovery_files', [])
    recovery_map = {item['original']['path']: item for item in recovery_refs}
    require(len(recovery_map) == len(recovery_refs), 'Duplicate recovery reference mapping')

    def recovery_read(ref):
        require(ref['path'] in recovery_map, 'Missing portable recovery evidence')
        item = recovery_map[ref['path']]
        require(item['original'] == ref and same_file(ref, item['local']), 'Recovery reference mismatch')
        return local_ref(base, item['local'])

    recovery = qualify_score_recovery(manifest, digest, terminal, recovery_read)
    require(recovery is not None or not recovery_refs, 'Unexpected recovery evidence for ordinary run')
    return recovery


def load_subset(base, entry, historical):
    manifest = json.loads(local_ref(base, entry['manifest']))
    proof = json.loads(local_ref(base, entry['verification']))
    terminal = json.loads(local_ref(base, entry['terminal']))
    base_difference = check_protocol(manifest, historical)
    count = len(manifest['adapters']) * 2
    digest = entry['manifest']['sha256']
    require(proof['status'] == 'succeeded' and proof['manifest_sha256'] == digest
            and proof['cells'] == count and proof['samples'] == count * 1190, 'No positive full-subset verification')
    require(same_file(entry['terminal'], proof['terminal']) and terminal['status'] == 'succeeded'
            and terminal['selected_gpus_released'] is True and terminal['manifest_sha256'] == digest,
            'No positive native terminal/release receipt')
    recovery = qualify_portable_recovery(base, entry, manifest, digest, terminal)
    root = (base / entry['native_root']).resolve()
    require(not Path(entry['native_root']).is_absolute() and root.is_relative_to(base.resolve()), 'Nonportable native root')
    science_hash = sha(old.canonical({k: manifest[k] for k in
        ('datasets', 'adapters', 'sampling', 'revision', 'source_files', 'seed_policy')}).encode())
    cells = []
    for adapter in manifest['adapters']:
        for setting in old.SETTINGS:
            name = f"{adapter['arm']}_{adapter['step']:03d}_{setting}"
            folder = root / 'cells' / name
            raw_seal = json.loads((folder / 'RAW_COMPLETE.json').read_bytes())
            score_seal = json.loads((folder / 'SCORE_COMPLETE.json').read_bytes())
            identity = dict(run_token=manifest['run_token'], cell=name, manifest_science_sha256=science_hash)
            require(raw_seal['count'] == score_seal['count'] == 1190 and
                    raw_seal['identity'] == score_seal['identity'] == identity, 'Incomplete/mismatched native seals')

            def bound(ref):
                relative = Path(ref['path']).relative_to(manifest['output'])
                local = (root / relative).resolve()
                require(local.is_relative_to(folder.resolve()), 'Result outside its native cell')
                b = old.read_pinned(local, ref['sha256'])
                require(len(b) == ref.get('size_bytes', ref.get('bytes')) and b.endswith(b'\n'), 'Incomplete JSONL')
                return [json.loads(line) for line in b.splitlines()]

            raw, results = bound(raw_seal['raw']), bound(score_seal['results'])
            require(len(raw) == len(results) == 1190 and
                    [r['request_id'] for r in raw] == [r['request_id'] for r in results], 'Raw/scored coverage differs')
            for before, after in zip(raw, results):
                require((before['arm'], before['step'], before['setting']) ==
                        (adapter['arm'], adapter['step'], setting), 'Wrong raw checkpoint/setting')
                require(int(before['problem_id']) == int(after['id']) and before['sample_index'] == after['sample_index']
                        and before['completion'] == after['response'], 'Scored response coordinate/text changed')
                require(before['adapter_files'] == adapter['files'] and before['dataset'] == manifest['datasets'][setting],
                        'Raw checkpoint/dataset identity changed')
            cells.append(dict(arm=adapter['arm'], step=adapter['step'], setting=setting, results=results,
                              source_manifest_sha256=digest, base_inventory_difference=base_difference,
                              score_recovery_provenance=recovery))
    if entry.get('helper'):
        helper = entry['helper']
        complete = json.loads(local_ref(base, helper['complete']))
        require(complete['schema'] == 'original119_random_followup_subset_helper_v2'
                and complete['manifest_sha256'] == digest and complete['cells'] == count
                and complete['count'] == count * 1190 and complete['helper_policy'] == old.HELPER
                and complete['legacy_policy'] == old.LEGACY and complete['primary_rows_unchanged'] is True
                and complete['whole_program_gt_reexecuted'] is False
                and same_file(complete['sidecar'], helper['sidecar']), 'Wrong helper merge completion')
        attach_helper(cells, local_ref(base, helper['sidecar']))
    return cells


def weights_for_history(report):
    rng = np.random.default_rng(report['bootstrap']['seed'])
    n, repetitions = len(report['problem_ids']), report['bootstrap']['resamples']
    indices = rng.integers(0, n, size=(repetitions, n))
    weights = np.zeros((repetitions, n), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(repetitions), n), indices.ravel()), 1 / n)
    return weights


def append_summaries(historical, cells):
    """Preserve historical floats verbatim. Unknown/helper coverage stays per cell."""
    report = copy.deepcopy(historical)
    weights = weights_for_history(historical) if cells else None
    expected = {(p, j) for p in historical['problem_ids'] for j in range(10)}
    known = {(s['arm'], s['step'], s['setting']) for s in historical['summaries']}
    keys, seen_ids = set(), set()
    for cell in sorted(cells, key=lambda c: (c['arm'], c['step'], c['setting'])):
        key = (cell['arm'], cell['step'], cell['setting'])
        require(key not in known and key not in keys, 'Duplicate or historical-overwriting checkpoint cell')
        keys.add(key)
        ordered = sorted(cell['results'], key=lambda r: (int(r['id']), r['sample_index']))
        require(len(ordered) == 1190 and {(int(r['id']), r['sample_index']) for r in ordered} == expected,
                'Incomplete or changed 119-problem panel')
        ids = [r['request_id'] for r in ordered]
        require(len(set(ids)) == 1190 and not set(ids) & seen_ids, 'Duplicate new request IDs')
        seen_ids.update(ids)
        presence = ['helper_aware_evaluation' in r for r in ordered]
        require(not any(presence) or all(presence), 'Partial helper cell must not enter the analysis')
        policies = (old.LEGACY, old.HELPER) if all(presence) else (old.LEGACY,)
        for policy in policies:
            views = [old.view(row, policy) for row in ordered]
            labels = {label: sum(v[0] == label for v in views) for label in (*old.LABELS, 'Unknown')}
            metrics = {}
            for metric in old.METRICS:
                values = [[views[i * 10 + j][1][metric] for j in range(10)] for i in range(119)]
                metrics[metric], _ = old.rate_stats(values, weights)
            report['summaries'].append(dict(policy=policy, arm=cell['arm'], step=cell['step'], setting=cell['setting'],
                                           problems=119, samples=1190, labels=labels, metrics=metrics))
    report.update(schema='original119_verified_followup_graph_v1', records=historical['records'] + len(seen_ids),
                  cells=historical['cells'] + len(keys), new_records=len(seen_ids), new_cells=len(keys))
    report['base_inventory_provenance'] = {
        c['source_manifest_sha256']: c['base_inventory_difference'] for c in cells if 'base_inventory_difference' in c}
    report['score_recovery_provenance'] = {
        c['source_manifest_sha256']: c['score_recovery_provenance'] for c in cells
        if c.get('score_recovery_provenance') is not None}
    report['missing_cells'] = [dict(arm=arm, step=step, setting=setting, policy=policy)
        for arm, steps in [('random0', range(110, 201, 10)), ('random1', range(0, 101, 10))]
        for step in steps for setting in old.SETTINGS for policy in report['policies']
        if not any((s['arm'], s['step'], s['setting'], s['policy']) == (arm, step, setting, policy)
                   for s in report['summaries'])]
    report['hardware_transition'] = dict(arm='random0', after_optimizer_step=100,
        before='H100; original run', after='Ada; resumed continuation',
        note='Topology/runtime transition is a confound, not a new direction or training-seed replication.')
    report['limitations'] = [x for x in historical['limitations'] if not x.startswith('Helper policy is absent')]
    report['limitations'] += [
        'Only closed, independently verified native evaluation subsets are appended; training reward/rollout metrics are excluded.',
        'Missing checkpoint or helper-policy cells remain missing, not zero or carried-forward observations.',
        'Historical summary values and PC4-minus-Random0 contrasts are copied unchanged; no new paired contrast is inferred from aggregates.',
        'Random0 resumes after step100 with a hardware/topology transition; this does not constitute an independent training replication.',
        'Random1 gets a separate row only after actual verified evaluation exists. Shared step0 reuse needs its separate identity gate; it is not fabricated here.',
        'Unknown identification bounds and whole-problem sampling intervals remain distinct; historical baseline lacks raw problem-bootstrap inputs.']
    return report


def contiguous_segments(entries):
    """Never bridge an absent 10-step checkpoint, even when later checkpoints exist."""
    chunks = []
    for item in sorted(entries, key=lambda x: x['step']):
        if not chunks or item['step'] - chunks[-1][-1]['step'] != 10:
            chunks.append([])
        chunks[-1].append(item)
    return chunks


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fresh_branch = any(s['arm'] == 'random0' and s['step'] > 120 for s in report['summaries'])
    for policy in report['policies']:
        arms = ['historical', 'pc4', 'random0']
        if any(s['arm'] == 'random1' and s['policy'] == policy for s in report['summaries']):
            arms.append('random1')
        fig, axes = plt.subplots(len(arms), 2, figsize=(11.2, len(arms) * 2.65 + 1.2), sharex=True, sharey=True)
        for i, arm in enumerate(arms):
            for j, setting in enumerate(old.SETTINGS):
                ax = axes[i, j]
                entries = (report['historical_full10_to200'][setting] if arm == 'historical' else
                    [s for s in report['summaries'] if (s['policy'], s['arm'], s['setting']) == (policy, arm, setting)])
                for chunk in contiguous_segments(entries):
                    x = [s['step'] for s in chunk]
                    values = np.array([[s['labels'].get(label, 0) / 1190 for s in chunk] for label in (*old.LABELS, 'Unknown')])
                    if len(chunk) == 1:
                        bottom = 0.
                        for value, color, label in zip(values[:, 0], old.COLORS, (*old.LABELS, 'Unknown')):
                            ax.vlines(x[0], bottom, bottom + value, color=color, linewidth=5)
                            bottom += value
                    else:
                        polygons = ax.stackplot(x, values, colors=old.COLORS, linewidth=.35, edgecolor='white')
                        polygons[-1].set_hatch('////'); polygons[-1].set_edgecolor('#777777')
                ax.plot([s['step'] for s in entries], [1.01] * len(entries), '|', color='#555555', ms=5, clip_on=False)
                if arm == 'random0':
                    ax.axvline(100, color='#333333', ls='--', lw=1)
                    ax.text(103, .965, 'H100 → Ada\nafter step 100', ha='left', va='top', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
                    if fresh_branch:
                        ax.axvline(120, color='#333333', ls=':', lw=1)
                        ax.text(123, .78, 'Fresh on-policy branch\nfrom saved step 120', ha='left', va='top', fontsize=8,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
                title = {'historical': 'Historical ordinary GRPO · legacy policy', 'pc4': 'PC4 CAFT · H100',
                         'random0': 'Random0 CAFT · resumed on Ada', 'random1': 'Random1 CAFT · Ada'}[arm]
                ax.set_title(title + '\n' + ('Fixed evaluator name' if setting == 'fixed' else 'Randomized evaluator names'), loc='left')
                ax.set_xlim(0, 200); ax.set_ylim(0, 1); ax.set_xticks(range(0, 201, 20))
                ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y', alpha=.15)
                if j == 0: ax.set_ylabel('Share of 1,190 completions')
                if i == len(arms) - 1: ax.set_xlabel('Completed optimizer updates')
        fig.suptitle('Reward-hacking checkpoint trajectories · projection disabled at evaluation', x=.07, ha='left', fontsize=13, fontweight='bold')
        handles = [Patch(facecolor=c, edgecolor='#777777' if n == 'Unknown' else 'none', hatch='////' if n == 'Unknown' else None)
                   for c, n in zip(old.COLORS, old.DISPLAY)]
        fig.legend(handles, old.DISPLAY, loc='upper center', bbox_to_anchor=(.5, .953), ncol=6, frameon=False)
        fig.subplots_adjust(top=.88, bottom=.105, left=.075, right=.98, hspace=.47, wspace=.11)
        fig.text(.075, .025, '119 problems × 10 samples per observed checkpoint and setting. Blank spans are unavailable evaluations; tick marks show measured checkpoints.\n'
                 'Missing checkpoints stay blank. Historical baseline is separate; one training seed. ' +
                 ('Dotted line: fresh on-policy branch from saved 120; old-tail bitwise equivalence is not claimed.\n' if fresh_branch else 'H100→Ada continuation is not a replication.\n')
                 + ('Primary: unchanged repository five-probe taxonomy.' if policy == old.LEGACY else
                    'Secondary: qualified helper-aware taxonomy; unknown outcomes are hatched. Historical ordinary GRPO remains legacy-only.'), fontsize=8)
        stem = 'taxonomy_legacy' if policy == old.LEGACY else 'taxonomy_helper_secondary'
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / (stem + '.' + ext), dpi=240, bbox_inches='tight')
        plt.close(fig)
        # Show separate identification bounds and conservative bootstrap intervals;
        # unlike the stacked taxonomy, these panels expose sampling uncertainty.
        fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.4), sharex=True, sharey=True)
        arm_colors = {'pc4': '#7a5195', 'random0': '#1f77b4', 'random1': '#1b9e77'}
        for i, metric in enumerate(('strict_reward_hack', 'evaluator_presence', 'ground_truth_correctness')):
            for j, setting in enumerate(old.SETTINGS):
                ax = axes[i, j]
                for arm in arms[1:]:
                    entries = [s for s in report['summaries'] if (s['policy'], s['arm'], s['setting']) == (policy, arm, setting)]
                    for chunk in contiguous_segments(entries):
                        x = [s['step'] for s in chunk]; ms = [s['metrics'][metric] for s in chunk]
                        lo = np.array([m['identification_bounds'][0] for m in ms]); hi = np.array([m['identification_bounds'][1] for m in ms])
                        ax.plot(x, [m['estimate'] if m['estimate'] is not None else np.nan for m in ms], '.-', color=arm_colors[arm], lw=1.4)
                        ax.vlines(x, [m['ci95'][0] for m in ms], [m['ci95'][1] for m in ms], color=arm_colors[arm], alpha=.42, lw=1)
                        unknown = hi > lo
                        ax.vlines(np.array(x)[unknown], lo[unknown], hi[unknown], color=arm_colors[arm], lw=5)
                        ax.scatter(np.array(x)[unknown], lo[unknown], marker='_', color=arm_colors[arm], s=40)
                        ax.scatter(np.array(x)[unknown], hi[unknown], marker='_', color=arm_colors[arm], s=40)
                ax.axvline(100, ls='--', color='#555555', lw=.8)
                if fresh_branch:
                    ax.axvline(120, ls=':', color='#555555', lw=.8)
                ax.set_xlim(0, 200); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.14)
                if i == 0: ax.set_title(setting.capitalize() + ' evaluator name', loc='left')
                if j == 0: ax.set_ylabel(metric.replace('_', ' ').capitalize())
                if i == 2: ax.set_xlabel('Completed optimizer updates')
        fig.suptitle('Checkpoint outcomes and whole-problem uncertainty · ' + ('primary' if policy == old.LEGACY else 'helper-aware secondary'), x=.075, ha='left', fontweight='bold', fontsize=13)
        fig.legend([plt.Line2D([], [], color=arm_colors[a], marker='.') for a in arms[1:]], [a.upper() if a == 'pc4' else a.capitalize() for a in arms[1:]],
                   loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
        fig.subplots_adjust(top=.86, bottom=.14, left=.08, right=.98, hspace=.25, wspace=.1)
        fig.text(.08, .025, 'Points: known rates. Thin whiskers: pointwise 95% whole-problem bootstrap (10,000 resamples; ten samples kept together).\n'
                 'Thick capped bars: unknown-outcome identification bounds; no midpoint estimate is substituted. Missing checkpoints remain blank.\n'
                 'Dashed: Random0 H100→Ada after 100. ' + ('Dotted: fresh on-policy branch from 120. ' if fresh_branch else '') +
                 'Projection off; no-loophole capability is not measured.', fontsize=8)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / ('rates_' + stem + '.' + ext), dpi=240, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True); parser.add_argument('--inputs-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.inputs).resolve()
    raw = old.read_pinned(path, args.inputs_sha)
    index = json.loads(raw)
    require(index['schema'] == 'rh_followup_plot_inputs_v1' and isinstance(index['subsets'], list), 'Wrong input index')
    history, manifest = frozen_inputs()
    cells = []
    for entry in index['subsets']:
        cells.extend(load_subset(path.parent, entry, manifest))
    report = append_summaries(history, cells)
    import matplotlib
    report['inputs'] = dict(historical_primary_sha256=PRIMARY_SHA, historical_secondary_sha256=SECONDARY_SHA,
        historical_manifest_sha256=MANIFEST_SHA, qualified_source_sha256=SOURCE_SHA,
        inputs_sha256=args.inputs_sha, input_bindings=index, source_sha256=sha(Path(__file__).read_bytes()),
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    (output / 'inputs.original.json').write_bytes(raw)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for name, rows in [
        ('category_counts.csv', [dict(policy=s['policy'], arm=s['arm'], step=s['step'], setting=s['setting'], category=k, count=v)
                                for s in report['summaries'] for k, v in s['labels'].items()]),
        ('metrics.csv', [dict(policy=s['policy'], arm=s['arm'], step=s['step'], setting=s['setting'], metric=k, **v)
                         for s in report['summaries'] for k, v in s['metrics'].items()]),
        ('paired_differences.historical_unchanged.csv', report['paired_differences'])]:
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(report, output)
    files = {p.name: dict(sha256=sha(p.read_bytes()), size_bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'artifact_manifest.json').write_text(json.dumps(dict(algorithm='sha256', files=files), indent=2, sort_keys=True) + '\n')
    print(old.canonical(dict(status='verified_available_cells_plotted', cells=report['cells'], records=report['records'],
                            new_cells=report['new_cells'], new_records=report['new_records'], output=str(output))))


if __name__ == '__main__':
    main()
