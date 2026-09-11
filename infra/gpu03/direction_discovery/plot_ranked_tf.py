"""Export an exploratory TF-ranking figure from a verified combined package.

This consumes the same strict selection specification as select_behavior and
does not generate a behavior request plan, run a model, or inspect activations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

try:
    from . import select_behavior as selection
except ImportError:
    import select_behavior as selection


def label(row):
    candidate = row['candidate_id']
    layer = candidate.split('.')[0]
    if '.pc' in candidate:
        return f"{layer}  PC {candidate.rsplit('pc', 1)[1]}"
    control = 'incorrect control' if 'harmful_incorrect_vs_benign_incorrect' in candidate else 'both controls'
    space = 'M60' if candidate.endswith('.v60') else 'checkpoint change'
    return f'{layer}  mean: {control}, {space}'


def render(ranking, selected_ids, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = ranking['target_ranking'][:20]
    selection.behavior.require(len(rows) >= 3 and selected_ids == [v['condition_id'] for v in rows[:3]],
                               'Figure selection must preserve the strict first-three order')
    selected = set(selected_ids)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True, gridspec_kw={'width_ratios': [1.25, 1]})
    summaries = []
    for index, row in enumerate(rows):
        score = row['priority_beyond_random_mean']
        capability = row['metrics']['benign_correct_nll_increase']
        color = '#126eae' if row['condition_id'] in selected else '#777777'
        for ax, values in zip(axes, (score, capability)):
            selection.behavior.require(all(type(values[key]) in (int, float) and math.isfinite(values[key])
                                           for key in ('mean', 'p025', 'p975')) and values['p025'] <= values['p975'],
                                       'Invalid source interval')
            ax.plot([values['p025'], values['p975']], [index, index], color=color, linewidth=1.5)
            ax.plot(values['mean'], index, 'o', color=color, markersize=5)
        summaries.append({'condition_id': row['condition_id'], 'label': label(row),
                          'selected_for_screening': row['condition_id'] in selected,
                          'tf_priority_beyond_random': score, 'clean_correct_nll_increase': capability})
    axes[0].set_yticks(range(len(rows)), [label(row) for row in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title('TF priority beyond matched random controls', fontsize=11)
    axes[1].set_title('Clean-correct NLL increase', fontsize=11)
    for ax in axes:
        ax.axvline(0, color='#222222', linestyle='--', linewidth=.8)
        ax.set_xlabel('Nats per token')
        ax.grid(axis='x', color='#eeeeee')
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Evaluator transition: combined coarse and neighbor TF ranking', fontsize=14, y=.98)
    fig.text(.01, .03,
             '12 validation problems; 2,000 paired problem bootstrap resamples. Bars show 95% percentile intervals.\n'
             'Blue marks the exact first three targets for screening. Intervals do not adjust for candidate selection.\n'
             'Higher left-panel priority and lower right-panel NLL change are favorable; behavioral efficacy is untested.',
             fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .95))
    for name in ('combined_tf_ranking.png', 'combined_tf_ranking.pdf'):
        fig.savefig(output / name, dpi=180, metadata={'Creator': 'plot_ranked_tf.py'})
    plt.close(fig)
    return summaries


def run(spec, output):
    selection.behavior.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Plotting must hide CUDA')
    output = Path(output)
    selection.behavior.require(output.is_absolute() and not output.exists(), 'Figure output must be fresh and absolute')
    selected = selection.build_selection(**spec)
    provenance = selected['ranking_provenance']
    root = Path(provenance['ranking_root'])
    ranking = selection.bound_json(root / 'tf_ranking.json', {'sha256': provenance['ranking_sha256']})
    input_roots = [root, Path(provenance['master_path']).parent, Path(provenance['parent_master_path']).parent]
    for phase in provenance['source_phases']:
        input_roots.append(Path(phase['request_plan']).parent.parent)
    input_roots.extend(Path(layer['path']).parent for target in selected['conditions'].values() for layer in target['layers'])
    selection.behavior.require(all(not output.resolve().is_relative_to(path.resolve()) for path in input_roots),
                               'Do not write figures into frozen input packages')
    output.mkdir(parents=True)
    summaries = render(ranking, selected['selected_condition_ids_in_priority_order'], output)
    result = {'status': 'exploratory_likelihood_figure', 'source': provenance,
              'selected_condition_ids': selected['selected_condition_ids_in_priority_order'],
              'requests_verified': ranking['requests_verified'], 'problem_ids': ranking['problem_ids'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'displayed_targets': summaries, 'model_work_launched': False,
              'behavioral_efficacy_established': False}
    (output / 'figure_data.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    manifest = {'algorithm': 'sha256', 'files': {path.name: {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()}}
    (output / 'artifact_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    for path in output.iterdir():
        path.chmod(0o400)
    return {'status': result['status'], 'output': str(output),
            'artifact_manifest_sha256': hashlib.sha256((output / 'artifact_manifest.json').read_bytes()).hexdigest(),
            'model_work_launched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(selection.parse_json(args.spec.read_bytes()), args.output), sort_keys=True))
