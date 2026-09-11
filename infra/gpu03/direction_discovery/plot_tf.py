"""Render verified TF prioritization as a descriptive scientific figure."""
import argparse
import hashlib
import json
from pathlib import Path


def render(ranking_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=Path(ranking_path).read_bytes();report=json.loads(raw)
    if report['status']!='exploratory_TF_prioritization_only' or not report['target_ranking']:
        raise ValueError('Expected a complete exploratory TF ranking')
    rows=report['target_ranking'];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.6,1]})
    families=[('Mean direction',lambda r:'.mean.' in r['candidate_id'],'#146c94','o'),
              ('Individual PC',lambda r:'.mean.' not in r['candidate_id'],'#aa4a44','x')]
    for name,accept,color,marker in families:
        group=[row for row in rows if accept(row)]
        ax.scatter([row['layer'] for row in group],[row['priority_beyond_random_mean']['mean'] for row in group],
                   s=22,alpha=.75,label=name,color=color,marker=marker)
    ax.axhline(0,color='#555555',linewidth=.8);ax.set(xlabel='Transformer block (zero-based)',ylabel='Penalized NLL priority above matched random mean')
    ax.set_title('All tested transition candidates');ax.legend(frameon=False)
    best=rows[:5]
    for y,row in enumerate(reversed(best)):
        value=row['priority_beyond_random_mean'];color='#146c94' if '.mean.' in row['candidate_id'] else '#aa4a44'
        bx.plot([value['p025'],value['p975']],[y,y],color=color,linewidth=2)
        bx.scatter(value['mean'],y,color=color,s=28)
    bx.axvline(0,color='#555555',linewidth=.8)
    labels=[row['candidate_id'].replace('.transition.',' ').replace('harmful_incorrect_vs_benign_incorrect','incorrect contrast').replace('harmful_vs_benign','combined contrast').replace('mean.','') for row in reversed(best)]
    bx.set_yticks(range(len(best)),labels);bx.set(xlabel='Priority, paired 95% bootstrap interval',title='Five highest validation priorities')
    fig.suptitle('Likelihood prioritization; behavioral benefit remains unestablished',fontsize=12)
    fig.text(.02,.015,f"{report['requests_verified']:,} verified trials; {len(report['problem_ids'])} validation problems. Intervals are exploratory and unadjusted for selection.",fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.94])
    for suffix in ['png','pdf']:fig.savefig(output/('tf_prioritization.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    metadata={'input_sha256':hashlib.sha256(raw).hexdigest(),'master_plan_sha256':report['master_plan_sha256'],
              'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'requests':report['requests_verified'],
              'interpretation':'Descriptive prioritization only. Axes use target effect above mean of three matched random controls. Intervals do not correct for model selection.'}
    (output/'figure_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ranking',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(render(a.ranking,a.output)))
