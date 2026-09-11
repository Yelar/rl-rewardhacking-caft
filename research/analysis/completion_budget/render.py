"""Render only closed statistics; never re-run estimators or model/code evaluation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent
LEGACY='repository_five_probe_legacy_v1'
def main():
 report=json.loads((D/'analysis.json').read_text());assert report['cells']==3 and report['records']==3570 and report['whole_four_cell_complete'] is False
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(4,2,figsize=(10.4,11.4),sharex=True,sharey='row')
 metrics=[('evaluator_presence','Evaluator presence'),('strict_reward_hack','Native strict RH'),('truncation','Length-finished truncation'),('ground_truth_correctness','Ground-truth correctness')]
 configurations=[(1536,'off'),(1536,'on'),(3072,'off'),(3072,'on')];points=[]
 for row,(metric,label) in enumerate(metrics):
  ymax=.4 if row<2 else 1.
  for col,setting in enumerate(['fixed','randomized']):
   ax=axes[row,col]
   for x,(cap,mode) in enumerate(configurations):
    entries=report['reference1536_rates'] if cap==1536 else report['rates']
    if cap==3072 and mode=='on' and setting=='randomized':
     ax.text(x,.10,'Pending',ha='center',va='bottom',rotation=90,fontsize=8,color='#666666',transform=ax.get_xaxis_transform())
     continue
    e=next(r for r in entries if (r['condition'],r['setting'],r['policy'],r['metric'])==(mode,setting,LEGACY,metric));r=e['rate']
    lo,hi=r['ci95'];value=r['identification_bounds'][0];ymax=max(ymax,hi*100*1.30)
    color='#51799c' if mode=='off' else '#c65d50';marker='o' if cap==1536 else 's'
    ax.errorbar(x,value*100,yerr=np.array([[max(0,value-lo)*100],[max(0,hi-value)*100]]),fmt=marker,color=color,capsize=4,markersize=7,clip_on=False,alpha=.65 if cap==1536 else 1.)
    ax.annotate(f"{r['successes']}/1190",(x,hi*100),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8,color=color)
    points.append(dict(metric=metric,setting=setting,condition=mode,max_completion_tokens=cap,successes=r['successes'],total=1190,ci95=r['ci95']))
   ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlim(-.45,3.45)
   if row==0:ax.set_title('Fixed evaluator name' if setting=='fixed' else 'Randomized evaluator name',fontsize=12)
  for ax in axes[row]:ax.set_ylim(0,min(100,ymax))
  axes[row,0].set_ylabel(label+' (%)')
 for ax in axes[-1]:ax.set_xticks(range(4),['OFF\n1536','ON\n1536','OFF\n3072','ON\n3072']);ax.set_xlabel('Projection mode / maximum completion tokens')
 fig.suptitle('Partial completion-budget test at frozen Random0 checkpoint 100',fontsize=14,y=.98)
 fig.text(.09,.025,'Three completed3072 cells; ON randomized pending (40 partial rows excluded).\n119 problems × 10 samples per cell; same checkpoint, prompts and seed policy.\nBars: 95% whole-problem bootstrap intervals. Native classifier shown; helper unknown bounds reported separately.\nZero-event intervals collapse, not population upper bounds. ON includes materialization/rounding. Cross-budget pairing is exploratory.',fontsize=8.2,va='bottom')
 fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.935,hspace=.25,wspace=.15)
 outputs=[]
 for ext in ['png','pdf']:
  path=D/f'completion_budget_projection.{ext}';assert not path.exists();fig.savefig(path,dpi=240);outputs.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size))
 plt.close(fig)
 with(D/'FIGURE_BINDINGS.json').open('x')as out:json.dump(dict(points=points,figures=outputs,analysis_sha256=hashlib.sha256((D/'analysis.json').read_bytes()).hexdigest(),estimators_recomputed=False),out,indent=2,sort_keys=True);out.write('\n')
 print(json.dumps(dict(figures=outputs,points=len(points))))
if __name__=='__main__':main()
