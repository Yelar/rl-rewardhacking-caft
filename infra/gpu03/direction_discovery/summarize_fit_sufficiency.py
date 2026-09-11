#!/usr/bin/env python3
"""Independent terminal-output verification and scientific summary/plots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    from . import candidates as c
    from . import fit_sufficiency as s
except ImportError:
    import candidates as c
    import fit_sufficiency as s


PRIMARY = ('harmful_vs_benign_correct', 'harmful_incorrect_vs_benign_incorrect')
LABELS = {'harmful_vs_benign_correct': 'RH − clean correct',
          'harmful_incorrect_vs_benign_incorrect': 'RH − clean incorrect'}
METHOD_LABELS = {'definition_predictor':'Definition predictor', 'definition':'Definition token',
                 'body_predictor':'First-body predictor', 'body':'First-body token',
                 'pre_definition':'Pre-definition window', 'pre_body':'Pre-body window',
                 'transition':'Transition window', 'early_body':'Early-body window'}


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def stat(cell, family, group, comparison='pair_signed_cosine', quantile='p50'):
    stats = cell['mean'][family]['v_change']['groups'][group][comparison]['statistics']
    return stats[quantile] if stats else None


def load_cells(root, digest):
    cells = []
    for layer in range(36):
        folder = root/'results'/f'layer_{layer:02d}'
        done = json.loads((folder/'complete.json').read_text())
        c.require(done['status']=='complete' and done['cells']==16 and done['manifest_sha256']==digest,
                  'missing/mismatched layer completion')
        for region in s.REGIONS:
            for method in s.METHODS:
                cell = json.loads((folder/f'{region}__{method}.json').read_text())
                c.require((cell['layer'],cell['region'],cell['method'])==(layer,region,method), 'cell identity mismatch')
                c.require(cell['study_manifest_sha256']==digest and cell['fitting_problems']==111, 'cell provenance mismatch')
                cells.append(cell)
    return cells


def verify(root, destination):
    start=time.monotonic()
    manifest_path=root/'control/study_manifest.json';digest=c.sha256_file(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    unit=manifest['run_token']+'-study'
    raw=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','Result',
                                  '-p','ExecMainStatus','-p','MainPID','-p','ControlGroup'],text=True)
    state=dict(line.split('=',1) for line in raw.splitlines())
    c.require(state['ActiveState']=='inactive' and state['SubState']=='dead' and state['Result']=='success' and
              state['ExecMainStatus']=='0' and state['MainPID']=='0', 'producer has not positively exited successfully')
    complete=json.loads((root/'control/study_completion.json').read_text())
    c.require(complete['status']=='complete' and complete['finished_layers']==list(range(36)) and
              complete['owned_workers_reaped'] and complete['manifest_sha256']==digest, 'incomplete study')
    cg=(Path('/sys/fs/cgroup')/state['ControlGroup'].lstrip('/')) if state['ControlGroup'] else None
    if cg is not None and cg.exists():
        c.require(not (cg/'cgroup.procs').read_text().strip(),'owned cgroup still has processes')
    for line in (root/'control/study_journal.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['event']=='started':
            cmd=Path('/proc')/str(event['pid'])/'cmdline'
            if cmd.exists():
                c.require(str(root).encode() not in cmd.read_bytes(),'owned worker process still running')
    cells=load_cells(root,digest)
    c.require(len(list((root/'results').glob('layer_*/*__*.json')))==576,'unexpected result cell count')
    files={}
    for relative,bound in manifest['inputs'].items():
        path=c.safe_child(root,relative);c.verify_file(path,bound);files[relative]=bound
    transfer=json.loads((root/'control/transfer_plan.json').read_text())
    for entry in transfer['files']:
        path=c.safe_child(root/'raw',entry['path']);c.verify_file(path,entry)
        files['raw/'+entry['path']]={'sha256':entry['sha256'],'size_bytes':entry['size_bytes']}
    matrices=0;max_orth=0.;max_metric_error=0.
    for cell in cells:
        stem=root/'results'/f"layer_{cell['layer']:02d}"/f"{cell['region']}__{cell['method']}"
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as arrays:
            expected={f'{family}__{kind}' for family in c.FAMILIES for kind in s.KINDS}|{'full_pcs','solver_repeat_pcs'}|{key+'_pcs' for key in cell['pca']['groups']}
            c.require(set(arrays.files)==expected,'missing/unexpected fitted vectors')
            for key in arrays.files:
                value=arrays[key];c.require(value.dtype==np.float32 and np.isfinite(value).all(),'invalid stored vectors')
                if not key.endswith('_pcs'):
                    c.require(value.shape==(2560,),'wrong mean vector dimensions');continue
                shape=(2560,10) if key in ('full_pcs','solver_repeat_pcs') else (12,2,2560,10)
                c.require(value.shape==shape,'wrong PCA vector dimensions')
                for matrix in value.reshape(-1,2560,10):
                    gram=matrix.astype(np.float64).T @ matrix.astype(np.float64)
                    err=float(np.max(np.abs(gram-np.eye(10))));max_orth=max(max_orth,err)
                    c.require(err<5e-5,'independent stored-PCA orthogonality failure');matrices+=1
                if key not in ('full_pcs','solver_repeat_pcs'):
                    group=cell['pca']['groups'][key[:-4]]
                    # Independently recompute every pair's reported cosine and subspace metrics.
                    for pair,reported in zip(value,group['pair_details']):
                        observed=s.compare_pcs(pair[0],pair[1])
                        for k in ('same_index_absolute_cosine','assignment_absolute_cosine'):
                            err=float(np.max(np.abs(np.asarray(observed[k])-reported[k])));max_metric_error=max(max_metric_error,err)
                            c.require(err<1e-10,'saved vector/pair metric disagreement')
            for family in c.FAMILIES:
                for kind in s.KINDS:
                    norm=float(np.linalg.norm(arrays[f'{family}__{kind}'].astype(np.float64)))
                    c.require(np.isclose(norm,cell['mean'][family][kind]['full_norm'],rtol=1e-6,atol=1e-7),'mean reference norm disagreement')
        for path in (stem.with_suffix('.json'),stem.with_suffix('.npz')):
            files[str(path.relative_to(root))]={'sha256':c.sha256_file(path),'size_bytes':path.stat().st_size}
    for path in sorted((root/'control').glob('*.json'))+sorted((root/'control').glob('*.log'))+sorted((root/'results').glob('layer_*/complete.json')):
        files[str(path.relative_to(root))]={'sha256':c.sha256_file(path),'size_bytes':path.stat().st_size}
    for name in ('study_journal.jsonl','launch_command.sh'):
        path=root/'control'/name;files[str(path.relative_to(root))]={'sha256':c.sha256_file(path),'size_bytes':path.stat().st_size}
    c.write_json(destination/'artifact_manifest.json',{'root':str(root),'study_manifest_sha256':digest,'files':files})
    c.write_json(destination/'independent_verification.json',{'status':'passed','cells':576,'raw_files':666,
        'pca_matrices_verified':matrices,'max_orthonormality_error':max_orth,'max_pair_metric_error':max_metric_error,
        'actual_service_state':state,'owned_workers_released':True,'gpu_allocations':[],
        'seconds':time.monotonic()-start,'artifact_manifest_sha256':c.sha256_file(destination/'artifact_manifest.json')})
    return cells,manifest


def summarize(cells,manifest):
    index={(v['layer'],v['region'],v['method']):v for v in cells}
    result={'fitting_problems':111,'total_original_problems':187,'cells':len(cells),'by_region_method':{},
            'numerical_refinement_candidates':[],'mean_group_counts':{},'manifest_runtime_seconds':None}
    for region in s.REGIONS:
        result['by_region_method'][region]={}
        for method in s.METHODS:
            selected=[index[l,region,method] for l in range(36)]
            values={}
            for family in PRIMARY:
                values[family]={}
                for group in selected[0]['mean'][family]['v_change']['groups']:
                    pair=[stat(v,family,group) for v in selected]
                    reference=[stat(v,family,group,'reference_signed_cosine') for v in selected]
                    values[family][group]={'median_across_layer_pair_medians':median(pair),
                        'median_across_layer_reference_medians':median(reference),
                        'layers_pair_median_at_least_09':sum(v is not None and v>=.9 for v in pair),
                        'layers_with_nonzero_pairs':sum(v is not None for v in pair)}
            result['by_region_method'][region][method]=values
    for family in PRIMARY:
        result['mean_group_counts'][family]={}
        for group in cells[0]['mean'][family]['v_change']['groups']:
            numbers=[stat(v,family,group) for v in cells]
            result['mean_group_counts'][family][group]={'valid_cells':sum(v is not None for v in numbers),
                'cells_pair_median_at_least_09':sum(v is not None and v>=.9 for v in numbers),
                'median_across_cells':median(numbers)}
    for cell in cells:
        pca=cell['pca'];same=pca['solver_repeat_agreement']['same_index_absolute_cosine']
        if pca['approximation_warning'] or min(same)<.99:
            result['numerical_refinement_candidates'].append({'layer':cell['layer'],'region':cell['region'],'method':cell['method'],
                'warning':pca['approximation_warning'],'min_same_data_solver_cosine':min(same),
                'max_resample_relative_residual':pca['all_resample_max_relative_residual']['p975']})
    chosen=index[21,'evaluator','transition'];result['previously_tested_L21_transition']={
        'means':{f:chosen['mean'][f]['v_change'] for f in PRIMARY},'pca':chosen['pca']}
    result['pca_disjoint55_by_region_method']={}
    for region in s.REGIONS:
        result['pca_disjoint55_by_region_method'][region]={}
        for method in s.METHODS:
            p=[index[l,region,method]['pca']['groups']['disjoint_55']['pair'] for l in range(36)]
            result['pca_disjoint55_by_region_method'][region][method]={
                'median_layer_PC0_cosine':median([v['same_index_absolute_cosine'][0]['p50'] for v in p]),
                'median_layer_PC4_cosine':median([v['same_index_absolute_cosine'][4]['p50'] for v in p]),
                'median_layer_top5_subspace_overlap':median([v['subspaces']['5']['overlap']['p50'] for v in p]),
                'median_layer_top10_subspace_overlap':median([v['subspaces']['10']['overlap']['p50'] for v in p])}
    return result


def plots(cells,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    idx={(v['layer'],v['region'],v['method']):v for v in cells}
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for i,region in enumerate(s.REGIONS):
        for j,family in enumerate(PRIMARY):
            raw=[[stat(idx[l,region,m],family,'disjoint_55') for m in s.METHODS] for l in range(36)]
            data=np.array([[np.nan if v is None else v for v in row] for row in raw])
            im=axes[i,j].imshow(data,aspect='auto',origin='lower',vmin=-.25,vmax=1,cmap='viridis')
            axes[i,j].set_title(f'{region.capitalize()}: {LABELS[family]}')
            axes[i,j].set_xticks(range(8),[METHOD_LABELS[m].replace(' ','\n',1) for m in s.METHODS],rotation=40,ha='right')
            axes[i,j].set_ylabel('Transformer block (zero based)')
    fig.colorbar(im,ax=axes,label='Median signed cosine: disjoint 55 / 55 problem fits',shrink=.7)
    fig.suptitle('Checkpoint-induced paired means: anchor versus window stability')
    fig.savefig(out/'mean_disjoint_stability.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    chosen=idx[21,'evaluator','transition']['pca']
    for ax,scheme,ns,comparison in [(axes[0],'disjoint',[30,45,55],'pair'),(axes[1],'bootstrap',[30,60,90,111],'reference')]:
        for pc in [0,4]:
            values=[chosen['groups'][f'{scheme}_{n}'][comparison]['same_index_absolute_cosine'][pc] for n in ns]
            ys=np.array([v['p50'] for v in values]);lo=np.array([v['p025'] for v in values]);hi=np.array([v['p975'] for v in values])
            ax.plot(ns,ys,marker='o',label=f'PC{pc}');ax.fill_between(ns,lo,hi,alpha=.12)
        values=[chosen['groups'][f'{scheme}_{n}'][comparison]['subspaces']['5']['overlap']['p50'] for n in ns]
        ax.plot(ns,values,marker='s',linestyle='--',label='Top-5 subspace overlap')
        ax.set_ylim(0,1.02);ax.set_xticks(ns);ax.set_xlabel('Problems per side' if scheme=='disjoint' else 'Bootstrap draws (with replacement)');ax.set_ylabel('Agreement');ax.legend()
        ax.set_title('Independent disjoint fits' if scheme=='disjoint' else 'Bootstrap fit versus full 111-problem fit')
    fig.suptitle('Previously tested direction: evaluator transition, block 21\nShading: descriptive resampling quantiles; PCA uses 12 pairs per size')
    fig.savefig(out/'previous_direction_learning_curve.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,region in enumerate(s.REGIONS):
        for j,family in enumerate(PRIMARY):
            for method in ['body_predictor','pre_body','transition','early_body']:
                ys=[median([stat(idx[l,region,method],family,f'disjoint_{n}') for l in range(36)]) for n in [30,45,55]]
                axes[i,j].plot([30,45,55],ys,marker='o',label=METHOD_LABELS[method])
            axes[i,j].set_title(f'{region.capitalize()}: {LABELS[family]}');axes[i,j].set_xticks([30,45,55]);axes[i,j].set_xlabel('Problems per independent side');axes[i,j].set_ylim(-.2,1);axes[i,j].set_ylabel('Median over 36 layer-wise median cosines');axes[i,j].legend(fontsize=8)
    fig.suptitle('Mean-direction learning curves without shared problems')
    fig.savefig(out/'mean_learning_curves.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    cells,manifest=verify(args.root,args.output);summary=summarize(cells,manifest)
    c.write_json(args.output/'summary.json',summary);plots(cells,args.output)
    print(c.canonical({'status':'verified','cells':len(cells),'numerical_refinement_candidates':len(summary['numerical_refinement_candidates'])}),flush=True)


if __name__=='__main__':main()
