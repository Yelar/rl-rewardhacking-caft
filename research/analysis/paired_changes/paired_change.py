"""Paired problem-bootstrap change from verified Random0 checkpoint100 to110."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;C=D.parent
spec=importlib.util.spec_from_file_location('qualified_followup_plot',C/'rh_followup_plot_v2/plot_followup.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
old=p.old
ROOT=C.parent/'hpc-full-rh-eval-20260910-continue'

def change(a,b,weights):
    # Same conservative endpoint subtraction as the original paired-arm estimator.
    lower=b[0]-a[1];upper=b[1]-a[0]
    boot_lower=weights@lower;boot_upper=weights@upper
    return {'estimate':float(lower.mean()) if np.array_equal(lower,upper) else None,
        'identification_bounds':[float(lower.mean()),float(upper.mean())],
        'ci95':[float(np.quantile(boot_lower,.025)),float(np.quantile(boot_upper,.975))],
        'sampling_ci95_for_lower_identification_endpoint':np.quantile(boot_lower,[.025,.975]).tolist(),
        'sampling_ci95_for_upper_identification_endpoint':np.quantile(boot_upper,[.025,.975]).tolist(),
        'problem_lower_differences':lower.tolist(),'problem_upper_differences':upper.tolist()}

def old_cells(history,manifest):
    proof=json.loads(old.read_pinned(ROOT/'final_verification_v1/INDEPENDENT_VERIFICATION.json',
        '10c2299b6eee4d2d7987b62e720f451ce1e1ecd469f9f7da19e909c0b333bd97'))
    assert proof['status']=='succeeded' and proof['manifest_sha256']==p.MANIFEST_SHA and proof['cells']==44 and proof['samples']==52360
    root=ROOT/'mirror/output';cells=[];bindings={}
    science=old.digest(old.canonical({k:manifest[k] for k in ('datasets','adapters','sampling','revision','source_files','seed_policy')}).encode())
    adapter=next(a for a in manifest['adapters'] if a['arm']=='random0' and a['step']==100)
    for setting in old.SETTINGS:
        folder=root/'cells'/f'random0_100_{setting}'
        seals=[json.loads((folder/name).read_bytes()) for name in ['RAW_COMPLETE.json','SCORE_COMPLETE.json']]
        identity={'run_token':manifest['run_token'],'cell':folder.name,'manifest_science_sha256':science}
        assert all(s['count']==1190 and s['identity']==identity for s in seals)
        batches=[]
        for seal,key in zip(seals,['raw','results']):
            b=seal[key];path=(root/Path(b['path']).relative_to(manifest['output'])).resolve();assert path.is_relative_to(folder.resolve())
            data=old.read_pinned(path,b['sha256']);assert len(data)==b.get('size_bytes',b.get('bytes')) and data.endswith(b'\n')
            batches.append([json.loads(line) for line in data.splitlines()]);bindings[str(path.relative_to(ROOT))]=b
        raw,rows=batches;assert len(raw)==len(rows)==1190
        for a,b in zip(raw,rows):
            assert a['request_id']==b['request_id'] and int(a['problem_id'])==int(b['id']) and a['sample_index']==b['sample_index']
            assert a['completion']==b['response'] and a['adapter_files']==adapter['files'] and a['dataset']==manifest['datasets'][setting]
            assert a['engine_seed']==1
        cells.append({'arm':'random0','step':100,'setting':setting,'results':rows})
    target={r['request_id']:r for cell in cells for r in cell['results']}
    helper=ROOT/'full_eval_helper_execution_v1/mirror/merged/helper_sidecar.jsonl'
    h=hashlib.sha256();count=0
    with helper.open('rb') as f:
        for line in f:
            h.update(line);side=json.loads(line);rid=side['request_id']
            if rid not in target:continue
            r=target[rid];assert 'helper_aware_evaluation' not in r
            assert side['primary_gt_pass']==r['eq_correct'] and side['legacy_gt_sha256']==old.digest(old.canonical(r['gt_result']).encode())
            r['helper_aware_evaluation']=side['helper_aware_evaluation'];count+=1
    assert count==2380 and h.hexdigest()=='c39f289e8a49427b21d13f10f919c4e3f73d948857e7c46231380666bc955990'
    return cells,bindings

def main():
    index_path=C/'rh110_plot_inputs_v2.json';index=json.loads(index_path.read_bytes())
    history,manifest=p.frozen_inputs();weights=p.weights_for_history(history)
    previous,bindings=old_cells(history,manifest)
    current=[cell for entry in index['subsets'] for cell in p.load_subset(C,entry,manifest)]
    assert {(c['arm'],c['step'],c['setting']) for c in current}=={('random0',110,s) for s in old.SETTINGS}
    expected={(pid,j) for pid in history['problem_ids'] for j in range(10)}
    differences=[]
    for setting in old.SETTINGS:
        cells=[next(c for c in group if c['setting']==setting) for group in [previous,current]]
        ordered=[sorted(c['results'],key=lambda r:(int(r['id']),r['sample_index'])) for c in cells]
        assert all({(int(r['id']),r['sample_index']) for r in rows}==expected for rows in ordered)
        policies=[old.LEGACY]+([old.HELPER] if all('helper_aware_evaluation' in r for r in ordered[1]) else [])
        for policy in policies:
            views=[[old.view(r,policy)[1] for r in rows] for rows in ordered]
            for metric in ['ground_truth_correctness','strict_reward_hack','evaluator_presence']:
                rates=[];vectors=[]
                for view in views:
                    rates_and_vectors=old.rate_stats([[view[i*10+j][metric] for j in range(10)] for i in range(119)],weights)
                    rates.append(rates_and_vectors[0]);vectors.append(rates_and_vectors[1])
                frozen=next(s for s in history['summaries'] if (s['arm'],s['step'],s['setting'],s['policy'])==('random0',100,setting,policy))
                assert rates[0]==frozen['metrics'][metric],'Historical cell does not exactly reproduce original summary'
                differences.append({'setting':setting,'policy':policy,'metric':metric,'comparison':'Random0 step110 minus step100',
                    'before':rates[0],'after':rates[1],**change(*vectors,weights)})
    report={'kind':'paired_random0_100_to110_problem_bootstrap_v1','problems':119,'samples_per_problem':10,
        'sampling':'Same119 sorted problem IDs, ten samples retained together,10,000 shared whole-problem draws,seed6219',
        'differences':differences,'new_inputs_sha256':old.digest(index_path.read_bytes()),'historical_native_bindings':bindings,
        'limitations':['Projection disabled for both checkpoint evaluations.','Loophole-prompt GT success is not ordinary no-loophole capability.','The continuation changed H100 to Ada and eight to four training ranks after update100; this is not a training-seed replication.',
            'Unknown identification intervals are separate from percentile sampling intervals.','Empirical zero-event bootstrap intervals collapse to zero and do not establish that population RH risk is zero.','Problem resampling does not capture training-seed uncertainty or multiple-comparison selection.']}
    with (D/'paired_changes.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ['setting','policy','metric','estimate','ci95','identification_bounds']} for r in differences],indent=2))

if __name__=='__main__':main()
