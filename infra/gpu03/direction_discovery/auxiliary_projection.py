"""Offline assertion-present paired projections of already-fitted candidates.

No direction fitting, model forward, generated-code execution, or test activation
access. Fit and validation populations remain separate; PCs retain their fitted
sign and do not acquire a harmful interpretation from variance alone.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
try:
    from . import candidates as c, prepare_union as union
    from .fixed_cache_cross_audit import Package
except ImportError:
    import candidates as c
    import prepare_union as union
    from infra.gpu03.direction_discovery.fixed_cache_cross_audit import Package

SPACES = ('h0','h60','delta')


def paired_statistics(pair_scores, pairs, *, bootstrap=2000, seed=6201):
    """Equal problem means, then paired cluster resampling, never row bootstrap."""
    values=np.asarray(pair_scores,dtype=np.float64)
    c.require(values.shape==(len(pairs),) and np.isfinite(values).all() and len(pairs)>0,'Invalid paired projection scores')
    c.require(type(bootstrap) is int and 1<=bootstrap<=2000 and type(seed) is int,'Invalid diagnostic bootstrap bounds')
    groups=defaultdict(list)
    for i,pair in enumerate(pairs):
        c.require(pair['problem_split'] in ('direction_fit','configuration_validation'),'Test pair cannot enter projection diagnostic')
        groups[str(pair['problem_id'])].append(i)
    problems=sorted(groups)
    problem_means=np.asarray([values[groups[p]].mean() for p in problems])
    problem_positive=np.asarray([(np.sum(values[groups[p]]>0)+.5*np.sum(values[groups[p]]==0))/len(groups[p]) for p in problems])
    rng=np.random.default_rng(seed)
    chosen=rng.integers(0,len(problems),size=(bootstrap,len(problems)))
    replicates=problem_means[chosen].mean(axis=1)
    interval=np.quantile(replicates,[.025,.975])
    return {'pairs':len(pairs),'problems':len(problems),'problem_weighted_mean_harmful_minus_benign':float(problem_means.mean()),
        'bootstrap_percentile95':[float(x) for x in interval], 'problem_weighted_positive_pair_fraction':float(problem_positive.mean()),
        'bootstrap_replicates':bootstrap,'bootstrap_seed':seed,'bootstrap_unit':'paired problem cluster',
        'uncertainty_scope':'Conditional on already fitted candidate; does not include fitting uncertainty or multiplicity correction'}


def choose_candidates(catalog, layers=tuple(range(36)), windows=('transition',)):
    c.require(list(layers)==list(range(36)) and list(windows)==['transition'],'Diagnostic predeclares all36 layers at transition only')
    selected=[item for item in catalog if item['layer'] in layers and item['window'] in windows and
        (item['family'] in c.PRIMARY_FAMILIES and item['kind'] in ('v60','v_change') or item['family']=='pca' and item['kind']=='pc')]
    c.require(selected and len({x['candidate_id'] for x in selected})==len(selected) and len(selected)<=504,'Missing/duplicate/excess diagnostic candidates')
    for layer in layers:
        items=[x for x in selected if x['layer']==layer]
        expected={(family,kind) for family in c.PRIMARY_FAMILIES for kind in ('v60','v_change')}
        c.require({(x['family'],x['kind']) for x in items if x['family']!='pca'}==expected and
                  sum(x['family']=='pca' for x in items)<=10,'Primary candidate coverage differs from fixed protocol')
    c.require(all(x['rank']==1 for x in selected),'Only individual fitted vectors are diagnostic candidates')
    return sorted(selected,key=lambda x:(x['layer'],x['candidate_id']))


def row_vectors(reader, row, layer, window):
    c.require(row['problem_split'] in ('direction_fit','configuration_validation'),'Test activation access is forbidden')
    positions=c.valid_positions(row,window)
    h0=reader.read(row,'h0',layer,positions);h60=reader.read(row,'h60',layer,positions)
    c.require(h0.dtype==h60.dtype==np.float32 and h0.shape==h60.shape,'Native FP32 read conversion differs')
    delta=np.subtract(h60,h0,dtype=np.float32)
    return {key:value.mean(axis=0,dtype=np.float32) for key,value in (('h0',h0),('h60',h60),('delta',delta))}


def project_pair(harmful, benign, vector):
    q=np.asarray(vector)
    c.require(q.dtype==np.float32 and q.ndim==1 and np.isfinite(q).all(),'Candidate projection must be a finite FP32 vector')
    result={}
    for space in SPACES:
        a,b=np.asarray(harmful[space]),np.asarray(benign[space])
        c.require(a.dtype==b.dtype==np.float32 and a.shape==b.shape==q.shape,'Projection activation dimensions differ')
        # FP32 token differences/means precede this explicitly FP64 diagnostic
        # reduction, avoiding an unqualified local BLAS contraction.
        sa=float(np.einsum('i,i->',a.astype(np.float64),q.astype(np.float64),optimize=False))
        sb=float(np.einsum('i,i->',b.astype(np.float64),q.astype(np.float64),optimize=False))
        c.require(np.isfinite(sa) and np.isfinite(sb),'Projection score is nonfinite')
        result[space]={'harmful':sa,'benign':sb,'difference':sa-sb}
    return result


def checked_json(path,digest):
    path=Path(path);c.require(path.is_file() and not path.is_symlink() and c.sha256_file(path)==digest,'Changed bound diagnostic JSON')
    return json.loads(path.read_text())


def verify_small_package(root,expected):
    root=Path(root);c.require(c.sha256_file(root/'artifact_manifest.json')==expected,'Small package manifest changed')
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    c.require(manifest['algorithm']=='sha256','Unexpected manifest algorithm')
    for name,info in manifest['files'].items():c.verify_file(c.safe_child(root,name),info)
    return manifest


def bootstrap_groups(pair_results,pairs):
    groups=defaultdict(list)
    for i,pair in enumerate(pairs):groups[(pair['problem_split'],pair['auxiliary_group'])].append(i)
    summaries=[]
    for (split,group),indices in sorted(groups.items()):
        for space in SPACES:
            selected_pairs=[pairs[i] for i in indices]
            # The same problem resampling is used across candidates and spaces.
            values=[pair_results[i][space]['difference'] for i in indices]
            summaries.append({'split':split,'auxiliary_group':group,'score_space':space,
                              **paired_statistics(values,selected_pairs)})
    return summaries


def run(spec):
    import torch
    from safetensors import safe_open
    started=time.monotonic()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Projection diagnostic requires CUDA hidden')
    torch.set_num_threads(1)
    c.require(not torch.cuda.is_initialized(),'Projection diagnostic must remain CPU-only')
    runtime=spec.get('runtime_seconds',3600)
    c.require(type(runtime) is int and 0<runtime<=3600,'Diagnostic deadline exceeds one CPU hour')
    def check_time():c.require(time.monotonic()-started<runtime,'Auxiliary diagnostic deadline exceeded')
    output=Path(spec['output']);c.require(not output.exists(),'Preserve prior diagnostics; use fresh output')
    package=Path(spec['union_package'])
    c.require(c.sha256_file(package/'artifact_manifest.json')==spec['union_artifact_manifest_sha256'],'Union package identity changed')
    union_receipt=union.verify(package)
    pairs=json.loads((package/'input/auxiliary_selected_manifest.json').read_text())['pairs']
    rows=c.read_jsonl(package/'prepared_records.jsonl');by_id={r['record_id']:r for r in rows}
    c.require(len(pairs)==51 and all(p['problem_split']!='untouched_test' for p in pairs),'Auxiliary pair scope changed')
    core=Package(spec['core_package'],spec['core_artifact_manifest_sha256'],'core')
    aux=Package(spec['auxiliary_package'],spec['auxiliary_artifact_manifest_sha256'],'auxiliary')
    c.require({r['record_id']:r for r in core.rows}=={r['record_id']:r for r in rows[:561]} and
              {r['record_id']:r for r in aux.rows}=={r['record_id']:r for r in rows[561:]},'Cache prepared rows differ from frozen union')
    cross_root=Path(spec['cross_prefix_audit'])
    verify_small_package(cross_root,spec['cross_prefix_audit_artifact_sha256'])
    cross=json.loads((cross_root/'audit.json').read_text());cross_inputs=json.loads((cross_root/'inputs.json').read_text())
    c.require(cross['status']=='verified' and cross['completed_comparisons']==cross['expected_comparisons']==306 and
        cross['all_shared_prefixes_bitwise_equal'] is True and cross['no_test_activations_opened'] is True and
        cross_inputs['core_manifest_sha256']==spec['core_artifact_manifest_sha256'] and
        cross_inputs['auxiliary_manifest_sha256']==spec['auxiliary_artifact_manifest_sha256'],'Cross-cache numerical qualification is absent or mismatched')
    candidate_root=Path(spec['candidate_package'])
    proof=checked_json(spec['candidate_verification_receipt'],spec['candidate_verification_receipt_sha256'])
    c.require(proof['status']=='verified' and proof['mode']=='production' and proof['process_release_verified'] is True,
              'Candidate campaign is not independently verified and released')
    manifest=verify_small_package(candidate_root,proof['artifact_manifest_sha256'])
    reviewed=json.loads((candidate_root/'reviewed_manifest.json').read_text())
    c.require(reviewed['raw_manifest_sha256']==spec['core_artifact_manifest_sha256'] and
              Path(reviewed['raw_package']).resolve()==core.root,'Candidate lineage uses another raw cache (including obsolete variable-shape cache)')
    catalog=json.loads((candidate_root/'candidate_catalog.json').read_text())['candidates']
    selected=choose_candidates(catalog)
    controls=list(dict.fromkeys(p['paired_cached_control_record_id'] for p in pairs))
    selected_core=[by_id[rid] for rid in controls];selected_aux=[by_id[p['record_id']] for p in pairs]
    c.require(all(r['problem_split'] in ('direction_fit','configuration_validation') for r in selected_core+selected_aux),'Test native rows cannot enter auxiliary readers')
    readers={kind:c.RawReader(p.root,rr,list(p.index.values()),{'files':p.files},p.source_digest,36,2560)
        for kind,p,rr in (('core',core,selected_core),('auxiliary',aux,selected_aux))}
    roots=[package.resolve(),core.root,aux.root,candidate_root.resolve(),cross_root.resolve()]
    c.require(all(not output.resolve().is_relative_to(root) for root in roots),'Output cannot mutate diagnostic inputs')
    output.mkdir(parents=True)
    (output/'resolved_spec.json').write_text(c.canonical(spec)+'\n')
    (output/'selected_candidates.json').write_text(c.canonical({'candidates':selected})+'\n')
    (output/'source_sha256.json').write_text(c.canonical({str(Path(module.__file__).resolve()):c.sha256_file(module.__file__)
        for module in (c,union) } | {str(Path(__file__).resolve()):c.sha256_file(__file__)})+'\n')
    summary_rows=0
    try:
        with (output/'paired_scores.jsonl').open('x') as detail,(output/'summaries.jsonl').open('x') as summary:
            by_layer=defaultdict(list)
            for item in selected:by_layer[item['layer']].append(item)
            for layer,items in sorted(by_layer.items()):
                check_time()
                vectors={}
                for kind,reader,rr in (('core',readers['core'],selected_core),('auxiliary',readers['auxiliary'],selected_aux)):
                    for row in rr:
                        check_time();vectors[row['record_id']]=row_vectors(reader,row,layer,'transition')
                for item in items:
                    check_time();path=c.safe_child(candidate_root,item['tensor_file'])
                    with safe_open(str(path),framework='numpy') as handle:
                        q=handle.get_tensor(item['tensor_key'])
                        if 'column_indices' in item:q=q[:,item['column_indices']]
                    c.require(q.dtype==np.float32 and q.shape==(2560,1) and np.isfinite(q).all() and
                              abs(float(np.einsum('i,i->',q[:,0].astype(np.float64),q[:,0].astype(np.float64),optimize=False))-1)<5e-5,'Fitted candidate is not a normalized FP32 column')
                    pair_results=[project_pair(vectors[p['record_id']],vectors[p['paired_cached_control_record_id']],q[:,0]) for p in pairs]
                    canonical_space='h60' if item['kind']=='v60' else 'delta'
                    for pair,scores in zip(pairs,pair_results):
                        detail.write(c.canonical({'candidate_id':item['candidate_id'],'harmful_record_id':pair['record_id'],
                            'benign_record_id':pair['paired_cached_control_record_id'],'problem_id':pair['problem_id'],
                            'split':pair['problem_split'],'auxiliary_group':pair['auxiliary_group'],'scores':scores})+'\n')
                    for row in bootstrap_groups(pair_results,pairs):
                        summary.write(c.canonical({'candidate_id':item['candidate_id'],'family':item['family'],'kind':item['kind'],
                            'layer':layer,'window':'transition','canonical_score_space':canonical_space,
                            'score_sign':'positive fitted harmful-minus-benign orientation' if item['family']!='pca' else 'arbitrary fitted PC sign; no harmful orientation assumed',**row})+'\n')
                        summary_rows+=1
                del vectors
                detail.flush();summary.flush()
        check_time()
        result={'status':'succeeded','candidate_count':len(selected),'pairs':51,'summary_rows':summary_rows,
            'fit_correct_pairs':29,'validation_correct_pairs':11,'validation_strict_incorrect_pairs':11,
            'bootstrap_replicates':2000,'bootstrap_seed':6201,'no_fit_or_sign_selection_from_auxiliary':True,
            'no_test_activations_opened':True,'model_forwards':0,'new_generations':0,'raw_inputs_modified':False,
            'new_primary_direction_fit_records':0,'delta_arithmetic':'Native BF16 converted toFP32 before per-token subtraction; per-record meansFP32; scalar projectionsFP64 einsum',
            'limitations':['Assertion presence is matched; assertion count, evaluator length, syntax and modification subtype remain possible confounds.',
                'Fit auxiliary scores are descriptive and share fitting problems with primary direction estimation.',
                'Validation has11 problems per correctness group; groups overlap in problem identity and are never treated as independent combined evidence.',
                'Many candidate diagnostics are exploratory without multiplicity adjustment; results do not establish causal efficacy or intent.'],
            'elapsed_seconds':time.monotonic()-started,'cuda_initialized':torch.cuda.is_initialized(),
            'verified_native_file_counts':{key:len(reader.verified) for key,reader in readers.items()},
            'union_prepared_sha256':union_receipt['prepared_records_sha256']}
        (output/'SUCCESS.json').write_text(c.canonical(result)+'\n')
    except BaseException as error:
        (output/'FAILURE.json').write_text(c.canonical({'status':'failed','error':str(error),'type':type(error).__name__,
            'partial_diagnostics_retained':True,'summary_rows':summary_rows})+'\n');raise
    payload={str(path.relative_to(output)):{'sha256':c.sha256_file(path),'size_bytes':path.stat().st_size}
             for path in output.rglob('*') if path.is_file()}
    (output/'artifact_manifest.json').write_text(c.canonical({'algorithm':'sha256','files':payload})+'\n')
    for path in output.rglob('*'):
        if path.is_file():path.chmod(0o400)
    return {**result,'artifact_manifest_sha256':c.sha256_file(output/'artifact_manifest.json')}


def verify(output):
    output=Path(output)
    manifest=verify_small_package(output,c.sha256_file(output/'artifact_manifest.json'))
    c.require(not (output/'FAILURE.json').exists(),'Failed auxiliary diagnostic cannot verify')
    success=json.loads((output/'SUCCESS.json').read_text())
    c.require(success['status']=='succeeded' and success['no_test_activations_opened'] is True and
              success['new_primary_direction_fit_records']==0 and success['model_forwards']==0 and
              success['new_generations']==0 and success['cuda_initialized'] is False,'Auxiliary scope/completion changed')
    selected=json.loads((output/'selected_candidates.json').read_text())['candidates']
    c.require(choose_candidates(selected)==selected,'Saved candidate scope is not the predeclared all36 transition diagnostic')
    by_id={item['candidate_id']:item for item in selected}
    c.require(len(by_id)==len(selected)==success['candidate_count'],'Candidate diagnostic coverage mismatch')
    spec=json.loads((output/'resolved_spec.json').read_text())
    union_root=Path(spec['union_package'])
    c.require(c.sha256_file(union_root/'artifact_manifest.json')==spec['union_artifact_manifest_sha256'],'Bound union manifest changed before score verification')
    union_proof=union.verify(union_root)
    c.require(union_proof['prepared_records_sha256']==success['union_prepared_sha256'],'Diagnostic prepared union identity changed')
    source_pairs=json.loads((union_root/'input/auxiliary_selected_manifest.json').read_text())['pairs']
    expected_pairs={(pair['record_id'],pair['paired_cached_control_record_id']):pair for pair in source_pairs}
    c.require(len(source_pairs)==len(expected_pairs)==51,'Bound source pair population differs')
    seen=defaultdict(set)
    detail_by_candidate=defaultdict(list)
    for row in c.read_jsonl(output/'paired_scores.jsonl'):
        key=row['candidate_id'];c.require(key in by_id and row['split'] in ('direction_fit','configuration_validation'),'Unknown candidate or test diagnostic row')
        identity=(row['harmful_record_id'],row['benign_record_id'])
        c.require(identity in expected_pairs,'Diagnostic contains an unselected auxiliary/control pair')
        pair=expected_pairs[identity]
        c.require(str(row['problem_id'])==str(pair['problem_id']) and row['split']==pair['problem_split'] and
                  row['auxiliary_group']==pair['auxiliary_group'],'Saved auxiliary pair problem/split/group differs from exact source')
        c.require(identity not in seen[key],'Duplicate candidate-pair diagnostic')
        seen[key].add(identity)
        detail_by_candidate[key].append(row)
        c.require(set(row['scores'])==set(SPACES) and all(np.isfinite(value) for space in row['scores'].values()
                  for value in space.values()),'Nonfinite diagnostic score')
        c.require(all(set(space)=={'harmful','benign','difference'} and space['difference']==space['harmful']-space['benign']
                      for space in row['scores'].values()),'Paired scalar difference is not harmful minus benign')
    c.require(set(seen)==set(by_id) and all(pairs==set(expected_pairs) for pairs in seen.values()),
              'Auxiliary pair coverage differs across candidates or exact source population')
    summaries=c.read_jsonl(output/'summaries.jsonl')
    c.require(len(summaries)==success['summary_rows']==len(selected)*9,'Auxiliary group/space summary coverage is incomplete')
    summary_keys={(row['candidate_id'],row['split'],row['auxiliary_group'],row['score_space']) for row in summaries}
    c.require(len(summary_keys)==len(summaries),'Duplicate diagnostic summary')
    summary_lookup={(row['candidate_id'],row['split'],row['auxiliary_group'],row['score_space']):row for row in summaries}
    for candidate_id,details in detail_by_candidate.items():
        pairs=[{'problem_id':row['problem_id'],'problem_split':row['split'],'auxiliary_group':row['auxiliary_group']} for row in details]
        recomputed=bootstrap_groups([row['scores'] for row in details],pairs)
        c.require(len(recomputed)==9,'Diagnostic fit/validation/correctness groups changed')
        for expected in recomputed:
            key=(candidate_id,expected['split'],expected['auxiliary_group'],expected['score_space'])
            c.require(key in summary_lookup and all(summary_lookup[key].get(field)==value for field,value in expected.items()),
                      'Independent paired problem-bootstrap recomputation differs')
    return {'status':'verified','candidate_count':len(selected),'pairs_per_candidate':51,'summary_rows':len(summaries),
            'artifact_manifest_sha256':c.sha256_file(output/'artifact_manifest.json'),'files':len(manifest['files'])}


def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec',type=Path);group.add_argument('--verify',type=Path);a=p.parse_args()
    print(c.canonical(verify(a.verify) if a.verify else run(json.loads(a.spec.read_text()))))
if __name__=='__main__':main()
