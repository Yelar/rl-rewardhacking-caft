"""Fit-only numerical-repair diagnostics; never select or promote candidates.

Compare verified old/new directions, PCA subspaces, and correlations with the
original completion length. Cached raw integrity receipts avoid another full
native-payload scan; every opened raw file remains stat-identity bound.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from . import candidates as c, sweep
except ImportError:
    import candidates as c
    import sweep

OLD_ARTIFACT = "5dad334834d1647631f7791095c18c4d4fc592699c910f376448fe303d2cdc9e"


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    c.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid cosine inputs")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator < 1e-12 else float(np.clip(np.einsum('i,i->', a, b, optimize=False) / denominator, -1, 1))


def subspace_cosines(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    c.require(a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[1] > 0 and
              np.isfinite(a).all() and np.isfinite(b).all(), "Invalid paired PCA bases")
    for q in (a, b):
        c.require(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))) < 5e-5, "PCA basis is not orthonormal")
    return np.clip(np.linalg.svd(a.T @ b, compute_uv=False), 0, 1).tolist()


def nuisance_basis(rows):
    c.require(rows and all(r['problem_split'] == 'direction_fit' for r in rows), "Only fitting rows may enter repair diagnostic")
    columns = [np.ones(len(rows))]
    for key in ('problem_id_key', 'outcome_presence_class'):
        values = [str(r[key]) for r in rows]
        columns.extend(np.asarray([v == level for v in values], dtype=np.float64) for level in sorted(set(values))[1:])
    # The source ID prefix is the campaign's preserved legacy/new provenance.
    columns.append(np.asarray([r['record_id'].startswith('existing-') for r in rows], dtype=np.float64))
    design = np.stack(columns, axis=1)
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(s > max(design.shape) * np.finfo(float).eps * s[0]))
    return u[:, :rank]


def length_correlations(scores, lengths, basis):
    scores, lengths = np.asarray(scores, dtype=np.float64), np.asarray(lengths, dtype=np.float64)
    c.require(scores.ndim == 2 and scores.shape[0] == len(lengths) == len(basis) and
              np.isfinite(scores).all() and np.isfinite(lengths).all(), "Invalid length score matrix")
    residual_scores = scores - basis @ (basis.T @ scores)
    residual_lengths = lengths - basis @ (basis.T @ lengths)
    centered_lengths = lengths - lengths.mean()
    return [{'raw_pearson_r': cosine(scores[:, i] - scores[:, i].mean(), centered_lengths),
             'partial_r_problem_class_legacy_source': cosine(residual_scores[:, i], residual_lengths)}
            for i in range(scores.shape[1])]


def load_package(root, proof_path, proof_sha):
    root, proof_path = Path(root), Path(proof_path)
    c.require(c.sha256_file(proof_path) == proof_sha, "Candidate verification receipt changed")
    proof = json.loads(proof_path.read_text())
    c.require(proof['process_release_verified'] is True, "Candidate processes have not been independently released")
    catalog, artifact = sweep.load_verified_catalog(root, proof)
    reviewed = json.loads((root / 'reviewed_manifest.json').read_text())
    c.require(proof['run_token'] == reviewed['run_token'], "Candidate receipt/run identity mismatch")
    raw = Path(reviewed['raw_package'])
    c.require(c.sha256_file(raw / 'artifact_manifest.json') == reviewed['raw_manifest_sha256'], "Raw cache identity changed")
    raw_manifest = json.loads((raw / 'artifact_manifest.json').read_text())
    for name in ('activation_index.jsonl', 'extraction_summary.json'):
        c.verify_file(raw / name, raw_manifest['files'][name])
    c.verify_file(reviewed['prepared_records'], raw_manifest['files']['input/prepared_records.jsonl'])
    for key in ('prepared_records', 'exclusions', 'experiment_plan'):
        c.verify_file(reviewed[key], reviewed['bound_files'][reviewed[key]])
    rows, dataset = c.validate_records(c.read_jsonl(reviewed['prepared_records']), json.loads(Path(reviewed['exclusions']).read_text()))
    c.require(len(rows) == 333 and dataset['fitting_problems'] == 111, "Fitting population changed")
    receipt_path = root / 'raw_integrity_receipt.json'
    c.verify_file(receipt_path, artifact['files']['raw_integrity_receipt.json'])
    c.require(receipt_path.stat().st_uid == os.getuid() and receipt_path.stat().st_mode & 0o222 == 0, "Integrity receipt is not owned/read-only")
    integrity = json.loads(receipt_path.read_text())
    c.require(integrity['status'] == 'verified' and integrity['raw_manifest_sha256'] == reviewed['raw_manifest_sha256'] and
              integrity['prepared_records_sha256'] == c.sha256_file(reviewed['prepared_records']) and
              integrity['exclusion_manifest_sha256'] == c.sha256_file(reviewed['exclusions']) and
              set(integrity['fitting_record_ids']) == {r['record_id'] for r in rows}, "Raw integrity receipt input mismatch")
    summary = json.loads((raw / 'extraction_summary.json').read_text())
    reader = c.RawReader(raw, rows, c.read_jsonl(raw / 'activation_index.jsonl'), raw_manifest,
                         summary['manifest_sha256'], 36, 2560, integrity)
    return {'root': root, 'catalog': catalog, 'proof': proof, 'reviewed': reviewed, 'rows': rows, 'reader': reader}


def reports_by_region(package):
    result = {}
    for filename in sorted({item['report_file'] for item in package['catalog']}):
        report = json.loads(c.safe_child(package['root'], filename).read_text())
        key = (report['layer'], report['window'])
        c.require(key not in result, "Duplicate candidate region report")
        result[key] = (filename, report)
    c.require(set(result) == {(layer, window) for layer in range(36) for window in c.WINDOWS}, "Missing candidate region coverage")
    return result


def fitting_contexts(report, fit_ids):
    result = []
    for pc in report['pc_contexts']:
        c.require(pc['fitting_contexts_only'] is True, "Context is not explicitly fitting-only")
        for sign in ('negative', 'positive'):
            c.require(all(row['record_id'] in fit_ids for row in pc[sign]), "Held-out token context entered diagnostic")
        result.append(pc)
    return result


def row_means(reader, rows, layer, window):
    """Bound memory to three [333,2560] arrays plus one token window."""
    means = {key: np.empty((len(rows), 2560), dtype=np.float32) for key in ('h60', 'delta')}
    for i, row in enumerate(rows):
        c.require(row['problem_split'] == 'direction_fit', "Held-out raw access is forbidden")
        positions = c.valid_positions(row, window)
        h0 = reader.read(row, 'h0', layer, positions)
        h60 = reader.read(row, 'h60', layer, positions)
        means['h60'][i] = h60.mean(0, dtype=np.float32)
        means['delta'][i] = np.subtract(h60, h0, dtype=np.float32).mean(0, dtype=np.float32)
    return means


def alignment_rows(layer, window, a, b):
    result=[]
    for family in c.FAMILIES:
        for kind in ('v0', 'v60', 'v_change'):
            key=family+'.'+kind
            result.append({'layer':layer,'window':window,'family':family,'kind':kind,
                'old_new_cosine':cosine(a[key],b[key]) if key in a and key in b else None,
                'old_raw_norm':float(np.linalg.norm(a[family+'.raw_'+kind].astype(np.float64))),
                'new_raw_norm':float(np.linalg.norm(b[family+'.raw_'+kind].astype(np.float64)))})
    result.append({'layer':layer,'window':window,'family':'pca','kind':'subspace',
        'principal_cosines':subspace_cosines(a['pca.pcs'],b['pca.pcs']),
        'interpretation':'Basis rotations/signs do not change a subspace; individual PC matching is not assumed.'})
    return result


def row_metadata(rows):
    return [{key:r[key] for key in ('record_id','problem_id_key','outcome_presence_class','problem_split','completion_token_count')}
            for r in rows]


def run(spec):
    from safetensors.numpy import load_file, save_file
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', "Repair diagnostic must hide CUDA")
    c.require(0 < spec.get('runtime_seconds', 1200) <= 3600, "Diagnostic deadline exceeds one CPU hour")
    start = time.monotonic()
    numeric = c.numerical_preflight()
    packages = {tag: load_package(spec[tag + '_root'], spec[tag + '_verification'], spec[tag + '_verification_sha256']) for tag in ('old', 'new')}
    old, new = packages['old'], packages['new']
    c.require(old['proof']['artifact_manifest_sha256'] == OLD_ARTIFACT and new['proof']['artifact_manifest_sha256'] != OLD_ARTIFACT,
              "Repair diagnostic requires original and distinct repaired packages")
    c.require(old['rows'] == new['rows'], "Old/new fitting population or original tokens differ")
    plan = json.loads(Path(new['reviewed']['experiment_plan']).read_text())
    c.require(plan['plan_version'] == 2 and plan['inputs']['raw_package'] == new['reviewed']['raw_package'] and
              plan['inputs']['raw_artifact_manifest_sha256'] == new['reviewed']['raw_manifest_sha256'] and
              plan['amendment']['superseded_candidates_status'] == 'diagnostic_only_do_not_promote', "Repaired plan lineage is absent")
    output = Path(spec['output'])
    c.require(not output.exists(), "Diagnostic output must be fresh")
    input_roots=[root for p in packages.values() for root in (p['root'],Path(p['reviewed']['raw_package']))]
    c.require(all(not output.resolve().is_relative_to(root.resolve()) for root in input_roots), "Diagnostic cannot write into candidate/raw package")
    rows = new['rows']; fit_ids = {row['record_id'] for row in rows}
    lengths = np.asarray([row['completion_token_count'] for row in rows], dtype=np.float64)
    basis = nuisance_basis(rows)
    reports = {tag: reports_by_region(p) for tag, p in packages.items()}
    output.mkdir(parents=True)
    c.write_json(output / 'resolved_spec.json', spec)
    c.write_json(output / 'source_sha256.json', {Path(module.__file__).name:c.sha256_file(module.__file__)
                                                for module in (c, sweep)} | {Path(__file__).name:c.sha256_file(__file__)})
    c.write_json(output / 'fitting_rows.json', row_metadata(rows))
    alignments, stability, contexts, correlations = [], [], [], []
    score_tensors = {}
    try:
        for layer in range(36):
            for window in c.WINDOWS:
                c.require(time.monotonic() - start < spec.get('runtime_seconds', 1200), "Diagnostic deadline exceeded")
                tensor_sets = {}
                for tag, package in packages.items():
                    filename, report = reports[tag][layer, window]
                    tensor_sets[tag] = load_file(c.safe_child(package['root'], filename.replace('.json', '.safetensors')))
                    stability.append({'cache':tag, 'layer':layer, 'window':window, 'mean_candidates':report['mean_candidates'], 'pca':report['pca']})
                    contexts.append({'cache':tag, 'layer':layer, 'window':window, 'fitting_contexts_only':True,
                                     'pc_contexts':fitting_contexts(report, fit_ids)})
                a, b = tensor_sets['old'], tensor_sets['new']
                alignments.extend(alignment_rows(layer,window,a,b))
                for tag, package in packages.items():
                    selected = sorted((item for item in package['catalog'] if item['layer']==layer and item['window']==window and
                        (item['kind'] in ('v60','v_change') or item['kind']=='pc')), key=lambda item:item['candidate_id'])
                    means = row_means(package['reader'], rows, layer, window)
                    scores = []
                    for item in selected:
                        q = tensor_sets[tag][item['tensor_key']]
                        if 'column_indices' in item: q = q[:, item['column_indices']]
                        c.require(q.shape == (2560,1), "Expected individual candidate column")
                        source = means['h60' if item['kind']=='v60' else 'delta']
                        scores.append(source.astype(np.float64) @ q[:,0].astype(np.float64))
                    scores = np.stack(scores, axis=1)
                    score_key = f'{tag}.L{layer:02d}.{window}'
                    score_tensors[score_key] = scores
                    for column, (item, stats) in enumerate(zip(selected, length_correlations(scores, lengths, basis))):
                        correlations.append({'cache':tag, 'candidate_id':item['candidate_id'], 'layer':layer, 'window':window,
                            'score_tensor':score_key, 'column':column, 'score_space':'h60' if item['kind']=='v60' else 'delta', **stats})
                    del means
            print(c.canonical({'completed_layer':layer,'elapsed_seconds':time.monotonic()-start}), flush=True)
        save_file(score_tensors, output/'fitting_projection_scores.safetensors')
        for name, value in [('alignment',alignments),('stability',stability),('fitting_token_contexts',contexts),('length_correlations',correlations)]:
            c.write_json(output/(name+'.json'), value)
        success={'status':'succeeded','fit_problems':111,'fit_records':333,'layer_windows':144,'cuda_hidden':True,
            'held_out_raw_files_opened':0,'old_candidates_used_for_selection':False,'elapsed_seconds':time.monotonic()-start,
            'numerical_preflight':numeric,'old_artifact_sha256':old['proof']['artifact_manifest_sha256'],
            'new_artifact_sha256':new['proof']['artifact_manifest_sha256'],
            'limitations':['Fitting-only descriptive comparisons; no behavioral efficacy or held-out claim.',
                'Length associations persist for semantic and syntactic reasons; partial correlations do not establish causal artifacts.',
                'PCA bootstrap is conditional on its fitted randomized range; independent-half PCA fits use the full space.',
                'v_change/PC scores use per-token FP32 differences before averaging; scalar diagnostic reductions use FP64.',
                'Old candidates remain diagnostic only; no layer, strength, sign, or candidate selection occurs here.']}
        c.write_json(output/'SUCCESS.json',success)
    except BaseException as error:
        c.write_json(output/'FAILURE.json',{'error':str(error),'partial_outputs_retained':True}); raise
    files={str(p.relative_to(output)):{'sha256':c.sha256_file(p),'size_bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
    c.write_json(output/'artifact_manifest.json',{'algorithm':'sha256','files':files})
    for path in output.rglob('*'):
        if path.is_file(): path.chmod(0o400)
    return success


def verify(output):
    """Independently hash outputs and recompute every persisted correlation."""
    from safetensors.numpy import load_file
    output = Path(output)
    artifact = json.loads((output/'artifact_manifest.json').read_text())
    c.require(artifact['algorithm']=='sha256' and not (output/'FAILURE.json').exists(), 'Incomplete repair diagnostic')
    for name, info in artifact['files'].items(): c.verify_file(c.safe_child(output,name),info)
    success=json.loads((output/'SUCCESS.json').read_text())
    c.require(success['status']=='succeeded' and success['old_candidates_used_for_selection'] is False and
              success['held_out_raw_files_opened']==0 and success['layer_windows']==144 and
              success['old_artifact_sha256']==OLD_ARTIFACT and success['new_artifact_sha256']!=OLD_ARTIFACT,
              'Repair diagnostic scope changed')
    rows=json.loads((output/'fitting_rows.json').read_text())
    c.require(len(rows)==333 and len({r['record_id'] for r in rows})==333 and
              len({r['problem_id_key'] for r in rows})==111, 'Repair fitting population differs')
    spec=json.loads((output/'resolved_spec.json').read_text())
    packages={tag:load_package(spec[tag+'_root'],spec[tag+'_verification'],spec[tag+'_verification_sha256']) for tag in ('old','new')}
    c.require(packages['old']['rows']==packages['new']['rows'] and rows==row_metadata(packages['new']['rows']),
              'Persisted fitting metadata differ from both bound sources')
    for tag in packages:
        c.require(success[tag+'_artifact_sha256']==packages[tag]['proof']['artifact_manifest_sha256'], 'Persisted candidate identity changed')
    basis=nuisance_basis(rows);lengths=np.asarray([r['completion_token_count'] for r in rows])
    scores=load_file(output/'fitting_projection_scores.safetensors')
    correlations=json.loads((output/'length_correlations.json').read_text())
    by_key={}
    for row in correlations:
        key=(row['score_tensor'],row['column'])
        c.require(key not in by_key,'Duplicate correlation identity');by_key[key]=row
    c.require(len(scores)==288 and set(by_key)=={(key,i) for key,value in scores.items() for i in range(value.shape[1])},
              'Persisted fitting score/correlation coverage mismatch')
    for key, values in scores.items():
        for i, stats in enumerate(length_correlations(values,lengths,basis)):
            c.require(all(by_key[key,i][name]==value for name,value in stats.items()), 'Independent correlation recomputation differs')
    fit_ids={r['record_id'] for r in rows}
    contexts=json.loads((output/'fitting_token_contexts.json').read_text())
    c.require(len(contexts)==288, 'Incomplete fit context coverage')
    for item in contexts: fitting_contexts(item,fit_ids)
    alignment=json.loads((output/'alignment.json').read_text())
    expected_keys={(l,w,f,k) for l in range(36) for w in c.WINDOWS for f in c.FAMILIES for k in ('v0','v60','v_change')}
    expected_keys|={(l,w,'pca','subspace') for l in range(36) for w in c.WINDOWS}
    c.require(len(alignment)==1872 and {(a['layer'],a['window'],a['family'],a['kind']) for a in alignment}==expected_keys,
              'Incomplete direction/subspace comparison coverage')
    reports={tag:reports_by_region(p) for tag,p in packages.items()}
    expected_alignment=[];expected_stability=[];expected_contexts=[]
    for layer in range(36):
        for window in c.WINDOWS:
            tensors={}
            for tag,package in packages.items():
                filename,report=reports[tag][layer,window]
                tensors[tag]=load_file(c.safe_child(package['root'],filename.replace('.json','.safetensors')))
                expected_stability.append({'cache':tag,'layer':layer,'window':window,'mean_candidates':report['mean_candidates'],'pca':report['pca']})
                expected_contexts.append({'cache':tag,'layer':layer,'window':window,'fitting_contexts_only':True,
                    'pc_contexts':fitting_contexts(report,fit_ids)})
            expected_alignment.extend(alignment_rows(layer,window,tensors['old'],tensors['new']))
    c.require(alignment==expected_alignment and contexts==expected_contexts and
              json.loads((output/'stability.json').read_text())==expected_stability,
              'Comparison/stability/context copies differ from bound source tensors/reports')
    return {'status':'verified','correlations':len(correlations),'layer_windows':144,'fitting_only':True,
            'artifact_manifest_sha256':c.sha256_file(output/'artifact_manifest.json')}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec',type=Path);group.add_argument('--verify',type=Path);args=parser.parse_args()
    print(c.canonical(verify(args.verify) if args.verify else run(json.loads(args.spec.read_text()))))
