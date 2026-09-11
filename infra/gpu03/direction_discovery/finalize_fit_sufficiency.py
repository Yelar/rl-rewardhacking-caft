#!/usr/bin/env python3
"""Verify separately refined PCA outputs and publish a consolidated summary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import candidates as c
import fit_sufficiency as s
import refine_fit_sufficiency as r
import summarize_fit_sufficiency as a


def main(root):
    proof=json.loads((root/'audit_initial/independent_verification.json').read_text())
    c.require(proof['status']=='passed','initial independent audit did not pass')
    c.require(c.sha256_file(root/'audit_initial/artifact_manifest.json')==proof['artifact_manifest_sha256'], 'initial artifact manifest changed')
    parent_digest=c.sha256_file(root/'control/study_manifest.json')
    parent=json.loads((root/'control/study_manifest.json').read_text())
    mp=root/'refinement_control/manifest.json';digest=c.sha256_file(mp);manifest=json.loads(mp.read_text())
    c.require(manifest['parent_manifest_sha256']==parent_digest,'refinement parent mismatch')
    unit=parent['run_token']+'-refine'
    state=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','MainPID'],text=True).splitlines())
    c.require(state['ActiveState']=='inactive' and state['SubState']=='dead' and state['Result']=='success' and
              state['ExecMainStatus']=='0' and state['MainPID']=='0','refinement has not positively exited successfully')
    for relative,bound in manifest['inputs'].items():c.verify_file(c.safe_child(root,relative),bound)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:value=path.read_bytes()
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        c.require(str(root/'analysis_source/refine_fit_sufficiency.py').encode() not in value,'refinement worker still alive')
    cells=a.load_cells(root,parent_digest);lookup={(x['layer'],x['region'],x['method']):x for x in cells}
    verified_matrices=0;max_orth=0.;details=[];files={}
    for spec in manifest['cells']:
        key=(spec['layer'],spec['region'],spec['method']);stem=root/'refinement_results'/f'layer_{key[0]:02d}__{key[1]}__{key[2]}'
        report=json.loads(stem.with_suffix('.json').read_text())
        c.require(report['parent_cell_sha256']==spec['parent_cell_sha256'] and report['refinement_manifest_sha256']==digest,'refined cell provenance mismatch')
        c.require(not r.needs_refinement(report),'numerical warning remains after recheck')
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as arrays:
            c.require(len(arrays.files)==12,'wrong refined vector count')
            for name in arrays.files:
                value=arrays[name];shape=(2560,10) if name in ('full_pcs','solver_repeat_pcs') else (12,2,2560,10)
                c.require(value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all(),'invalid refined vectors')
                for matrix in value.reshape(-1,2560,10):
                    v=matrix.astype(np.float64);error=float(np.max(np.abs(v.T@v-np.eye(10))));max_orth=max(max_orth,error)
                    c.require(error<5e-5,'refined basis not orthonormal');verified_matrices+=1
                if name not in ('full_pcs','solver_repeat_pcs'):
                    group=report['pca']['groups'][name[:-4]]
                    for pair,expected in zip(value,group['pair_details']):
                        actual=s.compare_pcs(pair[0],pair[1])
                        c.require(np.allclose(actual['same_index_absolute_cosine'],expected['same_index_absolute_cosine'],atol=1e-12,rtol=0),'refined pair metrics disagree')
        lookup[key]['pca']=report['pca'];lookup[key]['numerical_refinement_manifest_sha256']=digest
        details.append({'layer':key[0],'region':key[1],'method':key[2],
            'minimum_same_data_solver_cosine':min(report['pca']['solver_repeat_agreement']['same_index_absolute_cosine']),
            'residual_maxima_p975':report['pca']['all_resample_max_relative_residual']['p975']})
    c.require(verified_matrices==manifest['expected_additional_pca_fits']==726,'wrong refinement fit count')
    summary=a.summarize(cells,parent);c.require(not summary['numerical_refinement_candidates'],'unresolved PCA numerics')
    summary['numeric_rechecks']=details
    summary['total_verified_pca_fits']=proof['pca_matrices_verified']+verified_matrices
    output=root/'final_audit';output.mkdir(exist_ok=False)
    c.write_json(output/'summary.json',summary)
    c.write_json(output/'independent_verification.json',{'status':'passed','original_independent_verification_sha256':c.sha256_file(root/'audit_initial/independent_verification.json'),
        'refinement_manifest_sha256':digest,'refined_cells':3,'refined_pca_matrices_verified':verified_matrices,
        'total_verified_pca_fits':summary['total_verified_pca_fits'],'max_refined_orthonormality_error':max_orth,
        'actual_refinement_service_state':state,'owned_refinement_workers_released':True,'remaining_numerical_flags':0})
    for folder in ['refinement_control','refinement_results','supplementary_diagnostics','analysis_source','finalization_source','audit_initial','final_audit']:
        for path in sorted((root/folder).iterdir()):
            if path.is_file():files[str(path.relative_to(root))]={'sha256':c.sha256_file(path),'size_bytes':path.stat().st_size}
    c.write_json(output/'artifact_manifest.json',{'original_package_artifact_manifest_sha256':proof['artifact_manifest_sha256'],'files':files})
    print(c.canonical({'status':'passed','total_verified_pca_fits':summary['total_verified_pca_fits'],'remaining_numeric_flags':0,'refinement':details}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
