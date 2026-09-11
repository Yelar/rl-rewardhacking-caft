#!/usr/bin/env python3
"""Separate, manifest-bound numerical rechecks; original fits remain immutable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

try:
    from . import candidates as c
    from . import fit_sufficiency as s
except ImportError:
    import candidates as c
    import fit_sufficiency as s


def needs_refinement(cell):
    p=cell['pca']
    return bool(p['approximation_warning'] or
                min(p['solver_repeat_agreement']['same_index_absolute_cosine'])<.99 or
                max(p['full']['relative_covariance_residuals'])>.05 or
                max(p['solver_repeat']['relative_covariance_residuals'])>.05)


def run(manifest_path,digest,index):
    c.require(c.sha256_file(manifest_path)==digest,'refinement manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text());root=Path(manifest['root'])
    c.require(0<=index<len(manifest['cells']),'unreviewed cell index')
    spec=manifest['cells'][index]
    for relative,bound in manifest['inputs'].items():
        c.verify_file(c.safe_child(root,relative),bound)
    parent=json.loads((root/'control/study_manifest.json').read_text())
    source=root/'results'/f"layer_{spec['layer']:02d}"/f"{spec['region']}__{spec['method']}.json"
    c.require(c.sha256_file(source)==spec['parent_cell_sha256'],'parent cell changed')
    c.require(needs_refinement(json.loads(source.read_text())),'cell not selected by numerical rule')
    meta=root/'source_metadata'
    rows,inventory=c.validate_records(c.read_jsonl(meta/'prepared_records.jsonl'),json.loads((meta/'exclusions.json').read_text()))
    c.require(len(rows)==333 and inventory['fitting_problems']==111,'fitting cohort changed')
    reader=c.RawReader(root/'raw',rows,c.read_jsonl(meta/'activation_index.jsonl'),
        json.loads((meta/'raw_artifact_manifest.json').read_text()),s.SOURCE_DIGEST,36,2560,
        json.loads((root/'control/copy_verification.json').read_text()))
    start=time.monotonic()
    ps=[s.positions(row,spec['region'],spec['method']) for row in rows]
    offsets=np.cumsum([0]+[len(x) for x in ps])
    h0=np.concatenate([reader.read(row,'h0',spec['layer'],p) for row,p in zip(rows,ps)])
    h60=np.concatenate([reader.read(row,'h60',spec['layer'],p) for row,p in zip(rows,ps)])
    data=c.WindowData(rows,ps,h0,h60,offsets,np.repeat(np.arange(len(rows)),np.diff(offsets)),
        np.asarray([p for positions in ps for p in positions]))
    weights,token_problem,ids=c.balanced_token_weights(data)
    with np.load(root/'control/sampling_counts.npz',allow_pickle=False) as npz:
        groups={k:npz[k] for k in npz.files}
    report,arrays=s.pca_study(data.delta,token_problem,weights*111,groups,parent['pca_pair_repeats'],
        c.stable_seed(parent['seed'],spec['layer'],spec['region'],spec['method']),manifest['pca_config'])
    output=Path(manifest['output']);stem=f"layer_{spec['layer']:02d}__{spec['region']}__{spec['method']}"
    with (output/(stem+'.npz')).open('xb') as stream:np.savez(stream,**arrays)
    c.write_json(output/(stem+'.json'),{'layer':spec['layer'],'region':spec['region'],'method':spec['method'],
        'parent_cell_sha256':spec['parent_cell_sha256'],'refinement_manifest_sha256':digest,
        'seconds':time.monotonic()-start,'pca':report})
    print(c.canonical({'index':index,'cell':stem,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();run(a.manifest,a.manifest_sha256,a.index)
