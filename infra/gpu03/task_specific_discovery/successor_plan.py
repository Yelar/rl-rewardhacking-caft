"""Approved master-6008 successor preparation; no model, evaluation or dispatch.

Old requests/claims stay untouched. The original finite numerical qualification
is independently re-audited from its retained files, never regenerated. A new
plan changes only approved seeds/IDs and operational token/provenance bindings.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import os
from pathlib import Path
import socket

from . import build_causal_plan as original
from . import prepare_causal_waves as old_waves
from . import causal_qualification_v2 as audit_io
from . import causal_worker as worker
from .io_contract import canonical, hash_value

ROOT = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-pilot-20260922-113907'
TOKEN = Path(ROOT).name
SOURCE_ROOT = ROOT + '/source-successor-v1'
ORIGINAL_ROOT = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
ORIGINAL_PLAN_SHA = 'e9afc37f8186f9a2e7a69c4aaee40232b160dc831068c3bd2cc4adbdd87e3023'
PROPOSAL_SHA = 'd6e2e906caad20230cab0fa849b5ddc0067f8ede98c5b470f3a7caadbf946bd4'
QUALIFIED_SHA = 'ec85da9b150d83811638e7b0fe5ea88b9f1281cbc33663a8cac08875fbb4c25a'
DOMAIN = 'task-specific-causal-successor-proposal-20260922-v1'
SCHEMA = 'task_specific_approved_successor_v1'
ADMISSION_SCHEMA = 'task_specific_successor_numerical_admission_v1'
require = worker.require
binding = old_waves.binding
read_ref = old_waves.read_ref


def paired_seed(problem, sample, master):
    require(type(sample) is int and sample in (0, 1), 'Invalid seed sample coordinate')
    return int.from_bytes(hashlib.sha256(canonical([master, str(problem), 'primary', None, sample])).digest()[:8], 'big') % (2**31 - 1)


def derive_requests(plan):
    rows = plan['pilot_requests']
    require(len(rows) == 15168 and len({r['request_id'] for r in rows}) == 15168,
            'Incomplete or duplicate original pilot')
    fresh = []
    old_seeds = set()
    for old in rows:
        require(old['scope'] == 'primary' and old['seed'] == paired_seed(old['problem_id'], old['sample_index'], 6007),
                'Original seed/scope differs')
        new = copy.deepcopy(old)
        new['seed'] = paired_seed(old['problem_id'], old['sample_index'], 6008)
        new['request_id'] = hash_value({'domain': DOMAIN, 'original_request_id': old['request_id'], 'new_seed': new['seed']})
        fresh.append(new); old_seeds.add(old['seed'])
    seeds = {r['seed'] for r in fresh}
    old_ids = {r['request_id'] for r in rows + plan['qualification_requests']}
    require(len(seeds) == len(old_seeds) == 64 and not seeds & old_seeds and
            len({r['request_id'] for r in fresh}) == 15168 and not {r['request_id'] for r in fresh} & old_ids,
            'Successor seeds/IDs are not unique and disjoint')
    return fresh


def authorization(value, proposal_ref, root):
    expected = {'schema':'task_specific_successor_direct_authorization_v1', 'status':'approved',
        'host':'gpu-03', 'remote_root':str(root), 'run_token':Path(root).name,
        'proposal_sha256':proposal_ref['sha256'], 'proposal_size_bytes':proposal_ref['size_bytes'],
        'seed_master':6008, 'requests':15168, 'reuse_qualification_requests':0,
        'generation_window_seconds':86400, 'wave_work_seconds':14400,
        'evaluation_work_seconds':14400, 'numerical_admission_required':True}
    require(all(value.get(k) == v and type(value.get(k)) is type(v) for k,v in expected.items()),
            'Explicit successor authorization differs')
    require(value.get('direct_user_message') == 'Approved.' and
            set(value.get('excluded', [])) >= {'RL','other hosts','AWS compute','old request replay',
                                             'old deadline extension','gate relaxation'},
            'Successor authorization provenance/exclusions missing')


def validate_proposal(proposal, plan, fresh):
    require(proposal['schema'] == 'task_specific_causal_successor_local_proposal_v1' and
            proposal['proposal_id'] == DOMAIN and proposal['pilot_requests'] == fresh and
            proposal['request_budget'] == {'fresh_pilot_requests':15168,'reused_old_controls':0,
                'conditions':79,'problems':32,'protocols':3,'samples_per_problem_protocol_condition':2},
            'Approved proposal request universe differs')
    excluded = {'schema','status','run_token','pilot_requests','qualification_requests','budgets',
                'generation','evaluation','training','qualification_contract','qualification_condition_ids'}
    require(proposal['retained_original_science'] == {k:v for k,v in plan.items() if k not in excluded},
            'Approved proposal changes retained original science')
    require(proposal['seed_policy']['original_master'] == 6007 and proposal['seed_policy']['proposed_master'] == 6008,
            'Approved proposal seed masters differ')
    batches = [fresh[i:i+8] for i in range(0,len(fresh),8)]
    require(proposal['batch_policy']['batch_sha256'] == [hash_value(b) for b in batches] and
            proposal['batch_policy']['size'] == 8 and proposal['batch_policy']['count'] == 1896,
            'Approved proposal physical batch order differs')


def operational_source(ref, evidence, root):
    source = Path(root) / 'source-successor-v1'
    require(Path(ref['path']) == source / 'SOURCE_FILES.json', 'Wrong successor source inventory location')
    inventory = read_ref(ref, evidence)
    required = {'infra/gpu03/task_specific_discovery/'+n for n in
                ('successor_plan.py','prepare_successor_waves.py','bind_successor_wave.py')}
    require(required <= set(inventory), 'Successor operational source inventory incomplete')
    for name, value in inventory.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
                'Nonrelative successor operational source')
        evidence.bind(source / name, value)
    # The imported admission helper itself must be the frozen source, not a
    # same-named helper loaded from a different writable worktree.
    own = Path(__file__).resolve(); relative = 'infra/gpu03/task_specific_discovery/successor_plan.py'
    evidence.bind(own, inventory[relative])
    return inventory


def reaudit_numerical(qualified_ref, evidence):
    """Recompute finite qualification from closed bytes, without external calls."""
    proof = read_ref(qualified_ref, evidence)
    require(proof['requests'] == 312 and proof['auditor_version'] == audit_io.AUDITOR_VERSION,
            'Original numerical proof is not the complete V2 audit')
    helper = proof['audit_io_provenance']['helper']
    evidence.bind(Path(audit_io.__file__).resolve(), binding(helper))
    manifest = proof['manifest']
    fresh = audit_io.audit(manifest['path'], manifest['sha256'])
    # Audit helper location/provenance can differ in an additive snapshot; every
    # scientific result, producer identity and independently hashed file stays equal.
    omitted = {'audit_io_provenance'}
    require({k:v for k,v in fresh.items() if k not in omitted} ==
            {k:v for k,v in proof.items() if k not in omitted},
            'Independent numerical re-audit differs from the original finite proof')
    for path, value in fresh['independently_hashed_files'].items():
        evidence.bind(path, binding(value))
    evidence.unchanged()
    return fresh


def plan_claim(root, out, refs, source_inventory_ref, requests):
    return {'schema':'task_specific_successor_plan_binding_v1',
        'run_token':Path(root).name,**refs,'operational_source_inventory':source_inventory_ref,
        'output':str(out),'new_request_ids_sha256':hash_value([r['request_id'] for r in requests]),
        'generation_performed':False,'dispatch_authorized_by_this_file':False}


def pinned_refs(original_ref, qualified_ref, proposal_ref):
    require(original_ref['sha256'] == ORIGINAL_PLAN_SHA and qualified_ref['sha256'] == QUALIFIED_SHA and
            proposal_ref['sha256'] == PROPOSAL_SHA, 'Evidence does not match the directly approved predecessor/proposal')


def source_admission_fields(plan):
    return {k:copy.deepcopy(plan[k]) for k in ('model_revision','model_snapshot','checkpoint','prepared_rows',
        'vectors','sampling','batch_size','profile','source_root','qualified_source_files','conditions',
        'qualification_contract')}


def assemble_plan(old, fresh, provenance, extra_bindings):
    plan = copy.deepcopy(old)
    plan['run_token'] = Path(provenance['root']).name
    plan['pilot_requests'] = fresh
    plan['budgets'] = {'qualification':312,'pilot_including_reuse':15168,'reusable_pilot_requests':0}
    plan['successor'] = {k:v for k,v in provenance.items() if k != 'root'}
    plan['bound_files'].update(extra_bindings)
    return plan


def validate_admission(receipt, old, refs, evidence):
    require(receipt['schema'] == ADMISSION_SCHEMA and receipt['status'] == 'admitted_unchanged_finite_numerical_profile' and
            all(receipt[k] == refs[k] for k in ('original_plan','qualification','proposal','authorization')) and
            receipt['retained_science'] == source_admission_fields(old) and receipt['new_generation_master'] == 6008 and
            receipt['new_pilot_requests'] == 15168 and receipt['old_qualification_requests_reused_as_pilot'] == 0 and
            receipt['model_generation_performed'] is False and receipt['evaluation_performed'] is False,
            'Successor numerical admission receipt differs')
    original_proof = read_ref(refs['qualification'], evidence)
    fresh = read_ref(receipt['independent_numerical_reaudit'], evidence)
    require({k:v for k,v in fresh.items() if k != 'audit_io_provenance'} ==
            {k:v for k,v in original_proof.items() if k != 'audit_io_provenance'},
            'Recorded independent numerical re-audit differs')
    for path,value in fresh['independently_hashed_files'].items(): evidence.bind(path,binding(value))
    evidence.bind(Path(audit_io.__file__).resolve(), binding(original_proof['audit_io_provenance']['helper']))


def checked_plan(plan_ref, qualified_ref, evidence):
    """Validate successor plus original evidence; return (plan, {}, extra bindings)."""
    plan = read_ref(plan_ref,evidence); s = plan['successor']
    require(set(s) == {'schema','original_plan','qualification','proposal','authorization','numerical_admission',
                      'operational_source_root','operational_source_inventory','plan_binding'},
            'Unknown or missing successor provenance')
    pinned_refs(s['original_plan'], qualified_ref, s['proposal'])
    require(s['schema'] == SCHEMA and plan['run_token'] == Path(s['operational_source_root']).parent.name and
            Path(s['operational_source_root']).name == 'source-successor-v1', 'Successor operational scope differs')
    root = Path(s['operational_source_root']).parent
    old, old_reuse, extra = old_waves.checked_plan(s['original_plan'],qualified_ref,evidence)
    require(len(old_reuse) == 168 and qualified_ref == s['qualification'], 'Original proof binding/reuse differs')
    fresh = derive_requests(old)
    proposal = read_ref(s['proposal'],evidence); auth = read_ref(s['authorization'],evidence)
    authorization(auth,s['proposal'],root); validate_proposal(proposal,old,fresh)
    operational_source(s['operational_source_inventory'],evidence,root)
    refs = {k:s[k] for k in ('original_plan','qualification','proposal','authorization')}
    admission = read_ref(s['numerical_admission'],evidence)
    validate_admission(admission,old,refs,evidence)
    claim = read_ref(s['plan_binding'], evidence)
    require(Path(s['plan_binding']['path']) == root / 'SUCCESSOR_PLAN_BINDING.json' and
            claim == plan_claim(root, Path(plan_ref['path']).parent, refs, s['operational_source_inventory'], fresh),
            'Successor exclusive plan claim differs')
    required_extra = {ref['path']:binding(ref) for ref in refs.values()}
    required_extra[s['plan_binding']['path']] = binding(s['plan_binding'])
    required_extra[s['numerical_admission']['path']] = binding(s['numerical_admission'])
    required_extra[s['operational_source_inventory']['path']] = binding(s['operational_source_inventory'])
    required_extra[admission['independent_numerical_reaudit']['path']] = binding(admission['independent_numerical_reaudit'])
    expected = assemble_plan(old,fresh,{'root':str(root),**s},required_extra)
    require(plan == expected, 'Successor PLAN contains unapproved science/provenance changes')
    for path,value in plan['bound_files'].items(): evidence.bind(path,value)
    evidence.unchanged()
    return plan, {}, {**extra,**required_extra}


def prepare(original_ref, qualified_ref, proposal_ref, authorization_ref, source_inventory_ref, output):
    """Write new plan and admission, preserving every old file and claim."""
    out = Path(output); root = out.parent
    require(out.is_absolute() and out.parent.resolve()==out.parent and not out.exists() and not out.is_symlink(),
            'Fresh canonical successor plan output required')
    require(out.name == 'causal-plan-v1' and not (root/'SUCCESSOR_PLAN_BINDING.json').exists() and
            not (root/'SUCCESSOR_PLAN_BINDING.json').is_symlink(), 'Successor plan preparation already consumed')
    pinned_refs(original_ref, qualified_ref, proposal_ref)
    evidence = audit_io.Evidence()
    old, reused, extra = old_waves.checked_plan(original_ref,qualified_ref,evidence)
    require(len(reused)==168,'Original qualification panel changed')
    proposed = read_ref(proposal_ref,evidence); auth = read_ref(authorization_ref,evidence)
    authorization(auth,proposal_ref,root)
    requests = derive_requests(old); validate_proposal(proposed,old,requests)
    operational_source(source_inventory_ref,evidence,root)
    fresh_proof = reaudit_numerical(qualified_ref,evidence)
    require(old['profile']==worker.PROFILE and old['source_root'] != str(root/'source-successor-v1'),
            'Generation must use the original qualified worker snapshot')
    evidence.unchanged()
    refs={'original_plan':original_ref,'qualification':qualified_ref,'proposal':proposal_ref,'authorization':authorization_ref}
    worker.write(root/'SUCCESSOR_PLAN_BINDING.json', plan_claim(root,out,refs,source_inventory_ref,requests))
    claim_ref=original.ref(root/'SUCCESSOR_PLAN_BINDING.json')
    out.mkdir(mode=0o700)
    proof_path=out/'INDEPENDENT_NUMERICAL_REAUDIT.json';worker.write(proof_path,fresh_proof)
    fresh_ref=original.ref(proof_path)
    receipt={'schema':ADMISSION_SCHEMA,'status':'admitted_unchanged_finite_numerical_profile',**refs,
        'independent_numerical_reaudit':fresh_ref,'retained_science':source_admission_fields(old),
        'new_generation_master':6008,'new_pilot_requests':15168,'old_qualification_requests_reused_as_pilot':0,
        'model_generation_performed':False,'evaluation_performed':False,
        'finite_qualification_limit':'The original312 requests qualify this unchanged numerical profile; no claim of null parity on new seeds or causal efficacy.',
        'request_seed_api':'Original immutable worker accepts supplied per-request seeds; all other model/vector/source/profile inputs remain identical.'}
    admission_path=out/'NUMERICAL_ADMISSION.json';worker.write(admission_path,receipt);admission_ref=original.ref(admission_path)
    provenance={'schema':SCHEMA,'root':str(root),**refs,'numerical_admission':admission_ref,
        'operational_source_root':str(root/'source-successor-v1'),'operational_source_inventory':source_inventory_ref,
        'plan_binding':claim_ref}
    extra_bound={r['path']:binding(r) for r in (*refs.values(),admission_ref,fresh_ref,source_inventory_ref,claim_ref)}
    plan=assemble_plan(old,requests,provenance,extra_bound)
    path=out/'PLAN.json';worker.write(path,plan)
    evidence.unchanged()
    return original.ref(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    names=('original-plan','qualification','proposal','authorization','source-inventory')
    for name in names:
        parser.add_argument('--'+name,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    require(socket.gethostname()=='gpu-03' and os.getuid()==3357 and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Successor preparation requires accepted owner CPU-only environment')
    require(Path(args.output)==Path(ROOT)/'causal-plan-v1','Wrong successor output')
    refs=[]
    for name in names:
        ref=original.ref(getattr(args,name.replace('-','_')))
        require(ref['sha256']==getattr(args,name.replace('-','_')+'_sha256'),'Explicit input SHA differs')
        refs.append(ref)
    require(refs[0]['sha256']==ORIGINAL_PLAN_SHA and refs[1]['sha256']==QUALIFIED_SHA and
            refs[2]['sha256']==PROPOSAL_SHA,'Not the directly approved original/proof/proposal')
    require(refs[0]['path']==ORIGINAL_ROOT+'/causal-plan-v1/PLAN.json' and
            refs[1]['path']==ORIGINAL_ROOT+'/CAUSAL_QUALIFICATION_PROOF_V2.json' and
            refs[4]['path']==SOURCE_ROOT+'/SOURCE_FILES.json', 'Approved evidence/source path differs')
    require(all(Path(refs[i]['path']).is_relative_to(ROOT) for i in (2,3,4)),
            'Successor proposal/authorization/source must be in new root')
    print(canonical(prepare(*refs,args.output)).decode())

if __name__=='__main__':main()
