"""Authored CPU staging smoke test; never reserves or launches a GPU phase.

Uses the real current plan, core input, and request ledger, with explicitly fake
candidate files, fake idle hardware, launch-refusing worker stubs and mocked
qualification/model rehash. All synthetic plans/manifests live in a TemporaryDirectory
outside the campaign root and are removed before the diagnostic receipt is saved.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

try:
    from . import behavior_plan as behavior
except ImportError:
    import behavior_plan as behavior
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_phase
import phase_budget


def run(reference_stage, output, *, expected_generations, expected_tf):
    reference_stage, output = Path(reference_stage), Path(output)
    behavior.require(not output.exists(), 'Dry-run receipt needs a fresh path')
    master_path = reference_stage/'input/experiment_plan.json'
    master = json.loads(master_path.read_text()); digest = behavior.sha256(master_path)
    parent_path = reference_stage/'input/parent_experiment_plan.json'
    parent = json.loads(parent_path.read_text())
    behavior.validate_master(master,digest,parent=parent,parent_sha=behavior.sha256(parent_path))
    behavior.require(master.get('plan_version') == 2,'Authored smoke test must use the current repaired plan')
    prior_path = reference_stage/'input/prior_raw_reviewed_manifest.json'
    behavior.require(behavior.sha256(prior_path) == master['inputs']['prior_raw_review_manifest_sha256'],'Prior input identity changed')
    prior = json.loads(prior_path.read_text())
    prepared = Path(prior['prepared'])/'prepared_records.jsonl'
    behavior.require(behavior.sha256(prepared) == master['inputs']['prepared_records_sha256'],'Frozen core input changed')
    rows = [json.loads(line) for line in prepared.read_text().splitlines()]
    ledger = phase_budget.account(reference_stage.parent,digest,master['parent_plan_sha256'])
    behavior.require(ledger['generation_requests'] == expected_generations and ledger['tf_requests'] == expected_tf,
                     'Actual current request ledger differs from the supplied dry-run expectation')
    previous = behavior.counts_from_phase_ledger(reference_stage.parent,digest,master['parent_plan_sha256'])
    original_model_bindings = {name:record for name,record in prior['bound_files'].items()
                              if name.startswith(prior['model_snapshot']+'/') or name.startswith(prior['checkpoint']+'/') or '/venv/' in name}
    original_sha = build_phase.supervisor.sha256
    def mocked_prior_model_rehash(path):
        return original_model_bindings[str(path)]['sha256'] if str(path) in original_model_bindings else original_sha(path)
    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix='codex-authored-behavior-dryrun-') as directory:
        root = Path(directory); stage = root/'codex-authored-behavior-smoke-stage'
        inputs = stage/'input'; source = stage/'source/infra/gpu03/direction_discovery'
        inputs.mkdir(parents=True); source.mkdir(parents=True)
        # Any accidental invocation of either generated command fails before
        # imports, model loads, GPU probing, or external process creation.
        for name in ('engine.py','supervisor.py'):
            (source/name).write_text('raise SystemExit("Authored staging fixture: execution forbidden")\n')
        for original,name in [(master_path,'experiment_plan.json'),(parent_path,'parent_experiment_plan.json'),
                              (prior_path,'prior_raw_reviewed_manifest.json')]:
            shutil.copyfile(original,inputs/name)
        selected={}
        for layer in (4,8,12):
            candidate=root/f'AUTHORED-NOT-A-REAL-CANDIDATE-L{layer}.safetensors'
            candidate.write_bytes(b'Authored staging fixture; tensor loading is forbidden.\n')
            selected[f'target:authored_layer{layer}']={'role':'target','layers':[{
                'layer':layer,'kind':'candidate','path':str(candidate),'sha256':behavior.sha256(candidate),
                'selectors':[{'key':'authored_only','column':0}]}]}
        request_plan=behavior.build_request_plan(rows,master,digest,selected,phase='screening',previous_counts=previous,
                                                parent_master=parent,parent_master_sha256=behavior.sha256(parent_path))
        request_plan['prepared_records']=str(prepared)
        (inputs/'request_plan.json').write_text(behavior.canonical(request_plan)+'\n')
        fake_inventory=[{'index':gpu,'uuid':f'GPU-00000000-0000-0000-0000-{gpu:012d}',
                         'name':build_phase.supervisor.raw.engine.EXPECTED_GPU_NAME,'memory_total_mib':32768,
                         'memory_used_mib':0,'memory_free_mib':32768,'utilization_percent':0,'processes':[]} for gpu in range(8)]
        with patch.object(build_phase.supervisor,'RUN_ROOT',root), \
             patch.object(build_phase.supervisor,'sha256',side_effect=mocked_prior_model_rehash), \
             patch.object(build_phase.supervisor.raw,'gpu_snapshot',return_value=fake_inventory), \
             patch.object(build_phase,'verify_causal_qualification',return_value=[]), \
             patch.object(phase_budget,'account',return_value=ledger), \
             patch.object(build_phase.supervisor.subprocess,'Popen',side_effect=AssertionError('CPU dry run may not launch a process')), \
             redirect_stdout(captured):
            build_phase.build(stage,'authored_screening_dryrun',list(range(8)),3600,digest)
            manifest_path=stage/'reviewed_manifest.json'
            manifest=build_phase.supervisor.load_manifest(manifest_path,original_sha(manifest_path))
        tasks=[json.loads(Path(worker['command'][3]).read_text()) for worker in manifest['workers']]
        planned={r['request_id']:r for r in request_plan['requests']}
        actual={r['request_id']:r for task in tasks for r in task['requests']}
        behavior.require(len(planned)==len(actual)==780 and actual==planned and sum(len(t['requests']) for t in tasks)==780,
                         'Authored build lost/duplicated/changed planned requests')
        behavior.require(all(t['mode']=='generate' and t['sampling']==master['sampling'] and t['conditions']==request_plan['conditions'] and
                             t['teacher_forced_padded_sequence_length']==2176 and t['attention_policy']=='exclusive_math' for t in tasks),
                         'Generation task science differs from the current plan')
        baseline=[r for r in planned.values() if r['condition_id']=='baseline']
        behavior.require(len(baseline)==60 and Counter(r['scope'] for r in baseline)=={'primary':24,'local':36},
                         'Original matched baseline request coverage changed')
        by_cell={}
        for r in planned.values():
            key=(str(r['problem_id']),r['scope'],r['record_id'] if r['scope']=='local' else None,r['sample_index'])
            by_cell.setdefault(key,[]).append(r)
        behavior.require(len(by_cell)==60 and all(len(v)==13 and len({r['seed'] for r in v})==1 for v in by_cell.values()),
                         'Matched generation seed/condition cells changed')
        behavior.require(not Path(manifest['output']).exists() and not Path(manifest['runtime']).exists(),
                         'Authored staging created a worker runtime/output')
        receipt={'status':'authored_cpu_staging_passed','master_plan_sha256':digest,'parent_plan_sha256':master['parent_plan_sha256'],
                 'generation_requests_before':ledger['generation_requests'],'tf_requests_before':ledger['tf_requests'],
                 'hypothetical_screening_generations':780,'hypothetical_generation_total':ledger['generation_requests']+780,
                 'workers':8,'requests_per_worker':[len(t['requests']) for t in tasks],'conditions':13,
                 'baseline_cells':60,'baseline_scope_counts':dict(Counter(r['scope'] for r in baseline)),
                 'paired_seed_cells':60,'exact_task_request_coverage':True,
                 'mocked_boundaries':['GPU inventory','causal qualification','prior model/runtime content rehash','campaign root and fixed ledger snapshot'],
                 'no_model_or_generated_code_execution':True,'no_scientific_target_selected':True,'no_budget_reserved':True,
                 'temporary_manifests_and_fake_candidates_removed':False,'qualified_engine_source_changed':False,
                 'source_sha256':{p.name:behavior.sha256(p) for p in [Path(__file__),Path(build_phase.__file__),Path(behavior.__file__),Path(phase_budget.__file__)]}}
    behavior.require(not root.exists(),'Temporary synthetic stage was not removed')
    after=phase_budget.account(reference_stage.parent,digest,master['parent_plan_sha256'])
    behavior.require(after['generation_requests']==ledger['generation_requests'] and after['tf_requests']==ledger['tf_requests'],
                     'Request ledger changed during dry run; do not report unchanged-budget proof')
    receipt['temporary_manifests_and_fake_candidates_removed']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(behavior.canonical(receipt)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-stage',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expected-generations',required=True,type=int);p.add_argument('--expected-tf',required=True,type=int)
    args=p.parse_args()
    print(behavior.canonical(run(args.reference_stage,args.output,expected_generations=args.expected_generations,expected_tf=args.expected_tf)))
